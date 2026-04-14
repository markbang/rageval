from __future__ import annotations

import argparse
import asyncio
from itertools import islice
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from rageval.config import ModelConfig, VectorRAGConfig
from rageval.data_loader import DatasetLoader
from rageval.models import ExperimentDocument, QAPair, RAGRunResult
from rageval.rag.lightrag_rag import LightRAGSystem
from rageval.rag.vector_rag import VectorRAGSystem
from rageval.token_tracking import TokenCounter


DEFAULT_DATASET_PATH = Path("./dataset/processed/docfinqa/docfinqa_unified.jsonl")
DEFAULT_LIGHTRAG_WORKSPACE = Path("/tmp/rageval_smoke_lightrag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small end-to-end smoke test for VectorRAG and LightRAG using "
            "the first two documents from DocFinQA."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Processed JSONL file used for the smoke test.",
    )
    parser.add_argument(
        "--document-count",
        type=int,
        default=2,
        help="Number of documents to test. Defaults to 2.",
    )
    parser.add_argument(
        "--max-qa-per-document",
        type=int,
        default=None,
        help="Optional limit on QA pairs per document.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--lightrag-workspace-root",
        type=Path,
        default=DEFAULT_LIGHTRAG_WORKSPACE,
        help="Temporary LightRAG workspace root. It will be cleared between documents.",
    )
    return parser.parse_args()


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    model_config = ModelConfig.from_env(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        tokenizer_model=args.tokenizer_model,
        temperature=args.temperature,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )
    model_config.validate()
    return model_config


def build_vector_config(args: argparse.Namespace) -> VectorRAGConfig:
    return VectorRAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )


def load_documents(dataset_path: Path, document_count: int) -> list[ExperimentDocument]:
    if document_count <= 0:
        raise ValueError("--document-count must be positive.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    loader = DatasetLoader(
        dataset_dir=dataset_path.parent,
        dataset_globs=(dataset_path.name,),
    )
    documents = list(islice(loader.iter_documents(), document_count))
    if not documents:
        raise ValueError(f"No documents found in {dataset_path}")
    return documents


def compose_query_text(qa_pair: QAPair) -> str:
    instruction = (qa_pair.instruction or "").strip()
    question = qa_pair.question.strip()
    if instruction and question:
        return f"Instruction:\n{instruction}\n\nQuestion:\n{question}"
    if instruction:
        return instruction
    return question


def format_usage(payload: dict[str, Any]) -> str:
    if not payload:
        return "{}"
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def print_block(title: str, content: str) -> None:
    print(f"\n[{title}]")
    print(content if content else "<empty>")


def print_result(system_name: str, result: RAGRunResult) -> None:
    print_block(f"{system_name} Retrieved Context", result.retrieved_context)
    print_block(f"{system_name} Answer", result.answer)
    print_block(f"{system_name} Token Usage", format_usage(result.token_usage))
    if result.error:
        print_block(f"{system_name} Error", result.error)


def clear_workspace(workspace_root: Path) -> None:
    if workspace_root.exists():
        shutil.rmtree(workspace_root, ignore_errors=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    gc.collect()


async def build_vector_system(
    document: ExperimentDocument,
    model_config: ModelConfig,
    vector_config: VectorRAGConfig,
    token_counter: TokenCounter,
) -> tuple[VectorRAGSystem | None, dict[str, Any], float, str | None]:
    system = VectorRAGSystem(model_config, vector_config, token_counter)
    started = time.perf_counter()
    try:
        usage = system.index_document(document.document, document.doc_id)
        elapsed = time.perf_counter() - started
        return system, usage, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, {}, elapsed, str(exc)


async def build_lightrag_system(
    document: ExperimentDocument,
    model_config: ModelConfig,
    token_counter: TokenCounter,
    workspace_root: Path,
) -> tuple[LightRAGSystem | None, dict[str, Any], float, str | None]:
    system = LightRAGSystem(model_config, token_counter, workspace_root=workspace_root)
    started = time.perf_counter()
    try:
        usage = await system.initialize(document.doc_id, document.document)
        elapsed = time.perf_counter() - started
        return system, usage, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        try:
            await system.close()
        except Exception:
            pass
        return None, {}, elapsed, str(exc)


async def run_document_smoke_test(
    document: ExperimentDocument,
    model_config: ModelConfig,
    vector_config: VectorRAGConfig,
    token_counter: TokenCounter,
    workspace_root: Path,
    max_qa_per_document: int | None = None,
) -> None:
    clear_workspace(workspace_root)

    vector_system, vector_build_usage, vector_build_seconds, vector_build_error = await build_vector_system(
        document,
        model_config,
        vector_config,
        token_counter,
    )
    lightrag_system, lightrag_build_usage, lightrag_build_seconds, lightrag_build_error = await build_lightrag_system(
        document,
        model_config,
        token_counter,
        workspace_root,
    )

    print("=" * 100)
    print(
        f"Document: {document.doc_id}\n"
        f"Split: {document.split or '-'}\n"
        f"QA pairs: {len(document.qa_pairs)}\n"
        f"Doc tokens: {document.doc_length_tokens or token_counter.count_text(document.document)}"
    )
    print_block(
        "VectorRAG Build",
        f"time_seconds={vector_build_seconds:.3f}\nusage={format_usage(vector_build_usage)}",
    )
    if vector_build_error:
        print_block("VectorRAG Build Error", vector_build_error)

    print_block(
        "LightRAG Build",
        f"time_seconds={lightrag_build_seconds:.3f}\nusage={format_usage(lightrag_build_usage)}",
    )
    if lightrag_build_error:
        print_block("LightRAG Build Error", lightrag_build_error)

    qa_pairs = document.qa_pairs[:max_qa_per_document] if max_qa_per_document else document.qa_pairs

    try:
        for qa_index, qa_pair in enumerate(qa_pairs, start=1):
            query_text = compose_query_text(qa_pair)
            print("\n" + "-" * 100)
            print(f"QA {qa_index}: {qa_pair.q_id}")
            print_block("Question", qa_pair.question or qa_pair.instruction or "<empty>")
            print_block("Ground Truth", qa_pair.ground_truth)

            if vector_system is not None:
                try:
                    vector_result = vector_system.answer(query_text)
                except Exception as exc:
                    vector_result = RAGRunResult.from_error(str(exc))
            else:
                vector_result = RAGRunResult.from_error(
                    vector_build_error or "VectorRAG initialization failed."
                )

            if lightrag_system is not None:
                try:
                    lightrag_result = await lightrag_system.query(query_text)
                except Exception as exc:
                    lightrag_result = RAGRunResult.from_error(str(exc))
            else:
                lightrag_result = RAGRunResult.from_error(
                    lightrag_build_error or "LightRAG initialization failed."
                )

            print_result("VectorRAG", vector_result)
            print_result("LightRAG", lightrag_result)
    finally:
        if vector_system is not None:
            vector_system.reset()
        if lightrag_system is not None:
            await lightrag_system.close()
        clear_workspace(workspace_root)


async def async_main() -> int:
    args = parse_args()
    try:
        model_config = build_model_config(args)
        vector_config = build_vector_config(args)
        token_counter = TokenCounter(model_config.tokenizer_model)
        documents = load_documents(args.dataset_path, args.document_count)
    except Exception as exc:
        print(f"[Smoke Test Setup Failed] {exc}", file=sys.stderr)
        return 1

    print(
        "Smoke test configuration:\n"
        f"  dataset_path={args.dataset_path}\n"
        f"  document_count={len(documents)}\n"
        f"  llm_model={model_config.llm_model}\n"
        f"  embedding_model={model_config.embedding_model}\n"
        f"  openai_base_url={model_config.openai_base_url or 'default'}\n"
        f"  lightrag_workspace_root={args.lightrag_workspace_root}"
    )

    for index, document in enumerate(documents, start=1):
        print(f"\n\n### Smoke Test Document {index}/{len(documents)} ###")
        await run_document_smoke_test(
            document=document,
            model_config=model_config,
            vector_config=vector_config,
            token_counter=token_counter,
            workspace_root=args.lightrag_workspace_root,
            max_qa_per_document=args.max_qa_per_document,
        )

    print("\nSmoke test finished.")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
