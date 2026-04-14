from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

from tqdm import tqdm

from rageval.config import ExperimentConfig, ModelConfig, VectorRAGConfig
from rageval.data_loader import DatasetLoader
from rageval.logging_utils import setup_logging
from rageval.models import ExperimentDocument, QAPair, RAGRunResult
from rageval.rag.lightrag_rag import LightRAGSystem
from rageval.rag.vector_rag import VectorRAGSystem
from rageval.token_tracking import TokenCounter, TokenUsageRecorder
from rageval.utils import ensure_parent_dir, slugify


CSV_HEADERS = [
    "dataset_name",
    "split",
    "domain",
    "language",
    "level",
    "set_id",
    "doc_id",
    "q_id",
    "instruction",
    "question",
    "doc_length_tokens",
    "ground_truth",
    "vector_retrieved_context",
    "vector_answer",
    "lightrag_retrieved_context",
    "lightrag_answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Vector RAG and LightRAG on processed document-level datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./dataset/processed"),
        help="Directory containing processed JSONL files with document-level qa_pairs.",
    )
    parser.add_argument(
        "--dataset-glob",
        action="append",
        default=None,
        help="Glob pattern(s) under dataset-dir. Defaults to **/*.jsonl",
    )
    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Also read JSON array files. Processed datasets should normally use JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("./results/experiment_results.csv"),
    )
    parser.add_argument(
        "--token-usage-jsonl",
        type=Path,
        default=Path("./results/token_usage.jsonl"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name. CLI > RAGEVAL_LLM_MODEL > default(gpt-4o-mini).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name. CLI > RAGEVAL_EMBEDDING_MODEL > default(text-embedding-3-small).",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=None,
        help="Embedding vector dimension for LightRAG. Required when using a non-1536 embedding model.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer model used for token counting and LightRAG chunking. Useful for OpenAI-compatible custom models.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Override OPENAI_API_KEY from CLI.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Override OPENAI_BASE_URL from CLI, for example http://host:port/v1.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
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
    vector_config = VectorRAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    dataset_globs = tuple(args.dataset_glob or ["**/*.jsonl"])
    return ExperimentConfig.defaults(
        dataset_dir=args.dataset_dir,
        dataset_globs=dataset_globs,
        include_json=args.include_json,
        output_csv=args.output_csv,
        token_usage_jsonl=args.token_usage_jsonl,
        log_dir=args.log_dir,
        limit=args.limit,
        model=model_config,
        vector_rag=vector_config,
    )


def prepare_csv(output_csv: Path) -> None:
    ensure_parent_dir(output_csv)
    if output_csv.exists():
        return
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()


def load_completed_qids(output_csv: Path) -> dict[str, set[str]]:
    if not output_csv.exists():
        return {}

    csv.field_size_limit(sys.maxsize)
    completed_qids: dict[str, set[str]] = {}
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            q_id = (row.get("q_id") or "").strip()
            if not doc_id or not q_id:
                continue
            completed_qids.setdefault(doc_id, set()).add(q_id)
    return completed_qids


def open_csv_appender(output_csv: Path) -> tuple[Any, csv.DictWriter]:
    ensure_parent_dir(output_csv)
    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0
    handle = output_csv.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
    if needs_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def compose_query_text(qa_pair: QAPair) -> str:
    instruction = (qa_pair.instruction or "").strip()
    question = qa_pair.question.strip()
    if instruction and question:
        return f"Instruction:\n{instruction}\n\nQuestion:\n{question}"
    if instruction:
        return instruction
    return question


def make_error_result(
    error_message: str,
    token_usage: dict[str, Any] | None = None,
) -> RAGRunResult:
    return RAGRunResult(
        answer="ERROR",
        retrieved_context="",
        token_usage=token_usage or {},
        error=error_message,
    )


def cleanup_lightrag_workspace(
    doc_id: str,
    workspace_root: Path,
    logger: logging.Logger,
) -> None:
    workspace_dir = workspace_root / slugify(doc_id)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)
        logger.info("Removed LightRAG workspace for doc_id=%s at %s", doc_id, workspace_dir)


async def run_vector_branch(
    system: VectorRAGSystem | None,
    query_text: str,
    qa_pair: QAPair,
    doc_id: str,
    logger: logging.Logger,
    build_error: str | None = None,
) -> RAGRunResult:
    if build_error is not None:
        return make_error_result(f"VectorRAG index failed: {build_error}")
    if system is None:
        return make_error_result("VectorRAG index is unavailable.")
    try:
        return system.answer(query_text)
    except Exception as exc:
        logger.exception("VectorRAG failed for doc_id=%s q_id=%s", doc_id, qa_pair.q_id)
        return make_error_result(
            str(exc),
            token_usage={
                "estimated_question_tokens": system.token_counter.count_text(query_text),
                "estimated_answer_tokens": system.token_counter.count_text("ERROR"),
            },
        )


async def run_lightrag_branch(
    system: LightRAGSystem | None,
    query_text: str,
    qa_pair: QAPair,
    doc_id: str,
    doc_length_tokens: int,
    token_counter: TokenCounter,
    logger: logging.Logger,
    build_error: str | None = None,
) -> RAGRunResult:
    query_tokens = token_counter.count_text(query_text)
    error_usage = {
        "estimated_document_tokens": doc_length_tokens,
        "estimated_question_tokens": query_tokens,
        "estimated_answer_tokens": token_counter.count_text("ERROR"),
        "main_estimated_document_tokens": doc_length_tokens,
        "main_estimated_query_tokens": query_tokens,
        "main_estimated_answer_tokens": token_counter.count_text("ERROR"),
    }
    if build_error is not None:
        return make_error_result(f"LightRAG index failed: {build_error}", error_usage)
    if system is None:
        return make_error_result("LightRAG index is unavailable.", error_usage)
    try:
        result = await system.query(query_text)
    except Exception as exc:
        logger.exception("LightRAG failed for doc_id=%s q_id=%s", doc_id, qa_pair.q_id)
        return make_error_result(str(exc), error_usage)

    answer_tokens = token_counter.count_text(result.answer)
    result.token_usage.setdefault("estimated_document_tokens", doc_length_tokens)
    result.token_usage.setdefault("estimated_question_tokens", query_tokens)
    result.token_usage.setdefault("estimated_answer_tokens", answer_tokens)
    result.token_usage["main_estimated_document_tokens"] = doc_length_tokens
    result.token_usage["main_estimated_query_tokens"] = query_tokens
    result.token_usage["main_estimated_answer_tokens"] = answer_tokens
    return result


def build_vector_system(
    document: ExperimentDocument,
    config: ExperimentConfig,
    token_counter: TokenCounter,
    logger: logging.Logger,
) -> tuple[VectorRAGSystem | None, dict[str, Any], str | None]:
    system = VectorRAGSystem(config.model, config.vector_rag, token_counter)
    try:
        build_usage = system.index_document(document.document, document.doc_id)
        return system, build_usage, None
    except Exception as exc:
        logger.exception("VectorRAG indexing failed for doc_id=%s", document.doc_id)
        return None, {}, str(exc)


async def build_lightrag_system(
    document: ExperimentDocument,
    config: ExperimentConfig,
    token_counter: TokenCounter,
    logger: logging.Logger,
) -> tuple[LightRAGSystem | None, dict[str, Any], str | None]:
    document_tokens = document.doc_length_tokens or token_counter.count_text(document.document)
    system = LightRAGSystem(
        config.model,
        token_counter,
        workspace_root=config.lightrag_workspace_root,
    )
    try:
        build_usage = await system.initialize(document.doc_id, document.document)
        build_usage.setdefault("estimated_document_tokens", document_tokens)
        build_usage["main_estimated_document_tokens"] = document_tokens
        return system, build_usage, None
    except Exception as exc:
        logger.exception("LightRAG indexing failed for doc_id=%s", document.doc_id)
        await system.close()
        return (
            None,
            {
                "estimated_document_tokens": document_tokens,
                "main_estimated_document_tokens": document_tokens,
            },
            str(exc),
        )


async def cleanup_document_resources(
    doc_id: str,
    vector_system: VectorRAGSystem | None,
    lightrag_system: LightRAGSystem | None,
    workspace_root: Path,
    logger: logging.Logger,
) -> None:
    if vector_system is not None:
        try:
            vector_system.reset()
        except Exception:
            logger.exception("VectorRAG cleanup failed for doc_id=%s", doc_id)

    if lightrag_system is not None:
        try:
            await lightrag_system.close()
        except Exception:
            logger.exception("LightRAG close failed for doc_id=%s", doc_id)

    cleanup_lightrag_workspace(doc_id, workspace_root, logger)
    gc.collect()


async def run_experiment(config: ExperimentConfig) -> None:
    config.model.validate()

    logger = setup_logging(config.log_dir)
    logger.info(
        "Starting experiment with dataset_dir=%s llm_model=%s embedding_model=%s embedding_dimension=%s openai_base_url=%s",
        config.dataset_dir,
        config.model.llm_model,
        config.model.embedding_model,
        config.model.embedding_dimension,
        config.model.openai_base_url or "default",
    )

    loader = DatasetLoader(
        dataset_dir=config.dataset_dir,
        dataset_globs=config.dataset_globs,
        include_json=config.include_json,
    )
    documents = loader.iter_documents()
    token_counter = TokenCounter(config.model.tokenizer_model)
    token_recorder = TokenUsageRecorder(config.token_usage_jsonl)

    prepare_csv(config.output_csv)
    completed_qids = load_completed_qids(config.output_csv)
    csv_handle, csv_writer = open_csv_appender(config.output_csv)

    processed = 0
    try:
        for document in tqdm(documents, desc="Running experiment", unit="document"):
            if config.limit is not None and processed >= config.limit:
                break

            doc_length_tokens = document.doc_length_tokens or token_counter.count_text(
                document.document
            )
            existing_qids = completed_qids.get(document.doc_id, set())
            pending_qa_pairs = [
                qa_pair for qa_pair in document.qa_pairs if qa_pair.q_id not in existing_qids
            ]

            if not pending_qa_pairs:
                logger.info(
                    "Skipping doc_id=%s because all %s QA rows already exist in %s",
                    document.doc_id,
                    len(document.qa_pairs),
                    config.output_csv,
                )
                continue

            if existing_qids:
                logger.warning(
                    "Resuming partially completed doc_id=%s with %s/%s pending QA pairs",
                    document.doc_id,
                    len(pending_qa_pairs),
                    len(document.qa_pairs),
                )

            vector_system, vector_build_usage, vector_build_error = build_vector_system(
                document,
                config,
                token_counter,
                logger,
            )
            lightrag_system, lightrag_build_usage, lightrag_build_error = await build_lightrag_system(
                document,
                config,
                token_counter,
                logger,
            )
            token_recorder.append(
                {
                    "event_type": "document_build",
                    "dataset_name": document.dataset_name,
                    "split": document.split,
                    "doc_id": document.doc_id,
                    "source_file": str(document.source_file),
                    "record_index": document.record_index,
                    "doc_length_tokens": doc_length_tokens,
                    "qa_pair_count": len(document.qa_pairs),
                    "pending_qa_pair_count": len(pending_qa_pairs),
                    "skipped_existing_qa_pair_count": len(existing_qids),
                    "vector_build_usage": vector_build_usage,
                    "lightrag_build_usage": lightrag_build_usage,
                    "lightrag_build_input_tokens": doc_length_tokens,
                    "vector_build_error": vector_build_error,
                    "lightrag_build_error": lightrag_build_error,
                }
            )

            try:
                for qa_pair in pending_qa_pairs:
                    query_text = compose_query_text(qa_pair)
                    try:
                        vector_result = await run_vector_branch(
                            vector_system,
                            query_text,
                            qa_pair,
                            document.doc_id,
                            logger,
                            build_error=vector_build_error,
                        )
                        lightrag_result = await run_lightrag_branch(
                            lightrag_system,
                            query_text,
                            qa_pair,
                            document.doc_id,
                            doc_length_tokens,
                            token_counter,
                            logger,
                            build_error=lightrag_build_error,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Unexpected QA failure for doc_id=%s q_id=%s",
                            document.doc_id,
                            qa_pair.q_id,
                        )
                        vector_result = make_error_result(str(exc))
                        lightrag_result = make_error_result(
                            str(exc),
                            {
                                "estimated_document_tokens": doc_length_tokens,
                                "estimated_question_tokens": token_counter.count_text(query_text),
                                "estimated_answer_tokens": token_counter.count_text("ERROR"),
                                "main_estimated_document_tokens": doc_length_tokens,
                                "main_estimated_query_tokens": token_counter.count_text(
                                    query_text
                                ),
                                "main_estimated_answer_tokens": token_counter.count_text("ERROR"),
                            },
                        )

                    csv_writer.writerow(
                        {
                            "dataset_name": document.dataset_name,
                            "split": document.split or "",
                            "domain": document.domain or "",
                            "language": document.language or "",
                            "level": document.level or "",
                            "set_id": document.set_id or "",
                            "doc_id": document.doc_id,
                            "q_id": qa_pair.q_id,
                            "instruction": qa_pair.instruction or "",
                            "question": qa_pair.question,
                            "doc_length_tokens": doc_length_tokens,
                            "ground_truth": qa_pair.ground_truth,
                            "vector_retrieved_context": vector_result.retrieved_context,
                            "vector_answer": vector_result.answer,
                            "lightrag_retrieved_context": lightrag_result.retrieved_context,
                            "lightrag_answer": lightrag_result.answer,
                        }
                    )
                    csv_handle.flush()
                    completed_qids.setdefault(document.doc_id, set()).add(qa_pair.q_id)

                    token_recorder.append(
                        {
                            "event_type": "qa_query",
                            "dataset_name": document.dataset_name,
                            "split": document.split,
                            "domain": document.domain,
                            "language": document.language,
                            "level": document.level,
                            "set_id": document.set_id,
                            "doc_id": document.doc_id,
                            "q_id": qa_pair.q_id,
                            "instruction": qa_pair.instruction,
                            "question": qa_pair.question,
                            "query_text": query_text,
                            "ground_truth": qa_pair.ground_truth,
                            "doc_length_tokens": doc_length_tokens,
                            "vector_query_usage": vector_result.token_usage,
                            "lightrag_query_usage": lightrag_result.token_usage,
                            "vector_error": vector_result.error,
                            "lightrag_error": lightrag_result.error,
                        }
                    )
            finally:
                await cleanup_document_resources(
                    document.doc_id,
                    vector_system,
                    lightrag_system,
                    config.lightrag_workspace_root,
                    logger,
                )
                del vector_system
                del lightrag_system

            processed += 1
    finally:
        csv_handle.close()

    logger.info("Experiment finished. Processed %s documents.", processed)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
