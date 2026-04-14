from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile


@dataclass(slots=True)
class ModelConfig:
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    tokenizer_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout_seconds: float = 120.0
    max_retries: int = 2
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    @classmethod
    def from_env(
        cls,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        tokenizer_model: str | None = None,
        temperature: float = 0.0,
        request_timeout_seconds: float = 120.0,
        max_retries: int = 2,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
    ) -> "ModelConfig":
        resolved_llm_model = llm_model or os.getenv("RAGEVAL_LLM_MODEL") or "gpt-4o-mini"
        resolved_embedding_model = (
            embedding_model
            or os.getenv("RAGEVAL_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )
        raw_embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else os.getenv("RAGEVAL_EMBEDDING_DIMENSION")
        )
        resolved_embedding_dimension = int(raw_embedding_dimension or 1536)
        resolved_tokenizer_model = (
            tokenizer_model
            or os.getenv("RAGEVAL_TOKENIZER_MODEL")
            or resolved_llm_model
        )
        return cls(
            llm_model=resolved_llm_model,
            embedding_model=resolved_embedding_model,
            embedding_dimension=resolved_embedding_dimension,
            tokenizer_model=resolved_tokenizer_model,
            temperature=temperature,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            openai_base_url=openai_base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Export the key before running the experiment."
            )
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer.")


@dataclass(slots=True)
class VectorRAGConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 100
    top_k: int = 5


@dataclass(slots=True)
class ExperimentConfig:
    dataset_dir: Path
    dataset_globs: tuple[str, ...]
    include_json: bool
    output_csv: Path
    token_usage_jsonl: Path
    log_dir: Path
    lightrag_workspace_root: Path
    limit: int | None
    model: ModelConfig
    vector_rag: VectorRAGConfig

    @classmethod
    def defaults(
        cls,
        dataset_dir: Path = Path("./dataset/processed"),
        dataset_globs: tuple[str, ...] = ("**/*.jsonl",),
        include_json: bool = False,
        output_csv: Path = Path("./results/experiment_results.csv"),
        token_usage_jsonl: Path = Path("./results/token_usage.jsonl"),
        log_dir: Path = Path("./logs"),
        lightrag_workspace_root: Path | None = None,
        limit: int | None = None,
        model: ModelConfig | None = None,
        vector_rag: VectorRAGConfig | None = None,
    ) -> "ExperimentConfig":
        workspace_root = lightrag_workspace_root or Path(tempfile.gettempdir()) / "rageval_lightrag"
        return cls(
            dataset_dir=dataset_dir,
            dataset_globs=dataset_globs,
            include_json=include_json,
            output_csv=output_csv,
            token_usage_jsonl=token_usage_jsonl,
            log_dir=log_dir,
            lightrag_workspace_root=workspace_root,
            limit=limit,
            model=model or ModelConfig.from_env(),
            vector_rag=vector_rag or VectorRAGConfig(),
        )
