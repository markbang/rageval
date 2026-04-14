from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class QAPair:
    q_id: str
    question: str
    ground_truth: str
    instruction: str | None = None


@dataclass(slots=True)
class ExperimentDocument:
    dataset_name: str
    doc_id: str
    document: str
    qa_pairs: list[QAPair]
    doc_length_tokens: int | None
    source_file: Path
    record_index: int
    split: str | None = None
    domain: str | None = None
    language: str | None = None
    level: int | None = None
    set_id: int | None = None


@dataclass(slots=True)
class RAGRunResult:
    answer: str
    retrieved_context: str
    token_usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_error(cls, error_message: str) -> "RAGRunResult":
        return cls(
            answer=f"[ERROR] {error_message}",
            retrieved_context="",
            token_usage={},
            error=error_message,
        )
