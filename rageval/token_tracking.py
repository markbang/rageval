from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from rageval.utils import ensure_parent_dir

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency managed at runtime
    tiktoken = None


class TokenCounter:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = self._build_encoding(model_name)

    def _build_encoding(self, model_name: str) -> Any:
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding is None:
            return len(text.split())
        return len(self.encoding.encode(text, disallowed_special=()))

    def count_texts(self, texts: list[str]) -> int:
        return sum(self.count_text(text) for text in texts)


class TokenUsageRecorder:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        ensure_parent_dir(output_path)

    def append(self, payload: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def extract_langchain_usage(response: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage.update(usage_metadata)

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            usage.update(token_usage)

    return usage
