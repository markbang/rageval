from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        flattened = [normalize_text(item) for item in value]
        return "\n\n".join(item for item in flattened if item)
    if isinstance(value, dict):
        if "text" in value:
            return normalize_text(value["text"])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def slugify(value: str, default: str = "sample") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or default

