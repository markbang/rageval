from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import sys


def select_rows_for_shard(
    rows: list[dict[str, str]],
    num_shards: int | None,
    shard_index: int | None,
) -> list[dict[str, str]]:
    if num_shards is None and shard_index is None:
        return rows
    if num_shards is None or shard_index is None:
        raise ValueError("num_shards and shard_index must be provided together.")
    if num_shards <= 0:
        raise ValueError("num_shards must be a positive integer.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def build_pair_order(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    return {
        (row.get("doc_id", ""), row.get("q_id", "")): index
        for index, row in enumerate(rows)
    }


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def merge_shard_detail_rows(
    detail_paths: list[Path],
    pair_order: dict[tuple[str, str], int],
) -> list[dict[str, str]]:
    merged_rows: list[dict[str, str]] = []
    for path in detail_paths:
        if path.exists():
            merged_rows.extend(load_csv_rows(path))
    merged_rows.sort(
        key=lambda row: pair_order.get((row.get("doc_id", ""), row.get("q_id", "")), 10**18)
    )
    return merged_rows
