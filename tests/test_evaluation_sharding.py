from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rageval.evaluation_sharding import (
    build_pair_order,
    merge_shard_detail_rows,
    select_rows_for_shard,
)


class EvaluationShardingTests(unittest.TestCase):
    def test_select_rows_for_shard_round_robin(self) -> None:
        rows = [{"doc_id": f"doc{i}", "q_id": f"q{i}"} for i in range(10)]
        shards = [select_rows_for_shard(rows, 4, idx) for idx in range(4)]

        lengths = [len(shard) for shard in shards]
        self.assertEqual(sum(lengths), 10)
        self.assertLessEqual(max(lengths) - min(lengths), 1)

        pairs = {
            (row["doc_id"], row["q_id"])
            for shard in shards
            for row in shard
        }
        self.assertEqual(len(pairs), 10)

    def test_merge_shard_detail_rows_preserves_original_pair_order(self) -> None:
        rows = [
            {"doc_id": "doc1", "q_id": "q1"},
            {"doc_id": "doc2", "q_id": "q2"},
            {"doc_id": "doc3", "q_id": "q3"},
        ]
        pair_order = build_pair_order(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            shard_a = Path(tmp_dir) / "a.csv"
            shard_b = Path(tmp_dir) / "b.csv"
            shard_a.write_text(
                "doc_id,q_id\n"
                "doc3,q3\n"
                "doc1,q1\n",
                encoding="utf-8",
            )
            shard_b.write_text(
                "doc_id,q_id\n"
                "doc2,q2\n",
                encoding="utf-8",
            )

            merged = merge_shard_detail_rows([shard_a, shard_b], pair_order)

        self.assertEqual(
            [(row["doc_id"], row["q_id"]) for row in merged],
            [("doc1", "q1"), ("doc2", "q2"), ("doc3", "q3")],
        )


if __name__ == "__main__":
    unittest.main()
