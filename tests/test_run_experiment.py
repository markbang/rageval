from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_with_ragas import EvaluationConfig, MetricSuite
from scripts.evaluate_with_ragas import load_input_rows
from run_experiment import CSV_HEADERS, load_completed_qids


class LoadCompletedQidsTests(unittest.TestCase):
    def test_load_completed_qids_handles_large_context_fields(self) -> None:
        original_limit = csv.field_size_limit()
        csv.field_size_limit(131072)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = Path(tmp_dir) / "experiment_results.csv"
                large_context = "x" * 200000
                with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "dataset_name": "DocFinQA",
                            "split": "train",
                            "domain": "financial",
                            "language": "en",
                            "level": "",
                            "set_id": "",
                            "doc_id": "doc-1",
                            "q_id": "q-1",
                            "instruction": "",
                            "question": "question",
                            "doc_length_tokens": "10",
                            "ground_truth": "answer",
                            "vector_retrieved_context": large_context,
                            "vector_answer": "answer",
                            "lightrag_retrieved_context": large_context,
                            "lightrag_answer": "answer",
                        }
                    )

                completed = load_completed_qids(csv_path)

                self.assertEqual(completed, {"doc-1": {"q-1"}})
        finally:
            csv.field_size_limit(original_limit)

    def test_load_input_rows_handles_large_context_fields(self) -> None:
        original_limit = csv.field_size_limit()
        csv.field_size_limit(131072)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = Path(tmp_dir) / "experiment_results.csv"
                large_context = "y" * 200000
                with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "dataset_name": "DocFinQA",
                            "split": "train",
                            "domain": "financial",
                            "language": "en",
                            "level": "",
                            "set_id": "",
                            "doc_id": "doc-2",
                            "q_id": "q-2",
                            "instruction": "",
                            "question": "question",
                            "doc_length_tokens": "10",
                            "ground_truth": "answer",
                            "vector_retrieved_context": large_context,
                            "vector_answer": "answer",
                            "lightrag_retrieved_context": large_context,
                            "lightrag_answer": "answer",
                        }
                    )

                rows = load_input_rows(csv_path)

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["doc_id"], "doc-2")
        finally:
            csv.field_size_limit(original_limit)

    def test_metric_suite_balanced_uses_async_compatible_llm_and_embeddings(self) -> None:
        config = EvaluationConfig(
            input_csv=Path("results/experiment_results_repaired.csv"),
            output_dir=Path("results/evaluation/test"),
            metric_profile="balanced",
            judge_model="gpt-5.4-mini",
            embedding_model="text-embedding-3-small",
            openai_api_key="dummy",
            openai_base_url="https://api.openai.com/v1",
            max_retries=2,
            metric_timeout_seconds=120.0,
            limit=None,
            manifest_path=None,
            num_shards=None,
            shard_index=None,
        )

        metric_suite = MetricSuite(config)

        answer_accuracy_llm = metric_suite.metrics["answer_accuracy"].llm
        response_relevancy_embeddings = metric_suite.metrics["response_relevancy"].embeddings

        self.assertTrue(hasattr(answer_accuracy_llm, "agenerate_text"))
        self.assertIsNotNone(response_relevancy_embeddings)


if __name__ == "__main__":
    unittest.main()
