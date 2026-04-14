from __future__ import annotations

import unittest

from scripts.generate_analysis_outputs import (
    build_doc_metric_points,
    infer_experiment_source,
)


class GenerateAnalysisOutputsTests(unittest.TestCase):
    def test_infer_experiment_source_distinguishes_main_stress_ultra(self) -> None:
        self.assertEqual(
            infer_experiment_source(
                {
                    "dataset_name": "DocFinQA",
                    "evaluation_set": "main_or_unknown",
                }
            ),
            "main",
        )
        self.assertEqual(
            infer_experiment_source(
                {
                    "dataset_name": "DocFinQA",
                    "evaluation_set": "stress",
                }
            ),
            "stress",
        )
        self.assertEqual(
            infer_experiment_source(
                {
                    "dataset_name": "UltraLongFinance",
                    "evaluation_set": "main_or_unknown",
                }
            ),
            "ultra",
        )

    def test_build_doc_metric_points_aggregates_detail_rows_to_doc_level(self) -> None:
        detail_rows = [
            {
                "system": "vector",
                "dataset_name": "DocFinQA",
                "doc_id": "doc-1",
                "evaluation_set": "main_or_unknown",
                "length_bucket": "short",
                "doc_length_tokens": 1000,
                "answer_quality_score": 0.2,
                "is_error": False,
            },
            {
                "system": "vector",
                "dataset_name": "DocFinQA",
                "doc_id": "doc-1",
                "evaluation_set": "main_or_unknown",
                "length_bucket": "short",
                "doc_length_tokens": 1000,
                "answer_quality_score": 0.4,
                "is_error": False,
            },
            {
                "system": "lightrag",
                "dataset_name": "UltraLongFinance",
                "doc_id": "doc-2",
                "evaluation_set": "main_or_unknown",
                "length_bucket": "long",
                "doc_length_tokens": 500000,
                "answer_quality_score": 0.6,
                "is_error": False,
            },
        ]

        doc_points = build_doc_metric_points(detail_rows, "answer_quality_score")

        self.assertEqual(len(doc_points), 2)
        vector_point = next(point for point in doc_points if point["system"] == "vector")
        ultra_point = next(point for point in doc_points if point["system"] == "lightrag")
        self.assertAlmostEqual(vector_point["answer_quality_score"], 0.3)
        self.assertEqual(vector_point["experiment_source"], "main")
        self.assertEqual(ultra_point["experiment_source"], "ultra")

    def test_infer_experiment_source_prefers_stress_over_main(self) -> None:
        self.assertEqual(
            infer_experiment_source(
                {
                    "dataset_name": "DocFinQA",
                    "evaluation_set": "stress",
                }
            ),
            "stress",
        )


if __name__ == "__main__":
    unittest.main()
