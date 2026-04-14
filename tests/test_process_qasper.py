from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rageval.token_tracking import TokenCounter
from scripts.process_qasper import build_unified_records


SAMPLE_QASPER = {
    "1909.00694": {
        "title": "Minimally Supervised Learning of Affective Events Using Discourse Relations",
        "abstract": "This is the abstract of the paper",
        "full_text": [
            {
                "section_name": "Introduction",
                "paragraphs": [
                    "A short paragraph",
                    "Another intro paragraph",
                ],
            },
            {
                "section_name": "Proposed Method ::: Polarity Function",
                "paragraphs": [
                    "",
                    "Method paragraph using seed lexicon",
                ],
            },
            {
                "section_name": "Conclusion",
                "paragraphs": ["Conclusion paragraph"],
            },
        ],
        "qas": [
            {
                "question": "What is the seed lexicon?",
                "question_id": "q-freeform",
                "question_writer": "writer0",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "a vocabulary",
                            "evidence": ["Method paragraph using seed lexicon"],
                        },
                        "annotation_id": "ann-1",
                        "worker_id": "writer0",
                    },
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": ["using seed lexicon"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["Method paragraph using seed lexicon"],
                        },
                        "annotation_id": "ann-2",
                        "worker_id": "writer1",
                    },
                ],
            },
            {
                "question": "Are there three?",
                "question_id": "q-bool",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": True,
                            "free_form_answer": "",
                            "evidence": ["Method paragraph using seed lexicon"],
                        },
                        "annotation_id": "ann-3",
                        "worker_id": "writer0",
                    }
                ],
            },
            {
                "question": "Are there four?",
                "question_id": "q-none",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": True,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["Method paragraph using seed lexicon"],
                        },
                        "annotation_id": "ann-4",
                        "worker_id": "writer0",
                    }
                ],
            },
            {
                "question": "Is this extractive?",
                "question_id": "q-extractive",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": ["Conclusion paragraph"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["Conclusion paragraph"],
                        },
                        "annotation_id": "ann-5",
                        "worker_id": "writer0",
                    }
                ],
            },
        ],
        "figures_and_tables": [],
    }
}


class ProcessQasperTests(unittest.TestCase):
    def test_build_unified_records_normalizes_answers_and_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "qasper-train-v0.3.json"
            input_path.write_text(json.dumps(SAMPLE_QASPER), encoding="utf-8")

            records = build_unified_records(
                input_dir=Path(tmp_dir),
                input_paths=[input_path],
                token_counter=TokenCounter("gpt-4o-mini"),
                splits={"train"},
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["dataset_name"], "QASPER")
        self.assertEqual(record["split"], "train")
        self.assertEqual(record["source_article_id"], "1909.00694")
        self.assertIn("Abstract\nThis is the abstract of the paper", record["document"])
        self.assertIn("Introduction", record["document"])
        self.assertIn("Method paragraph using seed lexicon", record["document"])
        self.assertGreater(record["doc_length_tokens"], 0)
        self.assertEqual(record["full_text_section_count"], 3)

        qa_pairs = {item["q_id"]: item for item in record["qa_pairs"]}
        self.assertEqual(set(qa_pairs), {"q-freeform", "q-bool", "q-extractive"})
        self.assertEqual(qa_pairs["q-freeform"]["ground_truth"], "a vocabulary")
        self.assertEqual(qa_pairs["q-freeform"]["alternative_answers"], ["using seed lexicon"])
        self.assertEqual(qa_pairs["q-freeform"]["source_answer_type"], "abstractive")
        self.assertEqual(qa_pairs["q-bool"]["ground_truth"], "Yes")
        self.assertEqual(qa_pairs["q-bool"]["source_answer_type"], "boolean")
        self.assertEqual(qa_pairs["q-extractive"]["ground_truth"], "Conclusion paragraph")
        self.assertEqual(qa_pairs["q-extractive"]["source_answer_type"], "extractive")

    def test_build_unified_records_can_keep_unanswerable_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "validation.json"
            input_path.write_text(json.dumps(SAMPLE_QASPER), encoding="utf-8")

            records = build_unified_records(
                input_dir=Path(tmp_dir),
                input_paths=[input_path],
                token_counter=TokenCounter("gpt-4o-mini"),
                splits={"validation"},
                include_unanswerable=True,
            )

        qa_pairs = {item["q_id"]: item for item in records[0]["qa_pairs"]}
        self.assertIn("q-none", qa_pairs)
        self.assertEqual(qa_pairs["q-none"]["ground_truth"], "Unanswerable")
        self.assertEqual(qa_pairs["q-none"]["source_answer_type"], "none")


if __name__ == "__main__":
    unittest.main()
