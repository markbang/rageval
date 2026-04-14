from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rageval.token_tracking import TokenCounter
from scripts.process_financebench import build_unified_records
from scripts.process_financebench import extract_pdf_text
from scripts.process_financebench import resolve_pdf_path


class ProcessFinanceBenchTests(unittest.TestCase):
    def test_build_unified_records_groups_questions_by_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            questions_path = root / "financebench_open_source.jsonl"
            document_info_path = root / "financebench_document_information.jsonl"
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()

            questions = [
                {
                    "financebench_id": "fb-1",
                    "doc_name": "ACME_2023_10K",
                    "question": "What is revenue?",
                    "answer": "$10",
                    "dataset_subset_label": "OPEN_SOURCE",
                    "question_type": "metrics-generated",
                    "question_reasoning": "Information extraction",
                    "justification": "Look at the income statement.",
                },
                {
                    "financebench_id": "fb-2",
                    "doc_name": "ACME_2023_10K",
                    "question": "What is net income?",
                    "answer": "$3",
                    "dataset_subset_label": "OPEN_SOURCE",
                    "question_type": "novel-generated",
                    "question_reasoning": "Numerical reasoning",
                    "justification": "Look at the income statement.",
                },
            ]
            metadata = [
                {
                    "doc_name": "ACME_2023_10K",
                    "company": "Acme Corp",
                    "gics_sector": "Industrials",
                    "doc_type": "10k",
                    "doc_period": 2023,
                    "doc_link": "https://example.com/acme.pdf",
                }
            ]
            questions_path.write_text(
                "\n".join(json.dumps(item) for item in questions) + "\n",
                encoding="utf-8",
            )
            document_info_path.write_text(
                "\n".join(json.dumps(item) for item in metadata) + "\n",
                encoding="utf-8",
            )

            def fake_download(doc_link: str, destination: Path) -> Path:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"%PDF-1.4 test")
                return destination

            def fake_extract(pdf_path: Path, **_: object) -> str:
                self.assertEqual(pdf_path.name, "ACME_2023_10K.pdf")
                return "Revenue was 10 and net income was 3."

            records = build_unified_records(
                questions_path=questions_path,
                document_info_path=document_info_path,
                pdf_dir=pdf_dir,
                token_counter=TokenCounter("gpt-4o-mini"),
                download_missing_pdfs=True,
                cache_dir=None,
                extract_document_text=fake_extract,
                download_document=fake_download,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["dataset_name"], "FinanceBench")
            self.assertEqual(record["split"], "open_source")
            self.assertEqual(record["doc_id"], "FinanceBench_ACME_2023_10K")
            self.assertEqual(record["company"], "Acme Corp")
            self.assertEqual(record["doc_type"], "10k")
            self.assertEqual(record["doc_period"], 2023)
            self.assertEqual(len(record["qa_pairs"]), 2)
            self.assertEqual(record["qa_pairs"][0]["q_id"], "fb-1")
            self.assertGreater(record["doc_length_tokens"], 0)

    def test_resolve_pdf_path_requires_local_pdf_without_download_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError):
                resolve_pdf_path(
                    doc_name="ACME_2023_10K",
                    doc_link="https://example.com/acme.pdf",
                    pdf_dir=pdf_dir,
                    download_missing_pdfs=False,
                )

    def test_extract_pdf_text_uses_cache_before_invoking_pdftotext(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            cache_dir = root / "cache"
            cache_dir.mkdir()
            (cache_dir / "sample.txt").write_text("cached text", encoding="utf-8")

            with patch("scripts.process_financebench.subprocess.run") as run_mock:
                text = extract_pdf_text(pdf_path, cache_dir=cache_dir)

            self.assertEqual(text, "cached text")
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
