from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rageval.token_tracking import TokenCounter
from scripts.prepare_ultra_long_finance import (
    build_financebench_doc_records,
    build_ultra_long_finance_records,
)


class PrepareUltraLongFinanceTests(unittest.TestCase):
    def test_build_ultra_long_finance_records_bundles_company_docs_above_threshold(self) -> None:
        token_counter = TokenCounter("gpt-4o-mini")
        financebench_records = [
            {
                "dataset_name": "FinanceBench",
                "split": "open_source",
                "doc_id": "FinanceBench_ACME_2020_10K",
                "document": "alpha " * 1000,
                "doc_length_tokens": token_counter.count_text("alpha " * 1000),
                "domain": "financial",
                "language": "en",
                "company": "Acme",
                "doc_type": "10k",
                "doc_period": 2020,
                "source_document_name": "ACME_2020_10K",
                "qa_pairs": [{"q_id": "q1", "question": "Q1", "ground_truth": "A1"}],
            },
            {
                "dataset_name": "FinanceBench",
                "split": "open_source",
                "doc_id": "FinanceBench_ACME_2021_10K",
                "document": "beta " * 1000,
                "doc_length_tokens": token_counter.count_text("beta " * 1000),
                "domain": "financial",
                "language": "en",
                "company": "Acme",
                "doc_type": "10k",
                "doc_period": 2021,
                "source_document_name": "ACME_2021_10K",
                "qa_pairs": [{"q_id": "q2", "question": "Q2", "ground_truth": "A2"}],
            },
            {
                "dataset_name": "FinanceBench",
                "split": "open_source",
                "doc_id": "FinanceBench_ACME_2022_10K",
                "document": "gamma " * 1000,
                "doc_length_tokens": token_counter.count_text("gamma " * 1000),
                "domain": "financial",
                "language": "en",
                "company": "Acme",
                "doc_type": "10k",
                "doc_period": 2022,
                "source_document_name": "ACME_2022_10K",
                "qa_pairs": [{"q_id": "q3", "question": "Q3", "ground_truth": "A3"}],
            },
            {
                "dataset_name": "FinanceBench",
                "split": "open_source",
                "doc_id": "FinanceBench_OTHER_2022_10K",
                "document": "delta " * 500,
                "doc_length_tokens": token_counter.count_text("delta " * 500),
                "domain": "financial",
                "language": "en",
                "company": "Other",
                "doc_type": "10k",
                "doc_period": 2022,
                "source_document_name": "OTHER_2022_10K",
                "qa_pairs": [{"q_id": "q4", "question": "Q4", "ground_truth": "A4"}],
            },
        ]

        bundles = build_ultra_long_finance_records(
            financebench_records,
            token_counter=token_counter,
            min_total_tokens=2500,
            min_doc_count=3,
        )

        self.assertEqual(len(bundles), 1)
        bundle = bundles[0]
        self.assertEqual(bundle["dataset_name"], "UltraLongFinance")
        self.assertEqual(bundle["company"], "Acme")
        self.assertEqual(bundle["source_document_count"], 3)
        self.assertEqual(len(bundle["qa_pairs"]), 3)
        self.assertGreaterEqual(bundle["doc_length_tokens"], 2500)
        self.assertIn("Source Document: ACME_2020_10K", bundle["document"])

    def test_build_financebench_doc_records_skips_failed_downloads(self) -> None:
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
                },
                {
                    "financebench_id": "fb-2",
                    "doc_name": "BROKEN_2023_10K",
                    "question": "What is profit?",
                    "answer": "$5",
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
                },
                {
                    "doc_name": "BROKEN_2023_10K",
                    "company": "Broken Corp",
                    "gics_sector": "Industrials",
                    "doc_type": "10k",
                    "doc_period": 2023,
                    "doc_link": "https://example.com/broken.pdf",
                },
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
                if "broken" in doc_link:
                    raise RuntimeError("download failed")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"%PDF-1.4")
                return destination

            def fake_extract(pdf_path: Path, **_: object) -> str:
                return "Revenue was 10."

            records = build_financebench_doc_records(
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
            self.assertEqual(records[0]["company"], "Acme Corp")


if __name__ == "__main__":
    unittest.main()
