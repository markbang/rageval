from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir
from scripts.process_financebench import (
    clean_text,
    download_pdf,
    extract_pdf_text,
    read_jsonl,
    resolve_pdf_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct an experiment-ready ultra_long_finance dataset by bundling multiple "
            "FinanceBench source documents per company into super-long financial documents."
        )
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("./dataset/FinanceBench/data/financebench_open_source.jsonl"),
    )
    parser.add_argument(
        "--document-info-path",
        type=Path,
        default=Path("./dataset/FinanceBench/data/financebench_document_information.jsonl"),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("./dataset/FinanceBench/pdfs"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./dataset/FinanceBench/text_cache"),
    )
    parser.add_argument(
        "--download-missing-pdfs",
        action="store_true",
        help="Download missing PDFs via FinanceBench doc_link.",
    )
    parser.add_argument(
        "--min-total-tokens",
        type=int,
        default=400000,
        help="Minimum combined token length required for a company bundle.",
    )
    parser.add_argument(
        "--min-doc-count",
        type=int,
        default=4,
        help="Minimum number of source documents required in a bundle.",
    )
    parser.add_argument(
        "--max-bundles",
        type=int,
        default=None,
        help="Optional limit on number of company bundles to output.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/experiment/ultra_long_finance.jsonl"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("./dataset/processed/experiment/ultra_long_finance_manifest.json"),
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer model used for token counting.",
    )
    return parser.parse_args()


def slugify_company(company: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", company).strip("_")
    return slug or "unknown_company"


def sort_key_for_doc(record: dict[str, Any]) -> tuple[int, str, str]:
    raw_period = record.get("doc_period")
    try:
        period = int(raw_period)
    except (TypeError, ValueError):
        period = 0
    return (
        period,
        clean_text(record.get("doc_type")),
        clean_text(record.get("source_document_name")),
    )


def build_financebench_doc_records(
    *,
    questions_path: Path,
    document_info_path: Path,
    pdf_dir: Path,
    token_counter: TokenCounter,
    cache_dir: Path | None = None,
    download_missing_pdfs: bool = False,
    extract_document_text: Callable[..., str] = extract_pdf_text,
    download_document: Callable[[str, Path], Path] = download_pdf,
) -> list[dict[str, Any]]:
    question_rows = read_jsonl(questions_path)
    metadata_by_doc = {
        clean_text(row.get("doc_name")): row
        for row in read_jsonl(document_info_path)
        if clean_text(row.get("doc_name"))
    }

    grouped_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(question_rows, start=1):
        doc_name = clean_text(row.get("doc_name"))
        question = clean_text(row.get("question"))
        answer = clean_text(row.get("answer"))
        if not doc_name or not question or not answer:
            continue
        enriched = dict(row)
        enriched["source_row"] = row_index
        grouped_questions[doc_name].append(enriched)

    records: list[dict[str, Any]] = []
    for doc_name in tqdm(sorted(grouped_questions), desc="FinanceBench-docs", unit="doc"):
        metadata = metadata_by_doc.get(doc_name)
        if metadata is None:
            continue

        doc_link = clean_text(metadata.get("doc_link"))
        try:
            pdf_path = resolve_pdf_path(
                doc_name=doc_name,
                doc_link=doc_link,
                pdf_dir=pdf_dir,
                download_missing_pdfs=download_missing_pdfs,
                download_document=download_document,
            )
            document = extract_document_text(
                pdf_path,
                cache_dir=cache_dir,
                pdftotext_binary="pdftotext",
            )
        except Exception:
            continue

        qa_pairs: list[dict[str, Any]] = []
        for qa_index, row in enumerate(grouped_questions[doc_name], start=1):
            q_id = clean_text(row.get("financebench_id")) or f"{doc_name}_q{qa_index}"
            qa_pairs.append(
                {
                    "q_id": q_id,
                    "question": clean_text(row.get("question")),
                    "ground_truth": clean_text(row.get("answer")),
                    "question_type": clean_text(row.get("question_type")),
                    "question_reasoning": clean_text(row.get("question_reasoning")),
                    "justification": clean_text(row.get("justification")),
                    "source_row": row["source_row"],
                    "source_document_name": doc_name,
                }
            )

        records.append(
            {
                "dataset_name": "FinanceBench",
                "split": "open_source",
                "doc_id": f"FinanceBench_{doc_name}",
                "document": document,
                "doc_length_tokens": token_counter.count_text(document),
                "domain": "financial",
                "language": "en",
                "company": clean_text(metadata.get("company")),
                "gics_sector": clean_text(
                    metadata.get("gics_sector") or metadata.get("comany_sector_gics")
                ),
                "doc_type": clean_text(metadata.get("doc_type")),
                "doc_period": metadata.get("doc_period"),
                "source_document_name": doc_name,
                "source_document_url": doc_link,
                "qa_pairs": qa_pairs,
            }
        )
    return records


def build_ultra_long_finance_records(
    financebench_records: list[dict[str, Any]],
    *,
    token_counter: TokenCounter,
    min_total_tokens: int,
    min_doc_count: int,
    max_bundles: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in financebench_records:
        company = clean_text(record.get("company"))
        if company:
            grouped[company].append(record)

    bundles: list[dict[str, Any]] = []
    for company, docs in sorted(grouped.items()):
        docs_sorted = sorted(docs, key=sort_key_for_doc)
        if len(docs_sorted) < min_doc_count:
            continue

        source_doc_names = [clean_text(doc.get("source_document_name")) for doc in docs_sorted]
        combined_document_parts = []
        combined_qa_pairs: list[dict[str, Any]] = []
        for doc in docs_sorted:
            source_name = clean_text(doc.get("source_document_name"))
            combined_document_parts.append(
                f"===== Source Document: {source_name} =====\n\n{doc['document'].strip()}"
            )
            for qa in doc.get("qa_pairs", []):
                enriched = dict(qa)
                enriched["source_document_name"] = source_name
                combined_qa_pairs.append(enriched)

        combined_document = "\n\n".join(combined_document_parts).strip()
        combined_tokens = token_counter.count_text(combined_document)
        if combined_tokens < min_total_tokens:
            continue

        bundles.append(
            {
                "dataset_name": "UltraLongFinance",
                "split": "ultra_long_finance",
                "doc_id": f"UltraLongFinance_{slugify_company(company)}",
                "document": combined_document,
                "doc_length_tokens": combined_tokens,
                "domain": "financial",
                "language": "en",
                "company": company,
                "source_dataset": "FinanceBench",
                "source_document_count": len(docs_sorted),
                "source_document_names": source_doc_names,
                "qa_pairs": combined_qa_pairs,
            }
        )

    bundles.sort(key=lambda item: (-int(item["doc_length_tokens"]), item["doc_id"]))
    if max_bundles is not None:
        bundles = bundles[:max_bundles]
    return bundles


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_manifest(
    path: Path,
    *,
    input_doc_count: int,
    bundle_records: list[dict[str, Any]],
    min_total_tokens: int,
    min_doc_count: int,
) -> None:
    payload = {
        "source_dataset": "FinanceBench",
        "input_document_count": input_doc_count,
        "bundle_count": len(bundle_records),
        "min_total_tokens": min_total_tokens,
        "min_doc_count": min_doc_count,
        "bundles": [
            {
                "doc_id": record["doc_id"],
                "company": record.get("company"),
                "doc_length_tokens": record.get("doc_length_tokens"),
                "source_document_count": record.get("source_document_count"),
                "qa_pair_count": len(record.get("qa_pairs", [])),
                "source_document_names": record.get("source_document_names", []),
            }
            for record in bundle_records
        ],
    }
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    token_counter = TokenCounter(args.tokenizer_model)

    doc_records = build_financebench_doc_records(
        questions_path=args.questions_path,
        document_info_path=args.document_info_path,
        pdf_dir=args.pdf_dir,
        token_counter=token_counter,
        cache_dir=args.cache_dir,
        download_missing_pdfs=args.download_missing_pdfs,
    )
    bundle_records = build_ultra_long_finance_records(
        doc_records,
        token_counter=token_counter,
        min_total_tokens=args.min_total_tokens,
        min_doc_count=args.min_doc_count,
        max_bundles=args.max_bundles,
    )
    write_jsonl(args.output_path, bundle_records)
    write_manifest(
        args.manifest_path,
        input_doc_count=len(doc_records),
        bundle_records=bundle_records,
        min_total_tokens=args.min_total_tokens,
        min_doc_count=args.min_doc_count,
    )
    print(
        f"UltraLongFinance preparation complete. "
        f"source_docs={len(doc_records)} bundles={len(bundle_records)} output={args.output_path}"
    )


if __name__ == "__main__":
    main()
