from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
from urllib.request import Request, urlopen

from tqdm import tqdm

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate FinanceBench into a document-level JSONL format with qa_pairs. "
            "By default the script expects the official open-source question metadata plus local PDFs."
        )
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("./dataset/FinanceBench/data/financebench_open_source.jsonl"),
        help="Path to the official FinanceBench open-source question JSONL.",
    )
    parser.add_argument(
        "--document-info-path",
        type=Path,
        default=Path("./dataset/FinanceBench/data/financebench_document_information.jsonl"),
        help="Path to the official FinanceBench document metadata JSONL.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("./dataset/FinanceBench/pdfs"),
        help="Directory containing FinanceBench source PDFs named <doc_name>.pdf.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/financebench/financebench_unified.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer model used for doc_length_tokens.",
    )
    parser.add_argument(
        "--download-missing-pdfs",
        action="store_true",
        help="Download missing PDFs from the official doc_link field into --pdf-dir.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./dataset/FinanceBench/text_cache"),
        help="Optional directory for cached pdftotext output.",
    )
    parser.add_argument(
        "--pdftotext-binary",
        default="pdftotext",
        help="pdftotext executable used to extract source documents.",
    )
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=None,
        help="Optional debug limit on the number of documents processed.",
    )
    return parser.parse_args()


def clean_text(text: Any) -> str:
    value = normalize_text(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \u00a0]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def required_data_message() -> str:
    return (
        "FinanceBench source files are missing. Download the official open-source metadata files "
        "(financebench_open_source.jsonl and financebench_document_information.jsonl) into "
        "dataset/FinanceBench/data/, and place PDFs under dataset/FinanceBench/pdfs/ or rerun "
        "with --download-missing-pdfs. Official source: "
        "https://github.com/patronus-ai/financebench"
    )


def cache_path_for_pdf(pdf_path: Path, cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{pdf_path.stem}.txt"


def extract_pdf_text(
    pdf_path: Path,
    *,
    cache_dir: Path | None = None,
    pdftotext_binary: str = "pdftotext",
) -> str:
    cached_path = cache_path_for_pdf(pdf_path, cache_dir)
    if cached_path is not None and cached_path.exists():
        return clean_text(cached_path.read_text(encoding="utf-8"))

    command = [
        pdftotext_binary,
        "-layout",
        "-nopgbrk",
        "-enc",
        "UTF-8",
        str(pdf_path),
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftotext failed for {pdf_path}: {(result.stderr or result.stdout).strip()}"
        )

    text = clean_text(result.stdout)
    if not text:
        raise ValueError(f"Extracted empty text from {pdf_path}")

    if cached_path is not None:
        ensure_parent_dir(cached_path)
        cached_path.write_text(text, encoding="utf-8")

    return text


def download_pdf(doc_link: str, destination: Path) -> Path:
    ensure_parent_dir(destination)
    request = Request(
        doc_link,
        headers={
            "User-Agent": "rageval-financebench/1.0 (+https://github.com/patronus-ai/financebench)"
        },
    )
    with urlopen(request, timeout=120) as response:  # nosec B310 - official dataset download
        destination.write_bytes(response.read())
    return destination


def resolve_pdf_path(
    *,
    doc_name: str,
    doc_link: str,
    pdf_dir: Path,
    download_missing_pdfs: bool,
    download_document: Callable[[str, Path], Path] = download_pdf,
) -> Path:
    candidate = pdf_dir / f"{doc_name}.pdf"
    if candidate.exists():
        return candidate

    if download_missing_pdfs:
        return download_document(doc_link, candidate)

    raise FileNotFoundError(
        f"Missing PDF for {doc_name}: expected {candidate}. "
        "Provide the official PDF locally or rerun with --download-missing-pdfs."
    )


def build_unified_records(
    *,
    questions_path: Path,
    document_info_path: Path,
    pdf_dir: Path,
    token_counter: TokenCounter,
    download_missing_pdfs: bool = False,
    cache_dir: Path | None = None,
    pdftotext_binary: str = "pdftotext",
    limit_documents: int | None = None,
    extract_document_text: Callable[..., str] = extract_pdf_text,
    download_document: Callable[[str, Path], Path] = download_pdf,
) -> list[dict[str, Any]]:
    if not questions_path.exists() or not document_info_path.exists():
        raise FileNotFoundError(required_data_message())

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

    selected_doc_names = sorted(grouped_questions)
    if limit_documents is not None:
        selected_doc_names = selected_doc_names[:limit_documents]

    records: list[dict[str, Any]] = []
    for doc_name in tqdm(selected_doc_names, desc="FinanceBench", unit="doc"):
        metadata = metadata_by_doc.get(doc_name)
        if metadata is None:
            raise KeyError(f"Missing metadata for FinanceBench doc {doc_name}")

        doc_link = clean_text(metadata.get("doc_link"))
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
            pdftotext_binary=pdftotext_binary,
        )

        qa_rows = grouped_questions[doc_name]
        subset_labels = {
            clean_text(row.get("dataset_subset_label")).lower()
            for row in qa_rows
            if clean_text(row.get("dataset_subset_label"))
        }
        split = next(iter(sorted(subset_labels))) if subset_labels else None

        record = {
            "dataset_name": "FinanceBench",
            "split": split,
            "doc_id": f"FinanceBench_{doc_name}",
            "document": document,
            "doc_length_tokens": token_counter.count_text(document),
            "domain": "financial",
            "language": "en",
            "company": clean_text(metadata.get("company")),
            "gics_sector": clean_text(metadata.get("gics_sector") or metadata.get("comany_sector_gics")),
            "doc_type": clean_text(metadata.get("doc_type")),
            "doc_period": metadata.get("doc_period"),
            "source_document_name": doc_name,
            "source_document_url": doc_link,
            "qa_pairs": [],
        }

        for qa_index, row in enumerate(qa_rows, start=1):
            q_id = clean_text(row.get("financebench_id")) or f"{record['doc_id']}_q{qa_index}"
            qa_record = {
                "q_id": q_id,
                "question": clean_text(row.get("question")),
                "ground_truth": clean_text(row.get("answer")),
                "question_type": clean_text(row.get("question_type")),
                "question_reasoning": clean_text(row.get("question_reasoning")),
                "justification": clean_text(row.get("justification")),
                "source_row": row["source_row"],
            }
            record["qa_pairs"].append(qa_record)

        records.append(record)

    return records


def main() -> None:
    args = parse_args()

    token_counter = TokenCounter(args.tokenizer_model)
    records = build_unified_records(
        questions_path=args.questions_path,
        document_info_path=args.document_info_path,
        pdf_dir=args.pdf_dir,
        token_counter=token_counter,
        download_missing_pdfs=args.download_missing_pdfs,
        cache_dir=args.cache_dir,
        pdftotext_binary=args.pdftotext_binary,
        limit_documents=args.limit_documents,
    )

    ensure_parent_dir(args.output_path)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    total_qas = sum(len(record["qa_pairs"]) for record in records)
    print(
        f"FinanceBench processing complete. Output={args.output_path} "
        f"documents={len(records)} qa_pairs={total_qas}"
    )


if __name__ == "__main__":
    main()
