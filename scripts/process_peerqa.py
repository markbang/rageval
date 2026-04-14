from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate PeerQA into a document-level JSONL format with qa_pairs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./dataset/PeerQA/data"),
        help="Directory containing PeerQA data files such as papers.jsonl and qa.jsonl.",
    )
    parser.add_argument(
        "--papers-path",
        type=Path,
        default=None,
        help="Override path to papers.jsonl.",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=None,
        help="Override path to qa.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/peerqa/peerqa_unified.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer model used for doc_length_tokens.",
    )
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Keep unanswerable questions and map their ground truth to a fixed string.",
    )
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=None,
        help="Optional debug limit on the number of output documents.",
    )
    return parser.parse_args()


def clean_text(text: Any) -> str:
    value = normalize_text(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
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


def required_data_message(data_dir: Path) -> str:
    return (
        f"PeerQA data files are missing under {data_dir}. Expected files include papers.jsonl "
        "and qa.jsonl. Prepare them by following dataset/PeerQA/README.md: "
        "download the PeerQA question package, then download/extract the paper PDFs to produce papers.jsonl."
    )


def load_papers(papers_path: Path) -> dict[str, dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(papers_path):
        paper_id = clean_text(row.get("paper_id"))
        if not paper_id:
            continue
        grouped_rows[paper_id].append(row)

    papers: dict[str, dict[str, Any]] = {}
    for paper_id, rows in grouped_rows.items():
        rows = sorted(rows, key=lambda item: int(item.get("idx", 0)))
        title = ""
        body_parts: list[str] = []
        for row in rows:
            content = clean_text(row.get("content"))
            if not content:
                continue
            if row.get("type") == "title" and not title:
                title = content
            body_parts.append(content)
        document = "\n\n".join(body_parts).strip()
        if not document:
            continue
        papers[paper_id] = {
            "title": title,
            "document": document,
        }
    return papers


def build_unified_records(
    papers_path: Path,
    qa_path: Path,
    token_counter: TokenCounter,
    include_unanswerable: bool = False,
    limit_documents: int | None = None,
) -> list[dict[str, Any]]:
    papers = load_papers(papers_path)
    qa_rows = read_jsonl(qa_path)

    grouped: dict[str, dict[str, Any]] = {}

    for row_index, row in enumerate(tqdm(qa_rows, desc="PeerQA", unit="qa"), start=1):
        paper_id = clean_text(row.get("paper_id"))
        question_id = clean_text(row.get("question_id"))
        question = clean_text(row.get("question"))
        answer = clean_text(row.get("answer_free_form"))
        answerable = row.get("answerable_mapped")

        paper = papers.get(paper_id)
        if paper is None or not question_id or not question:
            continue

        if not answer:
            if include_unanswerable and answerable is False:
                answer = "The paper does not contain enough information to answer this question."
            else:
                continue

        entry = grouped.get(paper_id)
        if entry is None:
            title = clean_text(paper.get("title"))
            document = clean_text(paper["document"])
            if title and not document.startswith(title):
                document = f"{title}\n\n{document}"
            source_prefix = paper_id.split("-", 1)[0] if "-" in paper_id else paper_id
            entry = {
                "dataset_name": "PeerQA",
                "doc_id": f"PeerQA_{paper_id}",
                "document": document,
                "doc_length_tokens": token_counter.count_text(document),
                "domain": "scientific",
                "language": "en",
                "source_paper_id": paper_id,
                "paper_source": source_prefix,
                "qa_pairs": [],
            }
            grouped[paper_id] = entry

        entry["qa_pairs"].append(
            {
                "q_id": question_id,
                "question": question,
                "ground_truth": answer,
                "source_row": row_index,
            }
        )

    records = sorted(grouped.values(), key=lambda item: item["doc_id"])
    if limit_documents is not None:
        records = records[:limit_documents]
    return records


def main() -> None:
    args = parse_args()
    papers_path = args.papers_path or (args.data_dir / "papers.jsonl")
    qa_path = args.qa_path or (args.data_dir / "qa.jsonl")

    if not papers_path.exists() or not qa_path.exists():
        raise FileNotFoundError(required_data_message(args.data_dir))

    token_counter = TokenCounter(args.tokenizer_model)
    records = build_unified_records(
        papers_path=papers_path,
        qa_path=qa_path,
        token_counter=token_counter,
        include_unanswerable=args.include_unanswerable,
        limit_documents=args.limit_documents,
    )

    ensure_parent_dir(args.output_path)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    total_qas = sum(len(record["qa_pairs"]) for record in records)
    print(
        f"PeerQA processing complete. Output={args.output_path} "
        f"documents={len(records)} qa_pairs={total_qas}"
    )


if __name__ == "__main__":
    main()
