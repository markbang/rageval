from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir, normalize_text


DEFAULT_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate NarrativeQA into a document-level JSONL format with qa_pairs. "
            "By default this uses Wikipedia summaries as documents."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./dataset/narrativeqa"),
        help="NarrativeQA repository directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/narrativeqa/narrativeqa_unified.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--stories-dir",
        type=Path,
        default=None,
        help="Directory containing downloaded full stories, e.g. dataset/narrativeqa/tmp.",
    )
    parser.add_argument(
        "--use-full-story-if-available",
        action="store_true",
        help="Prefer downloaded full stories when available; otherwise fall back to summaries.",
    )
    parser.add_argument(
        "--full-story-min-chars",
        type=int,
        default=19000,
        help="Minimum character length required to accept a downloaded story file.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to include. Defaults to train valid test.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer model used for doc_length_tokens.",
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**8)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(10**8)
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_document_metadata(input_dir: Path) -> dict[str, dict[str, str]]:
    path = input_dir / "documents.csv"
    rows = read_csv_rows(path)
    return {row["document_id"]: row for row in rows if row.get("document_id")}


def load_summaries(input_dir: Path) -> dict[str, dict[str, str]]:
    path = input_dir / "third_party" / "wikipedia" / "summaries.csv"
    rows = read_csv_rows(path)
    return {row["document_id"]: row for row in rows if row.get("document_id")}


def load_story_text(
    stories_dir: Path | None,
    document_id: str,
    min_chars: int,
) -> tuple[str | None, str]:
    if stories_dir is None:
        return None, "summary"

    story_path = stories_dir / f"{document_id}.content"
    if not story_path.exists():
        return None, "summary"

    content = clean_text(story_path.read_text(encoding="utf-8", errors="ignore"))
    if len(content) < min_chars:
        return None, "summary"

    return content, "story"


def build_unified_records(
    input_dir: Path,
    token_counter: TokenCounter,
    splits: set[str],
    use_full_story_if_available: bool,
    stories_dir: Path | None,
    full_story_min_chars: int,
    limit_documents: int | None = None,
) -> list[dict[str, Any]]:
    documents_by_id = load_document_metadata(input_dir)
    summaries_by_id = load_summaries(input_dir)

    grouped: dict[str, dict[str, Any]] = {}

    for row_index, row in enumerate(
        tqdm(iter_csv_rows(input_dir / "qaps.csv"), desc="NarrativeQA", unit="row"),
        start=1,
    ):
        document_id = row.get("document_id", "").strip()
        question = clean_text(row.get("question"))
        answer1 = clean_text(row.get("answer1"))
        answer2 = clean_text(row.get("answer2"))
        meta = documents_by_id.get(document_id)
        summary = summaries_by_id.get(document_id)

        if not document_id or meta is None or summary is None:
            continue

        split = (meta.get("set") or summary.get("set") or row.get("set") or "").strip()
        if split not in splits:
            continue

        if not question or not answer1:
            continue

        entry = grouped.get(document_id)
        if entry is None:
            story_text = None
            document_source = "summary"
            if use_full_story_if_available:
                story_text, document_source = load_story_text(
                    stories_dir=stories_dir,
                    document_id=document_id,
                    min_chars=full_story_min_chars,
                )

            document_text = clean_text(story_text or summary.get("summary"))
            if not document_text:
                continue

            wiki_title = clean_text(meta.get("wiki_title"))
            if wiki_title:
                document_text = f"{wiki_title}\n\n{document_text}"

            entry = {
                "dataset_name": "NarrativeQA",
                "split": split,
                "doc_id": f"NarrativeQA_{document_id}",
                "document": document_text,
                "doc_length_tokens": token_counter.count_text(document_text),
                "domain": clean_text(meta.get("kind")) or None,
                "language": "en",
                "source_document_id": document_id,
                "document_source": document_source,
                "wiki_title": wiki_title,
                "story_url": clean_text(meta.get("story_url")),
                "wiki_url": clean_text(meta.get("wiki_url")),
                "story_word_count": clean_text(meta.get("story_word_count")),
                "qa_pairs": [],
            }
            grouped[document_id] = entry

        qa_index = len(entry["qa_pairs"]) + 1
        qa_record: dict[str, Any] = {
            "q_id": f"{entry['doc_id']}_q{qa_index}",
            "question": question,
            "ground_truth": answer1,
            "source_row": row_index,
        }
        if answer2 and answer2 != answer1:
            qa_record["alternative_answers"] = [answer2]
        entry["qa_pairs"].append(qa_record)

    records = sorted(grouped.values(), key=lambda item: item["doc_id"])
    if limit_documents is not None:
        records = records[:limit_documents]
    return records


def main() -> None:
    args = parse_args()
    token_counter = TokenCounter(args.tokenizer_model)
    stories_dir = args.stories_dir
    if stories_dir is None and args.use_full_story_if_available:
        stories_dir = args.input_dir / "tmp"

    records = build_unified_records(
        input_dir=args.input_dir,
        token_counter=token_counter,
        splits=set(args.splits),
        use_full_story_if_available=args.use_full_story_if_available,
        stories_dir=stories_dir,
        full_story_min_chars=args.full_story_min_chars,
        limit_documents=args.limit_documents,
    )

    ensure_parent_dir(args.output_path)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    total_qas = sum(len(record["qa_pairs"]) for record in records)
    print(
        f"NarrativeQA processing complete. Output={args.output_path} "
        f"documents={len(records)} qa_pairs={total_qas}"
    )


if __name__ == "__main__":
    main()
