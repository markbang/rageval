from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - dependency managed at runtime
    def tqdm(iterable, **_kwargs):
        return iterable

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir, normalize_text


DEFAULT_SPLITS = ("train", "validation", "test")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "validation",
    "validation": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate QASPER into a document-level JSONL format with qa_pairs. "
            "This expects official QASPER JSON/JSONL files and preserves split metadata."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./dataset/QASPER"),
        help="Directory containing official QASPER JSON/JSONL files.",
    )
    parser.add_argument(
        "--input-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit input file paths. If omitted, JSON/JSONL files are discovered under --input-dir.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/qasper/qasper_unified.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--default-split",
        default=None,
        help="Fallback split label to use when it cannot be inferred from an input filename.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to include. Defaults to train validation test.",
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


def canonicalize_split(split: str | None) -> str | None:
    if split is None:
        return None
    return SPLIT_ALIASES.get(clean_text(split).lower())


def infer_split_from_path(path: Path, default_split: str | None = None) -> str | None:
    tokens = re.split(r"[^a-z0-9]+", path.as_posix().lower())
    for token in tokens:
        canonical = canonicalize_split(token)
        if canonical is not None:
            return canonical
    return canonicalize_split(default_split)


def discover_input_paths(input_dir: Path, input_paths: list[Path] | None = None) -> list[Path]:
    if input_paths:
        return sorted(path for path in input_paths if path.is_file())

    discovered = sorted(path for path in input_dir.rglob("*.json") if path.is_file())
    discovered.extend(sorted(path for path in input_dir.rglob("*.jsonl") if path.is_file()))
    return sorted(set(discovered))


def safe_doc_suffix(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return slug or "unknown"


def iter_articles_from_json(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for article_id, article in payload.items():
            if not isinstance(article, dict):
                continue
            merged = dict(article)
            merged.setdefault("id", article_id)
            yield str(article_id), merged
        return

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            article_id = clean_text(item.get("id") or item.get("article_id"))
            if article_id:
                yield article_id, item


def iter_articles_from_jsonl(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            article_id = clean_text(item.get("id") or item.get("article_id"))
            if article_id:
                yield article_id, item


def iter_input_articles(
    input_dir: Path,
    input_paths: list[Path] | None,
    allowed_splits: set[str],
    default_split: str | None = None,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    paths = discover_input_paths(input_dir, input_paths=input_paths)
    if not paths:
        raise FileNotFoundError(
            f"No QASPER JSON/JSONL files found under {input_dir}. Download the official dataset from "
            "https://allenai.org/data/qasper or export allenai/qasper from Hugging Face, then rerun this script."
        )

    for path in paths:
        split = infer_split_from_path(path, default_split=default_split)
        if split is None:
            raise ValueError(
                f"Unable to infer split from {path}. Rename the file to include train/validation/test "
                "or pass --default-split."
            )
        if split not in allowed_splits:
            continue

        if path.suffix == ".json":
            iterator = iter_articles_from_json(path)
        elif path.suffix == ".jsonl":
            iterator = iter_articles_from_jsonl(path)
        else:
            continue

        for article_id, article in iterator:
            yield split, article_id, article


def build_document_text(article: dict[str, Any]) -> str:
    parts: list[str] = []

    title = clean_text(article.get("title"))
    if title:
        parts.append(title)

    abstract = clean_text(article.get("abstract"))
    if abstract:
        parts.append(f"Abstract\n{abstract}")

    for section in article.get("full_text") or []:
        if not isinstance(section, dict):
            continue
        section_name = clean_text(section.get("section_name"))
        if section_name:
            parts.append(section_name)
        for paragraph in section.get("paragraphs") or []:
            paragraph_text = clean_text(paragraph)
            if paragraph_text:
                parts.append(paragraph_text)

    return "\n\n".join(parts).strip()


def normalize_answer(answer_payload: dict[str, Any]) -> tuple[str, str, list[str]]:
    evidence = [clean_text(item) for item in answer_payload.get("evidence") or []]
    evidence = [item for item in evidence if item]

    if answer_payload.get("unanswerable", False):
        return "Unanswerable", "none", evidence

    if answer_payload.get("yes_no") is not None:
        return ("Yes" if answer_payload.get("yes_no") else "No"), "boolean", evidence

    extractive_spans = [clean_text(item) for item in answer_payload.get("extractive_spans") or []]
    extractive_spans = [item for item in extractive_spans if item]
    if extractive_spans:
        return ", ".join(extractive_spans), "extractive", evidence

    free_form = clean_text(answer_payload.get("free_form_answer"))
    return free_form, "abstractive", evidence


def build_qa_pairs(
    article: dict[str, Any],
    doc_id: str,
    include_unanswerable: bool = False,
) -> list[dict[str, Any]]:
    qa_pairs: list[dict[str, Any]] = []

    for qa_index, qa in enumerate(article.get("qas") or [], start=1):
        if not isinstance(qa, dict):
            continue

        question = clean_text(qa.get("question"))
        question_id = clean_text(qa.get("question_id")) or f"{doc_id}_q{qa_index}"
        if not question:
            continue

        normalized_answers: list[dict[str, Any]] = []
        for annotation in qa.get("answers") or []:
            if not isinstance(annotation, dict):
                continue
            answer_payload = annotation.get("answer")
            if not isinstance(answer_payload, dict):
                continue
            answer_text, answer_type, evidence = normalize_answer(answer_payload)
            if not answer_text:
                continue
            if answer_text == "Unanswerable" and not include_unanswerable:
                continue
            normalized_answers.append(
                {
                    "text": answer_text,
                    "answer_type": answer_type,
                    "evidence": evidence,
                    "annotation_id": clean_text(annotation.get("annotation_id")),
                    "worker_id": clean_text(annotation.get("worker_id")),
                }
            )

        if not normalized_answers:
            continue

        selected = normalized_answers[0]
        alternative_answers: list[str] = []
        for item in normalized_answers[1:]:
            text = item["text"]
            if text != selected["text"] and text not in alternative_answers:
                alternative_answers.append(text)

        qa_record: dict[str, Any] = {
            "q_id": question_id,
            "question": question,
            "ground_truth": selected["text"],
            "source_answer_type": selected["answer_type"],
            "source_annotation_count": len(normalized_answers),
            "source_question_writer": clean_text(qa.get("question_writer")) or None,
            "source_nlp_background": clean_text(qa.get("nlp_background")) or None,
            "source_topic_background": clean_text(qa.get("topic_background")) or None,
            "source_paper_read": clean_text(qa.get("paper_read")) or None,
            "source_search_query": clean_text(qa.get("search_query")) or None,
        }
        if selected["annotation_id"]:
            qa_record["source_annotation_id"] = selected["annotation_id"]
        if selected["worker_id"]:
            qa_record["source_worker_id"] = selected["worker_id"]
        if selected["evidence"]:
            qa_record["evidence"] = selected["evidence"]
        if alternative_answers:
            qa_record["alternative_answers"] = alternative_answers

        qa_pairs.append({key: value for key, value in qa_record.items() if value is not None})

    return qa_pairs


def build_unified_records(
    input_dir: Path,
    token_counter: TokenCounter,
    splits: set[str],
    input_paths: list[Path] | None = None,
    default_split: str | None = None,
    include_unanswerable: bool = False,
    limit_documents: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for split, article_id, article in tqdm(
        iter_input_articles(
            input_dir=input_dir,
            input_paths=input_paths,
            allowed_splits=splits,
            default_split=default_split,
        ),
        desc="QASPER",
        unit="doc",
    ):
        document = build_document_text(article)
        if not document:
            continue

        safe_article_id = safe_doc_suffix(article_id)
        doc_id = f"QASPER_{split}_{safe_article_id}"
        qa_pairs = build_qa_pairs(
            article=article,
            doc_id=doc_id,
            include_unanswerable=include_unanswerable,
        )
        if not qa_pairs:
            continue

        figures_and_tables = article.get("figures_and_tables") or []
        full_text = article.get("full_text") or []
        records.append(
            {
                "dataset_name": "QASPER",
                "split": split,
                "doc_id": doc_id,
                "document": document,
                "doc_length_tokens": token_counter.count_text(document),
                "domain": "scientific",
                "language": "en",
                "source_article_id": article_id,
                "title": clean_text(article.get("title")),
                "full_text_section_count": sum(1 for item in full_text if isinstance(item, dict)),
                "figure_table_count": len(figures_and_tables) if isinstance(figures_and_tables, list) else 0,
                "qa_pairs": qa_pairs,
            }
        )

    records = sorted(records, key=lambda item: (item["split"], item["doc_id"]))
    if limit_documents is not None:
        records = records[:limit_documents]
    return records


def main() -> None:
    args = parse_args()
    token_counter = TokenCounter(args.tokenizer_model)
    records = build_unified_records(
        input_dir=args.input_dir,
        input_paths=args.input_paths,
        token_counter=token_counter,
        splits={canonicalize_split(split) or split for split in args.splits},
        default_split=args.default_split,
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
        f"QASPER processing complete. Output={args.output_path} "
        f"documents={len(records)} qa_pairs={total_qas}"
    )


if __name__ == "__main__":
    main()
