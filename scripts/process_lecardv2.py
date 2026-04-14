from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LeCaRDv2 query cases into a document-level QA dataset. "
            "This uses query_allcontext.json as the source document corpus."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./dataset/LeCaRDv2"),
        help="LeCaRDv2 repository directory.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("./dataset/LeCaRDv2/query/query_allcontext.json"),
        help="Path to query_allcontext.json.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/lecardv2/lecardv2_unified.jsonl"),
        help="Output JSONL path.",
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
        help="Optional debug limit on the number of cases processed.",
    )
    return parser.parse_args()


def clean_text(text: Any) -> str:
    value = normalize_text(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    value = re.sub(r"[ \u00a0]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_split_map(input_dir: Path) -> dict[int, str]:
    split_map: dict[int, str] = {}
    for split_name, file_name in (("train", "train_query.json"), ("test", "test_query.json")):
        path = input_dir / "query" / file_name
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            qid = int(row["id"])
            split_map[qid] = split_name
    return split_map


def join_list(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        parts = [clean_text(item) for item in values]
        return "；".join(part for part in parts if part)
    return clean_text(values)


def build_document_text(record: dict[str, Any]) -> str:
    sections: list[str] = []
    title = clean_text(record.get("query")).split("：", 1)[0]
    if title:
        sections.append(f"标题\n{title}")

    for label, field_name in (
        ("全文", "query"),
        ("案情事实", "fact"),
        ("诉辩主张", "claim"),
        ("裁判理由", "reason"),
        ("判决结果", "result"),
    ):
        value = clean_text(record.get(field_name))
        if value:
            sections.append(f"{label}\n{value}")

    law_text = join_list(record.get("law"))
    if law_text:
        sections.append(f"涉及罪名\n{law_text}")

    xf_text = join_list(record.get("xf"))
    if xf_text:
        sections.append(f"适用法条\n{xf_text}")

    return "\n\n".join(sections).strip()


def build_qa_pairs(record: dict[str, Any], doc_id: str) -> list[dict[str, Any]]:
    qa_specs = [
        ("fact", "请概括本案的基本案情事实。", clean_text(record.get("fact"))),
        ("claim", "本案中公诉机关或当事人的主要主张是什么？", clean_text(record.get("claim"))),
        ("reason", "法院的裁判理由是什么？", clean_text(record.get("reason"))),
        ("result", "本案的判决结果是什么？", clean_text(record.get("result"))),
        ("law", "本案涉及的罪名是什么？", join_list(record.get("law"))),
        ("xf", "本案适用了哪些法条编号？", join_list(record.get("xf"))),
    ]

    qa_pairs: list[dict[str, Any]] = []
    for suffix, question, ground_truth in qa_specs:
        if not ground_truth:
            continue
        qa_pairs.append(
            {
                "q_id": f"{doc_id}_{suffix}",
                "question": question,
                "ground_truth": ground_truth,
            }
        )
    return qa_pairs


def build_unified_records(
    input_path: Path,
    input_dir: Path,
    token_counter: TokenCounter,
    limit_documents: int | None = None,
) -> list[dict[str, Any]]:
    split_map = load_split_map(input_dir)
    records: list[dict[str, Any]] = []
    for case_index, row in enumerate(tqdm(iter_jsonl(input_path), desc="LeCaRDv2", unit="case"), start=1):
        if limit_documents is not None and case_index > limit_documents:
            break
        case_id = int(row["id"])
        doc_id = f"LeCaRDv2_{case_id}"
        document = build_document_text(row)
        qa_pairs = build_qa_pairs(row, doc_id)
        if not document or not qa_pairs:
            continue

        records.append(
            {
                "dataset_name": "LeCaRDv2",
                "split": split_map.get(case_id, "all"),
                "doc_id": doc_id,
                "document": document,
                "doc_length_tokens": token_counter.count_text(document),
                "domain": "legal",
                "language": "zh",
                "source_case_id": case_id,
                "qa_pairs": qa_pairs,
            }
        )

    return sorted(records, key=lambda item: item["doc_id"])


def main() -> None:
    args = parse_args()
    token_counter = TokenCounter(args.tokenizer_model)
    records = build_unified_records(
        input_path=args.input_path,
        input_dir=args.input_dir,
        token_counter=token_counter,
        limit_documents=args.limit_documents,
    )

    ensure_parent_dir(args.output_path)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    total_qas = sum(len(record["qa_pairs"]) for record in records)
    print(
        f"LeCaRDv2 processing complete. Output={args.output_path} "
        f"documents={len(records)} qa_pairs={total_qas}"
    )


if __name__ == "__main__":
    main()
