from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import ExitStack
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir


DEFAULT_SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate DocFinQA into a document-level JSONL format with qa_pairs. "
            "This implementation uses on-disk sharding to avoid loading multi-GB JSON files into memory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./dataset/DocFinQA"),
        help="Directory containing DocFinQA split files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to process. Defaults to train dev test.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/docfinqa/docfinqa_unified.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer model used for doc_length_tokens.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debug limit on the number of raw QA rows processed per split.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=64,
        help="Number of temporary shards used during low-memory aggregation.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Optional directory for temporary shard files. Defaults to a temporary directory near the output file.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    cleaned = re.sub(r"[ \u00a0]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def stable_doc_hash(document: str) -> str:
    return hashlib.sha1(document.encode("utf-8")).hexdigest()[:16]


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    in_array = False
    chunk_size = 4 * 1024 * 1024

    with path.open("r", encoding="utf-8") as handle:
        reached_eof = False
        while True:
            if position >= len(buffer) and reached_eof:
                break

            if not reached_eof and len(buffer) - position < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer = buffer[position:] + chunk
                    position = 0
                else:
                    reached_eof = True
                    buffer = buffer[position:]
                    position = 0

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if not in_array:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} is not a JSON array.")
                    in_array = True
                    position += 1
                    continue

                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if position >= len(buffer):
                    break

                if buffer[position] == "]":
                    return
                if buffer[position] == ",":
                    position += 1
                    continue

                try:
                    obj, end_position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if reached_eof:
                        raise
                    chunk = handle.read(chunk_size)
                    if chunk:
                        buffer = buffer[position:] + chunk
                        position = 0
                        continue
                    reached_eof = True
                    buffer = buffer[position:]
                    position = 0
                    continue

                position = end_position
                if isinstance(obj, dict):
                    yield obj

            if reached_eof and position >= len(buffer):
                break


def normalize_row(row: dict[str, Any]) -> tuple[str, str, str] | None:
    document = clean_text(str(row.get("Context", "")))
    question = clean_text(str(row.get("Question", "")))
    ground_truth = clean_text(str(row.get("Answer", "")))
    if not document or not question or not ground_truth:
        return None
    return document, question, ground_truth


def spool_split_rows(
    split: str,
    input_path: Path,
    temp_dir: Path,
    shard_count: int,
    limit_records: int | None = None,
) -> tuple[list[Path], int]:
    shard_paths = [temp_dir / f"{split}_shard_{index:03d}.jsonl" for index in range(shard_count)]
    processed_rows = 0

    with ExitStack() as stack:
        shard_handles = [stack.enter_context(path.open("w", encoding="utf-8")) for path in shard_paths]
        for row_index, row in enumerate(tqdm(iter_json_array(input_path), desc=f"DocFinQA {split}", unit="row"), start=1):
            if limit_records is not None and row_index > limit_records:
                break

            normalized = normalize_row(row)
            if normalized is None:
                continue

            document, question, ground_truth = normalized
            doc_hash = stable_doc_hash(document)
            shard_index = int(doc_hash[:8], 16) % shard_count
            shard_handles[shard_index].write(
                json.dumps(
                    {
                        "doc_hash": doc_hash,
                        "document": document,
                        "question": question,
                        "ground_truth": ground_truth,
                        "source_row": row_index,
                    },
                    ensure_ascii=False,
                )
            )
            shard_handles[shard_index].write("\n")
            processed_rows += 1

    return shard_paths, processed_rows


def iter_aggregated_shard_records(
    split: str,
    shard_path: Path,
    token_counter: TokenCounter,
) -> Iterator[dict[str, Any]]:
    if not shard_path.exists() or shard_path.stat().st_size == 0:
        return

    documents: dict[str, dict[str, Any]] = {}
    with shard_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_hash = row["doc_hash"]
            doc_id = f"DocFinQA_{split}_{doc_hash}"
            entry = documents.get(doc_hash)
            if entry is None:
                entry = {
                    "dataset_name": "DocFinQA",
                    "split": split,
                    "doc_id": doc_id,
                    "document": row["document"],
                    "doc_length_tokens": token_counter.count_text(row["document"]),
                    "qa_pairs": [],
                }
                documents[doc_hash] = entry

            qa_index = len(entry["qa_pairs"]) + 1
            entry["qa_pairs"].append(
                {
                    "q_id": f"{doc_id}_q{qa_index}",
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "source_row": row["source_row"],
                }
            )

    for item in sorted(documents.values(), key=lambda record: record["doc_id"]):
        yield item

    shard_path.unlink(missing_ok=True)


def process_split(
    split: str,
    input_path: Path,
    token_counter: TokenCounter,
    output_handle,
    shard_count: int,
    temp_root: Path,
    limit_records: int | None = None,
) -> tuple[int, int]:
    split_temp_dir = temp_root / split
    split_temp_dir.mkdir(parents=True, exist_ok=True)

    shard_paths, processed_rows = spool_split_rows(
        split=split,
        input_path=input_path,
        temp_dir=split_temp_dir,
        shard_count=shard_count,
        limit_records=limit_records,
    )

    total_docs = 0
    total_qas = 0
    for shard_path in tqdm(shard_paths, desc=f"DocFinQA {split} shards", unit="shard"):
        for item in iter_aggregated_shard_records(split, shard_path, token_counter):
            output_handle.write(json.dumps(item, ensure_ascii=False))
            output_handle.write("\n")
            total_docs += 1
            total_qas += len(item["qa_pairs"])

    shutil.rmtree(split_temp_dir, ignore_errors=True)
    print(f"{split}: processed {processed_rows} valid QA rows into {total_docs} documents")
    return total_docs, total_qas


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be a positive integer.")

    token_counter = TokenCounter(args.tokenizer_model)
    ensure_parent_dir(args.output_path)

    total_docs = 0
    total_qas = 0

    if args.temp_dir is None:
        temp_root_context = tempfile.TemporaryDirectory(
            prefix="rageval_docfinqa_",
            dir=str(args.output_path.parent),
        )
        temp_root = Path(temp_root_context.name)
    else:
        temp_root_context = None
        temp_root = args.temp_dir
        temp_root.mkdir(parents=True, exist_ok=True)

    try:
        with args.output_path.open("w", encoding="utf-8") as handle:
            for split in args.splits:
                input_path = args.input_dir / f"{split}.json"
                if not input_path.exists():
                    raise FileNotFoundError(f"Missing DocFinQA split file: {input_path}")

                print(f"Processing {split} from {input_path}")
                split_docs, split_qas = process_split(
                    split=split,
                    input_path=input_path,
                    token_counter=token_counter,
                    output_handle=handle,
                    shard_count=args.shard_count,
                    temp_root=temp_root,
                    limit_records=args.limit_records,
                )
                total_docs += split_docs
                total_qas += split_qas
    finally:
        if temp_root_context is not None:
            temp_root_context.cleanup()

    print(
        f"DocFinQA processing complete. Output={args.output_path} "
        f"documents={total_docs} qa_pairs={total_qas}"
    )


if __name__ == "__main__":
    main()
