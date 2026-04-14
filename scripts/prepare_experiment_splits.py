from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT_GLOB = "*_unified.jsonl"
DEFAULT_OUTPUT_DIR = Path("./dataset/processed/experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create main_experiment.jsonl and stress_test.jsonl from processed unified datasets."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./dataset/processed"),
        help="Directory containing the processed *_unified.jsonl files.",
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob for unified input files inside dataset-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for main_experiment.jsonl and stress_test.jsonl.",
    )
    parser.add_argument(
        "--stress-top-k",
        type=int,
        default=15,
        help="Number of globally longest documents moved into stress_test.jsonl.",
    )
    parser.add_argument(
        "--sample-per-bin",
        type=int,
        default=10,
        help="Number of documents sampled per dataset x length bin.",
    )
    parser.add_argument(
        "--max-qa-per-doc",
        type=int,
        default=5,
        help="Maximum number of QA pairs retained for each sampled main-experiment document.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for document sampling and QA sub-sampling.",
    )
    return parser.parse_args()


def discover_input_files(dataset_dir: Path, input_glob: str) -> list[Path]:
    files = sorted(
        path
        for path in dataset_dir.rglob(input_glob)
        if path.is_file() and path.parent.name != "experiment"
    )
    if not files:
        raise FileNotFoundError(
            f"No unified JSONL files found under {dataset_dir} with glob {input_glob!r}."
        )
    return files


def load_records(files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"{path}:{line_no} is not a JSON object.")
                if "qa_pairs" not in record or "document" not in record:
                    raise ValueError(f"{path}:{line_no} is not a unified document record.")
                record["_source_file"] = str(path)
                record["_source_line"] = line_no
                record["dataset_name"] = record.get("dataset_name") or path.parent.name
                record["doc_length_tokens"] = int(record.get("doc_length_tokens") or 0)
                records.append(record)
    return records


def split_sorted_docs_into_bins(
    docs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    sorted_docs = sorted(
        docs,
        key=lambda item: (int(item["doc_length_tokens"]), str(item.get("doc_id", ""))),
    )
    n_docs = len(sorted_docs)
    base_size = n_docs // 3
    remainder = n_docs % 3
    bucket_sizes = [base_size + (1 if i < remainder else 0) for i in range(3)]

    bins: dict[str, list[dict[str, Any]]] = {}
    start = 0
    for label, size in zip(("short", "medium", "long"), bucket_sizes, strict=True):
        end = start + size
        bins[label] = sorted_docs[start:end]
        start = end
    return bins


def sample_qa_pairs(
    qa_pairs: list[dict[str, Any]],
    max_qa_per_doc: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(qa_pairs) <= max_qa_per_doc:
        return qa_pairs

    indexed_pairs = list(enumerate(qa_pairs))
    selected = rng.sample(indexed_pairs, max_qa_per_doc)
    selected.sort(key=lambda item: item[0])
    return [qa for _, qa in selected]


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    cleaned.pop("_source_file", None)
    cleaned.pop("_source_line", None)
    return cleaned


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def quantile_summary(docs: list[dict[str, Any]]) -> dict[str, int | None]:
    if not docs:
        return {"min": None, "q33_proxy": None, "q66_proxy": None, "max": None}

    sorted_lengths = sorted(int(doc["doc_length_tokens"]) for doc in docs)
    n_docs = len(sorted_lengths)
    first_cut = max(0, (n_docs + 2) // 3 - 1)
    second_cut = max(0, (2 * n_docs + 2) // 3 - 1)
    return {
        "min": sorted_lengths[0],
        "q33_proxy": sorted_lengths[first_cut],
        "q66_proxy": sorted_lengths[second_cut],
        "max": sorted_lengths[-1],
    }


def build_manifest(
    input_files: list[Path],
    all_records: list[dict[str, Any]],
    stress_records: list[dict[str, Any]],
    main_records: list[dict[str, Any]],
    remaining_by_dataset: dict[str, list[dict[str, Any]]],
    sampled_by_dataset_bin: dict[str, dict[str, list[dict[str, Any]]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "stress_top_k": args.stress_top_k,
        "sample_per_bin": args.sample_per_bin,
        "max_qa_per_doc": args.max_qa_per_doc,
        "input_files": [str(path) for path in input_files],
        "total_input_documents": len(all_records),
        "stress_test_documents": len(stress_records),
        "stress_test_total_qa_pairs": sum(len(record["qa_pairs"]) for record in stress_records),
        "main_experiment_documents": len(main_records),
        "main_experiment_total_qa_pairs": sum(len(record["qa_pairs"]) for record in main_records),
        "datasets": {},
    }

    for dataset_name, docs in sorted(remaining_by_dataset.items()):
        bins = sampled_by_dataset_bin[dataset_name]
        manifest["datasets"][dataset_name] = {
            "remaining_documents_after_stress_filter": len(docs),
            "length_summary": quantile_summary(docs),
            "sampled_documents": {
                label: len(bin_docs) for label, bin_docs in bins.items()
            },
            "sampled_qa_pairs": {
                label: sum(len(doc["qa_pairs"]) for doc in bin_docs)
                for label, bin_docs in bins.items()
            },
        }

    manifest["stress_test_doc_ids"] = [
        {
            "doc_id": record.get("doc_id"),
            "dataset_name": record.get("dataset_name"),
            "doc_length_tokens": record.get("doc_length_tokens"),
            "qa_pair_count": len(record.get("qa_pairs", [])),
        }
        for record in stress_records
    ]
    return manifest


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_files = discover_input_files(args.dataset_dir, args.input_glob)
    all_records = load_records(input_files)
    if len(all_records) < args.stress_top_k:
        raise ValueError(
            f"stress_top_k={args.stress_top_k} is larger than total documents={len(all_records)}."
        )

    all_records_sorted = sorted(
        all_records,
        key=lambda item: (
            -int(item["doc_length_tokens"]),
            str(item.get("dataset_name", "")),
            str(item.get("doc_id", "")),
        ),
    )
    stress_records_raw = all_records_sorted[: args.stress_top_k]
    stress_doc_ids = {str(record.get("doc_id")) for record in stress_records_raw}
    remaining_records = [
        record for record in all_records if str(record.get("doc_id")) not in stress_doc_ids
    ]

    remaining_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in remaining_records:
        remaining_by_dataset.setdefault(str(record["dataset_name"]), []).append(record)

    sampled_by_dataset_bin: dict[str, dict[str, list[dict[str, Any]]]] = {}
    main_records: list[dict[str, Any]] = []

    for dataset_name, docs in sorted(remaining_by_dataset.items()):
        bins = split_sorted_docs_into_bins(docs)
        sampled_by_dataset_bin[dataset_name] = {}
        for label in ("short", "medium", "long"):
            bucket_docs = bins[label]
            sample_size = min(args.sample_per_bin, len(bucket_docs))
            sampled_docs = rng.sample(bucket_docs, sample_size) if sample_size else []
            sampled_records_for_bin: list[dict[str, Any]] = []
            for record in sampled_docs:
                sampled_record = sanitize_record(record)
                sampled_record["qa_pairs"] = sample_qa_pairs(
                    list(record["qa_pairs"]),
                    args.max_qa_per_doc,
                    rng,
                )
                sampled_records_for_bin.append(sampled_record)
            sampled_by_dataset_bin[dataset_name][label] = sampled_records_for_bin
            main_records.extend(sampled_records_for_bin)

    rng.shuffle(main_records)

    stress_records = [sanitize_record(record) for record in stress_records_raw]

    output_dir = args.output_dir
    main_output = output_dir / "main_experiment.jsonl"
    stress_output = output_dir / "stress_test.jsonl"
    manifest_output = output_dir / "split_manifest.json"

    write_jsonl(main_output, main_records)
    write_jsonl(stress_output, stress_records)
    manifest_output.write_text(
        json.dumps(
            build_manifest(
                input_files=input_files,
                all_records=all_records,
                stress_records=stress_records,
                main_records=main_records,
                remaining_by_dataset=remaining_by_dataset,
                sampled_by_dataset_bin=sampled_by_dataset_bin,
                args=args,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Input files: {len(input_files)}")
    print(f"Total input documents: {len(all_records)}")
    print(
        f"Stress test: {len(stress_records)} docs, "
        f"{sum(len(record['qa_pairs']) for record in stress_records)} QA pairs -> {stress_output}"
    )
    print(
        f"Main experiment: {len(main_records)} docs, "
        f"{sum(len(record['qa_pairs']) for record in main_records)} QA pairs -> {main_output}"
    )
    print(f"Manifest -> {manifest_output}")


if __name__ == "__main__":
    main()
