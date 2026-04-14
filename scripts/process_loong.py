from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir, normalize_text


DEFAULT_DOCS_ZIP_URL = "http://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Loong into a document-level JSONL format with qa_pairs. "
            "This implementation avoids keeping all long documents in memory at once."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("./dataset/Loong/data/loong.jsonl"),
        help="Path to the original loong.jsonl file.",
    )
    parser.add_argument(
        "--doc-root",
        type=Path,
        default=Path("./dataset/Loong/data/doc"),
        help="Root directory containing Loong full documents (paper/legal/financial).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./dataset/processed/loong/loong_unified.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer model used to compute doc_length_tokens.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debug limit on the number of Loong rows processed.",
    )
    parser.add_argument(
        "--download-docs-if-missing",
        action="store_true",
        help="Download and extract official Loong documents if doc-root is missing.",
    )
    parser.add_argument(
        "--docs-zip-url",
        default=DEFAULT_DOCS_ZIP_URL,
        help="Official Loong docs zip URL.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload the docs zip even if it already exists locally.",
    )
    return parser.parse_args()


def stable_bundle_hash(parts: list[str]) -> str:
    payload = "\n".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def download_with_progress(url: str, output_path: Path) -> None:
    ensure_parent_dir(output_path)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0"))
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading Loong docs") as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))


def ensure_loong_docs(
    doc_root: Path,
    docs_zip_url: str,
    force_redownload: bool = False,
) -> None:
    if (doc_root / "paper").exists() and (doc_root / "financial").exists() and (doc_root / "legal").exists():
        return

    parent_dir = doc_root.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    zip_path = parent_dir / "doc.zip"

    if force_redownload and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        download_with_progress(docs_zip_url, zip_path)

    if doc_root.exists():
        shutil.rmtree(doc_root)

    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.startswith("doc/") and not name.startswith("__MACOSX/")]
        zf.extractall(parent_dir, members=members)


def recover_zip_filename(name: str) -> str:
    try:
        return name.encode("cp437").decode("utf-8")
    except Exception:
        return name


def extract_financial_aliases(path: Path) -> set[str]:
    aliases = {path.stem, path.name}

    recovered_name = recover_zip_filename(path.name)
    recovered_stem = Path(recovered_name).stem
    aliases.add(recovered_name)
    aliases.add(recovered_stem)

    try:
        preview = path.read_text(encoding="utf-8", errors="ignore")[:400]
    except Exception:
        preview = ""

    for line in preview.splitlines()[:3]:
        line = line.strip()
        if not line:
            continue
        aliases.add(line)
        aliases.update(re.findall(r"[\u4e00-\u9fffA-Za-z0-9 .,&'()/-]{2,}", line))

    return {alias for alias in aliases if alias}


def resolve_financial_file(financial_root: Path, doc_name: str, level: int) -> Path:
    patterns = [f"*{doc_name}*.txt"]
    if level != 4:
        patterns.insert(0, f"*2024-{doc_name}*.txt")

    for pattern in patterns:
        matches = sorted(path for path in financial_root.glob(pattern) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            exact_name_matches = [path for path in matches if doc_name in path.stem]
            if exact_name_matches:
                return exact_name_matches[0]
            return matches[0]

    for path in sorted(financial_root.glob("*.txt")):
        aliases = extract_financial_aliases(path)
        if any(doc_name in alias for alias in aliases):
            return path

    raise FileNotFoundError(f"Unable to resolve financial document for {doc_name}")


def read_loong_document(
    doc_root: Path,
    sample: dict[str, Any],
    doc_name: str,
    doc_index: int,
    legal_payload: dict[str, Any] | None = None,
) -> str:
    doc_type = sample["type"]
    level = int(sample["level"])

    if doc_type == "paper":
        path = doc_root / "paper" / doc_name
        content = path.read_text(encoding="utf-8").strip()
        first_line = content.splitlines()[0] if content else doc_name
        title = first_line.lstrip("#").strip() or Path(doc_name).stem
        return f"《{title}》\n{content}\n"

    if doc_type == "financial":
        path = resolve_financial_file(doc_root / "financial", doc_name, level)
        content = path.read_text(encoding="utf-8").strip()
        return f"《{doc_name}》\n{content}\n"

    if doc_type == "legal":
        if legal_payload is None:
            raise ValueError("legal_payload is required for legal Loong documents")
        record = legal_payload[doc_name]
        instruction = normalize_text(sample.get("instruction"))
        if level == 4 and "阅读以上判决文书，我将给你若干份判决结果" in instruction:
            content = normalize_text(record.get("content"))
        else:
            content = normalize_text(record.get("content")) + "\n" + normalize_text(record.get("result"))
        return f"《{doc_name or f'判决文书{doc_index + 1}'}》\n{content.strip()}\n"

    raise ValueError(f"Unsupported Loong document type: {doc_type}")


def normalize_answer(value: Any) -> str:
    if isinstance(value, str):
        return normalize_text(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def iter_loong_rows(
    input_path: Path,
    limit_records: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    with input_path.open("r", encoding="utf-8") as handle:
        yielded = 0
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yielded += 1
            if limit_records is not None and yielded > limit_records:
                break
            yield row_index, json.loads(line)


def build_bundle_entries(
    input_path: Path,
    limit_records: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row_index, sample in tqdm(
        iter_loong_rows(input_path, limit_records=limit_records),
        desc="Collecting Loong bundles",
        unit="row",
    ):
        bundle_key = (
            sample["type"],
            sample["language"],
            int(sample["set"]),
            int(sample["level"]),
            tuple(sample["doc"]),
        )
        entry = grouped.get(bundle_key)
        if entry is None:
            doc_hash = stable_bundle_hash(
                [
                    sample["type"],
                    sample["language"],
                    str(sample["set"]),
                    str(sample["level"]),
                    *sample["doc"],
                ]
            )
            entry = {
                "dataset_name": "Loong",
                "split": f"set{sample['set']}",
                "doc_id": f"Loong_{sample['type']}_set{sample['set']}_level{sample['level']}_{doc_hash}",
                "domain": sample["type"],
                "language": sample["language"],
                "level": int(sample["level"]),
                "set": int(sample["set"]),
                "source_doc_names": sample["doc"],
                "sample_for_document": {
                    "type": sample["type"],
                    "language": sample["language"],
                    "set": sample["set"],
                    "level": sample["level"],
                    "doc": sample["doc"],
                    "instruction": sample.get("instruction"),
                },
                "qa_pairs": [],
            }
            grouped[bundle_key] = entry

        entry["qa_pairs"].append(
            {
                "q_id": normalize_text(sample.get("id")) or f"{entry['doc_id']}_q{len(entry['qa_pairs']) + 1}",
                "instruction": normalize_text(sample.get("instruction")) or "",
                "question": normalize_text(sample.get("question")) or "",
                "ground_truth": normalize_answer(sample.get("answer")),
                "source_record_index": row_index,
            }
        )

    return sorted(grouped.values(), key=lambda item: item["doc_id"])


def write_unified_records(
    entries: list[dict[str, Any]],
    output_path: Path,
    doc_root: Path,
    token_counter: TokenCounter,
) -> tuple[int, int]:
    legal_path = doc_root / "legal" / "legal.json"
    legal_payload = None
    if any(entry["domain"] == "legal" for entry in entries):
        legal_payload = json.loads(legal_path.read_text(encoding="utf-8"))

    total_qas = 0
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in tqdm(entries, desc="Writing Loong documents", unit="doc"):
            sample = entry["sample_for_document"]
            docs_text = []
            for doc_index, doc_name in enumerate(sample["doc"]):
                docs_text.append(
                    read_loong_document(
                        doc_root=doc_root,
                        sample=sample,
                        doc_name=doc_name,
                        doc_index=doc_index,
                        legal_payload=legal_payload,
                    )
                )
            combined_document = "\n\n".join(docs_text).strip()

            record = {
                "dataset_name": entry["dataset_name"],
                "split": entry["split"],
                "doc_id": entry["doc_id"],
                "document": combined_document,
                "doc_length_tokens": token_counter.count_text(combined_document),
                "domain": entry["domain"],
                "language": entry["language"],
                "level": entry["level"],
                "set": entry["set"],
                "source_doc_names": entry["source_doc_names"],
                "qa_pairs": entry["qa_pairs"],
            }
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            total_qas += len(record["qa_pairs"])

    return len(entries), total_qas


def main() -> None:
    args = parse_args()

    if args.download_docs_if_missing:
        ensure_loong_docs(
            doc_root=args.doc_root,
            docs_zip_url=args.docs_zip_url,
            force_redownload=args.force_redownload,
        )
    elif not args.doc_root.exists():
        raise FileNotFoundError(
            f"Loong doc root not found: {args.doc_root}. "
            "Run with --download-docs-if-missing or place the official docs under this path."
        )

    token_counter = TokenCounter(args.tokenizer_model)
    entries = build_bundle_entries(
        input_path=args.input_path,
        limit_records=args.limit_records,
    )
    total_docs, total_qas = write_unified_records(
        entries=entries,
        output_path=args.output_path,
        doc_root=args.doc_root,
        token_counter=token_counter,
    )

    print(
        f"Loong processing complete. Output={args.output_path} "
        f"documents={total_docs} qa_pairs={total_qas}"
    )


if __name__ == "__main__":
    main()
