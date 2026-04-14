from __future__ import annotations

import argparse
import json
import re
import time
import subprocess
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from rageval.utils import ensure_parent_dir


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
LEADING_BOILERPLATE_PATTERNS = (
    r"^published as ",
    r"^under review as ",
    r"^conference paper",
    r"^preprint",
    r"^anonymous authors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supplement PeerQA papers.jsonl with missing OpenReview papers by using "
            "downloaded PDFs and pdftotext. This avoids heavy GROBID dependencies."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./dataset/PeerQA/data"),
        help="PeerQA data directory containing qa.jsonl, papers.jsonl, and openreview PDFs.",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=None,
        help="Override path to qa.jsonl.",
    )
    parser.add_argument(
        "--papers-path",
        type=Path,
        default=None,
        help="Override path to papers.jsonl.",
    )
    parser.add_argument(
        "--errors-path",
        type=Path,
        default=Path("./dataset/PeerQA/data/openreview_extraction_errors.jsonl"),
        help="Path to append extraction errors.",
    )
    parser.add_argument(
        "--limit-papers",
        type=int,
        default=None,
        help="Optional debug limit on the number of missing papers to process.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=None,
        help="Specific PeerQA paper_id(s) to process, for example openreview/ICLR-2022-conf/ABC123.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-extract OpenReview papers even if they already exist in papers.jsonl.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for each OpenReview PDF download request.",
    )
    parser.add_argument(
        "--seconds-between-requests",
        type=float,
        default=0.75,
        help="Sleep between PDF downloads to stay under OpenReview rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum download retries for each paper.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=15.0,
        help="Base backoff seconds for 429/timeout retry handling.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_existing_paper_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    paper_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_ids.add(row["paper_id"])
    return paper_ids


def load_openreview_targets(
    qa_path: Path,
    existing_paper_ids: set[str],
    override: bool = False,
) -> list[str]:
    paper_ids: set[str] = set()
    with qa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            paper_id = row["paper_id"]
            if not paper_id.startswith("openreview/"):
                continue
            if not override and paper_id in existing_paper_ids:
                continue
            paper_ids.add(paper_id)
    return sorted(paper_ids)


def paper_id_to_pdf_path(data_dir: Path, paper_id: str) -> Path:
    _, conference, forum_id = paper_id.split("/")
    return data_dir / "openreview" / conference / forum_id / "paper.pdf"


def openreview_pdf_url(paper_id: str) -> str:
    forum_id = paper_id.split("/")[-1]
    return f"https://openreview.net/pdf?id={forum_id}"


def is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except Exception:
        return False


def download_pdf(
    paper_id: str,
    pdf_path: Path,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> None:
    ensure_parent_dir(pdf_path)
    url = openreview_pdf_url(paper_id)
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": f"https://openreview.net/forum?id={paper_id.split('/')[-1]}",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
            if not is_valid_pdf(pdf_path):
                content_type = response.headers.get("Content-Type", "")
                raise ValueError(
                    f"Downloaded content is not a valid PDF. content_type={content_type!r}"
                )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_seconds = retry_backoff_seconds * attempt
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error


def extract_pdf_text(pdf_path: Path) -> str:
    completed = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return normalize_text(completed.stdout)


def paragraphize(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                chunks.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        chunks.append(" ".join(current))

    cleaned: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if len(chunk) < 2:
            continue
        cleaned.append(chunk)
    return cleaned


def build_paper_rows(paper_id: str, text: str) -> list[dict[str, Any]]:
    paragraphs = paragraphize(text)
    if not paragraphs:
        raise ValueError("No extractable text paragraphs found")

    while paragraphs:
        first = paragraphs[0].strip().lower()
        if any(re.match(pattern, first) for pattern in LEADING_BOILERPLATE_PATTERNS):
            paragraphs.pop(0)
            continue
        break

    if not paragraphs:
        raise ValueError("Only boilerplate text found after extraction")

    rows: list[dict[str, Any]] = []
    title = paragraphs[0]
    rows.append(
        {
            "idx": 0,
            "pidx": 0,
            "sidx": 0,
            "type": "title",
            "content": title,
            "last_heading": None,
            "paper_id": paper_id,
        }
    )

    for paragraph_index, paragraph in enumerate(paragraphs[1:], start=1):
        rows.append(
            {
                "idx": len(rows),
                "pidx": paragraph_index,
                "sidx": 0,
                "type": "paragraph",
                "content": paragraph,
                "last_heading": None,
                "paper_id": paper_id,
            }
        )

    return rows


def append_error(errors_path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(errors_path)
    with errors_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def main() -> None:
    args = parse_args()
    qa_path = args.qa_path or (args.data_dir / "qa.jsonl")
    papers_path = args.papers_path or (args.data_dir / "papers.jsonl")

    existing_paper_ids = load_existing_paper_ids(papers_path)
    if args.paper_id:
        targets = sorted(set(args.paper_id))
    else:
        targets = load_openreview_targets(
            qa_path=qa_path,
            existing_paper_ids=existing_paper_ids,
            override=args.override,
        )
    if args.limit_papers is not None:
        targets = targets[: args.limit_papers]

    if not targets:
        print("No missing OpenReview papers to supplement.")
        return

    ensure_parent_dir(papers_path)
    processed = 0
    with papers_path.open("a", encoding="utf-8") as handle:
        for paper_id in tqdm(targets, desc="Supplementing PeerQA OpenReview", unit="paper"):
            pdf_path = paper_id_to_pdf_path(args.data_dir, paper_id)

            try:
                if args.override or not is_valid_pdf(pdf_path):
                    download_pdf(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        timeout_seconds=args.request_timeout_seconds,
                        max_retries=args.max_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                    )
                    if args.seconds_between_requests > 0:
                        time.sleep(args.seconds_between_requests)
                text = extract_pdf_text(pdf_path)
                rows = build_paper_rows(paper_id, text)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")
                processed += 1
            except Exception as exc:
                append_error(
                    args.errors_path,
                    {"paper_id": paper_id, "pdf_path": str(pdf_path), "error": str(exc)},
                )

    print(
        f"PeerQA OpenReview supplementation complete. papers_added={processed} "
        f"targeted={len(targets)} output={papers_path}"
    )


if __name__ == "__main__":
    main()
