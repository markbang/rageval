from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
import json
import logging

from rageval.models import ExperimentDocument, QAPair
from rageval.utils import normalize_text


logger = logging.getLogger(__name__)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "doc_id": ("doc_id", "id", "qid", "question_id"),
    "question": ("question", "Question", "query", "prompt", "instruction"),
    "document": ("document", "Document", "context", "Context", "passage", "content"),
    "ground_truth": ("ground_truth", "GroundTruth", "answer", "Answer", "label"),
    "q_id": ("q_id", "id", "question_id", "qid"),
    "instruction": ("instruction", "Instruction"),
}


class DatasetLoader:
    def __init__(
        self,
        dataset_dir: Path,
        dataset_globs: tuple[str, ...] = ("**/*.jsonl",),
        include_json: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_globs = dataset_globs
        self.include_json = include_json

    def iter_documents(self) -> Iterator[ExperimentDocument]:
        for file_path in self.iter_dataset_files():
            yield from self._load_file(file_path)

    def iter_dataset_files(self) -> list[Path]:
        discovered: set[Path] = set()
        for pattern in self.dataset_globs:
            discovered.update(self.dataset_dir.glob(pattern))
        if self.include_json:
            discovered.update(self.dataset_dir.glob("**/*.json"))
        return sorted(path for path in discovered if path.is_file())

    def _load_file(self, file_path: Path) -> Iterator[ExperimentDocument]:
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for index, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        yield self._coerce_record(record, file_path, index)
                    except Exception as exc:
                        logger.warning(
                            "Skipping invalid JSONL record in %s line %s: %s",
                            file_path,
                            index,
                            exc,
                        )
            return

        if file_path.suffix == ".json" and self.include_json:
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skipping unreadable JSON file %s: %s", file_path, exc)
                return

            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                logger.warning("Skipping unsupported JSON file %s: top level is not a list", file_path)
                return

            for index, record in enumerate(payload, start=1):
                try:
                    yield self._coerce_record(record, file_path, index)
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid JSON record in %s item %s: %s",
                        file_path,
                        index,
                        exc,
                    )

    def _coerce_record(
        self,
        record: dict[str, Any],
        source_file: Path,
        record_index: int,
    ) -> ExperimentDocument:
        if not isinstance(record, dict):
            raise TypeError("record is not a JSON object")

        if "qa_pairs" in record:
            return self._coerce_document_record(record, source_file, record_index)

        return self._coerce_single_qa_record(record, source_file, record_index)

    def _coerce_document_record(
        self,
        record: dict[str, Any],
        source_file: Path,
        record_index: int,
    ) -> ExperimentDocument:
        dataset_name = normalize_text(record.get("dataset_name")) or source_file.parent.name
        doc_id = str(record.get("doc_id") or f"{dataset_name}-{record_index}")
        document = normalize_text(record.get("document"))
        split = normalize_text(record.get("split") or record.get("dataset_split")) or None
        doc_length_tokens = record.get("doc_length_tokens")
        domain = normalize_text(record.get("domain") or record.get("type")) or None
        language = normalize_text(record.get("language")) or None
        level = record.get("level")
        set_id = record.get("set") or record.get("set_id")

        if not document:
            raise ValueError("missing document field")

        qa_pairs_raw = record.get("qa_pairs")
        if not isinstance(qa_pairs_raw, list) or not qa_pairs_raw:
            raise ValueError("missing qa_pairs field")

        qa_pairs: list[QAPair] = []
        for qa_index, qa_record in enumerate(qa_pairs_raw, start=1):
            if not isinstance(qa_record, dict):
                raise TypeError("qa_pairs entry is not a JSON object")
            q_id = normalize_text(self._extract_first(qa_record, "q_id")) or f"q{qa_index}"
            question = normalize_text(self._extract_first(qa_record, "question"))
            ground_truth = normalize_text(self._extract_first(qa_record, "ground_truth"))
            instruction = normalize_text(self._extract_first(qa_record, "instruction")) or None
            if not question or not ground_truth:
                if not instruction or not ground_truth:
                    raise ValueError("qa_pairs entry missing query text or ground_truth")
            qa_pairs.append(
                QAPair(
                    q_id=q_id,
                    question=question,
                    ground_truth=ground_truth,
                    instruction=instruction,
                )
            )

        parsed_doc_length_tokens = int(doc_length_tokens) if doc_length_tokens is not None else None
        parsed_level = int(level) if level is not None else None
        parsed_set_id = int(set_id) if set_id is not None else None

        return ExperimentDocument(
            dataset_name=dataset_name,
            doc_id=doc_id,
            document=document,
            qa_pairs=qa_pairs,
            doc_length_tokens=parsed_doc_length_tokens,
            source_file=source_file,
            record_index=record_index,
            split=split,
            domain=domain,
            language=language,
            level=parsed_level,
            set_id=parsed_set_id,
        )

    def _coerce_single_qa_record(
        self,
        record: dict[str, Any],
        source_file: Path,
        record_index: int,
    ) -> ExperimentDocument:
        doc_id = str(self._extract_first(record, "doc_id") or f"{source_file.stem}-{record_index}")
        question = normalize_text(self._extract_first(record, "question"))
        document = normalize_text(self._extract_first(record, "document"))
        ground_truth = normalize_text(self._extract_first(record, "ground_truth"))
        instruction = normalize_text(self._extract_first(record, "instruction")) or None

        if not question and not instruction:
            raise ValueError("missing question field")
        if not document:
            raise ValueError("missing document field")
        if not ground_truth:
            raise ValueError("missing ground_truth field")

        return ExperimentDocument(
            dataset_name=normalize_text(record.get("dataset_name")) or source_file.parent.name,
            doc_id=doc_id,
            document=document,
            qa_pairs=[
                QAPair(
                    q_id="q1",
                    question=question,
                    ground_truth=ground_truth,
                    instruction=instruction,
                )
            ],
            doc_length_tokens=record.get("doc_length_tokens"),
            source_file=source_file,
            record_index=record_index,
            split=normalize_text(record.get("split") or record.get("dataset_split")) or None,
            domain=normalize_text(record.get("domain") or record.get("type")) or None,
            language=normalize_text(record.get("language")) or None,
            level=int(record["level"]) if record.get("level") is not None else None,
            set_id=int(record["set"]) if record.get("set") is not None else None,
        )

    def _extract_first(self, record: dict[str, Any], field_name: str) -> Any:
        for alias in FIELD_ALIASES[field_name]:
            if alias in record:
                return record[alias]
        return None
