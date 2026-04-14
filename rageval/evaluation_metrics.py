from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
import re
from typing import Any

from ragas.metrics import numeric_metric


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?|[\u4e00-\u9fff]")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_GUARD_RE = re.compile(r"^[\s\-\+\(\)\[\],.%$¥￥€£:/A-Za-z\u4e00-\u9fff]+$")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?")
_PLAIN_NUMBER_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

_NUMERIC_HINTS = (
    "how many",
    "how much",
    "what percentage",
    "what percent",
    "what portion",
    "what ratio",
    "growth rate",
    "change in",
    "total amount",
    "总额",
    "比例",
    "百分比",
    "百分点",
    "增长率",
    "变化率",
    "多少",
    "几成",
    "金额",
)

_NUMERIC_UNITS = (
    "%",
    "percent",
    "percentage",
    "pct",
    "billion",
    "million",
    "thousand",
    "美元",
    "元",
    "万元",
    "亿元",
    "倍",
    "个",
)


def normalize_answer_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = lowered.replace("\u2212", "-").replace("，", ",").replace("％", "%")
    lowered = _WHITESPACE_RE.sub(" ", lowered)
    return lowered


def tokenize_answer(text: str) -> list[str]:
    normalized = normalize_answer_text(text)
    return [_normalize_token(token) for token in _TOKEN_RE.findall(normalized)]


def _normalize_token(token: str) -> str:
    if _PLAIN_NUMBER_TOKEN_RE.fullmatch(token) is None:
        return token
    try:
        numeric = float(token)
    except ValueError:
        return token
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.12f}".rstrip("0").rstrip(".")


def harmonic_mean(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    if first <= 0 or second <= 0:
        return 0.0
    return 2 * first * second / (first + second)


def clamp_unit_interval(value: float | None) -> float | None:
    if value is None:
        return None
    if not isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def weighted_score(parts: list[tuple[str, float | None, float]]) -> float | None:
    usable = [(name, value, weight) for name, value, weight in parts if value is not None]
    if not usable:
        return None
    total_weight = sum(weight for _, _, weight in usable)
    if total_weight <= 0:
        return None
    return sum(value * weight for _, value, weight in usable) / total_weight


def is_error_answer(answer: str) -> bool:
    normalized = normalize_answer_text(answer)
    if not normalized:
        return True
    return normalized == "error" or normalized.startswith("[error]")


def looks_numeric_reference(reference: str) -> bool:
    normalized = normalize_answer_text(reference)
    if not normalized or len(normalized) > 64:
        return False
    if _NUMERIC_GUARD_RE.fullmatch(normalized) is None:
        return False
    if _NUMBER_RE.search(normalized) is None:
        return False
    stripped = normalized
    for unit in _NUMERIC_UNITS:
        stripped = stripped.replace(unit, " ")
    stripped = _NUMBER_RE.sub(" ", stripped)
    stripped = _WHITESPACE_RE.sub(" ", stripped).strip(" .,:;/")
    return not stripped


def is_numeric_question(question: str, reference: str) -> bool:
    if looks_numeric_reference(reference):
        return True
    normalized_question = normalize_answer_text(question)
    normalized_reference = normalize_answer_text(reference)
    return any(hint in normalized_question for hint in _NUMERIC_HINTS) and _NUMBER_RE.search(
        normalized_reference
    ) is not None


def _extract_numeric_value(text: str) -> float | None:
    normalized = normalize_answer_text(text)
    if not normalized:
        return None

    sign = -1.0 if normalized.startswith("(") and normalized.endswith(")") else 1.0
    match = _NUMBER_RE.search(normalized)
    if match is None:
        return None

    raw_number = match.group(0).replace(",", "")
    try:
        return sign * float(raw_number)
    except ValueError:
        return None


@numeric_metric(name="numeric_tolerance_score", allowed_values=(0.0, 1.0))
def numeric_tolerance_score(response: str, reference: str) -> float:
    reference_value = _extract_numeric_value(reference)
    response_value = _extract_numeric_value(response)

    if reference_value is None or response_value is None:
        return 1.0 if normalize_answer_text(response) == normalize_answer_text(reference) else 0.0

    if reference_value != 0 and response_value != 0 and reference_value * response_value < 0:
        return 0.0

    abs_diff = abs(response_value - reference_value)
    if abs_diff == 0:
        return 1.0

    denominator = max(abs(reference_value), 1e-12)
    relative_error = abs_diff / denominator
    if relative_error <= 0.01:
        return 1.0
    if relative_error <= 0.05:
        return 0.5
    return 0.0


@numeric_metric(name="token_f1_score", allowed_values=(0.0, 1.0))
def token_f1_score(response: str, reference: str) -> float:
    response_tokens = tokenize_answer(response)
    reference_tokens = tokenize_answer(reference)

    if not response_tokens and not reference_tokens:
        return 1.0
    if not response_tokens or not reference_tokens:
        return 0.0

    overlap = Counter(response_tokens) & Counter(reference_tokens)
    overlap_count = sum(overlap.values())
    if overlap_count == 0:
        return 0.0

    precision = overlap_count / len(response_tokens)
    recall = overlap_count / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_reference_alignment(
    *,
    is_numeric: bool,
    answer_accuracy: float | None,
    semantic_similarity: float | None,
    exact_match: float | None,
    token_f1: float | None,
    numeric_tolerance: float | None,
) -> float | None:
    if is_numeric:
        return weighted_score(
            [
                ("numeric_tolerance", numeric_tolerance, 0.55),
                ("answer_accuracy", answer_accuracy, 0.25),
                ("token_f1", token_f1, 0.10),
                ("exact_match", exact_match, 0.10),
            ]
        )
    return weighted_score(
        [
            ("answer_accuracy", answer_accuracy, 0.45),
            ("semantic_similarity", semantic_similarity, 0.25),
            ("token_f1", token_f1, 0.20),
            ("exact_match", exact_match, 0.10),
        ]
    )


def compute_answer_quality(
    *,
    reference_alignment: float | None,
    faithfulness: float | None,
    response_relevancy: float | None,
) -> float | None:
    return weighted_score(
        [
            ("reference_alignment", reference_alignment, 0.50),
            ("faithfulness", faithfulness, 0.30),
            ("response_relevancy", clamp_unit_interval(response_relevancy), 0.20),
        ]
    )


def compute_retrieval_quality(
    *,
    context_precision: float | None,
    context_recall: float | None,
    context_entity_recall: float | None,
) -> float | None:
    precision_recall_f1 = harmonic_mean(context_precision, context_recall)
    return weighted_score(
        [
            ("context_pr_f1", precision_recall_f1, 0.70),
            ("context_entity_recall", context_entity_recall, 0.30),
        ]
    )


def compute_overall_quality(
    *,
    answer_quality: float | None,
    retrieval_quality: float | None,
) -> float | None:
    return weighted_score(
        [
            ("answer_quality", answer_quality, 0.80),
            ("retrieval_quality", retrieval_quality, 0.20),
        ]
    )


@dataclass(slots=True)
class MetricFormulaDescription:
    answer_quality: str
    retrieval_quality: str
    overall_quality: str


def default_formula_description() -> MetricFormulaDescription:
    return MetricFormulaDescription(
        answer_quality=(
            "AnswerQuality = 0.50 * ReferenceAlignment + 0.30 * Faithfulness + "
            "0.20 * clamp(ResponseRelevancy)"
        ),
        retrieval_quality=(
            "RetrievalQuality = 0.70 * harmonic_mean(ContextPrecision, ContextRecall) + "
            "0.30 * ContextEntityRecall"
        ),
        overall_quality=(
            "OverallQuality = 0.80 * AnswerQuality + 0.20 * RetrievalQuality "
            "(weights are renormalized over available metrics)"
        ),
    )


def describe_custom_metric_rules() -> dict[str, Any]:
    return {
        "numeric_tolerance_score": {
            "exact_match": "1.0 when parsed numeric values match exactly",
            "tolerance_1_percent": "1.0 when relative error <= 1%",
            "tolerance_5_percent": "0.5 when relative error <= 5%",
            "sign_mismatch": "0.0 when non-zero values have opposite signs",
        },
        "token_f1_score": {
            "tokenization": "English alphanumeric tokens plus per-character CJK tokens",
            "formula": "standard token-level F1 on normalized answers",
        },
    }
