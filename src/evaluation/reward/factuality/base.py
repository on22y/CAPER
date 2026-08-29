"""Shared types and helpers for the Factuality / Source-Grounded Content Reward.

This module intentionally has no torch/transformers dependency so that pure
deterministic scoring can run without loading any model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# FACTUAL_UNIT vs CONTENT_STATE taxonomy (see project brief section 5 / 6-2).
#
# FACTUAL_UNIT: externally checkable entities/events (who/what/when/where/how many).
# CONTENT_STATE: polarity, stance, modality, degree, subjective attributes -
# not "objective fact" in the News sense, but semantic content that must not
# be silently flipped or dropped by a rewrite.
# ---------------------------------------------------------------------------

FACTUAL_UNIT_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE_TIME",
    "NUMBER_QUANTITY",
    "PRODUCT_OBJECT",
    "ATTRIBUTION",
    "EVENT_ACTION",
}

CONTENT_STATE_TYPES = {
    "POLARITY",
    "MODALITY_STATUS",
    "DEGREE_SCOPE",
    "ATTRIBUTE_VALUE",
    "CLAIM_RELATION",
}

# Map the existing corruption-type vocabulary (from generate_controlled_candidates_v2.py
# / reevaluate_controlled_full_pilot.py) onto the two-way taxonomy above. Used only for
# diagnostic slicing, not for scoring itself.
CORRUPTION_TYPE_TO_DIMENSION = {
    "DATE_TIME": "FACTUAL_UNIT",
    "NUMBER_QUANTITY": "FACTUAL_UNIT",
    "ORGANIZATION": "FACTUAL_UNIT",
    "LOCATION": "FACTUAL_UNIT",
    "PERSON_ROLE": "FACTUAL_UNIT",
    "PRODUCT_OBJECT": "FACTUAL_UNIT",
    "ATTRIBUTION": "FACTUAL_UNIT",
    "EVENT_ACTION": "FACTUAL_UNIT",
    "EVENT_ARGUMENT": "FACTUAL_UNIT",
    "RELATION": "FACTUAL_UNIT",
    "ATTRIBUTE_VALUE": "CONTENT_STATE",
    "MODALITY_STATUS": "CONTENT_STATE",
    "POLARITY": "CONTENT_STATE",
    "DEGREE_SCOPE": "CONTENT_STATE",
    "CLAIM_RELATION": "CONTENT_STATE",
    "CERTAINTY": "CONTENT_STATE",
    "PRODUCT_ATTRIBUTE": "CONTENT_STATE",
}


SCORE_FIELDS = [
    "factuality_score",
    "factual_unit_score",
    "content_state_score",
    "entity_preservation",
    "number_preservation",
    "date_preservation",
    "event_consistency",
    "attribution_consistency",
    "polarity_consistency",
    "modality_consistency",
    "degree_scope_consistency",
    "subjective_attribute_consistency",
    "nli_entailment",
    "nli_contradiction",
    "unsupported_content_score",
    "hard_violation",
]


def empty_score() -> dict[str, Any]:
    out = {k: None for k in SCORE_FIELDS}
    out["details"] = {}
    return out


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_PARTICLE_SUFFIXES = (
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "에게", "께",
    "도", "만", "과", "와", "으로", "로", "이라고", "라고", "이다", "다",
)


def strip_trailing_particle(token: str) -> str:
    """Loosely strip a single trailing Korean particle/copula for fuzzy matching.

    This is NOT a morphological analyzer (none is installed in the `hclt` env).
    It only removes the longest matching suffix so that e.g. '공론화위원회는'
    and '공론화위원회가' both reduce toward '공론화위원회'.
    """
    for suf in sorted(_PARTICLE_SUFFIXES, key=len, reverse=True):
        if len(token) > len(suf) + 1 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    sa, sb = char_ngrams(a, n), char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def loose_contains(haystack: str, needle: str, fuzzy_threshold: float = 0.7) -> bool:
    """Return True if `needle` (a short entity/number span) is present in
    `haystack` either literally or as a close morphological/character variant.
    """
    needle = needle.strip()
    if not needle:
        return False
    if needle in haystack:
        return True
    stripped = strip_trailing_particle(needle)
    if stripped and stripped in haystack:
        return True
    # Fuzzy fallback for short spans (handles spacing/morpheme drift), gated
    # by a minimum length so single-character particles do not fuzzy-match.
    if len(needle) >= 3:
        for window in _sliding_windows(haystack, len(needle)):
            if char_ngram_jaccard(needle, window) >= fuzzy_threshold:
                return True
    return False


def _sliding_windows(text: str, width: int):
    span = max(width - 1, 1)
    for i in range(0, max(len(text) - span + 1, 1)):
        yield text[i : i + span + 2]


@dataclass
class Span:
    type: str
    text: str
    start: int
    end: int
    source: str = "unknown"  # 'ner' | 'regex'
    confidence: float = 1.0


@dataclass
class Metadata:
    domain: str | None = None
    sample_id: str | None = None
    sentence_id: int | None = None
    candidate_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Metadata":
        d = d or {}
        known = {"domain", "sample_id", "sentence_id", "candidate_id"}
        return cls(
            domain=d.get("domain"),
            sample_id=d.get("sample_id"),
            sentence_id=d.get("sentence_id"),
            candidate_id=d.get("candidate_id"),
            extra={k: v for k, v in d.items() if k not in known},
        )


def safe_mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)
