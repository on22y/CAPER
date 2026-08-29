"""Baseline A: deterministic factual-unit preservation checks.

Uses a general-purpose Korean NER model (KLUE-NER fine-tune, already cached
locally as `chunwoolee0/klue_ner_roberta_model`) plus a small regex safety net
for numbers/percentages that NER sometimes misses. This intentionally does
NOT reuse the regex slot detector in `generate_controlled_candidates_v2.py`:
that detector is a synthetic-corruption-*target* picker hand-tuned to the
News/Petition/SNS pilot vocabulary (e.g. hardcoded '립글로우', '푸드뱅크'), not a
general factuality evaluator (see project brief section 4).

Scope: explicit factual units only (PERSON, ORGANIZATION, LOCATION,
DATE_TIME, NUMBER_QUANTITY). Event/attribution/modality/polarity semantics
are deliberately left to the NLI/semantic baseline (nli.py) because they
cannot be reliably captured by span matching.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from .base import Span, char_ngram_jaccard, loose_contains, normalize_text, safe_mean

NER_MODEL_NAME = "chunwoolee0/klue_ner_roberta_model"

_KLUE_TO_FACTUAL_UNIT = {
    "PS": "PERSON",
    "OG": "ORGANIZATION",
    "LC": "LOCATION",
    "DT": "DATE_TIME",
    "TI": "DATE_TIME",
    "QT": "NUMBER_QUANTITY",
}

_NUMBER_RE = re.compile(
    r"\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:%|퍼센트|원|만원|억원|천원|개|명|분|마리|회|점|건|층|대|kg|g|km|m|시간|일|개월|년|살|위)?"
)

_lock = threading.Lock()
_ner_pipeline = None
_ner_load_failed = False


def _get_ner_pipeline():
    global _ner_pipeline, _ner_load_failed
    if _ner_pipeline is not None or _ner_load_failed:
        return _ner_pipeline
    with _lock:
        if _ner_pipeline is not None or _ner_load_failed:
            return _ner_pipeline
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            device = 0 if torch.cuda.is_available() else -1
            tok = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
            model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
            _ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tok,
                aggregation_strategy="simple",
                device=device,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[deterministic] WARNING: failed to load NER model ({exc}); "
                  f"falling back to regex-only factual-unit detection.")
            _ner_load_failed = True
            _ner_pipeline = None
    return _ner_pipeline


def extract_spans(text: str) -> list[Span]:
    text = text or ""
    spans: list[Span] = []
    ner = _get_ner_pipeline()
    if ner is not None and text.strip():
        try:
            for ent in ner(text):
                typ = _KLUE_TO_FACTUAL_UNIT.get(ent.get("entity_group"))
                if typ is None:
                    continue
                span_text = ent["word"].strip()
                if len(span_text) < 1:
                    continue
                spans.append(
                    Span(
                        type=typ,
                        text=span_text,
                        start=int(ent.get("start", -1) or -1),
                        end=int(ent.get("end", -1) or -1),
                        source="ner",
                        confidence=float(ent.get("score", 0.0)),
                    )
                )
        except Exception as exc:  # pragma: no cover
            print(f"[deterministic] WARNING: NER inference failed on a span ({exc}); skipping.")
    seen_number_spans = {(s.start, s.end) for s in spans if s.type == "NUMBER_QUANTITY"}
    for m in _NUMBER_RE.finditer(text):
        span_text = m.group(0).strip()
        if not any(ch.isdigit() for ch in span_text):
            continue
        key = (m.start(), m.end())
        if key in seen_number_spans:
            continue
        # skip if this numeric span is already covered by an NER DATE_TIME span
        if any(s.type == "DATE_TIME" and s.start <= m.start() and m.end() <= s.end for s in spans):
            continue
        spans.append(Span(type="NUMBER_QUANTITY", text=span_text, start=m.start(), end=m.end(), source="regex", confidence=0.9))
    return spans


def _dedupe_spans(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    seen: set[tuple[str, str]] = set()
    for s in spans:
        key = (s.type, s.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _preservation(source_spans: list[Span], rewrite_text: str) -> tuple[float | None, int, int]:
    if not source_spans:
        return None, 0, 0
    matched = sum(1 for s in source_spans if loose_contains(rewrite_text, s.text))
    return matched / len(source_spans), matched, len(source_spans)


def _unsupported(rewrite_spans: list[Span], source_text: str) -> tuple[float | None, int, int]:
    if not rewrite_spans:
        return None, 0, 0
    unsupported = sum(1 for s in rewrite_spans if not loose_contains(source_text, s.text))
    return unsupported / len(rewrite_spans), unsupported, len(rewrite_spans)


def score(source_text: str, rewrite_text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic factual-unit preservation score.

    Returns a partial score dict (only the fields this baseline can speak to);
    callers merge it with other baselines. All *_score/*_preservation/*_consistency
    fields are higher-is-better in [0, 1].
    """
    source_text = normalize_text(source_text)
    rewrite_text = normalize_text(rewrite_text)

    source_spans = _dedupe_spans(extract_spans(source_text))
    rewrite_spans = _dedupe_spans(extract_spans(rewrite_text)) if rewrite_text else []

    by_type: dict[str, list[Span]] = {}
    for s in source_spans:
        by_type.setdefault(s.type, []).append(s)

    per_type_scores: dict[str, float | None] = {}
    per_type_counts: dict[str, tuple[int, int]] = {}
    for typ, spans in by_type.items():
        p, matched, total = _preservation(spans, rewrite_text)
        per_type_scores[typ] = p
        per_type_counts[typ] = (matched, total)

    entity_types = ["PERSON", "ORGANIZATION", "LOCATION"]
    entity_scores = [per_type_scores[t] for t in entity_types if t in per_type_scores]
    entity_preservation = safe_mean(entity_scores)

    number_preservation = per_type_scores.get("NUMBER_QUANTITY")
    date_preservation = per_type_scores.get("DATE_TIME")

    unsupported_score, unsup_count, unsup_total = _unsupported(rewrite_spans, source_text)

    factual_unit_components = [v for v in [entity_preservation, number_preservation, date_preservation] if v is not None]
    factual_unit_score = safe_mean(factual_unit_components)

    # Hard violation: at least one salient factual-unit type (<=2 source
    # instances, i.e. not a bulk list) is almost entirely dropped/substituted.
    hard_violation = 0
    for typ, (matched, total) in per_type_counts.items():
        if typ not in {"PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME", "NUMBER_QUANTITY"}:
            continue
        if total == 0:
            continue
        rate = matched / total
        if total <= 2 and rate == 0.0:
            hard_violation = 1
            break
        if rate < 0.34:
            hard_violation = 1
            break

    surface_overlap = char_ngram_jaccard(source_text, rewrite_text) if rewrite_text else 0.0

    return {
        "factual_unit_score": factual_unit_score,
        "entity_preservation": entity_preservation,
        "number_preservation": number_preservation,
        "date_preservation": date_preservation,
        "unsupported_content_score": unsupported_score,
        "hard_violation": hard_violation,
        "details": {
            "deterministic": {
                "source_spans": [s.__dict__ for s in source_spans],
                "rewrite_spans": [s.__dict__ for s in rewrite_spans],
                "per_type_preservation": per_type_scores,
                "per_type_counts": {k: {"matched": v[0], "total": v[1]} for k, v in per_type_counts.items()},
                "unsupported_spans": unsup_count,
                "unsupported_total": unsup_total,
                "surface_char3_jaccard": surface_overlap,
            }
        },
    }
