#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any


MATCH_TYPES = {"exact", "near_verbatim", "paraphrastic", "structural"}
ELEMENT_TYPES = {
    "fact_idea", "stock_expression", "name_number_term", "third_party_quote",
    "creative_wording", "selection_arrangement", "concrete_tone_voice",
}
PROTECTABILITY = {"unlikely", "uncertain", "plausible"}
REASON_CODES = {
    "FACT", "IDEA", "NAME_NUMBER", "TECHNICAL_TERM", "STOCK_EXPRESSION",
    "THIRD_PARTY_QUOTE", "CONSTRAINED_EXPRESSION",
}
RISK_EFFECTS = {"negligible", "minor", "moderate", "substantial"}
UNCERTAINTY = {"low", "medium", "high"}
DIMENSIONS = {
    "literal_expression", "paraphrastic_expression", "selection_arrangement_structure",
    "qualitative_salience", "quantitative_extent",
}
REQUIRED_FIELDS = {
    "representative_matches", "document_structure", "excluded_overlaps",
    "material_differences", "dimension_scores", "overall_concern", "uncertainty",
    "uncertainty_reasons", "short_rationale",
}


def parse_json_object(raw: str | None) -> tuple[dict[str, Any] | None, str, str | None]:
    if raw is None or not raw.strip():
        return None, "empty_response", "response is empty"
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            parsed, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, "parsed", None
    return None, "json_parse_error", "no complete JSON object found"


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _normalized_quote(text: str) -> str:
    translation = str.maketrans({
        "“": '"', "”": '"', "„": '"', "＂": '"',
        "‘": "'", "’": "'", "‚": "'", "＇": "'",
    })
    return re.sub(r"\s+", "", text.translate(translation)).lower()


def overlap_metrics(source_text: str, target_text: str, n: int = 5) -> dict[str, float]:
    source = _normalized(source_text)
    target = _normalized(target_text)
    if not source or not target:
        return {"sequence_ratio": 0.0, "char_ngram_containment": 0.0}
    sequence_ratio = SequenceMatcher(None, source, target, autojunk=False).ratio()
    source_ngrams = {source[i:i + n] for i in range(max(0, len(source) - n + 1))}
    target_ngrams = {target[i:i + n] for i in range(max(0, len(target) - n + 1))}
    denominator = min(len(source_ngrams), len(target_ngrams))
    containment = len(source_ngrams & target_ngrams) / denominator if denominator else float(source == target)
    return {
        "sequence_ratio": round(sequence_ratio, 6),
        "char_ngram_containment": round(containment, 6),
    }


def _score(value: Any, path: str, errors: list[str]) -> bool:
    if type(value) is not int or not 0 <= value <= 4:
        errors.append(f"{path}:expected_integer_0_4")
        return False
    return True


def _enum(value: Any, allowed: set[str], path: str, errors: list[str]) -> None:
    if value not in allowed:
        errors.append(f"{path}:invalid_enum")


def _text(
    value: Any,
    path: str,
    errors: list[str],
    *,
    max_length: int,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, str):
        errors.append(f"{path}:expected_string")
    elif not allow_empty and not value.strip():
        errors.append(f"{path}:empty_string")
    elif len(value) > max_length:
        errors.append(f"{path}:too_long:{len(value)}>{max_length}")


def _quote(value: Any, document: str, path: str, errors: list[str]) -> None:
    _text(value, path, errors, max_length=180)
    if (
        isinstance(value, str)
        and value.strip()
        and _normalized_quote(value) not in _normalized_quote(document)
    ):
        errors.append(f"{path}:not_in_input_document")


def validate_document_response(
    parsed: dict[str, Any],
    source_text: str,
    target_text: str,
    *,
    generation_reached_limit: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(parsed))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))

    matches = parsed.get("representative_matches")
    if not isinstance(matches, list):
        errors.append("representative_matches:expected_list")
        matches = []
    elif len(matches) > 4:
        errors.append(f"representative_matches:too_many:{len(matches)}>4")
    for index, item in enumerate(matches):
        path = f"representative_matches[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{path}:expected_object")
            continue
        required = {
            "source_quote", "target_quote", "match_type", "element_type",
            "protectability", "similarity", "qualitative_importance", "reason",
        }
        absent = sorted(required - set(item))
        if absent:
            errors.append(f"{path}:missing_fields:{','.join(absent)}")
        _quote(item.get("source_quote"), source_text, f"{path}.source_quote", errors)
        _quote(item.get("target_quote"), target_text, f"{path}.target_quote", errors)
        _enum(item.get("match_type"), MATCH_TYPES, f"{path}.match_type", errors)
        _enum(item.get("element_type"), ELEMENT_TYPES, f"{path}.element_type", errors)
        _enum(item.get("protectability"), PROTECTABILITY, f"{path}.protectability", errors)
        _score(item.get("similarity"), f"{path}.similarity", errors)
        _score(item.get("qualitative_importance"), f"{path}.qualitative_importance", errors)
        _text(item.get("reason"), f"{path}.reason", errors, max_length=240)

    structure = parsed.get("document_structure")
    if not isinstance(structure, dict):
        errors.append("document_structure:expected_object")
        structure = {}
    structure_required = {
        "similarity", "protectability", "order_preserved", "shared_sequence_summary", "reason",
    }
    absent = sorted(structure_required - set(structure))
    if absent:
        errors.append("document_structure:missing_fields:" + ",".join(absent))
    _score(structure.get("similarity"), "document_structure.similarity", errors)
    _enum(structure.get("protectability"), PROTECTABILITY, "document_structure.protectability", errors)
    if type(structure.get("order_preserved")) is not bool:
        errors.append("document_structure.order_preserved:expected_boolean")
    _text(
        structure.get("shared_sequence_summary"), "document_structure.shared_sequence_summary",
        errors, max_length=500, allow_empty=True,
    )
    _text(structure.get("reason"), "document_structure.reason", errors, max_length=500)

    excluded = parsed.get("excluded_overlaps")
    if not isinstance(excluded, list):
        errors.append("excluded_overlaps:expected_list")
        excluded = []
    elif len(excluded) > 4:
        errors.append(f"excluded_overlaps:too_many:{len(excluded)}>4")
    for index, item in enumerate(excluded):
        path = f"excluded_overlaps[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{path}:expected_object")
            continue
        required = {"source_quote", "target_quote", "reason_code", "reason"}
        absent = sorted(required - set(item))
        if absent:
            errors.append(f"{path}:missing_fields:{','.join(absent)}")
        _quote(item.get("source_quote"), source_text, f"{path}.source_quote", errors)
        _quote(item.get("target_quote"), target_text, f"{path}.target_quote", errors)
        _enum(item.get("reason_code"), REASON_CODES, f"{path}.reason_code", errors)
        _text(item.get("reason"), f"{path}.reason", errors, max_length=240)

    differences = parsed.get("material_differences")
    if not isinstance(differences, list):
        errors.append("material_differences:expected_list")
        differences = []
    elif len(differences) > 4:
        errors.append(f"material_differences:too_many:{len(differences)}>4")
    for index, item in enumerate(differences):
        path = f"material_differences[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{path}:expected_object")
            continue
        _text(item.get("description"), f"{path}.description", errors, max_length=240)
        _enum(item.get("risk_reducing_effect"), RISK_EFFECTS, f"{path}.risk_reducing_effect", errors)

    dimensions = parsed.get("dimension_scores")
    if not isinstance(dimensions, dict):
        errors.append("dimension_scores:expected_object")
        dimensions = {}
    absent = sorted(DIMENSIONS - set(dimensions))
    if absent:
        errors.append("dimension_scores:missing_fields:" + ",".join(absent))
    for name in sorted(DIMENSIONS):
        _score(dimensions.get(name), f"dimension_scores.{name}", errors)

    overall_valid = _score(parsed.get("overall_concern"), "overall_concern", errors)
    _enum(parsed.get("uncertainty"), UNCERTAINTY, "uncertainty", errors)
    uncertainty_reasons = parsed.get("uncertainty_reasons")
    if not isinstance(uncertainty_reasons, list):
        errors.append("uncertainty_reasons:expected_list")
    else:
        for index, reason in enumerate(uncertainty_reasons):
            _text(reason, f"uncertainty_reasons[{index}]", errors, max_length=240)
    _text(parsed.get("short_rationale"), "short_rationale", errors, max_length=500)

    metrics = overlap_metrics(source_text, target_text)
    metrics["diagnostic_only_not_legal_score"] = True
    if generation_reached_limit:
        warnings.append("generation_reached_max_new_tokens:possible_truncation")
    overall = parsed.get("overall_concern")
    plausible_match = any(
        isinstance(item, dict) and item.get("protectability") == "plausible" for item in matches
    )
    structural_evidence = (
        structure.get("protectability") == "plausible"
        and type(structure.get("similarity")) is int
        and structure.get("similarity") >= 2
    )
    if overall_valid and overall >= 3 and not plausible_match and not structural_evidence:
        warnings.append("high_overall_without_plausible_match_or_structural_evidence")
    structure_score = dimensions.get("selection_arrangement_structure")
    if type(structure_score) is int and structure_score >= 3:
        if not str(structure.get("shared_sequence_summary", "")).strip() or not str(structure.get("reason", "")).strip():
            warnings.append("high_structure_score_without_structure_explanation")
    quantitative = dimensions.get("quantitative_extent")
    containment = metrics["char_ngram_containment"]
    if type(quantitative) is int and quantitative >= 3 and containment < 0.10:
        warnings.append("high_quantitative_extent_with_very_low_text_overlap")
    if type(quantitative) is int and quantitative <= 1 and containment > 0.80:
        warnings.append("low_quantitative_extent_with_very_high_text_overlap")
    literal = dimensions.get("literal_expression")
    plausible_literal_match = any(
        isinstance(item, dict)
        and item.get("protectability") == "plausible"
        and item.get("match_type") in {"exact", "near_verbatim"}
        and type(item.get("similarity")) is int
        and item.get("similarity") >= 3
        for item in matches
    )
    if plausible_literal_match and type(literal) is int and literal == 0:
        warnings.append("plausible_exact_or_near_match_with_zero_literal_score")

    if overall_valid:
        parsed["derived_concern_label"] = (
            "unlikely" if overall <= 1 else "borderline" if overall == 2 else "likely"
        )
    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }
