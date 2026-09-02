#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from pecab import PeCab


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results/final_metrics"
INPUT = PROJECT / "data/preferences/dpo_preferences_all_survivors_501.jsonl"
CANONICAL = PROJECT / "src/evaluation/canonical_automatic_metrics.py"
PER_SAMPLE = ROOT / "per_sample_final_metrics.jsonl"
SUMMARY = ROOT / "final_metric_summary.json"
REPORT = ROOT / "final_metric_summary.md"
MANIFEST = ROOT / "final_metric_manifest.json"

METRICS = (
    "Entity F1",
    "Numeric/Date F1",
    "BiNLI Support",
    "BiNLI Coverage",
    "BiNLI F1",
    "Korean BERTScore Precision",
    "Korean BERTScore Recall",
    "Korean BERTScore F1",
    "QA-based factuality",
    "chrF++",
    "Morpheme ROUGE-L",
    "Extractive Coverage",
    "Extractive Density",
    "Novel-2",
    "Novel-4",
)
FACTUAL = set(METRICS[:9])
LOWER_DUPLICATE = {"chrF++", "Morpheme ROUGE-L", "Extractive Coverage", "Extractive Density"}
HIGHER_NOVELTY = {"Novel-2", "Novel-4"}
DOMAINS = ("news", "petition", "sns")
ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION"}
pecab = PeCab()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def checkpoint(records: list[dict[str, Any]]) -> None:
    atomic_text(PER_SAMPLE, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records))


def import_canonical():
    scripts_dir = str(CANONICAL.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("canonical_final_metrics", CANONICAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {CANONICAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_entities(text: str, metric) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for chunk, _ in metric.char_chunks(text):
        for span in metric.det_mod.extract_spans(chunk):
            if span.type in ENTITY_TYPES:
                normalized = metric.normalize_text(span.text)
                if normalized:
                    found.add((span.type, normalized))
    return sorted(found)


def entity_f1(source: str, rewrite: str, metric) -> tuple[float | None, dict[str, int]]:
    src = extract_entities(source, metric)
    rew = extract_entities(rewrite, metric)
    if not src:
        return None, {"source": 0, "rewrite": len(rew), "matched_source": 0, "matched_rewrite": 0}
    matched_source = sum(metric.det_mod.loose_contains(rewrite, value) for _, value in src)
    matched_rewrite = sum(metric.det_mod.loose_contains(source, value) for _, value in rew)
    recall = matched_source / len(src)
    precision = matched_rewrite / len(rew) if rew else 0.0
    return harmonic(precision, recall), {
        "source": len(src), "rewrite": len(rew),
        "matched_source": matched_source, "matched_rewrite": matched_rewrite,
    }


def numeric_date_f1(source: str, rewrite: str, metric) -> tuple[float | None, dict[str, int]]:
    src = Counter(metric.mention_norm(m.group(0)) for m in metric.NUM_DATE_RE.finditer(source) if any(c.isdigit() for c in m.group(0)))
    rew = Counter(metric.mention_norm(m.group(0)) for m in metric.NUM_DATE_RE.finditer(rewrite) if any(c.isdigit() for c in m.group(0)))
    source_n = sum(src.values())
    rewrite_n = sum(rew.values())
    matched = sum(min(count, rew[key]) for key, count in src.items())
    if source_n == 0:
        return None, {"source": 0, "rewrite": rewrite_n, "matched": 0}
    recall = matched / source_n
    precision = matched / rewrite_n if rewrite_n else 0.0
    return harmonic(precision, recall), {"source": source_n, "rewrite": rewrite_n, "matched": matched}


def morphs(text: str) -> list[str]:
    return pecab.morphs(text or "")


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def ngram_stats(reference: list[str], hypothesis: list[str], n: int) -> tuple[int, int, int]:
    ref = Counter(ngrams(reference, n))
    hyp = Counter(ngrams(hypothesis, n))
    matches = sum(min(count, ref[key]) for key, count in hyp.items())
    return matches, sum(hyp.values()), sum(ref.values())


def chrf_pp(reference: str, hypothesis: str, beta: float = 2.0) -> float:
    ref_chars = list(re.sub(r"\s+", "", reference))
    hyp_chars = list(re.sub(r"\s+", "", hypothesis))
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    precisions, recalls = [], []
    for order in range(1, 7):
        matched, hyp_total, ref_total = ngram_stats(ref_chars, hyp_chars, order)
        precisions.append(matched / hyp_total if hyp_total else 0.0)
        recalls.append(matched / ref_total if ref_total else 0.0)
    for order in range(1, 3):
        matched, hyp_total, ref_total = ngram_stats(ref_words, hyp_words, order)
        precisions.append(matched / hyp_total if hyp_total else 0.0)
        recalls.append(matched / ref_total if ref_total else 0.0)
    precision = statistics.fmean(precisions)
    recall = statistics.fmean(recalls)
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0


def lcs_len(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for index, right in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def morpheme_rouge_l(reference: list[str], hypothesis: list[str]) -> float | None:
    if not reference or not hypothesis:
        return None
    matched = lcs_len(reference, hypothesis)
    return harmonic(matched / len(hypothesis), matched / len(reference))


def extractive_fragments(source: list[str], rewrite: list[str]) -> list[int]:
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(source):
        positions.setdefault(token, []).append(index)
    lengths = []
    rewrite_index = 0
    while rewrite_index < len(rewrite):
        best = 0
        for source_index in positions.get(rewrite[rewrite_index], []):
            length = 0
            while (rewrite_index + length < len(rewrite) and source_index + length < len(source)
                   and rewrite[rewrite_index + length] == source[source_index + length]):
                length += 1
            best = max(best, length)
        if best:
            lengths.append(best)
            rewrite_index += best
        else:
            rewrite_index += 1
    return lengths


def surface_metrics(source: str, rewrite: str) -> dict[str, float | None]:
    source_morphs = morphs(source)
    rewrite_morphs = morphs(rewrite)
    fragments = extractive_fragments(source_morphs, rewrite_morphs)
    denominator = len(rewrite_morphs)
    values: dict[str, float | None] = {
        "chrF++": chrf_pp(source, rewrite),
        "Morpheme ROUGE-L": morpheme_rouge_l(source_morphs, rewrite_morphs),
        "Extractive Coverage": sum(fragments) / denominator if denominator else None,
        "Extractive Density": sum(length * length for length in fragments) / denominator if denominator else None,
    }
    source_sets = {n: set(ngrams(source_morphs, n)) for n in (2, 4)}
    for n in (2, 4):
        rewrite_ngrams = ngrams(rewrite_morphs, n)
        values[f"Novel-{n}"] = (
            sum(item not in source_sets[n] for item in rewrite_ngrams) / len(rewrite_ngrams)
            if rewrite_ngrams else None
        )
    return values


def finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def build_summary(records: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["sample_id"], row["comparison"]): row for row in records}
    try:
        from scipy.stats import wilcoxon
        scipy_available = True
    except ImportError:
        wilcoxon = None
        scipy_available = False
    scopes: dict[str, Any] = {}
    for scope in ("overall",) + DOMAINS:
        scope_pairs = pairs if scope == "overall" else [row for row in pairs if row["domain"] == scope]
        scopes[scope] = {}
        for name in METRICS:
            paired = []
            for pair in scope_pairs:
                chosen = lookup[(pair["sample_id"], "source↔chosen")]["metrics"].get(name)
                rejected = lookup[(pair["sample_id"], "source↔rejected")]["metrics"].get(name)
                if finite(chosen) and finite(rejected):
                    paired.append((float(chosen), float(rejected)))
            chosen_values = [row[0] for row in paired]
            rejected_values = [row[1] for row in paired]
            deltas = [left - right for left, right in paired]
            pvalue = None
            if scipy_available and paired and any(abs(value) > 1e-12 for value in deltas):
                try:
                    pvalue = float(wilcoxon(chosen_values, rejected_values).pvalue)
                except ValueError:
                    pvalue = None
            scopes[scope][name] = {
                "valid_n": len(paired),
                "chosen_mean": statistics.fmean(chosen_values) if chosen_values else None,
                "rejected_mean": statistics.fmean(rejected_values) if rejected_values else None,
                "chosen_minus_rejected_delta": statistics.fmean(deltas) if deltas else None,
                "wilcoxon_p": pvalue,
            }
    failures = {}
    for name in METRICS:
        chosen_valid = sum(finite(row["metrics"].get(name)) for row in records if row["comparison"] == "source↔chosen")
        rejected_valid = sum(finite(row["metrics"].get(name)) for row in records if row["comparison"] == "source↔rejected")
        failures[name] = {
            "chosen_valid": chosen_valid,
            "chosen_nan_or_failure": 501 - chosen_valid,
            "rejected_valid": rejected_valid,
            "rejected_nan_or_failure": 501 - rejected_valid,
        }
    return {
        "input_pairs": len(pairs),
        "evaluation_rows": len(records),
        "domain_counts": dict(Counter(row["domain"] for row in pairs)),
        "chosen_identity_counts": dict(Counter(row["chosen_identity"] for row in pairs)),
        "metric_directions": {
            "higher_is_better_factual_preservation": [name for name in METRICS if name in FACTUAL],
            "lower_is_less_source_expression_duplicate": [name for name in METRICS if name in LOWER_DUPLICATE],
            "higher_is_more_expression_novelty": [name for name in METRICS if name in HIGHER_NOVELTY],
        },
        "metric_status": {name: ("BLOCKED" if name == "QA-based factuality" else "COMPLETED") for name in METRICS},
        "validity": failures,
        "scipy_wilcoxon_available": scipy_available,
        "scopes": scopes,
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# DPO 501 Final Metric Summary", "",
        "Measurement only. No pair addition, deletion, mutation, filtering, thresholding, or training was performed.", "",
        f"- Input pairs: {summary['input_pairs']}",
        f"- Evaluation rows: {summary['evaluation_rows']}",
        f"- Domain counts: {summary['domain_counts']}",
        f"- Chosen identity counts: {summary['chosen_identity_counts']}",
        "- QA-based factuality: BLOCKED (no existing local implementation and no cached local QA model)", "",
        "## Definitions", "",
        "- BiNLI Support: canonical chunked NLI entailment for SOURCE premise -> REWRITE hypothesis.",
        "- BiNLI Coverage: the same canonical calculation in reverse, REWRITE premise -> SOURCE hypothesis.",
        "- BiNLI F1: harmonic mean of Support and Coverage.",
        "- Korean BERTScore: canonical chunked BERTScore P/R/F1 using locally cached xlm-roberta-large.",
        "- Morpheme metrics: PeCab 1.0.8 surface morphemes.", "",
        "| Scope | Metric | Valid N | Chosen mean | Rejected mean | Chosen - Rejected | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope in ("overall",) + DOMAINS:
        for name in METRICS:
            row = summary["scopes"][scope][name]
            lines.append(
                f"| {scope} | {name} | {row['valid_n']} | {fmt(row['chosen_mean'])} | "
                f"{fmt(row['rejected_mean'])} | {fmt(row['chosen_minus_rejected_delta'])} | {fmt(row['wilcoxon_p'])} |"
            )
    lines.extend(["", "Full validity/failure counts and metric directions are stored in `outputs/final_metric_summary.json`."])
    atomic_text(REPORT, "\n".join(lines) + "\n")


def make_manifest(metric) -> dict[str, Any]:
    return {
        "input": str(INPUT),
        "canonical_reused_script": str(CANONICAL),
        "expected_pairs": 501,
        "expected_evaluation_rows": 1002,
        "metrics": list(METRICS),
        "models": {
            "entity": {"id": metric.NER_ID, "local_only": True},
            "BiNLI": {"id": metric.NLI_ID, "local_only": True},
            "Korean BERTScore": {"id": metric.BERTSCORE_ID, "local_only": True},
            "QA-based factuality": {"status": "BLOCKED", "reason": "No existing local implementation and no cached local QA model."},
        },
        "tokenizers": {
            "morpheme": {"name": "PeCab", "version": importlib.metadata.version("pecab")},
            "BiNLI": metric.NLI_ID,
            "Korean BERTScore": metric.BERTSCORE_ID,
        },
        "definitions": {
            "Entity F1": "NER entity precision/recall F1; existing KLUE NER extractor and loose containment matcher; valid when SOURCE has >=1 entity.",
            "Numeric/Date F1": "Multiset precision/recall F1 over canonical numeric/date regex mentions; valid when SOURCE has >=1 mention.",
            "BiNLI Support": "Canonical chunked entailment SOURCE->REWRITE.",
            "BiNLI Coverage": "Canonical chunked entailment REWRITE->SOURCE.",
            "BiNLI F1": "Harmonic mean of BiNLI Support and Coverage.",
            "Korean BERTScore Precision/Recall/F1": "Canonical token-safe cross-chunk BERTScore with best-match token weighting.",
            "QA-based factuality": "BLOCKED; no substitute metric used.",
            "chrF++": "Sentence-level chrF++, character orders 1..6 plus whitespace-word orders 1..2, beta=2.",
            "Morpheme ROUGE-L": "LCS precision/recall F1 over PeCab morphemes.",
            "Extractive Coverage": "Sum of maximal extractive fragment lengths divided by rewrite morpheme count.",
            "Extractive Density": "Sum of squared maximal extractive fragment lengths divided by rewrite morpheme count.",
            "Novel-2/Novel-4": "Fraction of rewrite PeCab morpheme n-gram occurrences absent from the SOURCE n-gram set.",
        },
        "canonical_chunking": {
            "BiNLI": "premise windows <=240 tokens with one-sentence overlap; hypothesis chunks <=120; best premise-window entailment per hypothesis chunk; hypothesis-token-weighted mean",
            "Korean BERTScore": "token-safe chunks <= model limit-24; all cross-chunk scores; best-match token-weighted precision/recall",
            "entity": "350-character chunks with 60-character overlap and deduplication",
        },
        "environment": {"conda_env": "hclt", "offline": True},
    }


def preflight(pairs: list[dict[str, Any]], metric) -> None:
    required = {"sample_id", "domain", "source", "chosen", "rejected", "chosen_identity"}
    if len(pairs) != 501:
        raise RuntimeError(f"Expected 501 pairs, found {len(pairs)}")
    if len({row["sample_id"] for row in pairs}) != 501:
        raise RuntimeError("Duplicate sample_id")
    if any(not required <= row.keys() for row in pairs):
        raise RuntimeError("Missing required input fields")
    if Counter(row["domain"] for row in pairs) != Counter({"news": 140, "petition": 172, "sns": 189}):
        raise RuntimeError("Domain count mismatch")
    if Counter(row["chosen_identity"] for row in pairs) != Counter({"TRANSFORMATIVE": 456, "CONSERVATIVE": 45}):
        raise RuntimeError("Chosen identity count mismatch")
    atomic_json(MANIFEST, make_manifest(metric))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    pairs = read_jsonl(INPUT)
    metric = import_canonical()
    preflight(pairs, metric)
    if args.preflight:
        print("PREFLIGHT_OK pairs=501 rows=1002 metrics=15 qa=BLOCKED", flush=True)
        return

    records: list[dict[str, Any]] = []
    for pair in pairs:
        for system, comparison in (("chosen", "source↔chosen"), ("rejected", "source↔rejected")):
            rewrite = pair[system]
            entity, entity_counts = entity_f1(pair["source"], rewrite, metric)
            numeric, numeric_counts = numeric_date_f1(pair["source"], rewrite, metric)
            values = {name: None for name in METRICS}
            values.update({"Entity F1": entity, "Numeric/Date F1": numeric, **surface_metrics(pair["source"], rewrite)})
            records.append({
                "sample_id": pair["sample_id"], "domain": pair["domain"],
                "chosen_identity": pair["chosen_identity"], "comparison": comparison, "system": system,
                "source_text": pair["source"], "rewrite_text": rewrite, "metrics": values,
                "diagnostics": {"entity_counts": entity_counts, "numeric_date_counts": numeric_counts},
            })
    checkpoint(records)
    metric.det_mod._ner_pipeline = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    canonical_records = [{
        "sample_id": row["sample_id"], "domain": row["domain"], "system": row["system"],
        "source_text": row["source_text"], "rewrite_text": row["rewrite_text"],
    } for row in records]
    forward, nli_forward_overflow = metric.run_nli(canonical_records, device)
    reverse_input = [{
        "sample_id": row["sample_id"], "domain": row["domain"], "system": row["system"],
        "source_text": row["rewrite_text"], "rewrite_text": row["source_text"],
    } for row in records]
    reverse, nli_reverse_overflow = metric.run_nli(reverse_input, device)
    for row, fwd, rev in zip(records, forward, reverse):
        support = fwd["nli_entailment"]
        coverage = rev["nli_entailment"]
        row["metrics"].update({
            "BiNLI Support": support,
            "BiNLI Coverage": coverage,
            "BiNLI F1": harmonic(support, coverage),
        })
        row["diagnostics"]["binli"] = {
            "support_contradiction": fwd["nli_contradiction_probability"],
            "coverage_contradiction": rev["nli_contradiction_probability"],
            "support_source_chunks": fwd["nli_source_chunks"],
            "support_rewrite_chunks": fwd["nli_rewrite_chunks"],
            "coverage_source_chunks": rev["nli_source_chunks"],
            "coverage_rewrite_chunks": rev["nli_rewrite_chunks"],
        }
    checkpoint(records)

    bert_input = [{
        "sample_id": row["sample_id"], "domain": row["domain"], "system": row["system"],
        "source_text": row["source_text"], "rewrite_text": row["rewrite_text"],
    } for row in records]
    bert_rows, bert_overflow = metric.run_bertscore(bert_input, device)
    for row, scored in zip(records, bert_rows):
        row["metrics"].update({
            "Korean BERTScore Precision": scored["bertscore_precision"],
            "Korean BERTScore Recall": scored["bertscore_recall"],
            "Korean BERTScore F1": scored["bertscore_f1"],
        })
        row["diagnostics"]["bertscore"] = {
            "source_chunks": scored["bertscore_source_chunks"],
            "rewrite_chunks": scored["bertscore_rewrite_chunks"],
        }
    checkpoint(records)

    manifest = make_manifest(metric)
    manifest["overflow"] = {
        "BiNLI Support": nli_forward_overflow,
        "BiNLI Coverage": nli_reverse_overflow,
        "Korean BERTScore": bert_overflow,
    }
    atomic_json(MANIFEST, manifest)
    summary = build_summary(records, pairs)
    atomic_json(SUMMARY, summary)
    write_report(summary)
    print("FINAL_METRICS_COMPLETE pairs=501 rows=1002 metrics_completed=14 metrics_blocked=1", flush=True)


if __name__ == "__main__":
    main()
