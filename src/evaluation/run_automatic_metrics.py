#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results/automatic_metrics"
INPUT = PROJECT / "results/generation/base_vs_dpo501_rewrites_51.jsonl"
CANONICAL = PROJECT / "src/evaluation/metric_suite.py"
PER_SAMPLE = ROOT / "per_sample_metrics_102.jsonl"
SUMMARY = ROOT / "automatic_metric_summary.json"
REPORT = ROOT / "automatic_metric_summary.md"
MANIFEST = ROOT / "automatic_metric_manifest.json"

METRICS = (
    "Entity F1", "Numeric/Date F1", "BiNLI Support", "BiNLI Coverage", "BiNLI F1",
    "Korean BERTScore Precision", "Korean BERTScore Recall", "Korean BERTScore F1",
    "chrF++", "Morpheme ROUGE-L", "Extractive Coverage", "Extractive Density", "Novel-2", "Novel-4",
)
DOMAINS = ("news", "petition", "sns")


def load_final_metric_module():
    spec = importlib.util.spec_from_file_location("dpo501_final_metrics", CANONICAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {CANONICAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def summarize(records: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    lookup = {(row["sample_id"], row["comparison"]): row for row in records}
    scopes: dict[str, Any] = {}
    for scope in ("overall",) + DOMAINS:
        selected = pairs if scope == "overall" else [row for row in pairs if row["domain"] == scope]
        scopes[scope] = {}
        for metric_name in METRICS:
            paired = []
            for pair in selected:
                base = lookup[(pair["sample_id"], "source↔base")]["metrics"].get(metric_name)
                dpo = lookup[(pair["sample_id"], "source↔dpo")]["metrics"].get(metric_name)
                if finite(base) and finite(dpo):
                    paired.append((float(base), float(dpo)))
            base_values = [value[0] for value in paired]
            dpo_values = [value[1] for value in paired]
            deltas = [base - dpo for base, dpo in paired]
            pvalue = None
            if paired and any(abs(value) > 1e-12 for value in deltas):
                try:
                    pvalue = float(wilcoxon(base_values, dpo_values).pvalue)
                except ValueError:
                    pass
            scopes[scope][metric_name] = {
                "valid_n": len(paired),
                "base_mean": statistics.fmean(base_values) if base_values else None,
                "dpo_mean": statistics.fmean(dpo_values) if dpo_values else None,
                "base_minus_dpo_delta": statistics.fmean(deltas) if deltas else None,
                "wilcoxon_p": pvalue,
            }
    validity = {}
    for metric_name in METRICS:
        validity[metric_name] = {
            system: sum(finite(row["metrics"].get(metric_name)) for row in records if row["system"] == system)
            for system in ("base", "dpo")
        }
    return {
        "input_pairs": len(pairs), "evaluation_rows": len(records),
        "domain_counts": dict(Counter(row["domain"] for row in pairs)),
        "metrics": list(METRICS), "validity": validity, "scopes": scopes,
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Base vs DPO501 Final51 Automatic Metrics", "",
        "Canonical DPO501 final metric definitions, models, tokenization, chunking, and paired Wilcoxon test reused unchanged.", "",
        "| Scope | Metric | Valid N | Base mean | DPO mean | Base - DPO | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope in ("overall",) + DOMAINS:
        for name in METRICS:
            row = summary["scopes"][scope][name]
            lines.append(
                f"| {scope} | {name} | {row['valid_n']} | {fmt(row['base_mean'])} | {fmt(row['dpo_mean'])} | "
                f"{fmt(row['base_minus_dpo_delta'])} | {fmt(row['wilcoxon_p'])} |"
            )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    fm = load_final_metric_module()
    metric = fm.import_canonical()
    pairs = read_jsonl(INPUT)
    required = {"sample_id", "domain", "source_text", "base_rewrite", "dpo_rewrite"}
    if len(pairs) != 51 or len({row["sample_id"] for row in pairs}) != 51:
        raise RuntimeError("Expected 51 unique pairs")
    if any(not required <= row.keys() for row in pairs):
        raise RuntimeError("Missing paired input fields")
    if Counter(row["domain"] for row in pairs) != Counter({"news": 17, "petition": 17, "sns": 17}):
        raise RuntimeError("Domain counts differ from 17/17/17")

    records = []
    for pair in pairs:
        for system, comparison, rewrite_key in (
            ("base", "source↔base", "base_rewrite"), ("dpo", "source↔dpo", "dpo_rewrite")
        ):
            rewrite = pair[rewrite_key]
            entity, entity_counts = fm.entity_f1(pair["source_text"], rewrite, metric)
            numeric, numeric_counts = fm.numeric_date_f1(pair["source_text"], rewrite, metric)
            values = {name: None for name in METRICS}
            values.update({"Entity F1": entity, "Numeric/Date F1": numeric, **fm.surface_metrics(pair["source_text"], rewrite)})
            records.append({
                "sample_id": pair["sample_id"], "domain": pair["domain"], "comparison": comparison,
                "system": system, "source_text": pair["source_text"], "rewrite_text": rewrite,
                "metrics": values, "diagnostics": {"entity_counts": entity_counts, "numeric_date_counts": numeric_counts},
            })
    write_jsonl(PER_SAMPLE, records)
    metric.det_mod._ner_pipeline = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    canonical_records = [{
        "sample_id": row["sample_id"], "domain": row["domain"],
        "system": "chosen" if row["system"] == "base" else "rejected",
        "source_text": row["source_text"], "rewrite_text": row["rewrite_text"],
    } for row in records]
    forward, forward_overflow = metric.run_nli(canonical_records, device)
    reverse_records = [{
        "sample_id": row["sample_id"], "domain": row["domain"],
        "system": "chosen" if row["system"] == "base" else "rejected",
        "source_text": row["rewrite_text"], "rewrite_text": row["source_text"],
    } for row in records]
    reverse, reverse_overflow = metric.run_nli(reverse_records, device)
    for row, fwd, rev in zip(records, forward, reverse):
        support, coverage = fwd["nli_entailment"], rev["nli_entailment"]
        row["metrics"].update({"BiNLI Support": support, "BiNLI Coverage": coverage, "BiNLI F1": fm.harmonic(support, coverage)})
        row["diagnostics"]["binli"] = {"forward": fwd, "reverse": rev}
    write_jsonl(PER_SAMPLE, records)

    bert_rows, bert_overflow = metric.run_bertscore(canonical_records, device)
    for row, scored in zip(records, bert_rows):
        row["metrics"].update({
            "Korean BERTScore Precision": scored["bertscore_precision"],
            "Korean BERTScore Recall": scored["bertscore_recall"],
            "Korean BERTScore F1": scored["bertscore_f1"],
        })
        row["diagnostics"]["bertscore"] = {
            "source_chunks": scored["bertscore_source_chunks"], "rewrite_chunks": scored["bertscore_rewrite_chunks"],
        }
    write_jsonl(PER_SAMPLE, records)
    summary = summarize(records, pairs)
    write_json(SUMMARY, summary)
    write_report(summary)
    write_json(MANIFEST, {
        "input": str(INPUT), "input_sha256": sha256(INPUT), "input_pairs": 51, "evaluation_rows": 102,
        "canonical_final_metric_runner": str(CANONICAL), "metrics": list(METRICS), "qa_based_factuality_excluded": True,
        "models": {"entity": metric.NER_ID, "BiNLI": metric.NLI_ID, "Korean BERTScore": metric.BERTSCORE_ID},
        "tokenization": {"morpheme": "PeCab", "chunking": "canonical DPO501 final metric runner unchanged"},
        "overflow": {"BiNLI Support": forward_overflow, "BiNLI Coverage": reverse_overflow, "Korean BERTScore": bert_overflow},
        "output_sha256": {"per_sample": sha256(PER_SAMPLE), "summary": sha256(SUMMARY), "report": sha256(REPORT)},
    })
    print("AUTOMATIC_METRICS_COMPLETE pairs=51 rows=102 metrics=14", flush=True)


if __name__ == "__main__":
    main()
