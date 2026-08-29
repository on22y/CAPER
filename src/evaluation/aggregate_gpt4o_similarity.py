#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, atomic_json, atomic_jsonl, atomic_text, read_jsonl, sha256


def qstats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
        "p25": float(np.quantile(array, .25)), "p75": float(np.quantile(array, .75)),
        "min": float(array.min()), "max": float(array.max()),
    }


def usage_sum(rows: list[dict], pricing: dict) -> dict:
    result = {key: sum(row.get(key) or 0 for row in rows) for key in (
        "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"
    )}
    result["uncached_input_tokens"] = result["input_tokens"] - result["cached_input_tokens"]
    result["api_calls"] = len(rows)
    result["estimated_cost_usd"] = (
        result["uncached_input_tokens"] * pricing["input_usd_per_1m"]
        + result["cached_input_tokens"] * pricing["cached_input_usd_per_1m"]
        + result["output_tokens"] * pricing["output_usd_per_1m"]
    ) / 1e6
    return result


def main() -> None:
    raw_path = ROOT / "gpt4o/raw/substantive_similarity_final.jsonl"
    rows = read_jsonl(raw_path)
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["sample_id"], row["order"])
        if row["order"] in grouped[row["sample_id"]]:
            raise RuntimeError(f"duplicate key: {key}")
        grouped[row["sample_id"]][row["order"]] = row
    processed = []
    for sample_id, pair in sorted(grouped.items()):
        if set(pair) != {"original", "swapped"}:
            continue
        original, swapped = pair["original"], pair["swapped"]
        base_a = original["candidate_a_similarity"]
        dpo_b = original["candidate_b_similarity"]
        dpo_a = swapped["candidate_a_similarity"]
        base_b = swapped["candidate_b_similarity"]
        base_mean = (base_a + base_b) / 2
        dpo_mean = (dpo_a + dpo_b) / 2
        gap = dpo_mean - base_mean
        direction = "base_less_similar" if gap > 0 else "same_similarity" if gap == 0 else "dpo_less_similar"
        processed.append({
            "sample_id": sample_id, "domain": original["domain"],
            "base_similarity_A_position": base_a, "base_similarity_B_position": base_b,
            "dpo_similarity_A_position": dpo_a, "dpo_similarity_B_position": dpo_b,
            "base_similarity_mean": base_mean, "dpo_similarity_mean": dpo_mean,
            "similarity_gap_dpo_minus_base": gap, "ordinal_direction": direction,
            "base_position_abs_diff": abs(base_a - base_b), "dpo_position_abs_diff": abs(dpo_a - dpo_b),
        })
    if len(rows) != 102 or len(processed) != 51:
        raise RuntimeError(f"Incomplete GPT evaluation: judgments={len(rows)} pairs={len(processed)}")

    pair_path = ROOT / "gpt4o/processed/substantive_similarity_pair_scores_51.jsonl"
    atomic_jsonl(pair_path, processed)
    summary = {
        "pairs": len(processed), "judgments": len(rows),
        "base_score_distribution": dict(Counter(value for row in processed for value in (
            row["base_similarity_A_position"], row["base_similarity_B_position"]
        ))),
        "dpo_score_distribution": dict(Counter(value for row in processed for value in (
            row["dpo_similarity_A_position"], row["dpo_similarity_B_position"]
        ))),
        "base_similarity": qstats([row["base_similarity_mean"] for row in processed]),
        "dpo_similarity": qstats([row["dpo_similarity_mean"] for row in processed]),
        "similarity_gap_dpo_minus_base": qstats([row["similarity_gap_dpo_minus_base"] for row in processed]),
        "ordinal_direction": dict(Counter(row["ordinal_direction"] for row in processed)),
        "presentation_stability": {
            "base_exact": sum(row["base_position_abs_diff"] == 0 for row in processed),
            "dpo_exact": sum(row["dpo_position_abs_diff"] == 0 for row in processed),
            "base_mean_abs_diff": float(np.mean([row["base_position_abs_diff"] for row in processed])),
            "dpo_mean_abs_diff": float(np.mean([row["dpo_position_abs_diff"] for row in processed])),
        },
        "by_domain": {},
    }
    for domain in ("news", "petition", "sns"):
        subset = [row for row in processed if row["domain"] == domain]
        summary["by_domain"][domain] = {
            "pairs": len(subset), "ordinal_direction": dict(Counter(row["ordinal_direction"] for row in subset)),
            "base_similarity": qstats([row["base_similarity_mean"] for row in subset]),
            "dpo_similarity": qstats([row["dpo_similarity_mean"] for row in subset]),
            "gap_dpo_minus_base": qstats([row["similarity_gap_dpo_minus_base"] for row in subset]),
        }
    summary_path = ROOT / "gpt4o/processed/substantive_similarity_summary.json"
    atomic_json(summary_path, summary)

    pricing = {
        "model": "gpt-4o", "pricing_basis": "per_1m_tokens", "input_usd_per_1m": 2.5,
        "cached_input_usd_per_1m": 1.25, "output_usd_per_1m": 10.0,
        "checked_at": "2026-08-26", "source": "https://developers.openai.com/api/docs/models/gpt-4o",
        "source_description": "OpenAI official GPT-4o model documentation",
    }
    atomic_json(ROOT / "metadata/gpt4o_pricing_used.json", pricing)
    usage = read_jsonl(ROOT / "gpt4o/raw/substantive_similarity_api_usage.jsonl")
    final_ids = {row["response_id"] for row in rows}
    final_usage = [row for row in usage if row["run_type"] == "final" and row.get("response_id") in final_ids]
    overhead = [row for row in usage if row not in final_usage]
    usage_summary = {
        "model": "gpt-4o", "final_evaluation": usage_sum(final_usage, pricing),
        "validation_and_retry_overhead": usage_sum(overhead, pricing),
        "entire_experiment": usage_sum(usage, pricing), "pricing": pricing,
    }
    usage_path = ROOT / "gpt4o/processed/substantive_similarity_api_usage_summary.json"
    atomic_json(usage_path, usage_summary)
    directions = summary["ordinal_direction"]
    report = [
        "# GPT-4o Substantive Expressive Similarity: Base vs DPO501", "",
        "Canonical substantive_similarity_v1 prompt, structured decoding, original/swapped protocol, parser, and aggregation reused.", "",
        "| Result | Count | Ratio |", "|---|---:|---:|",
        f"| Base less similar | {directions.get('base_less_similar', 0)} | {directions.get('base_less_similar', 0)/51:.2%} |",
        f"| Same similarity | {directions.get('same_similarity', 0)} | {directions.get('same_similarity', 0)/51:.2%} |",
        f"| DPO less similar | {directions.get('dpo_less_similar', 0)} | {directions.get('dpo_less_similar', 0)/51:.2%} |",
        "", f"- Base similarity mean: `{summary['base_similarity']['mean']:.6f}`",
        f"- DPO similarity mean: `{summary['dpo_similarity']['mean']:.6f}`",
        f"- Gap mean (DPO - Base): `{summary['similarity_gap_dpo_minus_base']['mean']:.6f}`",
    ]
    report_path = ROOT / "reports/gpt4o_substantive_similarity_report.md"
    atomic_text(report_path, "\n".join(report) + "\n")
    cost = usage_summary["entire_experiment"]
    cost_path = ROOT / "reports/gpt4o_api_usage_cost_report.md"
    atomic_text(cost_path, "\n".join([
        "# GPT-4o API Usage and Estimated Cost", "",
        f"- API calls: {cost['api_calls']}", f"- Input tokens: {cost['input_tokens']}",
        f"- Cached input tokens: {cost['cached_input_tokens']}", f"- Output tokens: {cost['output_tokens']}",
        f"- Total tokens: {cost['total_tokens']}", f"- Estimated cost (USD): {cost['estimated_cost_usd']:.6f}",
        f"- Pricing checked at: {pricing['checked_at']}",
    ]) + "\n")
    atomic_json(ROOT / "metadata/gpt4o_evaluation_manifest.json", {
        "protocol": "substantive_similarity_v1 original/swapped", "expected_calls_without_retries": 102,
        "prompt": str(ROOT / "prompts/gpt4o_substantive_expressive_similarity_system_prompt_v1.txt"),
        "prompt_sha256": sha256(ROOT / "prompts/gpt4o_substantive_expressive_similarity_system_prompt_v1.txt"),
        "raw_sha256": sha256(raw_path), "pair_scores_sha256": sha256(pair_path),
        "summary_sha256": sha256(summary_path), "usage_sha256": sha256(usage_path),
    })
    print(json.dumps({"pairs": 51, "judgments": 102, "direction": directions, "api_calls": cost["api_calls"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
