#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT = Path("/home/user20250805/gayeon/HCLT2026")
ROOT = PROJECT / "work/hcx_dpo900_mode_jury_v1"
REUSED_SCRIPT = ROOT / "scripts/create_shortened_heldout51.py"
EXCLUSION_MANIFEST = ROOT / "data/splits/final_heldout_source_51_shortened_manifest.json"
OUTPUT = ROOT / "data/splits/final_heldout_source_51_final700.jsonl"
MANIFEST = ROOT / "data/splits/final_heldout_source_51_final700_manifest.json"
SEED = 20260829
MAX_CHARS = 700
PER_DOMAIN = 17
DOMAINS = ("news", "petition", "sns")


def load_reused() -> Any:
    spec = importlib.util.spec_from_file_location("heldout_shortened_reused", REUSED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REUSED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.MAX_CHARS = MAX_CHARS
    return module


def main() -> None:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("final700 output already exists; refusing to overwrite")

    reused = load_reused()
    previous = json.loads(EXCLUSION_MANIFEST.read_text(encoding="utf-8"))
    exclusion_paths = {
        name: Path(info["path"])
        for name, info in previous["exclusions"].items()
    }
    exclusion_details, all_exact, all_norm, all_ids = reused.load_exclusions(exclusion_paths)
    pools = reused.load_raw_candidates(all_exact, all_norm, all_ids)

    final_rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        rng = random.Random(f"{SEED}|{domain}|final700")
        selected = rng.sample(pools[domain], PER_DOMAIN)
        for index, candidate in enumerate(selected, 1):
            final_rows.append({
                "sample_id": f"dpo900_final700_{domain}_{index:03d}",
                "sampling_seed": SEED,
                "selection_rule": "deterministic random sample from sorted unused same-domain canonical raw candidates with 200 <= len(source_text) <= 700",
                **candidate,
            })

    domain_counts = dict(Counter(row["domain"] for row in final_rows))
    expected_counts = {domain: PER_DOMAIN for domain in DOMAINS}
    if domain_counts != expected_counts or len(final_rows) != 51:
        raise RuntimeError(f"count mismatch: {domain_counts}, rows={len(final_rows)}")

    lengths = [len(reused.source_text(row)) for row in final_rows]
    if max(lengths) > MAX_CHARS:
        raise RuntimeError(f"length eligibility failed: max={max(lengths)}")
    exact_texts = [reused.source_text(row) for row in final_rows]
    norm_texts = [reused.normalize_text(text) for text in exact_texts]
    if len(set(exact_texts)) != 51 or len(set(norm_texts)) != 51:
        raise RuntimeError("duplicate exact or normalized source text in final700")

    overlap_checks = {
        name: reused.overlap_counts(final_rows, path)
        for name, path in exclusion_paths.items()
    }
    if any(value != 0 for check in overlap_checks.values() for value in check.values()):
        raise RuntimeError(f"exclusion overlap detected: {overlap_checks}")

    reused.write_jsonl(OUTPUT, final_rows)
    output_sha = reused.file_sha256(OUTPUT)
    manifest = {
        "output_path": str(OUTPUT),
        "output_sha256": output_sha,
        "selection_seed": SEED,
        "eligibility_rule": "Python len(source_text), including whitespace/newlines, must satisfy 200 <= length <= 700; canonical raw eligibility rules unchanged",
        "selection_rule": "Within each domain, uniformly sample 17 rows with a domain-specific deterministic RNG from eligible candidates sorted by original_sample_id/source_hash.",
        "selection_uses_model_outputs_or_metrics": False,
        "reused_candidate_loader": str(REUSED_SCRIPT),
        "reused_exclusion_manifest": str(EXCLUSION_MANIFEST),
        "exclusion_sources": exclusion_details,
        "eligible_candidate_pool_counts": {domain: len(rows) for domain, rows in pools.items()},
        "domain_counts": domain_counts,
        "char_length": {
            "mean": mean(lengths),
            "median": median(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "overlap_checks": overlap_checks,
        "final_unique_exact_texts": len(set(exact_texts)),
        "final_unique_normalized_texts": len(set(norm_texts)),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
