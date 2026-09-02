#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "work/hcx_dpo900_mode_jury_v1"
RAW = PROJECT / "data/raw"
SOURCE = ROOT / "data/splits/final_heldout_source_51.jsonl"
OUTPUT = ROOT / "data/splits/final_heldout_source_51_shortened.jsonl"
MANIFEST = ROOT / "data/splits/final_heldout_source_51_shortened_manifest.json"
SHA_FILE = ROOT / "data/splits/final_heldout_source_51_shortened.sha256"
SEED = 20260829
DOMAINS = ("news", "petition", "sns")
MIN_CHARS = 200
MAX_CHARS = 1000

REPLACEMENTS = {
    "news": [
        "dpo900_heldout_news_017",
        "dpo900_heldout_news_004",
        "dpo900_heldout_news_003",
        "dpo900_heldout_news_002",
    ],
    "petition": [
        "dpo900_heldout_petition_005",
        "dpo900_heldout_petition_001",
        "dpo900_heldout_petition_011",
        "dpo900_heldout_petition_017",
    ],
}

BASE_EXCLUSIONS = {
    "dpo_construction_900": ROOT / "data/splits/dpo_train_source_900.jsonl",
    "judge_pilot_28": PROJECT / "work/hcx_v3_direct_expr_jury28_v1/data/jury_input_pairs.jsonl",
    "judge_validation_120": PROJECT / "work/hcx_v3_direct_expr_jury120_validation_v1/data/validation_pairs_120.jsonl",
    "old_heldout_60": PROJECT / "work/hcx_final_heldout60_v1/data/heldout/final_heldout_source_manifest_60.jsonl",
    "current_heldout_51": SOURCE,
    "rewrite_prompt_v2_pilot_30": PROJECT / "work/hcx_rewrite_prompt_v2_pilot30/data/pilot_source_manifest_30.jsonl",
    "rewrite_prompt_v3_pilot_30": PROJECT / "work/hcx_rewrite_prompt_v3_mode_margin_pilot30_v1/data/pilot_source_manifest_30.jsonl",
    "v3_final_method_150": PROJECT / "work/hcx_v3_expansion150_v1/data/final150_source_manifest.jsonl",
}
FINAL_REGISTRY = PROJECT / "work/hcx_final_heldout60_v1/data/exclusion/final_exclusion_registry.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def source_text(row: dict[str, Any]) -> str:
    return str(row.get("source_text") or row.get("text") or "")


def source_id(row: dict[str, Any]) -> str:
    for key in ("source_id", "original_sample_id", "pilot_sample_id", "sample_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return hashlib.sha256(normalize_text(source_text(row)).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusion_paths() -> dict[str, Path]:
    paths = dict(BASE_EXCLUSIONS)
    registry = json.loads(FINAL_REGISTRY.read_text(encoding="utf-8"))
    for index, item in enumerate(registry.get("canonical_inputs", []), 1):
        paths[f"historical_final_registry_{index}"] = Path(item["path"])
    return paths


def load_exclusions(paths: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], set[str], set[str], set[tuple[str, str]]]:
    details: dict[str, dict[str, Any]] = {}
    all_exact: set[str] = set()
    all_norm: set[str] = set()
    all_ids: set[tuple[str, str]] = set()
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing exclusion file: {path}")
        rows = read_jsonl(path)
        exact = {source_text(row) for row in rows if source_text(row)}
        norm = {normalize_text(text) for text in exact}
        ids = {(str(row.get("domain", "")), source_id(row)) for row in rows}
        details[name] = {
            "path": str(path),
            "rows": len(rows),
            "unique_exact_texts": len(exact),
            "unique_normalized_texts": len(norm),
        }
        all_exact.update(exact)
        all_norm.update(norm)
        all_ids.update(ids)
    return details, all_exact, all_norm, all_ids


def load_raw_candidates(all_exact: set[str], all_norm: set[str], all_ids: set[tuple[str, str]]) -> dict[str, list[dict[str, Any]]]:
    sys.path.insert(0, str(PROJECT / "scripts"))
    from prepare_experiment_data_100 import eligible_iterator
    from prepare_pilot_data import iter_news, iter_petition, iter_sns

    iterators = {"news": iter_news, "petition": iter_petition, "sns": iter_sns}
    candidates: dict[str, dict[str, dict[str, Any]]] = {domain: {} for domain in DOMAINS}
    for domain in DOMAINS:
        stats: dict[str, Any] = {"raw_seen": 0, "excluded_pilot_overlap": 0, "splittable_seen": 0, "filter_reasons": {}}
        for item in eligible_iterator(iterators[domain](RAW), domain, set(), 3, 3, stats):
            text = source_text(item)
            char_len = len(text)
            if char_len < MIN_CHARS or char_len > MAX_CHARS:
                continue
            norm = normalize_text(text)
            sid = source_id(item)
            if text in all_exact or norm in all_norm or (domain, sid) in all_ids:
                continue
            h = text_hash(text)
            candidates[domain].setdefault(h, {
                "domain": domain,
                "original_sample_id": sid,
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "category": item.get("category"),
                "source_dataset": item.get("source_dataset"),
                "source_file": item.get("source_file"),
                "source_text": text,
                "source_hash": h,
                "num_chars": char_len,
                "num_sentences": len(item.get("_precomputed_sentences") or []),
                "raw_provenance": item.get("source_file"),
            })
    return {
        domain: sorted(rows.values(), key=lambda row: (row["original_sample_id"], row["source_hash"]))
        for domain, rows in candidates.items()
    }


def overlap_counts(rows: list[dict[str, Any]], path: Path) -> dict[str, int]:
    external = read_jsonl(path)
    row_exact = {source_text(row) for row in rows}
    row_norm = {normalize_text(text) for text in row_exact}
    ext_exact = {source_text(row) for row in external if source_text(row)}
    ext_norm = {normalize_text(text) for text in ext_exact}
    return {
        "exact_text_overlap": len(row_exact & ext_exact),
        "normalized_text_overlap": len(row_norm & ext_norm),
    }


def main() -> None:
    if OUTPUT.exists() or MANIFEST.exists() or SHA_FILE.exists():
        raise FileExistsError("shortened held-out outputs already exist; refusing to overwrite")

    original = read_jsonl(SOURCE)
    target_ids = {sample_id for ids in REPLACEMENTS.values() for sample_id in ids}
    found = {row["sample_id"] for row in original if row["sample_id"] in target_ids}
    if found != target_ids:
        raise RuntimeError(f"replacement target mismatch: missing={sorted(target_ids - found)}")
    if any(len(source_text(row)) <= MAX_CHARS for row in original if row["sample_id"] in target_ids):
        raise RuntimeError("a requested replacement target is not longer than 1000 characters")

    paths = exclusion_paths()
    exclusion_details, all_exact, all_norm, all_ids = load_exclusions(paths)
    pools = load_raw_candidates(all_exact, all_norm, all_ids)

    replacement_map: dict[str, dict[str, Any]] = {}
    replacement_records: list[dict[str, Any]] = []
    for domain, old_ids in REPLACEMENTS.items():
        rng = random.Random(f"{SEED}|{domain}|shortened_replacement")
        selected = rng.sample(pools[domain], len(old_ids))
        for index, (old_id, candidate) in enumerate(zip(old_ids, selected, strict=True), 1):
            new_row = {
                "sample_id": f"dpo900_heldout_{domain}_short_{index:03d}",
                "sampling_seed": SEED,
                "replacement_for": old_id,
                "replacement_selection_rule": "uniform random sample from sorted unused same-domain canonical raw candidates with 200 <= char_len <= 1000",
                **candidate,
            }
            replacement_map[old_id] = new_row
            old_row = next(row for row in original if row["sample_id"] == old_id)
            replacement_records.append({
                "old_sample_id": old_id,
                "old_domain": domain,
                "old_char_len": len(source_text(old_row)),
                "new_sample_id": new_row["sample_id"],
                "new_original_sample_id": new_row["original_sample_id"],
                "new_char_len": len(new_row["source_text"]),
                "new_source_hash": new_row["source_hash"],
            })

    final_rows = [replacement_map.get(row["sample_id"], row) for row in original]
    retained = [row for row in original if row["sample_id"] not in target_ids]
    retained_ids = {row["sample_id"] for row in retained}
    final_retained = [row for row in final_rows if row["sample_id"] in retained_ids]
    if retained != final_retained:
        raise RuntimeError("the 43 retained rows changed or moved")
    if len(final_rows) != 51 or len(retained) != 43:
        raise RuntimeError("final/retained row count mismatch")
    domain_counts = dict(Counter(row["domain"] for row in final_rows))
    if domain_counts != {"news": 17, "petition": 17, "sns": 17}:
        raise RuntimeError(f"domain count mismatch: {domain_counts}")
    lengths = [len(source_text(row)) for row in final_rows]
    if max(lengths) > MAX_CHARS:
        raise RuntimeError(f"final max char length exceeds {MAX_CHARS}: {max(lengths)}")
    exact_texts = [source_text(row) for row in final_rows]
    norm_texts = [normalize_text(text) for text in exact_texts]
    if len(set(exact_texts)) != 51 or len(set(norm_texts)) != 51:
        raise RuntimeError("duplicate exact or normalized text in final held-out")

    external_overlap_checks = {
        name: overlap_counts(final_rows, path)
        for name, path in paths.items()
        if name != "current_heldout_51"
    }
    if any(value != 0 for check in external_overlap_checks.values() for value in check.values()):
        raise RuntimeError(f"external overlap detected: {external_overlap_checks}")
    replacement_rows = list(replacement_map.values())
    replacement_vs_old = overlap_counts(replacement_rows, SOURCE)
    if any(replacement_vs_old.values()):
        raise RuntimeError(f"replacement overlap with old held-out: {replacement_vs_old}")

    write_jsonl(OUTPUT, final_rows)
    output_sha = file_sha256(OUTPUT)
    SHA_FILE.write_text(f"{output_sha}  {OUTPUT.name}\n", encoding="utf-8")
    manifest = {
        "source_path": str(SOURCE),
        "source_sha256": file_sha256(SOURCE),
        "output_path": str(OUTPUT),
        "output_sha256": output_sha,
        "selection_seed": SEED,
        "selection_uses_model_outputs_or_metrics": False,
        "selection_rule": "For each replacement domain, uniformly sample four rows with a domain-specific deterministic RNG from canonical raw candidates sorted by original_sample_id/source_hash, after all exclusions and 200 <= source character length <= 1000.",
        "character_length_definition": "Python len(source_text), including whitespace and newlines",
        "original_rows": 51,
        "retained_rows_unchanged": 43,
        "replacement_rows": 8,
        "domain_counts": domain_counts,
        "char_length": {
            "mean": mean(lengths),
            "median": median(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "eligible_candidate_pool_counts": {domain: len(rows) for domain, rows in pools.items()},
        "replacements": replacement_records,
        "exclusions": exclusion_details,
        "overlap_checks_final_vs_external": external_overlap_checks,
        "overlap_checks_replacements_vs_old_heldout": replacement_vs_old,
        "final_unique_exact_texts": len(set(exact_texts)),
        "final_unique_normalized_texts": len(set(norm_texts)),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
