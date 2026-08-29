#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT = Path("/home/user20250805/gayeon/HCLT2026")
ROOT = PROJECT / "work/hcx_dpo900_mode_jury_v1"
RAW = PROJECT / "data/raw"
SEED = 20260828
HELDOUT_SEED = 20260828
TRAIN_SEED = 20260829
DOMAINS = ("news", "petition", "sns")
MIN_CHARS = 200
MIN_SENTENCES = 3
MIN_FINAL_PUNCT = 3
EXPECTED_V3_PROMPT_SHA = "2dc0e8c95264168dca7d19568cf9c6fb7e0cc273994bcce6a592e7493459f8e5"

DEV28 = PROJECT / "work/hcx_v3_direct_expr_jury28_v1/data/jury_input_pairs.jsonl"
VAL120 = PROJECT / "work/hcx_v3_direct_expr_jury120_validation_v1/data/validation_pairs_120.jsonl"
OLD_HELDOUT60 = PROJECT / "work/hcx_final_heldout60_v1/data/heldout/final_heldout_source_manifest_60.jsonl"

V3_ROOT = PROJECT / "work/hcx_v3_expansion150_v1"
V3_GEN = PROJECT / "work/hcx_rewrite_prompt_v3_mode_margin_pilot30_v1/scripts/run_v3_generation.py"
BASE_GEN = PROJECT / "work/hcx_pairwise_expand300_v1/scripts/run_expand300_rewrite_generation.py"
GEN_CONFIG = PROJECT / "work/hcx_pairwise_expand300_v1/configs/hcx_generation_config.json"
GEN_PROMPT_SRC = V3_ROOT / "prompts/document_rewrite_instruction_v3_mode_controlled.txt"

QWEN_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_v1"
GRANITE_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_granite_command_v1"
MINISTRAL_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_command_ministral_v3"
QWEN_SCRIPT = QWEN_ROOT / "scripts/run_direct_expr_jury.py"
GRANITE_SCRIPT = GRANITE_ROOT / "scripts/run_granite_command_jury28.py"
MINISTRAL_SCRIPT = MINISTRAL_ROOT / "scripts/run_command_ministral_v3.py"

TRAIN = ROOT / "data/splits/dpo_train_source_900.jsonl"
HELDOUT = ROOT / "data/splits/final_heldout_source_51.jsonl"
EXCLUSION = ROOT / "data/splits/exclusion_manifest.json"
SPLIT_REPORT = ROOT / "data/splits/split_report.md"
PAIR_INPUT = ROOT / "outputs/generation/complete_ct_pairs.jsonl"
ASSIGNMENT = ROOT / "data/jury/candidate_assignment.jsonl"
CONSERVATIVE = ROOT / "outputs/generation/conservative.jsonl"
TRANSFORMATIVE = ROOT / "outputs/generation/transformative.jsonl"
GEN_RAW = ROOT / "outputs/generation/raw_attempts.jsonl"
STABLE = ROOT / "outputs/jury/aggregation/stable_vote_results.jsonl"
AGG = ROOT / "outputs/jury/aggregation/aggregation_results.jsonl"
PREF_JSONL = ROOT / "outputs/preferences/dpo_preferences_all_survivors.jsonl"
PREF_CSV = ROOT / "outputs/preferences/dpo_preferences_all_survivors.csv"
PREF_MANIFEST = ROOT / "outputs/preferences/dpo_dataset_manifest.json"
FINAL_REPORT = ROOT / "reports/final_pipeline_report.md"
SETUP = ROOT / "reports/setup_and_runtime.json"

PROMPT = ROOT / "prompts/generation_prompt_reference.txt"
QWEN_PROMPT = ROOT / "prompts/qwen_frozen_prompt_reference.txt"
QWEN_USER = ROOT / "prompts/qwen_frozen_user_template.txt"
GRANITE_PROMPT = ROOT / "prompts/granite_frozen_prompt_reference.txt"
DIRECT_USER = ROOT / "prompts/direct_expressive_user_template.txt"
MINISTRAL_PROMPT = ROOT / "prompts/ministral_frozen_prompt_reference.txt"
JUDGES = ("qwen3", "granite", "ministral3")
ORDERS = ("original", "swapped")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def source_id(row: dict[str, Any]) -> str:
    for key in ("source_id", "original_sample_id", "sample_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return text_hash(row.get("source_text") or row.get("text") or "")


def source_text(row: dict[str, Any]) -> str:
    return str(row.get("source_text") or row.get("text") or "")


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    for rel in (
        "data/splits", "data/jury", "prompts", "outputs/generation",
        "outputs/jury/raw", "outputs/jury/aggregation", "outputs/preferences",
        "reports", "logs", "configs", "outputs/raw",
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def copy_frozen_prompts() -> None:
    shutil.copyfile(GEN_PROMPT_SRC, PROMPT)
    shutil.copyfile(QWEN_ROOT / "prompts/short_direct_pairwise_expressive_independence_system_prompt.txt", QWEN_PROMPT)
    shutil.copyfile(QWEN_ROOT / "prompts/short_direct_pairwise_expressive_independence_user_template.txt", QWEN_USER)
    shutil.copyfile(GRANITE_ROOT / "prompts/granite_direct_expressive_v1.txt", GRANITE_PROMPT)
    shutil.copyfile(GRANITE_ROOT / "prompts/direct_expressive_user_template.txt", DIRECT_USER)
    shutil.copyfile(MINISTRAL_ROOT / "prompts/ministral_positive_hierarchical_expressive_v3.txt", MINISTRAL_PROMPT)
    if sha256_file(PROMPT) != EXPECTED_V3_PROMPT_SHA:
        raise RuntimeError("v3 generation prompt SHA mismatch")


def load_raw_candidates() -> list[dict[str, Any]]:
    sys.path.insert(0, str(PROJECT / "scripts"))
    from prepare_pilot_data import iter_news, iter_petition, iter_sns  # noqa: PLC0415
    from prepare_experiment_data_100 import eligible_iterator  # noqa: PLC0415

    iterators = {"news": iter_news, "petition": iter_petition, "sns": iter_sns}
    excluded_keys: set[tuple[str, str | None]] = set()
    stats: dict[str, Any] = {}
    raw: list[dict[str, Any]] = []
    for domain in DOMAINS:
        domain_stats = {"raw_seen": 0, "excluded_pilot_overlap": 0, "splittable_seen": 0, "filter_reasons": {}}
        for item in eligible_iterator(iterators[domain](RAW), domain, excluded_keys, MIN_SENTENCES, MIN_FINAL_PUNCT, domain_stats):
            text = source_text(item)
            if len(text) < MIN_CHARS:
                continue
            sid = source_id(item)
            raw.append({
                "domain": domain,
                "original_sample_id": sid,
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "category": item.get("category"),
                "source_dataset": item.get("source_dataset"),
                "source_file": item.get("source_file"),
                "source_text": text,
                "source_hash": text_hash(text),
                "num_chars": len(text),
                "num_sentences": len(item.get("_precomputed_sentences") or []),
                "raw_provenance": item.get("source_file"),
            })
        stats[domain] = domain_stats
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw:
        unique.setdefault((row["domain"], row["source_hash"]), row)
    write_json(ROOT / "data/splits/raw_load_stats.json", {"raw_rows": len(raw), "unique_rows": len(unique), "domain_stats": stats})
    return list(unique.values())


def exclusion_sets() -> dict[str, Any]:
    paths = {"dev28": DEV28, "validation120": VAL120, "oldheldout60": OLD_HELDOUT60}
    rows_by_name = {name: read_jsonl(path) for name, path in paths.items()}
    hashes = {name: {text_hash(source_text(row)) for row in rows if source_text(row)} for name, rows in rows_by_name.items()}
    ids = {name: {(str(row.get("domain")), source_id(row)) for row in rows} for name, rows in rows_by_name.items()}
    return {"paths": {k: str(v) for k, v in paths.items()}, "rows": {k: len(v) for k, v in rows_by_name.items()}, "hashes": hashes, "ids": ids}


def split() -> None:
    ensure_dirs()
    copy_frozen_prompts()
    raw = load_raw_candidates()
    exc = exclusion_sets()
    all_excluded_hashes = set().union(*exc["hashes"].values())
    all_excluded_ids = set().union(*exc["ids"].values())
    eligible = [row for row in raw if row["source_hash"] not in all_excluded_hashes and (row["domain"], row["original_sample_id"]) not in all_excluded_ids]
    heldout_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        domain_rows = sorted([row for row in eligible if row["domain"] == domain], key=lambda r: (r["original_sample_id"], r["source_hash"]))
        if len(domain_rows) < 317:
            raise RuntimeError(f"insufficient eligible {domain}: {len(domain_rows)}")
        h_rng = random.Random(f"{HELDOUT_SEED}|{domain}|heldout")
        t_rng = random.Random(f"{TRAIN_SEED}|{domain}|train")
        heldout_idx = set(h_rng.sample(range(len(domain_rows)), 17))
        domain_heldout = [domain_rows[i] for i in sorted(heldout_idx)]
        remaining = [row for i, row in enumerate(domain_rows) if i not in heldout_idx]
        domain_train = t_rng.sample(remaining, 300)
        for i, row in enumerate(domain_heldout, 1):
            heldout_rows.append({"sample_id": f"dpo900_heldout_{domain}_{i:03d}", "sampling_seed": HELDOUT_SEED, **row})
        for i, row in enumerate(domain_train, 1):
            train_rows.append({"sample_id": f"dpo900_train_{domain}_{i:03d}", "sampling_seed": TRAIN_SEED, **row})

    train_hashes = {row["source_hash"] for row in train_rows}
    heldout_hashes = {row["source_hash"] for row in heldout_rows}
    if len(train_rows) != 900 or len(heldout_rows) != 51 or train_hashes & heldout_hashes:
        raise RuntimeError("split count or train/heldout overlap failed")
    checks = {
        "train_heldout_hash_overlap": len(train_hashes & heldout_hashes),
        "train_dev28_hash_overlap": len(train_hashes & exc["hashes"]["dev28"]),
        "train_validation120_hash_overlap": len(train_hashes & exc["hashes"]["validation120"]),
        "train_oldheldout60_hash_overlap": len(train_hashes & exc["hashes"]["oldheldout60"]),
        "heldout_dev28_hash_overlap": len(heldout_hashes & exc["hashes"]["dev28"]),
        "heldout_validation120_hash_overlap": len(heldout_hashes & exc["hashes"]["validation120"]),
        "heldout_oldheldout60_hash_overlap": len(heldout_hashes & exc["hashes"]["oldheldout60"]),
    }
    if any(checks.values()):
        raise RuntimeError(f"leakage checks failed: {checks}")
    write_jsonl(TRAIN, sorted(train_rows, key=lambda r: r["sample_id"]))
    write_jsonl(HELDOUT, sorted(heldout_rows, key=lambda r: r["sample_id"]))
    write_json(EXCLUSION, {
        "seed": SEED,
        "excluded_paths": exc["paths"],
        "excluded_rows": exc["rows"],
        "checks": checks,
        "raw_unique_domain_counts": dict(Counter(row["domain"] for row in raw)),
        "eligible_domain_counts": dict(Counter(row["domain"] for row in eligible)),
        "train_domain_counts": dict(Counter(row["domain"] for row in train_rows)),
        "heldout_domain_counts": dict(Counter(row["domain"] for row in heldout_rows)),
        "train_sha256": sha256_file(TRAIN),
        "heldout_sha256": sha256_file(HELDOUT),
    })
    (TRAIN.with_suffix(".sha256")).write_text(f"{sha256_file(TRAIN)}  {TRAIN.name}\n", encoding="utf-8")
    (HELDOUT.with_suffix(".sha256")).write_text(f"{sha256_file(HELDOUT)}  {HELDOUT.name}\n", encoding="utf-8")
    SPLIT_REPORT.write_text(
        "\n".join([
            "# DPO900 Split Report", "",
            f"- train: 900 {dict(Counter(row['domain'] for row in train_rows))}",
            f"- heldout: 51 {dict(Counter(row['domain'] for row in heldout_rows))}",
            f"- leakage checks: {checks}",
            f"- generation prompt SHA256: {sha256_file(PROMPT)}",
            f"- train SHA256: {sha256_file(TRAIN)}",
            f"- heldout SHA256: {sha256_file(HELDOUT)}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"SPLIT_COMPLETE train=900 heldout=51 checks={checks}", flush=True)


def split_prompt(text: str) -> tuple[str, str]:
    marker = "--------------------------------------------------\nUSER MESSAGE TEMPLATE\n--------------------------------------------------"
    if text.count(marker) != 1:
        raise RuntimeError("mode prompt marker mismatch")
    system, template = text.split(marker, 1)
    return system.rstrip(), template.strip()


def generation_final(row: dict[str, Any], mode: str, result: dict[str, Any], prompt_sha: str) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "original_sample_id": row["original_sample_id"],
        "domain": row["domain"],
        "source_text": row["source_text"],
        "source_hash": row["source_hash"],
        "mode": mode,
        "rewrite_text": result["rewrite_text"] if result.get("valid") else None,
        "seed": result.get("seed"),
        "attempt": result.get("attempt"),
        "valid": bool(result.get("valid")),
        "error": None if result.get("valid") else result.get("errors"),
        "latency_sec": result.get("latency_sec"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "model_identifier": "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
        "model_path": str(PROJECT / "models/HyperCLOVAX-SEED-Think-14B"),
        "prompt_sha256": prompt_sha,
    }


def generate() -> None:
    v3 = import_module("dpo900_v3_generation", V3_GEN)
    base = v3.load_base_module()
    config = json.loads(GEN_CONFIG.read_text(encoding="utf-8"))
    prompt_sha = sha256_file(PROMPT)
    system, template = split_prompt(PROMPT.read_text(encoding="utf-8"))
    rows = read_jsonl(TRAIN)
    if len(rows) != 900:
        raise RuntimeError("train split missing")
    existing_c = {row["sample_id"] for row in read_jsonl(CONSERVATIVE) if row.get("valid") and row.get("rewrite_text")}
    existing_t = {row["sample_id"] for row in read_jsonl(TRANSFORMATIVE) if row.get("valid") and row.get("rewrite_text")}
    print(f"GENERATION_START sources=900 existing_c={len(existing_c)} existing_t={len(existing_t)}", flush=True)
    model, tokenizer, revision = base.load_hcx()
    print(f"MODEL_LOADED model=HyperCLOVAX revision={revision} adapter=false", flush=True)
    v3.RAW = GEN_RAW
    for idx, row in enumerate(rows, 1):
        adapted = {"pilot_sample_id": row["sample_id"], **row}
        if row["sample_id"] not in existing_c:
            result = v3.generate(base, model, tokenizer, adapted, "CONSERVATIVE", system, template, config)
            append_jsonl(CONSERVATIVE, generation_final(row, "CONSERVATIVE", result, prompt_sha))
            print(f"GEN_PROGRESS source={idx}/900 mode=CONSERVATIVE valid={result.get('valid')}", flush=True)
        if row["sample_id"] not in existing_t:
            result = v3.generate(base, model, tokenizer, adapted, "TRANSFORMATIVE", system, template, config)
            append_jsonl(TRANSFORMATIVE, generation_final(row, "TRANSFORMATIVE", result, prompt_sha))
            print(f"GEN_PROGRESS source={idx}/900 mode=TRANSFORMATIVE valid={result.get('valid')}", flush=True)
    del model, tokenizer
    gc.collect()
    print("GENERATION_COMPLETE", flush=True)


def validate_generation() -> None:
    c = {row["sample_id"]: row for row in read_jsonl(CONSERVATIVE) if row.get("valid") and row.get("rewrite_text")}
    t = {row["sample_id"]: row for row in read_jsonl(TRANSFORMATIVE) if row.get("valid") and row.get("rewrite_text")}
    train = {row["sample_id"]: row for row in read_jsonl(TRAIN)}
    complete_ids = sorted(set(c) & set(t))
    pairs = []
    assignments = []
    for i, sid in enumerate(complete_ids):
        a = "CONSERVATIVE" if i % 2 == 0 else "TRANSFORMATIVE"
        b = "TRANSFORMATIVE" if a == "CONSERVATIVE" else "CONSERVATIVE"
        row = train[sid]
        pairs.append({
            "sample_id": sid,
            "original_sample_id": row["original_sample_id"],
            "domain": row["domain"],
            "source_text": row["source_text"],
            "conservative_text": c[sid]["rewrite_text"],
            "transformative_text": t[sid]["rewrite_text"],
            "original_A_identity": a,
            "original_B_identity": b,
        })
        assignments.append({
            "sample_id": sid,
            "original_A_identity": a,
            "original_B_identity": b,
            "swapped_A_identity": b,
            "swapped_B_identity": a,
        })
    write_jsonl(PAIR_INPUT, pairs)
    write_jsonl(ASSIGNMENT, assignments)
    print(f"GEN_VALIDATION_COMPLETE complete_pairs={len(pairs)} incomplete={900-len(pairs)}", flush=True)


def configure_qwen():
    qwen = import_module("dpo900_qwen", QWEN_SCRIPT)
    qwen.ROOT = ROOT
    qwen.PAIR_INPUT = PAIR_INPUT
    qwen.ASSIGNMENT = ASSIGNMENT
    qwen.SYSTEM_PROMPT = QWEN_PROMPT
    qwen.USER_TEMPLATE = QWEN_USER
    qwen.JUDGES = ("qwen3",)
    return qwen


def run_qwen() -> None:
    qwen = configure_qwen()
    qwen.run_judge("qwen3")


def run_granite() -> None:
    granite = import_module("dpo900_granite", GRANITE_SCRIPT)
    granite.ROOT = ROOT
    granite.PAIR_INPUT = PAIR_INPUT
    granite.ASSIGNMENT = ASSIGNMENT
    granite.USER_TEMPLATE = DIRECT_USER
    granite.PROMPTS = {"granite": GRANITE_PROMPT}
    granite.PROMPT_VERSIONS = {"granite": "granite_frozen_direct_v1"}
    granite.JUDGES = ("granite",)
    granite.SPECS = {"granite": granite.SPECS["granite"]}
    granite.run_judge("granite")


def configure_ministral():
    ministral = import_module("dpo900_ministral", MINISTRAL_SCRIPT)
    prompts = {"ministral3": MINISTRAL_PROMPT}
    versions = {"ministral3": "ministral_positive_hierarchical_expressive_v3_frozen"}
    specs = {"ministral3": ministral.SPECS["ministral3"]}
    ministral.ROOT = ROOT
    ministral.PAIR_INPUT = PAIR_INPUT
    ministral.ASSIGNMENT = ASSIGNMENT
    ministral.USER_TEMPLATE = DIRECT_USER
    ministral.PROMPTS = prompts
    ministral.PROMPT_VERSIONS = versions
    ministral.JUDGES = ("ministral3",)
    ministral.SPECS = specs
    ministral.common.ROOT = ROOT
    ministral.common.PAIR_INPUT = PAIR_INPUT
    ministral.common.ASSIGNMENT = ASSIGNMENT
    ministral.common.USER_TEMPLATE = DIRECT_USER
    ministral.common.PROMPTS = prompts
    ministral.common.PROMPT_VERSIONS = versions
    ministral.common.JUDGES = ("ministral3",)
    ministral.common.SPECS = specs
    return ministral


def run_ministral() -> None:
    ministral = configure_ministral()
    ministral.common.run_judge("ministral3")


def latest_valid(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "valid":
            out[(row["sample_id"], row["presentation"])] = row
    return out


def aggregate() -> None:
    pairs = read_jsonl(PAIR_INPUT)
    raw = {judge: latest_valid(ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    stable_rows = []
    agg_rows = []
    prefs = []
    reason_counts: Counter[str] = Counter()
    for pair in pairs:
        votes: list[str] = []
        row: dict[str, Any] = {"sample_id": pair["sample_id"], "domain": pair["domain"]}
        for judge in JUDGES:
            original = raw[judge].get((pair["sample_id"], "original"), {})
            swapped = raw[judge].get((pair["sample_id"], "swapped"), {})
            vote_o = original.get("mapped_candidate")
            vote_s = swapped.get("mapped_candidate")
            stable = bool(vote_o and vote_o == vote_s)
            vote = vote_o if stable else None
            row[f"{judge}_stable"] = stable
            row[f"{judge}_vote"] = vote
            row[f"{judge}_original"] = vote_o
            row[f"{judge}_swapped"] = vote_s
            if vote:
                votes.append(vote)
        counts = Counter(votes)
        winner, winner_votes = (counts.most_common(1)[0] if counts else (None, 0))
        stable_vote_count = len(votes)
        stable_conflict = len(counts) > 1
        accepted = bool(winner and winner_votes >= 2)
        if not accepted:
            reason = "insufficient_stable_votes" if stable_vote_count < 2 else "stable_conflict"
            reason_counts[reason] += 1
        else:
            reason = None
        agg = row | {
            "stable_votes": votes,
            "stable_vote_count": stable_vote_count,
            "winner": winner,
            "winner_vote_count": winner_votes,
            "accepted": accepted,
            "reject_reason": reason,
            "aggregation_pattern": "/".join(votes) if votes else "none",
        }
        stable_rows.append(row)
        agg_rows.append(agg)
        if accepted:
            chosen_identity = winner
            rejected_identity = "CONSERVATIVE" if winner == "TRANSFORMATIVE" else "TRANSFORMATIVE"
            prefs.append({
                "sample_id": pair["sample_id"],
                "domain": pair["domain"],
                "source": pair["source_text"],
                "prompt": pair["source_text"],
                "chosen": pair["transformative_text"] if chosen_identity == "TRANSFORMATIVE" else pair["conservative_text"],
                "rejected": pair["conservative_text"] if chosen_identity == "TRANSFORMATIVE" else pair["transformative_text"],
                "chosen_identity": chosen_identity,
                "rejected_identity": rejected_identity,
                "qwen_stable": agg["qwen3_stable"],
                "qwen_vote": agg["qwen3_vote"],
                "granite_stable": agg["granite_stable"],
                "granite_vote": agg["granite_vote"],
                "ministral_stable": agg["ministral3_stable"],
                "ministral_vote": agg["ministral3_vote"],
                "stable_vote_count": stable_vote_count,
                "winner_vote_count": winner_votes,
                "aggregation_pattern": agg["aggregation_pattern"],
            })
    write_jsonl(STABLE, stable_rows)
    write_jsonl(AGG, agg_rows)
    write_jsonl(PREF_JSONL, prefs)
    if prefs:
        with PREF_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in prefs for k in row}))
            writer.writeheader()
            writer.writerows(prefs)
    raw_counts = {judge: len(read_jsonl(ROOT / f"outputs/raw/{judge}.jsonl")) for judge in JUDGES}
    valid_counts = {judge: len(raw[judge]) for judge in JUDGES}
    manifest = {
        "initial_source_n": 900,
        "generation_complete_n": len(pairs),
        "jury_expected": len(pairs) * 3 * 2,
        "jury_raw_rows": raw_counts,
        "jury_valid_keys": valid_counts,
        "aggregation_survivor_n": len(prefs),
        "survivor_rate": len(prefs) / len(pairs) if pairs else 0,
        "domain_counts": dict(Counter(row["domain"] for row in prefs)),
        "chosen_identity_counts": dict(Counter(row["chosen_identity"] for row in prefs)),
        "rejection_reason_counts": dict(reason_counts),
        "aggregation_rule": "accept iff >=2 stable judges support the same actual candidate; original/swapped are not counted as separate votes",
        "gpt_openai_calls": 0,
    }
    write_json(PREF_MANIFEST, manifest)
    FINAL_REPORT.write_text(
        "\n".join([
            "# DPO900 Mode Jury Pipeline Report", "",
            f"- train sources: 900",
            f"- heldout sources frozen, unused after split: 51",
            f"- complete C/T pairs: {len(pairs)}",
            f"- expected jury judgments: {len(pairs) * 6}",
            f"- valid jury keys: {valid_counts}",
            f"- DPO survivors: {len(prefs)}",
            f"- survivor rate: {manifest['survivor_rate']:.4f}",
            f"- chosen identity counts: {manifest['chosen_identity_counts']}",
            f"- domain counts: {manifest['domain_counts']}",
            f"- reject reasons: {manifest['rejection_reason_counts']}",
            "- GPT/OpenAI calls: 0",
        ]) + "\n",
        encoding="utf-8",
    )
    for judge in JUDGES:
        src = ROOT / f"outputs/raw/{judge}.jsonl"
        dst = ROOT / f"outputs/jury/raw/{judge}.jsonl"
        if src.exists():
            shutil.copyfile(src, dst)
    print(f"AGGREGATION_COMPLETE pairs={len(pairs)} survivors={len(prefs)}", flush=True)


def preflight() -> None:
    split()
    setup = {
        "root": str(ROOT),
        "raw_root": str(RAW),
        "train_target": {"news": 300, "petition": 300, "sns": 300},
        "heldout_target": {"news": 17, "petition": 17, "sns": 17},
        "generation_expected": 1800,
        "jury_expected_max": 5400,
        "frozen_prompts": {
            "generation": {"path": str(PROMPT), "sha256": sha256_file(PROMPT)},
            "qwen": {"path": str(QWEN_PROMPT), "sha256": sha256_file(QWEN_PROMPT)},
            "qwen_user": {"path": str(QWEN_USER), "sha256": sha256_file(QWEN_USER)},
            "granite": {"path": str(GRANITE_PROMPT), "sha256": sha256_file(GRANITE_PROMPT)},
            "direct_user": {"path": str(DIRECT_USER), "sha256": sha256_file(DIRECT_USER)},
            "ministral": {"path": str(MINISTRAL_PROMPT), "sha256": sha256_file(MINISTRAL_PROMPT)},
        },
        "reused_scripts": {
            "generation_v3": str(V3_GEN),
            "generation_base": str(BASE_GEN),
            "qwen": str(QWEN_SCRIPT),
            "granite": str(GRANITE_SCRIPT),
            "ministral": str(MINISTRAL_SCRIPT),
        },
        "aggregation_rule": ">=2 stable judges same actual winner",
        "gpt_openai_calls": 0,
        "rough_eta": "HCX generation dominates; based on prior 240 rewrites about 83-90 min, 1800 rewrites roughly 10-12 hours plus jury time",
    }
    write_json(SETUP, setup)
    print("PREFLIGHT_OK train=900 heldout=51 expected_generation=1800 expected_jury_max=5400", flush=True)


def pipeline() -> None:
    if not TRAIN.exists():
        split()
    generate()
    validate_generation()
    run_qwen()
    run_granite()
    run_ministral()
    aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "split", "generate", "validate-generation", "qwen", "granite", "ministral", "aggregate", "pipeline"))
    args = parser.parse_args()
    started = time.perf_counter()
    print(f"STAGE_START {args.stage}", flush=True)
    try:
        if args.stage == "preflight":
            preflight()
        elif args.stage == "split":
            split()
        elif args.stage == "generate":
            generate()
        elif args.stage == "validate-generation":
            validate_generation()
        elif args.stage == "qwen":
            run_qwen()
        elif args.stage == "granite":
            run_granite()
        elif args.stage == "ministral":
            run_ministral()
        elif args.stage == "aggregate":
            aggregate()
        else:
            pipeline()
    finally:
        print(f"STAGE_END {args.stage} elapsed_sec={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
