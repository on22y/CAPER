#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT = Path("/home/user20250805/gayeon/HCLT2026")
WORK = PROJECT / "work/hcx_dpo501_train_v1"
INPUT = PROJECT / "work/hcx_dpo900_mode_jury_v1/outputs/preferences/dpo_preferences_all_survivors.jsonl"
CANONICAL_PREF = PROJECT / "work/hcx_dpo_expanded367_v1/data/preferences/dpo_preferences_combined_367.jsonl"
MODEL = PROJECT / "models/HyperCLOVAX-SEED-Think-14B"
TRAIN = WORK / "data/dpo/dpo_train_501_chatformatted.jsonl"
TOKEN_PREFLIGHT = WORK / "data/dpo/token_length_preflight.json"
MANIFEST = WORK / "metadata/preflight_manifest.json"
REPORT = WORK / "reports/dpo_length_audit.md"
CONFIG = WORK / "configs/lora_dpo501_config.json"
MAX_LENGTH = 3072


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    position = (len(xs) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(xs[lower])
    return xs[lower] + (xs[upper] - xs[lower]) * (position - lower)


def stats(values: list[int]) -> dict[str, Any]:
    return {
        "min": min(values) if values else None,
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values) if values else None,
    }


def canonical_prompt() -> tuple[str, str, str]:
    row = read_jsonl(CANONICAL_PREF)[0]
    parsed = json.loads(row["prompt"])
    source = row["source_text"]
    user = parsed["user"]
    start = user.index(source)
    return parsed["system"], user[:start], user[start + len(source):]


def main() -> None:
    rows = read_jsonl(INPUT)
    required = {"sample_id", "domain", "source", "chosen", "rejected", "chosen_identity"}
    if len(rows) != 501:
        raise RuntimeError(f"Expected 501 rows, found {len(rows)}")
    if len({row["sample_id"] for row in rows}) != 501:
        raise RuntimeError("Duplicate sample_id")
    if any(not required <= row.keys() for row in rows):
        raise RuntimeError("Missing required fields")
    if any(not str(row["chosen"]).strip() or not str(row["rejected"]).strip() for row in rows):
        raise RuntimeError("Empty chosen/rejected")
    if any(row["chosen"] == row["rejected"] for row in rows):
        raise RuntimeError("chosen equals rejected")
    if Counter(row["domain"] for row in rows) != Counter({"news": 140, "petition": 172, "sns": 189}):
        raise RuntimeError("Domain count mismatch")
    if Counter(row["chosen_identity"] for row in rows) != Counter({"TRANSFORMATIVE": 456, "CONSERVATIVE": 45}):
        raise RuntimeError("Chosen identity count mismatch")

    system, user_prefix, user_suffix = canonical_prompt()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lengths = {key: [] for key in ("prompt", "chosen_completion", "rejected_completion", "prompt_chosen", "prompt_rejected")}
    train_rows = []
    prompt_drops = []
    completion_truncation = []
    untruncated = []
    all_lengths = []
    for row in rows:
        user = user_prefix.replace("[DOMAIN]\nnews", f"[DOMAIN]\n{row['domain']}", 1) + row["source"] + user_suffix
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, skip_reasoning=True)
        chosen = json.dumps({"rewrite": row["chosen"]}, ensure_ascii=False)
        rejected = json.dumps({"rewrite": row["rejected"]}, ensure_ascii=False)
        prompt_n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        chosen_n = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
        rejected_n = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
        length = {
            "sample_id": row["sample_id"], "domain": row["domain"],
            "prompt": prompt_n, "chosen_completion": chosen_n, "rejected_completion": rejected_n,
            "prompt_chosen": prompt_n + chosen_n, "prompt_rejected": prompt_n + rejected_n,
        }
        all_lengths.append(length)
        for key in lengths:
            lengths[key].append(length[key])
        if prompt_n >= MAX_LENGTH:
            prompt_drops.append(length)
        elif length["prompt_chosen"] > MAX_LENGTH or length["prompt_rejected"] > MAX_LENGTH:
            completion_truncation.append(length)
        else:
            untruncated.append(length)
        train_rows.append({
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
            "sample_id": row["sample_id"], "domain": row["domain"],
            "chosen_identity": row["chosen_identity"], "preference_source": "dpo900_mode_jury_v1",
        })

    write_jsonl(TRAIN, train_rows)
    effective = len(rows) - len(prompt_drops)
    estimated_steps = math.ceil(effective / 4) * 3
    token_info = {
        "tokenizer_path": str(MODEL),
        "num_examples": len(rows),
        "method": "DPOTrainer plain tokenizer over preformatted prompt/chosen/rejected strings",
        "stats": {key: stats(values) for key, values in lengths.items()},
        "max_observed_total": max(max(lengths["prompt_chosen"]), max(lengths["prompt_rejected"])),
        "recommended_max_length_no_truncation": MAX_LENGTH,
        "effective_max_length_for_training": MAX_LENGTH,
        "truncation_mode": "keep_start",
        "canonical_input_rows": len(rows),
        "prompt_filter_passed": effective,
        "prompt_length_dropped": len(prompt_drops),
        "completion_untruncated": len(untruncated),
        "completion_partially_truncated": len(completion_truncation),
        "effective_training_rows": effective,
        "estimated_total_optimization_steps": estimated_steps,
        "prompt_length_drops": prompt_drops,
        "completion_truncated_examples": completion_truncation,
        "longest_examples": sorted(all_lengths, key=lambda row: max(row["prompt_chosen"], row["prompt_rejected"]), reverse=True)[:30],
    }
    write_json(TOKEN_PREFLIGHT, token_info)
    config = {
        "reused_training_script": str(PROJECT / "work/hcx_dpo_expanded367_v1/scripts/run_lora_dpo_expanded367.py"),
        "base_model": str(MODEL), "dtype": "bfloat16", "device_map": "auto", "use_cache": False,
        "fresh_lora_from_base": True, "old_adapter_loaded": False,
        "input_path": str(INPUT), "canonical_input_rows": 501, "effective_training_rows": effective,
        "dpo": {
            "beta": 0.1, "learning_rate": 5e-6, "num_train_epochs": 3.0,
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
            "max_length": 3072, "truncation_mode": "keep_start", "bf16": True,
            "gradient_checkpointing": True, "precompute_ref_log_probs": False,
            "remove_unused_columns": False, "dataloader_pin_memory": False,
            "optim": "adamw_torch_fused", "lr_scheduler_type": "linear", "seed": 42,
            "weight_decay": 0.0, "max_grad_norm": 1.0,
        },
        "lora": {
            "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        },
        "expected_parameter_counts": {"total": 14809126912, "trainable": 61014016},
        "estimated_total_optimization_steps": estimated_steps,
    }
    write_json(CONFIG, config)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT), "input_sha256": sha256(INPUT),
        "canonical_prompt_source": str(CANONICAL_PREF), "canonical_prompt_source_sha256": sha256(CANONICAL_PREF),
        "system_prompt_characters": len(system), "user_prefix": user_prefix, "user_suffix": user_suffix,
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "chosen_identity_counts": dict(Counter(row["chosen_identity"] for row in rows)),
        "token_length_preflight": token_info,
        "train_path": str(TRAIN), "train_sha256": sha256(TRAIN),
        "config_path": str(CONFIG),
    }
    write_json(MANIFEST, manifest)
    report = [
        "# DPO 501 Length Audit", "", f"- canonical input rows: `{len(rows)}`",
        f"- effective training rows: `{effective}`", f"- prompt-length dropped: `{len(prompt_drops)}`",
        f"- completion partially truncated: `{len(completion_truncation)}`",
        f"- estimated optimization steps: `{estimated_steps}`", "", "## Prompt-Length Drops", "",
        json.dumps(prompt_drops, ensure_ascii=False, indent=2),
    ]
    atomic_text(REPORT, "\n".join(report) + "\n")
    print(json.dumps({
        "event": "PREFLIGHT_OK", "input_rows": len(rows), "effective_training_rows": effective,
        "prompt_length_dropped": len(prompt_drops), "completion_partially_truncated": len(completion_truncation),
        "estimated_total_optimization_steps": estimated_steps,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
