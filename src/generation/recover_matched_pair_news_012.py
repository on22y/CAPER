#!/usr/bin/env python3
import gc
import importlib.util
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run_base_vs_dpo501_final51.py"
SPEC = importlib.util.spec_from_file_location("final51_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(runner)

TARGET = "dpo900_final700_news_012"
RECOVERY_SEED = runner.stable_seed(f"{TARGET}|heldout|matched_recovery_v1")
RECOVERY_AUDIT = runner.WORK / "outputs/recovery/matched_pair_news_012.json"


def generate(row, variant, cfg):
    model, tokenizer, revision = runner.load_hcx(adapter=(variant == "dpo"))
    if variant == "base" and hasattr(model, "peft_config"):
        raise RuntimeError("Base recovery unexpectedly loaded a PEFT adapter")
    raw, input_tokens, output_tokens, latency = runner.generate_once(
        model,
        tokenizer,
        runner.system_prompt(),
        runner.build_user_message(row["domain"], row["source_text"]),
        RECOVERY_SEED,
        cfg,
    )
    rewrite, valid, errors, parsed = runner.validate_rewrite(row["source_text"], raw)
    record = runner.raw_record(
        row, variant, 1, RECOVERY_SEED, raw, parsed, valid, errors,
        input_tokens, output_tokens, latency,
    )
    record["recovery"] = "matched_recovery_v1"
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rewrite, valid, errors, revision, record


def output_row(row, variant, rewrite, revision, record, cfg):
    return {
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "source_text": row["source_text"],
        "source_file": row.get("source_file"),
        "source_id": row.get("source_id"),
        "title": row.get("title"),
        "category": row.get("category"),
        "rewrite": rewrite,
        "model_variant": variant,
        "seed": RECOVERY_SEED,
        "generation_metadata": {
            "generator": runner.GENERATOR,
            "base_revision": revision,
            "adapter_path": str(runner.ADAPTER) if variant == "dpo" else None,
            "config": runner.metadata_base(cfg, {})["generation_config"],
            "created_at": runner.now(),
            "recovery": "matched_recovery_v1",
        },
        "validation": {"valid": True, "errors": [], "attempts": 1},
        "last_attempt_metrics": {
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
            "elapsed_sec": record["elapsed_sec"],
            "output_tokens_per_sec": record["output_tokens_per_sec"],
        },
    }


def replace_or_add(path, replacement):
    rows = [row for row in runner.read_jsonl(path) if row.get("sample_id") != TARGET]
    rows.append(replacement)
    runner.write_jsonl(path, sorted(rows, key=lambda row: row["sample_id"]))


def main() -> None:
    payload = runner.validate_inputs()
    target_rows = [row for row in payload["rows"] if row["sample_id"] == TARGET]
    if len(target_rows) != 1:
        raise RuntimeError(f"target lookup failed: {len(target_rows)}")
    row = target_rows[0]

    cfg_all = json.loads(runner.CANONICAL_CONFIG.read_text(encoding="utf-8"))
    keys = ["do_sample", "temperature", "top_p", "repetition_penalty", "max_new_tokens"]
    cfg = {key: cfg_all[key] for key in keys}
    runner.write_method_audit(cfg)

    base_rewrite, base_valid, base_errors, base_revision, base_record = generate(row, "base", cfg)
    dpo_rewrite, dpo_valid, dpo_errors, dpo_revision, dpo_record = generate(row, "dpo", cfg)
    audit = {
        "target": TARGET,
        "recovery_seed": RECOVERY_SEED,
        "base_valid": base_valid,
        "base_errors": base_errors,
        "dpo_valid": dpo_valid,
        "dpo_errors": dpo_errors,
        "committed": False,
        "base_raw": base_record,
        "dpo_raw": dpo_record,
        "completed_at": runner.now(),
    }
    runner.write_json(RECOVERY_AUDIT, audit)

    if not (base_valid and dpo_valid):
        print(json.dumps({key: audit[key] for key in (
            "target", "recovery_seed", "base_valid", "dpo_valid", "committed"
        )}, ensure_ascii=False))
        return

    replace_or_add(
        runner.BASE_OUT,
        output_row(row, "base", base_rewrite, base_revision, base_record, cfg),
    )
    replace_or_add(
        runner.DPO_OUT,
        output_row(row, "dpo", dpo_rewrite, dpo_revision, dpo_record, cfg),
    )
    runner.pair_outputs()

    manifest = json.loads(runner.MANIFEST_OUT.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["base_output"] = runner.sha256_file(runner.BASE_OUT)
    manifest["artifact_sha256"]["dpo_output"] = runner.sha256_file(runner.DPO_OUT)
    manifest["artifact_sha256"]["paired_output"] = runner.sha256_file(runner.PAIRED_OUT)
    manifest["counts"]["base_rows"] = len(runner.read_jsonl(runner.BASE_OUT))
    manifest["counts"]["dpo_rows"] = len(runner.read_jsonl(runner.DPO_OUT))
    manifest["counts"]["paired_rows"] = len(runner.read_jsonl(runner.PAIRED_OUT))
    audit["committed"] = True
    manifest["matched_pair_recovery"] = {
        key: audit[key] for key in (
            "target", "recovery_seed", "base_valid", "dpo_valid", "committed", "completed_at"
        )
    }
    runner.write_final_report(manifest)
    runner.write_json(runner.MANIFEST_OUT, manifest)
    runner.write_json(RECOVERY_AUDIT, audit)
    print(json.dumps(manifest["matched_pair_recovery"], ensure_ascii=False))


if __name__ == "__main__":
    main()
