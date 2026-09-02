#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "work/hcx_rewrite_prompt_v3_mode_margin_pilot30_v1"
BASE_SCRIPT = PROJECT / "work/hcx_pairwise_expand300_v1/scripts/run_expand300_rewrite_generation.py"
CONFIG = PROJECT / "work/hcx_pairwise_expand300_v1/configs/hcx_generation_config.json"
MODEL_PATH = PROJECT / "models/HyperCLOVAX-SEED-Think-14B"
MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
PROMPT = ROOT / "prompts/document_rewrite_instruction_v3_mode_controlled.txt"
MANIFEST = ROOT / "data/pilot_source_manifest_30.jsonl"
OUTPUT = ROOT / "outputs/rewrites/pilot_v3_mode_controlled_rewrites_30x2.jsonl"
RAW = ROOT / "outputs/rewrites/pilot_v3_mode_controlled_rewrites_raw.jsonl"
MODES = ("CONSERVATIVE", "TRANSFORMATIVE")


def load_base_module():
    spec = importlib.util.spec_from_file_location("hcx_base_generation_v3", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def split_prompt(text: str) -> tuple[str, str]:
    marker = "--------------------------------------------------\nUSER MESSAGE TEMPLATE\n--------------------------------------------------"
    if text.count(marker) != 1:
        raise RuntimeError("Canonical prompt must contain one USER MESSAGE TEMPLATE marker")
    system, template = text.split(marker, 1)
    return system.rstrip(), template.strip()


def generate(base, model, tokenizer, row, mode, system, template, config):
    base_seed = base.stable_seed(f"{row['original_sample_id']}|v3|{mode}")
    last = None
    for attempt in range(1, int(config.get("max_attempts_per_candidate", 3)) + 1):
        seed = base_seed if attempt == 1 else base.stable_seed(
            f"{row['original_sample_id']}|v3|{mode}|retry{attempt}"
        )
        user = template.format(domain=row["domain"], rewrite_mode=mode, source_text=row["source_text"])
        raw_text, input_tokens, output_tokens, latency = base.generate_hcx_once(
            model=model, tokenizer=tokenizer, system=system, user=user, seed=seed, config=config
        )
        rewrite, valid, errors, validation = base.validate_rewrite(row["source_text"], raw_text)
        attempt_row = {
            "pilot_sample_id": row["pilot_sample_id"],
            "original_sample_id": row["original_sample_id"],
            "domain": row["domain"],
            "rewrite_mode": mode,
            "attempt": attempt,
            "seed": seed,
            "valid": valid,
            "errors": errors,
            "validation": validation,
            "raw_text": raw_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_sec": latency,
        }
        append_jsonl(RAW, attempt_row)
        last = attempt_row | {"rewrite_text": rewrite}
        if valid:
            break
    return last


def main() -> None:
    base = load_base_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    prompt_sha = hashlib.sha256(PROMPT.read_bytes()).hexdigest()
    system, template = split_prompt(PROMPT.read_text(encoding="utf-8"))
    rows = read_jsonl(MANIFEST)
    completed = set()
    if OUTPUT.exists():
        completed = {(x["pilot_sample_id"], x["rewrite_mode"]) for x in read_jsonl(OUTPUT)}

    print(f"START sources={len(rows)} expected_generations=60 resume_completed={len(completed)}", flush=True)
    print(f"PROMPT_SHA256={prompt_sha}", flush=True)
    print(f"MODEL={MODEL_ID} DPO_ADAPTER_NOT_LOADED=True", flush=True)
    model, tokenizer, revision = base.load_hcx()
    print(
        f"MODEL_LOADED revision={revision} dtype=bf16 device_map="
        f"{json.dumps(getattr(model, 'hf_device_map', {}), ensure_ascii=False)}",
        flush=True,
    )
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        for mode in MODES:
            key = (row["pilot_sample_id"], mode)
            if key in completed:
                continue
            result = generate(base, model, tokenizer, row, mode, system, template, config)
            valid = bool(result["valid"])
            final = {
                "pilot_sample_id": row["pilot_sample_id"],
                "original_sample_id": row["original_sample_id"],
                "domain": row["domain"],
                "source_text": row["source_text"],
                "source_pool": row.get("source_pool"),
                "source_provenance": row.get("source_provenance"),
                "v1_rewrite_output_path": row.get("v1_rewrite_output_path"),
                "rewrite_mode": mode,
                "rewrite_text": result["rewrite_text"] if valid else None,
                "seed": result["seed"],
                "generation_status": "generated" if valid else "failed_validation",
                "latency_sec": result["latency_sec"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "attempt": result["attempt"],
                "validation_errors": result["errors"],
                "validation": result["validation"],
                "prompt_version": "document_rewrite_instruction_v3_mode_controlled",
                "prompt_path": str(PROMPT),
                "prompt_sha256": prompt_sha,
                "model_identifier": MODEL_ID,
                "model_path": str(MODEL_PATH),
            }
            append_jsonl(OUTPUT, final)
            completed.add(key)
            print(
                f"PROGRESS source={index}/{len(rows)} generations={len(completed)}/60 "
                f"sample={row['pilot_sample_id']} mode={mode} valid={valid}",
                flush=True,
            )
    print(
        f"COMPLETE generations={len(completed)} elapsed_sec={time.perf_counter() - started:.3f} output={OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
