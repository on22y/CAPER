#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/home/user20250805/gayeon/HCLT2026")
ROOT = PROJECT / "work/hcx_base_vs_dpo501_final51_eval_v1"
INPUT = PROJECT / "work/hcx_base_vs_dpo501_final51_v1/outputs/paired/base_vs_dpo501_rewrites_51.jsonl"
PROMPT = ROOT / "prompts/gpt4o_substantive_expressive_similarity_system_prompt_v1.txt"
GPT_RUNNER = ROOT / "scripts/run_gpt4o_substantive_similarity.py"
CANONICAL_GPT_RUNNER = PROJECT / "work/hcx_dpo367_dataset_audit_v1/scripts/run_gpt4o_substantive_similarity.py"
EXPECTED_INPUT_SHA = "a2638aec0cbf243bf012261caa7906c86521fa18b9178b1026109813b3f97cee"
EXPECTED_PROMPT_SHA = "ee4383acad495ee63d42d22bd7a10f2a5b0e216cbafd6aca9d0894522479c542"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight() -> dict:
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    check = {
        "input": str(INPUT), "input_rows": len(rows), "input_sha256": sha256(INPUT),
        "prompt": str(PROMPT), "prompt_sha256": sha256(PROMPT),
        "gpt_runner": str(GPT_RUNNER), "canonical_gpt_runner": str(CANONICAL_GPT_RUNNER),
        "gpt_runner_sha256": sha256(GPT_RUNNER), "canonical_gpt_runner_sha256": sha256(CANONICAL_GPT_RUNNER),
        "gpt_runner_byte_identical": sha256(GPT_RUNNER) == sha256(CANONICAL_GPT_RUNNER),
        "expected_gpt_calls_without_retries": 102,
        "metrics": [
            "Entity F1", "Numeric/Date F1", "BiNLI Support", "BiNLI Coverage", "BiNLI F1",
            "Korean BERTScore Precision", "Korean BERTScore Recall", "Korean BERTScore F1",
            "chrF++", "Morpheme ROUGE-L", "Extractive Coverage", "Extractive Density", "Novel-2", "Novel-4",
        ],
        "qa_based_factuality_excluded": True,
    }
    if check["input_rows"] != 51 or check["input_sha256"] != EXPECTED_INPUT_SHA:
        raise RuntimeError("Input preflight failed")
    if check["prompt_sha256"] != EXPECTED_PROMPT_SHA or not check["gpt_runner_byte_identical"]:
        raise RuntimeError("Canonical GPT evaluator/prompt preflight failed")
    write_json(ROOT / "metadata/preflight_manifest.json", check)
    return check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    check = preflight()
    print(json.dumps(check, ensure_ascii=False), flush=True)
    if args.preflight:
        return

    started = time.monotonic()
    started_at = now()
    commands = {
        "automatic_metrics": [sys.executable, str(ROOT / "scripts/run_automatic_metrics.py")],
        "gpt4o": [sys.executable, str(GPT_RUNNER), "--run-type", "final", "--max-retries", "3"],
    }
    processes = {}
    handles = {}
    stage_started = {}
    stage_finished = {}
    for stage, command in commands.items():
        log_path = ROOT / "logs" / ("automatic_metrics.log" if stage == "automatic_metrics" else "gpt4o_similarity.log")
        handle = log_path.open("w", encoding="utf-8")
        handles[stage] = handle
        stage_started[stage] = time.monotonic()
        processes[stage] = subprocess.Popen(command, cwd=ROOT / "scripts", stdout=handle, stderr=subprocess.STDOUT)
    pending = set(processes)
    while pending:
        for stage in list(pending):
            if processes[stage].poll() is not None:
                stage_finished[stage] = time.monotonic()
                pending.remove(stage)
        time.sleep(1)
    for handle in handles.values():
        handle.close()
    failures = {stage: process.returncode for stage, process in processes.items() if process.returncode != 0}
    if failures:
        raise RuntimeError(f"Evaluation stages failed: {failures}")

    aggregate_started = time.monotonic()
    aggregate_log = ROOT / "logs/gpt4o_aggregation.log"
    with aggregate_log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/aggregate_gpt4o_similarity.py")],
            cwd=ROOT / "scripts", stdout=handle, stderr=subprocess.STDOUT, check=True,
        )
    finished = time.monotonic()
    outputs = [
        ROOT / "automatic_metrics/outputs/per_sample_metrics_102.jsonl",
        ROOT / "automatic_metrics/outputs/automatic_metric_summary.json",
        ROOT / "automatic_metrics/reports/automatic_metric_summary.md",
        ROOT / "gpt4o/raw/substantive_similarity_final.jsonl",
        ROOT / "gpt4o/processed/substantive_similarity_pair_scores_51.jsonl",
        ROOT / "gpt4o/processed/substantive_similarity_summary.json",
        ROOT / "gpt4o/processed/substantive_similarity_api_usage_summary.json",
    ]
    write_json(ROOT / "metadata/final_runtime_and_artifacts.json", {
        "status": "COMPLETE", "started_at": started_at, "completed_at": now(),
        "runtime_seconds": finished - started,
        "stage_runtime_seconds": {
            stage: stage_finished[stage] - stage_started[stage] for stage in commands
        } | {"gpt4o_aggregation": finished - aggregate_started},
        "input_sha256": sha256(INPUT),
        "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
    })
    print(f"PIPELINE_COMPLETE runtime_seconds={finished - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
