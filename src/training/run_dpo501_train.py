#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / "results/training"
REUSED = PROJECT / "src/training/run_lora_dpo_checkpointed.py"


def load_reused():
    spec = importlib.util.spec_from_file_location("reused_expanded367_trainer", REUSED)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {REUSED}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.WORK = WORK
    module.MODEL_PATH = Path(__import__("os").environ.get("HCLT_BASE_MODEL", PROJECT / "models/HyperCLOVAX-SEED-Think-14B"))
    module.TRAIN_PATH = PROJECT / "data/preferences/dpo_train_501_chatformatted.jsonl"
    module.TOKEN_PREFLIGHT = PROJECT / "data/preferences/token_length_preflight.json"
    module.CONFIG_PATH = PROJECT / "configs/lora_dpo501_config.json"
    module.MANIFEST_PATH = WORK / "metadata/preflight_manifest.json"
    module.REPORT_PATH = WORK / "reports/training_summary.md"
    module.FINAL_ADAPTER = Path(__import__("os").environ.get("HCLT_DPO_ADAPTER_OUT", PROJECT / "models/adapters/dpo501_final_adapter"))
    module.SMOKE_ADAPTER = WORK / "smoke_test_adapter"
    module.OFFLOAD_DIR = WORK / "offload"
    return module


def write_summary() -> None:
    status_path = WORK / "metadata/train_status.json"
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    config = status["config"]
    lines = [
        "# DPO 501 Training Summary", "", "- status: `COMPLETED`",
        f"- input: `{PROJECT / 'work/hcx_dpo900_mode_jury_v1/outputs/preferences/dpo_preferences_all_survivors.jsonl'}`",
        f"- dataset_rows_loaded: `{config['dataset_rows']}`",
        f"- base_model: `{config['base_model']}`",
        f"- fresh_lora_initialized: `{config['fresh_lora_initialized']}`",
        f"- old_dpo_adapter_loaded: `{config['old_dpo_adapter_loaded']}`",
        f"- parameter_counts: `{config['parameter_counts']}`",
        f"- elapsed_sec: `{status['elapsed_sec']}`",
        f"- metrics: `{status['metrics']}`",
        f"- final_adapter: `{status['adapter_dir']}`",
    ]
    path = WORK / "reports/training_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    module = load_reused()
    module.main()
    if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "train":
        write_summary()


if __name__ == "__main__":
    main()
