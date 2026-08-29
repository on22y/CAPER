#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, HyperCLOVAXConfig, HyperCLOVAXForCausalLM
from trl import DPOConfig, DPOTrainer


ROOT = Path("/home/user20250805/gayeon/HCLT2026")
WORK = ROOT / "work/hcx_dpo_expanded367_v1"
MODEL_PATH = ROOT / "models/HyperCLOVAX-SEED-Think-14B"
TRAIN_PATH = WORK / "data/dpo/dpo_train_367_chatformatted.jsonl"
TOKEN_PREFLIGHT = WORK / "data/dpo/token_length_preflight.json"
CONFIG_PATH = WORK / "configs/lora_dpo_expanded367_config.json"
MANIFEST_PATH = WORK / "metadata/dpo_expanded367_manifest.json"
REPORT_PATH = WORK / "reports/dpo_expanded367_training_report.md"
FINAL_ADAPTER = WORK / "checkpoints/final_adapter"
SMOKE_ADAPTER = WORK / "checkpoints/smoke_test_adapter"
OFFLOAD_DIR = WORK / "checkpoints/offload"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_tokenizer() -> Any:
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model() -> Any:
    with (MODEL_PATH / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("auto_map", None)
    config = HyperCLOVAXConfig.from_dict(cfg)
    config.use_cache = False
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    model = HyperCLOVAXForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "22GiB", 1: "22GiB", "cpu": "128GiB"},
        offload_folder=str(OFFLOAD_DIR),
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def linear_suffixes(model: Any) -> list[str]:
    suffix_counts: Counter[str] = Counter()
    target_like = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "dense",
        "fc1",
        "fc2",
    }
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            suffix_counts[suffix] += 1
    preferred = [s for s in target_like if suffix_counts.get(s, 0) > 0]
    if preferred:
        return sorted(preferred)
    common = [name for name, _count in suffix_counts.most_common(12)]
    if not common:
        raise RuntimeError("No torch.nn.Linear modules found for LoRA target discovery.")
    return sorted(common)


def get_runtime_config(max_steps: int | None = None) -> dict[str, Any]:
    token_info = read_json(TOKEN_PREFLIGHT)
    max_length = int(token_info["recommended_max_length_no_truncation"])
    cfg = {
        "base_model": str(MODEL_PATH),
        "dtype": "bfloat16",
        "device_map": "auto",
        "use_cache": False,
        "trl_version": "runtime",
        "peft_mode": "LoRA adapter only",
        "full_parameter_finetuning": False,
        "qlora_or_quantization": False,
        "adapter_merge": False,
        "heldout_generation_or_evaluation": False,
        "dpo": {
            "beta": 0.1,
            "learning_rate": 5e-6,
            "num_train_epochs": 3.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_length": max_length,
            "truncation_mode": "keep_start",
            "bf16": True,
            "gradient_checkpointing": True,
            "precompute_ref_log_probs": False,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "save_strategy": "steps",
            "save_steps": 50,
            "save_total_limit": 2,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
    }
    if max_steps is not None:
        cfg["dpo"]["max_steps"] = max_steps
    return cfg


def dataset_for_mode(mode: str) -> Dataset:
    rows = read_jsonl(TRAIN_PATH)
    if mode == "smoke":
        token_info = read_json(TOKEN_PREFLIGHT)
        longest_ids = [x["sample_id"] for x in token_info.get("longest_examples", [])[:2]]
        by_id = {r["sample_id"]: r for r in rows}
        selected: list[dict[str, Any]] = []
        for sid in longest_ids:
            if sid in by_id:
                selected.append(by_id[sid])
        for row in rows:
            if row["sample_id"] not in {r["sample_id"] for r in selected}:
                selected.append(row)
            if len(selected) >= 4:
                break
        rows = selected
    return Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} | {"sample_id": r["sample_id"], "domain": r["domain"]} for r in rows])


def create_trainer(mode: str) -> tuple[DPOTrainer, dict[str, Any]]:
    tok = load_tokenizer()
    model = load_model()
    target_modules = linear_suffixes(model)
    cfg = get_runtime_config(max_steps=2 if mode == "smoke" else None)
    cfg["lora"]["target_modules"] = target_modules
    output_dir = WORK / ("checkpoints/smoke_run" if mode == "smoke" else "checkpoints/train_run")
    lora = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        bias=str(cfg["lora"]["bias"]),
        task_type=str(cfg["lora"]["task_type"]),
        target_modules=target_modules,
    )
    args = DPOConfig(
        output_dir=str(output_dir),
        beta=float(cfg["dpo"]["beta"]),
        learning_rate=float(cfg["dpo"]["learning_rate"]),
        num_train_epochs=float(cfg["dpo"]["num_train_epochs"]),
        max_steps=int(cfg["dpo"]["max_steps"]) if "max_steps" in cfg["dpo"] else -1,
        per_device_train_batch_size=int(cfg["dpo"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["dpo"]["gradient_accumulation_steps"]),
        max_length=int(cfg["dpo"]["max_length"]),
        truncation_mode=str(cfg["dpo"]["truncation_mode"]),
        bf16=bool(cfg["dpo"]["bf16"]),
        gradient_checkpointing=bool(cfg["dpo"]["gradient_checkpointing"]),
        precompute_ref_log_probs=bool(cfg["dpo"]["precompute_ref_log_probs"]),
        remove_unused_columns=bool(cfg["dpo"]["remove_unused_columns"]),
        dataloader_pin_memory=bool(cfg["dpo"]["dataloader_pin_memory"]),
        logging_steps=1 if mode == "smoke" else 5,
        save_strategy=str(cfg["dpo"]["save_strategy"]),
        save_steps=int(cfg["dpo"]["save_steps"]),
        save_total_limit=int(cfg["dpo"]["save_total_limit"]),
        prediction_loss_only=True,
        report_to=[],
        disable_tqdm=False,
    )
    data = dataset_for_mode(mode)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=data,
        processing_class=tok,
        peft_config=lora,
    )
    trainable = [(n, p.numel()) for n, p in trainer.model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(n for _name, n in trainable)
    cfg["parameter_counts"] = {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params else 0.0,
        "all_trainable_param_names_are_lora": all("lora" in name.lower() for name, _ in trainable),
        "trainable_param_name_examples": [name for name, _ in trainable[:20]],
    }
    cfg["model_device_map"] = getattr(model, "hf_device_map", {})
    cfg["dataset_rows"] = len(data)
    cfg["mode"] = mode
    cfg["old_dpo_adapter_loaded"] = False
    cfg["old_dpo_adapter_merged"] = False
    cfg["fresh_lora_initialized"] = True
    return trainer, cfg


def run_train(mode: str) -> dict[str, Any]:
    started = time.time()
    trainer, cfg = create_trainer(mode)
    print(json.dumps({"event": "trainer_ready", "mode": mode, "config": cfg}, ensure_ascii=False), flush=True)
    if not cfg["parameter_counts"]["all_trainable_param_names_are_lora"]:
        raise RuntimeError("Non-LoRA trainable parameters detected; refusing to train.")
    result = trainer.train()
    adapter_dir = SMOKE_ADAPTER if mode == "smoke" else FINAL_ADAPTER
    trainer.model.save_pretrained(adapter_dir)
    trainer.processing_class.save_pretrained(adapter_dir)
    metrics = dict(getattr(result, "metrics", {}) or {})
    finite_loss = True
    losses: list[float] = []
    for item in trainer.state.log_history:
        if "loss" in item:
            losses.append(float(item["loss"]))
    if losses:
        finite_loss = all(math.isfinite(x) for x in losses)
    status = {
        "mode": mode,
        "started_at": now(),
        "elapsed_sec": round(time.time() - started, 3),
        "adapter_dir": str(adapter_dir),
        "metrics": metrics,
        "losses": losses,
        "finite_loss": finite_loss,
        "adapter_files": sorted(p.name for p in adapter_dir.glob("*")),
        "config": cfg,
    }
    if not finite_loss:
        raise RuntimeError("Non-finite training loss detected.")
    write_json(WORK / f"metadata/{mode}_status.json", status)
    write_json(CONFIG_PATH, cfg)
    return status


def reload_check() -> dict[str, Any]:
    tok = load_tokenizer()
    model = load_model()
    if not FINAL_ADAPTER.exists():
        raise FileNotFoundError(f"Final adapter not found: {FINAL_ADAPTER}")
    peft_model = PeftModel.from_pretrained(model, FINAL_ADAPTER, is_trainable=False)
    text = "다음 문서를 재작성하세요.\n\n원문:\n서울시는 시민 의견을 수렴하기 위해 설명회를 개최했다."
    ids = tok(text, return_tensors="pt").to(next(p.device for p in peft_model.parameters() if p.device.type != "meta"))
    with torch.no_grad():
        out = peft_model(**ids)
    status = {
        "mode": "reload_check",
        "adapter_dir": str(FINAL_ADAPTER),
        "logits_shape": list(out.logits.shape),
        "finite_logits": bool(torch.isfinite(out.logits.detach().float()).all().item()),
        "heldout_generation_or_evaluation": False,
        "created_at": now(),
    }
    write_json(WORK / "metadata/reload_check_status.json", status)
    return status


def write_final_report(manifest: dict[str, Any]) -> None:
    heldout_manifest = {'validation': {'rows': 0, 'domain_counts': {}, 'existing300_exact_id_overlap': None, 'existing300_exact_normalized_text_overlap': None, 'existing300_near_duplicate_overlap': None, 'heldout_sha256': None}}
    dpo_validation = read_json(WORK / "data/preferences/combined_preference_validation.json")
    token_info = read_json(TOKEN_PREFLIGHT)
    train_status = read_json(WORK / "metadata/train_status.json") if (WORK / "metadata/train_status.json").exists() else {}
    smoke_status = read_json(WORK / "metadata/smoke_status.json") if (WORK / "metadata/smoke_status.json").exists() else {}
    reload_status = read_json(WORK / "metadata/reload_check_status.json") if (WORK / "metadata/reload_check_status.json").exists() else {}
    lines = [
        "# HyperCLOVAX LoRA-DPO Expanded367 Training Report",
        "",
        "## Status",
        "",
        f"- final_status: `{manifest.get('final_status')}`",
        f"- base_model: `{MODEL_PATH}`",
        f"- final_adapter: `{FINAL_ADAPTER}`",
        "- full_parameter_finetuning_executed: `NO`",
        "- qlora_or_quantization_executed: `NO`",
        "- adapter_merge_executed: `NO`",
        "- heldout_rewrite_generation_or_evaluation_executed: `NO`",
        "",
        "## Held-out Source Freeze",
        "",
        f"- rows: `{heldout_manifest['validation']['rows']}`",
        f"- domain_counts: `{heldout_manifest['validation']['domain_counts']}`",
        f"- existing300_exact_id_overlap: `{heldout_manifest['validation']['existing300_exact_id_overlap']}`",
        f"- existing300_exact_text_overlap: `{heldout_manifest['validation']['existing300_exact_normalized_text_overlap']}`",
        f"- existing300_near_duplicate_overlap: `{heldout_manifest['validation']['existing300_near_duplicate_overlap']}`",
        f"- heldout_sha256: `{heldout_manifest['validation']['heldout_sha256']}`",
        "",
        "## DPO Data",
        "",
        f"- rows: `{dpo_validation['rows']}`",
        f"- domain_counts: `{dpo_validation['domain_counts']}`",
        f"- preference_source_counts: `{dpo_validation.get('preference_source_counts')}`",
        f"- bad_count: `{dpo_validation['bad_count']}`",
        f"- label_type: `automatic pseudo-preference, not human gold`",
        "",
        "## Token Length",
        "",
        f"- recommended_max_length_no_truncation: `{token_info['recommended_max_length_no_truncation']}`",
        f"- max_observed_total: `{token_info['max_observed_total']}`",
        "",
        "## LoRA / DPO",
        "",
        f"- target_modules: `{train_status.get('config', smoke_status.get('config', {})).get('lora', {}).get('target_modules')}`",
        f"- parameter_counts: `{train_status.get('config', smoke_status.get('config', {})).get('parameter_counts')}`",
        f"- smoke_finite_loss: `{smoke_status.get('finite_loss')}`",
        f"- train_elapsed_sec: `{train_status.get('elapsed_sec')}`",
        f"- train_metrics: `{train_status.get('metrics')}`",
        f"- reload_check: `{reload_status}`",
        "",
        "## Files",
        "",
        f"- combined_preferences: `{WORK / 'data/preferences/dpo_preferences_combined_367.jsonl'}`",
        f"- dpo_train: `{TRAIN_PATH}`",
        f"- config: `{CONFIG_PATH}`",
        f"- manifest: `{MANIFEST_PATH}`",
        f"- report: `{REPORT_PATH}`",
    ]
    write_text(REPORT_PATH, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train", "reload_check", "report"], required=True)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "start", "mode": args.mode, "time": now()}, ensure_ascii=False), flush=True)
    if args.mode in {"smoke", "train"}:
        status = run_train(args.mode)
        print(json.dumps({"event": "done", "status": status}, ensure_ascii=False), flush=True)
    elif args.mode == "reload_check":
        status = reload_check()
        print(json.dumps({"event": "done", "status": status}, ensure_ascii=False), flush=True)
    else:
        manifest = read_json(MANIFEST_PATH)
        write_final_report(manifest)
        print(json.dumps({"event": "report_written", "report": str(REPORT_PATH)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
