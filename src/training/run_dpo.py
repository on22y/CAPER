#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/training/dpo_common.json"
DEFAULT_TRAIN = REPO / "data/processed/dpo_train_498.jsonl"
DEFAULT_PROMPT = REPO / "prompts/rewrite/document_rewrite_instruction_v1.txt"
PROMPT_MARKER = "--------------------------------------------------\nUSER MESSAGE TEMPLATE\n--------------------------------------------------"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_family(config: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(getattr(config, "architectures", None) or []).lower()
    joined = f"{model_type} {architectures}"
    if "hyperclovax" in joined:
        return "hyperclovax"
    if "qwen3" in joined:
        return "qwen3"
    if "gemma3" in joined or "gemma_3" in joined:
        return "gemma3"
    raise RuntimeError(
        f"Unsupported architecture model_type={model_type!r}, architectures={architectures!r}. "
        "Use --model-family only after adding an explicit preset."
    )


def split_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split(PROMPT_MARKER, 1)[0].strip()


def source_text(row: dict[str, Any]) -> str:
    return str(row.get("source") or row.get("source_text") or "").strip()


def render_prompt(tokenizer: Any, family: str, system: str, domain: str, source: str) -> tuple[str, str]:
    user = (
        "Rewrite the SOURCE document below according to the instructions above.\n\n"
        f"[DOMAIN]\n{domain}\n\n[SOURCE]\n{source}\n\n"
        "Return exactly one valid JSON object."
    )
    options: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if family == "qwen3":
        options["enable_thinking"] = False
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, **options), "system_and_user_roles"
    except Exception:
        # Gemma templates commonly reject a system role. Preserve the text by folding it into user.
        folded = [{"role": "user", "content": system + "\n\n" + user}]
        return tokenizer.apply_chat_template(folded, **options), "system_folded_into_user"


def build_dataset(
    rows: list[dict[str, Any]], tokenizer: Any, family: str, system: str, max_length: int
) -> tuple[Dataset, dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    modes: Counter[str] = Counter()
    prompts_at_or_above_max_length: list[dict[str, Any]] = []
    for row in rows:
        missing = {"sample_id", "domain", "chosen", "rejected"} - row.keys()
        if missing or not source_text(row):
            raise RuntimeError(f"Invalid training row {row.get('sample_id')}: missing={sorted(missing)}")
        prompt, mode = render_prompt(tokenizer, family, system, str(row["domain"]), source_text(row))
        modes[mode] += 1
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if prompt_tokens >= max_length:
            prompts_at_or_above_max_length.append(
                {"sample_id": row["sample_id"], "prompt_tokens": prompt_tokens}
            )
        formatted.append({
            "prompt": prompt,
            "chosen": json.dumps({"rewrite": row["chosen"]}, ensure_ascii=False),
            "rejected": json.dumps({"rewrite": row["rejected"]}, ensure_ascii=False),
            "sample_id": row["sample_id"],
            "domain": row["domain"],
        })
    return Dataset.from_list(formatted), {
        "rows": len(formatted),
        "unique_sample_ids": len({row["sample_id"] for row in formatted}),
        "domain_counts": dict(Counter(row["domain"] for row in formatted)),
        "chat_template_modes": dict(modes),
        "prompts_at_or_above_max_length": prompts_at_or_above_max_length,
        "rows_removed_for_length": 0,
    }


def discover_targets(model: Any, preset: dict[str, Any], family: str) -> list[str]:
    suffixes = set(preset["target_module_suffixes"])
    scope = preset.get("target_scope_contains")
    linear_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if family == "gemma3":
        targets = [
            name for name in linear_names
            if (not scope or scope in name) and name.rsplit(".", 1)[-1] in suffixes
        ]
    else:
        present = {name.rsplit(".", 1)[-1] for name in linear_names}
        targets = sorted(suffixes & present)
    missing = sorted(suffixes - {name.rsplit(".", 1)[-1] for name in targets})
    if missing:
        raise RuntimeError(f"LoRA target validation failed for {family}; missing suffixes={missing}")
    if not targets:
        raise RuntimeError(f"No LoRA targets found for {family}")
    return targets


def load_model(model_path: str, family: str, local_only: bool, trust_remote_code: bool) -> Any:
    common = {
        "pretrained_model_name_or_path": model_path,
        "local_files_only": local_only,
        "trust_remote_code": trust_remote_code,
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if family == "hyperclovax":
        from transformers import HyperCLOVAXForCausalLM
        model = HyperCLOVAXForCausalLM.from_pretrained(**common)
    else:
        model = AutoModelForCausalLM.from_pretrained(**common)
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable LoRA-DPO runner for HCLT multi-model comparison")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-family", choices=["auto", "hyperclovax", "qwen3", "gemma3"], default="auto")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    training = dict(config["training"])
    if args.per_device_train_batch_size is not None:
        training["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        training["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    effective_batch = training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
    if effective_batch != training["target_effective_batch_size"]:
        raise RuntimeError(
            f"Effective batch must remain {training['target_effective_batch_size']}; got {effective_batch}"
        )

    raw_rows = read_jsonl(args.train_file)
    if len(raw_rows) != 498 or len({row.get("sample_id") for row in raw_rows}) != 498:
        raise RuntimeError(f"Canonical comparison dataset must be 498 unique rows; found {len(raw_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Tokenizer has no official chat_template")
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    family = infer_family(model_config, args.model_family)
    preset = config["models"][family]
    dataset, dataset_audit = build_dataset(
        raw_rows, tokenizer, family, split_prompt(args.prompt_file), int(training["max_length"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "preflight_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PREFLIGHT_OK",
        "model_name_or_path": args.model_name_or_path,
        "model_family": family,
        "model_type": getattr(model_config, "model_type", None),
        "architectures": getattr(model_config, "architectures", None),
        "train_file": str(args.train_file.resolve()),
        "train_sha256": sha256(args.train_file),
        "prompt_file": str(args.prompt_file.resolve()),
        "prompt_sha256": sha256(args.prompt_file),
        "dataset": dataset_audit,
        "effective_batch_size_single_process": effective_batch,
        "training": training,
    })

    model = load_model(args.model_name_or_path, family, args.local_files_only, args.trust_remote_code)
    targets = discover_targets(model, preset, family)
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg["bias"]),
        task_type=str(lora_cfg["task_type"]),
        target_modules=targets,
        modules_to_save=lora_cfg.get("modules_to_save"),
    )
    trainer_args = DPOConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        beta=float(training["beta"]),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=args.smoke_steps if args.smoke_steps is not None else -1,
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_length=int(training["max_length"]),
        truncation_mode=str(training["truncation_mode"]),
        bf16=bool(training["bf16"]),
        fp16=bool(training["fp16"]),
        tf32=training["tf32"],
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        precompute_ref_log_probs=bool(training["precompute_ref_log_probs"]),
        remove_unused_columns=bool(training["remove_unused_columns"]),
        dataloader_pin_memory=bool(training["dataloader_pin_memory"]),
        optim=str(training["optim"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        warmup_steps=int(training["warmup_steps"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        seed=int(training["seed"]),
        data_seed=training["data_seed"],
        loss_type=str(training["loss_type"]),
        label_smoothing=float(training["label_smoothing"]),
        save_strategy=str(training["save_strategy"]),
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        logging_steps=1 if args.smoke_steps is not None else int(training["logging_steps"]),
        prediction_loss_only=True,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainable = [name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad]
    if not trainable or not all("lora" in name.lower() for name in trainable):
        raise RuntimeError("Trainable parameter audit failed: expected LoRA-only training")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_adapter = args.output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    write_json(args.output_dir / "train_status.json", {
        "status": "COMPLETED",
        "model_family": family,
        "metrics": result.metrics,
        "target_modules": targets,
        "trainable_parameter_names": trainable,
        "final_adapter": str(final_adapter.resolve()),
    })


if __name__ == "__main__":
    main()
