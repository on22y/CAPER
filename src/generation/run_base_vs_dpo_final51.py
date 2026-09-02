#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "data/processed/heldout_51.jsonl"
DEFAULT_PROMPT = REPO / "prompts/rewrite/document_rewrite_instruction_v1.txt"
DEFAULT_CONFIG = REPO / "configs/generation/comparison_generation.json"
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def infer_family(config: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    joined = (
        str(getattr(config, "model_type", "")) + " " + " ".join(getattr(config, "architectures", None) or [])
    ).lower()
    for family in ("hyperclovax", "qwen3", "gemma3"):
        if family in joined:
            return family
    if "gemma_3" in joined:
        return "gemma3"
    raise RuntimeError(f"Unsupported model architecture: {joined}")


def system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").split(PROMPT_MARKER, 1)[0].strip()


def user_message(domain: str, source: str) -> str:
    return (
        "Rewrite the SOURCE document below according to the instructions above.\n\n"
        f"[DOMAIN]\n{domain}\n\n[SOURCE]\n{source}\n\n"
        "Return exactly one valid JSON object."
    )


def template_options(family: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
        "add_generation_prompt": True,
    }
    if family == "hyperclovax":
        options["skip_reasoning"] = True
    elif family == "qwen3":
        options["enable_thinking"] = False
    return options


def encode(tokenizer: Any, family: str, system: str, user: str) -> tuple[dict[str, torch.Tensor], str]:
    options = template_options(family)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, **options), "system_and_user_roles"
    except Exception:
        folded = [{"role": "user", "content": system + "\n\n" + user}]
        return tokenizer.apply_chat_template(folded, **options), "system_folded_into_user"


def infer_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(
    model_path: str, adapter_path: Path | None, family: str, local_only: bool, trust_remote_code: bool
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=local_only, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Tokenizer has no official chat_template")
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
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.set_adapter("default")
    elif isinstance(model, PeftModel) or hasattr(model, "peft_config"):
        raise RuntimeError("Base model unexpectedly contains a PEFT adapter")
    model.config.use_cache = True
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    model.eval()
    return model, tokenizer


def parse_rewrite(raw: str) -> tuple[str, list[str]]:
    errors: list[str] = []
    try:
        parsed = json.loads((raw or "").strip())
    except Exception:
        return "", ["json_parse_failure"]
    if not isinstance(parsed, dict) or set(parsed) != {"rewrite"}:
        errors.append("schema_keys_not_exact")
    rewrite = parsed.get("rewrite", "") if isinstance(parsed, dict) else ""
    if not isinstance(rewrite, str) or not rewrite.strip():
        errors.append("empty_rewrite")
        rewrite = ""
    return rewrite.strip(), errors


def generate_one(
    model: Any, tokenizer: Any, family: str, system: str, row: dict[str, Any], config: dict[str, Any],
    variant: str, raw_path: Path,
) -> dict[str, Any]:
    source = str(row["source_text"])
    user = user_message(str(row["domain"]), source)
    max_attempts = int(config["max_attempts"])
    for attempt in range(1, max_attempts + 1):
        suffix = "" if attempt == 1 else f"|retry{attempt}"
        seed = stable_seed(f"{row['sample_id']}|heldout|rewrite{suffix}")
        set_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        inputs, template_mode = encode(tokenizer, family, system, user)
        input_tokens = int(inputs["input_ids"].shape[-1])
        device = infer_device(model)
        inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
        kwargs = {
            "max_new_tokens": int(config["max_new_tokens"]),
            "do_sample": bool(config["do_sample"]),
            "temperature": float(config["temperature"]),
            "top_p": float(config["top_p"]),
            "repetition_penalty": float(config["repetition_penalty"]),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": bool(config["use_cache"]),
        }
        if config.get("top_k") is not None:
            kwargs["top_k"] = int(config["top_k"])
        started = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        completion = generated[0, input_tokens:]
        raw = tokenizer.decode(completion, skip_special_tokens=True).strip()
        rewrite, errors = parse_rewrite(raw)
        if rewrite == source.strip():
            errors.append("source_exact_copy")
        record = {
            "sample_id": row["sample_id"], "domain": row["domain"], "variant": variant,
            "attempt": attempt, "seed": seed, "valid": not errors, "errors": errors,
            "input_tokens": input_tokens, "output_tokens": int(completion.shape[-1]),
            "latency_sec": time.time() - started, "chat_template_mode": template_mode, "raw_output": raw,
        }
        append_jsonl(raw_path, record)
        if not errors:
            return {
                "sample_id": row["sample_id"], "domain": row["domain"], "source_text": source,
                "rewrite": rewrite, "seed": seed, "attempts": attempt,
            }
    raise RuntimeError(f"Generation failed after {max_attempts} attempts: {row['sample_id']} {variant}")


def run_variant(
    variant: str, model_path: str, adapter: Path | None, family: str, rows: list[dict[str, Any]],
    system: str, config: dict[str, Any], output_dir: Path, local_only: bool, trust_remote_code: bool,
) -> list[dict[str, Any]]:
    model, tokenizer = load_model(model_path, adapter, family, local_only, trust_remote_code)
    raw_path = output_dir / f"{variant}_rewrites_51_raw.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    outputs = [generate_one(model, tokenizer, family, system, row, config, variant, raw_path) for row in rows]
    write_jsonl(output_dir / f"{variant}_rewrites_51.jsonl", outputs)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched Base/DPO rewrites for the frozen held-out 51")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model-family", choices=["auto", "hyperclovax", "qwen3", "gemma3"], default="auto")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    domains = Counter(row.get("domain") for row in rows)
    if len(rows) != 51 or len({row.get("sample_id") for row in rows}) != 51:
        raise RuntimeError("Held-out input must contain exactly 51 unique sample IDs")
    if domains != Counter({"news": 17, "petition": 17, "sns": 17}):
        raise RuntimeError(f"Held-out domain mismatch: {domains}")
    required = {"sample_id", "domain", "source_text"}
    if any(not required <= row.keys() for row in rows):
        raise RuntimeError("Held-out rows must contain sample_id/domain/source_text")
    model_config = AutoConfig.from_pretrained(
        args.base_model, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code
    )
    family = infer_family(model_config, args.model_family)
    config = read_json(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    system = system_prompt(args.prompt)

    base_rows = run_variant(
        "base", args.base_model, None, family, rows, system, config, args.output,
        args.local_files_only, args.trust_remote_code,
    )
    dpo_rows = run_variant(
        "dpo", args.base_model, args.adapter, family, rows, system, config, args.output,
        args.local_files_only, args.trust_remote_code,
    )
    base_by_id = {row["sample_id"]: row for row in base_rows}
    dpo_by_id = {row["sample_id"]: row for row in dpo_rows}
    paired = [{
        "sample_id": row["sample_id"], "domain": row["domain"], "source_text": row["source_text"],
        "base_rewrite": base_by_id[row["sample_id"]]["rewrite"],
        "dpo_rewrite": dpo_by_id[row["sample_id"]]["rewrite"],
        "base_seed": base_by_id[row["sample_id"]]["seed"],
        "dpo_seed": dpo_by_id[row["sample_id"]]["seed"],
    } for row in rows]
    paired_path = args.output / "base_vs_dpo_rewrites_51.jsonl"
    write_jsonl(paired_path, paired)
    write_json(args.output / "generation_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE",
        "model": args.base_model, "model_family": family, "adapter": str(args.adapter.resolve()),
        "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
        "prompt": str(args.prompt.resolve()), "prompt_sha256": sha256(args.prompt),
        "config": config, "pairs": len(paired), "paired_sha256": sha256(paired_path),
    })


if __name__ == "__main__":
    main()
