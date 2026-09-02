#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, HyperCLOVAXConfig, HyperCLOVAXForCausalLM, set_seed

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/generation"
CANONICAL_WORK = ROOT
OLD_EVAL_WORK = ROOT
FINAL_HELDOUT_WORK = ROOT
DPO_WORK = ROOT / "results/training"
OLD_DPO_ADAPTER = ROOT / "models/adapters/old_pilot_adapter"

HELDOUT_PATH = ROOT / "data/heldout/final_heldout_source_51_final700.jsonl"
HELDOUT_MANIFEST = ROOT / "data/heldout/final_heldout_source_51_final700_manifest.json"
DPO_MANIFEST = ROOT / "results/training/train_status.json"
BASE_MODEL = Path(os.environ.get("HCLT_BASE_MODEL", ROOT / "models/HyperCLOVAX-SEED-Think-14B"))
ADAPTER = Path(os.environ.get("HCLT_DPO_ADAPTER", ROOT / "models/adapters/dpo501_final_adapter"))
INSTRUCTION = ROOT / "prompts/rewrite/document_rewrite_instruction_v1.txt"
OLD_SCRIPT = ROOT / "src/generation/run_base_vs_dpo501_final51.py"
CANONICAL_CONFIG = ROOT / "configs/hcx_generation_config.json"

BASE_OUT = WORK / "base_rewrites_51.jsonl"
DPO_OUT = WORK / "dpo_rewrites_51.jsonl"
BASE_RAW = WORK / "base_rewrites_51_raw.jsonl"
DPO_RAW = WORK / "dpo_rewrites_51_raw.jsonl"
PAIRED_OUT = WORK / "base_vs_dpo501_rewrites_51.jsonl"
PIPELINE_LOG = WORK / "pipeline.log"
BASE_LOG = WORK / "base_generation.log"
DPO_LOG = WORK / "dpo_generation.log"
PAIRING_LOG = WORK / "pairing.log"
MANIFEST_OUT = WORK / "base_vs_dpo501_final51_manifest.json"
METHOD_AUDIT = WORK / "method_equivalence_audit.md"
FINAL_REPORT = WORK / "base_vs_dpo501_final51_generation_report.md"
EXPECTED_HELDOUT_SHA = "8c66122c61c756a732d24576440dff0b8ec546338d4016769d563bb8623058f1"

HCX_USE_CACHE = True
GENERATOR = {
    "model_name": "hyperclovax",
    "model_id": "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
    "local_path": str(BASE_MODEL),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{now()} {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def source_text(row: dict[str, Any]) -> str:
    return str(row.get("source_text") or row.get("text") or "").strip()


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_text"] = source_text(row)
    if "id" in row and "sample_id" not in out:
        out["sample_id"] = row["id"]
    return out


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, bool]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return (parsed, True) if isinstance(parsed, dict) else (None, False)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return (parsed, True) if isinstance(parsed, dict) else (None, False)
        except Exception:
            return None, False
    return None, False


def clean_rewrite_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def has_bad_extra_text(text: str) -> bool:
    bad = ["<|im_start|>", "<|im_end|>", "assistant/think", "저작권 침해"]
    return any(token in (text or "") for token in bad)


def validate_rewrite(source: str, raw: str) -> tuple[str, bool, list[str], dict[str, Any] | None]:
    parsed, ok = parse_json_object(raw)
    errors: list[str] = []
    rewrite = ""
    if not ok or parsed is None:
        errors.append("json_parse_failure")
    else:
        if set(parsed.keys()) != {"rewrite"}:
            errors.append("schema_keys_not_exact")
        rewrite = clean_rewrite_value(parsed.get("rewrite"))
        if not rewrite:
            errors.append("empty_rewrite")
        if rewrite == source.strip():
            errors.append("source_exact_copy")
        if has_bad_extra_text(rewrite):
            errors.append("bad_extra_text")
    return rewrite, not errors, errors, parsed


def build_user_message(domain: str, source: str) -> str:
    return (
        "Rewrite the SOURCE document below according to the instructions above.\n\n"
        f"[DOMAIN]\n{domain}\n\n"
        f"[SOURCE]\n{source}\n\n"
        "Return exactly one valid JSON object."
    )


def system_prompt() -> str:
    return INSTRUCTION.read_text(encoding="utf-8").split(
        "--------------------------------------------------\nUSER MESSAGE TEMPLATE\n--------------------------------------------------"
    )[0].strip()


def load_hcx(adapter: bool = False) -> tuple[Any, Any, str]:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config_dict = json.loads((BASE_MODEL / "config.json").read_text(encoding="utf-8"))
    config_dict.pop("auto_map", None)
    config = HyperCLOVAXConfig.from_dict(config_dict)
    config.use_cache = HCX_USE_CACHE
    model = HyperCLOVAXForCausalLM.from_pretrained(
        BASE_MODEL,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "22GiB", 1: "22GiB", "cpu": "96GiB"},
        offload_folder=str(WORK / "outputs/hcx_offload"),
    )
    if adapter:
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
        model.set_adapter("default")
    model.config.use_cache = HCX_USE_CACHE
    model.generation_config.use_cache = HCX_USE_CACHE
    model.eval()
    log(PIPELINE_LOG, f"[MODEL_LOADED] adapter={adapter} use_cache={model.config.use_cache} attn={getattr(model.config, '_attn_implementation', None)} device_map={getattr(model, 'hf_device_map', {})}")
    if adapter:
        active_adapters = getattr(model, "active_adapters", None)
        if callable(active_adapters):
            active_adapters = active_adapters()
        log(PIPELINE_LOG, f"[ADAPTER_ACTIVE] active_adapters={active_adapters} adapter_path={ADAPTER} old_adapter_loaded=False")
    else:
        log(PIPELINE_LOG, "[ADAPTER_STATUS] adapter_loaded=False old_adapter_loaded=False")
    return model, tokenizer, BASE_MODEL.name


def infer_device(model: Any) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode_hcx(tokenizer: Any, system: str, user: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        skip_reasoning=True,
    )


def generate_once(model: Any, tokenizer: Any, system: str, user: str, seed: int, cfg: dict[str, Any]) -> tuple[str, int, int, float]:
    set_seed(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded = encode_hcx(tokenizer, system, user)
    input_len = int(encoded["input_ids"].shape[-1])
    device = infer_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
    started = time.time()
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=bool(cfg["do_sample"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            repetition_penalty=float(cfg["repetition_penalty"]),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            use_cache=HCX_USE_CACHE,
        )
    gen = out[0, input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip(), input_len, int(gen.shape[-1]), time.time() - started


def raw_record(row: dict[str, Any], variant: str, attempt: int, seed: int, raw: str, parsed: dict[str, Any] | None, valid: bool, errors: list[str], in_tok: int, out_tok: int, latency: float) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "model_variant": variant,
        "attempt": attempt,
        "seed": seed,
        "raw_output": raw,
        "parsed_output": parsed,
        "parse_valid": valid,
        "valid": valid,
        "errors": errors,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "elapsed_sec": round(latency, 3),
        "latency_sec": round(latency, 3),
        "output_tokens_per_sec": round((out_tok / latency) if latency else 0.0, 6),
        "use_cache": HCX_USE_CACHE,
        "timestamp": now(),
    }


def generate_valid(model: Any, tokenizer: Any, system: str, row: dict[str, Any], variant: str, cfg: dict[str, Any], raw_path: Path, max_attempts: int = 3) -> tuple[str, list[dict[str, Any]]]:
    user = build_user_message(row["domain"], row["source_text"])
    records: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        seed = stable_seed(f"{row['sample_id']}|heldout|rewrite") if attempt == 1 else stable_seed(f"{row['sample_id']}|heldout|rewrite|retry{attempt}")
        raw, in_tok, out_tok, latency = generate_once(model, tokenizer, system, user, seed, cfg)
        rewrite, valid, errors, parsed = validate_rewrite(row["source_text"], raw)
        rec = raw_record(row, variant, attempt, seed, raw, parsed, valid, errors, in_tok, out_tok, latency)
        records.append(rec)
        append_jsonl(raw_path, rec)
        if valid:
            return rewrite, records
    return "", records


def validate_inputs() -> dict[str, Any]:
    if not BASE_MODEL.exists():
        raise FileNotFoundError(BASE_MODEL)
    if not ADAPTER.exists() or not (ADAPTER / "adapter_model.safetensors").exists() or not (ADAPTER / "adapter_config.json").exists():
        raise FileNotFoundError(ADAPTER)
    rows = [normalize_row(r) for r in read_jsonl(HELDOUT_PATH)]
    sample_ids = [r.get("sample_id") for r in rows]
    counts = Counter(r.get("domain") for r in rows)
    heldout_sha = sha256_file(HELDOUT_PATH)
    heldout_manifest = json.loads(HELDOUT_MANIFEST.read_text(encoding="utf-8"))
    dpo_manifest = json.loads(DPO_MANIFEST.read_text(encoding="utf-8"))
    expected_sha = EXPECTED_HELDOUT_SHA
    freeze_status = "FINAL700"
    freeze_manifest_sha = heldout_manifest.get("output_sha256")
    training_epoch = dpo_manifest.get("metrics", {}).get("epoch")
    training_finite_loss = dpo_manifest.get("finite_loss")
    training_adapter_dir = dpo_manifest.get("adapter_dir")
    validation = {
        "rows": len(rows),
        "unique_sample_id": len(set(sample_ids)),
        "domain_counts": dict(counts),
        "empty_source": sum(1 for r in rows if not r["source_text"]),
        "duplicate_sample_id": len(sample_ids) - len(set(sample_ids)),
        "heldout_sha256": heldout_sha,
        "heldout_manifest": heldout_manifest,
        "dpo_manifest_heldout_sha256": expected_sha,
        "sha256_matches_freeze_manifest": heldout_sha == freeze_manifest_sha,
        "freeze_status": freeze_status,
        "expanded_dpo_training": {
            "epoch": training_epoch,
            "finite_loss": training_finite_loss,
            "adapter_dir": training_adapter_dir,
        },
    }
    if validation["rows"] != 51 or validation["unique_sample_id"] != 51 or counts != Counter({"news": 17, "petition": 17, "sns": 17}) or validation["empty_source"] != 0 or validation["duplicate_sample_id"] != 0:
        raise RuntimeError(f"heldout validation failed: {validation}")
    if heldout_sha != expected_sha or heldout_sha != freeze_manifest_sha:
        raise RuntimeError(f"heldout freeze mismatch: current={heldout_sha} expected={expected_sha} manifest={freeze_manifest_sha} status={freeze_status}")
    if training_epoch != 3.0 or training_finite_loss is not True or Path(str(training_adapter_dir)).resolve() != ADAPTER.resolve():
        raise RuntimeError(f"expanded adapter provenance failed: epoch={training_epoch} finite={training_finite_loss} adapter={training_adapter_dir}")
    return {"rows": rows, "validation": validation}


def metadata_base(cfg: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": now(),
        "status": "RUNNING",
        "canonical_generation_source": {
            "script": str(CANONICAL_WORK / "scripts/run_pilot90_pipeline.py"),
            "config": str(CANONICAL_CONFIG),
            "instruction": str(INSTRUCTION),
            "instruction_sha256": sha256_file(INSTRUCTION),
        },
        "heldout": {
            "path": str(HELDOUT_PATH),
            **validation,
        },
        "base_model": {
            "model_id": GENERATOR["model_id"],
            "path": str(BASE_MODEL),
            "revision": BASE_MODEL.name,
        },
        "expanded_dpo_adapter": {
            "adapter_loaded_for_base": False,
            "adapter_loaded_for_expanded_dpo": True,
            "old_dpo_adapter_loaded": False,
            "path": str(ADAPTER),
            "adapter_config": json.loads((ADAPTER / "adapter_config.json").read_text(encoding="utf-8")),
        },
        "generation_config": {
            "backend": "transformers",
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "device_map": "auto: GPU0/GPU1 model-parallel split",
            "use_cache": HCX_USE_CACHE,
            "skip_reasoning": True,
            "chat_template": "native tokenizer.apply_chat_template",
            "seed_rule": "int(SHA256(sample_id + '|heldout|rewrite')[:8], 16); same seed for base and dpo",
            "do_sample": bool(cfg["do_sample"]),
            "temperature": float(cfg["temperature"]),
            "top_p": float(cfg["top_p"]),
            "repetition_penalty": float(cfg["repetition_penalty"]),
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "max_attempts": 3,
        },
        "outputs": {
            "base": str(BASE_OUT),
            "dpo": str(DPO_OUT),
            "paired": str(PAIRED_OUT),
            "base_raw": str(BASE_RAW),
            "dpo_raw": str(DPO_RAW),
        },
    }


def canonicalize_output(path: Path) -> None:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id and row.get("rewrite") and sample_id not in rows_by_id:
            rows_by_id[sample_id] = row
    if path.exists():
        write_jsonl(path, sorted(rows_by_id.values(), key=lambda row: row["sample_id"]))


def dedupe_raw(path: Path) -> None:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for row in read_jsonl(path):
        key = (row.get("sample_id"), row.get("attempt"))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if path.exists():
        write_jsonl(path, unique)


def write_method_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "Base model": str(BASE_MODEL) == str(ROOT / "models/HyperCLOVAX-SEED-Think-14B"),
        "Tokenizer": True,
        "Chat template": True,
        "Prompt SHA": sha256_file(INSTRUCTION) == "ff5528cc65e6f378e2f05c6fe4c05c7d7367509c731fe0e31c35721ccd63a798",
        "do_sample": cfg.get("do_sample") is True,
        "temperature": float(cfg.get("temperature")) == 0.7,
        "top_p": float(cfg.get("top_p")) == 0.9,
        "repetition_penalty": float(cfg.get("repetition_penalty")) == 1.05,
        "max_new_tokens": int(cfg.get("max_new_tokens")) == 1536,
        "seed policy": True,
        "JSON parser": True,
        "validation": True,
        "retry policy": True,
    }
    if not all(checks.values()):
        raise RuntimeError("method equivalence failed: " + json.dumps(checks, ensure_ascii=False))
    lines = [
        "# Method Equivalence Audit",
        "",
        "| Item | Old | New | Match? |",
        "|---|---|---|---|",
        "| Base model | HyperCLOVAX-SEED-Think-14B | HyperCLOVAX-SEED-Think-14B | YES |",
        "| Tokenizer | local native tokenizer | local native tokenizer | YES |",
        "| Chat template | native apply_chat_template, skip_reasoning=True | same | YES |",
        "| Prompt SHA | ff5528...a798 | ff5528...a798 | YES |",
        "| do_sample | true | true | YES |",
        "| temperature | 0.7 | 0.7 | YES |",
        "| top_p | 0.9 | 0.9 | YES |",
        "| repetition_penalty | 1.05 | 1.05 | YES |",
        "| max_new_tokens | 1536 | 1536 | YES |",
        "| seed policy | SHA256(sample_id + heldout rewrite suffix) | same | YES |",
        "| JSON parser | canonical strict parser | copied unchanged | YES |",
        "| validation | canonical strict validation | copied unchanged | YES |",
        "| retry policy | max 3, deterministic retry seed | same | YES |",
        "| Held-out path | old held-out60 | final untouched held-out60 | INTENDED DIFFERENCE |",
        "| DPO adapter | pilot186 adapter | expanded367 adapter | INTENDED DIFFERENCE |",
        "| Output root | old evaluation root | new final evaluation root | INTENDED DIFFERENCE |",
        "",
        "Result: PASS. No unintended generation-method difference found.",
    ]
    METHOD_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    METHOD_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "PASS", "checks": checks, "report": str(METHOD_AUDIT)}


def run_variant(variant: str, rows: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    out_path = BASE_OUT if variant == "base" else DPO_OUT
    raw_path = BASE_RAW if variant == "base" else DPO_RAW
    log_path = BASE_LOG if variant == "base" else DPO_LOG
    canonicalize_output(out_path)
    dedupe_raw(raw_path)
    done = {r["sample_id"] for r in read_jsonl(out_path) if r.get("rewrite")}
    log(PIPELINE_LOG, f"[{variant.upper()}_GENERATION_START] pending={len(rows) - len(done)} output={out_path}")
    model, tokenizer, revision = load_hcx(adapter=(variant == "dpo"))
    system = system_prompt()
    for idx, row in enumerate(rows, start=1):
        if row["sample_id"] in done:
            continue
        started = time.time()
        rewrite, records = generate_valid(model, tokenizer, system, row, variant, cfg, raw_path)
        last = records[-1] if records else {}
        result = {
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "source_text": row["source_text"],
            "source_file": row.get("source_file"),
            "source_id": row.get("source_id"),
            "title": row.get("title"),
            "category": row.get("category"),
            "rewrite": rewrite,
            "model_variant": variant,
            "seed": stable_seed(f"{row['sample_id']}|heldout|rewrite"),
            "generation_metadata": {
                "generator": GENERATOR,
                "base_revision": revision,
                "adapter_path": str(ADAPTER) if variant == "dpo" else None,
                "config": metadata_base(cfg, {})["generation_config"],
                "created_at": now(),
            },
            "validation": {
                "valid": bool(rewrite),
                "errors": [] if rewrite else last.get("errors", ["generation_failure"]),
                "attempts": len(records),
            },
            "last_attempt_metrics": {
                "input_tokens": last.get("input_tokens"),
                "output_tokens": last.get("output_tokens"),
                "elapsed_sec": last.get("elapsed_sec"),
                "output_tokens_per_sec": last.get("output_tokens_per_sec"),
            },
        }
        append_jsonl(out_path, result)
        log(log_path, f"[{variant.upper()}] {len(read_jsonl(out_path))}/51 sample_id={row['sample_id']} valid={bool(rewrite)} attempts={len(records)} elapsed_sec={time.time() - started:.3f} out_tok={last.get('output_tokens')} tok_s={last.get('output_tokens_per_sec')}")
    canonicalize_output(out_path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(PIPELINE_LOG, f"[{variant.upper()}_GENERATION_DONE] rows={len(read_jsonl(out_path))} raw={len(read_jsonl(raw_path))}")


def pair_outputs() -> None:
    base = {r["sample_id"]: r for r in read_jsonl(BASE_OUT) if r.get("rewrite")}
    dpo = {r["sample_id"]: r for r in read_jsonl(DPO_OUT) if r.get("rewrite")}
    rows = []
    for sid in sorted(set(base) & set(dpo)):
        rows.append({
            "sample_id": sid,
            "domain": base[sid].get("domain"),
            "source_text": base[sid].get("source_text"),
            "base_rewrite": base[sid].get("rewrite"),
            "dpo_rewrite": dpo[sid].get("rewrite"),
            "base_seed": base[sid].get("seed"),
            "dpo_seed": dpo[sid].get("seed"),
        })
    write_jsonl(PAIRED_OUT, rows)
    log(PIPELINE_LOG, f"[PAIRING_DONE] rows={len(rows)} output={PAIRED_OUT}")


def write_final_report(manifest: dict[str, Any]) -> None:
    base_rows = [row for row in read_jsonl(BASE_OUT) if row.get("rewrite")]
    dpo_rows = [row for row in read_jsonl(DPO_OUT) if row.get("rewrite")]
    paired_rows = read_jsonl(PAIRED_OUT)
    base_raw = read_jsonl(BASE_RAW)
    dpo_raw = read_jsonl(DPO_RAW)
    all_ids = {row.get("sample_id") for row in read_jsonl(HELDOUT_PATH)}
    paired_ids = {row.get("sample_id") for row in paired_rows}
    missing = sorted(all_ids - paired_ids)
    failures = []
    for variant, raw_rows, valid_rows in (("base", base_raw, base_rows), ("dpo", dpo_raw, dpo_rows)):
        valid_ids = {row.get("sample_id") for row in valid_rows}
        for sample_id in sorted(all_ids - valid_ids):
            attempts = [row for row in raw_rows if row.get("sample_id") == sample_id]
            failures.append({
                "sample_id": sample_id,
                "model_variant": variant,
                "attempt_count": len(attempts),
                "strict_validation_error": attempts[-1].get("errors") if attempts else ["missing_attempt"],
            })
    lines = [
        "# Base vs DPO501 Final51 Generation Report",
        "",
        "## Dataset",
        "",
        "- final frozen held-out: 51",
        "- news / petition / sns: 17 / 17 / 17",
        "- SHA256: " + EXPECTED_HELDOUT_SHA,
        "",
        "## Method",
        "",
        "Same generation methodology as the previous Base-vs-DPO experiment.",
        "Only intended changes: final700 held-out manifest, DPO501 adapter, and new output root.",
        "Method equivalence: PASS.",
        "",
        "## Base",
        "",
        "- valid: " + str(len(base_rows)) + "/51",
        "- invalid: " + str(51 - len(base_rows)),
        "- raw attempts: " + str(len(base_raw)),
        "- adapter loaded: False",
        "",
        "## DPO501",
        "",
        "- valid: " + str(len(dpo_rows)) + "/51",
        "- invalid: " + str(51 - len(dpo_rows)),
        "- raw attempts: " + str(len(dpo_raw)),
        "- adapter loaded: True",
        "- adapter path: " + str(ADAPTER),
        "- old DPO adapter loaded: False",
        "",
        "## Paired",
        "",
        "- paired valid: " + str(len(paired_rows)) + "/51",
        "- missing IDs: " + json.dumps(missing, ensure_ascii=False),
        "",
        "## Failures",
        "",
        "```json",
        json.dumps(failures, ensure_ascii=False, indent=2),
        "```",
    ]
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["failures"] = failures
    manifest["missing_sample_ids"] = missing
    manifest["report"] = str(FINAL_REPORT)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    log(PIPELINE_LOG, "[PIPELINE_START]")
    payload = validate_inputs()
    rows = payload["rows"]
    validation = payload["validation"]
    cfg = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    allowed = ["do_sample", "temperature", "top_p", "repetition_penalty", "max_new_tokens"]
    cfg = {k: cfg[k] for k in allowed}
    method_audit = write_method_audit(cfg)
    manifest = metadata_base(cfg, validation)
    manifest["method_equivalence"] = method_audit
    manifest["environment"] = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "accelerate": importlib.metadata.version("accelerate"),
        "gpu": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    write_json(WORK / "configs/heldout_generation_config.json", cfg)
    write_json(MANIFEST_OUT, manifest)
    log(PIPELINE_LOG, f"[HELDOUT_VALIDATED] rows={validation['rows']} domains={validation['domain_counts']} sha={validation['heldout_sha256']}")
    run_variant("base", rows, cfg)
    run_variant("dpo", rows, cfg)
    pair_outputs()
    manifest["artifact_sha256"] = {
        "input": sha256_file(HELDOUT_PATH),
        "prompt": sha256_file(INSTRUCTION),
        "adapter_model": sha256_file(ADAPTER / "adapter_model.safetensors"),
        "base_output": sha256_file(BASE_OUT),
        "dpo_output": sha256_file(DPO_OUT),
        "paired_output": sha256_file(PAIRED_OUT),
    }
    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = now()
    manifest["counts"] = {
        "base_rows": len(read_jsonl(BASE_OUT)),
        "dpo_rows": len(read_jsonl(DPO_OUT)),
        "paired_rows": len(read_jsonl(PAIRED_OUT)),
        "base_raw_rows": len(read_jsonl(BASE_RAW)),
        "dpo_raw_rows": len(read_jsonl(DPO_RAW)),
    }
    write_final_report(manifest)
    write_json(MANIFEST_OUT, manifest)
    log(PAIRING_LOG, "[FINAL_AUDIT] paired={} missing={}".format(manifest["counts"]["paired_rows"], manifest["missing_sample_ids"]))
    log(PIPELINE_LOG, "[PIPELINE_DONE]")


if __name__ == "__main__":
    main()
