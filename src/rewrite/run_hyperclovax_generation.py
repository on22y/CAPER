#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, HyperCLOVAXConfig, HyperCLOVAXForCausalLM, set_seed


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/hcx_pairwise_expand300_v1"
GENERATOR = {
    "model_name": "hyperclovax",
    "model_id": "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
    "local_path": str(ROOT / "models/HyperCLOVAX-SEED-Think-14B"),
}
JUDGE_MODELS = ["exaone", "qwen3", "ministral3"]
VALID_PREFS = {"A", "B", "TIE"}
HCX_USE_CACHE = True


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


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
    if not isinstance(value, str):
        return ""
    return value.strip()


def has_bad_extra_text(text: str) -> bool:
    bad = ["<|im_start|>", "<|im_end|>", "assistant/think", "저작권 침해"]
    return any(token in (text or "") for token in bad)


def build_user_message(domain: str, source_text: str) -> str:
    return (
        "Rewrite the SOURCE document below according to the instructions above.\n\n"
        f"[DOMAIN]\n{domain}\n\n"
        f"[SOURCE]\n{source_text}\n\n"
        "Return exactly one valid JSON object."
    )


def load_hcx() -> tuple[Any, Any, str]:
    path = GENERATOR["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with open(Path(path) / "config.json", "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict.pop("auto_map", None)
    config = HyperCLOVAXConfig.from_dict(config_dict)
    config.use_cache = HCX_USE_CACHE
    model = HyperCLOVAXForCausalLM.from_pretrained(
        path,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "22GiB", 1: "22GiB", "cpu": "96GiB"},
        offload_folder=str(WORK / "outputs/hcx_offload"),
    )
    model.config.use_cache = HCX_USE_CACHE
    model.generation_config.use_cache = HCX_USE_CACHE
    model.eval()
    print(f"model.config.use_cache={model.config.use_cache}", flush=True)
    print(f"model.generation_config.use_cache={model.generation_config.use_cache}", flush=True)
    print(f"generate_use_cache={HCX_USE_CACHE}", flush=True)
    print("DPO_ADAPTER_NOT_LOADED=True", flush=True)
    print(f"attention_implementation={getattr(model.config, '_attn_implementation', None)}", flush=True)
    print(f"hf_device_map={getattr(model, 'hf_device_map', {})}", flush=True)
    return model, tokenizer, Path(path).name


def encode_hcx(tokenizer: Any, system: str, user: str) -> dict[str, torch.Tensor]:
    kwargs = {
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
        "add_generation_prompt": True,
        "skip_reasoning": True,
    }
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, **kwargs)


def infer_device(model: Any) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_hcx_once(
    model: Any,
    tokenizer: Any,
    system: str,
    user: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[str, int, int, float]:
    set_seed(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded = encode_hcx(tokenizer, system, user)
    device = infer_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
    input_len = int(encoded["input_ids"].shape[-1])
    started = time.time()
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=int(config["max_new_tokens"]),
            do_sample=bool(config["do_sample"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            repetition_penalty=float(config["repetition_penalty"]),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            use_cache=HCX_USE_CACHE,
        )
    gen = out[0, input_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip(), input_len, int(gen.shape[-1]), time.time() - started


def validate_rewrite(source: str, raw: str) -> tuple[str, bool, list[str], dict[str, Any] | None]:
    parsed, ok = parse_json_object(raw)
    errors = []
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


def generation_config_metadata(config: dict[str, Any], revision: str) -> dict[str, Any]:
    return {
        "model_id": GENERATOR["model_id"],
        "model_revision": revision,
        "backend": "transformers",
        "inference_backend": "transformers",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "device_map": "auto: GPU0/GPU1 model-parallel split",
        "use_cache": HCX_USE_CACHE,
        "skip_reasoning": True,
        "do_sample": bool(config["do_sample"]),
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "repetition_penalty": float(config["repetition_penalty"]),
        "max_new_tokens": int(config["max_new_tokens"]),
        "seed_rule": "int(SHA256(sample_id + '|A' or '|B')[:8], 16)",
    }


def has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def is_clear_truncation(raw: str, rewrite: str) -> bool:
    stripped = (raw or "").strip()
    if not rewrite:
        return True
    if stripped.endswith("...") or stripped.endswith("…"):
        return True
    return False


def raw_record(
    row: dict[str, Any],
    label: str,
    attempt: int,
    seed: int,
    raw: str,
    parsed: dict[str, Any] | None,
    valid: bool,
    errors: list[str],
    in_tok: int,
    out_tok: int,
    latency: float,
) -> dict[str, Any]:
    tps = (out_tok / latency) if latency else 0.0
    return {
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "candidate": label,
        "candidate_label": label,
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
        "output_tokens_per_sec": round(tps, 6),
        "use_cache": HCX_USE_CACHE,
        "timestamp": now(),
    }


def smoke_quality_checks(source: str, rewrite_a: str, rewrite_b: str, raw_a: str, raw_b: str) -> dict[str, Any]:
    return {
        "rewrite_a_non_empty": bool(rewrite_a),
        "rewrite_b_non_empty": bool(rewrite_b),
        "rewrite_a_korean": has_hangul(rewrite_a),
        "rewrite_b_korean": has_hangul(rewrite_b),
        "rewrite_text_markdown_code_fence": "```" in rewrite_a or "```" in rewrite_b,
        "raw_markdown_code_fence": "```" in raw_a or "```" in raw_b,
        "reasoning_or_debug_text": any(t in (rewrite_a + rewrite_b) for t in ["assistant/think", "분석:", "추론", "debug"]),
        "legal_commentary": any(t in (rewrite_a + rewrite_b) for t in ["저작권 침해", "법적 책임", "법률 자문"]),
        "source_exact_copy": rewrite_a.strip() == source.strip() or rewrite_b.strip() == source.strip(),
        "a_b_exact_duplicate": bool(rewrite_a and rewrite_a == rewrite_b),
        "clear_truncation": is_clear_truncation(raw_a, rewrite_a) or is_clear_truncation(raw_b, rewrite_b),
    }


def generate_valid_candidate(
    model: Any,
    tokenizer: Any,
    system: str,
    row: dict[str, Any],
    label: str,
    base_seed: int,
    raw_path: Path | None,
    max_attempts: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    user = build_user_message(row["domain"], row["source_text"])
    records: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        seed = base_seed if attempt == 1 else stable_seed(f"{row['sample_id']}|{label}|retry{attempt}")
        raw, in_tok, out_tok, latency = generate_hcx_once(model, tokenizer, system, user, seed, config=json.loads((WORK / "configs/hcx_generation_config.json").read_text(encoding="utf-8")))
        rewrite, valid, errors, parsed = validate_rewrite(row["source_text"], raw)
        rec = raw_record(row, label, attempt, seed, raw, parsed, valid, errors, in_tok, out_tok, latency)
        records.append(rec)
        if raw_path is not None:
            append_jsonl(raw_path, rec)
        if valid:
            return rewrite, records
    return "", records


def write_use_cache_smoke_report(smoke_rows: list[dict[str, Any]], reproducibility: dict[str, Any]) -> None:
    lines = [
        "# use_cache=True Smoke Report",
        "",
        "- Backend: Transformers",
        f"- use_cache: `{HCX_USE_CACHE}`",
        "- Scope: News 1 / Petition 1 / SNS 1, A/B each",
        "",
        "## Reproducibility Check",
        "",
        f"- sample_id: `{reproducibility.get('sample_id')}`",
        f"- candidate: `{reproducibility.get('candidate')}`",
        f"- seed: `{reproducibility.get('seed')}`",
        f"- exact_output_reproduced: `{reproducibility.get('exact_output_reproduced')}`",
        f"- note: {reproducibility.get('note', '')}",
        "",
        "## Smoke Generations",
        "",
        "| sample_id | domain | A valid | B valid | A tok/s | B tok/s | quality flags |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in smoke_rows:
        flags = ", ".join(k for k, v in r["quality_checks"].items() if v and k not in {"rewrite_a_non_empty", "rewrite_b_non_empty", "rewrite_a_korean", "rewrite_b_korean"}) or "none"
        lines.append(
            f"| {r['sample_id']} | {r['domain']} | {r['a_valid']} | {r['b_valid']} | {r['a_tps']:.3f} | {r['b_tps']:.3f} | {flags} |"
        )
    lat = [x for r in smoke_rows for x in [r["a_elapsed_sec"], r["b_elapsed_sec"]]]
    tps = [x for r in smoke_rows for x in [r["a_tps"], r["b_tps"]]]
    if lat:
        lines += [
            "",
            "## Throughput",
            "",
            f"- mean elapsed sec: `{statistics.mean(lat):.3f}`",
            f"- mean output tok/s: `{statistics.mean(tps):.3f}`",
        ]
    (WORK / "reports/use_cache_true_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(WORK / "reports/use_cache_true_smoke_report.json", {"reproducibility": reproducibility, "smoke_rows": smoke_rows})


def run_use_cache_smoke(model: Any, tokenizer: Any, pilot: list[dict[str, Any]], system: str) -> None:
    smoke_ids = []
    for domain in ["news", "petition", "sns"]:
        smoke_ids.append(next(r["sample_id"] for r in pilot if r["domain"] == domain))
    write_json(WORK / "reports/hcx_smoke_samples.json", {"sample_ids": smoke_ids, "use_cache": HCX_USE_CACHE})
    by_id = {r["sample_id"]: r for r in pilot}

    repro_row = by_id[smoke_ids[0]]
    repro_seed = stable_seed(repro_row["sample_id"] + "|A")
    rw1, recs1 = generate_valid_candidate(model, tokenizer, system, repro_row, "A", repro_seed, raw_path=None, max_attempts=1)
    rw2, recs2 = generate_valid_candidate(model, tokenizer, system, repro_row, "A", repro_seed, raw_path=None, max_attempts=1)
    reproducibility = {
        "sample_id": repro_row["sample_id"],
        "candidate": "A",
        "seed": repro_seed,
        "exact_output_reproduced": bool(rw1 and rw1 == rw2),
        "first_valid": bool(rw1),
        "second_valid": bool(rw2),
        "note": "If false, likely sampling plus multi-GPU nondeterminism; seed is set before each generation.",
        "first_metrics": recs1[0] if recs1 else None,
        "second_metrics": recs2[0] if recs2 else None,
    }

    smoke_rows = []
    smoke_out = WORK / "outputs/hcx_expand300_smoke_rewrites.jsonl"
    for sid in smoke_ids:
        row = by_id[sid]
        seed_a = stable_seed(row["sample_id"] + "|A")
        seed_b = stable_seed(row["sample_id"] + "|B")
        rewrite_a, rec_a = generate_valid_candidate(model, tokenizer, system, row, "A", seed_a, raw_path=None, max_attempts=2)
        rewrite_b, rec_b = generate_valid_candidate(model, tokenizer, system, row, "B", seed_b, raw_path=None, max_attempts=2)
        raw_a = rec_a[-1]["raw_output"] if rec_a else ""
        raw_b = rec_b[-1]["raw_output"] if rec_b else ""
        out_row = {
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "source_text": row["source_text"],
            "rewrite_a": rewrite_a,
            "rewrite_b": rewrite_b,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "use_cache": HCX_USE_CACHE,
            "generation_metadata": {"generator": GENERATOR, "config": generation_config_metadata(json.loads((WORK / "configs/hcx_generation_config.json").read_text(encoding="utf-8")), Path(GENERATOR["local_path"]).name), "created_at": now()},
        }
        append_jsonl(smoke_out, out_row)
        smoke_rows.append({
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "a_valid": bool(rewrite_a),
            "b_valid": bool(rewrite_b),
            "a_input_tokens": rec_a[-1]["input_tokens"] if rec_a else None,
            "b_input_tokens": rec_b[-1]["input_tokens"] if rec_b else None,
            "a_output_tokens": rec_a[-1]["output_tokens"] if rec_a else None,
            "b_output_tokens": rec_b[-1]["output_tokens"] if rec_b else None,
            "a_elapsed_sec": rec_a[-1]["elapsed_sec"] if rec_a else None,
            "b_elapsed_sec": rec_b[-1]["elapsed_sec"] if rec_b else None,
            "a_tps": rec_a[-1]["output_tokens_per_sec"] if rec_a else 0.0,
            "b_tps": rec_b[-1]["output_tokens_per_sec"] if rec_b else 0.0,
            "quality_checks": smoke_quality_checks(row["source_text"], rewrite_a, rewrite_b, raw_a, raw_b),
        })
    write_use_cache_smoke_report(smoke_rows, reproducibility)
    mean_tps = statistics.mean([x for r in smoke_rows for x in [r["a_tps"], r["b_tps"]] if x])
    print(f"[SMOKE use_cache] rows={len(smoke_rows)} mean_output_tok_s={mean_tps:.3f}", flush=True)
    if mean_tps < 10:
        raise RuntimeError(f"use_cache smoke throughput too low: {mean_tps:.3f} tok/s")


def maybe_write_10gen_report(raw_path: Path, start_time: float) -> None:
    report = WORK / "reports/use_cache_true_throughput_10gen.md"
    if report.exists():
        return
    rows = [r for r in read_jsonl(raw_path) if r.get("parse_valid") and r.get("use_cache") is True]
    if len(rows) < 10:
        return
    first10 = rows[:10]
    elapsed = [r["elapsed_sec"] for r in first10]
    input_tokens = [r["input_tokens"] for r in first10]
    output_tokens = [r["output_tokens"] for r in first10]
    tps = [r["output_tokens_per_sec"] for r in first10]
    total_elapsed = time.time() - start_time
    remaining_generations = 600 - len(rows)
    eta_sec = remaining_generations * statistics.mean(elapsed)
    lines = [
        "# use_cache=True Throughput After 10 Successful Generations",
        "",
        f"- successful generations counted: `{len(first10)}`",
        f"- mean input tokens: `{statistics.mean(input_tokens):.3f}`",
        f"- mean output tokens: `{statistics.mean(output_tokens):.3f}`",
        f"- mean elapsed sec: `{statistics.mean(elapsed):.3f}`",
        f"- median elapsed sec: `{statistics.median(elapsed):.3f}`",
        f"- mean output tok/s: `{statistics.mean(tps):.3f}`",
        f"- total elapsed since full-run start sec: `{total_elapsed:.3f}`",
        f"- estimated remaining rewrite generation sec: `{eta_sec:.3f}`",
        f"- estimated remaining rewrite generation hours: `{eta_sec/3600:.3f}`",
        "",
        "## First 10 Rows",
        "",
        "| sample_id | candidate | input | output | elapsed | tok/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in first10:
        lines.append(f"| {r['sample_id']} | {r['candidate']} | {r['input_tokens']} | {r['output_tokens']} | {r['elapsed_sec']:.3f} | {r['output_tokens_per_sec']:.3f} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[10GEN report] {report}", flush=True)


def generation_smoke_and_full() -> None:
    pilot = read_jsonl(WORK / "data/source/expanded_source_manifest_300.jsonl")
    system = (WORK / "instructions/document_rewrite_instruction_v1.txt").read_text(encoding="utf-8").split(
        "--------------------------------------------------\nUSER MESSAGE TEMPLATE\n--------------------------------------------------"
    )[0].strip()
    config = json.loads((WORK / "configs/hcx_generation_config.json").read_text(encoding="utf-8"))
    out_path = WORK / "outputs/hcx_expand300_rewrites.jsonl"
    raw_path = WORK / "outputs/hcx_expand300_rewrites_raw.jsonl"
    done = {r["sample_id"] for r in read_jsonl(out_path) if r.get("rewrite_a") and r.get("rewrite_b")}
    print(f"[START rewrite expand300] pending={len(pilot)-len(done)} output={out_path}", flush=True)
    model, tokenizer, revision = load_hcx()
    config.update(generation_config_metadata(config, revision))
    config["model_revision"] = revision
    config["instruction_sha256"] = sha256_file(WORK / "instructions/document_rewrite_instruction_v1.txt")
    write_json(WORK / "configs/hcx_generation_config.json", config)

    if not read_jsonl(out_path) and not read_jsonl(raw_path):
        run_use_cache_smoke(model, tokenizer, pilot, system)

    full_started = time.time()
    for row in pilot:
        done = {r["sample_id"] for r in read_jsonl(out_path) if r.get("rewrite_a") and r.get("rewrite_b")}
        if row["sample_id"] in done:
            continue
        result: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "source_text": row["source_text"],
            "source_file": row.get("source_file"),
            "source_row_index": row.get("source_row_index"),
            "rewrite_a": "",
            "rewrite_b": "",
            "seed_a": stable_seed(row["sample_id"] + "|A"),
            "seed_b": stable_seed(row["sample_id"] + "|B"),
            "generation_metadata": {**generation_config_metadata(config, revision), "generator": GENERATOR, "created_at": now(), "dpo_adapter_loaded": False},
            "validation": {},
            "use_cache": HCX_USE_CACHE,
        }
        label_errors: dict[str, list[str]] = {}
        label_records: dict[str, list[dict[str, Any]]] = {}
        for label in ["A", "B"]:
            base_seed = result["seed_a"] if label == "A" else result["seed_b"]
            rewrite, records = generate_valid_candidate(model, tokenizer, system, row, label, base_seed, raw_path=raw_path, max_attempts=3)
            label_records[label] = records
            if rewrite:
                result["rewrite_a" if label == "A" else "rewrite_b"] = rewrite
            else:
                label_errors[label] = records[-1]["errors"] if records else ["generation_failure"]
            maybe_write_10gen_report(raw_path, full_started)
        if result["rewrite_a"] == result["rewrite_b"] and result["rewrite_a"]:
            label_errors["B"] = ["a_b_exact_duplicate"]
            for attempt in range(3, 5):
                seed = stable_seed(f"{row['sample_id']}|B|duplicate_retry{attempt}")
                raw, in_tok, out_tok, latency = generate_hcx_once(model, tokenizer, system, build_user_message(row["domain"], row["source_text"]), seed, config)
                rewrite, valid, errors, parsed = validate_rewrite(row["source_text"], raw)
                rec = raw_record(row, "B", attempt, seed, raw, parsed, valid, errors, in_tok, out_tok, latency)
                append_jsonl(raw_path, rec)
                maybe_write_10gen_report(raw_path, full_started)
                if valid and rewrite != result["rewrite_a"]:
                    result["rewrite_b"] = rewrite
                    label_errors.pop("B", None)
                    break
        result["validation"] = {
            "rewrite_a_valid": bool(result["rewrite_a"]),
            "rewrite_b_valid": bool(result["rewrite_b"]),
            "a_b_exact_duplicate": bool(result["rewrite_a"] and result["rewrite_a"] == result["rewrite_b"]),
            "errors": label_errors,
            "use_cache": HCX_USE_CACHE,
        }
        append_jsonl(out_path, result)
        print(f"[rewrite] {len(read_jsonl(out_path))}/300 {row['sample_id']} valid={not label_errors}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[END rewrite expand300] rows={len(read_jsonl(out_path))} raw={len(read_jsonl(raw_path))}", flush=True)


def validate_nonpilot_input() -> None:
    rows = read_jsonl(WORK / "data/source/expanded_source_manifest_300.jsonl")
    ids = [r.get("sample_id") for r in rows]
    if len(rows) != 300:
        raise RuntimeError(f"expected 300 source rows, got {len(rows)}")
    if len(set(ids)) != 300:
        raise RuntimeError("expand300 source ids are not unique")
    if any(not (r.get("source_text") or "").strip() for r in rows):
        raise RuntimeError("empty source_text exists")
    print(f"[INPUT validated expand300] rows={len(rows)} unique_ids={len(set(ids))}", flush=True)


def main() -> None:
    for d in ["data", "outputs", "reports", "configs", "instructions", "metadata"]:
        (WORK / d).mkdir(parents=True, exist_ok=True)
    validate_nonpilot_input()
    generation_smoke_and_full()


if __name__ == "__main__":
    main()
