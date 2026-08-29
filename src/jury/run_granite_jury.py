#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

PROJECT = Path("/home/user20250805/gayeon/HCLT2026")
ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_granite_command_v1"
CANONICAL_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_v1"
BASE_SCRIPT = PROJECT / "work/hcx_v3_overall_jury6vote_pilot_v1/scripts/run_overall_jury6vote.py"
PAIR_INPUT = ROOT / "data/jury_input_pairs_28.jsonl"
ASSIGNMENT = ROOT / "data/candidate_assignment.jsonl"
USER_TEMPLATE = ROOT / "prompts/direct_expressive_user_template.txt"
PROMPTS = {
    "granite": ROOT / "prompts/granite_direct_expressive_v1.txt",
    "command_r7b": ROOT / "prompts/command_r7b_direct_expressive_v1.txt",
}
PROMPT_VERSIONS = {
    "granite": "granite_direct_expressive_v1",
    "command_r7b": "command_r7b_direct_expressive_v1",
}
PAIRWISE = ROOT / "outputs/aggregation/granite_command_pairwise_results.jsonl"
REPORT = ROOT / "reports/granite_command_jury28_report.md"
SETUP = ROOT / "reports/setup_and_runtime.txt"
GPT_REFERENCE = CANONICAL_ROOT / "outputs/aggregation/direct_expr_6vote_results.jsonl"
JUDGES = ("granite", "command_r7b")
ORDERS = ("original", "swapped")
MAX_NEW_TOKENS = 4
SPECS = {
    "granite": {
        "model_id": "ibm-granite/granite-3.3-8b-instruct",
        "hf_id": "ibm-granite/granite-3.3-8b-instruct",
        "local_path": str(PROJECT / "models/judges/granite-3.3-8b-instruct"),
        "revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "family": "causal_lm",
        "trust_remote_code": False,
    },
    "command_r7b": {
        "model_id": "CohereLabs/c4ai-command-r7b-12-2024",
        "hf_id": "CohereLabs/c4ai-command-r7b-12-2024",
        "local_path": str(PROJECT / "models/judges/c4ai-command-r7b-12-2024"),
        "revision": "4f3d0aa6856e322f2f4480fe65420d5d53d297b8",
        "family": "causal_lm",
        "trust_remote_code": False,
    },
}


def load_base():
    spec = importlib.util.spec_from_file_location("granite_command_jury_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_text(pair: dict[str, Any], identity: str) -> str:
    return pair["conservative_text"] if identity == "CONSERVATIVE" else pair["transformative_text"]


def make_user(pair: dict[str, Any], a_identity: str, b_identity: str) -> str:
    return USER_TEMPLATE.read_text(encoding="utf-8").format(
        domain=pair["domain"],
        source_text=pair["source_text"],
        candidate_a=identity_text(pair, a_identity),
        candidate_b=identity_text(pair, b_identity),
    )


def prepare() -> None:
    canonical_pairs = read_jsonl(CANONICAL_ROOT / "data/jury_input_pairs.jsonl")
    canonical_assignments = read_jsonl(CANONICAL_ROOT / "data/candidate_assignment.jsonl")
    pairs = read_jsonl(PAIR_INPUT)
    assignments = read_jsonl(ASSIGNMENT)
    domains = Counter(row["domain"] for row in pairs)
    ids = [row["sample_id"] for row in pairs]
    if pairs != canonical_pairs or assignments != canonical_assignments:
        raise RuntimeError("Copied pair or assignment artifact differs from canonical Direct Jury input")
    if len(pairs) != 28 or domains != Counter({"news": 9, "petition": 9, "sns": 10}):
        raise RuntimeError(f"Unexpected canonical input: N={len(pairs)} domains={dict(domains)}")
    if len(ids) != len(set(ids)) or ids != [row["sample_id"] for row in assignments]:
        raise RuntimeError("Duplicate or misaligned sample IDs")
    preflight: dict[str, Any] = {}
    pair = pairs[0]
    for judge in JUDGES:
        model_path = Path(SPECS[judge]["local_path"])
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise RuntimeError(f"Incomplete local model directory: {model_path}")
        if not list(model_path.glob("*.safetensors")):
            raise RuntimeError(f"No safetensors found in {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(f"Official chat template missing for {judge}")
        encoded = base.format_messages(
            tokenizer,
            SPECS[judge],
            PROMPTS[judge].read_text(encoding="utf-8"),
            make_user(pair, pair["original_A_identity"], pair["original_B_identity"]),
        )
        preflight[judge] = {
            "local_path": str(model_path),
            "official_chat_template": True,
            "first_input_tokens": int(encoded["input_ids"].shape[-1]),
        }
        del tokenizer, encoded
    setup = {
        "reused_script_layer": str(BASE_SCRIPT),
        "canonical_pair_input": str(CANONICAL_ROOT / "data/jury_input_pairs.jsonl"),
        "canonical_assignment": str(CANONICAL_ROOT / "data/candidate_assignment.jsonl"),
        "pair_input": str(PAIR_INPUT),
        "assignment": str(ASSIGNMENT),
        "pairs": len(pairs),
        "domains": dict(domains),
        "expected_judgments": 112,
        "orders": list(ORDERS),
        "models": SPECS,
        "prompts": {judge: {"path": str(PROMPTS[judge]), "sha256": sha256(PROMPTS[judge])} for judge in JUDGES},
        "user_template": {"path": str(USER_TEMPLATE), "sha256": sha256(USER_TEMPLATE)},
        "preflight": preflight,
        "decoding": {"dtype": "bfloat16", "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "use_cache": True, "max_retry": 1},
        "oa_pass": "stable >= 17/28 (60.71%)",
        "gpt_openai_calls": 0,
        "eta": "roughly 4-10 minutes including two sequential model loads and aggregation",
    }
    SETUP.parent.mkdir(parents=True, exist_ok=True)
    SETUP.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PREFLIGHT_OK pairs=28 domains={dict(domains)} expected=112", flush=True)


def run_judge(judge: str) -> None:
    spec = SPECS[judge]
    output = ROOT / f"outputs/raw/{judge}.jsonl"
    previous = read_jsonl(output)
    done = {
        (row["sample_id"], row["presentation"])
        for row in previous
        if row.get("status") == "valid"
    }
    system = PROMPTS[judge].read_text(encoding="utf-8")
    pairs = read_jsonl(PAIR_INPUT)
    print(f"LOAD_START judge={judge} model={spec['model_id']} resume_valid={len(done)}", flush=True)
    model, tokenizer, model_path, _ = base.load_model(spec)
    device = base.input_device(model)
    print(f"MODEL_LOADED judge={judge} path={model_path} revision={spec['revision']} device={device}", flush=True)
    for pair_index, pair in enumerate(pairs, 1):
        for presentation in ORDERS:
            key = (pair["sample_id"], presentation)
            if key in done:
                continue
            if presentation == "original":
                a_identity, b_identity = pair["original_A_identity"], pair["original_B_identity"]
            else:
                a_identity, b_identity = pair["original_B_identity"], pair["original_A_identity"]
            user = make_user(pair, a_identity, b_identity)
            attempts: list[dict[str, Any]] = []
            choice = None
            latency_total = 0.0
            for attempt in range(1, 3):
                started = time.perf_counter()
                encoded = base.format_messages(tokenizer, spec, system, user)
                encoded = {name: tensor.to(device) for name, tensor in encoded.items() if torch.is_tensor(tensor)}
                input_tokens = int(encoded["input_ids"].shape[-1])
                with torch.inference_mode():
                    generated_all = model.generate(
                        **encoded,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                generated = generated_all[0, input_tokens:]
                raw = base.decode_generated(tokenizer, generated)
                latency = time.perf_counter() - started
                latency_total += latency
                choice = base.parse_choice(raw)
                attempts.append({"attempt": attempt, "raw_output": raw, "parsed_choice": choice, "latency_sec": latency})
                if choice is not None:
                    break
            mapped = a_identity if choice == "A" else b_identity if choice == "B" else None
            row = {
                "sample_id": pair["sample_id"],
                "domain": pair["domain"],
                "judge": judge,
                "model_id": spec["model_id"],
                "prompt_version": PROMPT_VERSIONS[judge],
                "presentation": presentation,
                "displayed_A_identity": a_identity,
                "displayed_B_identity": b_identity,
                "raw_output": attempts[-1]["raw_output"],
                "parsed_choice": choice,
                "mapped_candidate": mapped,
                "input_tokens": input_tokens,
                "output_tokens": int(generated.shape[-1]),
                "latency_sec": latency_total,
                "retry_count": len(attempts) - 1,
                "status": "valid" if choice else "invalid_output",
                "attempts": attempts,
                "model_path": model_path,
                "model_revision": spec["revision"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            base.append_jsonl(output, row)
            if choice is not None:
                done.add(key)
            print(f"JUDGED judge={judge} pair={pair_index}/28 order={presentation} status={row['status']} mapped={mapped}", flush=True)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"JUDGE_COMPLETE judge={judge} valid_keys={len(done)} output={output}", flush=True)


def latest_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[(row["sample_id"], row["presentation"])] = row
    return latest


def aggregate() -> None:
    pairs = read_jsonl(PAIR_INPUT)
    raw = {judge: latest_rows(ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    gpt_by_id = {row["sample_id"]: row.get("gpt4o_winner") for row in read_jsonl(GPT_REFERENCE)}
    pairwise: list[dict[str, Any]] = []
    for pair in pairs:
        sid = pair["sample_id"]
        result: dict[str, Any] = {"sample_id": sid, "domain": pair["domain"], "gpt4o_winner": gpt_by_id.get(sid)}
        for judge in JUDGES:
            original = raw[judge].get((sid, "original"), {})
            swapped = raw[judge].get((sid, "swapped"), {})
            o_vote, s_vote = original.get("mapped_candidate"), swapped.get("mapped_candidate")
            result[f"{judge}_original"] = o_vote
            result[f"{judge}_swapped"] = s_vote
            result[f"{judge}_stable"] = bool(o_vote and o_vote == s_vote)
        pairwise.append(result)
    write_jsonl(PAIRWISE, pairwise)

    total_unique = sum(len(rows) for rows in raw.values())
    all_latest = [row for rows in raw.values() for row in rows.values()]
    valid = [row for row in all_latest if row.get("status") == "valid"]
    retries = sum(int(row.get("retry_count", 0)) for row in all_latest)
    lines = [
        "# Granite and Command R7B Direct Expressive Independence Jury28",
        "",
        "## A. Completeness",
        "",
        f"- Pair N: {len(pairs)}",
        "- Expected judgments: 112",
        f"- Completed unique judgments: {total_unique}",
        f"- Valid: {len(valid)}",
        f"- Invalid: {total_unique - len(valid)}",
        f"- Retries: {retries}",
        "- GPT/OpenAI calls: 0",
        "",
        "## B. Primary Result",
        "",
        "| Judge | Stable | Unstable | OA % | >=60% PASS? |",
        "|---|---:|---:|---:|---|",
    ]
    stats: dict[str, dict[str, Any]] = {}
    for judge in JUDGES:
        complete = [row for row in pairwise if row[f"{judge}_original"] and row[f"{judge}_swapped"]]
        stable = [row for row in complete if row[f"{judge}_stable"]]
        unstable = len(complete) - len(stable)
        oa = 100 * len(stable) / len(complete) if complete else 0.0
        passed = len(stable) >= 17 and len(complete) == 28
        stats[judge] = {"complete": complete, "stable": stable, "unstable": unstable, "oa": oa, "passed": passed}
        lines.append(f"| {judge} | {len(stable)} | {unstable} | {oa:.2f}% | {'PASS' if passed else 'FAIL'} |")

    lines += ["", "## C. Raw Position Bias", "", "| Judge | A | B | A % | B % |", "|---|---:|---:|---:|---:|"]
    for judge in JUDGES:
        rows = [row for row in raw[judge].values() if row.get("status") == "valid"]
        choices = Counter(row.get("parsed_choice") for row in rows)
        n = sum(choices.values())
        lines.append(f"| {judge} | {choices['A']} | {choices['B']} | {100*choices['A']/n if n else 0:.2f}% | {100*choices['B']/n if n else 0:.2f}% |")
        if n and max(choices.values()) / n >= 0.90:
            lines.append(f"\nWarning: {judge} shows severe label collapse (>=90%).")

    lines += ["", "## D. Candidate Preference", "", "| Judge | T | C | T % |", "|---|---:|---:|---:|"]
    for judge in JUDGES:
        rows = [row for row in raw[judge].values() if row.get("status") == "valid"]
        mapped = Counter(row.get("mapped_candidate") for row in rows)
        n = mapped["TRANSFORMATIVE"] + mapped["CONSERVATIVE"]
        lines.append(f"| {judge} | {mapped['TRANSFORMATIVE']} | {mapped['CONSERVATIVE']} | {100*mapped['TRANSFORMATIVE']/n if n else 0:.2f}% |")

    lines += ["", "## E. Stable Subset", "", "| Judge | Stable N | Stable T | Stable C |", "|---|---:|---:|---:|"]
    for judge in JUDGES:
        stable = stats[judge]["stable"]
        selected = Counter(row[f"{judge}_original"] for row in stable)
        lines.append(f"| {judge} | {len(stable)} | {selected['TRANSFORMATIVE']} | {selected['CONSERVATIVE']} |")

    lines += [
        "",
        "## F. Existing Judge Reference",
        "",
        "| Judge | OA |",
        "|---|---:|",
        "| Qwen3 frozen Direct | 75.00% |",
        "| EXAONE frozen Direct | 39.29% |",
        "| Ministral3 frozen Direct | 53.57% |",
        f"| Granite | {stats['granite']['oa']:.2f}% |",
        f"| Command R7B | {stats['command_r7b']['oa']:.2f}% |",
        "",
        "## G. GPT Stable-subset Agreement",
        "",
        "| Judge | Agreement |",
        "|---|---:|",
    ]
    for judge in JUDGES:
        with_gpt = [row for row in stats[judge]["stable"] if row.get("gpt4o_winner")]
        agree = sum(row[f"{judge}_original"] == row["gpt4o_winner"] for row in with_gpt)
        lines.append(f"| {judge} | {agree}/{len(with_gpt)} ({100*agree/len(with_gpt) if with_gpt else 0:.2f}%) |")
    lines += [
        "",
        "OA is the primary metric. Raw A/B bias is secondary, GPT agreement is a reference diagnostic, and T/C preference is descriptive only.",
        "This is a 28-pair development pilot; PASS does not establish final judge performance.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AGGREGATION_COMPLETE pairs={len(pairs)} valid={len(valid)}/112 report={REPORT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "judge", "aggregate"))
    parser.add_argument("--judge", choices=JUDGES)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "judge":
        if not args.judge:
            parser.error("--judge is required")
        run_judge(args.judge)
    else:
        aggregate()


if __name__ == "__main__":
    main()
