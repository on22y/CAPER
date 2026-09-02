#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_v1"
PREV_ROOT = PROJECT / "work/hcx_v3_shortprompt_overall_jury6vote_pilot_v1"
PREV_SCRIPT = PREV_ROOT / "scripts/run_shortprompt_jury.py"
SOURCE = PROJECT / "work/hcx_rewrite_prompt_v3_mode_margin_pilot30_v1/outputs/rewrites/pilot_v3_mode_controlled_rewrites_30x2.jsonl"
ATTACHMENT = Path("/home/user20250805/.codex/attachments/e08cb2f3-5f55-45cd-8f86-fbb5e7698777/pasted-text.txt")
PAIR_INPUT = ROOT / "data/jury_input_pairs.jsonl"
ASSIGNMENT = ROOT / "data/candidate_assignment.jsonl"
SYSTEM_PROMPT = ROOT / "prompts/short_direct_pairwise_expressive_independence_system_prompt.txt"
USER_TEMPLATE = ROOT / "prompts/short_direct_pairwise_expressive_independence_user_template.txt"
DIAGNOSTIC = ROOT / "diagnostics/ministral_input_sanity.json"
AGGREGATE = ROOT / "outputs/aggregation/direct_expr_6vote_results.jsonl"
REPORT = ROOT / "reports/direct_expr_jury_report.md"
SETUP = ROOT / "reports/setup_and_runtime.txt"
GPT_RESULTS = PROJECT / "work/hcx_rewrite_prompt_v3_mode_margin_pilot30_v1/outputs/gpt4o_direct_pairwise/results.jsonl"
JUDGES = ("exaone", "qwen3", "ministral3")
ORDERS = ("original", "swapped")
MAX_NEW_TOKENS = 48


def load_previous():
    spec = importlib.util.spec_from_file_location("previous_shortprompt_jury", PREV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PREV_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prev = load_previous()
base = prev.base


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_block(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Cannot uniquely extract {begin}")
    return text.split(begin, 1)[1].split(end, 1)[0].strip() + "\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    valid_modes: dict[str, set[str]] = {}
    for row in read_jsonl(SOURCE):
        if row.get("generation_status") == "generated" and row.get("rewrite_text"):
            valid_modes.setdefault(row["pilot_sample_id"], set()).add(row["rewrite_mode"])
    valid_ids = sorted(sid for sid, modes in valid_modes.items() if modes == {"CONSERVATIVE", "TRANSFORMATIVE"})
    pairs = read_jsonl(PREV_ROOT / "data/jury_input_pairs.jsonl")
    assignments = read_jsonl(PREV_ROOT / "data/candidate_assignment.jsonl")
    if [row["sample_id"] for row in pairs] != valid_ids or [row["sample_id"] for row in assignments] != valid_ids:
        raise RuntimeError("Previous pair/assignment artifacts do not match the 28 canonical valid pairs")
    write_jsonl(PAIR_INPUT, pairs)
    write_jsonl(ASSIGNMENT, assignments)
    request = ATTACHMENT.read_text(encoding="utf-8")
    system = extract_block(request, "---------------- BEGIN SYSTEM PROMPT ----------------", "---------------- END SYSTEM PROMPT ----------------")
    user = extract_block(request, "---------------- BEGIN USER TEMPLATE ----------------", "---------------- END USER TEMPLATE ----------------")
    SYSTEM_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT.write_text(system, encoding="utf-8")
    USER_TEMPLATE.write_text(user, encoding="utf-8")
    domains = Counter(row["domain"] for row in pairs)
    specs = base.model_specs()
    setup = {
        "input": str(SOURCE), "valid_pairs": len(pairs), "domain_breakdown": dict(domains),
        "candidate_assignment_reused_from": str(PREV_ROOT / "data/candidate_assignment.jsonl"),
        "expected_judgments": len(pairs) * 6,
        "models": {judge: specs[judge] for judge in JUDGES},
        "prompt_paths": {"system": str(SYSTEM_PROMPT), "user": str(USER_TEMPLATE)},
        "prompt_sha256": {"system": sha(SYSTEM_PROMPT), "user": sha(USER_TEMPLATE)},
        "orders": list(ORDERS), "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "max_retry": 1},
        "ministral_input": "combined_system_and_user_single_UserMessage",
        "strength": [1, 2, 3, 4], "soft_logits": False, "gpt_openai_api_calls": 0,
        "eta": {
            "exaone": "roughly 1-2 minutes including load",
            "qwen3": "roughly 1-2 minutes including load",
            "ministral3": "roughly 1-3 minutes including load",
            "aggregation": "under 1 minute", "total_after_gpu_available": "roughly 4-8 minutes",
            "basis": "previous identical 168-judgment run had summed inference latencies 12.69/14.73/52.39 sec; allowance added for model loads and longer JSON output",
        },
    }
    SETUP.parent.mkdir(parents=True, exist_ok=True)
    SETUP.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(pairs) != 28 or domains != Counter({"news": 9, "petition": 9, "sns": 10}):
        raise RuntimeError(f"Unexpected pair breakdown: N={len(pairs)} domains={dict(domains)}")
    print(f"PREPARED pairs={len(pairs)} domains={dict(domains)} expected={len(pairs)*6}", flush=True)


def text_for(pair: dict, identity: str) -> str:
    return pair["conservative_text"] if identity == "CONSERVATIVE" else pair["transformative_text"]


def parse_result(raw: str) -> tuple[str | None, int | None]:
    text = (raw or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        choice, strength = data.get("more_independent"), data.get("difference_strength")
        if choice in {"A", "B"} and type(strength) is int and 1 <= strength <= 4:
            return choice, strength
    return None, None


def ministral_sanity() -> None:
    pair = sorted(read_jsonl(PAIR_INPUT), key=lambda row: len(row["source_text"]))[len(read_jsonl(PAIR_INPUT)) // 2]
    spec = base.model_specs()["ministral3"]
    from run_judge_v5_document import MistralCommonAdapter
    adapter = MistralCommonAdapter(spec["local_path"], spec["hf_id"])
    system = SYSTEM_PROMPT.read_text(encoding="utf-8")
    template = USER_TEMPLATE.read_text(encoding="utf-8")
    a_id, b_id = pair["original_A_identity"], pair["original_B_identity"]
    a_text, b_text = text_for(pair, a_id), text_for(pair, b_id)
    user = template.format(domain=pair["domain"], source_text=pair["source_text"], candidate_a=a_text, candidate_b=b_text)
    encoded = prev.fixed_format_messages(adapter, spec, system, user)
    decoded = adapter.decode(encoded["input_ids"][0])
    normalized = " ".join(decoded.split())
    checks = {
        "source_marker": "[SOURCE]" in decoded,
        "candidate_a_marker": "[CANDIDATE A]" in decoded,
        "candidate_b_marker": "[CANDIDATE B]" in decoded,
        "source_content": " ".join(pair["source_text"].split())[:80] in normalized,
        "candidate_a_content": " ".join(a_text.split())[:80] in normalized,
        "candidate_b_content": " ".join(b_text.split())[:80] in normalized,
    }
    result = {"sample_id": pair["sample_id"], "input_construction": "combined_system_and_user_single_UserMessage", "input_tokens": int(encoded["input_ids"].shape[-1]), "non_padding_tokens": int(encoded["attention_mask"].sum()), "checks": checks, "status": "passed" if all(checks.values()) else "failed"}
    DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MINISTRAL_SANITY status={result['status']} tokens={result['input_tokens']} checks={checks}", flush=True)
    if result["status"] != "passed":
        raise RuntimeError("Ministral input sanity failed")


def run_judge(judge: str) -> None:
    spec = base.model_specs()[judge]
    output = ROOT / f"outputs/raw/{judge}.jsonl"
    previous = read_jsonl(output)
    done = {(row["sample_id"], row["presentation"]) for row in previous}
    if len(done) != len(previous):
        raise RuntimeError(f"Duplicate checkpoint rows for {judge}")
    system = SYSTEM_PROMPT.read_text(encoding="utf-8")
    template = USER_TEMPLATE.read_text(encoding="utf-8")
    pairs = read_jsonl(PAIR_INPUT)
    print(f"LOAD_START judge={judge} model={spec['model_id']} resume={len(done)}", flush=True)
    model, adapter, model_path, revision = base.load_model(spec)
    device = base.input_device(model)
    print(f"MODEL_LOADED judge={judge} path={model_path} revision={revision} device={device}", flush=True)
    for pair_index, pair in enumerate(pairs, 1):
        for presentation in ORDERS:
            key = (pair["sample_id"], presentation)
            if key in done:
                continue
            if presentation == "original":
                a_identity, b_identity = pair["original_A_identity"], pair["original_B_identity"]
            else:
                a_identity, b_identity = pair["original_B_identity"], pair["original_A_identity"]
            user = template.format(domain=pair["domain"], source_text=pair["source_text"], candidate_a=text_for(pair, a_identity), candidate_b=text_for(pair, b_identity))
            attempts, choice, strength, latency_total = [], None, None, 0.0
            for attempt in range(1, 3):
                started = time.perf_counter()
                encoded = prev.fixed_format_messages(adapter, spec, system, user)
                encoded = {name: tensor.to(device) for name, tensor in encoded.items() if torch.is_tensor(tensor)}
                input_tokens = int(encoded["input_ids"].shape[-1])
                with torch.inference_mode():
                    generated_all = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=getattr(adapter, "eos_token_id", None) or getattr(getattr(adapter, "tokenizer", None), "eos_token_id", None))
                generated = generated_all[0, input_tokens:]
                raw = base.decode_generated(adapter, generated)
                latency = time.perf_counter() - started
                latency_total += latency
                choice, strength = parse_result(raw)
                attempts.append({"attempt": attempt, "raw_output": raw, "parsed_choice": choice, "difference_strength": strength, "latency_sec": latency})
                if choice is not None:
                    break
            mapped = a_identity if choice == "A" else b_identity if choice == "B" else None
            row = {
                "sample_id": pair["sample_id"], "domain": pair["domain"], "judge": judge, "presentation": presentation,
                "displayed_A_identity": a_identity, "displayed_B_identity": b_identity,
                "raw_output": attempts[-1]["raw_output"], "parsed_choice": choice, "mapped_candidate": mapped,
                "difference_strength": strength, "input_tokens": input_tokens, "output_tokens": int(generated.shape[-1]),
                "latency_sec": latency_total, "retry_count": len(attempts) - 1,
                "status": "valid" if choice else "invalid_output", "attempts": attempts,
                "model_id": spec["model_id"], "model_path": model_path, "model_revision": revision,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(output, row)
            done.add(key)
            print(f"JUDGED judge={judge} pair={pair_index}/{len(pairs)} order={presentation} status={row['status']} mapped={mapped} strength={strength}", flush=True)
    del model, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"JUDGE_COMPLETE judge={judge} rows={len(done)} output={output}", flush=True)


def stats(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "valid"]
    choices, mapped = Counter(row["parsed_choice"] for row in valid), Counter(row["mapped_candidate"] for row in valid)
    strengths = [row.get("difference_strength") for row in valid if row.get("difference_strength") is not None]
    per_order = {}
    for order in ORDERS:
        count = Counter(row["parsed_choice"] for row in valid if row["presentation"] == order)
        per_order[order] = count
    return {"n": len(valid), "A": choices["A"], "B": choices["B"], "mapped": mapped, "strength": Counter(strengths), "mean": statistics.fmean(strengths) if strengths else None, "median": statistics.median(strengths) if strengths else None, "orders": per_order}


def order_stats(lookup: dict, pairs: list[dict], judge: str) -> tuple[int, int]:
    same = different = 0
    for pair in pairs:
        original = lookup.get((pair["sample_id"], judge, "original"), {})
        swapped = lookup.get((pair["sample_id"], judge, "swapped"), {})
        if original.get("mapped_candidate") and swapped.get("mapped_candidate"):
            if original["mapped_candidate"] == swapped["mapped_candidate"]: same += 1
            else: different += 1
    return same, different


def aggregate_report() -> None:
    pairs = read_jsonl(PAIR_INPUT)
    raw = {judge: read_jsonl(ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    lookup = {(row["sample_id"], row["judge"], row["presentation"]): row for rows in raw.values() for row in rows}
    gpt = {row["pilot_sample_id"]: row for row in read_jsonl(GPT_RESULTS)}
    results = []
    for pair in pairs:
        sid, votes = pair["sample_id"], []
        row = {"sample_id": sid, "domain": pair["domain"]}
        for judge in JUDGES:
            original = lookup.get((sid, judge, "original"), {})
            swapped = lookup.get((sid, judge, "swapped"), {})
            for order, item in (("original", original), ("swapped", swapped)):
                row[f"{judge}_{order}"] = item.get("mapped_candidate")
                row[f"{judge}_{order}_strength"] = item.get("difference_strength")
            row[f"{judge}_order_agreement"] = bool(original.get("mapped_candidate") and original.get("mapped_candidate") == swapped.get("mapped_candidate"))
            row[f"{judge}_strength_agreement"] = bool(original.get("difference_strength") is not None and original.get("difference_strength") == swapped.get("difference_strength"))
            if original.get("difference_strength") is not None and swapped.get("difference_strength") is not None:
                row[f"{judge}_strength_abs_diff"] = abs(original["difference_strength"] - swapped["difference_strength"])
            votes += [x for x in (original.get("mapped_candidate"), swapped.get("mapped_candidate")) if x in {"CONSERVATIVE", "TRANSFORMATIVE"}]
        c, t = votes.count("CONSERVATIVE"), votes.count("TRANSFORMATIVE")
        if len(votes) < 6: winner = "incomplete_judgments"
        elif t > c: winner = "TRANSFORMATIVE"
        elif c > t: winner = "CONSERVATIVE"
        else: winner = "no_consensus"
        row.update({"transformative_votes": t, "conservative_votes": c, "vote_pattern": f"{max(t,c)}:{min(t,c)}" if len(votes) == 6 else f"incomplete_{len(votes)}/6", "vote_margin": abs(t-c), "final_winner": winner})
        if sid in gpt:
            row["gpt4o_winner"] = gpt[sid].get("more_independent_mode")
            row["gpt4o_strength"] = gpt[sid].get("difference_strength")
        results.append(row)
    write_jsonl(AGGREGATE, results)

    old_raw = {judge: read_jsonl(PREV_ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    old_lookup = {(row["sample_id"], row["judge"], row["presentation"]): row for rows in old_raw.values() for row in rows}
    all_rows = [row for rows in raw.values() for row in rows]
    valid = [row for row in all_rows if row.get("status") == "valid"]
    retries = sum(int(row.get("retry_count", 0)) for row in all_rows)
    invalid = len(all_rows) - len(valid)
    winners, patterns = Counter(row["final_winner"] for row in results), Counter(row["vote_pattern"] for row in results)
    lines = ["# Direct Pairwise Expressive Independence Local Jury", "", "## Completeness", "", f"- Valid pairs: {len(pairs)}", f"- Expected judgments: {len(pairs)*6}", f"- Completed: {len(all_rows)}", f"- Valid: {len(valid)}", f"- Invalid: {invalid}", f"- Retries: {retries}", "", "## Raw A/B Bias", "", "| Judge | A | B | A % | Original A/B | Swapped A/B |", "|---|---:|---:|---:|---:|---:|"]
    summaries = {}
    for judge in JUDGES:
        s = stats(raw[judge]); summaries[judge] = s
        lines.append(f"| {judge} | {s['A']} | {s['B']} | {100*s['A']/s['n'] if s['n'] else 0:.2f}% | {s['orders']['original']['A']}/{s['orders']['original']['B']} | {s['orders']['swapped']['A']}/{s['orders']['swapped']['B']} |")
    lines += ["", "## Order Agreement", "", "| Judge | Same | Different | Agreement % |", "|---|---:|---:|---:|"]
    for judge in JUDGES:
        same, different = order_stats(lookup, pairs, judge)
        lines.append(f"| {judge} | {same} | {different} | {100*same/(same+different) if same+different else 0:.2f}% |")
    lines += ["", "## Candidate Preference", "", "| Judge | Transformative | Conservative |", "|---|---:|---:|"]
    for judge in JUDGES:
        s = summaries[judge]; lines.append(f"| {judge} | {s['mapped']['TRANSFORMATIVE']} | {s['mapped']['CONSERVATIVE']} |")
    lines += ["", "## Stable Subset", "", "| Judge | Stable N | Stable T | Stable C |", "|---|---:|---:|---:|"]
    for judge in JUDGES:
        stable = []
        for pair in pairs:
            o, s = lookup.get((pair["sample_id"], judge, "original"), {}), lookup.get((pair["sample_id"], judge, "swapped"), {})
            if o.get("mapped_candidate") and o.get("mapped_candidate") == s.get("mapped_candidate"): stable.append(o)
        count = Counter(row["mapped_candidate"] for row in stable)
        lines.append(f"| {judge} | {len(stable)} | {count['TRANSFORMATIVE']} | {count['CONSERVATIVE']} |")
    lines += ["", "## Difference Strength", "", "| Judge | S1 | S2 | S3 | S4 | Mean | Median | >=3 | Strength exact agreement | Mean abs diff |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for judge in JUDGES:
        s = summaries[judge]
        exact = sum(bool(row[f"{judge}_strength_agreement"]) for row in results)
        diffs = [row[f"{judge}_strength_abs_diff"] for row in results if row.get(f"{judge}_strength_abs_diff") is not None]
        lines.append(f"| {judge} | {s['strength'][1]} | {s['strength'][2]} | {s['strength'][3]} | {s['strength'][4]} | {s['mean']:.3f} | {s['median']:.3f} | {s['strength'][3]+s['strength'][4]} ({100*(s['strength'][3]+s['strength'][4])/s['n'] if s['n'] else 0:.2f}%) | {exact}/{len(diffs)} | {statistics.fmean(diffs) if diffs else 0:.3f} |")
    lines += ["", "## Six Presentation-Level Votes", "", f"- Transformative winner: {winners['TRANSFORMATIVE']}", f"- Conservative winner: {winners['CONSERVATIVE']}", f"- 3:3 no consensus: {winners['no_consensus']}", f"- Incomplete: {winners['incomplete_judgments']}", ""]
    for pattern in ("6:0", "5:1", "4:2", "3:3"): lines.append(f"- {pattern}: {patterns[pattern]}")
    lines += ["", "## Overall Prompt vs Direct Prompt Diagnostic", "", "| Judge | Old A % | New A % | Old Order Agreement | New Order Agreement |", "|---|---:|---:|---:|---:|"]
    for judge in JUDGES:
        old_s, new_s = stats(old_raw[judge]), summaries[judge]
        old_same, old_diff = order_stats(old_lookup, pairs, judge)
        new_same, new_diff = order_stats(lookup, pairs, judge)
        lines.append(f"| {judge} | {100*old_s['A']/old_s['n'] if old_s['n'] else 0:.2f}% | {100*new_s['A']/new_s['n'] if new_s['n'] else 0:.2f}% | {100*old_same/(old_same+old_diff) if old_same+old_diff else 0:.2f}% | {100*new_same/(new_same+new_diff) if new_same+new_diff else 0:.2f}% |")
    lines += ["", "## GPT-4o Reference", "", f"- Existing GPT-4o direct results merged: {sum('gpt4o_winner' in row for row in results)}/{len(results)}.", "- GPT-4o was not called in this experiment; its single-order results are reference only."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT_COMPLETE rows={len(results)} report={REPORT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "diagnostic", "judge", "aggregate"))
    parser.add_argument("--judge", choices=JUDGES)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "diagnostic": ministral_sanity()
    elif args.command == "judge":
        if not args.judge: parser.error("--judge is required")
        run_judge(args.judge)
    else: aggregate_report()


if __name__ == "__main__":
    main()
