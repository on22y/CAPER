#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "work/hcx_v3_shortprompt_overall_jury6vote_pilot_v1"
OLD_ROOT = PROJECT / "work/hcx_v3_overall_jury6vote_pilot_v1"
OLD_SCRIPT = OLD_ROOT / "scripts/run_overall_jury6vote.py"
SOURCE = PROJECT / "work/hcx_rewrite_prompt_v3_mode_margin_pilot30_v1/outputs/rewrites/pilot_v3_mode_controlled_rewrites_30x2.jsonl"
ATTACHMENT = Path("/home/user20250805/.codex/attachments/ea6d43b2-099d-4875-a225-ec854b97608d/pasted-text.txt")
PAIR_INPUT = ROOT / "data/jury_input_pairs.jsonl"
ASSIGNMENT = ROOT / "data/candidate_assignment.jsonl"
SYSTEM_PROMPT = ROOT / "prompts/short_overall_forced_ab_system_prompt.txt"
USER_TEMPLATE = ROOT / "prompts/short_overall_forced_ab_user_template.txt"
DIAGNOSTIC = ROOT / "diagnostics/ministral_input_diagnostic.txt"
AGGREGATE = ROOT / "outputs/aggregation/jury_6vote_results.jsonl"
REPORT = ROOT / "reports/shortprompt_jury_report.md"
SETUP = ROOT / "reports/setup_and_runtime.txt"
JUDGES = ("exaone", "qwen3", "ministral3")


def load_base_module():
    spec = importlib.util.spec_from_file_location("previous_overall_jury", OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {OLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
old_format_messages = base.format_messages


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def extract_block(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Cannot uniquely extract {begin}")
    return text.split(begin, 1)[1].split(end, 1)[0].strip() + "\n"


def patch_base() -> None:
    base.ROOT = ROOT
    base.SOURCE = SOURCE
    base.ATTACHMENT = ATTACHMENT
    base.PAIR_INPUT = PAIR_INPUT
    base.ASSIGNMENT = ASSIGNMENT
    base.SYSTEM_PROMPT = SYSTEM_PROMPT
    base.USER_TEMPLATE = USER_TEMPLATE
    base.AGGREGATE = AGGREGATE
    base.REPORT = REPORT
    base.SETUP = SETUP
    base.format_messages = fixed_format_messages


def fixed_format_messages(adapter: Any, spec: dict, system: str, user: str) -> dict[str, torch.Tensor]:
    if spec.get("family") == "ministral3":
        # MistralCommonAdapter consumes only messages[0], so preserve both prompts in one UserMessage.
        combined = system.rstrip() + "\n\n" + user.lstrip()
        return adapter.apply_chat_template(
            [{"role": "user", "content": combined}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if spec.get("enable_thinking") is False:
        kwargs["enable_thinking"] = False
    try:
        encoded = adapter.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError):
        encoded = adapter.apply_chat_template(
            [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ],
            **kwargs,
        )
    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
    return encoded


def prepare() -> None:
    valid_modes: dict[str, set[str]] = {}
    for row in read_jsonl(SOURCE):
        if row.get("generation_status") == "generated" and row.get("rewrite_text"):
            valid_modes.setdefault(row["pilot_sample_id"], set()).add(row["rewrite_mode"])
    valid_ids = sorted(sid for sid, modes in valid_modes.items() if modes == {"CONSERVATIVE", "TRANSFORMATIVE"})
    old_pairs = read_jsonl(OLD_ROOT / "data/jury_input_pairs.jsonl")
    old_assignments = read_jsonl(OLD_ROOT / "data/candidate_assignment.jsonl")
    if [row["sample_id"] for row in old_pairs] != valid_ids:
        raise RuntimeError("Previous canonical pair input does not match current valid v3 pairs")
    if [row["sample_id"] for row in old_assignments] != valid_ids:
        raise RuntimeError("Previous counterbalanced assignment does not match current valid v3 pairs")
    write_jsonl(PAIR_INPUT, old_pairs)
    write_jsonl(ASSIGNMENT, old_assignments)

    request = ATTACHMENT.read_text(encoding="utf-8")
    system = extract_block(request, "---------------- BEGIN SYSTEM PROMPT ----------------", "---------------- END SYSTEM PROMPT ----------------")
    template = extract_block(request, "---------------- BEGIN USER TEMPLATE ----------------", "---------------- END USER TEMPLATE ----------------")
    SYSTEM_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT.write_text(system, encoding="utf-8")
    USER_TEMPLATE.write_text(template, encoding="utf-8")
    domains = Counter(row["domain"] for row in old_pairs)
    specs = base.model_specs()
    setup = {
        "input": str(SOURCE),
        "valid_pairs": len(old_pairs),
        "domain_breakdown": dict(domains),
        "candidate_assignment": str(OLD_ROOT / "data/candidate_assignment.jsonl"),
        "expected_judgments": len(old_pairs) * 6,
        "models": {judge: specs[judge] for judge in JUDGES},
        "prompt_paths": {"system": str(SYSTEM_PROMPT), "user": str(USER_TEMPLATE)},
        "ministral_fix": "combine system and user into one UserMessage because canonical adapter consumes messages[0] only",
        "decoding": {"do_sample": False, "max_new_tokens": 4, "max_retry": 1},
        "gpt_openai_api_calls": 0,
        "eta": "roughly 3-6 minutes, based on previous local 2520-request/479.5-sec run with generation/model-load allowance",
    }
    SETUP.parent.mkdir(parents=True, exist_ok=True)
    SETUP.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PREPARED pairs={len(old_pairs)} domains={dict(domains)} expected={len(old_pairs) * 6}", flush=True)


def identity_text(pair: dict, identity: str) -> str:
    return pair["conservative_text"] if identity == "CONSERVATIVE" else pair["transformative_text"]


def normalize(text: str) -> str:
    return " ".join((text or "").split())


def inspect_decoded(decoded: str, source: str, candidate_a: str, candidate_b: str) -> dict:
    decoded_norm = normalize(decoded)
    return {
        "source_start_present": normalize(source)[:80] in decoded_norm,
        "candidate_a_start_present": normalize(candidate_a)[:80] in decoded_norm,
        "candidate_b_start_present": normalize(candidate_b)[:80] in decoded_norm,
        "candidate_b_end_present": normalize(candidate_b)[-80:] in decoded_norm,
    }


def diagnostic() -> None:
    pairs = read_jsonl(PAIR_INPUT)
    ranked = sorted(pairs, key=lambda row: len(row["source_text"]) + len(row["conservative_text"]) + len(row["transformative_text"]))
    selected = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    spec = base.model_specs()["ministral3"]
    from run_judge_v5_document import MistralCommonAdapter
    adapter = MistralCommonAdapter(spec["local_path"], spec["hf_id"])
    system = SYSTEM_PROMPT.read_text(encoding="utf-8")
    template = USER_TEMPLATE.read_text(encoding="utf-8")
    rows = []
    for size_label, pair in zip(("short", "medium", "long"), selected):
        a_identity, b_identity = pair["original_A_identity"], pair["original_B_identity"]
        candidate_a, candidate_b = identity_text(pair, a_identity), identity_text(pair, b_identity)
        user = template.format(domain=pair["domain"], source_text=pair["source_text"], candidate_a=candidate_a, candidate_b=candidate_b)
        rendered = system.rstrip() + "\n\n" + user.lstrip()
        old_encoded = old_format_messages(adapter, spec, system, user)
        fixed_encoded = fixed_format_messages(adapter, spec, system, user)
        old_decoded = adapter.decode(old_encoded["input_ids"][0])
        fixed_decoded = adapter.decode(fixed_encoded["input_ids"][0])
        rows.append({
            "size_label": size_label,
            "sample_id": pair["sample_id"],
            "raw_character_lengths": {"source": len(pair["source_text"]), "candidate_a": len(candidate_a), "candidate_b": len(candidate_b)},
            "rendered_prompt": {
                "total_characters": len(rendered),
                "beginning": rendered[:180].replace("\n", "\\n"),
                "ending": rendered[-180:].replace("\n", "\\n"),
                "system_included": system.strip() in rendered,
                "user_included": user.strip() in rendered,
                "source_marker": "[SOURCE]" in rendered,
                "candidate_a_marker": "[CANDIDATE A]" in rendered,
                "candidate_b_marker": "[CANDIDATE B]" in rendered,
            },
            "old_tokenization": {
                "input_ids_shape": int(old_encoded["input_ids"].shape[-1]),
                "non_padding_tokens": int(old_encoded["attention_mask"].sum()),
                "padding": "none; attention mask is all ones",
                "truncation": False,
                "decoded_content": inspect_decoded(old_decoded, pair["source_text"], candidate_a, candidate_b),
            },
            "fixed_tokenization": {
                "pre_padding_actual_tokens": int(fixed_encoded["attention_mask"].sum()),
                "input_ids_shape": int(fixed_encoded["input_ids"].shape[-1]),
                "non_padding_tokens": int(fixed_encoded["attention_mask"].sum()),
                "padding": "none; attention mask is all ones",
                "truncation": False,
                "tokenizer_max_length": None,
                "decoded_content": inspect_decoded(fixed_decoded, pair["source_text"], candidate_a, candidate_b),
            },
        })
    fixed_ok = all(all(row["fixed_tokenization"]["decoded_content"].values()) for row in rows)
    old_missing = all(not any(row["old_tokenization"]["decoded_content"].values()) for row in rows)
    classification = "CASE B. Actual truncation/input construction bug" if fixed_ok and old_missing else "CASE C. Cause not fully isolated"
    result = {
        "classification": classification,
        "cause": "MistralCommonAdapter.apply_chat_template uses only messages[0]; the previous two-message call encoded only the system prompt, producing constant 1202 tokens and omitting SOURCE/A/B.",
        "fix": "System and user prompts are combined into one UserMessage for Ministral only. No truncation or padding is requested.",
        "fixed_candidate_content_verified": fixed_ok,
        "proceed_with_rerun": fixed_ok,
        "samples": rows,
    }
    DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DIAGNOSTIC_COMPLETE classification={classification} fixed_content_verified={fixed_ok}", flush=True)
    if not fixed_ok:
        raise RuntimeError("Fixed Ministral tokenization did not preserve SOURCE/A/B; aborting inference")


def enrich_ministral_output() -> None:
    path = ROOT / "outputs/raw/ministral3.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row["non_padding_input_tokens"] = row["input_tokens"]
        row["input_padding"] = "none"
        row["input_construction"] = "combined_system_and_user_single_UserMessage"
    write_jsonl(path, rows)


def raw_stats(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "valid"]
    choices = Counter(row.get("parsed_choice") for row in valid)
    presentations = {}
    for presentation in ("original", "swapped"):
        subset = [row for row in valid if row["presentation"] == presentation]
        count = Counter(row["parsed_choice"] for row in subset)
        presentations[presentation] = {"A": count["A"], "B": count["B"]}
    mapped = Counter(row.get("mapped_candidate") for row in valid)
    return {"n": len(valid), "A": choices["A"], "B": choices["B"], "presentations": presentations, "mapped": mapped}


def order_stats(aggregate_rows: list[dict], judge: str) -> tuple[int, int, float]:
    flags = [row[f"{judge}_order_agreement"] for row in aggregate_rows if row.get(f"{judge}_original") and row.get(f"{judge}_swapped")]
    same = sum(flags); different = len(flags) - same
    return same, different, 100 * same / len(flags) if flags else 0.0


def aggregate_and_report() -> None:
    patch_base()
    base.aggregate()
    results = read_jsonl(AGGREGATE)
    new_raw = {judge: read_jsonl(ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    old_raw = {judge: read_jsonl(OLD_ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    old_agg = read_jsonl(OLD_ROOT / "outputs/aggregation/jury_6vote_results.jsonl")
    completed = sum(len(rows) for rows in new_raw.values())
    invalid = sum(row.get("status") != "valid" for rows in new_raw.values() for row in rows)
    retries = sum(int(row.get("retry_count", 0)) for rows in new_raw.values() for row in rows)
    winners = Counter(row["final_winner"] or row["consensus"] for row in results)
    patterns = Counter(row["vote_pattern"] for row in results)
    diagnostic_data = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    lines = [
        "# Short-Prompt Overall Jury Rerun", "", "## Ministral Diagnostic", "",
        f"- Classification: {diagnostic_data['classification']}",
        f"- Cause: {diagnostic_data['cause']}",
        f"- SOURCE/A/B included after fix: {diagnostic_data['fixed_candidate_content_verified']}",
        f"- Fix: {diagnostic_data['fix']}", "", "## Completeness", "",
        f"- Valid pairs: {len(results)}", f"- Expected judgments: {len(results) * 6}",
        f"- Completed: {completed}", f"- Invalid: {invalid}", f"- Retry: {retries}", "",
        "## Raw Label Distribution", "", "| Judge | Raw A | Raw B | A % | Original A/B | Swapped A/B |", "|---|---:|---:|---:|---:|---:|",
    ]
    new_stats = {}
    for judge in JUDGES:
        stat = raw_stats(new_raw[judge]); new_stats[judge] = stat
        lines.append(f"| {judge} | {stat['A']} | {stat['B']} | {100 * stat['A'] / stat['n'] if stat['n'] else 0:.2f}% | {stat['presentations']['original']['A']}/{stat['presentations']['original']['B']} | {stat['presentations']['swapped']['A']}/{stat['presentations']['swapped']['B']} |")
    lines += ["", "## Order Agreement", "", "| Judge | Same | Different | Agreement % |", "|---|---:|---:|---:|"]
    for judge in JUDGES:
        same, different, pct = order_stats(results, judge)
        lines.append(f"| {judge} | {same} | {different} | {pct:.2f}% |")
    lines += ["", "## Candidate Preference", "", "| Judge | Transformative | Conservative |", "|---|---:|---:|"]
    for judge in JUDGES:
        stat = new_stats[judge]
        lines.append(f"| {judge} | {stat['mapped']['TRANSFORMATIVE']} | {stat['mapped']['CONSERVATIVE']} |")
    lines += ["", "## Final Aggregation", "", f"- Transformative winner: {winners['TRANSFORMATIVE']}", f"- Conservative winner: {winners['CONSERVATIVE']}", f"- 3:3 no consensus: {winners['no_consensus']}"]
    for pattern in ("6:0", "5:1", "4:2", "3:3"):
        lines.append(f"- {pattern}: {patterns[pattern]}")
    lines += ["", "## Previous vs Short Prompt", "", "| Judge | Old A % | New A % | Old Order Agreement | New Order Agreement |", "|---|---:|---:|---:|---:|"]
    for judge in JUDGES:
        old_stat = raw_stats(old_raw[judge]); new_stat = new_stats[judge]
        _, _, old_pct = order_stats(old_agg, judge); _, _, new_pct = order_stats(results, judge)
        lines.append(f"| {judge} | {100 * old_stat['A'] / old_stat['n'] if old_stat['n'] else 0:.2f}% | {100 * new_stat['A'] / new_stat['n'] if new_stat['n'] else 0:.2f}% | {old_pct:.2f}% | {new_pct:.2f}% |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT_COMPLETE report={REPORT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "diagnostic", "judge", "aggregate"))
    parser.add_argument("--judge", choices=JUDGES)
    args = parser.parse_args()
    patch_base()
    if args.command == "prepare":
        prepare()
    elif args.command == "diagnostic":
        diagnostic()
    elif args.command == "judge":
        if not args.judge:
            parser.error("--judge is required")
        base.run_judge(args.judge)
        if args.judge == "ministral3":
            enrich_ministral_output()
    else:
        aggregate_and_report()


if __name__ == "__main__":
    main()
