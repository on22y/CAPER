#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_command_ministral_v3"
CANONICAL_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_v1"
COMMON_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_granite_command_v1"
COMMON_SCRIPT = COMMON_ROOT / "scripts/run_granite_command_jury28.py"
LOADER_SCRIPT = PROJECT / "work/hcx_v3_shortprompt_overall_jury6vote_pilot_v1/scripts/run_shortprompt_jury.py"
MODEL_SPECIFIC_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_model_specific_v1"
V2_ROOT = PROJECT / "work/hcx_v3_direct_expr_jury28_command_ministral_v2"
PAIR_INPUT = ROOT / "data/jury_input_pairs_28.jsonl"
ASSIGNMENT = ROOT / "data/candidate_assignment.jsonl"
USER_TEMPLATE = ROOT / "prompts/direct_expressive_user_template.txt"
PROMPTS = {
    "command_r7b": ROOT / "prompts/command_r7b_independent_checklist_v3.txt",
    "ministral3": ROOT / "prompts/ministral_positive_hierarchical_expressive_v3.txt",
}
PROMPT_VERSIONS = {
    "command_r7b": "command_r7b_independent_checklist_v3",
    "ministral3": "ministral_positive_hierarchical_expressive_v3",
}
DIAGNOSTIC = ROOT / "diagnostics/ministral_input_sanity.json"
PAIRWISE = ROOT / "outputs/aggregation/command_ministral_v3_pairwise_results.jsonl"
REPORT = ROOT / "reports/command_ministral_jury28_v3_report.md"
SETUP = ROOT / "reports/setup_and_runtime.txt"
GPT_REFERENCE = CANONICAL_ROOT / "outputs/aggregation/direct_expr_6vote_results.jsonl"
PREVIOUS_COMMAND = V2_ROOT / "outputs/aggregation/command_ministral_pairwise_results.jsonl"
PREVIOUS_MINISTRAL = MODEL_SPECIFIC_ROOT / "outputs/aggregation/model_specific_pairwise_results.jsonl"
JUDGES = ("command_r7b", "ministral3")
ORDERS = ("original", "swapped")
SPECS = {
    "command_r7b": {
        "model_id": "CohereLabs/c4ai-command-r7b-12-2024",
        "hf_id": "CohereLabs/c4ai-command-r7b-12-2024",
        "local_path": str(PROJECT / "models/judges/c4ai-command-r7b-12-2024"),
        "revision": "4f3d0aa6856e322f2f4480fe65420d5d53d297b8",
        "family": "causal_lm",
        "trust_remote_code": False,
    },
    "ministral3": {
        "model_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "hf_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "local_path": str(PROJECT / "models/judges/Ministral-3-8B-Instruct-2512"),
        "revision": "5b26027e7b19eeb4b7352e1fed3926375dd2cb4d",
        "family": "ministral3",
        "trust_remote_code": False,
    },
}


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = import_module("command_ministral_common", COMMON_SCRIPT)
loader = import_module("command_ministral_fixed_loader", LOADER_SCRIPT)
common.ROOT = ROOT
common.PAIR_INPUT = PAIR_INPUT
common.ASSIGNMENT = ASSIGNMENT
common.USER_TEMPLATE = USER_TEMPLATE
common.PROMPTS = PROMPTS
common.PROMPT_VERSIONS = PROMPT_VERSIONS
common.JUDGES = JUDGES
common.SPECS = SPECS
common.base.format_messages = loader.fixed_format_messages
original_load_model = common.base.load_model


def load_model_with_adapter_padding(spec: dict[str, Any]):
    loaded = original_load_model(spec)
    tokenizer = loaded[1]
    if not hasattr(tokenizer, "pad_token_id"):
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)
    return loaded


common.base.load_model = load_model_with_adapter_padding


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return common.read_jsonl(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    common.write_jsonl(path, rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join((text or "").split())


def prepare() -> None:
    canonical_pairs = read_jsonl(CANONICAL_ROOT / "data/jury_input_pairs.jsonl")
    canonical_assignments = read_jsonl(CANONICAL_ROOT / "data/candidate_assignment.jsonl")
    pairs = read_jsonl(PAIR_INPUT)
    assignments = read_jsonl(ASSIGNMENT)
    domains = Counter(row["domain"] for row in pairs)
    if pairs != canonical_pairs or assignments != canonical_assignments:
        raise RuntimeError("V3 pair or assignment artifact differs from canonical Direct Jury input")
    if len(pairs) != 28 or domains != Counter({"news": 9, "petition": 9, "sns": 10}):
        raise RuntimeError(f"Unexpected input: pairs={len(pairs)} domains={dict(domains)}")
    if [row["sample_id"] for row in pairs] != [row["sample_id"] for row in assignments]:
        raise RuntimeError("Pair and assignment IDs differ")
    for judge, spec in SPECS.items():
        model_path = Path(spec["local_path"])
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise RuntimeError(f"Missing local model: {judge} {model_path}")
        if not list(model_path.glob("*.safetensors")):
            raise RuntimeError(f"No safetensors found: {judge} {model_path}")
    command_tokenizer = AutoTokenizer.from_pretrained(SPECS["command_r7b"]["local_path"], local_files_only=True)
    if not getattr(command_tokenizer, "chat_template", None):
        raise RuntimeError("Command official chat template missing")
    pair = pairs[0]
    encoded = loader.fixed_format_messages(
        command_tokenizer,
        SPECS["command_r7b"],
        PROMPTS["command_r7b"].read_text(encoding="utf-8"),
        common.make_user(pair, pair["original_A_identity"], pair["original_B_identity"]),
    )
    setup = {
        "reused_command_runner": str(COMMON_SCRIPT),
        "reused_fixed_ministral_formatter": str(LOADER_SCRIPT),
        "canonical_input": str(CANONICAL_ROOT / "data/jury_input_pairs.jsonl"),
        "canonical_assignment": str(CANONICAL_ROOT / "data/candidate_assignment.jsonl"),
        "pairs": 28,
        "domains": dict(domains),
        "models": SPECS,
        "prompts": {judge: {"path": str(PROMPTS[judge]), "sha256": sha256(PROMPTS[judge])} for judge in JUDGES},
        "user_template": {"path": str(USER_TEMPLATE), "sha256": sha256(USER_TEMPLATE)},
        "command_official_chat_template": True,
        "command_first_input_tokens": int(encoded["input_ids"].shape[-1]),
        "ministral_input": "combined_system_and_user_single_UserMessage",
        "decoding": {"dtype": "bfloat16", "do_sample": False, "max_new_tokens": 4, "use_cache": True, "max_retry": 1, "strength": False},
        "expected_new_judgments": 112,
        "oa_pass": "stable >= 17/28 (60.71%)",
        "qwen_inference": 0,
        "granite_inference": 0,
        "exaone_inference": 0,
        "gpt_openai_calls": 0,
        "eta": "roughly 3-6 minutes including sequential model loads and aggregation",
    }
    SETUP.parent.mkdir(parents=True, exist_ok=True)
    SETUP.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PREFLIGHT_OK pairs=28 domains={dict(domains)} expected=112 qwen=0 granite=0 gpt=0", flush=True)


def ministral_sanity() -> None:
    from run_judge_v5_document import MistralCommonAdapter

    pairs = read_jsonl(PAIR_INPUT)
    ranked = sorted(pairs, key=lambda row: len(row["source_text"]) + len(row["conservative_text"]) + len(row["transformative_text"]))
    pair = ranked[len(ranked) // 2]
    spec = SPECS["ministral3"]
    adapter = MistralCommonAdapter(spec["local_path"], spec["hf_id"])
    a_identity, b_identity = pair["original_A_identity"], pair["original_B_identity"]
    a_text = common.identity_text(pair, a_identity)
    b_text = common.identity_text(pair, b_identity)
    user = common.make_user(pair, a_identity, b_identity)
    encoded = loader.fixed_format_messages(adapter, spec, PROMPTS["ministral3"].read_text(encoding="utf-8"), user)
    decoded = adapter.decode(encoded["input_ids"][0])
    decoded_norm = normalized(decoded)
    checks = {
        "source_marker": "[SOURCE]" in decoded,
        "candidate_a_marker": "[CANDIDATE A]" in decoded,
        "candidate_b_marker": "[CANDIDATE B]" in decoded,
        "source_content": normalized(pair["source_text"]) in decoded_norm,
        "candidate_a_full": normalized(a_text) in decoded_norm,
        "candidate_b_full": normalized(b_text) in decoded_norm,
    }
    result = {
        "sample_id": pair["sample_id"],
        "input_construction": "combined_system_and_user_single_UserMessage",
        "input_tokens": int(encoded["input_ids"].shape[-1]),
        "non_padding_tokens": int(encoded["attention_mask"].sum()),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }
    DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MINISTRAL_SANITY status={result['status']} tokens={result['input_tokens']} checks={checks}", flush=True)
    if result["status"] != "passed":
        raise RuntimeError("Ministral input sanity failed")


def latest_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[(row["sample_id"], row["presentation"])] = row
    return latest


def aggregate() -> None:
    pairs = read_jsonl(PAIR_INPUT)
    raw = {judge: latest_rows(ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    gpt_by_id = {row["sample_id"]: row.get("gpt4o_winner") for row in read_jsonl(GPT_REFERENCE)}
    previous_command = {row["sample_id"]: row for row in read_jsonl(PREVIOUS_COMMAND)}
    previous_ministral = {row["sample_id"]: row for row in read_jsonl(PREVIOUS_MINISTRAL)}
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
        result["command_r7b_previous_stable"] = bool(previous_command.get(sid, {}).get("command_r7b_stable"))
        result["ministral3_previous_stable"] = bool(previous_ministral.get(sid, {}).get("ministral3_stable"))
        pairwise.append(result)
    write_jsonl(PAIRWISE, pairwise)

    all_latest = [row for rows in raw.values() for row in rows.values()]
    valid = [row for row in all_latest if row.get("status") == "valid"]
    retries = sum(int(row.get("retry_count", 0)) for row in all_latest)
    stats: dict[str, dict[str, Any]] = {}
    for judge in JUDGES:
        complete = [row for row in pairwise if row[f"{judge}_original"] and row[f"{judge}_swapped"]]
        stable = [row for row in complete if row[f"{judge}_stable"]]
        stats[judge] = {
            "complete": complete,
            "stable": stable,
            "unstable": len(complete) - len(stable),
            "oa": 100 * len(stable) / len(complete) if complete else 0.0,
            "pass": len(complete) == 28 and len(stable) >= 17,
        }
    previous_best = {"command_r7b": (11, 39.29), "ministral3": (16, 57.14)}
    lines = [
        "# Command R7B and Ministral3 V3 Prompt Jury28",
        "",
        "## A. Completeness",
        "",
        "- Pair N: 28",
        "- Expected judgments: 112",
        f"- Completed unique judgments: {sum(len(rows) for rows in raw.values())}",
        f"- Valid: {len(valid)}",
        f"- Invalid: {len(all_latest)-len(valid)}",
        f"- Retries: {retries}",
        "- Qwen inference: 0",
        "- Granite inference: 0",
        "- GPT/OpenAI calls: 0",
        "",
        "## B. Primary Result",
        "",
        "| Judge | Stable | Unstable | OA % | PASS/FAIL |",
        "|---|---:|---:|---:|---|",
    ]
    for judge in JUDGES:
        item = stats[judge]
        lines.append(f"| {judge} | {len(item['stable'])} | {item['unstable']} | {item['oa']:.2f}% | {'PASS' if item['pass'] else 'FAIL'} |")
    lines += [
        "", "## C. Version History", "",
        "| Judge | Version | Stable | OA % |", "|---|---|---:|---:|",
        "| command_r7b | V1 | 9/28 | 32.14% |",
        "| command_r7b | V2 | 11/28 | 39.29% |",
        f"| command_r7b | V3 | {len(stats['command_r7b']['stable'])}/28 | {stats['command_r7b']['oa']:.2f}% |",
        "| ministral3 | Direct | 15/28 | 53.57% |",
        "| ministral3 | Model-specific Gate V1 | 16/28 | 57.14% |",
        "| ministral3 | V2 | 1/28 | 3.57% |",
        f"| ministral3 | V3 | {len(stats['ministral3']['stable'])}/28 | {stats['ministral3']['oa']:.2f}% |",
        "", "## D. Previous Best vs V3", "",
        "| Judge | Previous Best OA | V3 OA | Delta pp | Stable Delta N |", "|---|---:|---:|---:|---:|",
    ]
    for judge in JUDGES:
        old_n, old_oa = previous_best[judge]
        new_n, new_oa = len(stats[judge]["stable"]), stats[judge]["oa"]
        lines.append(f"| {judge} | {old_oa:.2f}% | {new_oa:.2f}% | {new_oa-old_oa:+.2f} | {new_n-old_n:+d} |")
    lines += ["", "## E. Position Bias", "", "| Judge | A | B | A % | B % |", "|---|---:|---:|---:|---:|"]
    for judge in JUDGES:
        rows = [row for row in raw[judge].values() if row.get("status") == "valid"]
        choices = Counter(row.get("parsed_choice") for row in rows)
        n = sum(choices.values())
        lines.append(f"| {judge} | {choices['A']} | {choices['B']} | {100*choices['A']/n if n else 0:.2f}% | {100*choices['B']/n if n else 0:.2f}% |")
        if n and max(choices.values()) / n >= 0.90:
            lines.append(f"\nWarning: {judge} shows severe label collapse (>=90%).")
    lines += ["", "## F. Candidate Preference", "", "| Judge | T | C | T % |", "|---|---:|---:|---:|"]
    for judge in JUDGES:
        rows = [row for row in raw[judge].values() if row.get("status") == "valid"]
        mapped = Counter(row.get("mapped_candidate") for row in rows)
        n = mapped["TRANSFORMATIVE"] + mapped["CONSERVATIVE"]
        lines.append(f"| {judge} | {mapped['TRANSFORMATIVE']} | {mapped['CONSERVATIVE']} | {100*mapped['TRANSFORMATIVE']/n if n else 0:.2f}% |")
    lines += ["", "## G. Stable Subset", "", "| Judge | Stable N | Stable T | Stable C |", "|---|---:|---:|---:|"]
    for judge in JUDGES:
        stable = stats[judge]["stable"]
        selected = Counter(row[f"{judge}_original"] for row in stable)
        lines.append(f"| {judge} | {len(stable)} | {selected['TRANSFORMATIVE']} | {selected['CONSERVATIVE']} |")
    lines += [
        "",
        "## H. Frozen References",
        "",
        "| Judge | Prompt Status | Stable | OA | PASS |",
        "|---|---|---:|---:|---|",
        "| Qwen3 | Frozen | 21/28 | 75.00% | PASS |",
        "| Granite | Frozen | 17/28 | 60.71% | PASS |",
        f"| Ministral3 V3 | New | {len(stats['ministral3']['stable'])}/28 | {stats['ministral3']['oa']:.2f}% | {'PASS' if stats['ministral3']['pass'] else 'FAIL'} |",
        f"| Command R7B V3 | New | {len(stats['command_r7b']['stable'])}/28 | {stats['command_r7b']['oa']:.2f}% | {'PASS' if stats['command_r7b']['pass'] else 'FAIL'} |",
        "",
        "## I. GPT Stable-subset Agreement",
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
        "OA is the primary prompt-selection metric. Position bias and GPT agreement are secondary; T/C preference is descriptive only.",
        "This is the final prompt-development round; no V4 prompt was generated or executed.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AGGREGATION_COMPLETE pairs=28 valid={len(valid)}/112 report={REPORT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "sanity", "judge", "aggregate"))
    parser.add_argument("--judge", choices=JUDGES)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "sanity":
        ministral_sanity()
    elif args.command == "judge":
        if not args.judge:
            parser.error("--judge is required")
        common.run_judge(args.judge)
    else:
        aggregate()


if __name__ == "__main__":
    main()
