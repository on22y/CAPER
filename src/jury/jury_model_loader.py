#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT = Path("/home/user20250805/gayeon/HCLT2026")
ROOT = PROJECT / "work/hcx_v3_overall_jury6vote_pilot_v1"
SOURCE = PROJECT / "work/hcx_rewrite_prompt_v3_mode_margin_pilot30_v1/outputs/rewrites/pilot_v3_mode_controlled_rewrites_30x2.jsonl"
ATTACHMENT = Path("/home/user20250805/.codex/attachments/d644de71-6047-4cef-9b96-f623b319b3b0/pasted-text.txt")
MODEL_CONFIG = PROJECT / "work/hcx_pairwise_expand300_v1/configs/evaluator_models.json"
JURY_CODE = PROJECT / "experiments/copyright_jury/scripts"
sys.path.insert(0, str(JURY_CODE))
from run_judge_v5_document import decode_generated, input_device, load_model  # noqa: E402

PAIR_INPUT = ROOT / "data/jury_input_pairs.jsonl"
ASSIGNMENT = ROOT / "data/candidate_assignment.jsonl"
SYSTEM_PROMPT = ROOT / "prompts/overall_fact_preserving_expressive_forced_ab_system_prompt.txt"
USER_TEMPLATE = ROOT / "prompts/overall_forced_ab_user_template.txt"
AGGREGATE = ROOT / "outputs/aggregation/jury_6vote_results.jsonl"
REPORT = ROOT / "reports/jury_6vote_report.md"
SETUP = ROOT / "reports/setup_and_runtime.txt"
JUDGES = ("exaone", "qwen3", "ministral3")
ORDERS = ("original", "swapped")
CHOICE_RE = re.compile(r"^\s*([AB])\s*[.,!?;:]*\s*$", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_block(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Cannot uniquely extract block: {begin}")
    return text.split(begin, 1)[1].split(end, 1)[0].strip() + "\n"


def model_specs() -> dict[str, dict[str, Any]]:
    specs = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    for judge, spec in specs.items():
        spec["hf_id"] = spec["model_id"]
        if judge == "qwen3":
            spec.update({"family": "causal_lm", "enable_thinking": False, "trust_remote_code": True})
        elif judge == "exaone":
            spec.update({"family": "causal_lm", "trust_remote_code": True})
        elif judge == "ministral3":
            spec.update({"family": "ministral3", "trust_remote_code": False})
    return specs


def prepare() -> None:
    rows_by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in read_jsonl(SOURCE):
        if row.get("generation_status") == "generated" and row.get("rewrite_text"):
            rows_by_sample[row["pilot_sample_id"]][row["rewrite_mode"]] = row
    valid_ids = sorted(sid for sid, modes in rows_by_sample.items() if all(mode in modes for mode in ("CONSERVATIVE", "TRANSFORMATIVE")))
    pairs, assignments = [], []
    for index, sample_id in enumerate(valid_ids):
        modes = rows_by_sample[sample_id]
        a_identity = "CONSERVATIVE" if index % 2 == 0 else "TRANSFORMATIVE"
        b_identity = "TRANSFORMATIVE" if a_identity == "CONSERVATIVE" else "CONSERVATIVE"
        base = modes[a_identity]
        pair = {
            "sample_id": sample_id,
            "original_sample_id": base["original_sample_id"],
            "domain": base["domain"],
            "source_text": base["source_text"],
            "conservative_text": modes["CONSERVATIVE"]["rewrite_text"],
            "transformative_text": modes["TRANSFORMATIVE"]["rewrite_text"],
            "original_A_identity": a_identity,
            "original_B_identity": b_identity,
        }
        pairs.append(pair)
        assignments.append({
            "sample_id": sample_id,
            "assignment_index": index,
            "assignment_rule": "sample_id_sorted_alternating",
            "original_A_identity": a_identity,
            "original_B_identity": b_identity,
            "swapped_A_identity": b_identity,
            "swapped_B_identity": a_identity,
        })
    if len(pairs) != 28:
        print(f"WARNING expected prior-report 28 pairs but found {len(pairs)}", flush=True)

    request = ATTACHMENT.read_text(encoding="utf-8")
    system = extract_block(request, "---------------- BEGIN JURY SYSTEM PROMPT ----------------", "---------------- END JURY SYSTEM PROMPT ----------------")
    template = extract_block(request, "---------------- BEGIN USER TEMPLATE ----------------", "---------------- END USER TEMPLATE ----------------")
    SYSTEM_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT.write_text(system, encoding="utf-8")
    USER_TEMPLATE.write_text(template, encoding="utf-8")
    write_jsonl(PAIR_INPUT, pairs)
    write_jsonl(ASSIGNMENT, assignments)
    domains = Counter(row["domain"] for row in pairs)
    specs = model_specs()
    setup = {
        "input": str(SOURCE),
        "valid_pairs": len(pairs),
        "domain_breakdown": dict(domains),
        "expected_judgments": len(pairs) * 3 * 2,
        "judges": {judge: {key: specs[judge][key] for key in ("model_id", "revision", "local_path")} for judge in JUDGES},
        "forced_choices": ["A", "B"],
        "orders": list(ORDERS),
        "system_prompt": str(SYSTEM_PROMPT),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.read_bytes()).hexdigest(),
        "user_template": str(USER_TEMPLATE),
        "user_template_sha256": hashlib.sha256(USER_TEMPLATE.read_bytes()).hexdigest(),
        "decoding": {"do_sample": False, "max_new_tokens": 4, "max_retry": 1, "soft_logits": False},
        "aggregation": "six mapped presentation-level votes; 4-6 majority, 3:3 no_consensus; no tie-break",
        "eta": {
            "exaone": "roughly 30-60 seconds including load",
            "qwen3": "roughly 30-60 seconds including load",
            "ministral3": "roughly 45-90 seconds including FP8 load",
            "aggregation": "under 1 minute",
            "total": "roughly 3-6 minutes",
            "basis": "prior 2520 forward judgments completed in 479.5 sec; this run has 168 short forced-label generations, with conservative allowance for generate() and model loading",
        },
        "gpt_openai_api_calls": 0,
    }
    SETUP.parent.mkdir(parents=True, exist_ok=True)
    SETUP.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PREPARED pairs={len(pairs)} domains={dict(domains)} expected={len(pairs) * 6}", flush=True)


def text_for_identity(pair: dict, identity: str) -> str:
    return pair["conservative_text"] if identity == "CONSERVATIVE" else pair["transformative_text"]


def format_messages(adapter: Any, spec: dict, system: str, user: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt", "return_dict": True}
    if spec.get("enable_thinking") is False:
        kwargs["enable_thinking"] = False
    try:
        encoded = adapter.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError):
        rich = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        encoded = adapter.apply_chat_template(rich, **kwargs)
    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
    return encoded


def parse_choice(raw: str) -> str | None:
    match = CHOICE_RE.fullmatch(raw or "")
    return match.group(1).upper() if match else None


def run_judge(judge: str) -> None:
    specs = model_specs()
    spec = specs[judge]
    output = ROOT / f"outputs/raw/{judge}.jsonl"
    done = {(row["sample_id"], row["presentation"]) for row in read_jsonl(output)}
    system = SYSTEM_PROMPT.read_text(encoding="utf-8")
    template = USER_TEMPLATE.read_text(encoding="utf-8")
    pairs = read_jsonl(PAIR_INPUT)
    print(f"LOAD_START judge={judge} model={spec['model_id']} resume={len(done)}", flush=True)
    model, adapter, model_path, revision = load_model(spec)
    device = input_device(model)
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
            user = template.format(
                domain=pair["domain"], source_text=pair["source_text"],
                candidate_a=text_for_identity(pair, a_identity), candidate_b=text_for_identity(pair, b_identity),
            )
            attempts, choice, latency_total = [], None, 0.0
            for attempt in range(1, 3):
                started = time.perf_counter()
                encoded = format_messages(adapter, spec, system, user)
                encoded = {name: tensor.to(device) for name, tensor in encoded.items() if torch.is_tensor(tensor)}
                input_tokens = int(encoded["input_ids"].shape[-1])
                with torch.inference_mode():
                    generated_all = model.generate(
                        **encoded,
                        max_new_tokens=4,
                        do_sample=False,
                        pad_token_id=getattr(adapter, "eos_token_id", None) or getattr(getattr(adapter, "tokenizer", None), "eos_token_id", None),
                    )
                generated = generated_all[0, input_tokens:]
                raw = decode_generated(adapter, generated)
                latency = time.perf_counter() - started
                latency_total += latency
                choice = parse_choice(raw)
                attempts.append({"attempt": attempt, "raw_output": raw, "parsed_choice": choice, "latency_sec": latency})
                if choice is not None:
                    break
            mapped = a_identity if choice == "A" else b_identity if choice == "B" else None
            row = {
                "sample_id": pair["sample_id"], "domain": pair["domain"], "judge": judge,
                "presentation": presentation, "displayed_A_identity": a_identity, "displayed_B_identity": b_identity,
                "raw_output": attempts[-1]["raw_output"], "parsed_choice": choice, "mapped_candidate": mapped,
                "latency_sec": latency_total, "status": "valid" if choice else "invalid_output",
                "retry_count": len(attempts) - 1, "attempts": attempts,
                "input_tokens": input_tokens, "output_tokens": int(generated.shape[-1]),
                "model_id": spec["model_id"], "model_path": model_path, "model_revision": revision,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(output, row)
            done.add(key)
            print(f"JUDGED judge={judge} pair={pair_index}/{len(pairs)} order={presentation} status={row['status']} mapped={mapped}", flush=True)
    del model, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"JUDGE_COMPLETE judge={judge} rows={len(done)} output={output}", flush=True)


def aggregate() -> None:
    raw = {judge: read_jsonl(ROOT / f"outputs/raw/{judge}.jsonl") for judge in JUDGES}
    lookup = {(row["sample_id"], row["judge"], row["presentation"]): row for rows in raw.values() for row in rows}
    results = []
    for pair in read_jsonl(PAIR_INPUT):
        sid = pair["sample_id"]
        row = {"sample_id": sid, "domain": pair["domain"]}
        votes = []
        for judge in JUDGES:
            original = lookup.get((sid, judge, "original"), {})
            swapped = lookup.get((sid, judge, "swapped"), {})
            o_vote, s_vote = original.get("mapped_candidate"), swapped.get("mapped_candidate")
            row[f"{judge}_original"] = o_vote
            row[f"{judge}_swapped"] = s_vote
            row[f"{judge}_order_agreement"] = o_vote is not None and o_vote == s_vote
            votes.extend(vote for vote in (o_vote, s_vote) if vote in {"CONSERVATIVE", "TRANSFORMATIVE"})
        c_votes, t_votes = votes.count("CONSERVATIVE"), votes.count("TRANSFORMATIVE")
        if len(votes) < 6:
            winner, consensus = None, "incomplete_judgments"
        elif t_votes > c_votes:
            winner, consensus = "TRANSFORMATIVE", "majority"
        elif c_votes > t_votes:
            winner, consensus = "CONSERVATIVE", "majority"
        else:
            winner, consensus = None, "no_consensus"
        row.update({
            "conservative_votes": c_votes, "transformative_votes": t_votes,
            "vote_pattern": f"{max(c_votes, t_votes)}:{min(c_votes, t_votes)}" if len(votes) == 6 else f"incomplete_{len(votes)}/6",
            "vote_margin": abs(t_votes - c_votes), "final_winner": winner, "consensus": consensus,
            "valid_presentation_votes": len(votes),
        })
        results.append(row)
    write_jsonl(AGGREGATE, results)

    all_raw = [row for rows in raw.values() for row in rows]
    valid = [row for row in all_raw if row.get("status") == "valid"]
    retries = sum(int(row.get("retry_count", 0)) for row in all_raw)
    invalid = len(all_raw) - len(valid)
    winners = Counter(row["final_winner"] or row["consensus"] for row in results)
    patterns = Counter(row["vote_pattern"] for row in results)
    lines = [
        "# Overall Fact-Preserving Expressive Rewriting Jury", "", "## Completeness", "",
        f"- Valid pairs: {len(results)}", f"- Expected judgments: {len(results) * 6}",
        f"- Completed judgment rows: {len(all_raw)}", f"- Valid outputs: {len(valid)}",
        f"- Invalid outputs: {invalid}", f"- Retry count: {retries}", "", "## Judge-Level Preference", "",
        "| Judge | Transformative | Conservative | Invalid |", "|---|---:|---:|---:|",
    ]
    for judge in JUDGES:
        rows = raw[judge]; count = Counter(row.get("mapped_candidate") for row in rows)
        lines.append(f"| {judge} | {count['TRANSFORMATIVE']} | {count['CONSERVATIVE']} | {count[None]} |")
    lines += ["", "## Presentation-Order Agreement", "", "| Judge | Same | Different | Agreement % |", "|---|---:|---:|---:|"]
    total_same = total_pairs = 0
    for judge in JUDGES:
        flags = [row[f"{judge}_order_agreement"] for row in results if row[f"{judge}_original"] and row[f"{judge}_swapped"]]
        same = sum(flags); different = len(flags) - same; total_same += same; total_pairs += len(flags)
        lines.append(f"| {judge} | {same} | {different} | {100 * same / len(flags) if flags else 0:.2f}% |")
    lines += [f"", f"Overall presentation-order agreement: {total_same}/{total_pairs} ({100 * total_same / total_pairs if total_pairs else 0:.2f}%)", "", "## Final Six-Vote Majority", ""]
    for label in ("TRANSFORMATIVE", "CONSERVATIVE", "no_consensus", "incomplete_judgments"):
        lines.append(f"- {label}: {winners[label]}/{len(results)} ({100 * winners[label] / len(results) if results else 0:.2f}%)")
    lines += ["", "Vote patterns:"]
    for pattern in ("6:0", "5:1", "4:2", "3:3"):
        lines.append(f"- {pattern}: {patterns[pattern]}")
    lines += ["", "## Domain Breakdown", "", "| Domain | N | T winner | C winner | No consensus |", "|---|---:|---:|---:|---:|"]
    for domain in ("news", "petition", "sns"):
        rows = [row for row in results if row["domain"] == domain]; counts = Counter(row["final_winner"] or row["consensus"] for row in rows)
        lines.append(f"| {domain} | {len(rows)} | {counts['TRANSFORMATIVE']} | {counts['CONSERVATIVE']} | {counts['no_consensus'] + counts['incomplete_judgments']} |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AGGREGATION_COMPLETE rows={len(results)} report={REPORT}", flush=True)


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
