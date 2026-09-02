#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from common import MODEL, PROMPT, PROMPT_VERSION, ROOT, append_jsonl, canonical_rows, read_jsonl

ENV = Path(os.environ.get("ENV_FILE", Path(__file__).resolve().parents[2] / ".env"))


def user_prompt(domain: str, source: str, candidate_a: str, candidate_b: str) -> str:
    return f"[DOMAIN]\n{domain}\n\n[SOURCE]\n{source}\n\n[Candidate A]\n{candidate_a}\n\n[Candidate B]\n{candidate_b}"


def usage_dict(response) -> dict:
    usage = response.usage.model_dump() if getattr(response, "usage", None) is not None else {}
    details = usage.get("input_tokens_details") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-type", choices=("validation", "final"), required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ENV)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    rows = canonical_rows()
    if args.limit:
        rows = rows[:args.limit]
    output = ROOT / f"gpt4o/raw/substantive_similarity_{args.run_type}.jsonl"
    usage_output = ROOT / "gpt4o/raw/substantive_similarity_api_usage.jsonl"
    existing = read_jsonl(output) if output.exists() else []
    done = {(r["sample_id"], r["order"]) for r in existing}
    if len(done) != len(existing):
        raise RuntimeError("duplicate checkpoint keys")
    schema = {
        "type": "object",
        "properties": {
            "candidate_a_similarity": {"type": "integer", "minimum": 1, "maximum": 4},
            "candidate_b_similarity": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "required": ["candidate_a_similarity", "candidate_b_similarity"],
        "additionalProperties": False,
    }
    instructions = PROMPT.read_text(encoding="utf-8")
    forbidden = ("chosen", "rejected")
    assert not any(word in instructions.lower() for word in forbidden)
    client = OpenAI()
    for row in rows:
        for order in ("original", "swapped"):
            key = (row["sample_id"], order)
            if key in done:
                continue
            if order == "original":
                role_a, role_b, a, b = "chosen", "rejected", row["chosen"], row["rejected"]
            else:
                role_a, role_b, a, b = "rejected", "chosen", row["rejected"], row["chosen"]
            prompt = user_prompt(row["domain"], row["source_text"], a, b)
            assert "chosen" not in prompt.lower() and "rejected" not in prompt.lower()
            last_error = None
            for retry in range(args.max_retries):
                try:
                    response = client.responses.create(
                        model=MODEL, instructions=instructions, input=prompt, temperature=0,
                        max_output_tokens=80,
                        text={"format": {"type": "json_schema", "name": "substantive_similarity", "strict": True, "schema": schema}},
                        store=False,
                    )
                    timestamp = datetime.now(timezone.utc).isoformat()
                    usage = usage_dict(response)
                    append_jsonl(usage_output, {
                        "sample_id": row["sample_id"], "domain": row["domain"], "order": order,
                        "run_type": args.run_type, "request_status": "response_received",
                        "response_id": getattr(response, "id", None), "model": getattr(response, "model", MODEL),
                        **usage, "retry_index": retry, "timestamp": timestamp,
                    })
                    parsed = json.loads(response.output_text)
                    score_a = parsed.get("candidate_a_similarity")
                    score_b = parsed.get("candidate_b_similarity")
                    if not (isinstance(score_a, int) and isinstance(score_b, int) and 1 <= score_a <= 4 and 1 <= score_b <= 4):
                        raise ValueError(f"invalid scores: {parsed!r}")
                    result = {
                        "sample_id": row["sample_id"], "domain": row["domain"], "order": order,
                        "candidate_a_internal_role": role_a, "candidate_b_internal_role": role_b,
                        "candidate_a_similarity": score_a, "candidate_b_similarity": score_b,
                        "model": MODEL, "response_model": getattr(response, "model", None),
                        "response_id": getattr(response, "id", None), "prompt_version": PROMPT_VERSION,
                        "run_type": args.run_type, **usage, "timestamp": timestamp,
                    }
                    append_jsonl(output, result)
                    done.add(key)
                    print(f"[{args.run_type}] {len(done)}/{len(rows) * 2} {row['sample_id']} {order} A={score_a} B={score_b}", flush=True)
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    print(f"WARNING {row['sample_id']} {order} retry={retry}: {last_error}", flush=True)
                    if retry + 1 < args.max_retries:
                        time.sleep(2 ** retry + random.random())
            else:
                append_jsonl(ROOT / f"gpt4o/raw/substantive_similarity_failures_{args.run_type}.jsonl", {
                    "sample_id": row["sample_id"], "order": order, "error": last_error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                raise RuntimeError(last_error)


if __name__ == "__main__":
    main()
