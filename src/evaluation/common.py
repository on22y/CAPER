#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results/gpt4o/substantive_similarity_v1"
CANONICAL = PROJECT / "results/generation/base_vs_dpo501_rewrites_51.jsonl"
PROMPT = PROJECT / "prompts/evaluation/gpt4o_expressive_independence_judge_v1.txt"
MODEL = "gpt-4o"
PROMPT_VERSION = "substantive_similarity_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n" for row in rows))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows() -> list[dict[str, Any]]:
    source_rows = read_jsonl(CANONICAL)
    rows = [{
        "sample_id": row["sample_id"], "domain": row["domain"], "source_text": row["source_text"],
        "chosen": row["base_rewrite"], "rejected": row["dpo_rewrite"],
    } for row in source_rows]
    ids = [row["sample_id"] for row in rows]
    assert len(rows) == 51 and len(ids) == len(set(ids))
    assert all(row["source_text"] and row["chosen"] and row["rejected"] for row in rows)
    return rows
