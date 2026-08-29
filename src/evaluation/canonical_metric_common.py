#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

PROJECT = Path("/home/user20250805/gayeon/HCLT2026")
ROOT = PROJECT / "work/hcx_dpo367_dataset_audit_v1"
CANONICAL = PROJECT / "work/hcx_dpo_expanded367_v1/data/preferences/dpo_preferences_combined_367.jsonl"
TRAIN = PROJECT / "work/hcx_dpo_expanded367_v1/data/dpo/dpo_train_367_chatformatted.jsonl"
PREF_INITIAL = PROJECT / "work/hcx_pairwise_nonpilot210_v1/preferences/dpo_pseudo_preferences_300_survivors.jsonl"
PREF_ADDITIONAL = PROJECT / "work/hcx_pairwise_expand300_v1/preferences/expand300_dpo_pseudo_preferences_survivors.jsonl"
PREFLIGHT = PROJECT / "work/hcx_dpo_expanded367_v1/data/dpo/token_length_preflight.json"
TRAIN_LOG = PROJECT / "work/hcx_dpo_expanded367_v1/logs/lora_dpo_training.log"
PROMPT = ROOT / "prompts/gpt4o_substantive_expressive_similarity_system_prompt_v1.txt"
MODEL = "gpt-4o"
PROMPT_VERSION = "substantive_similarity_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
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
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").split())

def canonical_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(CANONICAL)
    ids = [r.get("sample_id") for r in rows]
    assert len(rows) == 367 and len(ids) == len(set(ids))
    assert all(r.get("source_text") and r.get("chosen") and r.get("rejected") for r in rows)
    assert all(r["chosen"] != r["rejected"] for r in rows)
    return rows
