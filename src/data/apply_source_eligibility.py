#!/usr/bin/env python3
"""Prepare 100-document-per-domain experiment dataset from raw data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from prepare_pilot_data import (
    balanced_sample_records,
    clean_text,
    has_sentence_final_punctuation,
    iter_news,
    iter_petition,
    iter_sns,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PUNCT = tuple(".!?。！？…")


def fast_split_sentences(text: str) -> list[str]:
    normalized = clean_text(text)
    normalized = re.sub(r"\s*\n+\s*", " ", normalized)
    parts = re.split(r"(?<=[!?。！？…])\s+|(?<=[^0-9]\.)\s+", normalized)
    return [clean_text(part) for part in parts if clean_text(part)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pilot_source_keys(path: Path) -> set[tuple[str, str | None]]:
    keys = set()
    for row in read_jsonl(path):
        keys.add((row.get("domain"), str(row.get("source_id")) if row.get("source_id") is not None else None))
    return keys


def is_sentence_splittable(text: str, min_sentences: int, min_final_punct: int) -> tuple[bool, list[str], str | None]:
    if not text or not text.rstrip().endswith(FINAL_PUNCT):
        return False, [], "text_does_not_end_with_sentence_final_punctuation"
    if sum(1 for char in text if char in FINAL_PUNCT) < min_final_punct:
        return False, [], "not_enough_sentence_final_punctuation"
    sentences = fast_split_sentences(text)
    if len(sentences) < min_sentences:
        return False, sentences, "not_enough_split_sentences"
    if not has_sentence_final_punctuation(sentences):
        return False, sentences, "some_split_sentences_lack_final_punctuation"
    return True, sentences, None


def has_readable_spacing(text: str, domain: str) -> tuple[bool, str | None, dict[str, Any]]:
    hangul_count = len(re.findall(r"[가-힣]", text or ""))
    space_count = (text or "").count(" ")
    space_ratio = space_count / max(1, hangul_count)
    tokens = re.findall(r"\S+", text or "")
    max_hangul_token = max((len(re.findall(r"[가-힣]", token)) for token in tokens), default=0)
    punct_no_space = len(re.findall(r"[.!?。！？…][가-힣]", text or ""))
    metrics = {
        "hangul_count": hangul_count,
        "space_ratio": round(space_ratio, 4),
        "max_hangul_token": max_hangul_token,
        "punct_no_space_count": punct_no_space,
    }
    if domain == "petition":
        if space_ratio < 0.20:
            return False, "low_korean_spacing_ratio", metrics
        if max_hangul_token >= 18:
            return False, "long_unspaced_korean_token", metrics
        if punct_no_space >= 5 and space_ratio < 0.30:
            return False, "sentence_punctuation_without_following_space", metrics
    elif domain == "sns":
        if space_ratio < 0.16:
            return False, "low_korean_spacing_ratio", metrics
        if max_hangul_token >= 18:
            return False, "long_unspaced_korean_token", metrics
        if punct_no_space >= 10 and space_ratio < 0.28:
            return False, "sentence_punctuation_without_following_space", metrics
    else:
        if max_hangul_token >= 35:
            return False, "long_unspaced_korean_token", metrics
    return True, None, metrics


def eligible_iterator(
    iterator: Iterable[dict[str, Any]],
    domain: str,
    pilot_keys: set[tuple[str, str | None]],
    min_sentences: int,
    min_final_punct: int,
    stats: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    for item in iterator:
        stats["raw_seen"] += 1
        source_key = (domain, str(item.get("source_id")) if item.get("source_id") is not None else None)
        if source_key in pilot_keys:
            stats["excluded_pilot_overlap"] += 1
            continue
        ok, sentences, reason = is_sentence_splittable(item.get("text", ""), min_sentences, min_final_punct)
        if not ok:
            stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1
            continue
        quality_ok, quality_reason, spacing_metrics = has_readable_spacing(item.get("text", ""), domain)
        if not quality_ok:
            stats["filter_reasons"][quality_reason] = stats["filter_reasons"].get(quality_reason, 0) + 1
            continue
        enriched = dict(item)
        enriched["_precomputed_sentences"] = sentences
        stats["splittable_seen"] += 1
        yield enriched


def finalize_with_precomputed_sentences(records: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    finalized = []
    for idx, record in enumerate(records, start=1):
        sentences = record.pop("_precomputed_sentences", None)
        if sentences is None:
            sentences = fast_split_sentences(record["text"])
        out = dict(record)
        out["id"] = f"{domain}_{idx:04d}"
        out["sentences"] = [clean_text(sentence) for sentence in sentences if clean_text(sentence)]
        out["num_chars"] = len(out["text"])
        out["num_sentences"] = len(out["sentences"])
        finalized.append(out)
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/experiment_100")
    parser.add_argument("--pilot-path", type=Path, default=PROJECT_ROOT / "data/processed/pilot_all_30.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--min-sentences", type=int, default=3)
    parser.add_argument("--min-final-punct", type=int, default=3)
    args = parser.parse_args()

    domain_iterators: dict[str, Callable[[Path], Iterable[dict[str, Any]]]] = {
        "news": iter_news,
        "sns": iter_sns,
        "petition": iter_petition,
    }
    pilot_keys = load_pilot_source_keys(args.pilot_path)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "limit_per_domain": args.limit,
        "min_chars": args.min_chars,
        "min_sentences": args.min_sentences,
        "min_final_punctuation_count": args.min_final_punct,
        "excluded_pilot_source_ids": len(pilot_keys),
        "schema": [
            "id",
            "domain",
            "source_dataset",
            "source_file",
            "source_id",
            "title",
            "category",
            "text",
            "sentences",
            "num_chars",
            "num_sentences",
        ],
        "filter": "raw records only; excludes pilot source_ids; text must split into >= min_sentences, every split sentence must end with .!?。！？…, and Korean spacing quality must pass domain-specific thresholds",
        "domains": {},
    }

    all_records = []
    for offset, (domain, iterator_factory) in enumerate(domain_iterators.items()):
        stats: dict[str, Any] = {
            "raw_seen": 0,
            "excluded_pilot_overlap": 0,
            "splittable_seen": 0,
            "filter_reasons": {},
        }
        iterator = eligible_iterator(
            iterator_factory(args.raw_dir),
            domain,
            pilot_keys,
            args.min_sentences,
            args.min_final_punct,
            stats,
        )
        sampled, eligible, category_counts = balanced_sample_records(
            iterator,
            args.limit,
            args.seed + offset,
            args.min_chars,
            bucket_limit=max(20, args.limit),
        )
        records = finalize_with_precomputed_sentences(sampled, domain)
        out_path = args.out_dir / domain / f"experiment_{args.limit}.jsonl"
        write_jsonl(out_path, records)
        all_records.extend(records)
        manifest["domains"][domain] = {
            **stats,
            "eligible_after_min_chars": eligible,
            "written_records": len(records),
            "output": str(out_path.relative_to(PROJECT_ROOT)),
            "category_counts": category_counts,
        }

    combined_path = args.out_dir / f"experiment_all_{len(all_records)}.jsonl"
    write_jsonl(combined_path, all_records)
    manifest["combined_output"] = str(combined_path.relative_to(PROJECT_ROOT))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
