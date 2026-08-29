import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def clean_text(value):
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text):
    try:
        import kss

        sentences = kss.split_sentences(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?。！？]|[다요죠까네음임함됨])\s+", text)
    return [clean_text(sentence) for sentence in sentences if clean_text(sentence)]


def has_punctuation_rich_text(text, min_final_punct=5):
    final_punct = tuple('.!?。！？…')
    if not text.rstrip().endswith(final_punct):
        return False
    return sum(1 for char in text if char in final_punct) >= min_final_punct


def has_sentence_final_punctuation(sentences):
    if not sentences:
        return False
    final_punct = tuple('.!?。！？…')
    return all(sentence.rstrip().endswith(final_punct) for sentence in sentences)


def reservoir_add(sample, item, seen, limit, rng):
    seen += 1
    if len(sample) < limit:
        sample.append(item)
    else:
        idx = rng.randrange(seen)
        if idx < limit:
            sample[idx] = item
    return seen


def iter_json_array_items(path, array_key="data"):
    text = path.read_text(encoding="utf-8")
    key_pos = text.find(f'"{array_key}"')
    if key_pos < 0:
        return
    pos = text.find("[", key_pos)
    if pos < 0:
        return
    pos += 1
    decoder = json.JSONDecoder(strict=False)
    length = len(text)
    while pos < length:
        while pos < length and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= length or text[pos] == "]":
            break
        try:
            item, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            next_pos = text.find("\n\t\t},\n\t\t{", pos)
            if next_pos < 0:
                break
            pos = next_pos + len("\n\t\t},\n\t\t")
            continue
        yield item
        pos = end


def iter_news(raw_dir):
    for path in sorted((raw_dir / "news").glob("TS_*.json")):
        task = path.stem.removeprefix("TS_")
        for row in iter_json_array_items(path):
            paragraphs = row.get("paragraphs") or []
            text = "\n\n".join(
                clean_text(paragraph.get("context"))
                for paragraph in paragraphs
                if clean_text(paragraph.get("context"))
            )
            if not text:
                continue
            yield {
                "domain": "news",
                "source_dataset": task,
                "source_file": str(path.relative_to(raw_dir.parent.parent)),
                "source_id": row.get("doc_id"),
                "title": clean_text(row.get("doc_title")),
                "category": (row.get("doc_class") or {}).get("code"),
                "text": text,
            }


def iter_sns(raw_dir):
    for path in sorted((raw_dir / "sns").glob("TS_*.xlsx")):
        frame = pd.read_excel(path)
        for row in frame.to_dict(orient="records"):
            text = clean_text(row.get("상품평"))
            if not text:
                continue
            yield {
                "domain": "sns",
                "source_dataset": clean_text(row.get("데이터구분")) or "SNS",
                "source_file": str(path.relative_to(raw_dir.parent.parent)),
                "source_id": str(row.get("INDEX")) if row.get("INDEX") is not None else None,
                "title": clean_text(row.get("상품명")),
                "category": clean_text(row.get("카테고리")),
                "subdomain": clean_text(row.get("도메인")),
                "text": text,
            }


def iter_petition(raw_dir):
    data_dir = raw_dir / "petition" / "hf_snapshot" / "data"
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = clean_text(row.get("content"))
                if not text:
                    continue
                yield {
                    "domain": "petition",
                    "source_dataset": "heegyu/korean-petitions",
                    "source_file": str(path.relative_to(raw_dir.parent.parent)),
                    "source_id": str(row.get("petition_idx")) if row.get("petition_idx") is not None else None,
                    "title": clean_text(row.get("title")),
                    "category": clean_text(row.get("category")),
                    "status": clean_text(row.get("status")),
                    "begin": clean_text(row.get("begin")),
                    "end": clean_text(row.get("end")),
                    "num_agree": row.get("num_agree"),
                    "text": text,
                }


def sample_records(iterator, limit, seed, min_chars):
    rng = random.Random(seed)
    sample = []
    seen = 0
    for item in iterator:
        if len(item["text"]) < min_chars:
            continue
        seen = reservoir_add(sample, item, seen, limit, rng)
    return sample, seen


def balanced_sample_records(iterator, limit, seed, min_chars, bucket_limit=50):
    rng = random.Random(seed)
    buckets = {}
    seen_by_bucket = {}
    eligible = 0
    for item in iterator:
        if len(item["text"]) < min_chars:
            continue
        if item.get("domain") == "petition" and not has_punctuation_rich_text(item["text"]):
            continue
        eligible += 1
        key = clean_text(item.get("category")) or "UNKNOWN"
        buckets.setdefault(key, [])
        seen_by_bucket[key] = reservoir_add(
            buckets[key],
            item,
            seen_by_bucket.get(key, 0),
            bucket_limit,
            rng,
        )

    keys = [key for key, values in buckets.items() if values]
    keys.sort(key=lambda key: (-seen_by_bucket[key], key))
    rng.shuffle(keys)
    for values in buckets.values():
        rng.shuffle(values)

    selected = []
    while len(selected) < limit and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys

    category_counts = {}
    for item in selected:
        key = clean_text(item.get("category")) or "UNKNOWN"
        category_counts[key] = category_counts.get(key, 0) + 1
    return selected, eligible, category_counts


def finalize_records(records, domain):
    finalized = []
    for idx, record in enumerate(records, start=1):
        sentences = split_sentences(record["text"])
        record = dict(record)
        record["id"] = f"{domain}_{idx:04d}"
        record["sentences"] = sentences
        record["num_chars"] = len(record["text"])
        record["num_sentences"] = len(sentences)
        finalized.append(record)
    return finalized


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    domains = {
        "news": iter_news(args.raw_dir),
        "sns": iter_sns(args.raw_dir),
        "petition": iter_petition(args.raw_dir),
    }
    manifest = {
        "seed": args.seed,
        "limit_per_domain": args.limit,
        "min_chars": args.min_chars,
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
        "domains": {},
    }

    all_records = []
    for offset, (domain, iterator) in enumerate(domains.items()):
        sampled, eligible, category_counts = balanced_sample_records(
            iterator,
            args.limit,
            args.seed + offset,
            args.min_chars,
        )
        records = finalize_records(sampled, domain)
        out_path = args.out_dir / domain / f"pilot_{args.limit}.jsonl"
        write_jsonl(out_path, records)
        all_records.extend(records)
        manifest["domains"][domain] = {
            "eligible_records": eligible,
            "written_records": len(records),
            "output": str(out_path),
        }

    combined_path = args.out_dir / f"pilot_all_{len(all_records)}.jsonl"
    write_jsonl(combined_path, all_records)
    manifest["combined_output"] = str(combined_path)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
