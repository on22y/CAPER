#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import ROOT, atomic_json, atomic_text, canonical_rows, normalize_text

HCLT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HCLT / "scripts"))
from reward.factuality import deterministic as det_mod  # noqa: E402

NER_ID = os.environ.get("HCLT_NER_MODEL", "chunwoolee0/klue_ner_roberta_model")
NLI_ID = os.environ.get("HCLT_NLI_MODEL", "joeddav/xlm-roberta-large-xnli")
STS_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BERTSCORE_ID = os.environ.get("HCLT_BERTSCORE_MODEL", "xlm-roberta-large")
WORD_RE = re.compile(r"[가-힣A-Za-z0-9%._+-]+")
SENT_RE = re.compile(r"[^.!?。！？…\n]+[.!?。！？…]?")
NUM_DATE_RE = re.compile(
    r"(?:\d{4}\s*[-./년]\s*\d{1,2}\s*[-./월]\s*\d{1,2}\s*일?|\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"(?:오전|오후)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?|"
    r"\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:조\s*)?(?:억\s*)?(?:만\s*)?(?:천\s*)?"
    r"(?:%|퍼센트|원|만원|억원|천원|개|명|분|마리|회|점|건|층|대|kg|g|km|m|시간|일간|일|개월|년|살|위|배)?)"
)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in SENT_RE.finditer(text or "") if m.group(0).strip()] or [normalize_text(text)]


def token_chunks(text: str, tokenizer, max_tokens: int, overlap_sentences: int = 1) -> list[str]:
    sents = sentences(text)
    chunks, current = [], []
    for sent in sents:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        if len(ids) > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
            step = max_tokens
            chunks.extend(tokenizer.decode(ids[i:i + step], skip_special_tokens=True) for i in range(0, len(ids), step))
            continue
        candidate = " ".join(current + [sent])
        if current and len(tokenizer.encode(candidate, add_special_tokens=False)) > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] + [sent] if overlap_sentences else [sent]
            while len(tokenizer.encode(" ".join(current), add_special_tokens=False)) > max_tokens and len(current) > 1:
                current.pop(0)
        else:
            current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def char_chunks(text: str, size: int = 350, overlap: int = 60) -> list[tuple[str, int]]:
    text = normalize_text(text)
    if len(text) <= size:
        return [(text, 0)]
    out, start = [], 0
    while start < len(text):
        out.append((text[start:start + size], start))
        if start + size >= len(text): break
        start += size - overlap
    return out


def entity_metric(source: str, rewrite: str) -> dict:
    spans = []
    for chunk, offset in char_chunks(source):
        for s in det_mod.extract_spans(chunk):
            if s.type in {"PERSON", "ORGANIZATION", "LOCATION"}:
                spans.append((s.type, normalize_text(s.text)))
    entities = sorted(set(spans))
    rw = normalize_text(rewrite).replace(" ", "")
    matched = sum(normalize_text(x[1]).replace(" ", "") in rw for x in entities)
    return {"source_entity_count": len(entities), "matched_entity_count": matched, "entity_preservation": matched / len(entities) if entities else math.nan}


def mention_norm(x: str) -> str:
    x = unicodedata.normalize("NFC", x).lower()
    x = re.sub(r"[,\s]", "", x)
    return x.rstrip(".")


def numeric_date_metric(source: str, rewrite: str) -> dict:
    src = Counter(mention_norm(m.group(0)) for m in NUM_DATE_RE.finditer(source) if any(c.isdigit() for c in m.group(0)))
    rw = Counter(mention_norm(m.group(0)) for m in NUM_DATE_RE.finditer(rewrite) if any(c.isdigit() for c in m.group(0)))
    total = sum(src.values())
    matched = sum(min(n, rw[k]) for k, n in src.items())
    return {"source_numeric_date_count": total, "matched_numeric_date_count": matched, "numeric_date_match": matched / total if total else math.nan}


def char4(text: str) -> set[str]:
    compact = "".join(normalize_text(text).split())
    return {compact[i:i + 4] for i in range(max(0, len(compact) - 3))}


def lcs_len(a, b) -> int:
    if len(a) < len(b): a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def surface_metrics(source: str, rewrite: str) -> dict:
    gs, gr = char4(source), char4(rewrite)
    four = len(gs & gr) / len(gs | gr) if gs or gr else math.nan
    cs, cr = normalize_text(source), normalize_text(rewrite)
    clcs = lcs_len(cs, cr)
    ws, wr = WORD_RE.findall(source), WORD_RE.findall(rewrite)
    wlcs = lcs_len(ws, wr)
    p = wlcs / len(wr) if wr else 0.0
    r = wlcs / len(ws) if ws else 0.0
    rouge = 2 * p * r / (p + r) if p + r else 0.0
    return {"fourgram_overlap": four, "lcs_ratio": clcs / len(cs) if cs else math.nan, "rougeL_f1": rouge}


def run_nli(records: list[dict], device: str) -> tuple[list[dict], dict]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI_ID, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_ID, local_files_only=True).to(device).eval()
    if device.startswith("cuda"): model.half()
    overflow = {"source": 0, "chosen": 0, "rejected": 0}
    with torch.no_grad():
        for rec in tqdm(records, desc="NLI"):
            source = rec["source_text"]; rewrite = rec["rewrite_text"]
            sysname = rec["system"]
            source_len = len(tok.encode(source, add_special_tokens=True)); rewrite_len = len(tok.encode(rewrite, add_special_tokens=True))
            overflow["source"] += int(source_len > 512); overflow[sysname] += int(rewrite_len > 512)
            source_chunks = token_chunks(source, tok, 240, 1)
            rewrite_chunks = token_chunks(rewrite, tok, 120, 0)
            best, weights = [], []
            for rw in rewrite_chunks:
                pairs = [(s, rw) for s in source_chunks]
                probs_all = []
                for i in range(0, len(pairs), 16):
                    batch = pairs[i:i + 16]
                    enc = tok([x[0] for x in batch], [x[1] for x in batch], padding=True, truncation=False, return_tensors="pt").to(device)
                    probs_all.extend(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy().tolist())
                # joeddav labels: contradiction, neutral, entailment
                best.append(max(probs_all, key=lambda p: p[2]))
                weights.append(len(tok.encode(rw, add_special_tokens=False)))
            probs = np.average(np.asarray(best), axis=0, weights=np.asarray(weights))
            rec.update({"nli_contradiction_probability": float(probs[0]), "nli_neutral_probability": float(probs[1]), "nli_entailment": float(probs[2]), "nli_predicted_label": ["contradiction", "neutral", "entailment"][int(np.argmax(probs))], "nli_source_chunks": len(source_chunks), "nli_rewrite_chunks": len(rewrite_chunks), "nli_source_tokens": source_len, "nli_rewrite_tokens": rewrite_len})
    del model, tok; cleanup()
    return records, overflow


def run_sts(records: list[dict], device: str) -> tuple[list[dict], dict]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(STS_ID, device=device)
    tok = model.tokenizer; limit = int(model.max_seq_length)
    overflow = {"source": 0, "chosen": 0, "rejected": 0}
    for rec in tqdm(records, desc="STS"):
        src_len = len(tok.encode(rec["source_text"], add_special_tokens=True)); rw_len = len(tok.encode(rec["rewrite_text"], add_special_tokens=True))
        overflow["source"] += int(src_len > limit); overflow[rec["system"]] += int(rw_len > limit)
        sc = token_chunks(rec["source_text"], tok, limit - 8, 0); rc = token_chunks(rec["rewrite_text"], tok, limit - 8, 0)
        texts = sc + rc
        emb = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sw = np.array([len(tok.encode(x, add_special_tokens=False)) for x in sc], dtype=float); rw = np.array([len(tok.encode(x, add_special_tokens=False)) for x in rc], dtype=float)
        se = np.average(emb[:len(sc)], axis=0, weights=sw); re = np.average(emb[len(sc):], axis=0, weights=rw)
        rec.update({"sts_cosine": float(np.dot(se, re) / (np.linalg.norm(se) * np.linalg.norm(re))), "sts_source_chunks": len(sc), "sts_rewrite_chunks": len(rc), "sts_source_tokens": src_len, "sts_rewrite_tokens": rw_len})
    del model; cleanup()
    return records, overflow


def run_bertscore(records: list[dict], device: str) -> tuple[list[dict], dict]:
    from bert_score import BERTScorer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BERTSCORE_ID, local_files_only=True)
    scorer = BERTScorer(model_type=BERTSCORE_ID, num_layers=24, batch_size=8, device=device, rescale_with_baseline=False)
    limit = min(int(getattr(tok, "model_max_length", 512)), 512)
    overflow = {"source": 0, "chosen": 0, "rejected": 0}
    for rec in tqdm(records, desc="BERTScore"):
        src_len = len(tok.encode(rec["source_text"], add_special_tokens=True)); rw_len = len(tok.encode(rec["rewrite_text"], add_special_tokens=True))
        overflow["source"] += int(src_len > limit); overflow[rec["system"]] += int(rw_len > limit)
        sc = token_chunks(rec["source_text"], tok, limit - 24, 0); rc = token_chunks(rec["rewrite_text"], tok, limit - 24, 0)
        refs, cands = [], []
        for s in sc:
            for r in rc:
                refs.append(s); cands.append(r)
        p, r, _ = scorer.score(cands, refs, batch_size=8)
        pm = p.cpu().numpy().reshape(len(sc), len(rc)); rm = r.cpu().numpy().reshape(len(sc), len(rc))
        sw = np.array([len(tok.encode(x, add_special_tokens=False)) for x in sc], dtype=float); rw = np.array([len(tok.encode(x, add_special_tokens=False)) for x in rc], dtype=float)
        precision = float(np.average(pm.max(axis=0), weights=rw))
        recall = float(np.average(rm.max(axis=1), weights=sw))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rec.update({"bertscore_f1": f1, "bertscore_precision": precision, "bertscore_recall": recall, "bertscore_source_chunks": len(sc), "bertscore_rewrite_chunks": len(rc), "bertscore_source_tokens": src_len, "bertscore_rewrite_tokens": rw_len})
    del scorer, tok; cleanup()
    return records, overflow


def checkpoint(path: Path, records: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(x, ensure_ascii=False, allow_nan=True) + "\n" for x in records))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-name", default="automatic_metrics_734.jsonl")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    rows = canonical_rows()
    audit = {"canonical_pairs": len(rows), "domains": dict(Counter(r["domain"] for r in rows))}
    if args.limit: rows = rows[:args.limit]
    records = []
    for row in rows:
        for system, field in [("chosen", "chosen"), ("rejected", "rejected")]:
            rec = {"sample_id": row["sample_id"], "domain": row["domain"], "system": system, "source_text": row["source_text"], "rewrite_text": row[field]}
            rec.update(entity_metric(row["source_text"], row[field]))
            rec.update(numeric_date_metric(row["source_text"], row[field]))
            rec.update(surface_metrics(row["source_text"], row[field]))
            records.append(rec)
    stage = ROOT / "metrics/per_sample" / args.output_name
    checkpoint(stage, records)
    det_mod._ner_pipeline = None
    cleanup()
    records, nli_over = run_nli(records, args.device); checkpoint(stage, records)
    records, sts_over = run_sts(records, args.device); checkpoint(stage, records)
    records, bs_over = run_bertscore(records, args.device); checkpoint(stage, records)
    length = {"input_audit": audit, "models": {"nli": {"model": NLI_ID, "limit": 512, "overflow": nli_over, "handling": "source token windows <=240 and rewrite token chunks <=120; best source-window probability per rewrite chunk, rewrite-token-weighted mean"}, "sts": {"model": STS_ID, "limit": 128, "overflow": sts_over, "handling": "token-safe chunks; token-weighted mean chunk embedding"}, "bertscore": {"model": BERTSCORE_ID, "limit": 512, "overflow": bs_over, "handling": "token-safe chunks; all cross-chunk BERTScore, best-match precision/recall with token weighting"}}}
    atomic_json(ROOT / ("metrics/length_audit_validation.json" if args.limit else "metrics/length_audit.json"), length)
    print(f"completed automatic metric rows={len(records)} output={stage}")


if __name__ == "__main__":
    main()
