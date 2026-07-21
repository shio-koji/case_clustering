#!/usr/bin/env python3
"""
plan02 Stage 0: Data preparation + morphological tokenization.

- Fix the text composition: title + description + contents (NO updates).
- Normalize (NFKC), tokenize with SudachiPy mode C (long units),
  keep nouns/verbs/adjectives in dictionary form.
- Propose stopwords from document frequency (final call is the owner's).
- Emit quality-check material for the Stage 0 checkpoint.

Run from the repo root:  python plan02/s0_tokenize.py
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from sudachipy import dictionary, tokenizer

CACHE_DIR = Path("cache")
FEATURES_DIR = Path("plan02/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# --- Tokenization policy (frozen at Stage 0) ---
SPLIT_MODE = tokenizer.Tokenizer.SplitMode.C  # long units: keeps compounds together
KEEP_POS = {"名詞", "動詞", "形容詞"}
EXCLUDE_POS_SUB = {"数詞", "代名詞", "非自立可能"}  # low-content noun/verb subclasses
DF_STOPWORD_RATIO = 0.8  # words in >=80% of cases become stopword candidates
RE_SKIP_TOKEN = re.compile(r"^[\d\W_]+$")  # digits / punctuation-only tokens


def clean_html(text: str) -> str:
    """Same cleaning as 02_build_corpus.py (entities, tags, URLs, whitespace)."""
    if not text:
        return ""
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&rdquo;", '"')
    text = text.replace("&ldquo;", '"').replace("&nbsp;", " ")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_texts():
    """Text = title + description + contents (updates deliberately excluded)."""
    corpus = json.loads((CACHE_DIR / "corpus_clean.json").read_text(encoding="utf-8"))
    records = []
    for c in corpus:
        details = json.loads(
            (CACHE_DIR / f"case_{c['id']}.json").read_text(encoding="utf-8")
        )
        contents = clean_html(details.get("contents") or "")
        parts = [p for p in [c["title"], c["description"], contents] if p.strip()]
        text = unicodedata.normalize("NFKC", "\n".join(parts))
        records.append({
            "id": c["id"],
            "title": c["title"],
            "subject_tags": c["subject_tags"],
            "case_status": c["case_status"],
            "text": text,
            "text_length": len(text),
        })
    records.sort(key=lambda r: r["id"])  # same order for every later stage
    return records


def tokenize(records):
    tok = dictionary.Dictionary().create()
    for r in records:
        tokens = []
        for m in tok.tokenize(r["text"], SPLIT_MODE):
            pos = m.part_of_speech()
            if pos[0] not in KEEP_POS or pos[1] in EXCLUDE_POS_SUB:
                continue
            base = m.dictionary_form()
            if len(base) <= 1 or RE_SKIP_TOKEN.match(base):
                continue
            tokens.append(base)
        r["tokens"] = tokens
        r["n_tokens"] = len(tokens)
    return records


def propose_stopwords(records):
    n = len(records)
    df = Counter()
    for r in records:
        df.update(set(r["tokens"]))
    threshold = int(n * DF_STOPWORD_RATIO)
    proposal = [
        {"word": w, "df": c, "df_ratio": round(c / n, 3)}
        for w, c in df.most_common()
        if c >= threshold
    ]
    return proposal, df


def quality_report(records, proposal, df):
    n = len(records)
    print(f"=== Stage 0 quality report ({n} cases) ===\n")

    print("[1] Tokenization samples (first 40 tokens):")
    for r in records[:3]:
        print(f"\n  {r['id']} {r['title'][:36]}")
        print("   ", " / ".join(r["tokens"][:40]))

    counts = sorted(r["n_tokens"] for r in records)
    print(f"\n[2] Tokens per case: min={counts[0]}, p25={counts[n//4]}, "
          f"median={counts[n//2]}, p75={counts[3*n//4]}, max={counts[-1]}")

    print(f"\n[3] Stopword candidates (DF >= {int(DF_STOPWORD_RATIO*100)}% of cases): "
          f"{len(proposal)} words")
    for p in proposal:
        print(f"    {p['df']:3d} ({p['df_ratio']:.0%})  {p['word']}")

    print("\n[4] Top-30 DF words overall (context for the stopword decision):")
    for w, c in df.most_common(30):
        print(f"    {c:3d}  {w}")


def main():
    records = build_texts()
    records = tokenize(records)
    proposal, df = propose_stopwords(records)

    tokens_out = [
        {k: r[k] for k in
         ("id", "title", "subject_tags", "case_status", "text_length", "n_tokens", "tokens")}
        for r in records
    ]
    (FEATURES_DIR / "tokens.json").write_text(
        json.dumps(tokens_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (FEATURES_DIR / "stopwords_proposal.json").write_text(
        json.dumps({
            "df_threshold_ratio": DF_STOPWORD_RATIO,
            "note": "Candidates only - the owner decides what is actually removed (Stage 0 checkpoint).",
            "candidates": proposal,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    quality_report(records, proposal, df)
    print(f"\nSaved: {FEATURES_DIR/'tokens.json'} , {FEATURES_DIR/'stopwords_proposal.json'}")


if __name__ == "__main__":
    main()
