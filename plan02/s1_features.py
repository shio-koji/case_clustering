#!/usr/bin/env python3
"""
plan02 Stage 1: Feature representations (parallel matrices, one shared row order).

- A1 tags.npz   : one-hot of existing subject tags (null model / reference)
- A2 tfidf.npz  : morphological 1-gram + adjacent 2-gram TF-IDF (sublinear, L2)
     count.npz  : raw counts on the SAME vocabulary (LDA input)
- A3 emb.npz    : bge-m3 embeddings of the Stage-0 frozen text (updates excluded),
                  re-encoded here on purpose (the plan01 cache used updates-included
                  text and would break input parity across representations)

Row order for every matrix = case_index.json (case IDs ascending).
Run from the repo root:  python plan02/s1_features.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

FEATURES_DIR = Path("plan02/features")
TOKENS_PATH = FEATURES_DIR / "tokens.json"

# Frozen at the Stage 0 checkpoint (owner-approved, 20 words):
# 18 words with DF>=80% + 2 donation-boilerplate words (使途, 弁護団).
# Extended after the Stage 3 checkpoint (owner-approved):
# 事務所 (lawyer-profile boilerplate: 127/127 uses of 法律事務所 are team intros;
# dropping the unigram also removes the 法律_事務所 bigram) and ledge (English
# boilerplate fragment that leaked into NMF topic words).
STOPWORDS = {
    "こと", "訴訟", "ため", "もの", "よる", "いう", "裁判", "対する", "求める", "つく",
    "行う", "弁護士", "原告", "支援", "費用", "考える", "思う", "問題",
    "使途", "弁護団",
    "事務所", "ledge",
}


def load_tokens():
    records = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
    records.sort(key=lambda r: r["id"])  # canonical row order
    return records


def build_tags_matrix(records):
    all_tags = sorted({t for r in records for t in r["subject_tags"]})
    mat = np.zeros((len(records), len(all_tags)), dtype=np.float32)
    for i, r in enumerate(records):
        for t in r["subject_tags"]:
            mat[i, all_tags.index(t)] = 1.0
    return mat, all_tags


def build_ngram_docs(records):
    """Unigrams + adjacent bigrams from the frozen token stream, stopwords applied.

    Bigrams are formed BEFORE stopword removal so that '情報_公開' style pairs
    reflect true adjacency in the text; a bigram is kept only if neither part
    is a stopword.
    """
    docs = []
    for r in records:
        toks = r["tokens"]
        unigrams = [t for t in toks if t not in STOPWORDS]
        bigrams = [
            f"{a}_{b}" for a, b in zip(toks, toks[1:])
            if a not in STOPWORDS and b not in STOPWORDS
        ]
        docs.append(unigrams + bigrams)
    return docs


def build_lexical_matrices(docs):
    # One CountVectorizer -> one shared vocabulary for both Count (LDA) and TF-IDF.
    vectorizer = CountVectorizer(
        analyzer=lambda doc: doc,  # docs are already token lists
        min_df=2,
        max_df=0.90,
    )
    count = vectorizer.fit_transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(count)
    vocab = vectorizer.get_feature_names_out().tolist()
    return count, tfidf, vocab


def build_embeddings(records):
    """Re-encode the Stage-0 frozen text with bge-m3 (no updates, NFKC-normalized).

    Embeddings depend only on the raw text (not on tokens/stopwords), so reuse
    the existing matrix when present - stopword changes must not trigger a
    pointless re-encode.
    """
    emb_path = FEATURES_DIR / "emb.npz"
    if emb_path.exists():
        emb = np.load(emb_path)["matrix"]
        if emb.shape[0] == len(records):
            print("  [cache] emb.npz reused (text unchanged by stopword edits)")
            return emb

    sys.path.insert(0, "plan02")
    from s0_tokenize import build_texts  # single source of truth for the text

    texts_by_id = {r["id"]: r["text"] for r in build_texts()}
    texts = [texts_by_id[r["id"]] for r in records]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    model.max_seq_length = 8192
    emb = model.encode(texts, batch_size=4, show_progress_bar=True,
                       normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def quality_report(records, tfidf, vocab, tags_list, emb):
    n = len(records)
    print(f"\n=== Stage 1 quality report ({n} cases) ===")
    print(f"[1] Shapes: tags={n}x{len(tags_list)}, tfidf/count={tfidf.shape}, emb={emb.shape}")
    n_uni = sum("_" not in w for w in vocab)
    print(f"    Vocabulary: {len(vocab)} terms ({n_uni} unigrams, {len(vocab)-n_uni} bigrams)")

    print("\n[2] TF-IDF top terms per case (checkpoint: issue-specific words expected):")
    vocab_arr = np.array(vocab)
    for i in [0, 1, 30, 60, 90]:
        row = tfidf[i].toarray()[0]
        top = row.argsort()[-8:][::-1]
        print(f"  {records[i]['id']} {records[i]['title'][:32]}")
        print("    ", " / ".join(vocab_arr[j] for j in top))

    # Row-order invariant: embeddings were built from the same sorted IDs.
    assert emb.shape[0] == tfidf.shape[0] == n, "row count mismatch"
    print("\n[3] Row order: all matrices share case_index.json (IDs ascending) - OK")


def main():
    records = load_tokens()
    case_ids = [r["id"] for r in records]

    tags_mat, tags_list = build_tags_matrix(records)
    docs = build_ngram_docs(records)
    count, tfidf, vocab = build_lexical_matrices(docs)
    print(f"Lexical matrices done: vocab={len(vocab)}")
    emb = build_embeddings(records)

    (FEATURES_DIR / "case_index.json").write_text(
        json.dumps(case_ids, ensure_ascii=False, indent=1), encoding="utf-8")
    (FEATURES_DIR / "vocab.json").write_text(
        json.dumps({"terms": vocab, "tag_columns": tags_list,
                    "stopwords_applied": sorted(STOPWORDS)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(FEATURES_DIR / "tags.npz", matrix=tags_mat, columns=tags_list)
    sparse.save_npz(FEATURES_DIR / "tfidf.npz", tfidf.tocsr())
    sparse.save_npz(FEATURES_DIR / "count.npz", count.tocsr())
    np.savez_compressed(FEATURES_DIR / "emb.npz", matrix=emb)

    quality_report(records, tfidf, vocab, tags_list, emb)
    print(f"\nSaved to {FEATURES_DIR}/: case_index.json, vocab.json, tags.npz, tfidf.npz, count.npz, emb.npz")


if __name__ == "__main__":
    main()
