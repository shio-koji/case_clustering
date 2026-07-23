#!/usr/bin/env python3
"""
plan04 Stage 1: NMF K-sweep (K=4..KMAX) for soft classification of cases & words.

Goal: explore many topic counts (K=4 up to a bit past the 11 existing tags) and
gather the material to judge which K is "semantically right":
  - per-topic top words (H rows), with high-DF generic-noun flags
  - per-topic dominant-case counts, with tiny/over-concentrated flags
  - reconstruction error, per K
  - topic genealogy: how a topic at K splits as K -> K+1 (top-word Jaccard match)

Rebuilds TF-IDF from tokens.json + an editable stopword list, so the
"add generic nouns to stopwords and re-run" loop works: edit
plan04/features/stopwords.json, rerun this script.

Run from the repo root:  python plan04/s1_nmf_sweep.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

SEED = 42
KMIN, KMAX = 4, 14
HIGH_DF = 0.50          # a top word appearing in >=50% of cases is a generic-noun candidate
TINY_TOPIC = 2          # dominant-case count <= this -> tiny-topic flag
FEATURES_DIR = Path("plan04/features")
RESULTS_DIR = Path("plan04/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_tfidf():
    tok = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tok.sort(key=lambda r: r["id"])
    stop = set(json.loads((FEATURES_DIR / "stopwords.json").read_text(encoding="utf-8"))["stopwords"])
    docs = []
    for r in tok:
        toks = r["tokens"]
        uni = [t for t in toks if t not in stop]
        bi = [f"{a}_{b}" for a, b in zip(toks, toks[1:]) if a not in stop and b not in stop]
        docs.append(uni + bi)
    vec = CountVectorizer(analyzer=lambda d: d, min_df=2, max_df=0.90)
    count = vec.fit_transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(count)
    vocab = np.array(vec.get_feature_names_out())
    df_ratio = np.asarray((count > 0).sum(axis=0)).ravel() / count.shape[0]
    meta = {"ids": [r["id"] for r in tok], "titles": [r["title"] for r in tok],
            "tags": [r["subject_tags"] for r in tok]}
    return tfidf, vocab, df_ratio, meta, len(stop)


def top_words(H_row, vocab, n=10):
    return [str(vocab[j]) for j in H_row.argsort()[-n:][::-1]]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    tfidf, vocab, df_ratio, meta, n_stop = build_tfidf()
    N = tfidf.shape[0]
    print(f"corpus: {N} cases, vocab={len(vocab)} (stopwords={n_stop})\n")

    sweep = {}
    prev_topics = None
    generic_hits = {}  # word -> set of K where it surfaced as a high-DF top word

    for K in range(KMIN, KMAX + 1):
        model = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=700)
        W = model.fit_transform(tfidf)
        H = model.components_
        dom = W.argmax(axis=1)
        topics = []
        for t in range(K):
            tw = top_words(H[t], vocab)
            size = int((dom == t).sum())
            # concentration: share of this topic's total mass held by its top-2 cases
            col = W[:, t]; top2 = np.sort(col)[-2:].sum()
            conc = float(top2 / (col.sum() + 1e-9))
            generic = [w for w in tw if df_ratio[np.where(vocab == w)[0][0]] >= HIGH_DF]
            for w in generic:
                generic_hits.setdefault(w, set()).add(K)
            topics.append({"top_words": tw, "dominant_size": size,
                           "top2_mass_share": round(conc, 2),
                           "tiny": size <= TINY_TOPIC, "generic_words": generic})
        # genealogy: match each topic to the most similar previous-K topic by word overlap
        genealogy = []
        if prev_topics is not None:
            for t in range(K):
                sims = [(pt_i, jaccard(topics[t]["top_words"], pt["top_words"]))
                        for pt_i, pt in enumerate(prev_topics)]
                best_i, best_s = max(sims, key=lambda x: x[1])
                genealogy.append({"topic": t, "from_prev_topic": best_i, "jaccard": round(best_s, 2)})
        sweep[K] = {"reconstruction_err": round(float(model.reconstruction_err_), 4),
                    "topics": topics, "genealogy_from_prevK": genealogy}
        prev_topics = topics

        n_tiny = sum(t["tiny"] for t in topics)
        n_gen = sum(1 for t in topics if t["generic_words"])
        print(f"K={K:2d}  err={sweep[K]['reconstruction_err']}  "
              f"tiny-topics={n_tiny}  topics-with-generic-word={n_gen}")

    # summary of generic-noun candidates across the sweep
    gen_summary = sorted(({"word": w, "appears_in_K": sorted(ks),
                           "df_ratio": round(float(df_ratio[np.where(vocab == w)[0][0]]), 2)}
                          for w, ks in generic_hits.items()),
                         key=lambda x: -len(x["appears_in_K"]))

    out = {"kmin": KMIN, "kmax": KMAX, "seed": SEED, "n_cases": N,
           "vocab_size": len(vocab), "n_stopwords": n_stop,
           "high_df_threshold": HIGH_DF, "generic_candidates": gen_summary, "sweep": sweep}
    (RESULTS_DIR / "nmf_sweep.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== generic-noun candidates (high-DF words that surfaced as top words) ===")
    if gen_summary:
        for g in gen_summary:
            print(f"  {g['word']}  (DF {g['df_ratio']:.0%}, K={g['appears_in_K']})")
    else:
        print("  none")
    print(f"\nSaved: {RESULTS_DIR/'nmf_sweep.json'}")


if __name__ == "__main__":
    main()
