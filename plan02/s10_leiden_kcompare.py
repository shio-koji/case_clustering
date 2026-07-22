#!/usr/bin/env python3
"""
Leiden K=5/6/7 comparison (semantic space).

Leiden has no K knob; it has a resolution knob. Sweep resolution finely,
pick a representative partition for each of 5/6/7 clusters, and describe
each with sizes, silhouette, c-TF-IDF feature words, medoid, and tag ARI/NMI.
Analogous to the NMF K-sweep, to justify (or not) the cluster count.

Run from the repo root:  python plan02/s10_leiden_kcompare.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg
from scipy import sparse
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

SEED = 42
N_NEIGHBORS = 10
FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")


def build_graph(X):
    sims = X @ X.T
    np.fill_diagonal(sims, -1)
    edges, weights, seen = [], [], set()
    for i in range(X.shape[0]):
        for j in np.argsort(sims[i])[-N_NEIGHBORS:]:
            a, b = min(i, int(j)), max(i, int(j))
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
                weights.append(float(max(sims[i, j], 0.0)))
    return ig.Graph(n=X.shape[0], edges=edges), weights


def leiden_at(g, weights, resolution):
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights=weights, resolution_parameter=resolution, seed=SEED)
    return np.array(part.membership)


def sil(X, labels):
    if len(set(labels.tolist())) < 2:
        return None
    return round(float(silhouette_score(X, labels, metric="cosine")), 4)


def ctfidf(labels, count, vocab, top_n=8):
    vocab_arr = np.array(vocab)
    classes = sorted(set(int(l) for l in labels))
    counts = np.vstack([np.asarray(count[labels == c].sum(axis=0)).ravel() for c in classes])
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + counts.mean(axis=1, keepdims=True).sum() / np.maximum(counts.sum(axis=0), 1))
    scores = tf * idf
    return {c: [str(vocab_arr[j]) for j in scores[i].argsort()[-top_n:][::-1]]
            for i, c in enumerate(classes)}


def medoid_title(members, X, titles):
    sub = X[members]
    sims = sub @ sub.T
    return titles[members[int(np.argmax(sims.mean(axis=1)))]]


def find_resolutions(g, weights, targets=(5, 6, 7)):
    """Scan resolution 0.3..2.5; record which cluster counts appear and pick one res each."""
    found = {}
    scan = {}
    for r in [round(x, 2) for x in np.arange(0.3, 2.51, 0.05)]:
        k = len(set(leiden_at(g, weights, r).tolist()))
        scan.setdefault(k, []).append(r)
    for t in targets:
        if t in scan:
            res_list = scan[t]
            found[t] = res_list[len(res_list) // 2]  # middle of the plateau
    return found, scan


def main():
    X = np.load(FEATURES_DIR / "emb_pca.npz")["matrix"]
    count = sparse.load_npz(FEATURES_DIR / "count.npz")
    vocab = json.loads((FEATURES_DIR / "vocab.json").read_text(encoding="utf-8"))["terms"]
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    titles = [r["title"] for r in tokens]
    firsts = [(r["subject_tags"][0] if r["subject_tags"] else "no_tag") for r in tokens]
    uniq = sorted(set(firsts))
    tag_ints = np.array([uniq.index(t) for t in firsts])

    g, weights = build_graph(X)
    res_for_k, scan = find_resolutions(g, weights)

    print("=== resolution → cluster-count plateau ===")
    for k in sorted(scan):
        rs = scan[k]
        print(f"  {k}クラスタ: resolution {rs[0]}–{rs[-1]}（{len(rs)}点）")

    out = {"neighbors": N_NEIGHBORS, "seed": SEED,
           "plateau": {str(k): [rs[0], rs[-1], len(rs)] for k, rs in scan.items()},
           "K": {}}

    for K, res in sorted(res_for_k.items()):
        labels = leiden_at(g, weights, res)
        sizes = np.bincount(labels).tolist()
        words = ctfidf(labels, count, vocab)
        clusters = {}
        for c in sorted(set(labels.tolist())):
            members = np.where(labels == c)[0].tolist()
            clusters[c] = {"size": len(members), "words": words[c],
                           "medoid": medoid_title(members, X, titles)}
        out["K"][K] = {
            "resolution": res,
            "silhouette": sil(X, labels),
            "sizes_sorted": sorted(sizes, reverse=True),
            "tag_ari": round(float(adjusted_rand_score(labels, tag_ints)), 4),
            "tag_nmi": round(float(normalized_mutual_info_score(labels, tag_ints)), 4),
            "clusters": clusters,
        }
        print(f"\n=== K={K} (resolution={res}) ===")
        print(f"  sizes={sorted(sizes, reverse=True)}  sil={out['K'][K]['silhouette']}  "
              f"tagARI={out['K'][K]['tag_ari']}")
        for c, cl in clusters.items():
            print(f"  C{c} (n={cl['size']}): {' / '.join(cl['words'][:6])}")
            print(f"       medoid: {cl['medoid'][:42]}")

    (RESULTS_DIR / "leiden_kcompare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR/'leiden_kcompare.json'}")


if __name__ == "__main__":
    main()
