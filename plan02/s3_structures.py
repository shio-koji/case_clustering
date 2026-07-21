#!/usr/bin/env python3
"""
plan02 Stage 3: Structure discovery on the frozen reduced spaces.

Hard clustering (comparison grid), all with seed=42:
- k-means (k=4-8 sweep, primary k=5) on SVD-67d and PCA-48d
- hierarchical (cosine + average linkage, cut at k=5) on both spaces
- kNN graph + Leiden (k=10, resolutions 0.25-1.5 -> closest to 5 clusters)
- BERTopic-equivalent: emb 1024d -> UMAP 5d -> HDBSCAN -> c-TF-IDF (outliers kept)
- null model: subject-tag one-hot, Jaccard + average linkage

Mixture membership:
- NMF on raw TF-IDF, K=4-10 sweep (error curve + topic words per K -> owner picks K)
- LDA on raw counts at the provisional K (re-run cheaply once K is approved)
- diagnostics: dominant topic + normalized entropy per case

Run from the repo root:  python plan02/s3_structures.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import silhouette_score

SEED = 42
PRIMARY_K = 5
FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)


def sil(X, labels):
    """Cosine silhouette; None when undefined (single cluster / all noise)."""
    mask = labels >= 0
    if mask.sum() < 3 or len(set(labels[mask])) < 2:
        return None
    return round(float(silhouette_score(X[mask], labels[mask], metric="cosine")), 4)


def save_json(name, obj):
    (RESULTS_DIR / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------- hard clustering ----------------

def run_kmeans(X, space):
    sweep = {}
    for k in range(4, 9):
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(X)
        sweep[k] = {"labels": labels.tolist(), "silhouette": sil(X, labels)}
        print(f"  kmeans {space} k={k}: sil={sweep[k]['silhouette']}")
    save_json(f"labels_kmeans_{space}.json",
              {"method": "kmeans", "space": space, "primary_k": PRIMARY_K,
               "labels": sweep[PRIMARY_K]["labels"], "sweep": sweep})
    return np.array(sweep[PRIMARY_K]["labels"])


def run_hier(X, space):
    Z = linkage(pdist(X, metric="cosine"), method="average")
    labels = fcluster(Z, PRIMARY_K, criterion="maxclust") - 1
    print(f"  hier {space} k={PRIMARY_K}: sil={sil(X, labels)}")
    save_json(f"labels_hier_{space}.json",
              {"method": "hierarchical_average_cosine", "space": space,
               "k": PRIMARY_K, "labels": labels.tolist(), "silhouette": sil(X, labels)})
    return labels, Z


def run_leiden(X, space, n_neighbors=10):
    import igraph as ig
    import leidenalg

    sims = X @ X.T  # cosine similarity (rows are L2-normalized)
    np.fill_diagonal(sims, -1)
    edges, weights, edges_set = [], [], set()
    for i in range(X.shape[0]):
        for j in np.argsort(sims[i])[-n_neighbors:]:
            a, b = min(i, int(j)), max(i, int(j))
            if (a, b) not in edges_set:
                edges_set.add((a, b))
                edges.append((a, b))
                weights.append(float(max(sims[i, j], 0.0)))
    g = ig.Graph(n=X.shape[0], edges=edges)

    trials = {}
    for res in [0.25, 0.5, 0.75, 1.0, 1.5]:
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights=weights, resolution_parameter=res, seed=SEED)
        labels = np.array(part.membership)
        trials[res] = {"labels": labels.tolist(),
                       "n_clusters": int(labels.max()) + 1,
                       "silhouette": sil(X, labels)}
        print(f"  leiden {space} res={res}: {trials[res]['n_clusters']} clusters, "
              f"sil={trials[res]['silhouette']}")

    # closest to PRIMARY_K clusters; tie -> higher silhouette
    best = min(trials, key=lambda r: (abs(trials[r]["n_clusters"] - PRIMARY_K),
                                      -(trials[r]["silhouette"] or -1)))
    save_json(f"labels_leiden_{space}.json",
              {"method": "knn_leiden", "space": space, "n_neighbors": n_neighbors,
               "adopted_resolution": best, "labels": trials[best]["labels"],
               "trials": {str(r): {k: v for k, v in t.items() if k != "labels"}
                          for r, t in trials.items()}})
    return np.array(trials[best]["labels"])


def run_bertopic_like(emb_raw, count, vocab):
    import umap as umap_lib
    import hdbscan

    coords5 = umap_lib.UMAP(n_components=5, n_neighbors=10, min_dist=0.0,
                            metric="cosine", random_state=SEED).fit_transform(emb_raw)
    labels = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(coords5)
    n_out = int((labels == -1).sum())
    n_clu = len(set(labels[labels >= 0]))
    print(f"  bertopic-like: {n_clu} topics, {n_out}/{len(labels)} outliers")

    top_words = ctfidf_top_words(labels, count, vocab)
    save_json("labels_bertopic_pca.json",
              {"method": "bertopic_like (emb->UMAP5->HDBSCAN->c-TF-IDF)",
               "labels": labels.tolist(), "n_outliers": n_out,
               "topic_words": top_words, "silhouette": sil(emb_raw, labels)})
    return labels


def run_tags_null(tags_mat):
    Z = linkage(pdist(tags_mat, metric="jaccard"), method="average")
    labels = fcluster(Z, PRIMARY_K, criterion="maxclust") - 1
    print(f"  tags null model k={PRIMARY_K}: sizes={np.bincount(labels).tolist()}")
    save_json("labels_tags_null.json",
              {"method": "tags_jaccard_average", "k": PRIMARY_K,
               "labels": labels.tolist()})
    return labels, Z


# ---------------- mixture membership ----------------

def ctfidf_top_words(labels, count, vocab, top_n=10):
    """BERTopic-style class TF-IDF: tf(class) * log(1 + A / f(term))."""
    vocab_arr = np.array(vocab)
    classes = sorted(set(l for l in labels if l >= 0))
    counts = np.vstack([np.asarray(count[np.array(labels) == c].sum(axis=0)).ravel()
                        for c in classes])
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + counts.mean(axis=1, keepdims=True).sum() / np.maximum(counts.sum(axis=0), 1))
    scores = tf * idf
    return {int(c): [vocab_arr[j] for j in scores[i].argsort()[-top_n:][::-1]]
            for i, c in enumerate(classes)}


def topic_top_words(components, vocab, top_n=10):
    vocab_arr = np.array(vocab)
    return {k: [vocab_arr[j] for j in comp.argsort()[-top_n:][::-1]]
            for k, comp in enumerate(components)}


def membership_diagnostics(W):
    ratios = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    dom = ratios.argmax(axis=1)
    K = ratios.shape[1]
    ent = -(ratios * np.log(np.maximum(ratios, 1e-12))).sum(axis=1) / np.log(K)
    return ratios, dom, ent


def run_nmf_sweep(tfidf, vocab):
    kselect = {}
    for K in range(4, 11):
        model = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=600)
        W = model.fit_transform(tfidf)
        kselect[K] = {
            "reconstruction_err": round(float(model.reconstruction_err_), 4),
            "topic_words": topic_top_words(model.components_, vocab),
            "W": W.tolist(),
        }
        print(f"  NMF K={K}: err={kselect[K]['reconstruction_err']}")

    errs = {K: kselect[K]["reconstruction_err"] for K in kselect}
    drops = {K: errs[K - 1] - errs[K] for K in range(5, 11)}
    provisional = max(drops, key=drops.get)  # largest marginal gain; owner decides finally
    save_json("nmf_kselect.json",
              {"provisional_K": provisional, "note": "K adoption pending owner checkpoint",
               "errors": errs,
               "topic_words": {K: kselect[K]["topic_words"] for K in kselect}})
    return kselect, provisional


def save_membership(name, ratios, dom, ent, topic_words, meta):
    save_json(f"membership_{name}.json", {
        **meta,
        "topic_words": topic_words,
        "ratios": np.round(ratios, 4).tolist(),
        "dominant_topic": dom.tolist(),
        "entropy_normalized": np.round(ent, 4).tolist(),
    })


def main():
    case_ids = json.loads((FEATURES_DIR / "case_index.json").read_text())
    vocab = json.loads((FEATURES_DIR / "vocab.json").read_text())["terms"]
    X_svd = np.load(FEATURES_DIR / "tfidf_svd.npz")["matrix"]
    X_pca = np.load(FEATURES_DIR / "emb_pca.npz")["matrix"]
    emb_raw = np.load(FEATURES_DIR / "emb.npz")["matrix"]
    tags_mat = np.load(FEATURES_DIR / "tags.npz")["matrix"]
    tfidf = sparse.load_npz(FEATURES_DIR / "tfidf.npz")
    count = sparse.load_npz(FEATURES_DIR / "count.npz")

    print("[3a] hard clustering")
    linkages = {}
    for space, X in [("svd", X_svd), ("pca", X_pca)]:
        run_kmeans(X, space)
        _, Z = run_hier(X, space)
        linkages[space] = Z.tolist()
        run_leiden(X, space)
    run_bertopic_like(emb_raw, count, vocab)
    _, Z_tags = run_tags_null(tags_mat)
    linkages["tags_null"] = Z_tags.tolist()
    save_json("linkages.json", {"case_ids": case_ids, "linkages": linkages})

    print("\n[3b] mixture membership")
    kselect, provisional = run_nmf_sweep(tfidf, vocab)
    print(f"  provisional K = {provisional} (owner checkpoint decides)")

    W = np.array(kselect[provisional]["W"])
    ratios, dom, ent = membership_diagnostics(W)
    save_membership("nmf", ratios, dom, ent, kselect[provisional]["topic_words"],
                    {"method": "NMF(tfidf)", "K": provisional, "provisional": True})

    lda = LatentDirichletAllocation(n_components=provisional, random_state=SEED,
                                    max_iter=30, learning_method="batch")
    W_lda = lda.fit_transform(count)
    ratios_l, dom_l, ent_l = membership_diagnostics(W_lda)
    save_membership("lda", ratios_l, dom_l, ent_l,
                    topic_top_words(lda.components_, vocab),
                    {"method": "LDA(count)", "K": provisional, "provisional": True})
    print(f"  LDA done at K={provisional}")

    print(f"\nSaved Stage 3 outputs to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
