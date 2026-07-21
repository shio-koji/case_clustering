#!/usr/bin/env python3
"""
plan02 Stage 2: Dimensionality reduction -> the shared reduced spaces.

- A2 TF-IDF -> TruncatedSVD, dimension chosen by cumulative explained variance
  (>=80%, capped at 100), then re-L2-normalized.
- A3 embeddings -> PCA, same criterion (>=80%, capped at 50), re-L2-normalized.
- Evidence mini-experiment: k-means on raw 1024-dim vs PCA space
  (silhouette + ARI vs first subject tag), saved as one comparison figure.
- Scree plots for both reductions.

All later clustering (Stage 3) runs in these spaces only.
Run from the repo root:  python plan02/s2_reduce.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

SEED = 42
VAR_TARGET = 0.80   # smallest dim reaching this cumulative explained variance
SVD_CAP = 100       # plan02.md Stage 2: ~50-100 for TF-IDF
PCA_CAP = 50        # plan02.md Stage 2: ~30-50 for embeddings

FEATURES_DIR = Path("plan02/features")


def choose_dim(cum_ratio, cap):
    d = int(np.searchsorted(cum_ratio, VAR_TARGET) + 1)
    return min(d, cap), float(cum_ratio[min(d, cap) - 1])


def reduce_tfidf(tfidf):
    max_comp = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=max_comp, random_state=SEED)
    full = svd.fit_transform(tfidf)
    cum = np.cumsum(svd.explained_variance_ratio_)
    dim, kept = choose_dim(cum, SVD_CAP)
    return normalize(full[:, :dim]), dim, kept, cum


def reduce_emb(emb):
    max_comp = min(emb.shape) - 1
    pca = PCA(n_components=max_comp, random_state=SEED)
    full = pca.fit_transform(emb)
    cum = np.cumsum(pca.explained_variance_ratio_)
    dim, kept = choose_dim(cum, PCA_CAP)
    return normalize(full[:, :dim]), dim, kept, cum


def tag_labels():
    """First subject tag as an integer label (same convention as plan01's ARI)."""
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    firsts = [(r["subject_tags"][0] if r["subject_tags"] else "no_tag") for r in tokens]
    uniq = sorted(set(firsts))
    return np.array([uniq.index(t) for t in firsts])


def mini_experiment(emb_raw, emb_pca, tags):
    """Evidence for '論点1': cluster quality in 1024-dim vs the PCA space."""
    rows = []
    for name, X in [("raw 1024d", emb_raw), (f"PCA {emb_pca.shape[1]}d", emb_pca)]:
        for k in range(4, 9):
            labels = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(X)
            rows.append({
                "space": name, "k": k,
                "silhouette": round(float(silhouette_score(X, labels, metric="cosine")), 4),
                "ari_vs_tags": round(float(adjusted_rand_score(tags, labels)), 4),
            })
    return rows


def plot_all(cum_svd, cum_pca, dim_svd, dim_pca, experiment, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, cum, dim, title, cap in [
        (axes[0], cum_svd, dim_svd, "TF-IDF SVD scree", 150),
        (axes[1], cum_pca, dim_pca, "Embedding PCA scree", 94),
    ]:
        n = min(len(cum), cap)
        ax.plot(range(1, n + 1), cum[:n])
        ax.axhline(VAR_TARGET, color="grey", ls="--", lw=0.8)
        ax.axvline(dim, color="red", ls="--", lw=0.8, label=f"adopted d={dim}")
        ax.set_xlabel("components"); ax.set_ylabel("cumulative variance ratio")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    for space, marker in [("raw 1024d", "o"), (None, "s")]:
        rows = [r for r in experiment if (r["space"] == space if space else r["space"] != "raw 1024d")]
        label = rows[0]["space"]
        ax.plot([r["k"] for r in rows], [r["silhouette"] for r in rows],
                marker=marker, label=f"{label} silhouette")
        ax.plot([r["k"] for r in rows], [r["ari_vs_tags"] for r in rows],
                marker=marker, ls="--", label=f"{label} ARI vs tags")
    ax.set_xlabel("k (k-means)"); ax.set_title("raw vs PCA: k-means quality")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    tfidf = sparse.load_npz(FEATURES_DIR / "tfidf.npz")
    emb = np.load(FEATURES_DIR / "emb.npz")["matrix"]

    tfidf_svd, dim_svd, kept_svd, cum_svd = reduce_tfidf(tfidf)
    emb_pca, dim_pca, kept_pca, cum_pca = reduce_emb(emb)
    print(f"TF-IDF SVD : {tfidf.shape[1]}d -> {dim_svd}d (cumvar {kept_svd:.1%})")
    print(f"Embedding  : {emb.shape[1]}d -> {dim_pca}d (cumvar {kept_pca:.1%})")

    tags = tag_labels()
    experiment = mini_experiment(emb, emb_pca, tags)
    print("\n[mini-experiment] k-means, raw 1024d vs PCA space:")
    for r in experiment:
        print(f"  {r['space']:<10} k={r['k']}: silhouette={r['silhouette']:+.4f}  ARI={r['ari_vs_tags']:+.4f}")

    np.savez_compressed(FEATURES_DIR / "tfidf_svd.npz", matrix=tfidf_svd)
    np.savez_compressed(FEATURES_DIR / "emb_pca.npz", matrix=emb_pca)
    fig_path = FEATURES_DIR / "s2_scree_and_experiment.png"
    plot_all(cum_svd, cum_pca, dim_svd, dim_pca, experiment, fig_path)

    meta = {
        "seed": SEED,
        "variance_target": VAR_TARGET,
        "tfidf_svd": {"dim": dim_svd, "cumulative_variance": round(kept_svd, 4)},
        "emb_pca": {"dim": dim_pca, "cumulative_variance": round(kept_pca, 4)},
        "mini_experiment_kmeans": experiment,
    }
    (FEATURES_DIR / "s2_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: tfidf_svd.npz, emb_pca.npz, s2_meta.json, {fig_path.name}")


if __name__ == "__main__":
    main()
