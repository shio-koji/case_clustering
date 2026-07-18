#!/usr/bin/env python3
"""
Step 3: Full analysis pipeline.
- TF-IDF + SVD embeddings (baseline)
- Multilingual E5 sentence embeddings
- UMAP dimensionality reduction
- k-means / hierarchical / HDBSCAN clustering
- Cluster interpretation (feature terms, representative cases)
- ARI/NMI validation
- Similarity search examples
"""

import json
import pickle
import warnings
import numpy as np
import re
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

# --- Load corpus ---
corpus = json.loads((CACHE_DIR / "corpus_clean.json").read_text(encoding="utf-8"))
texts = [c["combined_text"] for c in corpus]
ids = [c["id"] for c in corpus]
titles = [c["title"] for c in corpus]
subject_tags = [c["subject_tags"] for c in corpus]
statuses = [c["case_status"] for c in corpus]
N = len(corpus)
print(f"Loaded {N} cases")


# ============================================================
# 1. TF-IDF + SVD Baseline (Japanese character n-grams)
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

print("\n[1] Computing TF-IDF (char 2-4 ngrams) + SVD...")
tfidf_cache = CACHE_DIR / "tfidf_svd.pkl"
if tfidf_cache.exists():
    print("  Loading from cache...")
    with open(tfidf_cache, "rb") as f:
        tfidf_data = pickle.load(f)
    tfidf_matrix = tfidf_data["matrix"]
    svd_emb_tfidf = tfidf_data["svd_emb"]
    vectorizer = tfidf_data["vectorizer"]
    svd = tfidf_data["svd"]
else:
    # Japanese-friendly: character n-grams (2-4) to avoid needing tokenization
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.85,
        max_features=20000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")
    # SVD to 50 dims (scRNA-seq style: PCA before clustering)
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    svd_emb_tfidf = svd.fit_transform(tfidf_matrix)
    svd_emb_tfidf = normalize(svd_emb_tfidf)  # L2-normalize
    print(f"  SVD embedding: {svd_emb_tfidf.shape}")
    with open(tfidf_cache, "wb") as f:
        pickle.dump({"matrix": tfidf_matrix, "svd_emb": svd_emb_tfidf,
                     "vectorizer": vectorizer, "svd": svd}, f)
    print(f"  Cached to {tfidf_cache}")


# ============================================================
# 2. Multilingual E5 Sentence Embeddings
# ============================================================
print("\n[2] Computing multilingual-e5-small sentence embeddings...")
e5_cache = CACHE_DIR / "e5_embeddings.pkl"
if e5_cache.exists():
    print("  Loading from cache...")
    with open(e5_cache, "rb") as f:
        e5_data = pickle.load(f)
    emb_e5 = e5_data["embeddings"]
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    # E5 models require "passage: " prefix for passages
    prefixed_texts = ["passage: " + t[:2048] for t in texts]  # truncate to 2048 chars
    print(f"  Encoding {N} texts with multilingual-e5-small...")
    emb_e5 = model.encode(prefixed_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    print(f"  E5 embedding: {emb_e5.shape}")
    with open(e5_cache, "wb") as f:
        pickle.dump({"embeddings": emb_e5, "model_name": "intfloat/multilingual-e5-small"}, f)
    print(f"  Cached to {e5_cache}")


# ============================================================
# 3. Dimensionality Reduction (UMAP + PCA for visualization)
# ============================================================
print("\n[3] UMAP dimensionality reduction...")
import umap as umap_lib
from sklearn.decomposition import PCA

umap_cache = CACHE_DIR / "umap_coords.pkl"
if umap_cache.exists():
    print("  Loading from cache...")
    with open(umap_cache, "rb") as f:
        umap_data = pickle.load(f)
else:
    # UMAP with small N-appropriate settings: n_neighbors=10 (not default 15)
    # perplexity-equivalent: n_neighbors=10 for N=95
    reducer_e5 = umap_lib.UMAP(
        n_components=2,
        n_neighbors=10,    # small N: use 10 instead of default 15
        min_dist=0.1,
        metric="cosine",
        random_state=SEED,
    )
    coords_e5 = reducer_e5.fit_transform(emb_e5)

    reducer_tfidf = umap_lib.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        random_state=SEED,
    )
    coords_tfidf = reducer_tfidf.fit_transform(svd_emb_tfidf)

    # PCA 2D for comparison
    pca_2d = PCA(n_components=2, random_state=SEED)
    coords_pca_e5 = pca_2d.fit_transform(emb_e5)

    umap_data = {
        "coords_e5": coords_e5,
        "coords_tfidf": coords_tfidf,
        "coords_pca_e5": coords_pca_e5,
    }
    with open(umap_cache, "wb") as f:
        pickle.dump(umap_data, f)
    print(f"  UMAP done. Shapes: e5={coords_e5.shape}, tfidf={coords_tfidf.shape}")

coords_e5 = umap_data["coords_e5"]
coords_tfidf = umap_data["coords_tfidf"]
coords_pca_e5 = umap_data["coords_pca_e5"]


# ============================================================
# 4. Clustering
# ============================================================
print("\n[4] Clustering...")
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Helper: evaluate clustering
def evaluate_clustering(emb, labels, name=""):
    valid = labels[labels >= 0]
    n_clusters = len(set(valid))
    n_noise = (labels == -1).sum()
    valid_mask = labels >= 0
    if valid_mask.sum() < 2 or n_clusters < 2:
        return {"name": name, "n_clusters": n_clusters, "n_noise": n_noise, "silhouette": None, "db": None}
    sil = silhouette_score(emb[valid_mask], labels[valid_mask], metric="cosine")
    db = davies_bouldin_score(emb[valid_mask], labels[valid_mask])
    return {"name": name, "n_clusters": n_clusters, "n_noise": n_noise,
            "silhouette": round(sil, 4), "db": round(db, 4)}

# --- 4a. k-means on E5 embeddings, k=4..8 ---
print("\n  [4a] k-means sweep (k=4..8) on E5 embeddings:")
kmeans_results = {}
for k in range(4, 9):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels_km = km.fit_predict(emb_e5)
    ev = evaluate_clustering(emb_e5, labels_km, f"kmeans_k{k}")
    kmeans_results[k] = {"labels": labels_km.tolist(), "metrics": ev}
    print(f"    k={k}: silhouette={ev['silhouette']}, db={ev['db']}")

# Choose best k by silhouette
best_k = max(range(4, 9), key=lambda k: kmeans_results[k]["metrics"]["silhouette"] or 0)
print(f"  -> Best k by silhouette: k={best_k}")
labels_kmeans = np.array(kmeans_results[best_k]["labels"])

# Also compute for k=6 as a comparison
labels_kmeans_6 = np.array(kmeans_results[6]["labels"])

# --- 4b. Hierarchical clustering (Ward linkage) ---
print("\n  [4b] Hierarchical clustering (Ward) on E5 embeddings:")
# Compute pairwise cosine distance
cos_dist = pdist(emb_e5, metric="cosine")
Z_ward = linkage(cos_dist, method="ward")

# Try k=5 and k=6
labels_hier = {}
for k in [5, 6]:
    labels_hier[k] = fcluster(Z_ward, k, criterion="maxclust") - 1  # 0-indexed
    ev = evaluate_clustering(emb_e5, labels_hier[k], f"hier_k{k}")
    print(f"    k={k}: silhouette={ev['silhouette']}, db={ev['db']}")

labels_hierarchical = labels_hier[6]  # use k=6 for main comparison

# --- 4c. HDBSCAN ---
print("\n  [4c] HDBSCAN on E5 embeddings:")
# For N=95, use min_cluster_size=5 (5% of data)
hdb = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=3,
    metric="euclidean",  # on normalized vectors, euclidean ≈ cosine
    cluster_selection_method="eom",
)
labels_hdbscan = hdb.fit_predict(emb_e5)
ev_hdb = evaluate_clustering(emb_e5, labels_hdbscan, "hdbscan")
print(f"    HDBSCAN: n_clusters={ev_hdb['n_clusters']}, noise={ev_hdb['n_noise']}, silhouette={ev_hdb['silhouette']}")

# --- 4d. k-means on TF-IDF SVD (baseline for comparison) ---
print("\n  [4d] k-means k=6 on TF-IDF baseline:")
km_tfidf = KMeans(n_clusters=6, random_state=SEED, n_init=10)
labels_kmeans_tfidf = km_tfidf.fit_predict(svd_emb_tfidf)
ev_tfidf = evaluate_clustering(svd_emb_tfidf, labels_kmeans_tfidf, "kmeans_tfidf_k6")
print(f"    silhouette={ev_tfidf['silhouette']}, db={ev_tfidf['db']}")


# ============================================================
# 5. Cluster Interpretation
# ============================================================
print("\n[5] Cluster interpretation...")

def get_feature_terms(labels, texts, top_n=10):
    """Extract top discriminative char n-grams per cluster using c-TF-IDF."""
    # Use word-level TF-IDF on character bigrams of whole cluster
    from sklearn.feature_extraction.text import TfidfVectorizer
    unique_labels = sorted(set(l for l in labels if l >= 0))
    cluster_docs = {}
    for lbl in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        cluster_docs[lbl] = " ".join(texts[i] for i in idxs)

    # Use the existing vectorizer vocabulary for efficiency
    vect = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
    )
    # Fit on cluster docs
    cluster_texts_list = [cluster_docs[l] for l in unique_labels]
    mat = vect.fit_transform(cluster_texts_list)
    feature_names = np.array(vect.get_feature_names_out())

    terms_per_cluster = {}
    for i, lbl in enumerate(unique_labels):
        scores = mat[i].toarray()[0]
        top_idx = scores.argsort()[-top_n:][::-1]
        terms = [(feature_names[j], round(float(scores[j]), 4)) for j in top_idx if scores[j] > 0]
        terms_per_cluster[lbl] = terms
    return terms_per_cluster


def get_representative_cases(labels, emb, top_n=3):
    """Find cases closest to each cluster centroid."""
    unique_labels = sorted(set(l for l in labels if l >= 0))
    reps = {}
    for lbl in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        cluster_emb = emb[idxs]
        centroid = cluster_emb.mean(axis=0)
        # cosine similarity to centroid
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = cluster_emb @ centroid_norm
        top_local = sims.argsort()[-top_n:][::-1]
        top_global = [idxs[j] for j in top_local]
        reps[lbl] = top_global
    return reps


def get_tag_distribution(labels, subject_tags_list):
    """Cross-tab of clusters vs tags."""
    unique_labels = sorted(set(l for l in labels if l >= 0))
    # Get all unique tags
    all_tags = sorted(set(t for tags in subject_tags_list for t in tags))
    cross_tab = {}
    for lbl in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        tag_counts = {t: 0 for t in all_tags}
        for i in idxs:
            for t in subject_tags_list[i]:
                if t in tag_counts:
                    tag_counts[t] += 1
        cross_tab[lbl] = tag_counts
    return cross_tab, all_tags


# Interpret main clustering (k-means best k on E5)
print(f"  Feature terms for k-means k={best_k} (E5):")
terms_km = get_feature_terms(labels_kmeans, texts, top_n=15)
reps_km = get_representative_cases(labels_kmeans, emb_e5, top_n=3)
cross_km, all_tags = get_tag_distribution(labels_kmeans.tolist(), subject_tags)

# Interpret hierarchical k=6
terms_hier = get_feature_terms(labels_hierarchical, texts, top_n=15)
reps_hier = get_representative_cases(labels_hierarchical, emb_e5, top_n=3)
cross_hier, _ = get_tag_distribution(labels_hierarchical.tolist(), subject_tags)

# Interpret HDBSCAN
terms_hdb = get_feature_terms(labels_hdbscan, texts, top_n=15)
reps_hdb = get_representative_cases(labels_hdbscan, emb_e5, top_n=3)
cross_hdb, _ = get_tag_distribution(labels_hdbscan.tolist(), subject_tags)

# Print representative cases per cluster
for lbl in sorted(set(l for l in labels_kmeans if l >= 0)):
    case_idxs = reps_km[lbl]
    print(f"\n  Cluster {lbl}: {[titles[i][:30] for i in case_idxs]}")
    top_terms = [t[0] for t in terms_km[lbl][:8]]
    print(f"    Top terms: {top_terms}")
    tag_dist = cross_km[lbl]
    top_tags = sorted(tag_dist.items(), key=lambda x: -x[1])[:3]
    print(f"    Top tags: {top_tags}")


# ============================================================
# 6. Validation: ARI / NMI
# ============================================================
print("\n[6] Validation (ARI / NMI)...")
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Convert subject_tags to a single dominant tag for ARI computation
# (multi-label: use first non-empty tag as label)
tag_labels = []
for tags in subject_tags:
    if tags:
        tag_labels.append(tags[0])
    else:
        tag_labels.append("no_tag")

# Encode tags as integers
tag_set = sorted(set(tag_labels))
tag_int = {t: i for i, t in enumerate(tag_set)}
tag_ints = np.array([tag_int[t] for t in tag_labels])

def ari_nmi(labels_a, labels_b, name_a, name_b):
    valid = (labels_a >= 0) & (labels_b >= 0)
    if valid.sum() < 2:
        return None
    ari = adjusted_rand_score(labels_a[valid], labels_b[valid])
    nmi = normalized_mutual_info_score(labels_a[valid], labels_b[valid])
    print(f"  {name_a} vs {name_b}: ARI={ari:.3f}, NMI={nmi:.3f}")
    return {"ari": round(ari, 4), "nmi": round(nmi, 4)}

# vs subject tags
val_km_tags = ari_nmi(labels_kmeans, tag_ints, f"kmeans_k{best_k}(e5)", "subject_tags")
val_hier_tags = ari_nmi(labels_hierarchical, tag_ints, "hier_k6(e5)", "subject_tags")
val_hdb_tags = ari_nmi(labels_hdbscan, tag_ints, "hdbscan(e5)", "subject_tags")

# Method vs method
val_km_hier = ari_nmi(labels_kmeans, labels_hierarchical, f"kmeans_k{best_k}", "hier_k6")
val_km_tfidf = ari_nmi(labels_kmeans, labels_kmeans_tfidf, f"kmeans_e5_k{best_k}", "kmeans_tfidf_k6")
val_km_hdb = ari_nmi(labels_kmeans, labels_hdbscan, f"kmeans_k{best_k}", "hdbscan")


# ============================================================
# 7. Similarity search examples
# ============================================================
print("\n[7] Similarity search examples...")

def find_similar(query_idx, emb, top_k=5):
    """Find top_k most similar cases by cosine similarity."""
    q = emb[query_idx]
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    sims = emb @ q_norm
    sims[query_idx] = -1  # exclude self
    top_idx = sims.argsort()[-top_k:][::-1]
    return [(int(i), float(sims[i])) for i in top_idx]

# Example queries
query_examples = []
example_queries = [31, 0, 10, 20, 40]  # indices into corpus
for qi in example_queries:
    similar = find_similar(qi, emb_e5, top_k=4)
    query_examples.append({
        "query_id": ids[qi],
        "query_title": titles[qi],
        "query_cluster_km": int(labels_kmeans[qi]),
        "similar": [
            {"id": ids[i], "title": titles[i], "sim": round(s, 4),
             "cluster_km": int(labels_kmeans[i])}
            for i, s in similar
        ]
    })
    print(f"  Similar to: {titles[qi][:40]}")
    for item in query_examples[-1]["similar"]:
        print(f"    -> {item['sim']:.3f} {item['title'][:40]}")


# ============================================================
# 8. Save all results
# ============================================================
print("\n[8] Saving results...")

# Prepare linkage matrix for dendrogram (serializable)
Z_list = Z_ward.tolist()

# Build per-case cluster assignments
case_clusters = {}
for i, cid in enumerate(ids):
    case_clusters[cid] = {
        "kmeans_best": int(labels_kmeans[i]),
        "kmeans_k6": int(labels_kmeans_6[i]),
        "hierarchical_k6": int(labels_hierarchical[i]),
        "hdbscan": int(labels_hdbscan[i]),
        "kmeans_tfidf_k6": int(labels_kmeans_tfidf[i]),
        "umap_e5_x": float(coords_e5[i, 0]),
        "umap_e5_y": float(coords_e5[i, 1]),
        "umap_tfidf_x": float(coords_tfidf[i, 0]),
        "umap_tfidf_y": float(coords_tfidf[i, 1]),
        "pca_e5_x": float(coords_pca_e5[i, 0]),
        "pca_e5_y": float(coords_pca_e5[i, 1]),
    }

# Silhouette metrics summary
metrics_summary = {
    "kmeans_sweep": {k: kmeans_results[k]["metrics"] for k in range(4, 9)},
    "best_k": best_k,
    "hierarchical_k5": evaluate_clustering(emb_e5, labels_hier[5], "hier_k5"),
    "hierarchical_k6": evaluate_clustering(emb_e5, labels_hierarchical, "hier_k6"),
    "hdbscan": ev_hdb,
    "kmeans_tfidf_k6": ev_tfidf,
}

validation_results = {
    "km_vs_tags": val_km_tags,
    "hier_vs_tags": val_hier_tags,
    "hdb_vs_tags": val_hdb_tags,
    "km_vs_hier": val_km_hier,
    "km_vs_tfidf": val_km_tfidf,
    "km_vs_hdb": val_km_hdb,
}

# Cluster label names (based on top terms + tags - to be used in HTML report)
# We'll do a simple rule-based naming based on dominant tags
def name_cluster(cluster_id, cross_tab, all_tags):
    tag_counts = cross_tab[cluster_id]
    top = sorted(tag_counts.items(), key=lambda x: -x[1])[:2]
    top_tags = [t for t, c in top if c > 0]
    if top_tags:
        return " / ".join(top_tags[:2])
    return f"Cluster {cluster_id}"

km_cluster_names = {lbl: name_cluster(lbl, cross_km, all_tags)
                    for lbl in sorted(set(l for l in labels_kmeans if l >= 0))}
hier_cluster_names = {lbl: name_cluster(lbl, cross_hier, all_tags)
                      for lbl in sorted(set(l for l in labels_hierarchical if l >= 0))}
hdb_cluster_names = {lbl: name_cluster(lbl, cross_hdb, all_tags)
                     for lbl in sorted(set(l for l in labels_hdbscan if l >= 0))}

# Convert numpy int keys to int
def convert_keys(d):
    return {int(k): v for k, v in d.items()}

all_results = {
    "generated_at": datetime.now().isoformat(),
    "seed": SEED,
    "n_cases": N,
    "best_k": best_k,
    "case_clusters": case_clusters,
    "metrics_summary": metrics_summary,
    "validation": validation_results,
    "terms_kmeans": {int(k): v for k, v in terms_km.items()},
    "terms_hier": {int(k): v for k, v in terms_hier.items()},
    "terms_hdbscan": {int(k): v for k, v in terms_hdb.items()},
    "reps_kmeans": {int(k): [int(i) for i in v] for k, v in reps_km.items()},
    "reps_hier": {int(k): [int(i) for i in v] for k, v in reps_hier.items()},
    "reps_hdbscan": {int(k): [int(i) for i in v] for k, v in reps_hdb.items()},
    "cross_kmeans": {int(k): v for k, v in cross_km.items()},
    "cross_hier": {int(k): v for k, v in cross_hier.items()},
    "cross_hdbscan": {int(k): v for k, v in cross_hdb.items()},
    "all_tags": all_tags,
    "km_cluster_names": km_cluster_names,
    "hier_cluster_names": hier_cluster_names,
    "hdb_cluster_names": hdb_cluster_names,
    "similarity_examples": query_examples,
    "linkage_matrix": Z_list,
    "kmeans_k6_labels": labels_kmeans_6.tolist(),
    "hier_k6_labels": labels_hierarchical.tolist(),
    "hdbscan_labels": labels_hdbscan.tolist(),
    "kmeans_best_labels": labels_kmeans.tolist(),
    "tfidf_k6_labels": labels_kmeans_tfidf.tolist(),
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

results_path = RESULTS_DIR / "analysis_results.json"
results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, cls=NumpyEncoder), encoding="utf-8")
print(f"Results saved to {results_path}")
print("\nDone! All analysis complete.")
