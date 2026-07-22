#!/usr/bin/env python3
"""plan03 Stage 1 — Run the 5 tag-only method families (+ frequent-itemset reference).

All methods consume the same 95x11 binary tag matrix and seed=42. Labels are
saved in plan02-compatible JSON ({method, labels, ...}) so Stage 3 can build a
cross-method / cross-plan ARI matrix.

  1. LCA / Bernoulli mixture  (self-implemented EM, K=2..8, BIC selection)
  2. NMF on tags              (sklearn, K sweep 4..8, focus K=6 to match plan02)
  3. MCA                      (self-implemented CA on disjunctive indicator)
  4. Tag co-occurrence community + case-similarity Leiden (igraph + leidenalg)
  5. Jaccard hierarchical + spectral co-clustering (scipy + sklearn)
  ref. frequent tag itemsets (support >= 3)
"""
import json
import os
import numpy as np
from numpy.random import default_rng
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, SpectralCoclustering
import igraph as ig
import leidenalg as la

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "plan03", "features")
RES = os.path.join(ROOT, "plan03", "results")
os.makedirs(RES, exist_ok=True)

sz = np.load(os.path.join(FEAT, "tag_sim.npz"), allow_pickle=True)
M = sz["matrix"].astype(float)                 # 95 x 11
tags = [str(t) for t in sz["columns"]]
jac_s = sz["jaccard_sim"]
jac_d = sz["jaccard_dist"]
N, T = M.shape


def norm_entropy(P):
    """Row-wise normalized entropy in [0,1] for a mixture-ratio matrix."""
    K = P.shape[1]
    out = []
    for row in P:
        p = row[row > 0]
        h = -(p * np.log(p)).sum()
        out.append(float(h / np.log(K)) if K > 1 else 0.0)
    return out


def save_labels(name, obj):
    json.dump(obj, open(os.path.join(RES, f"labels_{name}.json"), "w"),
              ensure_ascii=False, indent=2)


# ======================================================================
# 1. LCA / Bernoulli mixture  (EM with restarts, BIC over K)
# ======================================================================
def bernoulli_mixture(X, K, rng, n_restarts=20, n_iter=300, tol=1e-6, eps=1e-6):
    n, d = X.shape
    best = None
    for _ in range(n_restarts):
        # init emission probs near tag means, jittered
        base = X.mean(0)
        p = np.clip(base + 0.25 * (rng.random((K, d)) - 0.5), eps, 1 - eps)
        pi = np.full(K, 1.0 / K)
        prev = -np.inf
        for _ in range(n_iter):
            # E-step (log-space)
            logp = (X @ np.log(p).T) + ((1 - X) @ np.log(1 - p).T) + np.log(pi)
            mx = logp.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(1))
            ll = lse.sum()
            R = np.exp(logp - lse[:, None])
            # M-step
            Nk = R.sum(0) + eps
            pi = Nk / n
            p = np.clip((R.T @ X) / Nk[:, None], eps, 1 - eps)
            if abs(ll - prev) < tol:
                break
            prev = ll
        n_params = (K - 1) + K * d
        bic = -2 * ll + n_params * np.log(n)
        if best is None or ll > best["ll"]:
            best = {"ll": float(ll), "bic": float(bic), "pi": pi, "p": p, "R": R}
    return best


rng = default_rng(SEED)
lca_sweep = {}
for K in range(2, 9):
    fit = bernoulli_mixture(M, K, default_rng(SEED + K))
    lca_sweep[K] = {"loglik": fit["ll"], "bic": fit["bic"], "fit": fit}
best_K = min(lca_sweep, key=lambda k: lca_sweep[k]["bic"])
fit6 = bernoulli_mixture(M, 6, default_rng(SEED + 6))  # K=6 to align with plan02

for tag_name, K, fit in [("bic", best_K, lca_sweep[best_K]["fit"]), ("k6", 6, fit6)]:
    R = fit["R"]
    labels = R.argmax(1).astype(int).tolist()
    save_labels(f"lca_{tag_name}", {
        "method": "bernoulli_mixture_LCA", "K": int(K), "selection": tag_name,
        "labels": labels, "pi": fit["pi"].tolist(),
        "emission_prob": fit["p"].tolist(), "tags": tags,
        "bic_sweep": {str(k): round(v["bic"], 2) for k, v in lca_sweep.items()},
    })
# soft membership for the aligned (K=6) model
json.dump({
    "method": "LCA", "K": 6, "tags": tags,
    "ratios": fit6["R"].tolist(),
    "dominant": fit6["R"].argmax(1).astype(int).tolist(),
    "entropy_normalized": norm_entropy(fit6["R"]),
    "emission_prob": fit6["p"].tolist(),
}, open(os.path.join(RES, "membership_lca.json"), "w"), ensure_ascii=False, indent=2)
print(f"[s1.1] LCA/Bernoulli: BIC-optimal K={best_K}; also fit K=6")

# ======================================================================
# 2. NMF on tags (K sweep, focus K=6)
# ======================================================================
nmf_sweep = {}
for K in range(4, 9):
    m = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=2000)
    W = m.fit_transform(M)
    nmf_sweep[K] = round(float(m.reconstruction_err_), 4)
K = 6
m = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=2000)
W = m.fit_transform(M)                 # 95 x K  (case loadings)
H = m.components_                       # K x 11  (archetype x tag)
ratios = W / (W.sum(1, keepdims=True) + 1e-12)
dominant = ratios.argmax(1).astype(int)
# characteristic tags per archetype (top by loading)
arche_tags = []
for k in range(K):
    ordv = np.argsort(-H[k])
    arche_tags.append([{"tag": tags[j], "weight": round(float(H[k, j]), 3)}
                       for j in ordv if H[k, j] > 1e-3][:5])
json.dump({
    "method": "NMF_tags", "K": K, "recon_sweep": nmf_sweep,
    "tags": tags, "archetype_tags": arche_tags,
    "ratios": ratios.tolist(), "dominant_topic": dominant.tolist(),
    "entropy_normalized": norm_entropy(ratios),
}, open(os.path.join(RES, "membership_nmf_tags.json"), "w"),
    ensure_ascii=False, indent=2)
save_labels("nmf_tags_k6", {"method": "NMF_tags_argmax", "K": K,
                            "labels": dominant.tolist(), "recon_sweep": nmf_sweep})
print(f"[s1.2] NMF on tags K=6; recon sweep {nmf_sweep}")

# ======================================================================
# 3. MCA (CA on disjunctive indicator: each tag -> [absent, present])
# ======================================================================
def mca(X, n_dims=5):
    n, d = X.shape
    # disjunctive indicator Z: n x 2d
    Z = np.zeros((n, 2 * d))
    Z[:, 0::2] = 1 - X        # absent
    Z[:, 1::2] = X            # present
    col_labels = []
    for t in tags:
        col_labels += [f"{t}=0", f"{t}=1"]
    total = Z.sum()
    P = Z / total
    r = P.sum(1)              # row masses (== 1/n each, since d cats per row)
    c = P.sum(0)              # col masses
    Dr_inv = np.diag(1.0 / np.sqrt(r))
    Dc_inv = np.diag(1.0 / np.sqrt(c))
    S = Dr_inv @ (P - np.outer(r, c)) @ Dc_inv
    U, sig, Vt = np.linalg.svd(S, full_matrices=False)
    eig = sig ** 2
    inertia = eig / eig.sum()
    row_coords = (Dr_inv @ U[:, :n_dims]) * sig[:n_dims]       # principal
    col_coords = (Dc_inv @ Vt.T[:, :n_dims]) * sig[:n_dims]
    return row_coords, col_coords, col_labels, inertia[:n_dims], c


row_coords, col_coords, col_labels, mca_inertia, col_mass = mca(M, n_dims=5)
# k-means on MCA coords (first 4 dims) as an MCA-derived exclusive label
mca_km = KMeans(n_clusters=6, n_init=50, random_state=SEED).fit_predict(row_coords[:, :4])
np.savez(os.path.join(FEAT, "mca.npz"),
         row_coords=row_coords, col_coords=col_coords,
         col_labels=np.array(col_labels), inertia=mca_inertia, col_mass=col_mass)
save_labels("mca_kmeans_k6", {"method": "MCA_kmeans", "K": 6,
                              "labels": mca_km.astype(int).tolist(),
                              "inertia_top5": [round(float(x), 4) for x in mca_inertia]})
print(f"[s1.3] MCA inertia top5: {[round(float(x),3) for x in mca_inertia]}")

# ======================================================================
# 4a. Tag co-occurrence community detection (meta-tags)
# ======================================================================
cooc = (M.T @ M)
np.fill_diagonal(cooc, 0)
edges, weights = [], []
for i in range(T):
    for j in range(i + 1, T):
        if cooc[i, j] > 0:
            edges.append((i, j)); weights.append(float(cooc[i, j]))
gtag = ig.Graph(n=T, edges=edges)
gtag.es["weight"] = weights
# resolution sweep: res=1.0 collapses to 2 coarse blocks; report the sweep and
# adopt the finest resolution that still yields >=5 interpretable meta-tags.
tag_res_sweep = {}
adopted_tag_res, tag_comm = None, None
for res in [0.5, 1.0, 1.5, 2.0, 3.0]:
    p = la.find_partition(gtag, la.RBConfigurationVertexPartition,
                          weights="weight", seed=SEED, resolution_parameter=res)
    mem = list(p.membership)
    tag_res_sweep[str(res)] = {"n_comm": len(set(mem)),
                               "groups": [[tags[i] for i in range(T) if mem[i] == c]
                                          for c in sorted(set(mem))]}
    if adopted_tag_res is None and len(set(mem)) >= 5:
        adopted_tag_res, tag_comm = res, mem
if tag_comm is None:                        # fallback to coarsest
    adopted_tag_res, p = 1.0, la.find_partition(
        gtag, la.RBConfigurationVertexPartition, weights="weight",
        seed=SEED, resolution_parameter=1.0)
    tag_comm = list(p.membership)
meta_tags = {}
for c in sorted(set(tag_comm)):
    meta_tags[str(c)] = [tags[i] for i in range(T) if tag_comm[i] == c]
json.dump({"method": "tag_cooccurrence_leiden", "adopted_resolution": adopted_tag_res,
           "resolution_sweep": tag_res_sweep, "communities": meta_tags,
           "membership": tag_comm, "tags": tags,
           "cooccurrence": cooc.astype(int).tolist()},
          open(os.path.join(RES, "tag_communities.json"), "w"),
          ensure_ascii=False, indent=2)
print(f"[s1.4a] tag co-occurrence -> adopted res={adopted_tag_res}, "
      f"{len(meta_tags)} meta-tag communities")

# 4b. Case-similarity KNN + Leiden (resolution sweep -> target ~5-6 clusters)
def knn_graph(S, k=10):
    n = S.shape[0]
    edges, w = [], []
    for i in range(n):
        nn = np.argsort(-S[i]); cnt = 0
        for j in nn:
            if j == i:
                continue
            if S[i, j] <= 0:
                break
            edges.append((i, int(j))); w.append(float(S[i, j])); cnt += 1
            if cnt >= k:
                break
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = w
    g.simplify(combine_edges="max")
    return g


gcase = knn_graph(jac_s, k=10)
leiden_trials = {}
best_res, best_labels = None, None
for res in [0.25, 0.5, 0.75, 1.0, 1.5]:
    p = la.find_partition(gcase, la.RBConfigurationVertexPartition,
                          weights="weight", seed=SEED, resolution_parameter=res)
    labs = list(p.membership)
    nc = len(set(labs))
    leiden_trials[str(res)] = {"n_clusters": nc}
    if best_res is None or abs(nc - 6) < abs(leiden_trials[str(best_res)]["n_clusters"] - 6):
        best_res, best_labels = res, labs
save_labels("leiden_tags", {"method": "knn_leiden_tags", "n_neighbors": 10,
                            "adopted_resolution": best_res,
                            "labels": [int(x) for x in best_labels],
                            "trials": leiden_trials})
print(f"[s1.4b] case Leiden: adopted res={best_res} "
      f"n_clusters={len(set(best_labels))}; trials {leiden_trials}")

# ======================================================================
# 5a. Jaccard hierarchical (average linkage), cut at k=5 and k=6
# ======================================================================
Z = linkage(squareform(jac_d, checks=False), method="average")
for k in [5, 6]:
    labs = fcluster(Z, t=k, criterion="maxclust") - 1
    save_labels(f"hier_tags_k{k}", {"method": "jaccard_hier_average", "k": int(k),
                                    "labels": [int(x) for x in labs]})
json.dump({"method": "jaccard_hier_average", "linkage": Z.tolist()},
          open(os.path.join(RES, "linkage_tags.json"), "w"), indent=2)
print("[s1.5a] Jaccard hierarchical: saved k=5,6 + linkage")

# 5b. Spectral co-clustering (cases x tags block structure)
cc = SpectralCoclustering(n_clusters=6, random_state=SEED)
cc.fit(M + 1e-9)                       # avoid all-zero degenerate rows
row_lab = cc.row_labels_.astype(int)
col_lab = cc.column_labels_.astype(int)
save_labels("cocluster_k6", {"method": "spectral_cocluster", "n_clusters": 6,
                             "labels": row_lab.tolist(),
                             "tag_labels": col_lab.tolist(), "tags": tags})
tag_blocks = {}
for c in sorted(set(col_lab)):
    tag_blocks[str(c)] = [tags[i] for i in range(T) if col_lab[i] == c]
print(f"[s1.5b] spectral co-clustering: tag blocks {tag_blocks}")

print("[s1] done.")
