#!/usr/bin/env python3
"""plan03 Stage 3 — Evaluation (3 layers + the plan02 comparison).

  Q1 internal validity : silhouette on Jaccard distance (tag space only; NOT
                         comparable to plan02's embedding-space silhouettes).
  Q2 method agreement   : ARI/NMI matrix across all plan03 tag-only methods.
  Q3 vs plan02 (MAIN)   : each tag-only method vs plan02 winners
                         (text Leiden K=5, text NMF K=6 dominant topic).
                         "how much of the text-driven structure do tags recover?"
  Q4 tag critique       : dominant-tag spread, meta-tags, integration / splitting.
"""
import json
import os
import numpy as np
from itertools import combinations
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "plan03", "features")
RES = os.path.join(ROOT, "plan03", "results")
P2 = os.path.join(ROOT, "plan02", "results")

sz = np.load(os.path.join(FEAT, "tag_sim.npz"), allow_pickle=True)
M = sz["matrix"].astype(float)
jac_d = sz["jaccard_dist"]
tags = [str(t) for t in sz["columns"]]
meta = json.load(open(os.path.join(FEAT, "s0_meta.json")))
N, T = M.shape


def load_labels(fn, base=RES):
    return np.asarray(json.load(open(os.path.join(base, fn)))["labels"])


# ---- assemble label sets --------------------------------------------------
tagsets = {
    "LCA(BIC K=2)": load_labels("labels_lca_bic.json"),
    "LCA(K=6)": load_labels("labels_lca_k6.json"),
    "NMF-tags(K=6)": load_labels("labels_nmf_tags_k6.json"),
    "MCA+kmeans(K=6)": load_labels("labels_mca_kmeans_k6.json"),
    "Leiden-tags": load_labels("labels_leiden_tags.json"),
    "Jaccard-hier(k=5)": load_labels("labels_hier_tags_k5.json"),
    "Jaccard-hier(k=6)": load_labels("labels_hier_tags_k6.json"),
    "co-cluster(K=6)": load_labels("labels_cocluster_k6.json"),
}
# plan02 winners (text-driven)
text_leiden = np.asarray(json.load(open(os.path.join(P2, "labels_leiden_pca.json")))["labels"])
text_nmf = np.asarray(json.load(open(os.path.join(P2, "membership_nmf.json")))["dominant_topic"])
plan02 = {"text-Leiden(K=5)": text_leiden, "text-NMF(K=6)": text_nmf}

# ---- Q1 internal validity (silhouette on Jaccard) -------------------------
internal = {}
for name, lab in tagsets.items():
    if len(set(lab)) > 1 and len(set(lab)) < N:
        try:
            s = silhouette_score(jac_d, lab, metric="precomputed")
        except Exception:
            s = None
    else:
        s = None
    internal[name] = {"K": int(len(set(lab))),
                      "silhouette_jaccard": round(float(s), 4) if s is not None else None,
                      "largest_cluster_frac": round(float(max(np.bincount(
                          lab - lab.min()) / N)), 3)}

# ---- Q2 method-agreement matrix (within plan03) ---------------------------
names = list(tagsets)
ari = np.eye(len(names))
nmi = np.eye(len(names))
for a, b in combinations(range(len(names)), 2):
    r = adjusted_rand_score(tagsets[names[a]], tagsets[names[b]])
    m = normalized_mutual_info_score(tagsets[names[a]], tagsets[names[b]])
    ari[a, b] = ari[b, a] = round(r, 3)
    nmi[a, b] = nmi[b, a] = round(m, 3)

# ---- Q3 vs plan02 (main deliverable) --------------------------------------
vs_plan02 = {}
for tname, tlab in tagsets.items():
    vs_plan02[tname] = {}
    for pname, plab in plan02.items():
        vs_plan02[tname][pname] = {
            "ARI": round(adjusted_rand_score(plab, tlab), 3),
            "NMI": round(normalized_mutual_info_score(plab, tlab), 3),
        }
# best tag-only recovery of each text structure
best_recovery = {}
for pname in plan02:
    ranked = sorted(vs_plan02.items(), key=lambda kv: -kv[1][pname]["ARI"])
    best_recovery[pname] = [{"method": k, "ARI": v[pname]["ARI"]} for k, v in ranked[:3]]

# ---- Q4 tag critique ------------------------------------------------------
# 4a. dominant-tag spread: for the reference text-NMF topics, how does each
#     subject tag distribute across topics? (a tag spread thin = ambiguous axis)
subj = meta["subject_tags_per_case"]
tag_spread = {}
for ti, tag in enumerate(tags):
    idx = [i for i in range(N) if M[i, ti] > 0]
    if not idx:
        continue
    dist = np.bincount(text_nmf[idx], minlength=int(text_nmf.max()) + 1)
    frac = dist / dist.sum()
    tag_spread[tag] = {"n": len(idx), "max_topic_frac": round(float(frac.max()), 2),
                       "n_topics_touched": int((dist > 0).sum()),
                       "topic_dist": dist.tolist()}
# 4b. meta-tags (from Stage 1 co-occurrence communities)
meta_tags = json.load(open(os.path.join(RES, "tag_communities.json")))["communities"]
# 4c. integration/splitting relative to text-NMF topics
#     splitting: a subject tag whose cases spread across many topics.
#     flag by (a) plurality <=50%, or (b) reaching >=5 of 6 topics (cross-cutting).
n_topics = int(text_nmf.max()) + 1
splitting = sorted([{"tag": t, **v} for t, v in tag_spread.items()
                    if v["n"] >= 5 and (v["max_topic_frac"] <= 0.5
                                        or v["n_topics_touched"] >= n_topics - 1)],
                   key=lambda d: (-d["n_topics_touched"], d["max_topic_frac"]))
#     integration: text-NMF topics that absorb multiple distinct dominant tags
#     (proxy: subject tags whose plurality topic is shared)
topic_to_tags = {}
for t, v in tag_spread.items():
    top = int(np.argmax(v["topic_dist"]))
    topic_to_tags.setdefault(top, []).append((t, v["n"]))
integration = {str(k): sorted(v, key=lambda x: -x[1])
               for k, v in topic_to_tags.items() if len(v) >= 2}

evaluation = {
    "note": "silhouettes are Jaccard-space only and NOT comparable to plan02's "
            "embedding-space values.",
    "resolution_ceiling": {"n_unique_tag_combinations": meta["n_unique_combinations"],
                           "frac_zero_similarity_pairs": round(float(
                               (sz["jaccard_sim"][~np.eye(N, dtype=bool)] == 0).mean()), 3)},
    "Q1_internal": internal,
    "Q2_method_names": names,
    "Q2_ari_matrix": ari.tolist(),
    "Q2_nmi_matrix": nmi.tolist(),
    "Q3_vs_plan02": vs_plan02,
    "Q3_best_recovery": best_recovery,
    "Q4_tag_spread_over_textNMF": tag_spread,
    "Q4_meta_tags": meta_tags,
    "Q4_splitting_tags": splitting,
    "Q4_integration_topics": integration,
}
json.dump(evaluation, open(os.path.join(RES, "evaluation.json"), "w"),
          ensure_ascii=False, indent=2)

# ---- console summary ------------------------------------------------------
print("[s3] Q1 internal validity (Jaccard silhouette | largest cluster frac):")
for n, v in internal.items():
    print(f"     {n:20s} K={v['K']} sil={v['silhouette_jaccard']} "
          f"biggest={v['largest_cluster_frac']}")
print("\n[s3] Q3 MAIN — best tag-only recovery of plan02 text structures:")
for pname, lst in best_recovery.items():
    print(f"     {pname}: " + ", ".join(f"{d['method']}={d['ARI']}" for d in lst))
print("\n[s3] Q4 splitting tags (spread across text-NMF topics, <=50% in top):")
for d in splitting:
    print(f"     {d['tag']} (n={d['n']}): top topic {int(d['max_topic_frac']*100)}%, "
          f"touches {d['n_topics_touched']} topics")
print("\n[s3] Q4 meta-tags (co-occurrence communities):")
for c, g in meta_tags.items():
    print(f"     [{c}] {g}")
print("\n[s3] wrote results/evaluation.json")
