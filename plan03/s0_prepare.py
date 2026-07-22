#!/usr/bin/env python3
"""plan03 Stage 0 — Tag preparation.

Loads the frozen 95x11 subject-tag matrix (shared from plan02), records the
distribution / co-occurrence, and builds binary similarity + distance matrices
(Jaccard as primary — ignores co-absence so the dominant "公正な手続" tag does
not inflate similarity across unrelated cases).

Inputs (shared, read-only):
  plan02/features/tags.npz        (matrix: 95x11 float, columns: 11 tag names)
  plan02/features/case_index.json (row order = case IDs, ID-ascending)
  cache/corpus_clean.json         (id -> title, subject_tags, case_status)

Outputs:
  plan03/features/tag_sim.npz     (jaccard/dice/ochiai similarity + distances)
  plan03/features/s0_meta.json    (distribution, co-occurrence, combos)
"""
import json
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "plan03", "features")
os.makedirs(FEAT, exist_ok=True)

# ---- load frozen tag matrix (shared from plan02) --------------------------
tz = np.load(os.path.join(ROOT, "plan02/features/tags.npz"), allow_pickle=True)
M = tz["matrix"].astype(float)              # 95 x 11, {0,1}
tags = [str(t) for t in tz["columns"]]
case_ids = json.load(open(os.path.join(ROOT, "plan02/features/case_index.json")))
corpus = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "cache/corpus_clean.json")))}
N, T = M.shape
assert N == len(case_ids), "row order mismatch"

# ---- distribution ---------------------------------------------------------
counts = M.sum(0).astype(int)
order = list(np.argsort(-counts))
tags_per_case = M.sum(1).astype(int)
combos = {}
for row in M:
    key = tuple(int(x) for x in row)
    combos[key] = combos.get(key, 0) + 1

# ---- co-occurrence (tag x tag) -------------------------------------------
cooc = (M.T @ M).astype(int)                # diagonal = counts

# ---- binary similarity / distance ----------------------------------------
# Jaccard distance via scipy; similarity = 1 - distance.
jac_d = squareform(pdist(M, metric="jaccard"))
np.fill_diagonal(jac_d, 0.0)
jac_s = 1.0 - jac_d

# Dice and Ochiai (cosine on binary) computed directly.
inter = M @ M.T
row_sum = M.sum(1)
dice_s = np.zeros((N, N))
och_s = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        a, b = row_sum[i], row_sum[j]
        c = inter[i, j]
        dice_s[i, j] = (2 * c) / (a + b) if (a + b) > 0 else 0.0
        och_s[i, j] = c / np.sqrt(a * b) if (a * b) > 0 else 0.0
dice_d = 1.0 - dice_s
och_d = 1.0 - och_s

np.savez(
    os.path.join(FEAT, "tag_sim.npz"),
    matrix=M, columns=np.array(tags),
    jaccard_sim=jac_s, jaccard_dist=jac_d,
    dice_sim=dice_s, dice_dist=dice_d,
    ochiai_sim=och_s, ochiai_dist=och_d,
)

meta = {
    "n_cases": int(N), "n_tags": int(T),
    "tags_by_count": [{"tag": tags[i], "count": int(counts[i])} for i in order],
    "tags_per_case": {
        "min": int(tags_per_case.min()), "median": float(np.median(tags_per_case)),
        "max": int(tags_per_case.max()), "mean": round(float(tags_per_case.mean()), 3),
        "hist": {str(k): int((tags_per_case == k).sum()) for k in sorted(set(tags_per_case))},
    },
    "n_unique_combinations": len(combos),
    "cooccurrence": {tags[i]: {tags[j]: int(cooc[i, j]) for j in range(T)} for i in range(T)},
    "distance_choice": "Jaccard primary (ignores co-absence; the dominant "
                       "'公正な手続' tag otherwise inflates similarity).",
    "case_ids": case_ids,
    "titles": [corpus[c]["title"] for c in case_ids],
    "case_status": [corpus[c].get("case_status") for c in case_ids],
    "subject_tags_per_case": [corpus[c].get("subject_tags", []) for c in case_ids],
}
json.dump(meta, open(os.path.join(FEAT, "s0_meta.json"), "w"),
          ensure_ascii=False, indent=2)

print(f"[s0] N={N} tags={T} unique_combos={len(combos)}")
print("[s0] tag counts:", {tags[i]: int(counts[i]) for i in order})
print("[s0] jaccard sim: mean(offdiag)=%.3f  frac zero-sim pairs=%.3f" % (
    jac_s[~np.eye(N, dtype=bool)].mean(),
    (jac_s[~np.eye(N, dtype=bool)] == 0).mean()))
print("[s0] wrote features/tag_sim.npz, features/s0_meta.json")
