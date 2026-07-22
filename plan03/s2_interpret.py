#!/usr/bin/env python3
"""plan03 Stage 2 — Interpret every tag-only clustering.

For each exclusive clustering and each mixture archetype we produce the same
three things (mirroring plan02 Stage 4, but with tags as the vocabulary):

  - enriched tags : in-cluster tag rate vs global rate (lift), the tag analog of
                    c-TF-IDF characteristic words.
  - representative case : exclusive -> medoid (min mean Jaccard distance within
                    cluster); mixture -> top-3 cases by archetype ratio.
  - name : a short mechanical label from the top enriched tags (no LLM, no tag
           majority vote borrowed from an external taxonomy — avoids circularity).
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "plan03", "features")
RES = os.path.join(ROOT, "plan03", "results")

sz = np.load(os.path.join(FEAT, "tag_sim.npz"), allow_pickle=True)
M = sz["matrix"].astype(float)
tags = [str(t) for t in sz["columns"]]
jac_d = sz["jaccard_dist"]
meta = json.load(open(os.path.join(FEAT, "s0_meta.json")))
case_ids = meta["case_ids"]
titles = meta["titles"]
N, T = M.shape
global_rate = M.mean(0)


def enriched_tags(mask, topn=4):
    """Tags over-represented in the masked cluster vs the global rate."""
    if mask.sum() == 0:
        return []
    rate = M[mask].mean(0)
    out = []
    for j in range(T):
        if rate[j] <= 0:
            continue
        lift = rate[j] / global_rate[j] if global_rate[j] > 0 else 0.0
        out.append({"tag": tags[j], "in_rate": round(float(rate[j]), 2),
                    "lift": round(float(lift), 2), "n": int(M[mask, j].sum())})
    out.sort(key=lambda d: (-d["lift"], -d["in_rate"]))
    return out[:topn]


def medoid(members):
    """Index (into case_ids) of the min-average-Jaccard-distance member."""
    if len(members) == 1:
        return members[0]
    sub = jac_d[np.ix_(members, members)]
    return members[int(sub.mean(1).argmin())]


def name_from(enr):
    if not enr:
        return "(空)"
    return "＋".join(e["tag"] for e in enr[:2])


def interpret_exclusive(labels):
    labels = np.asarray(labels)
    groups = []
    for c in sorted(set(labels)):
        members = [i for i in range(N) if labels[i] == c]
        enr = enriched_tags(labels == c)
        med = medoid(members)
        groups.append({
            "cluster": int(c), "n": len(members),
            "enriched_tags": enr, "name": name_from(enr),
            "medoid": {"id": case_ids[med], "title": titles[med]},
            "member_ids": [case_ids[i] for i in members],
        })
    return groups


def interpret_mixture(ratios, archetype_tags=None):
    ratios = np.asarray(ratios)
    K = ratios.shape[1]
    dom = ratios.argmax(1)
    groups = []
    for k in range(K):
        top = np.argsort(-ratios[:, k])[:3]
        enr = enriched_tags(dom == k) if (dom == k).any() else []
        # prefer archetype loading tags when available (NMF/LCA emission)
        atags = archetype_tags[k] if archetype_tags else enr
        groups.append({
            "archetype": k, "n_dominant": int((dom == k).sum()),
            "enriched_tags": enr,
            "archetype_tags": atags,
            "name": name_from(enr if enr else
                              [{"tag": a["tag"]} for a in (atags or [])]),
            "top_cases": [{"id": case_ids[i], "title": titles[i],
                           "ratio": round(float(ratios[i, k]), 2)} for i in top],
        })
    return groups


result = {"exclusive": {}, "mixture": {}}

# exclusive labelings
for fn, key in [("labels_lca_bic.json", "lca_bic"),
                ("labels_lca_k6.json", "lca_k6"),
                ("labels_nmf_tags_k6.json", "nmf_argmax_k6"),
                ("labels_mca_kmeans_k6.json", "mca_kmeans_k6"),
                ("labels_leiden_tags.json", "leiden_tags"),
                ("labels_hier_tags_k5.json", "hier_k5"),
                ("labels_hier_tags_k6.json", "hier_k6"),
                ("labels_cocluster_k6.json", "cocluster_k6")]:
    d = json.load(open(os.path.join(RES, fn)))
    result["exclusive"][key] = interpret_exclusive(d["labels"])

# mixtures
nmf = json.load(open(os.path.join(RES, "membership_nmf_tags.json")))
nmf_atags = [[{"tag": t["tag"]} for t in row] for row in nmf["archetype_tags"]]
result["mixture"]["nmf_tags"] = interpret_mixture(nmf["ratios"], nmf_atags)

lca = json.load(open(os.path.join(RES, "membership_lca.json")))
emis = np.asarray(lca["emission_prob"])       # K x T
lca_atags = []
for k in range(emis.shape[0]):
    ordv = np.argsort(-emis[k])
    lca_atags.append([{"tag": tags[j]} for j in ordv if emis[k, j] > 0.25][:4])
result["mixture"]["lca"] = interpret_mixture(lca["ratios"], lca_atags)

json.dump(result, open(os.path.join(RES, "interpretation.json"), "w"),
          ensure_ascii=False, indent=2)

# console summary
print("[s2] exclusive method summaries:")
for key, groups in result["exclusive"].items():
    names = [f"{g['name']}(n={g['n']})" for g in groups]
    print(f"  {key:16s} K={len(groups)}: " + " | ".join(names))
print("[s2] NMF-tag archetypes:")
for g in result["mixture"]["nmf_tags"]:
    print(f"  A{g['archetype']} (dom n={g['n_dominant']}): {g['name']}  "
          f"tags={[t['tag'] for t in g['archetype_tags']]}")
print("[s2] wrote results/interpretation.json")
