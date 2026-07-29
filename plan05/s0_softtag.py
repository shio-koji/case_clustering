#!/usr/bin/env python3
"""plan05 Stage 0 — Soft tagging from text embeddings (系統B-2).

Computes, per (case, tag), a discriminative affinity score that removes the
high common-mode of bge-m3 cosines, using leave-one-out prototypes:

    disc(i,t) = cos(emb_i, pos_proto_t[LOO if i has t]) - cos(emb_i, neg_proto_t)

From disc we derive:
  - ratio(i,t): softmax over a case's ASSIGNED tags (sums to 1) = per-case weight.
  - missing-tag candidates: unassigned (i,t) with high disc (percentile vs positives).
  - over-tag candidates: assigned (i,t) with low disc.
  - validation: per-tag ROC-AUC (LOO) + top-k self-recovery.

Inputs (shared, read-only):
  plan02/features/emb.npz        (95x1024, L2-normalized bge-m3)
  plan02/features/tags.npz       (95x11 binary)
  plan02/features/case_index.json
  cache/corpus_clean.json
"""
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "plan05", "results")
os.makedirs(RES, exist_ok=True)

E = np.load(os.path.join(ROOT, "plan02/features/emb.npz"))["matrix"].astype(np.float64)
E = E / np.linalg.norm(E, axis=1, keepdims=True)          # ensure unit norm
tz = np.load(os.path.join(ROOT, "plan02/features/tags.npz"), allow_pickle=True)
M = tz["matrix"].astype(int)
tags = [str(t) for t in tz["columns"]]
case_ids = json.load(open(os.path.join(ROOT, "plan02/features/case_index.json")))
corpus = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "cache/corpus_clean.json")))}
titles = [corpus[c]["title"] for c in case_ids]
N, T = M.shape


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# --- discriminative scores with LOO positive prototypes --------------------
disc = np.zeros((N, T))
pos_only = np.zeros((N, T))       # cos to positive prototype (for reference)
reliable = []                     # tags with >=2 positives (LOO possible)
for t in range(T):
    pos = np.where(M[:, t] == 1)[0]
    neg = np.where(M[:, t] == 0)[0]
    n_pos = len(pos)
    reliable.append(n_pos >= 2)
    neg_proto = unit(E[neg].mean(0)) if len(neg) else np.zeros(E.shape[1])
    pos_sum = E[pos].sum(0)
    for i in range(N):
        if M[i, t] == 1 and n_pos >= 2:
            proto = unit((pos_sum - E[i]) / (n_pos - 1))     # leave-one-out
        elif M[i, t] == 1 and n_pos == 1:
            proto = unit(pos_sum)                             # unreliable (self)
        else:
            proto = unit(pos_sum / n_pos) if n_pos else np.zeros(E.shape[1])
        pc = float(E[i] @ proto)
        nc = float(E[i] @ neg_proto)
        pos_only[i, t] = pc
        disc[i, t] = pc - nc

# --- per-case ratio over ASSIGNED tags (softmax on disc) -------------------
assigned_disc = np.array([disc[i, t] for i in range(N) for t in range(T) if M[i, t]])
tau = float(assigned_disc.std())                              # adaptive temperature
ratios = {}
for i in range(N):
    ts = [t for t in range(T) if M[i, t] == 1]
    d = np.array([disc[i, t] for t in ts])
    w = np.exp((d - d.max()) / tau)
    w = w / w.sum()
    ratios[case_ids[i]] = {tags[t]: round(float(w[k]), 3) for k, t in enumerate(ts)}

# --- missing-tag candidates (unassigned, high disc vs positives) -----------
missing = []
for t in range(T):
    pos = np.where(M[:, t] == 1)[0]
    if not reliable[t]:
        continue
    # positive LOO disc distribution for this tag
    pos_disc = np.sort([disc[i, t] for i in pos])
    p25 = np.percentile(pos_disc, 25)
    for i in range(N):
        if M[i, t] == 0 and disc[i, t] >= p25:
            # percentile rank of this case among positives
            pct = float((pos_disc <= disc[i, t]).mean())
            missing.append({
                "case_id": case_ids[i], "title": titles[i], "tag": tags[t],
                "disc": round(float(disc[i, t]), 4),
                "pct_vs_positives": round(pct, 2),
                "n_pos": int(len(pos)),
                "low_conf": len(pos) < 3,
            })
missing.sort(key=lambda d: -d["disc"])

# --- over-tag candidates (assigned but weak) -------------------------------
over = []
for t in range(T):
    pos = np.where(M[:, t] == 1)[0]
    neg = np.where(M[:, t] == 0)[0]
    if not reliable[t]:
        continue
    neg_disc = np.array([disc[i, t] for i in neg])
    p50_neg = np.percentile(neg_disc, 50)
    for i in pos:
        if disc[i, t] <= p50_neg:      # looks no more like t than a median non-t case
            over.append({
                "case_id": case_ids[i], "title": titles[i], "tag": tags[t],
                "disc": round(float(disc[i, t]), 4),
                "assigned_ratio": ratios[case_ids[i]].get(tags[t]),
            })
over.sort(key=lambda d: d["disc"])

# --- validation: per-tag ROC-AUC (LOO) + top-k recovery --------------------
auc = {}
for t in range(T):
    if not reliable[t]:
        auc[tags[t]] = None
        continue
    y = M[:, t]
    try:
        auc[tags[t]] = round(float(roc_auc_score(y, disc[:, t])), 3)
    except Exception:
        auc[tags[t]] = None
# top-k self-recovery: does an assigned tag appear in the case's top-k disc?
recov = {1: 0, 2: 0, 3: 0}
denom = 0
for i in range(N):
    assigned = set(np.where(M[i] == 1)[0])
    if not assigned:
        continue
    denom += 1
    order = list(np.argsort(-disc[i]))
    for k in recov:
        if assigned & set(order[:k]):
            recov[k] += 1
recovery = {f"top{k}": round(recov[k] / denom, 3) for k in recov}

out = {
    "method": "discriminative prototype (pos-neg centroid) + LOO, softmax ratio",
    "tau": round(tau, 4),
    "tags": tags,
    "reliable_tags": {tags[t]: bool(reliable[t]) for t in range(T)},
    "disc_matrix": disc.round(4).tolist(),
    "case_ids": case_ids,
    "titles": titles,
    "ratios": ratios,
    "missing_candidates": missing,
    "over_candidates": over,
    "validation": {"per_tag_auc": auc, "self_recovery": recovery,
                   "mean_auc_reliable": round(float(np.mean(
                       [v for v in auc.values() if v is not None])), 3)},
}
json.dump(out, open(os.path.join(RES, "soft_tags.json"), "w"),
          ensure_ascii=False, indent=2)

# --- console summary -------------------------------------------------------
print(f"[s0] tau={tau:.4f}  mean AUC (reliable tags)={out['validation']['mean_auc_reliable']}")
print("[s0] per-tag AUC:", {k: v for k, v in auc.items()})
print("[s0] self-recovery:", recovery)
print(f"[s0] missing-tag candidates: {len(missing)} (top 8):")
for m in missing[:8]:
    flag = " [low-conf]" if m["low_conf"] else ""
    print(f"     +{m['tag']:14s} pct={m['pct_vs_positives']:.2f} disc={m['disc']:.3f}  "
          f"{m['title'][:34]}{flag}")
print(f"[s0] over-tag candidates: {len(over)} (worst 5):")
for o in over[:5]:
    print(f"     -{o['tag']:14s} disc={o['disc']:.3f}  {o['title'][:34]}")
# sample ratios
print("[s0] sample per-case ratios:")
for cid in case_ids[:2]:
    print(f"     {cid}: {ratios[cid]}")
print("[s0] wrote results/soft_tags.json")
