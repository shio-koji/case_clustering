#!/usr/bin/env python3
"""
plan02 Stage 5: Evaluation - three questions, answered with numbers.

Q1 internal validity : silhouette per method, compared ONLY within the same
                       space, with the Stage-4 one-sentence-test column
Q2 robustness        : ARI/NMI agreement matrix across all methods +
                       bootstrap stability (80% subsample x 100) for the
                       main methods
Q3 tag critique      : method-vs-tag ARI/NMI, cluster x tag cross-tabs,
                       within-cluster tag Jaccard, merge/split callouts,
                       boundary cases (NMF entropy x method instability)

Run from the repo root:  python plan02/s6_evaluate.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

SEED = 42
N_BOOT = 100
BOOT_FRAC = 0.8
FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")

rng = np.random.default_rng(SEED)


# ---------------- loading ----------------

def load_labels():
    """All method labels, one array per method (same case order everywhere)."""
    def lab(fname):
        return np.array(json.loads((RESULTS_DIR / fname).read_text())["labels"])

    labels = {
        "kmeans_svd": lab("labels_kmeans_svd.json"),
        "kmeans_pca": lab("labels_kmeans_pca.json"),
        "hier_svd": lab("labels_hier_svd.json"),
        "hier_pca": lab("labels_hier_pca.json"),
        "leiden_svd": lab("labels_leiden_svd.json"),
        "leiden_pca": lab("labels_leiden_pca.json"),
        "bertopic": lab("labels_bertopic_pca.json"),
        "tags_null": lab("labels_tags_null.json"),
    }
    for name in ["nmf", "lda"]:
        mem = json.loads((RESULTS_DIR / f"membership_{name}.json").read_text())
        labels[f"{name}_dom"] = np.array(mem["dominant_topic"])
    return labels


def load_spaces():
    return {
        "svd": np.load(FEATURES_DIR / "tfidf_svd.npz")["matrix"],
        "pca": np.load(FEATURES_DIR / "emb_pca.npz")["matrix"],
    }


def load_meta():
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in tokens]
    titles = [r["title"] for r in tokens]
    tag_sets = [set(r["subject_tags"]) for r in tokens]
    firsts = [(r["subject_tags"][0] if r["subject_tags"] else "no_tag") for r in tokens]
    uniq = sorted(set(firsts))
    tag_ints = np.array([uniq.index(t) for t in firsts])
    return ids, titles, tag_sets, tag_ints


# ---------------- Q1 internal validity ----------------

METHOD_SPACE = {
    "kmeans_svd": "svd", "hier_svd": "svd", "leiden_svd": "svd", "nmf_dom": "svd",
    "lda_dom": "svd",
    "kmeans_pca": "pca", "hier_pca": "pca", "leiden_pca": "pca", "bertopic": "pca",
}


def sil(X, labels):
    mask = labels >= 0
    if mask.sum() < 3 or len(set(labels[mask].tolist())) < 2:
        return None
    return round(float(silhouette_score(X[mask], labels[mask], metric="cosine")), 4)


def internal_validity(labels, spaces):
    names_llm = json.loads((RESULTS_DIR / "names_llm.json").read_text(encoding="utf-8"))
    key_map = {"nmf_dom": "nmf", "lda_dom": "lda"}
    rows = {}
    for m, l in labels.items():
        if m == "tags_null":
            continue
        entry = {"space": METHOD_SPACE[m], "n_clusters": len(set(l[l >= 0].tolist())),
                 "n_outliers": int((l == -1).sum()),
                 "silhouette": sil(spaces[METHOD_SPACE[m]], l)}
        nm = names_llm.get(key_map.get(m, m))
        if nm:
            groups = [v for k, v in nm.items() if not k.startswith("_")]
            entry["one_sentence_pass"] = f"{sum(g.get('nameable') for g in groups)}/{len(groups)}"
        rows[m] = entry
    return rows


# ---------------- Q2 robustness ----------------

def agreement_matrix(labels):
    names = list(labels)
    ari = {a: {} for a in names}
    nmi = {a: {} for a in names}
    for a, b in combinations(names, 2):
        s_ari = round(float(adjusted_rand_score(labels[a], labels[b])), 4)
        s_nmi = round(float(normalized_mutual_info_score(labels[a], labels[b])), 4)
        ari[a][b] = ari[b][a] = s_ari
        nmi[a][b] = nmi[b][a] = s_nmi
    for a in names:
        ari[a][a] = nmi[a][a] = 1.0
    return names, ari, nmi


def leiden_fit(X, resolution, n_neighbors=10):
    import igraph as ig
    import leidenalg
    sims = X @ X.T
    np.fill_diagonal(sims, -1)
    edges, weights, seen = [], [], set()
    for i in range(X.shape[0]):
        for j in np.argsort(sims[i])[-n_neighbors:]:
            a, b = min(i, int(j)), max(i, int(j))
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
                weights.append(float(max(sims[i, j], 0.0)))
    g = ig.Graph(n=X.shape[0], edges=edges)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    weights=weights, resolution_parameter=resolution,
                                    seed=SEED)
    return np.array(part.membership)


def bootstrap_stability(labels, spaces, tfidf):
    n = len(next(iter(labels.values())))
    leiden_res = json.loads((RESULTS_DIR / "labels_leiden_pca.json").read_text()
                            )["adopted_resolution"]

    def one_round(idx):
        out = {}
        Xp = spaces["pca"][idx]
        out["kmeans_pca"] = KMeans(n_clusters=5, random_state=SEED, n_init=10
                                   ).fit_predict(Xp)
        out["leiden_pca"] = leiden_fit(Xp, float(leiden_res))
        W = NMF(n_components=6, init="nndsvda", random_state=SEED, max_iter=400
                ).fit_transform(tfidf[idx])
        out["nmf_dom"] = W.argmax(axis=1)
        return out

    dists = {m: [] for m in ["kmeans_pca", "leiden_pca", "nmf_dom"]}
    for _ in range(N_BOOT):
        idx = np.sort(rng.choice(n, size=int(n * BOOT_FRAC), replace=False))
        sub = one_round(idx)
        for m in dists:
            dists[m].append(round(float(
                adjusted_rand_score(labels[m][idx], sub[m])), 4))
    summary = {}
    for m, d in dists.items():
        d = np.array(d)
        summary[m] = {"ari_mean": round(float(d.mean()), 4),
                      "ari_p25": round(float(np.percentile(d, 25)), 4),
                      "ari_median": round(float(np.median(d)), 4),
                      "ari_p75": round(float(np.percentile(d, 75)), 4),
                      "distribution": d.tolist()}
    return summary


# ---------------- Q3 tag critique ----------------

def tag_alignment(labels, tag_ints):
    return {m: {"ari": round(float(adjusted_rand_score(l, tag_ints)), 4),
                "nmi": round(float(normalized_mutual_info_score(l, tag_ints)), 4)}
            for m, l in labels.items() if m != "tags_null"}


def cross_tab(l, tag_sets):
    tab = {}
    for c in sorted(set(int(x) for x in l if x >= 0)):
        counts = {}
        for i in np.where(l == c)[0]:
            for t in tag_sets[i]:
                counts[t] = counts.get(t, 0) + 1
        tab[c] = dict(sorted(counts.items(), key=lambda x: -x[1]))
    return tab


def within_cluster_jaccard(l, tag_sets):
    out = {}
    for c in sorted(set(int(x) for x in l if x >= 0)):
        idx = np.where(l == c)[0]
        vals = []
        for i, j in combinations(idx, 2):
            a, b = tag_sets[i], tag_sets[j]
            union = a | b
            vals.append(len(a & b) / len(union) if union else 0.0)
        out[c] = round(float(np.mean(vals)), 4) if vals else None
    return out


def merges_and_splits(l, tag_sets, min_count=3):
    """Merge: several tags concentrated in one cluster. Split: one tag scattered."""
    tab = cross_tab(l, tag_sets)
    merges = {c: [t for t, n in counts.items() if n >= min_count]
              for c, counts in tab.items()}
    merges = {c: ts for c, ts in merges.items() if len(ts) >= 2}

    all_tags = sorted({t for s in tag_sets for t in s})
    splits = {}
    for t in all_tags:
        idx = [i for i, s in enumerate(tag_sets) if t in s]
        if len(idx) < 4:
            continue
        clusters = [int(l[i]) for i in idx if l[i] >= 0]
        if not clusters:
            continue
        top_share = max(clusters.count(c) for c in set(clusters)) / len(clusters)
        splits[t] = {"n_cases": len(idx),
                     "n_clusters_spread": len(set(clusters)),
                     "top_cluster_share": round(top_share, 3)}
    splits = {t: v for t, v in splits.items() if v["top_cluster_share"] < 0.6}
    return merges, splits


def boundary_cases(labels, ids, titles, top_n=10):
    mem = json.loads((RESULTS_DIR / "membership_nmf.json").read_text())
    ent = np.array(mem["entropy_normalized"])

    # per-case instability from the co-assignment matrix of the clean methods
    methods = ["kmeans_svd", "kmeans_pca", "hier_pca", "leiden_svd", "leiden_pca",
               "nmf_dom"]
    n = len(ids)
    co = np.zeros((n, n))
    for m in methods:
        l = labels[m]
        co += (l[:, None] == l[None, :]).astype(float)
    co /= len(methods)
    np.fill_diagonal(co, np.nan)
    instab = np.nanmean(4 * co * (1 - co), axis=1)  # max when co-assignment ~ 0.5

    top_ent = ent.argsort()[-top_n:][::-1]
    top_ins = instab.argsort()[-top_n:][::-1]
    both = [int(i) for i in top_ent if i in set(top_ins.tolist())]
    fmt = lambda idxs, vals: [{"id": ids[i], "title": titles[i],
                               "value": round(float(vals[i]), 3)} for i in idxs]
    return {"nmf_entropy_top": fmt(top_ent, ent),
            "method_instability_top": fmt(top_ins, instab),
            "in_both_lists": [{"id": ids[i], "title": titles[i]} for i in both]}


def main():
    labels = load_labels()
    spaces = load_spaces()
    tfidf = sparse.load_npz(FEATURES_DIR / "tfidf.npz")
    ids, titles, tag_sets, tag_ints = load_meta()

    print("[Q1] internal validity")
    q1 = internal_validity(labels, spaces)
    for m, r in q1.items():
        print(f"  {m:<12} space={r['space']} k={r['n_clusters']} "
              f"sil={r['silhouette']} one-sentence={r.get('one_sentence_pass')}")

    print("\n[Q2] agreement matrix + bootstrap")
    names, ari, nmi = agreement_matrix(labels)
    boot = bootstrap_stability(labels, spaces, tfidf)
    for m, s in boot.items():
        print(f"  stability {m}: median ARI={s['ari_median']} "
              f"(IQR {s['ari_p25']}-{s['ari_p75']})")

    print("\n[Q3] tag critique")
    align = tag_alignment(labels, tag_ints)
    for m, v in sorted(align.items(), key=lambda x: -x[1]["ari"]):
        print(f"  vs tags {m:<12} ARI={v['ari']:+.3f} NMI={v['nmi']:.3f}")

    main_methods = {"nmf_dom": labels["nmf_dom"], "leiden_pca": labels["leiden_pca"],
                    "kmeans_pca": labels["kmeans_pca"]}
    tabs = {m: cross_tab(l, tag_sets) for m, l in main_methods.items()}
    jac = {m: within_cluster_jaccard(l, tag_sets) for m, l in main_methods.items()}
    ms = {m: merges_and_splits(l, tag_sets) for m, l in main_methods.items()}
    for m, (mg, sp) in ms.items():
        print(f"\n  [{m}] merges (>=2 tags with >=3 cases in one cluster):")
        for c, ts in mg.items():
            print(f"    C{c}: {ts}")
        print(f"  [{m}] splits (tag spread, top-cluster share < 60%):")
        for t, v in sp.items():
            print(f"    {t}: {v['n_cases']} cases over {v['n_clusters_spread']} clusters "
                  f"(top share {v['top_cluster_share']})")

    bounds = boundary_cases(labels, ids, titles)
    print("\n  boundary cases in BOTH lists (high NMF entropy AND method instability):")
    for b in bounds["in_both_lists"]:
        print(f"    {b['title'][:50]}")

    out = {
        "seed": SEED, "n_bootstrap": N_BOOT, "bootstrap_fraction": BOOT_FRAC,
        "q1_internal_validity": q1,
        "q2_agreement": {"methods": names, "ari": ari, "nmi": nmi},
        "q2_bootstrap_stability": boot,
        "q3_tag_alignment": align,
        "q3_cross_tabs": tabs,
        "q3_within_cluster_tag_jaccard": jac,
        "q3_merges": {m: v[0] for m, v in ms.items()},
        "q3_splits": {m: v[1] for m, v in ms.items()},
        "q3_boundary_cases": bounds,
    }
    (RESULTS_DIR / "evaluation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR/'evaluation.json'}")


if __name__ == "__main__":
    main()
