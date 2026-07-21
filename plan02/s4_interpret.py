#!/usr/bin/env python3
"""
plan02 Stage 4: Interpretation - the same 3-piece kit for every method.

For every cluster/topic of every Stage-3 method:
- descriptor words: c-TF-IDF top 10 (identical procedure across methods,
  so methods can be compared fairly)
- representative cases: medoid (hard clusters, in the method's own space)
  or top-3 ratio cases (topics)
- mechanical name: top-2 descriptor words joined - existing tags are NOT
  used for naming (that would be circular in a tag-critique analysis).
  LLM-assisted names live in a separate names_llm.json with evidence.

Run from the repo root:  python plan02/s4_interpret.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse

FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")


def load_all():
    case_ids = json.loads((FEATURES_DIR / "case_index.json").read_text())
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    titles = [r["title"] for r in tokens]
    tags = [r["subject_tags"] for r in tokens]
    vocab = json.loads((FEATURES_DIR / "vocab.json").read_text(encoding="utf-8"))["terms"]
    count = sparse.load_npz(FEATURES_DIR / "count.npz")
    spaces = {
        "svd": np.load(FEATURES_DIR / "tfidf_svd.npz")["matrix"],
        "pca": np.load(FEATURES_DIR / "emb_pca.npz")["matrix"],
        "tags": np.load(FEATURES_DIR / "tags.npz")["matrix"],
    }
    return case_ids, titles, tags, vocab, count, spaces


def ctfidf_words(labels, count, vocab, top_n=10):
    """BERTopic-style class TF-IDF over member documents (label -1 = skipped)."""
    vocab_arr = np.array(vocab)
    labels = np.asarray(labels)
    classes = sorted(set(int(l) for l in labels if l >= 0))
    counts = np.vstack([np.asarray(count[labels == c].sum(axis=0)).ravel()
                        for c in classes])
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + counts.mean(axis=1, keepdims=True).sum()
                 / np.maximum(counts.sum(axis=0), 1))
    scores = tf * idf
    return {c: [str(vocab_arr[j]) for j in scores[i].argsort()[-top_n:][::-1]]
            for i, c in enumerate(classes)}


def medoid(members, X):
    """Index (into `members`) of the case with the smallest mean cosine distance."""
    sub = X[members]
    sims = sub @ sub.T  # rows are L2-normalized in svd/pca spaces
    return members[int(np.argmax(sims.mean(axis=1)))]


def jaccard_medoid(members, tags_mat):
    sub = tags_mat[members]
    inter = sub @ sub.T
    sizes = sub.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return members[int(np.argmax(jac.mean(axis=1)))]


def interpret_hard(name, labels, space_key, spaces, count, vocab, case_ids, titles, tags):
    labels = np.asarray(labels)
    words = ctfidf_words(labels, count, vocab)
    clusters = {}
    for c in sorted(set(int(l) for l in labels if l >= 0)):
        members = np.where(labels == c)[0].tolist()
        med = (jaccard_medoid(members, spaces["tags"]) if space_key == "tags"
               else medoid(members, spaces[space_key]))
        top2 = words[c][:2]
        clusters[c] = {
            "size": len(members),
            "descriptor_words": words[c],
            "mechanical_name": "・".join(top2),
            "medoid": {"id": case_ids[med], "title": titles[med]},
            "members": [{"id": case_ids[i], "title": titles[i]} for i in members],
            "top_tags_reference_only": _tag_counts(members, tags),
        }
    out = {"method": name, "space": space_key, "clusters": clusters}
    n_out = int((labels == -1).sum())
    if n_out:
        out["outliers"] = [{"id": case_ids[i], "title": titles[i]}
                           for i in np.where(labels == -1)[0]]
    return out


def _tag_counts(members, tags):
    """Existing-tag distribution, kept as REFERENCE (never used for naming)."""
    counts = {}
    for i in members:
        for t in tags[i]:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1])[:3])


def interpret_mixture(name, mem, count, vocab, case_ids, titles, tags):
    ratios = np.array(mem["ratios"])
    dom = np.asarray(mem["dominant_topic"])
    words_dom = ctfidf_words(dom, count, vocab)  # same procedure as hard methods
    topics = {}
    for t in range(ratios.shape[1]):
        top3 = ratios[:, t].argsort()[-3:][::-1]
        members = np.where(dom == t)[0].tolist()
        desc = words_dom.get(t, mem["topic_words"][str(t)][:10])
        topics[t] = {
            "size_dominant": len(members),
            "descriptor_words": desc,
            "component_words": mem["topic_words"][str(t)][:10],
            "mechanical_name": "・".join(desc[:2]),
            "representatives": [
                {"id": case_ids[i], "title": titles[i],
                 "ratio": round(float(ratios[i, t]), 3)} for i in top3],
            "top_tags_reference_only": _tag_counts(members, tags),
        }
    return {"method": name, "K": ratios.shape[1], "topics": topics}


def main():
    case_ids, titles, tags, vocab, count, spaces = load_all()

    interpretation = {"hard": {}, "mixture": {}}

    hard_sources = [
        ("kmeans_svd", "labels_kmeans_svd.json", "svd"),
        ("kmeans_pca", "labels_kmeans_pca.json", "pca"),
        ("hier_svd", "labels_hier_svd.json", "svd"),
        ("hier_pca", "labels_hier_pca.json", "pca"),
        ("leiden_svd", "labels_leiden_svd.json", "svd"),
        ("leiden_pca", "labels_leiden_pca.json", "pca"),
        ("bertopic", "labels_bertopic_pca.json", "pca"),
        ("tags_null", "labels_tags_null.json", "tags"),
    ]
    for name, fname, space in hard_sources:
        data = json.loads((RESULTS_DIR / fname).read_text(encoding="utf-8"))
        interpretation["hard"][name] = interpret_hard(
            name, data["labels"], space, spaces, count, vocab, case_ids, titles, tags)

    for name in ["nmf", "lda"]:
        mem = json.loads((RESULTS_DIR / f"membership_{name}.json").read_text(encoding="utf-8"))
        interpretation["mixture"][name] = interpret_mixture(
            name, mem, count, vocab, case_ids, titles, tags)

    (RESULTS_DIR / "interpretation.json").write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- checkpoint table ----
    print("=== Stage 4: one-sentence test material ===")
    for name, data in interpretation["hard"].items():
        print(f"\n[{name}]")
        for c, cl in data["clusters"].items():
            print(f"  C{c} (n={cl['size']}): {cl['mechanical_name']}")
            print(f"     words: {' / '.join(cl['descriptor_words'][:6])}")
            print(f"     medoid: {cl['medoid']['title'][:44]}")
        if "outliers" in data:
            print(f"  outliers: {[o['title'][:20] for o in data['outliers']]}")
    for name, data in interpretation["mixture"].items():
        print(f"\n[{name} K={data['K']}]")
        for t, tp in data["topics"].items():
            reps = ", ".join(f"{r['title'][:20]}({r['ratio']})" for r in tp["representatives"][:2])
            print(f"  T{t} (n_dom={tp['size_dominant']}): {tp['mechanical_name']}")
            print(f"     words: {' / '.join(tp['descriptor_words'][:6])}")
            print(f"     reps: {reps}")

    print(f"\nSaved: {RESULTS_DIR/'interpretation.json'}")


if __name__ == "__main__":
    main()
