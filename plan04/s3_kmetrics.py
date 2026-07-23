#!/usr/bin/env python3
"""
plan04 Stage 3: quantitative K-selection metrics (there is no elbow).

For K=4..14, compute indicators that CAN guide K when reconstruction error
cannot:
  - coherence (UMass): do a topic's top words actually co-occur? higher(->0)=better
  - stability: refit NMF with init='random' over many seeds; mean pairwise ARI of
    dominant-topic assignments. drops once K exceeds the "natural" granularity.
  - redundancy: mean pairwise cosine of topic-word vectors (H rows). rises when
    topics start duplicating -> K too large. lower=better.
  - tag alignment (reference): ARI of dominant topic vs first existing tag.

Outputs a 4-panel figure + table (plan04/report/call4_plan04_kmetrics.html).
Run from the repo root:  python plan04/s3_kmetrics.py
"""

import io
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

KMIN, KMAX = 4, 14
N_SEEDS = 20
FEATURES_DIR = Path("plan04/features")
OUT = Path("plan04/report/call4_plan04_kmetrics.html")

JP_FONT_PATH = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
                "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/"
                "YuGothic-Medium.otf")
fm.fontManager.addfont(JP_FONT_PATH)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

def jp(size=10):
    return fm.FontProperties(fname=JP_FONT_PATH, size=size)


def build():
    tok = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tok.sort(key=lambda r: r["id"])
    stop = set(json.loads((FEATURES_DIR / "stopwords.json").read_text(encoding="utf-8"))["stopwords"])
    docs = []
    for r in tok:
        t = r["tokens"]
        uni = [w for w in t if w not in stop]
        bi = [f"{a}_{b}" for a, b in zip(t, t[1:]) if a not in stop and b not in stop]
        docs.append(uni + bi)
    vec = CountVectorizer(analyzer=lambda d: d, min_df=2, max_df=0.90)
    count = vec.fit_transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(count)
    firsts = [(r["subject_tags"][0] if r["subject_tags"] else "no_tag") for r in tok]
    uniq = sorted(set(firsts))
    tag_ints = np.array([uniq.index(x) for x in firsts])
    return tfidf, count, tag_ints


def umass_coherence(H, B, top_m=10):
    """Average UMass coherence over topics. B = binary doc-term (CSC)."""
    Df = np.asarray((B > 0).sum(axis=0)).ravel()
    vals = []
    for row in H:
        idx = row.argsort()[-top_m:][::-1]
        sub = B[:, idx].toarray()  # N x M
        co = sub.T @ sub           # M x M co-doc counts
        s = 0.0
        for a in range(1, len(idx)):
            for b in range(a):
                s += np.log((co[a, b] + 1) / max(Df[idx[b]], 1))
        vals.append(s)
    return float(np.mean(vals))


def redundancy(H):
    Hn = normalize(H)
    S = Hn @ Hn.T
    iu = np.triu_indices(len(H), k=1)
    return float(S[iu].mean())


def main():
    tfidf, count, tag_ints = build()
    B = (count > 0).tocsc()
    Ks = list(range(KMIN, KMAX + 1))
    coh, red, stab, tagari = [], [], [], []

    for K in Ks:
        base = NMF(n_components=K, init="nndsvda", random_state=42, max_iter=700).fit(tfidf)
        H = base.components_
        W = base.transform(tfidf)
        coh.append(umass_coherence(H, B))
        red.append(redundancy(H))
        tagari.append(round(float(adjusted_rand_score(W.argmax(1), tag_ints)), 4))
        # stability: random-init refits across seeds -> mean pairwise ARI of dominant labels
        labs = []
        for s in range(N_SEEDS):
            m = NMF(n_components=K, init="random", random_state=s, max_iter=400).fit_transform(tfidf)
            labs.append(m.argmax(1))
        aris = [adjusted_rand_score(labs[i], labs[j]) for i, j in combinations(range(N_SEEDS), 2)]
        stab.append(float(np.mean(aris)))
        print(f"K={K:2d}  coherence={coh[-1]:8.2f}  stability={stab[-1]:.3f}  "
              f"redundancy={red[-1]:.3f}  tagARI={tagari[-1]:.3f}")

    Path("plan04/results/kmetrics.json").write_text(json.dumps(
        {"Ks": Ks, "coherence": coh, "stability": stab, "redundancy": red,
         "tag_ari": tagari, "n_seeds": N_SEEDS}, ensure_ascii=False, indent=1), encoding="utf-8")

    # figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # kind: "peak" -> mark argmax (metric has a genuine optimum);
    #       "mono" -> monotone in K, annotate that it can't bound K
    panels = [
        (ax[0, 0], coh, "整合性 coherence（→0で良い）", "#2E86C1", "mono"),
        (ax[0, 1], stab, "安定性 stability（seed間ARI・高いほど良い）", "#27AE60", "peak"),
        (ax[1, 0], red, "冗長性 redundancy（低いほど良い）", "#C0392B", "mono"),
        (ax[1, 1], tagari, "既存タグとの一致 ARI（参考）", "#8E44AD", "peak"),
    ]
    for a, vals, title, c, kind in panels:
        a.plot(Ks, vals, "o-", color=c)
        a.set_xticks(Ks); a.set_xlabel("K", fontproperties=jp(9))
        a.set_title(title, fontproperties=jp(10)); a.grid(alpha=0.3)
        if kind == "peak":
            best = Ks[int(np.argmax(vals))]
            a.axvline(best, color=c, ls="--", lw=1.2, alpha=0.7)
            a.text(best, max(vals), f" ピーク K={best}", color=c, fontproperties=jp(9),
                   va="top")
        else:
            a.text(0.5, 0.06, "単調（Kを上げ続けると改善）＝K上限の判断には非力",
                   transform=a.transAxes, ha="center", fontproperties=jp(8), color="#888")
    fig.suptitle("plan04: トピック数K の選択指標（エルボーの代わり）", fontproperties=jp(13))
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    svg = buf.getvalue().decode("utf-8"); svg = svg[svg.find("<svg"):]

    rows = "".join(
        f"<tr><td>{K}</td><td>{coh[i]:.2f}</td><td>{stab[i]:.3f}</td>"
        f"<td>{red[i]:.3f}</td><td>{tagari[i]:.3f}</td></tr>"
        for i, K in enumerate(Ks))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>plan04 K選択指標</title>
<style>body{{font-family:"Hiragino Sans",sans-serif;margin:0;background:#f7f6f3;color:#222;line-height:1.8}}
.c{{max-width:1000px;margin:0 auto;padding:26px 20px 60px}}h1{{font-size:1.35em;border-bottom:3px solid #2E86C1;padding-bottom:8px}}
.fig{{background:#fff;border-radius:8px;padding:12px;margin:14px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow-x:auto}}
table{{border-collapse:collapse;background:#fff;font-size:.9em}}th,td{{border:1px solid #ccc;padding:5px 10px;text-align:right}}
th{{background:#e8edf5}}.note{{background:#eef3fa;border:1px solid #c9d8ee;border-radius:8px;padding:12px 16px}}</style>
</head><body><div class="c">
<h1>plan04: トピック数 K の選択指標</h1>
<div class="note">再構成誤差にエルボーが無いため別指標で判断。結果、指標は2種に分かれた：<br>
<b>①ピークを持つ指標</b>＝<b>安定性</b>（seedを変えた再現性）と<b>タグ一致</b>は、ともに <b>K=6 が最良</b>
（K=6は最も頑健で、plan02と同じ着地）。<br>
<b>②単調な指標</b>＝整合性・冗長性はKとともに改善し続け、<b>上限の歯止めにはならない</b>
（トピックが細かく専門特化するほど良く見えるだけ）。<br>
→ <b>客観的頑健性なら K=6</b>。<b>タグより細かい探索が目的なら K=11〜12</b>
（安定性が持ち直し・整合性も高く・中身も解釈可能。K=14は安定性最低＋断片化で非推奨）。
最終判断は各トピックの解釈可能性（系譜図・各K一覧）と併せて。</div>
<div class="fig">{svg}</div>
<table><tr><th>K</th><th>coherence↑</th><th>stability↑</th><th>redundancy↓</th><th>tagARI</th></tr>{rows}</table>
<p style="color:#778;font-size:.85em">seed={N_SEEDS}回(init=random)でstability測定。coherence=UMass(top10語)。redundancy=H行の平均コサイン。</p>
</div></body></html>""", encoding="utf-8")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
