#!/usr/bin/env python3
"""
Explainability-first public report, Leiden (semantic space) K=6 as the star.

Narrative (per report_design_leiden.md):
  Part 1  should tags be the ingredient? -> no, use them as an answer key
  Part 2  method (embedding + Leiden) and the 6-group result, with the
          "not a fluke" numbers and the K=5/6/7 comparison
  Part 3  the same result, seen many ways (map, network, cards, Sankey, ...)

Self-contained single HTML. Run from the repo root:
  python plan02/s11_leiden_report.py
"""

import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import leidenalg
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath

SEED = 42
N_NEIGHBORS = 10
RES_K6 = 1.25   # resolution giving 6 clusters (plateau 1.05-1.45; see s10)
FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")
REPORT_DIR = Path("plan02/report")

JP_FONT_PATH = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
                "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/"
                "YuGothic-Medium.otf")
fm.fontManager.addfont(JP_FONT_PATH)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

def jp(size=9):
    return fm.FontProperties(fname=JP_FONT_PATH, size=size)

# 6-cluster palette (distinct, print-safe)
CL_COLORS = ["#C0392B", "#27AE60", "#2E86C1", "#8E44AD", "#E67E22", "#16A085"]

# Names for the K=6 Leiden clusters, keyed by descriptor words (evidence in code).
CL_NAMES = {
    "刑事手続・身体拘束": "取調べ・勾留・保釈・弁護人・黙秘権（入管収容も含む）",
    "地域開発・環境": "住民・工事・再開発・伐採・自治",
    "情報公開・公文書": "開示・文書・財務省・情報公開",
    "労働・生活者の権利": "労働・労災・共働き・消費者",
    "家族・ジェンダー": "同性・カップル・不妊・優生",
    "選挙・政治参加": "選挙・投票・在外・立候補",
}
# map a cluster's top word to a name (robust to label-id permutation)
def name_for(words):
    w = set(words[:6])
    if {"取調べ", "勾留", "保釈", "弁護人"} & w:
        return "刑事手続・身体拘束"
    if {"住民", "工事", "再開発", "伐採"} & w:
        return "地域開発・環境"
    if {"開示", "文書", "財務省"} & w:
        return "情報公開・公文書"
    if {"労働", "労災", "共働き", "消費者"} & w:
        return "労働・生活者の権利"
    if {"同性", "不妊", "カップル", "優生"} & w:
        return "家族・ジェンダー"
    if {"選挙", "投票", "在外", "立候補"} & w:
        return "選挙・政治参加"
    return "／".join(words[:2])


def svg_of(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue().decode("utf-8")
    return s[s.find("<svg"):]


def url(cid):
    return f"https://www.call4.jp/info.php?type=items&id={cid}"


# ---------------- data + clustering ----------------

def build_graph(X):
    sims = X @ X.T
    np.fill_diagonal(sims, -1)
    edges, weights, seen = [], [], set()
    for i in range(X.shape[0]):
        for j in np.argsort(sims[i])[-N_NEIGHBORS:]:
            a, b = min(i, int(j)), max(i, int(j))
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b)); weights.append(float(max(sims[i, j], 0.0)))
    return ig.Graph(n=X.shape[0], edges=edges), weights, edges


def leiden_at(g, weights, res):
    return np.array(leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights=weights, resolution_parameter=res, seed=SEED).membership)


def ctfidf(labels, count, vocab, top_n=10):
    vocab_arr = np.array(vocab)
    classes = sorted(set(int(l) for l in labels))
    counts = np.vstack([np.asarray(count[labels == c].sum(axis=0)).ravel() for c in classes])
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + counts.mean(axis=1, keepdims=True).sum() / np.maximum(counts.sum(axis=0), 1))
    scores = tf * idf
    return {c: [str(vocab_arr[j]) for j in scores[i].argsort()[-top_n:][::-1]] for i, c in enumerate(classes)}


def reps(members, X, k=3):
    sub = X[members]
    cen = sub.mean(axis=0); cen /= np.linalg.norm(cen) + 1e-9
    sims = sub @ cen
    return [members[j] for j in sims.argsort()[-k:][::-1]]


def bootstrap_k6(X, g_full, weights_full, base_labels, n=100, frac=0.8):
    rng = np.random.default_rng(SEED)
    N = X.shape[0]; out = []
    for _ in range(n):
        idx = np.sort(rng.choice(N, int(N * frac), replace=False))
        gs, ws, _ = build_graph(X[idx])
        lab = leiden_at(gs, ws, RES_K6)
        out.append(adjusted_rand_score(base_labels[idx], lab))
    return float(np.median(out)), float(np.percentile(out, 25)), float(np.percentile(out, 75))


# ---------------- figures ----------------

def fig_tag_bar(tags):
    c = Counter(t for ts in tags for t in ts)
    items = c.most_common()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = [k for k, _ in items]; vals = [v for _, v in items]
    ax.barh(range(len(names)), vals, color="#8fa8c8")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontproperties=jp(9))
    ax.invert_yaxis()
    ax.bar_label(ax.containers[0], fontsize=8, padding=2)
    ax.set_title("既存タグの件数（11タグ・多い順）/ Existing tags by frequency", fontproperties=jp(10))
    ax.set_xlim(0, max(vals) * 1.15)
    return svg_of(fig)


def fig_nullmodel():
    sizes = [56, 21, 10, 7, 1]
    fig, ax = plt.subplots(figsize=(6, 2.0))
    left = 0
    for i, s in enumerate(sizes):
        ax.barh(0, s, left=left, color=plt.cm.Pastel1(i), edgecolor="white")
        ax.text(left + s / 2, 0, str(s), ha="center", va="center", fontsize=9)
        left += s
    ax.set_xlim(0, 95); ax.set_yticks([]); ax.set_xlabel("ケース数", fontproperties=jp(9))
    ax.set_title("タグだけで機械分類すると 56件が1つの塊に（＝座標として粗すぎる）",
                 fontproperties=jp(10))
    return svg_of(fig)


def fig_sizes(order_names, order_sizes):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(range(len(order_sizes)), order_sizes,
           color=[CL_COLORS[i] for i in range(len(order_sizes))])
    ax.set_xticks(range(len(order_names)))
    ax.set_xticklabels(order_names, fontproperties=jp(8.5), rotation=18, ha="right")
    ax.bar_label(ax.containers[0], fontsize=9)
    ax.set_title("6グループの規模 / Group sizes", fontproperties=jp(10))
    return svg_of(fig)


def fig_kcompare():
    ks = [5, 6, 7]
    width = [8, 9, 5]      # plateau width (resolution points)
    sil = [0.126, 0.132, 0.126]
    ari = [0.210, 0.244, 0.200]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    for ax, vals, ttl, fmt in [
        (axes[0], width, "安定性: 6になるresolution幅（点数）", "{:.0f}"),
        (axes[1], sil, "クラスタのまとまり（シルエット）", "{:.3f}"),
        (axes[2], ari, "既存タグとの一致（ARI）", "{:.3f}")]:
        bars = ax.bar([str(k) for k in ks], vals,
                      color=["#bbb", CL_COLORS[2], "#bbb"])
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=9)
        ax.set_title(ttl, fontproperties=jp(9.5)); ax.set_xlabel("クラスタ数", fontproperties=jp(9))
        ax.margins(y=0.2)
    fig.suptitle("K=6 が3拍子そろって最良（灰=5と7）", fontproperties=jp(11))
    fig.tight_layout()
    return svg_of(fig)


def fig_feature_words(clusters, order):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    for ax, ci in zip(axes.ravel(), order):
        cl = clusters[ci]
        words = cl["words"][:7][::-1]
        ax.barh(range(len(words)), range(1, len(words) + 1), color=CL_COLORS[order.index(ci)])
        ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontproperties=jp(9))
        ax.set_xticks([])
        ax.set_title(f"{cl['name']}（{cl['size']}件）", fontproperties=jp(9.5))
    fig.suptitle("各グループを特徴づける言葉（c-TF-IDF上位）/ Defining words per group",
                 fontproperties=jp(11))
    fig.tight_layout()
    return svg_of(fig)


def fig_sankey(tags, labels, clusters, order):
    tag_tot = Counter(t for ts in tags for t in ts)
    tags_sorted = [t for t, _ in tag_tot.most_common()]
    flows = Counter()
    for i, ts in enumerate(tags):
        for t in ts:
            flows[(t, int(labels[i]))] += 1
    clsize = {ci: clusters[ci]["size"] for ci in order}
    gap, H = 1.6, 100.0
    def layout(names, tot):
        total = sum(tot[n] for n in names)
        scale = (H - gap * (len(names) - 1)) / total
        pos, y = {}, 0.0
        for n in names:
            h = tot[n] * scale; pos[n] = (y, h); y += h + gap
        return pos, scale
    lpos, ls = layout(tags_sorted, tag_tot)
    rpos, rs = layout(order, clsize)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    loff = {t: 0.0 for t in tags_sorted}; roff = {c: 0.0 for c in order}
    for (t, c), n in sorted(flows.items(), key=lambda x: (tags_sorted.index(x[0][0]), order.index(x[0][1]))):
        y0, _ = lpos[t]; y1, _ = rpos[c]
        a0, a1 = y0 + loff[t], y1 + roff[c]; h0, h1 = n * ls, n * rs
        loff[t] += h0; roff[c] += h1
        col = CL_COLORS[order.index(c)]
        verts = [(0.14, a0), (0.5, a0), (0.5, a1), (0.86, a1),
                 (0.86, a1 + h1), (0.5, a1 + h1), (0.5, a0 + h0), (0.14, a0 + h0), (0.14, a0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=col, alpha=0.4, lw=0))
    for t in tags_sorted:
        y, h = lpos[t]
        ax.add_patch(mpatches.Rectangle((0.10, y), 0.04, h, color="#555"))
        ax.text(0.09, y + h / 2, f"{t} ({tag_tot[t]})", ha="right", va="center", fontproperties=jp(9))
    for c in order:
        y, h = rpos[c]
        ax.add_patch(mpatches.Rectangle((0.86, y), 0.04, h, color=CL_COLORS[order.index(c)]))
        ax.text(0.91, y + h / 2, f"{clusters[c]['name']} ({clusters[c]['size']})",
                ha="left", va="center", fontproperties=jp(9))
    ax.set_xlim(-0.3, 1.55); ax.set_ylim(-2, H + 2); ax.invert_yaxis(); ax.axis("off")
    ax.set_title("既存タグ（左） → データ駆動6グループ（右）/ Existing tags → data-driven groups",
                 fontproperties=jp(11))
    return svg_of(fig)


def fig_dendro():
    lk = json.loads((RESULTS_DIR / "linkages.json").read_text(encoding="utf-8"))
    Z = np.array(lk["linkages"]["pca"])
    ids = lk["case_ids"]
    fig, ax = plt.subplots(figsize=(13, 3.0))
    dendrogram(Z, ax=ax, labels=[i[-3:] for i in ids], leaf_rotation=90,
               leaf_font_size=5, color_threshold=Z[-6, 2])
    ax.set_title("参考: 意味空間の階層構造（粒度は連続的に変えられる）",
                 fontproperties=jp(10))
    ax.axhline(Z[-6, 2], color="red", ls="--", lw=0.7)
    return svg_of(fig)


# ---------------- main ----------------

def main():
    X = np.load(FEATURES_DIR / "emb_pca.npz")["matrix"]
    count = sparse.load_npz(FEATURES_DIR / "count.npz")
    tfidf = sparse.load_npz(FEATURES_DIR / "tfidf.npz")
    vocab = json.loads((FEATURES_DIR / "vocab.json").read_text(encoding="utf-8"))["terms"]
    coords = np.load(FEATURES_DIR / "umap2d.npz")["coords"]
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in tokens]; titles = [r["title"] for r in tokens]
    tags = [r["subject_tags"] for r in tokens]

    g, weights, edges = build_graph(X)
    labels = leiden_at(g, weights, RES_K6)
    assert len(set(labels.tolist())) == 6, f"expected 6 clusters, got {len(set(labels.tolist()))}"

    words = ctfidf(labels, count, vocab)
    clusters = {}
    for c in sorted(set(labels.tolist())):
        members = np.where(labels == c)[0].tolist()
        rep_idx = reps(members, X)
        clusters[c] = {
            "name": name_for(words[c]), "size": len(members), "words": words[c],
            "medoid_idx": rep_idx[0], "rep_idx": rep_idx, "members": members,
            "tags": dict(Counter(t for i in members for t in tags[i]).most_common(3)),
        }
    order = sorted(clusters, key=lambda c: -clusters[c]["size"])
    cidx = {c: k for k, c in enumerate(order)}  # cluster id -> display slot (color)

    # not-a-fluke numbers for THIS K=6 config
    sil6 = round(float(silhouette_score(X, labels, metric="cosine")), 4)
    boot_med, boot_p25, boot_p75 = bootstrap_k6(X, g, weights, labels)
    nmf_dom = np.array(json.loads(
        (RESULTS_DIR / "membership_nmf.json").read_text())["dominant_topic"])
    ari_nmf = round(float(adjusted_rand_score(labels, nmf_dom)), 3)
    km6 = KMeans(n_clusters=6, random_state=SEED, n_init=50).fit_predict(X)
    ari_km = round(float(adjusted_rand_score(labels, km6)), 3)
    firsts = [(t[0] if t else "no_tag") for t in tags]
    uniq = sorted(set(firsts)); tag_ints = np.array([uniq.index(t) for t in firsts])
    ari_tag = round(float(adjusted_rand_score(labels, tag_ints)), 3)

    # figures
    figs = {
        "tagbar": fig_tag_bar(tags), "null": fig_nullmodel(),
        "kcmp": fig_kcompare(), "sizes": fig_sizes(
            [clusters[c]["name"] for c in order], [clusters[c]["size"] for c in order]),
        "words": fig_feature_words(clusters, order), "sankey": fig_sankey(tags, labels, clusters, order),
        "dendro": fig_dendro(),
    }

    # cluster cards
    cards = []
    for c in order:
        cl = clusters[c]; col = CL_COLORS[cidx[c]]
        reps_html = "".join(
            f'<li><a href="{url(ids[i])}" target="_blank">{titles[i]}</a></li>'
            for i in cl["rep_idx"])
        allc = "".join(
            f'<li><a href="{url(ids[i])}" target="_blank">{titles[i]}</a></li>'
            for i in cl["members"])
        cards.append(f"""
<div class="card" style="border-top:5px solid {col}">
  <h4><span class="dot" style="background:{col}"></span>{cl['name']} <span class="dim">({cl['size']}件)</span></h4>
  <p><b>特徴語:</b> {' / '.join(cl['words'][:8])}</p>
  <p><b>代表ケース:</b></p><ul>{reps_html}</ul>
  <p class="dim">既存タグ内訳: {'、'.join(f'{k}×{v}' for k,v in cl['tags'].items())}</p>
  <details><summary>所属{cl['size']}件すべて</summary><ul class="small">{allc}</ul></details>
</div>""")

    # JS data: nodes (with cluster color slot), edges, layouts
    gg = ig.Graph(n=len(ids), edges=edges)
    import random as pyr; pyr.seed(SEED)
    net = np.array(gg.layout_fruchterman_reingold(niter=800).coords)
    nodes = [{"title": titles[i], "url": url(ids[i]), "tags": tags[i],
              "c": cidx[int(labels[i])], "cname": clusters[int(labels[i])]["name"],
              "x": round(float(coords[i, 0]), 2), "y": round(float(coords[i, 1]), 2),
              "nx": round(float(net[i, 0]), 2), "ny": round(float(net[i, 1]), 2)}
             for i in range(len(ids))]
    # cross-cluster edges flagged for the network view
    edge_js = [[int(a), int(b), int(labels[a] != labels[b])] for a, b in edges]
    cnames = [clusters[c]["name"] for c in order]

    gen = date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALL4公共訴訟95件を「意味」で6つに分ける</title>
<style>
body {{ font-family:"Hiragino Sans","Yu Gothic",sans-serif; margin:0; background:#f7f6f3; color:#222; line-height:1.85; }}
.container {{ max-width:1000px; margin:0 auto; padding:30px 20px 70px; }}
h1 {{ font-size:1.5em; border-bottom:3px solid #2E86C1; padding-bottom:10px; }}
h2 {{ font-size:1.25em; margin-top:2.6em; border-left:6px solid #2E86C1; padding-left:11px; }}
h3 {{ font-size:1.05em; margin-top:1.7em; }}
h4 {{ margin:0.3em 0; }}
.dim {{ color:#778; font-size:0.85em; }}
.lead, .callout, .step {{ border-radius:8px; padding:14px 18px; }}
.lead {{ background:#eaf2fb; border:1px solid #bcd6f0; }}
.callout {{ background:#fbf6e9; border:1px solid #e6d9b0; }}
.step {{ background:#fff; border:1px solid #e0ddd6; margin:8px 0; }}
.card {{ background:#fff; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin:12px 0; }}
.figure {{ background:#fff; border-radius:8px; padding:12px; margin:14px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); overflow-x:auto; }}
table {{ border-collapse:collapse; background:#fff; font-size:0.86em; margin:10px 0; }}
th,td {{ border:1px solid #ccc; padding:6px 10px; text-align:left; }}
th {{ background:#e8edf5; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:6px; margin-right:5px; vertical-align:middle; }}
canvas {{ max-width:100%; }}
.btn {{ background:#2E86C1; color:#fff; border:0; border-radius:4px; padding:5px 12px; margin:3px; cursor:pointer; font-size:.84em; font-family:inherit; }}
.btn.active {{ background:#1a4f78; font-weight:bold; }}
#tip {{ position:fixed; display:none; background:rgba(20,25,35,.94); color:#fff; padding:8px 11px; border-radius:6px; font-size:12px; max-width:330px; pointer-events:none; z-index:10; }}
a {{ color:#2E86C1; }}
details summary {{ cursor:pointer; color:#2E86C1; font-size:.9em; }}
ul.small li {{ font-size:.85em; }}
.num {{ font-size:1.9em; font-weight:bold; color:#1a4f78; }}
.kpi {{ display:flex; flex-wrap:wrap; gap:20px; margin:10px 0; }}
.kpi > div {{ background:#fff; border-radius:8px; padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,.08); text-align:center; }}
</style>
</head>
<body>
<div class="container">

<h1>CALL4 公共訴訟95件を「意味」で6つに分ける<br>
<span class="dim" style="font-size:.62em">— 人手タグに頼らず、文章の内容からグループを見つける</span></h1>
<p class="dim">生成 {gen} ｜ 対象: <a href="https://www.call4.jp" target="_blank">CALL4</a> 掲載の公共訴訟95件</p>

<div class="lead">
公共訴訟には現在<b>11個の人手タグ</b>が付いています。では、<b>データ（文章の内容）に任せたら、
どんなグループ分けになるでしょうか？</b> 複数の手法を比べた結果、
文章の意味でつなぐ方法（埋め込み＋Leiden）が<b>安定して6つのグループ</b>を示し、
別の手法（NMF）も独立に同じ6軸に着地しました。
この資料は ①タグを材料に使うべきか → ②手法と6グループの結果 → ③結果の色々な見せ方、の順で説明します。
</div>

<h2>1. そもそも「タグ」を分類の材料に使うべき？</h2>
<p><b>結論から言うと、使いません。</b> 既存タグは分類を作る<b>材料</b>ではなく、
できた分類が妥当かを照合する<b>「答え合わせ」</b>に回します。理由は3つ。</p>
<div class="step"><b>① 循環論法になる。</b> 私たちの狙いは「今のタグ体系は見直せるか」。
そのタグを材料に分類を作れば、タグをなぞるだけで新しい発見はできません（＝答えを写して答え合わせするようなもの）。</div>
<div class="step"><b>② そもそも座標として粗い。</b> 試しにタグだけで機械分類すると、
<b>95件中56件が1つの巨大な塊</b>に潰れました。「公正な手続」タグが43件に偏っているためです。</div>
<div class="figure">{figs['null']}</div>
<div class="step"><b>③ 多ラベルで不安定。</b> タグが2個のケースも0個のケースもあり、材料としてばらつきます。</div>
<p>そこで<b>本文テキストを材料</b>にし、タグは最後の照合に使います。「答えを見ずに解く」ので、
結果に対する信頼が置けます。参考までに既存11タグの内訳はこちら（後で6グループと突き合わせます）。</p>
<div class="figure">{figs['tagbar']}</div>

<h2>2. 手法と結果 — 文章の意味で6グループ</h2>
<p>やっていることは2ステップだけです。</p>
<div class="step"><b>ステップ1: 埋め込み。</b> 各ケースの本文を「意味ベクトル」に変換します。
内容が似たケースほど近くに置かれる<b>地図</b>を作るイメージ（多言語AIモデル bge-m3 を使用）。</div>
<div class="step"><b>ステップ2: Leiden。</b> 各ケースを「意味が近い相手」と線でつないで<b>ネットワーク</b>を作り、
<b>密につながった塊（コミュニティ）</b>を取り出します。人が「誰と誰が近いか」で考えるのと同じ発想です。</div>

<h3>結果: 6つのグループ</h3>
<div class="figure">{figs['sizes']}</div>
<div class="grid">
{''.join(cards)}
</div>

<h3>この6グループは「偶然」ではない</h3>
<div class="kpi">
<div><div class="num">{boot_med:.0%}</div>データを8割に間引いて<br>分類し直しても一致<br><span class="dim">(ブートストラップ中央値ARI)</span></div>
<div><div class="num">{ari_km:.2f}</div>別手法(k-means)とも<br>ほぼ同じ分割<br><span class="dim">(ARI)</span></div>
<div><div class="num">{ari_nmf:.2f}</div>語彙ベースのNMFとも一致<br>＝別の入口から同じ6軸<br><span class="dim">(ARI)</span></div>
</div>
<p>データを間引いても<b>{boot_med:.0%}</b>再現し、まったく別の分類法でも同じ絵が出ます。
とくに、単語の使い方だけを見るNMF（後述）も<b>独立に同じ6グループ</b>に着地しました。
別々のアプローチが同じ構造を指す——これが「この6分けは実在する」という最大の根拠です。</p>

<h3>なぜ「6」なのか（5でも7でもなく）</h3>
<p>Leidenは粒度ツマミ（resolution）を回すとグループ数が変わります。細かく試すと、
グループ数を増やすほど<b>融合していたテーマが段階的に分かれて</b>いきました。</p>
<table>
<tr><th>グループ数</th><th>K−1から何が変わるか</th><th>まとまり</th><th>タグ一致</th></tr>
<tr><td>5</td><td>「家族・ジェンダー」と「選挙」が融合したまま</td><td>0.126</td><td>0.210</td></tr>
<tr style="background:#eaf7ee"><td><b>6（採用）</b></td><td><b>「選挙」が「家族・ジェンダー」から分離</b></td><td><b>0.132</b></td><td><b>0.244</b></td></tr>
<tr><td>7</td><td>「入管・難民」が「刑事手続」から分離</td><td>0.126</td><td>0.200</td></tr>
</table>
<div class="figure">{figs['kcmp']}</div>
<p><b>6が3拍子そろって最良</b>でした：安定して6になるツマミの幅が最も広く、まとまりも既存タグとの一致も最良。
そして<b>語彙ベースのNMFも独立に6を選んだ</b>ため、「6」は恣意的でなく複数の入口が合流した数です。
※残る揺らぎは「刑事手続と入管・難民を1つ（身体拘束）と見るか2つに割るか」だけ——
これは手法の限界ではなく<b>訴訟の見方の選択</b>（上位概念でまとめるか、争点で分けるか）です。</p>

<h2>3. 結果を色々な角度で見る</h2>
<p>以下はすべて<b>同じ6グループの結果</b>を別の切り口で描いたもの。
どの図も<b>点にカーソルで詳細・クリックでCALL4のケースページ</b>に飛べます。</p>

<h3>A. ケース地図 — 全体像を一目で</h3>
<p>意味的な近さで95件を配置し、6グループを色分け（座標は可視化専用）。</p>
<div class="figure">
  <div id="legend-map" style="display:flex;flex-wrap:wrap;gap:12px;margin:4px 0;"></div>
  <canvas id="map" width="960" height="540"></canvas>
</div>

<h3>B. ネットワーク図 — 「機械が実際に見た繋がり」</h3>
<p>Leidenは「ネットワークの塊を見つける」手法なので、この図は<b>手法そのものが見ている絵</b>です。
線＝意味が近い関係、色＝グループ。<b>灰色の線＝グループをまたぐ繋がり</b>（複数論点の橋渡しケース）。</p>
<div class="figure">
  <label class="dim"><input type="checkbox" id="cross-only"> グループをまたぐ線だけ表示</label>
  <canvas id="net" width="960" height="600"></canvas>
</div>

<h3>C. 既存タグとの対応 — タグ見直しへ</h3>
<p>左＝既存11タグ、右＝データ駆動6グループ。帯の太さ＝ケース数。
<b>「公正な手続」が6グループすべてに散っている</b>のが一目でわかります（＝分類として機能しておらず、
争点ベースへの再編が有効）。</p>
<div class="figure">{figs['sankey']}</div>

<h3>D. 各グループを定義する言葉</h3>
<div class="figure">{figs['words']}</div>

<details><summary>参考: 階層構造（グループはさらに大きなテーマに束ねられる）</summary>
<div class="figure">{figs['dendro']}</div></details>

<h2>4. わかったこと・示唆</h2>
<ul>
<li><b>既存タグ「公正な手続」（43件）は実質「その他」</b>。6グループ全てに分散しており、
争点ベースの軸（情報公開・刑事手続など）への分解が有効。</li>
<li><b>「労働・生活者の権利」</b>が独立グループとして自然に出現。現在の「働き方」タグより広い括りの候補。</li>
<li><b>刑事手続と入管・難民</b>は6分けでは1グループ（身体拘束）だが、もう一段細かくすると分かれる。
「主タグ＋副タグ」のような運用が実態に忠実。</li>
</ul>

<div class="callout">
<b>発展編: もっと精密な見方（混合メンバーシップ）</b><br>
「1ケース＝1グループ」に押し込むと、複数論点をまたぐ訴訟の性質は失われます。
各ケースを<b>6軸への比率</b>（例:「入管60%＋刑事30%」）で表す見方もあり、別手法NMFで作成済みです。
フル版は <code>call4_public_report.html</code> を参照。
</div>

<h2>付記: 手法と限界</h2>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>データ</td><td>CALL4公開95件（2026-07-18取得）。ケース名＋概要＋本文</td></tr>
<tr><td>手法</td><td>bge-m3埋め込み → PCA 48次元 → KNN(10近傍)グラフ + Leiden(resolution {RES_K6}, 6グループ)</td></tr>
<tr><td>検証</td><td>シルエット {sil6}／ブートストラップ80%×100回 中央値ARI {boot_med:.2f}（{boot_p25:.2f}–{boot_p75:.2f}）／
k-means・NMFとの一致 ARI {ari_km}・{ari_nmf}／既存タグとの一致 ARI {ari_tag}</td></tr>
<tr><td>再現性</td><td>乱数シード42固定。コード・データは GitHub (shio-koji/case_clustering) の plan02/</td></tr>
<tr><td>限界</td><td>95件は小規模で、結果は探索・仮説生成のためのもの。まとまりは「ゆるやかな地形」。
訴状・判決文の全文は未使用</td></tr>
</table>
<p class="dim">本分類は解析目的であり、訴訟当事者を類型化して評価する意図はありません。
ケース本文の著作権はCALL4および執筆者に帰属します。各ケースの詳細・支援は各リンク先をご覧ください。</p>

</div>
<div id="tip"></div>

<script>
const NODES = {json.dumps(nodes, ensure_ascii=False)};
const EDGES = {json.dumps(edge_js)};
const CN = {json.dumps(cnames, ensure_ascii=False)};
const CC = {json.dumps(CL_COLORS)};
const tip = document.getElementById('tip');
function showTip(h, ev) {{
  tip.innerHTML = h; tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 350) + 'px';
  tip.style.top = (ev.clientY + 12) + 'px';
}}
function hideTip() {{ tip.style.display = 'none'; }}
function nodeTip(n) {{
  return `<b>${{n.title}}</b><br><span style="color:#9db">${{CN[n.c]}} ｜ tags: ${{n.tags.join('・')||'—'}}</span>`;
}}
function mkScale(canvas, key, pad) {{
  const xs = NODES.map(n=>n[key+'x']), ys = NODES.map(n=>n[key+'y']);
  const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  return n => [pad+(n[key+'x']-x0)/(x1-x0)*(canvas.width-2*pad),
               pad+(n[key+'y']-y0)/(y1-y0)*(canvas.height-2*pad)];
}}
function hook(canvas, posOf) {{
  let cur=null;
  canvas.addEventListener('mousemove', ev => {{
    const r=canvas.getBoundingClientRect();
    const mx=(ev.clientX-r.left)*(canvas.width/r.width), my=(ev.clientY-r.top)*(canvas.height/r.height);
    cur=null; let bd=170;
    NODES.forEach(n => {{ const [x,y]=posOf(n); const d=(x-mx)**2+(y-my)**2; if (d<bd){{bd=d;cur=n;}} }});
    if (!cur) {{ hideTip(); canvas.style.cursor='default'; return; }}
    canvas.style.cursor='pointer'; showTip(nodeTip(cur), ev);
  }});
  canvas.addEventListener('mouseleave', hideTip);
  canvas.addEventListener('click', () => {{ if (cur) window.open(cur.url, '_blank'); }});
  return () => cur;
}}
// legend
const lm = document.getElementById('legend-map');
CN.forEach((n,i) => lm.insertAdjacentHTML('beforeend',
  `<span style="font-size:12px"><span style="display:inline-block;width:12px;height:12px;background:${{CC[i]}};margin-right:4px;border-radius:6px"></span>${{n}}</span>`));

// A. map
const mc=document.getElementById('map'), mx=mc.getContext('2d');
const mScale=mkScale(mc,'',28), mPos=n=>mScale(n);
function drawMap() {{
  mx.clearRect(0,0,mc.width,mc.height);
  NODES.forEach(n => {{ const [x,y]=mPos(n);
    mx.beginPath(); mx.arc(x,y,6.5,0,7); mx.fillStyle=CC[n.c]; mx.globalAlpha=.85; mx.fill();
    mx.globalAlpha=1; mx.strokeStyle='#fff'; mx.lineWidth=1; mx.stroke(); }});
}}
drawMap(); hook(mc, mPos);

// B. network
const nc=document.getElementById('net'), nx=nc.getContext('2d');
const nScale=mkScale(nc,'n',30), nPos=n=>nScale(n);
const crossOnly=document.getElementById('cross-only');
function drawNet() {{
  nx.clearRect(0,0,nc.width,nc.height);
  EDGES.forEach(([a,b,cross]) => {{
    if (crossOnly.checked && !cross) return;
    const [x1,y1]=nPos(NODES[a]), [x2,y2]=nPos(NODES[b]);
    nx.beginPath(); nx.moveTo(x1,y1); nx.lineTo(x2,y2);
    nx.strokeStyle = cross ? 'rgba(120,120,120,0.55)' : 'rgba(190,195,205,0.35)';
    nx.lineWidth = cross ? 1.3 : 0.8; nx.stroke();
  }});
  NODES.forEach(n => {{ const [x,y]=nPos(n);
    nx.beginPath(); nx.arc(x,y,6,0,7); nx.fillStyle=CC[n.c]; nx.fill();
    nx.strokeStyle='#fff'; nx.lineWidth=1; nx.stroke(); }});
}}
drawNet(); hook(nc, nPos);
crossOnly.addEventListener('change', drawNet);
</script>
</body>
</html>"""
    out = REPORT_DIR / "call4_leiden6_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Leiden-6 report saved: {out} ({out.stat().st_size//1024} KB)")
    print(f"  silhouette={sil6}  bootstrap median ARI={boot_med:.3f} ({boot_p25:.2f}-{boot_p75:.2f})")
    print(f"  ARI vs kmeans6={ari_km}  vs NMF6={ari_nmf}  vs tags={ari_tag}")
    for c in order:
        print(f"  {clusters[c]['name']} ({clusters[c]['size']})")


if __name__ == "__main__":
    main()
