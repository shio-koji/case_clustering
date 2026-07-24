#!/usr/bin/env python3
"""
plan04 Summary: integrate plan01-04 into one HTML.

- existing-tag bipartite graph (case <-> tag), reproduced from plan03
- NMF soft classification for K=6 and K=12 (toggle): topic cards + member
  case lists, mixture stacked bar, vocabulary soft classification, and the
  network family (shared-word / mixture-similarity / cross-group bridges /
  case x topic bipartite / pie-marker map)
- a global "exclude archived cases" toggle that filters what is DISPLAYED
  (nodes/bars/edges/list rows) while the analysis itself (NMF fit, layouts,
  topic regions) is unchanged. Archived = title contains "アーカイブ".

Self-contained HTML. Run from the repo root:  python plan04/s6_summary.py
"""

import io
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import igraph as ig
from scipy.spatial import ConvexHull
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

SEED = 42
KS = [6, 12]
FEATURES_DIR = Path("plan04/features")
OUT = Path("plan04/report/call4_plan04_summary.html")

JP = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
      "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/YuGothic-Medium.otf")
fm.fontManager.addfont(JP)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False
def jp(s=10): return fm.FontProperties(fname=JP, size=s)

PAL = ["#C0392B", "#27AE60", "#2E86C1", "#8E44AD", "#E67E22", "#16A085",
       "#D4AC0D", "#7F8C8D", "#E84393", "#2C3E50", "#00A8A8", "#B9770E"]
TAGPAL = ["#4C8C2B", "#2B6CB0", "#B0532B", "#8A4FA8", "#C29B2C", "#2BA8A0",
          "#D81B60", "#5D6D7E", "#7CB342", "#00838F", "#6D4C41"]


def url(cid): return f"https://www.call4.jp/info.php?type=items&id={cid}"

def blend(row, colors):
    rgb = np.zeros(3)
    for t, r in enumerate(row):
        c = colors[t].lstrip("#"); rgb += r * np.array([int(c[i:i+2], 16) for i in (0, 2, 4)])
    return "#%02x%02x%02x" % tuple(int(min(v, 255)) for v in rgb)

def svg_of(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    s = buf.getvalue().decode("utf-8"); return s[s.find("<svg"):]


def build():
    tok = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tok.sort(key=lambda r: r["id"])
    stop = set(json.loads((FEATURES_DIR / "stopwords.json").read_text(encoding="utf-8"))["stopwords"])
    docs = []
    for r in tok:
        t = r["tokens"]
        docs.append([w for w in t if w not in stop]
                    + [f"{a}_{b}" for a, b in zip(t, t[1:]) if a not in stop and b not in stop])
    vec = CountVectorizer(analyzer=lambda d: d, min_df=2, max_df=0.90)
    count = vec.fit_transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(count).tocsr()
    vocab = np.array(vec.get_feature_names_out())
    meta = {"ids": [r["id"] for r in tok], "titles": [r["title"] for r in tok],
            "tags": [r.get("subject_tags", []) for r in tok],
            "arch": [int("アーカイブ" in r["title"]) for r in tok]}
    return tfidf, vocab, meta


def umap2d(n):
    p = FEATURES_DIR / "umap2d.npz"
    if p.exists():
        c = np.load(p)["coords"]
        if len(c) == n:
            return c
    import umap
    emb = np.load(FEATURES_DIR / "emb.npz")["matrix"]
    c = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.15, metric="cosine",
                  random_state=SEED).fit_transform(emb)
    np.savez_compressed(p, coords=c); return c


def fr(n, edges, weights=None, seed=SEED):
    import random as pyr; pyr.seed(seed)
    if not edges:
        return np.random.RandomState(seed).rand(n, 2)
    lay = np.array(ig.Graph(n=n, edges=edges).layout_fruchterman_reingold(
        weights=weights, niter=900).coords, dtype=float)
    lay -= lay.min(0); lay /= np.maximum(np.ptp(lay, 0), 1e-9)
    return lay


def group_gravity(dom, K, edges_sw):
    import random as pyr; pyr.seed(SEED)
    N = len(dom)
    ge, gw = [], []
    for a, b, _ in edges_sw:
        ge.append((a, b)); gw.append(3.0)
    for t in range(K):
        grp = [i for i in range(N) if dom[i] == t]
        for a, b in combinations(grp, 2):
            ge.append((a, b)); gw.append(0.35)
    lay = np.array(ig.Graph(n=N, edges=ge).layout_fruchterman_reingold(
        weights=gw, niter=1000).coords, dtype=float)
    lay -= lay.min(0); lay /= np.maximum(np.ptp(lay, 0), 1e-9)
    lay = 0.05 + lay * 0.90
    hulls = {}
    for t in range(K):
        pts = lay[[i for i in range(N) if dom[i] == t]]
        try:
            poly = pts[ConvexHull(pts).vertices]; c = poly.mean(0)
            hulls[t] = (c + (poly - c) * 1.13).round(3).tolist()
        except Exception:
            hulls[t] = None
    return lay.round(3), hulls


def chord_svg(pair_ct, sizes, names, colors, K, subtitle):
    import math
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ang = {t: 2 * math.pi * t / K + math.pi / 2 for t in range(K)}
    pos = {t: (math.cos(ang[t]), math.sin(ang[t])) for t in range(K)}
    maxc = max(pair_ct.values()) if pair_ct else 1
    for (i, j), c in pair_ct.items():
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#7a8290",
                lw=1 + 5 * c / maxc, alpha=0.45, solid_capstyle="round", zorder=1)
        mx, my = (pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2
        ax.text(mx, my, str(c), fontsize=8, ha="center", va="center", zorder=2,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#bbb"))
    for t in range(K):
        ax.scatter(*pos[t], s=280 + sizes[t] * 40, color=colors[t], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(pos[t][0] * 1.32, pos[t][1] * 1.32, names[t], ha="center", va="center",
                fontproperties=jp(8))
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title(f"K={K}: テーマ間の越境つながり（{subtitle}）", fontproperties=jp(10))
    return svg_of(fig)


def compute_for_K(K, tfidf, vocab, meta, case_words, edges_sw, coords):
    N = tfidf.shape[0]; arch = meta["arch"]
    model = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=700)
    W = model.fit_transform(tfidf); H = model.components_
    dom = W.argmax(1)
    ratios = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    Rn = normalize(ratios)
    colors = PAL[:K]
    topw = [[str(vocab[j]) for j in H[t].argsort()[-10:][::-1]] for t in range(K)]
    names = ["・".join(topw[t][:2]) for t in range(K)]
    sizes = [int((dom == t).sum()) for t in range(K)]

    sw_lay, hulls = group_gravity(dom, K, edges_sw)
    sims = Rn @ Rn.T; np.fill_diagonal(sims, -1)
    ms_edges = sorted({(min(i, int(j)), max(i, int(j)))
                       for i in range(N) for j in np.argsort(sims[i])[-3:]})
    ms_lay = fr(N, [(a, b) for a, b in ms_edges])

    cross = [(a, b, sh) for a, b, sh in edges_sw if dom[a] != dom[b]]
    def pairct(edges):
        return Counter(tuple(sorted((int(dom[a]), int(dom[b])))) for a, b, _ in edges)
    pc_full = pairct(cross)
    pc_noarch = pairct([e for e in cross if not arch[e[0]] and not arch[e[1]]])
    by_pair = defaultdict(list)
    for a, b, sh in cross:
        i, j = int(dom[a]), int(dom[b])
        if i > j: i, j, a, b = j, i, b, a
        by_pair[(i, j)].append((a, b, sh))

    bip = [(i, t) for i in range(N) for t in range(K) if ratios[i, t] > 0.15]
    bl = fr(N + K, [(a, N + b) for a, b in bip])
    bip_case, bip_topic = bl[:N], bl[N:]

    nodes = [{"i": i, "t": meta["titles"][i], "u": url(meta["ids"][i]), "f": int(dom[i]),
              "a": arch[i], "w": case_words[i][:6],
              "mix": [round(float(x), 3) for x in ratios[i]],
              "bl": blend(Rn[i] / Rn[i].sum(), colors),
              "swx": float(sw_lay[i, 0]), "swy": float(sw_lay[i, 1]),
              "msx": float(round(ms_lay[i, 0], 3)), "msy": float(round(ms_lay[i, 1], 3)),
              "bx": float(round(bip_case[i, 0], 3)), "by": float(round(bip_case[i, 1], 3)),
              "px": float(round(coords[i, 0], 2)), "py": float(round(coords[i, 1], 2))}
             for i in range(N)]

    Hn = H / np.maximum(H.sum(0, keepdims=True), 1e-12)
    cand = sorted({j for t in range(K) for j in H[t].argsort()[-20:]})
    bridge = []
    for j in cand:
        p = Hn[:, j]; order = p.argsort()[::-1]
        if p[order[1]] >= 0.25:
            bridge.append({"word": str(vocab[j]),
                           "mix": [[int(t), round(float(p[t]), 2)] for t in order[:3] if p[t] >= 0.1]})
    bridge.sort(key=lambda x: -x["mix"][1][1]); bridge = bridge[:16]

    return {"K": K, "colors": colors, "names": names, "sizes": sizes, "topw": topw,
            "dom": dom.tolist(), "nodes": nodes,
            "sw_edges": [[int(a), int(b), sh] for a, b, sh in edges_sw],
            "ms_edges": [[int(a), int(b)] for a, b in ms_edges],
            "hulls": hulls,
            "bip_edges": [[int(i), int(t)] for i, t in bip],
            "topic_pos": [[float(round(bip_topic[t, 0], 3)), float(round(bip_topic[t, 1], 3))] for t in range(K)],
            "cross": [[int(a), int(b), sh] for a, b, sh in cross],
            "chord_full": chord_svg(pc_full, sizes, names, colors, K, "全ケース"),
            "chord_noarch": chord_svg(pc_noarch, sizes, names, colors, K, "アーカイブ除外"),
            "by_pair": {f"{i}-{j}": [[int(a), int(b), sh] for a, b, sh in v] for (i, j), v in by_pair.items()},
            "pair_ct": {f"{i}-{j}": c for (i, j), c in pc_full.items()},
            "bridge_words": bridge}


def tag_bipartite(meta):
    tags = [t for t, _ in Counter(x for ts in meta["tags"] for x in ts).most_common()]
    T = len(tags); N = len(meta["ids"]); tj = {t: j for j, t in enumerate(tags)}
    M = np.zeros((N, T), int)
    for i, ts in enumerate(meta["tags"]):
        for x in ts:
            M[i, tj[x]] = 1
    edges = [(j, T + i) for i in range(N) for j in range(T) if M[i, j]]
    P = fr(T + N, edges)
    counts = M.sum(0)
    tag_nodes = [{"name": tags[j], "count": int(counts[j]), "color": TAGPAL[j % len(TAGPAL)],
                  "x": float(round(P[j, 0], 3)), "y": float(round(P[j, 1], 3))} for j in range(T)]
    case_nodes = [{"t": meta["titles"][i], "u": url(meta["ids"][i]), "a": meta["arch"][i],
                   "deg": int(M[i].sum()), "tags": meta["tags"][i],
                   "x": float(round(P[T + i, 0], 3)), "y": float(round(P[T + i, 1], 3))} for i in range(N)]
    return {"tags": tag_nodes, "cases": case_nodes,
            "edges": [[j, ci - T] for (j, ci) in edges]}


def main():
    tfidf, vocab, meta = build()
    N = tfidf.shape[0]; coords = umap2d(N)
    case_words = []
    for i in range(N):
        row = tfidf.getrow(i).toarray().ravel(); idx = row.argsort()[-20:][::-1]
        case_words.append([str(vocab[j]) for j in idx if row[j] > 0])
    wsets = [set(w) for w in case_words]
    edges_sw = []
    for a, b in combinations(range(N), 2):
        sh = [w for w in case_words[a] if w in wsets[b]]
        if len(sh) >= 2:
            edges_sw.append((a, b, sh[:6]))
    n_arch = sum(meta["arch"])
    print(f"corpus {N} (archived {n_arch}), shared-word edges {len(edges_sw)}")

    BYK = {K: compute_for_K(K, tfidf, vocab, meta, case_words, edges_sw, coords) for K in KS}
    TAGBIP = tag_bipartite(meta)

    def html_blocks(d):
        K = d["K"]
        def li_case(n, extra=""):
            cls = ' class="arch"' if n["a"] else ""
            return (f'<li{cls}><a href="{n["u"]}" target="_blank">{n["t"]}</a>{extra}'
                    f'<br><span class="cw">特徴語: {" / ".join(n["w"][:5])}</span></li>')
        def card(t):
            mem = sorted([n for n in d["nodes"] if n["f"] == t], key=lambda n: -n["mix"][t])
            lis = "".join(li_case(n, f' <span class="dim">(配合{int(n["mix"][t]*100)}%)</span>') for n in mem)
            return (f'<div class="card" style="border-top:4px solid {d["colors"][t]}">'
                    f'<h4><span class="dot" style="background:{d["colors"][t]}"></span>T{t}: {d["names"][t]} '
                    f'<span class="dim">({d["sizes"][t]}件)</span></h4>'
                    f'<div class="tw">{" / ".join(d["topw"][t][:8])}</div>'
                    f'<details><summary>このテーマのケース {d["sizes"][t]}件</summary>'
                    f'<ul class="cases">{lis}</ul></details></div>')
        cards = "".join(card(t) for t in range(K))
        perTopic = "".join(
            f'<li><span class="dot" style="background:{d["colors"][t]}"></span><b>T{t}</b>: '
            f'{" / ".join(d["topw"][t][:8])}</li>' for t in range(K))
        bw = "".join(
            f'<li>{b["word"]}: ' + " ／ ".join(
                f'<span style="color:{d["colors"][t]}">T{t} {int(fr*100)}%</span>' for t, fr in b["mix"]) + "</li>"
            for b in d["bridge_words"])
        blocks = ""
        for key, c in sorted(d["pair_ct"].items(), key=lambda x: -x[1]):
            i, j = map(int, key.split("-"))
            items = ""
            for a, b, sh in d["by_pair"][key]:
                na, nb = d["nodes"][a], d["nodes"][b]
                cls = ' class="arch"' if (na["a"] or nb["a"]) else ""
                items += (f'<li{cls}>[T{i}] <a href="{na["u"]}" target="_blank">{na["t"]}</a> × '
                          f'[T{j}] <a href="{nb["u"]}" target="_blank">{nb["t"]}</a>'
                          f'<br><span class="cw">共通語: {" / ".join(sh)}</span></li>')
            blocks += (f'<div class="pb"><h5><span class="dot" style="background:{d["colors"][i]}"></span>'
                       f'<span class="dot" style="background:{d["colors"][j]}"></span> '
                       f'{d["names"][i]} ↔ {d["names"][j]} <span class="dim">({c}組)</span></h5>'
                       f'<ul class="cases">{items}</ul></div>')
        return cards, perTopic, bw, blocks

    parts = {K: html_blocks(BYK[K]) for K in KS}
    def konly(K, s):
        return f'<div class="konly k{K}{"" if K == KS[0] else " hidden"}">{s}</div>'
    cards_html = "".join(konly(K, f'<div class="grid">{parts[K][0]}</div>') for K in KS)
    perTopic_html = "".join(konly(K, f'<ul class="soft">{parts[K][1]}</ul>') for K in KS)
    bw_html = "".join(konly(K, f'<ul class="soft">{parts[K][2]}</ul>') for K in KS)
    chord_html = "".join(
        konly(K, f'<span class="chordfull">{BYK[K]["chord_full"]}</span>'
                 f'<span class="chordnoarch hidden">{BYK[K]["chord_noarch"]}</span>') for K in KS)
    bridgelist_html = "".join(konly(K, f'<div class="bridges">{parts[K][3]}</div>') for K in KS)

    BYK_js = {str(K): {k: BYK[K][k] for k in
                       ("K", "colors", "names", "sizes", "dom", "nodes", "sw_edges",
                        "ms_edges", "hulls", "bip_edges", "topic_pos", "cross")} for K in KS}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.format(
        cards=cards_html, perTopic=perTopic_html, bridgewords=bw_html,
        chord=chord_html, bridgelist=bridgelist_html,
        byk=json.dumps(BYK_js, ensure_ascii=False), ks=json.dumps(KS),
        tagbip=json.dumps(TAGBIP, ensure_ascii=False), narch=n_arch, ntotal=N),
        encoding="utf-8")
    print(f"saved: {OUT} ({OUT.stat().st_size//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>plan04 総合レポート（タグ二部グラフ＋NMF K=6/12＋アーカイブ除外）</title>
<style>
body{{font-family:"Hiragino Sans",sans-serif;margin:0;background:#f7f6f3;color:#222;line-height:1.8}}
.c{{max-width:1050px;margin:0 auto;padding:24px 20px 70px}}
h1{{font-size:1.4em;border-bottom:3px solid #2E86C1;padding-bottom:9px}}
h2{{font-size:1.2em;margin-top:2em;border-left:5px solid #2E86C1;padding-left:10px}}
h3{{font-size:1.03em;margin-top:1.4em}}
.dim{{color:#778;font-size:.85em}} .hidden{{display:none}}
.bar{{position:sticky;top:0;background:#eef3fa;border-bottom:1px solid #ccd;padding:10px 14px;z-index:6;display:flex;flex-wrap:wrap;gap:16px;align-items:center}}
.kb{{background:#e2e8f0;border:1px solid #bcd;border-radius:6px;padding:5px 14px;margin:2px;cursor:pointer;font-family:inherit;font-size:.95em}}
.kb.active{{background:#2E86C1;color:#fff;font-weight:bold}}
.cnt{{font-size:.85em;color:#456}}
.rule{{background:#eef6f0;border:1px solid #bcdcc8;border-radius:8px;padding:9px 14px;font-size:.88em;margin:8px 0}}.rule b{{color:#2f7a4d}}
.card{{background:#fff;border-radius:8px;padding:9px 13px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}}
.tw{{font-size:.88em;color:#444}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:6px;margin-right:4px;vertical-align:middle}}
.fig{{background:#fff;border-radius:8px;padding:12px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow-x:auto}}
canvas{{max-width:100%}}
.soft{{columns:2;font-size:.9em}}.soft li{{margin:3px 0;break-inside:avoid}}
.bridges{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}}
.pb{{background:#fff;border:1px solid #e6ded3;border-radius:8px;padding:8px 12px}}.pb h5{{margin:.2em 0;font-size:.95em}}
.cases{{font-size:.85em}}.cw{{color:#8a5a2b;font-size:.85em}}
.panel{{margin-top:8px;padding:9px 13px;background:#faf7f2;border:1px solid #e2d8cc;border-radius:8px;font-size:.88em;min-height:2em}}
#tip{{position:fixed;display:none;background:rgba(20,25,35,.94);color:#fff;padding:8px 11px;border-radius:6px;font-size:12px;max-width:340px;pointer-events:none;z-index:10}}
a{{color:#2E86C1}}
body.hidearch li.arch{{display:none}}
label.chk{{font-size:.92em;cursor:pointer;user-select:none}}
</style></head><body><div class="c">

<h1>CALL4 訴訟の分類：総合レポート（plan01–04）</h1>
<p class="dim">既存タグの二部グラフ＋テキスト由来のNMFソフト分類（K=6 / K=12）。
「アーカイブ除外」は<b>解析結果（配置・トピック）はそのままに、表示からアーカイブ済みケースだけを隠します</b>。</p>

<div class="bar">
  <span><b>トピック数:</b> <span id="kbtns"></span></span>
  <label class="chk"><input type="checkbox" id="archchk"> アーカイブ済みを除外して表示</label>
  <span class="cnt" id="cnt"></span>
</div>

<h2>1. 既存タグの二部グラフ（ケース ↔ タグ）</h2>
<div class="rule"><b>構成ルール：</b>■＝タグ（11種・色つき四角・大きさ＝付与ケース数）、●＝ケース（灰色丸・大きさ＝付与タグ数）。
<b>線＝そのケースがそのタグを持つ</b>。配置は力学レイアウト。plan03で構築したものの再掲（相互作用版）。
ケースにカーソルで保有タグ、クリックでCALL4へ。</div>
<div class="fig"><canvas id="tagbip" width="1010" height="620"></canvas></div>

<h2>2. NMFトピック一覧（テーマ内のケース一覧つき）</h2>
<p class="dim">テーマ名は各トピックの最重み上位2語による機械ラベル（LLM要約ではない）。各カードの折りたたみでケース一覧。</p>
{cards}

<h2>3. 配合スタックバー（各ケースのテーマ混合比）</h2>
<div class="rule"><b>構成ルール：</b>棒1本＝1ケース。NMFの重みWを合計1に正規化した配合を積み上げ、優勢テーマ順に整列。</div>
<div class="fig"><div id="lg-stack" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px"></div>
<canvas id="stack" width="1010" height="300"></canvas></div>

<h2>4. 語彙のソフト分類</h2>
<div class="rule"><b>構成ルール：</b>NMFのH（テーマ×語）を使用。各語は全テーマに重みを持つ。
「橋渡し語」＝H列を正規化して2テーマ以上に25%超で分かれる語。</div>
<h3>各テーマの高重み語</h3>{perTopic}
<h3>橋渡し語（複数テーマにまたがる語）</h3>{bridgewords}

<h2>5. ネットワーク図（各構成ルール付き）</h2>

<h3>5a. 共通語ネットワーク</h3>
<div class="rule"><b>構成ルール：</b>色＝主テーマ。<b>線＝特徴語(TF-IDF上位20語)を2語以上共有</b>。
グループ重力配置＋薄い背景＝テーマ範囲。点クリックで共通語つき隣接一覧。</div>
<div class="fig"><div id="lg-net" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px"></div>
<canvas id="swnet" width="1010" height="560"></canvas><div id="sw-panel" class="panel">点クリックで共通語つき隣接ケース。</div></div>

<h3>5b. 混合類似ネットワーク</h3>
<div class="rule"><b>構成ルール：</b>色＝配合ブレンド。<b>線＝配合ベクトルのコサインで近い3件</b>（意味的近さ、共通語とは別基準）。</div>
<div class="fig"><canvas id="msnet" width="1010" height="540"></canvas></div>

<h3>5c. 越境ビュー（テーマを越える共通語つながり）</h3>
<div class="rule"><b>構成ルール：</b>5aの共通語リンクのうち<b>主テーマが異なるペアだけ</b>。コード図＝越境件数（太さ）。
アーカイブ除外時はコード図・一覧とも除外後の数に切り替わります。</div>
<div class="fig">{chord}</div>
{bridgelist}

<h3>5d. ケース×トピック 2部グラフ</h3>
<div class="rule"><b>構成ルール：</b>大丸＝テーマ、小丸＝ケース。<b>線＝配合15%超</b>（太さ＝配合）。複数本＝横断ケース。</div>
<div class="fig"><canvas id="bipnet" width="1010" height="600"></canvas></div>

<h3>5e. パイマーカー地図（UMAP）</h3>
<div class="rule"><b>構成ルール：</b>配置＝意味埋め込みのUMAP2D。各点を配合の円グラフで描画。近接＋別配色＝文章は似るが論点構成が違う。</div>
<div class="fig"><canvas id="piemap" width="1010" height="540"></canvas></div>

<p class="dim">アーカイブ＝タイトルに「アーカイブ」を含むケース（{narch}/{ntotal}件）。本分類は解析目的で当事者類型の評価を意図しません。</p>
</div><div id="tip"></div>

<script>
const BYK={byk}, KS={ks}, TAGBIP={tagbip}, NARCH={narch}, NTOTAL={ntotal};
let curK=String(KS[0]), hideArch=false;
const tip=document.getElementById('tip');
function showTip(h,ev){{tip.innerHTML=h;tip.style.display='block';tip.style.left=Math.min(ev.clientX+14,innerWidth-360)+'px';tip.style.top=(ev.clientY+12)+'px';}}
function hideTip(){{tip.style.display='none';}}
function D(){{return BYK[curK];}}
function vis(n){{return !(hideArch&&n.a);}}
function mixText(n){{const d=D();return n.mix.map((v,t)=>v>0.08?`<span style="color:${{d.colors[t]}}">■</span>${{d.names[t]}} ${{(v*100).toFixed(0)}}%`:null).filter(Boolean).join('<br>');}}
function wtext(n){{return n.w&&n.w.length?`<br><span style="color:#e0b48a">特徴語: ${{n.w.join(' / ')}}</span>`:'';}}
function legend(elid){{const d=D(),el=document.getElementById(elid);if(!el)return;el.innerHTML='';d.names.forEach((nm,t)=>el.insertAdjacentHTML('beforeend',`<span style="font-size:11px"><span style="display:inline-block;width:10px;height:10px;background:${{d.colors[t]}};margin-right:3px;border-radius:5px"></span>T${{t}} ${{nm}}</span>`));}}
function sc(cv,xs,ys,pad){{const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);return (x,y)=>[pad+(x-x0)/((x1-x0)||1)*(cv.width-2*pad),pad+(y-y0)/((y1-y0)||1)*(cv.height-2*pad)];}}
function nearest(cv,pos,ev,arr){{const r=cv.getBoundingClientRect(),mx=(ev.clientX-r.left)*(cv.width/r.width),my=(ev.clientY-r.top)*(cv.height/r.height);let best=null,bd=200;(arr||D().nodes).forEach(n=>{{if(!vis(n))return;const [x,y]=pos(n);const dd=(x-mx)**2+(y-my)**2;if(dd<bd){{bd=dd;best=n;}}}});return best;}}

// 1. tag bipartite
function drawTag(){{const cv=document.getElementById('tagbip'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const allx=TAGBIP.tags.map(t=>t.x).concat(TAGBIP.cases.map(c=>c.x)),ally=TAGBIP.tags.map(t=>t.y).concat(TAGBIP.cases.map(c=>c.y));
  const S=sc(cv,allx,ally,40),tp=j=>S(TAGBIP.tags[j].x,TAGBIP.tags[j].y),cp=i=>S(TAGBIP.cases[i].x,TAGBIP.cases[i].y);
  x.strokeStyle='rgba(190,195,205,.5)';x.lineWidth=.5;
  TAGBIP.edges.forEach(([j,i])=>{{if(hideArch&&TAGBIP.cases[i].a)return;const [x1,y1]=tp(j),[x2,y2]=cp(i);x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}});
  TAGBIP.cases.forEach((c,i)=>{{if(hideArch&&c.a)return;const [X,Y]=cp(i);x.beginPath();x.arc(X,Y,3.5+c.deg*1.6,0,7);x.fillStyle='#8593a6';x.globalAlpha=.85;x.fill();x.globalAlpha=1;x.strokeStyle='#fff';x.lineWidth=.5;x.stroke();}});
  TAGBIP.tags.forEach((t,j)=>{{const [X,Y]=tp(j);const s=9+t.count*.5;x.fillStyle=t.color;x.fillRect(X-s,Y-s,2*s,2*s);x.strokeStyle='#fff';x.lineWidth=1.5;x.strokeRect(X-s,Y-s,2*s,2*s);x.fillStyle='#fff';x.font='bold 11px sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(t.name,X,Y);x.textAlign='left';}});
  cv.onmousemove=ev=>{{const r=cv.getBoundingClientRect(),mx=(ev.clientX-r.left)*(cv.width/r.width),my=(ev.clientY-r.top)*(cv.height/r.height);
    let best=null,bd=260,bi=-1;TAGBIP.cases.forEach((c,i)=>{{if(hideArch&&c.a)return;const [X,Y]=cp(i);const dd=(X-mx)**2+(Y-my)**2;if(dd<bd){{bd=dd;best=c;bi=i;}}}});
    if(!best){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{best.t}}</b><br>タグ: ${{best.tags.join(' / ')||'—'}}${{best.a?'<br><span style=color:#e88>【アーカイブ】</span>':''}}`,ev);}};
  cv.onmouseleave=hideTip;
  cv.onclick=ev=>{{const r=cv.getBoundingClientRect(),mx=(ev.clientX-r.left)*(cv.width/r.width),my=(ev.clientY-r.top)*(cv.height/r.height);let best=null,bd=260;TAGBIP.cases.forEach((c,i)=>{{if(hideArch&&c.a)return;const [X,Y]=cp(i);const dd=(X-mx)**2+(Y-my)**2;if(dd<bd){{bd=dd;best=c;}}}});if(best)open(best.u,'_blank');}};}}

// 3. stacked bar
function drawStack(){{const d=D(),cv=document.getElementById('stack'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const order=d.nodes.filter(vis).slice().sort((a,b)=>(a.f-b.f)||(b.mix[b.f]-a.mix[a.f]));
  const BW=(cv.width-16)/Math.max(order.length,1);
  order.forEach((n,pos)=>{{let y=cv.height-14;n.mix.forEach((r,t)=>{{const h=r*(cv.height-24);x.fillStyle=d.colors[t];x.fillRect(10+pos*BW,y-h,Math.max(BW-.5,1),h);y-=h;}});}});
  cv.onmousemove=ev=>{{const r=cv.getBoundingClientRect(),p=Math.floor(((ev.clientX-r.left)*(cv.width/r.width)-10)/BW);if(p<0||p>=order.length){{hideTip();return;}}showTip(`<b>${{order[p].t}}</b><br>${{mixText(order[p])}}`,ev);cv.style.cursor='pointer';}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const r=cv.getBoundingClientRect(),p=Math.floor(((ev.clientX-r.left)*(cv.width/r.width)-10)/BW);if(p>=0&&p<order.length)open(order[p].u,'_blank');}};
  legend('lg-stack');}}

// 5a shared-word
let swsel=null;
function swNb(i){{const o=[];D().sw_edges.forEach(([a,b,sh])=>{{if(a===i)o.push([b,sh]);else if(b===i)o.push([a,sh]);}});return o.filter(([j])=>vis(D().nodes[j]));}}
function drawSW(){{const d=D(),cv=document.getElementById('swnet'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const pos=n=>[26+n.swx*(cv.width-52),n.swy*(cv.height-26)];
  for(const t in d.hulls){{const poly=d.hulls[t];if(!poly)continue;x.beginPath();poly.forEach((p,k)=>{{const [X,Y]=pos({{swx:p[0],swy:p[1]}});k?x.lineTo(X,Y):x.moveTo(X,Y);}});x.closePath();const c=d.colors[t].replace('#','');x.fillStyle=`rgba(${{parseInt(c.slice(0,2),16)}},${{parseInt(c.slice(2,4),16)}},${{parseInt(c.slice(4,6),16)}},.08)`;x.fill();}}
  const nb=swsel!=null?new Set(swNb(swsel).map(e=>e[0])):null;
  d.sw_edges.forEach(([a,b])=>{{if(!vis(d.nodes[a])||!vis(d.nodes[b]))return;if(swsel!=null&&a!==swsel&&b!==swsel)return;const [x1,y1]=pos(d.nodes[a]),[x2,y2]=pos(d.nodes[b]);x.strokeStyle=swsel!=null?'rgba(70,80,100,.6)':'rgba(140,145,155,.28)';x.lineWidth=swsel!=null?1.5:.8;x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}});
  d.nodes.forEach(n=>{{if(!vis(n))return;const [X,Y]=pos(n);const dim=swsel!=null&&n.i!==swsel&&!(nb&&nb.has(n.i));x.globalAlpha=dim?.15:1;x.beginPath();x.arc(X,Y,n.i===swsel?8:5.5,0,7);x.fillStyle=d.colors[n.f];x.fill();x.globalAlpha=1;x.strokeStyle=n.i===swsel?'#222':'#fff';x.lineWidth=n.i===swsel?2:1;x.stroke();}});
  legend('lg-net');
  cv.onmousemove=ev=>{{const n=nearest(cv,pos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br><span style="color:${{d.colors[n.f]}}">主: ${{d.names[n.f]}}</span>${{wtext(n)}}`,ev);}};
  cv.onmouseleave=hideTip;
  cv.onclick=ev=>{{const n=nearest(cv,pos,ev);swsel=n?n.i:null;drawSW();const P=document.getElementById('sw-panel');if(swsel==null){{P.innerHTML='点クリックで共通語つき隣接ケース。';return;}}const nb=swNb(swsel).sort((a,b)=>b[1].length-a[1].length);P.innerHTML=nb.length?`<b>「${{n.t}}」と特徴語を共有</b><ul class="cases">`+nb.map(([j,sh])=>`<li><a href="${{d.nodes[j].u}}" target="_blank">${{d.nodes[j].t}}</a><br><span class="cw">共通語: ${{sh.join(' / ')}}</span></li>`).join('')+'</ul>':'共有相手なし（表示中）';}};}}

// 5b mixture-similarity
function drawMS(){{const d=D(),cv=document.getElementById('msnet'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const pos=n=>[26+n.msx*(cv.width-52),26+n.msy*(cv.height-52)];
  x.strokeStyle='rgba(150,155,165,.35)';x.lineWidth=.9;
  d.ms_edges.forEach(([a,b])=>{{if(!vis(d.nodes[a])||!vis(d.nodes[b]))return;const [x1,y1]=pos(d.nodes[a]),[x2,y2]=pos(d.nodes[b]);x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}});
  d.nodes.forEach(n=>{{if(!vis(n))return;const [X,Y]=pos(n);x.beginPath();x.arc(X,Y,6,0,7);x.fillStyle=n.bl;x.fill();x.strokeStyle='#fff';x.lineWidth=1;x.stroke();}});
  cv.onmousemove=ev=>{{const n=nearest(cv,pos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br>${{mixText(n)}}${{wtext(n)}}`,ev);}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const n=nearest(cv,pos,ev);if(n)open(n.u,'_blank');}};}}

// 5d bipartite case-topic
function drawBip(){{const d=D(),cv=document.getElementById('bipnet'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const xs=d.nodes.map(n=>n.bx).concat(d.topic_pos.map(p=>p[0])),ys=d.nodes.map(n=>n.by).concat(d.topic_pos.map(p=>p[1]));
  const S=sc(cv,xs,ys,34),cpos=n=>S(n.bx,n.by),tpos=t=>S(d.topic_pos[t][0],d.topic_pos[t][1]);
  d.bip_edges.forEach(([i,t])=>{{if(!vis(d.nodes[i]))return;const [x1,y1]=cpos(d.nodes[i]),[x2,y2]=tpos(t);const c=d.colors[t].replace('#','');x.strokeStyle=`rgba(${{parseInt(c.slice(0,2),16)}},${{parseInt(c.slice(2,4),16)}},${{parseInt(c.slice(4,6),16)}},.35)`;x.lineWidth=Math.max(d.nodes[i].mix[t]*5,.5);x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}});
  d.nodes.forEach(n=>{{if(!vis(n))return;const [X,Y]=cpos(n);x.beginPath();x.arc(X,Y,4.5,0,7);x.fillStyle=d.colors[n.f];x.fill();x.strokeStyle='#fff';x.lineWidth=.8;x.stroke();}});
  d.topic_pos.forEach((p,t)=>{{const [X,Y]=tpos(t);x.beginPath();x.arc(X,Y,16,0,7);x.fillStyle=d.colors[t];x.fill();x.strokeStyle='#fff';x.lineWidth=2;x.stroke();x.fillStyle='#fff';x.font='bold 11px sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText('T'+t,X,Y);x.textAlign='left';}});
  cv.onmousemove=ev=>{{const n=nearest(cv,cpos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br>${{mixText(n)}}`,ev);}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const n=nearest(cv,cpos,ev);if(n)open(n.u,'_blank');}};}}

// 5e pie map
function drawPie(){{const d=D(),cv=document.getElementById('piemap'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const S=sc(cv,d.nodes.map(n=>n.px),d.nodes.map(n=>n.py),28),pos=n=>S(n.px,n.py);
  d.nodes.forEach(n=>{{if(!vis(n))return;const [X,Y]=pos(n);let a0=-Math.PI/2;n.mix.forEach((r,t)=>{{if(r<.03)return;const a1=a0+r*2*Math.PI;x.beginPath();x.moveTo(X,Y);x.arc(X,Y,8,a0,a1);x.closePath();x.fillStyle=d.colors[t];x.fill();a0=a1;}});x.beginPath();x.arc(X,Y,8,0,7);x.strokeStyle='#fff';x.lineWidth=1;x.stroke();}});
  cv.onmousemove=ev=>{{const n=nearest(cv,pos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br>${{mixText(n)}}`,ev);}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const n=nearest(cv,pos,ev);if(n)open(n.u,'_blank');}};}}

function updateCnt(){{const shown=hideArch?NTOTAL-NARCH:NTOTAL;document.getElementById('cnt').textContent=`表示 ${{shown}} / 全 ${{NTOTAL}}件`+(hideArch?`（アーカイブ${{NARCH}}件を非表示）`:'');}}
function redraw(){{swsel=null;drawTag();drawStack();drawSW();drawMS();drawBip();drawPie();updateCnt();}}
function applyArchClass(){{document.body.classList.toggle('hidearch',hideArch);
  document.querySelectorAll('.chordfull').forEach(e=>e.classList.toggle('hidden',hideArch));
  document.querySelectorAll('.chordnoarch').forEach(e=>e.classList.toggle('hidden',!hideArch));}}
function setK(K){{curK=String(K);document.querySelectorAll('.konly').forEach(e=>e.classList.add('hidden'));document.querySelectorAll('.k'+K).forEach(e=>e.classList.remove('hidden'));document.querySelectorAll('.kb').forEach(b=>b.classList.toggle('active',b.dataset.k==K));applyArchClass();redraw();}}
const kb=document.getElementById('kbtns');
KS.forEach(K=>kb.insertAdjacentHTML('beforeend',`<button class="kb" data-k="${{K}}">K=${{K}}</button>`));
kb.addEventListener('click',e=>{{if(e.target.dataset.k)setK(e.target.dataset.k);}});
document.getElementById('archchk').addEventListener('change',e=>{{hideArch=e.target.checked;applyArchClass();redraw();}});
setK(KS[0]);
</script></body></html>"""


if __name__ == "__main__":
    main()
