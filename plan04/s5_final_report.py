#!/usr/bin/env python3
"""
plan04 Report B: final NMF report for K=6 AND K=12 (toggle), with
- topic cards, mixture stacked bar
- vocabulary soft classification (per-topic top words + bridge-word table)
- several network views we have tried, EACH annotated with its construction rule:
    (a) shared-word network   (b) mixture-similarity network
    (c) cross-group bridges: chord + labeled bridge list
    (d) case x topic bipartite (e) pie-marker UMAP map

Self-contained HTML. Run from the repo root:  python plan04/s5_final_report.py
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
from scipy import sparse
from scipy.spatial import ConvexHull
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

SEED = 42
KS = [6, 12]
FEATURES_DIR = Path("plan04/features")
OUT = Path("plan04/report/call4_plan04_final.html")

JP = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
      "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/YuGothic-Medium.otf")
fm.fontManager.addfont(JP)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False
def jp(s=10): return fm.FontProperties(fname=JP, size=s)

PAL = ["#C0392B", "#27AE60", "#2E86C1", "#8E44AD", "#E67E22", "#16A085",
       "#D4AC0D", "#7F8C8D", "#E84393", "#2C3E50", "#00A8A8", "#B9770E"]


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
            "tags": [r["subject_tags"] for r in tok]}
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
    g = ig.Graph(n=n, edges=edges)
    lay = np.array(g.layout_fruchterman_reingold(weights=weights, niter=800).coords, dtype=float)
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


def chord_svg(pair_ct, sizes, names, colors, K):
    import math
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
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
        ax.scatter(*pos[t], s=300 + sizes[t] * 40, color=colors[t], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(pos[t][0] * 1.32, pos[t][1] * 1.32, names[t], ha="center", va="center",
                fontproperties=jp(8))
    lim = 1.7
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title(f"K={K}: テーマ間の越境つながり（線の太さ＝件数）", fontproperties=jp(10))
    return svg_of(fig)


def compute_for_K(K, tfidf, vocab, meta, case_words, wsets, edges_sw, coords):
    N = tfidf.shape[0]
    model = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=700)
    W = model.fit_transform(tfidf); H = model.components_
    dom = W.argmax(1)
    ratios = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    Rn = normalize(ratios)
    colors = PAL[:K]
    topw = [[str(vocab[j]) for j in H[t].argsort()[-10:][::-1]] for t in range(K)]
    names = ["・".join(topw[t][:2]) for t in range(K)]
    sizes = [int((dom == t).sum()) for t in range(K)]

    # (a) shared-word layout (group gravity by dominant topic)
    sw_lay, hulls = group_gravity(dom, K, edges_sw)

    # (b) mixture-similarity network: kNN top-3 by mixture cosine
    sims = Rn @ Rn.T; np.fill_diagonal(sims, -1)
    ms_edges = sorted({(min(i, int(j)), max(i, int(j)))
                       for i in range(N) for j in np.argsort(sims[i])[-3:]})
    ms_lay = fr(N, [(a, b) for a, b in ms_edges])

    # (c) cross-group bridges (shared-word edges whose dominant topic differs)
    cross = [(a, b, sh) for a, b, sh in edges_sw if dom[a] != dom[b]]
    pair_ct = Counter(tuple(sorted((int(dom[a]), int(dom[b])))) for a, b, _ in cross)
    by_pair = defaultdict(list)
    for a, b, sh in cross:
        i, j = int(dom[a]), int(dom[b])
        if i > j: i, j, a, b = j, i, b, a
        by_pair[(i, j)].append((a, b, sh))

    # (d) bipartite case-topic (ratio>0.15)
    bip = [(i, t) for i in range(N) for t in range(K) if ratios[i, t] > 0.15]
    bl = fr(N + K, [(a, N + b) for a, b in bip])
    bip_case, bip_topic = bl[:N], bl[N:]

    blends = [blend(Rn[i] / Rn[i].sum(), colors) for i in range(N)]
    dom_l = dom.tolist()
    nodes = [{"i": i, "t": meta["titles"][i], "u": url(meta["ids"][i]), "f": int(dom[i]),
              "w": case_words[i][:6], "mix": [round(float(x), 3) for x in ratios[i]],
              "bl": blends[i],
              "swx": float(sw_lay[i, 0]), "swy": float(sw_lay[i, 1]),
              "msx": float(round(ms_lay[i, 0], 3)), "msy": float(round(ms_lay[i, 1], 3)),
              "bx": float(round(bip_case[i, 0], 3)), "by": float(round(bip_case[i, 1], 3)),
              "px": float(round(coords[i, 0], 2)), "py": float(round(coords[i, 1], 2))}
             for i in range(N)]

    # vocabulary soft classification: bridge words (span >=2 topics)
    Hn = H / np.maximum(H.sum(0, keepdims=True), 1e-12)  # each word's distribution over topics
    cand = sorted({j for t in range(K) for j in H[t].argsort()[-20:]})
    bridge = []
    for j in cand:
        p = Hn[:, j]; order = p.argsort()[::-1]
        if p[order[1]] >= 0.25:
            bridge.append({"word": str(vocab[j]),
                           "mix": [[int(t), round(float(p[t]), 2)] for t in order[:3] if p[t] >= 0.1]})
    bridge.sort(key=lambda x: -x["mix"][1][1]); bridge = bridge[:16]

    return {"K": K, "colors": colors, "names": names, "sizes": sizes, "topw": topw,
            "dom": dom_l, "nodes": nodes,
            "sw_edges": [[int(a), int(b), sh] for a, b, sh in edges_sw],
            "ms_edges": [[int(a), int(b)] for a, b in ms_edges],
            "hulls": hulls,
            "bip_edges": [[int(i), int(t)] for i, t in bip],
            "topic_pos": [[float(round(bip_topic[t, 0], 3)), float(round(bip_topic[t, 1], 3))] for t in range(K)],
            "cross": [[int(a), int(b), sh] for a, b, sh in cross],
            "chord": chord_svg(pair_ct, sizes, names, colors, K),
            "bridge_list_pairs": {f"{i}-{j}": [[int(a), int(b), sh] for a, b, sh in v]
                                  for (i, j), v in by_pair.items()},
            "pair_ct": {f"{i}-{j}": c for (i, j), c in pair_ct.items()},
            "bridge_words": bridge}


def main():
    tfidf, vocab, meta = build()
    N = tfidf.shape[0]
    coords = umap2d(N)
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
    print(f"corpus {N}, shared-word edges {len(edges_sw)}")

    BYK = {}
    for K in KS:
        BYK[K] = compute_for_K(K, tfidf, vocab, meta, case_words, wsets, edges_sw, coords)
        print(f"K={K}: topics named, cross={len(BYK[K]['cross'])}, bridge-words={len(BYK[K]['bridge_words'])}")

    # ---- build per-K HTML blocks (cards, soft-vocab, chord, bridge list) ----
    def html_blocks(d):
        K = d["K"]
        cards = "".join(
            f'<div class="card" style="border-top:4px solid {d["colors"][t]}">'
            f'<h4><span class="dot" style="background:{d["colors"][t]}"></span>T{t}: {d["names"][t]} '
            f'<span class="dim">({d["sizes"][t]}件)</span></h4>'
            f'<div class="tw">{" / ".join(d["topw"][t][:8])}</div></div>' for t in range(K))
        # soft vocab: per-topic words + bridge words
        perTopic = "".join(
            f'<li><span class="dot" style="background:{d["colors"][t]}"></span>'
            f'<b>T{t}</b>: {" / ".join(d["topw"][t][:8])}</li>' for t in range(K))
        bw = "".join(
            f'<li>{b["word"]}: ' + " ／ ".join(
                f'<span style="color:{d["colors"][t]}">T{t} {int(fr*100)}%</span>' for t, fr in b["mix"])
            + "</li>" for b in d["bridge_words"])
        # bridge list by pair (desc by count)
        pairs = sorted(d["pair_ct"].items(), key=lambda x: -x[1])
        blocks = ""
        for key, c in pairs:
            i, j = map(int, key.split("-"))
            items = "".join(
                f'<li>[T{i}] <a href="{d["nodes"][a]["u"]}" target="_blank">{d["nodes"][a]["t"]}</a> × '
                f'[T{j}] <a href="{d["nodes"][b]["u"]}" target="_blank">{d["nodes"][b]["t"]}</a>'
                f'<br><span class="cw">共通語: {" / ".join(sh)}</span></li>'
                for a, b, sh in d["bridge_list_pairs"][key])
            blocks += (f'<div class="pb"><h5><span class="dot" style="background:{d["colors"][i]}"></span>'
                       f'<span class="dot" style="background:{d["colors"][j]}"></span> '
                       f'{d["names"][i]} ↔ {d["names"][j]} <span class="dim">({c}組)</span></h5>'
                       f'<ul class="cases">{items}</ul></div>')
        return cards, perTopic, bw, blocks

    parts = {}
    for K in KS:
        parts[K] = html_blocks(BYK[K])

    def konly(K, s):
        return f'<div class="konly k{K}"{"" if K == KS[0] else " hidden"}>{s}</div>'

    cards_html = "".join(konly(K, f'<div class="grid">{parts[K][0]}</div>') for K in KS)
    perTopic_html = "".join(konly(K, f'<ul class="soft">{parts[K][1]}</ul>') for K in KS)
    bridge_words_html = "".join(konly(K, f'<ul class="soft">{parts[K][2]}</ul>') for K in KS)
    chord_html = "".join(konly(K, BYK[K]["chord"]) for K in KS)
    bridge_list_html = "".join(konly(K, f'<div class="bridges">{parts[K][3]}</div>') for K in KS)

    BYK_js = {str(K): {k: BYK[K][k] for k in
                       ("K", "colors", "names", "sizes", "dom", "nodes", "sw_edges",
                        "ms_edges", "hulls", "bip_edges", "topic_pos", "cross")} for K in KS}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.format(
        cards=cards_html, perTopic=perTopic_html, bridgewords=bridge_words_html,
        chord=chord_html, bridgelist=bridge_list_html,
        byk=json.dumps(BYK_js, ensure_ascii=False), ks=json.dumps(KS)),
        encoding="utf-8")
    print(f"saved: {OUT} ({OUT.stat().st_size//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>plan04 最終レポート K=6 / K=12</title>
<style>
body{{font-family:"Hiragino Sans",sans-serif;margin:0;background:#f7f6f3;color:#222;line-height:1.8}}
.c{{max-width:1040px;margin:0 auto;padding:26px 20px 70px}}
h1{{font-size:1.4em;border-bottom:3px solid #2E86C1;padding-bottom:9px}}
h2{{font-size:1.2em;margin-top:2em;border-left:5px solid #2E86C1;padding-left:10px}}
h3{{font-size:1.03em;margin-top:1.5em}}
.dim{{color:#778;font-size:.85em}} .hidden{{display:none}}
.kbar{{position:sticky;top:0;background:#f7f6f3;padding:10px 0;z-index:5;border-bottom:1px solid #ddd}}
.kb{{background:#e8edf5;border:1px solid #bcd;border-radius:6px;padding:6px 16px;margin:3px;cursor:pointer;font-size:1em;font-family:inherit}}
.kb.active{{background:#2E86C1;color:#fff;font-weight:bold}}
.rule{{background:#eef6f0;border:1px solid #bcdcc8;border-radius:8px;padding:9px 14px;font-size:.88em;margin:8px 0}}
.rule b{{color:#2f7a4d}}
.card{{background:#fff;border-radius:8px;padding:9px 13px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}}
.tw{{font-size:.88em;color:#444}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:6px;margin-right:4px;vertical-align:middle}}
.fig{{background:#fff;border-radius:8px;padding:12px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow-x:auto}}
canvas{{max-width:100%}}
.soft{{columns:2;font-size:.9em}} .soft li{{margin:3px 0;break-inside:avoid}}
.bridges{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}}
.pb{{background:#fff;border:1px solid #e6ded3;border-radius:8px;padding:8px 12px}} .pb h5{{margin:.2em 0;font-size:.95em}}
.cases{{font-size:.85em}} .cw{{color:#8a5a2b;font-size:.85em}}
.panel{{margin-top:8px;padding:9px 13px;background:#faf7f2;border:1px solid #e2d8cc;border-radius:8px;font-size:.88em;min-height:2em}}
#tip{{position:fixed;display:none;background:rgba(20,25,35,.94);color:#fff;padding:8px 11px;border-radius:6px;font-size:12px;max-width:340px;pointer-events:none;z-index:10}}
a{{color:#2E86C1}}
</style></head><body><div class="c">

<h1>plan04 最終レポート：NMFソフト分類（K=6 / K=12 切替）</h1>
<p class="dim">TF-IDF＋NMF ／ N=95 ／ ストップワード35語 ／ seed=42。K=6=頑健／K=12=細かい探索。</p>
<div class="kbar"><b>トピック数：</b><span id="kbtns"></span></div>

<h2>1. トピック一覧</h2>
<p class="dim">テーマ名は各トピックの最重み上位2語による機械ラベル（LLM要約ではない）。</p>
{cards}

<h2>2. 配合スタックバー（各ケースのテーマ混合比）</h2>
<div class="rule"><b>構成ルール：</b>棒1本＝1ケース。NMFの重み行列Wを合計1に正規化した「テーマ配合」を色で積み上げ。
優勢テーマ順に整列。ホバーで内訳。</div>
<div class="fig"><div id="lg-stack" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px"></div>
<canvas id="stack" width="1000" height="300"></canvas></div>

<h2>3. 語彙のソフト分類</h2>
<div class="rule"><b>構成ルール：</b>NMFのもう一方の行列H（テーマ×語の重み）を使用。各語は全テーマに重みを持つ（ソフト）。
下段の「橋渡し語」は、H列を正規化して<b>2テーマ以上に25%超</b>で分かれる語＝複数テーマにまたがる語。</div>
<h3>各テーマの高重み語</h3>
{perTopic}
<h3>橋渡し語（複数テーマにまたがる語）</h3>
{bridgewords}

<h2>4. ネットワーク図（複数種類・各構成ルール付き）</h2>

<h3>4a. 共通語ネットワーク</h3>
<div class="rule"><b>構成ルール：</b>ノード＝ケース（色＝主テーマ）。<b>線＝2ケースが特徴語(TF-IDF上位20語)を2語以上共有</b>。
配置は「同じ主テーマが集まる」グループ重力＋薄い背景＝テーマ範囲。点クリックで共通語つきの隣接一覧。</div>
<div class="fig"><div id="lg-net" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px"></div>
<canvas id="swnet" width="1000" height="580"></canvas><div id="sw-panel" class="panel">点をクリックすると共通語つきで隣接ケースを表示。</div></div>

<h3>4b. 混合類似ネットワーク</h3>
<div class="rule"><b>構成ルール：</b>ノード＝ケース（色＝テーマ配合を混ぜた色）。<b>線＝テーマ配合ベクトルのコサイン類似で最も近い3件</b>
（意味的な近さ。共通語とは別基準）。配置は力学レイアウト。中間色＝複数テーマの橋渡し。</div>
<div class="fig"><canvas id="msnet" width="1000" height="560"></canvas></div>

<h3>4c. 越境ビュー（テーマを越える共通語つながり）</h3>
<div class="rule"><b>構成ルール：</b>4aの共通語リンクのうち<b>主テーマが異なるペアだけ</b>を抽出。
コード図＝テーマ間の越境件数（線の太さ）。下の一覧＝各越境ペアの2ケースと共通語。</div>
<div class="fig">{chord}</div>
{bridgelist}

<h3>4d. ケース×トピック 2部グラフ</h3>
<div class="rule"><b>構成ルール：</b>大きな丸＝テーマ、小さな丸＝ケース。<b>線＝そのケースのテーマ配合が15%超</b>（太さ＝配合）。
1本だけ＝単一テーマ、複数本＝横断ケース。配置は力学レイアウト。</div>
<div class="fig"><canvas id="bipnet" width="1000" height="600"></canvas></div>

<h3>4e. パイマーカー地図（UMAP）</h3>
<div class="rule"><b>構成ルール：</b>配置＝文章の意味埋め込みのUMAP2D座標（可視化専用）。各点を<b>テーマ配合の円グラフ</b>で描画。
近接なのに配色が違う＝文章は似るが論点構成が違うケース。</div>
<div class="fig"><canvas id="piemap" width="1000" height="560"></canvas></div>

<p class="dim">本分類は解析目的であり当事者の類型評価を意図しません。著作権はCALL4および執筆者に帰属。</p>
</div><div id="tip"></div>

<script>
const BYK={byk}; const KS={ks};
let curK=KS[0];
const tip=document.getElementById('tip');
function showTip(h,ev){{tip.innerHTML=h;tip.style.display='block';
  tip.style.left=Math.min(ev.clientX+14,window.innerWidth-360)+'px';tip.style.top=(ev.clientY+12)+'px';}}
function hideTip(){{tip.style.display='none';}}
function D(){{return BYK[curK];}}
function mixText(n){{const d=D();return n.mix.map((v,t)=>v>0.08?`<span style="color:${{d.colors[t]}}">■</span>${{d.names[t]}} ${{(v*100).toFixed(0)}}%`:null).filter(Boolean).join('<br>');}}
function wtext(n){{return n.w&&n.w.length?`<br><span style="color:#e0b48a">特徴語: ${{n.w.join(' / ')}}</span>`:'';}}
function legend(elid){{const d=D();const el=document.getElementById(elid);if(!el)return;el.innerHTML='';
  d.names.forEach((nm,t)=>el.insertAdjacentHTML('beforeend',`<span style="font-size:11px"><span style="display:inline-block;width:10px;height:10px;background:${{d.colors[t]}};margin-right:3px;border-radius:5px"></span>T${{t}} ${{nm}}</span>`));}}

// scalers
function sc(canvas,xs,ys,pad){{const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  return (x,y)=>[pad+(x-x0)/((x1-x0)||1)*(canvas.width-2*pad),pad+(y-y0)/((y1-y0)||1)*(canvas.height-2*pad)];}}
function nearest(canvas,pos,ev){{const r=canvas.getBoundingClientRect();const mx=(ev.clientX-r.left)*(canvas.width/r.width),my=(ev.clientY-r.top)*(canvas.height/r.height);
  let best=null,bd=200;D().nodes.forEach(n=>{{const [x,y]=pos(n);const dd=(x-mx)**2+(y-my)**2;if(dd<bd){{bd=dd;best=n;}}}});return best;}}

// 2. stacked bar
function drawStack(){{const d=D();const cv=document.getElementById('stack'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const order=[...d.nodes.keys()].sort((a,b)=>(d.nodes[a].f-d.nodes[b].f)|| (d.nodes[b].mix[d.nodes[b].f]-d.nodes[a].mix[d.nodes[a].f]));
  const BW=(cv.width-16)/order.length;
  order.forEach((ci,pos)=>{{const n=d.nodes[ci];let y=cv.height-14;n.mix.forEach((r,t)=>{{const h=r*(cv.height-24);x.fillStyle=d.colors[t];x.fillRect(10+pos*BW,y-h,Math.max(BW-.5,1),h);y-=h;}});}});
  cv.onmousemove=ev=>{{const r=cv.getBoundingClientRect();const p=Math.floor(((ev.clientX-r.left)*(cv.width/r.width)-10)/BW);if(p<0||p>=order.length){{hideTip();return;}}const n=d.nodes[order[p]];showTip(`<b>${{n.t}}</b><br>${{mixText(n)}}`,ev);cv.style.cursor='pointer';}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const r=cv.getBoundingClientRect();const p=Math.floor(((ev.clientX-r.left)*(cv.width/r.width)-10)/BW);if(p>=0&&p<order.length)window.open(d.nodes[order[p]].u,'_blank');}};
  legend('lg-stack');}}

// 4a shared-word net
let swsel=null;
function swNeighbors(i){{const o=[];D().sw_edges.forEach(([a,b,sh])=>{{if(a===i)o.push([b,sh]);else if(b===i)o.push([a,sh]);}});return o;}}
function drawSW(){{const d=D();const cv=document.getElementById('swnet'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const pos=n=>[26+n.swx*(cv.width-52),n.swy*(cv.height-26)];
  for(const t in d.hulls){{const poly=d.hulls[t];if(!poly)continue;x.beginPath();poly.forEach((p,k)=>{{const [X,Y]=pos({{swx:p[0],swy:p[1]}});k?x.lineTo(X,Y):x.moveTo(X,Y);}});x.closePath();const c=d.colors[t].replace('#','');x.fillStyle=`rgba(${{parseInt(c.slice(0,2),16)}},${{parseInt(c.slice(2,4),16)}},${{parseInt(c.slice(4,6),16)}},.09)`;x.fill();}}
  const nb=swsel!=null?new Set(swNeighbors(swsel).map(e=>e[0])):null;
  d.sw_edges.forEach(([a,b])=>{{if(swsel!=null&&a!==swsel&&b!==swsel)return;const [x1,y1]=pos(d.nodes[a]),[x2,y2]=pos(d.nodes[b]);x.strokeStyle=swsel!=null?'rgba(70,80,100,.6)':'rgba(140,145,155,.28)';x.lineWidth=swsel!=null?1.5:.8;x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}});
  d.nodes.forEach(n=>{{const [X,Y]=pos(n);const dim=swsel!=null&&n.i!==swsel&&!(nb&&nb.has(n.i));x.globalAlpha=dim?.15:1;x.beginPath();x.arc(X,Y,n.i===swsel?8:5.5,0,7);x.fillStyle=d.colors[n.f];x.fill();x.globalAlpha=1;x.strokeStyle=n.i===swsel?'#222':'#fff';x.lineWidth=n.i===swsel?2:1;x.stroke();}});
  legend('lg-net');
  cv.onmousemove=ev=>{{const n=nearest(cv,pos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br><span style="color:${{d.colors[n.f]}}">主: ${{d.names[n.f]}}</span>${{wtext(n)}}`,ev);}};
  cv.onmouseleave=hideTip;
  cv.onclick=ev=>{{const n=nearest(cv,pos,ev);swsel=n?n.i:null;drawSW();const P=document.getElementById('sw-panel');
    if(swsel==null){{P.innerHTML='点をクリックすると共通語つきで隣接ケースを表示。';return;}}
    const nb=swNeighbors(swsel).sort((a,b)=>b[1].length-a[1].length);
    P.innerHTML=nb.length?`<b>「${{n.t}}」と特徴語を共有</b><ul class="cases">`+nb.map(([j,sh])=>`<li><a href="${{d.nodes[j].u}}" target="_blank">${{d.nodes[j].t}}</a><br><span class="cw">共通語: ${{sh.join(' / ')}}</span></li>`).join('')+'</ul>':'共有相手なし';}};}}

// 4b mixture-similarity net
function drawMS(){{const d=D();const cv=document.getElementById('msnet'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const pos=n=>[26+n.msx*(cv.width-52),26+n.msy*(cv.height-52)];
  x.strokeStyle='rgba(150,155,165,.35)';x.lineWidth=.9;
  d.ms_edges.forEach(([a,b])=>{{const [x1,y1]=pos(d.nodes[a]),[x2,y2]=pos(d.nodes[b]);x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}});
  d.nodes.forEach(n=>{{const [X,Y]=pos(n);x.beginPath();x.arc(X,Y,6,0,7);x.fillStyle=n.bl;x.fill();x.strokeStyle='#fff';x.lineWidth=1;x.stroke();}});
  cv.onmousemove=ev=>{{const n=nearest(cv,pos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br>${{mixText(n)}}${{wtext(n)}}`,ev);}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const n=nearest(cv,pos,ev);if(n)window.open(n.u,'_blank');}};}}

// 4d bipartite
function drawBip(){{const d=D();const cv=document.getElementById('bipnet'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const xs=d.nodes.map(n=>n.bx).concat(d.topic_pos.map(p=>p[0])),ys=d.nodes.map(n=>n.by).concat(d.topic_pos.map(p=>p[1]));
  const S=sc(cv,xs,ys,34);const cpos=n=>S(n.bx,n.by),tpos=t=>S(d.topic_pos[t][0],d.topic_pos[t][1]);
  d.bip_edges.forEach(([i,t])=>{{const [x1,y1]=cpos(d.nodes[i]),[x2,y2]=tpos(t);const c=d.colors[t].replace('#','');x.strokeStyle=`rgba(${{parseInt(c.slice(0,2),16)}},${{parseInt(c.slice(2,4),16)}},${{parseInt(c.slice(4,6),16)}},.35)`;x.lineWidth=Math.max(d.nodes[i].mix[t]*5,.5);x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();}});
  d.nodes.forEach(n=>{{const [X,Y]=cpos(n);x.beginPath();x.arc(X,Y,4.5,0,7);x.fillStyle=d.colors[n.f];x.fill();x.strokeStyle='#fff';x.lineWidth=.8;x.stroke();}});
  d.topic_pos.forEach((p,t)=>{{const [X,Y]=tpos(t);x.beginPath();x.arc(X,Y,16,0,7);x.fillStyle=d.colors[t];x.fill();x.strokeStyle='#fff';x.lineWidth=2;x.stroke();x.fillStyle='#fff';x.font='bold 11px sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText('T'+t,X,Y);x.textAlign='left';}});
  cv.onmousemove=ev=>{{const n=nearest(cv,cpos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br>${{mixText(n)}}`,ev);}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const n=nearest(cv,cpos,ev);if(n)window.open(n.u,'_blank');}};}}

// 4e pie map
function drawPie(){{const d=D();const cv=document.getElementById('piemap'),x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);
  const S=sc(cv,d.nodes.map(n=>n.px),d.nodes.map(n=>n.py),28);const pos=n=>S(n.px,n.py);
  d.nodes.forEach(n=>{{const [X,Y]=pos(n);let a0=-Math.PI/2;n.mix.forEach((r,t)=>{{if(r<.03)return;const a1=a0+r*2*Math.PI;x.beginPath();x.moveTo(X,Y);x.arc(X,Y,8,a0,a1);x.closePath();x.fillStyle=d.colors[t];x.fill();a0=a1;}});x.beginPath();x.arc(X,Y,8,0,7);x.strokeStyle='#fff';x.lineWidth=1;x.stroke();}});
  cv.onmousemove=ev=>{{const n=nearest(cv,pos,ev);if(!n){{hideTip();cv.style.cursor='default';return;}}cv.style.cursor='pointer';showTip(`<b>${{n.t}}</b><br>${{mixText(n)}}`,ev);}};
  cv.onmouseleave=hideTip;cv.onclick=ev=>{{const n=nearest(cv,pos,ev);if(n)window.open(n.u,'_blank');}};}}

function redraw(){{swsel=null;drawStack();drawSW();drawMS();drawBip();drawPie();}}
function setK(K){{curK=String(K);
  document.querySelectorAll('.konly').forEach(e=>e.classList.add('hidden'));
  document.querySelectorAll('.k'+K).forEach(e=>e.classList.remove('hidden'));
  document.querySelectorAll('.kb').forEach(b=>b.classList.toggle('active',b.dataset.k==K));
  redraw();}}
const kb=document.getElementById('kbtns');
KS.forEach(K=>kb.insertAdjacentHTML('beforeend',`<button class="kb" data-k="${{K}}">K=${{K}}</button>`));
kb.addEventListener('click',e=>{{if(e.target.dataset.k)setK(e.target.dataset.k);}});
setK(KS[0]);
</script></body></html>"""


if __name__ == "__main__":
    main()
