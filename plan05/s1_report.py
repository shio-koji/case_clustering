#!/usr/bin/env python3
"""plan05 Stage 1 — Self-contained bilingual HTML report for soft tagging.

Figures (inline SVG): per-tag validation AUC bar, per-case tag-ratio stacked bar
(multi-tag cases). Interactive: searchable per-case ratio table (CALL4 links),
missing-tag and over-tag candidate tables.
"""
import io
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import igraph as ig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "plan05", "results")
REP = os.path.join(ROOT, "plan05", "report")
os.makedirs(REP, exist_ok=True)

JP_FONT_PATH = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
                "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/"
                "YuGothic-Medium.otf")
if os.path.exists(JP_FONT_PATH):
    fm.fontManager.addfont(JP_FONT_PATH)
    plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def jp(size=9):
    return fm.FontProperties(fname=JP_FONT_PATH, size=size) if os.path.exists(JP_FONT_PATH) \
        else fm.FontProperties(size=size)


def svg_of(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue().decode("utf-8")
    return s[s.find("<svg"):]


d = json.load(open(os.path.join(RES, "soft_tags.json")))
tags = d["tags"]
T = len(tags)
TAGPAL = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00A0B0",
          "#f032e6", "#9A6324", "#469990", "#808000", "#e6ac00"]
TCOL = {tags[j]: TAGPAL[j] for j in range(T)}
ratios = d["ratios"]
case_ids = d["case_ids"]
titles = d["titles"]
tit = {c: t for c, t in zip(case_ids, titles)}
corpus = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "cache/corpus_clean.json")))}
status_of = {c: corpus.get(c, {}).get("case_status", "") for c in case_ids}
n_active = sum(1 for c in case_ids if status_of[c] == "active")
auc = d["validation"]["per_tag_auc"]
recov = d["validation"]["self_recovery"]


def url(cid):
    return f"https://www.call4.jp/info.php?type=items&id={cid}"


# ---- Fig 1. validation AUC bar -------------------------------------------
def fig_auc():
    items = [(t, auc[t]) for t in tags if auc[t] is not None]
    items.sort(key=lambda x: x[1])
    names = [t for t, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=[TCOL[n] for n in names])
    ax.set_yticks(y); ax.set_yticklabels(names, fontproperties=jp(9))
    ax.set_xlim(0.5, 1.0); ax.axvline(0.5, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("ROC-AUC (LOO判別スコアがタグ有無を分離できるか)", fontproperties=jp(9))
    ax.set_title("検証：埋め込みは各タグをどれだけ言い当てるか\n"
                 "Validation: how well embeddings recover each tag (LOO ROC-AUC)",
                 fontproperties=jp(11))
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontproperties=jp(8))
    fig.tight_layout()
    return svg_of(fig)


# ---- Fig 2. per-case tag-ratio stacked bar (multi-tag cases) --------------
def fig_ratio_stack():
    multi = [(c, ratios[c]) for c in case_ids if len(ratios[c]) >= 2]

    def dom(item):
        r = item[1]
        top = max(r, key=r.get)
        return (tags.index(top), -r[top])
    multi.sort(key=dom)
    n = len(multi)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(n)
    bottom = np.zeros(n)
    for j, tag in enumerate(tags):
        vals = np.array([multi[i][1].get(tag, 0.0) for i in range(n)])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, width=0.92, color=TCOL[tag], label=tag)
        bottom += vals
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_ylabel("タグ割合 / tag share", fontproperties=jp(9))
    ax.set_xlabel(f"複数タグを持つ{n}ケース（優勢タグ順） / {n} multi-tag cases",
                  fontproperties=jp(9))
    ax.set_title("ケース内タグ割合（ソフトタグ）Per-case tag share within each case",
                 fontproperties=jp(11))
    ax.legend(ncol=6, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return svg_of(fig)


# ---- interactive per-case ratio table (JSON for JS) -----------------------
def ratio_table_json():
    rows = []
    for c in case_ids:
        r = ratios[c]
        parts = sorted(r.items(), key=lambda x: -x[1])
        rows.append({"id": c, "title": tit[c], "st": status_of[c],
                     "n": len(r),
                     "tags": [{"t": k, "v": v, "c": TCOL[k]} for k, v in parts]})
    return json.dumps(rows, ensure_ascii=False)


# ---- bipartite case<->tag graph (interactive, hover shows ratios) ---------
def _layouts_for(subset):
    """Compute the 6 layouts over tags + the given case subset (list of case_ids).
    Returns {layout: {"tags": [[x,y]*T], "cases": {case_id: [x,y]}}}."""
    import math
    m = len(subset)
    NN = T + m
    edges_idx, wts = [], []
    for k, cid in enumerate(subset):
        for tag, v in ratios[cid].items():
            edges_idx.append((tags.index(tag), T + k)); wts.append(v)
    g = ig.Graph(n=NN, edges=edges_idx)
    import random as _r
    _r.seed(42)
    try:
        ig.set_random_number_generator(_r)
    except Exception:
        pass
    kk_len = [max(0.12, 1.0 - 0.85 * w) for w in wts]     # high ratio -> short -> near

    P = {}
    P["force"] = np.array(g.layout_fruchterman_reingold(niter=1500).coords)
    P["force_r"] = np.array(g.layout_fruchterman_reingold(niter=1500, weights=wts).coords)
    try:
        P["kamada"] = np.array(g.layout_kamada_kawai().coords)
    except Exception:
        P["kamada"] = P["force"]
    try:
        P["kamada_r"] = np.array(g.layout_kamada_kawai(weights=kk_len).coords)
    except Exception:
        P["kamada_r"] = P["force_r"]
    # tag order by count WITHIN this subset (so active-only re-orders the circle too)
    tcount = [sum(1 for cid in subset if tags[j] in ratios[cid]) for j in range(T)]
    torder = list(np.argsort(-np.array(tcount)))
    tagpos = np.zeros((T, 2))
    for k, j in enumerate(torder):
        a = 2 * math.pi * k / max(T, 1)
        tagpos[j] = [math.cos(a), math.sin(a)]
    rad = np.zeros((NN, 2)); rad[:T] = tagpos * 1.15
    for k, cid in enumerate(subset):
        bc = np.zeros(2)
        for tag, v in ratios[cid].items():
            bc += v * tagpos[tags.index(tag)]
        ang = (T + k) * 2.399963229
        rad[T + k] = bc * 0.82 + 0.07 * np.array([math.cos(ang), math.sin(ang)])
    P["radial"] = rad
    col = np.zeros((NN, 2))
    for k, j in enumerate(torder):
        col[j] = [-1.0, 1.0 - 2.0 * k / max(T - 1, 1)]
    corder = sorted(range(m), key=lambda k: (
        torder.index(tags.index(max(ratios[subset[k]], key=ratios[subset[k]].get))),
        -max(ratios[subset[k]].values())))
    for rank, k in enumerate(corder):
        col[T + k] = [1.0, 1.0 - 2.0 * rank / max(m - 1, 1)]
    P["columns"] = col

    out = {}
    for name, A in P.items():
        out[name] = {
            "tags": [[round(float(A[j, 0]), 4), round(float(A[j, 1]), 4)] for j in range(T)],
            "cases": {cid: [round(float(A[T + k, 0]), 4), round(float(A[T + k, 1]), 4)]
                      for k, cid in enumerate(subset)}}
    return out


def build_bipartite_json():
    """Two layout sets — all cases and active-only — so the graph can re-layout
    (not just hide) when filtered. Edges weighted by soft ratio."""
    full = _layouts_for(case_ids)
    active_ids = [c for c in case_ids if status_of[c] == "active"]
    active = _layouts_for(active_ids)
    LAYS = list(full.keys())
    miss_by_case = {}
    for mm in d["missing_candidates"]:
        miss_by_case.setdefault(mm["case_id"], []).append(mm)

    tag_nodes = []
    for j in range(T):
        tag_nodes.append({
            "name": tags[j], "count": int(sum(1 for c in case_ids if tags[j] in ratios[c])),
            "color": TCOL[tags[j]],
            "pos": {ly: full[ly]["tags"][j] for ly in LAYS},
            "posA": {ly: active[ly]["tags"][j] for ly in LAYS}})
    case_nodes = []
    for cid in case_ids:
        parts = sorted(ratios[cid].items(), key=lambda x: -x[1])
        miss = [{"t": mm["tag"], "c": TCOL[mm["tag"]], "pct": mm["pct_vs_positives"],
                 "lc": mm["low_conf"]}
                for mm in sorted(miss_by_case.get(cid, []), key=lambda x: -x["disc"])[:3]]
        node = {
            "id": cid, "title": tit[cid], "st": status_of[cid],
            "ndeg": len(parts), "domc": TCOL[parts[0][0]],
            "tags": [{"t": k, "v": round(v, 3), "c": TCOL[k]} for k, v in parts],
            "miss": miss,
            "pos": {ly: full[ly]["cases"][cid] for ly in LAYS}}
        if status_of[cid] == "active":
            node["posA"] = {ly: active[ly]["cases"][cid] for ly in LAYS}
        case_nodes.append(node)
    edge_list = []
    for ci, cid in enumerate(case_ids):
        for tag in ratios[cid]:
            edge_list.append([tags.index(tag), ci, round(ratios[cid][tag], 3)])
    return (json.dumps(tag_nodes, ensure_ascii=False),
            json.dumps(case_nodes, ensure_ascii=False), json.dumps(edge_list))


# ---- missing / over candidate tables --------------------------------------
def missing_rows(limit=25):
    rows = []
    for m in d["missing_candidates"][:limit]:
        conf = '<span class="lc">低信頼 low-conf</span>' if m["low_conf"] else ""
        bar = int(m["pct_vs_positives"] * 100)
        rows.append(
            f'<tr><td><a href="{url(m["case_id"])}" target="_blank">{m["title"]}</a></td>'
            f'<td><span class="tg" style="background:{TCOL[m["tag"]]}">{m["tag"]}</span>{conf}</td>'
            f'<td>{m["disc"]:.3f}</td>'
            f'<td><div class="pb"><div class="pf" style="width:{bar}%"></div></div>'
            f'{bar}%</td></tr>')
    return "\n".join(rows)


def over_rows():
    rows = []
    for o in d["over_candidates"]:
        rows.append(
            f'<tr><td><a href="{url(o["case_id"])}" target="_blank">{o["title"]}</a></td>'
            f'<td><span class="tg" style="background:{TCOL[o["tag"]]}">{o["tag"]}</span></td>'
            f'<td>{o["disc"]:.3f}</td>'
            f'<td>{o["assigned_ratio"]}</td></tr>')
    return "\n".join(rows)


n_multi = sum(1 for c in case_ids if len(ratios[c]) >= 2)
mean_auc = d["validation"]["mean_auc_reliable"]
n_missing = len(d["missing_candidates"])
bip_tags_json, bip_cases_json, bip_edges_json = build_bipartite_json()

HTML = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CALL4 ソフトタグ・レポート (plan05)</title>
<style>
:root{{--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--accent:#2B6CB0;}}
*{{box-sizing:border-box;}}
body{{font-family:-apple-system,"Hiragino Sans","Yu Gothic",sans-serif;color:var(--fg);
max-width:1080px;margin:0 auto;padding:28px 20px 80px;line-height:1.7;}}
h1{{font-size:1.65rem;border-bottom:3px solid var(--accent);padding-bottom:10px;}}
h2{{font-size:1.26rem;margin-top:2.3em;border-left:5px solid var(--accent);padding-left:10px;}}
h4{{margin:.3em 0;}}
.en{{color:var(--muted);font-weight:400;font-size:.72em;}}
.lead{{background:#f5f8fc;border:1px solid var(--line);border-radius:8px;padding:16px 20px;}}
.warn{{background:#fff8e6;border:1px solid #e6c34c;border-radius:8px;padding:12px 18px;font-size:.92em;}}
.ok{{background:#eef6ef;border:1px solid #9cc1a4;border-left:6px solid #4C8C2B;border-radius:8px;padding:12px 18px;}}
figure{{margin:1.2em 0;text-align:center;}} svg{{max-width:100%;height:auto;}}
figcaption{{color:var(--muted);font-size:.84em;margin-top:6px;}}
table{{border-collapse:collapse;width:100%;font-size:.9em;margin:1em 0;}}
th,td{{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top;}}
th{{background:#f2f2f2;}}
.kpi{{display:flex;gap:20px;flex-wrap:wrap;margin:1em 0;}}
.kpi div{{background:#f5f8fc;border:1px solid var(--line);border-radius:8px;padding:12px 18px;min-width:150px;}}
.big{{font-size:2rem;font-weight:800;color:var(--accent);}}
.tg{{display:inline-block;color:#fff;border-radius:4px;padding:1px 7px;font-size:.85em;font-weight:600;}}
.lc{{color:#b0532b;font-size:.78em;margin-left:6px;}}
.pb{{display:inline-block;width:90px;height:9px;background:#eee;border-radius:5px;overflow:hidden;vertical-align:middle;margin-right:5px;}}
.pf{{height:100%;background:#4C8C2B;}}
code{{background:#f2f2f2;padding:1px 5px;border-radius:3px;}}
#search{{padding:7px 11px;width:280px;border:1px solid var(--line);border-radius:6px;font-size:14px;}}
#rt{{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:8px;}}
#rt table{{margin:0;}} #rt th{{position:sticky;top:0;}}
.rbar{{display:flex;height:16px;border-radius:3px;overflow:hidden;min-width:180px;}}
.rseg{{color:#fff;font-size:10px;text-align:center;line-height:16px;white-space:nowrap;overflow:hidden;}}
.rlbl{{font-size:.82em;color:#444;}}
.detail{{font-size:.96em;}}
.detail h3{{font-size:1.05rem;margin-top:1.5em;color:#243b53;}}
.gloss{{background:#f7f9fb;border-left:3px solid #b9c6d6;padding:2px 10px;margin:.4em 0;
color:#3a4a5a;font-size:.92em;border-radius:0 4px 4px 0;}}
.gloss b{{color:#243b53;}}
#bip-wrap{{border:1px solid var(--line);border-radius:8px;padding:8px;background:#fbfcfe;}}
#bip{{width:100%;height:auto;display:block;cursor:default;}}
#bip-legend{{display:flex;flex-wrap:wrap;gap:6px 12px;margin:8px 4px 2px;}}
#bip-legend span{{font-size:11.5px;}}
#bip-btns{{display:flex;flex-wrap:wrap;gap:6px;margin:2px 4px 8px;}}
.lbtn{{font-size:12px;padding:4px 12px;border:1px solid #c3ccd6;background:#fff;
border-radius:15px;cursor:pointer;color:#334;}}
.lbtn.on{{background:var(--accent);color:#fff;border-color:var(--accent);}}
.af{{font-size:13px;color:#334;margin-left:10px;cursor:pointer;user-select:none;}}
.af input{{vertical-align:middle;margin-right:4px;}}
#tip{{position:fixed;display:none;background:#20293a;color:#fff;padding:9px 12px;
border-radius:6px;font-size:12px;max-width:300px;box-shadow:0 3px 12px rgba(0,0,0,.28);
z-index:50;line-height:1.5;pointer-events:none;}}
#tip a{{color:#9ecbff;}}
#tip .trow{{display:flex;align-items:center;gap:6px;margin:2px 0;}}
#tip .tsw{{display:inline-block;width:10px;height:10px;border-radius:2px;flex:none;}}
#tip .tbar{{flex:1;height:8px;background:#46536b;border-radius:4px;overflow:hidden;min-width:70px;}}
#tip .tbf{{display:block;height:100%;border-radius:4px;min-width:2px;}}
.hint{{color:var(--muted);font-size:.84em;margin:.3em 0 .6em;}}
</style></head><body>

<h1>CALL4：ソフトタグ —— ケース内タグ割合と貼り忘れ検出<br>
<span class="en">Soft tags: per-case tag weights & missing-tag detection — plan05</span></h1>

<div class="lead">
<b>これは何か / What.</b> 既存タグは0/1（付いている・いない）だけで<b>重みが無い</b>。
本レポートは plan02 の文章埋め込み（bge-m3）を使い、
<b>(1) 1ケースの中で各タグがどれだけ中心的か（割合）</b>と、
<b>(2) テキスト上は該当するのに付いていないタグ（貼り忘れ候補）</b>を推定する。系統B・LLM不使用。
<br><span class="en">Binary tags carry no weight. Using text embeddings we estimate, per case, how central each
assigned tag is, and which unassigned tags the text nonetheless matches (missing-tag candidates).</span>
</div>

<div class="kpi">
<div><div class="big">{mean_auc}</div>平均AUC（埋め込みのタグ再現力）<br><span class="en">mean tag-recovery AUC</span></div>
<div><div class="big">{recov['top1']:.0%}</div>最上位スコアが実タグと一致<br><span class="en">top-1 self-recovery</span></div>
<div><div class="big">{n_multi}</div>複数タグを持つケース<br><span class="en">multi-tag cases</span></div>
<div><div class="big">{n_missing}</div>貼り忘れ候補<br><span class="en">missing-tag candidates</span></div>
</div>

<h2>1. 方法 <span class="en">Method</span></h2>
<p>各タグ t について、<b>そのタグを持つケース群の重心(pos)</b>と<b>持たない群の重心(neg)</b>を作り、
ケース i の判別スコアを <code>disc(i,t)=cos(emb_i, pos)−cos(emb_i, neg)</code> と定義する。
bge-m3 はコサインの下限が高い（全文書が法律文書として似る共通成分）ため、素のコサインでは差が出ない。
pos と neg の差を取ることで<b>共通成分を打ち消し、タグ固有の向き</b>だけを残す（最近傍重心分類の決定値に相当）。
自己参照の水増しを避けるため、i が t を持つ場合は pos を <b>i を除いて（Leave-One-Out）</b>計算する。</p>
<ul>
<li><b>割合</b>：ケースが持つタグについて disc を softmax 正規化（合計1、温度τ={d['tau']}・データから自動）。</li>
<li><b>貼り忘れ</b>：付いていないタグで disc が高いもの。その値が「実際にtが付くケースの何％より上か」で基準化。</li>
<li><b>信頼度</b>：陽性1件のタグ（個人情報）はプロトタイプが作れず<b>除外</b>、3件未満（沖縄）は<b>低信頼</b>表示。</li>
</ul>

<details open><summary style="cursor:pointer;font-weight:600;color:#243b53">▼ 手法の詳しい解説（クリックで開閉）／ Detailed method walk-through</summary>
<div class="detail">

<h3>(a) 埋め込みとコサイン類似度 / Embeddings & cosine</h3>
<p><b>埋め込み（embedding）</b>とは、文章を「意味を要約した数値ベクトル」に変換したもの。ここでは各ケースの
テキスト（ケース名＋概要＋本文）を <b>bge-m3</b> という多言語モデルで <b>1024次元のベクトル</b>にしている
（plan02 で計算済みのものを再利用）。2つのベクトルの近さは <b>コサイン類似度（cosine similarity）</b>で測る。</p>
<div class="gloss"><b>コサイン類似度</b>：2つのベクトルのなす角の余弦。1に近い＝ほぼ同じ向き（似た意味）、
0＝無関係、負＝逆向き。ベクトルの長さを1に揃えて（<b>L2正規化</b>）あるので、内積＝コサインになる。</div>

<h3>(b) なぜ素のコサインではダメか / The high-floor problem</h3>
<p>本データでは、無関係な2ケースでもコサインが中央値 0.57 と高い。「どのケースも“公共訴訟の文章”として
似ている」<b>共通成分（common-mode）</b>がベクトルに乗っているためだ。だから
「ケースとタグ重心のコサイン」をそのまま使うと、どのタグでも 0.5〜0.7 に張り付いて<b>差が出ない</b>。</p>
<div class="gloss">バイオ対応：全細胞に共通して光るハウスキーピング遺伝子の発現が、
細胞型の違いを覆い隠してしまう状況に似ている。まず<b>共通の底上げを引き算</b>したい。</div>

<h3>(c) プロトタイプと判別スコア / Prototype & discriminative score</h3>
<p><b>プロトタイプ（prototype＝原型・重心）</b>とは、あるタグを持つケース群のベクトルの平均。
「そのタグらしさの中心」を表す。素朴には <code>cos(ケース, タグの原型)</code> でタグらしさを測れるが、(b)の底上げが乗る。
そこで<b>そのタグを持つ群の原型(pos)</b>と<b>持たない群の原型(neg)</b>の両方を作り、その差を取る：</p>
<p style="text-align:center"><code>disc(ケース, タグ) = cos(ケース, pos) − cos(ケース, neg)</code></p>
<p>共通成分は pos・neg どちらにも等しく乗るので、引き算で<b>相殺</b>される。残るのは「そのタグ特有の向きに、
このケースがどれだけ寄っているか」。値が正なら「そのタグらしい」、負なら「らしくない」。</p>
<div class="gloss"><b>これは何をしているか</b>：2グループ（保有/非保有）の重心の中点を境界にした
<b>最近傍重心分類（nearest-centroid classifier）</b>の決定値そのもの。統計でいう線形判別分析(LDA)の素朴版に相当し、
「タグを分ける方向」への射影を測っている。</div>

<h3>(d) Leave-One-Out（自己参照の除去）/ LOO</h3>
<p>あるケースが「自分の持つタグ」の原型に自分自身も含まれていると、<b>自分で自分を似ていると判定</b>してしまい
点数が水増しされる（<b>情報リーク／leakage</b>）。これを避けるため、ケース i を採点するときは、
i が属するタグの原型を <b>i を除いて（Leave-One-Out＝1つ抜き）</b>計算する。
検証(第2章)で使う AUC も、このLOO点数で測るので<b>公正</b>。</p>
<div class="gloss">バイオ対応：参照アトラスに“これから分類したい細胞そのもの”を混ぜてラベルを当てると
出来過ぎになる——それを防ぐ交差検証と同じ発想。</div>

<h3>(e) 割合の作り方：softmaxと温度 / From scores to shares</h3>
<p>ケースが持つ複数タグの disc を、合計が1になる<b>割合</b>に変換するのに <b>softmax（ソフトマックス）</b>を使う。</p>
<div class="gloss"><b>softmax</b>：数値の組を「大きいものほど大きい割合」になるよう0〜1（合計1）に変換する関数。
指数関数 exp を使うので、差が強調される。<b>温度（temperature τ）</b>は強調の度合いを決める調整つまみで、
小さいほど一番大きいタグに偏り、大きいほど均等に近づく。本レポートは τ を<b>データのばらつき（標準偏差）から自動設定</b>
({d['tau']})し、恣意性を避けている。</div>
<p>結果、例えば「同性パートナー遺族給付金」訴訟は <b>ジェンダー0.89／刑事0.11</b> のように、
主題と傍論が数値で分かれる。</p>

<h3>(f) 貼り忘れの見つけ方：パーセンタイル基準 / Missing-tag via percentile</h3>
<p>あるタグを<b>持たない</b>ケースでも disc が高ければ「本来は該当するのでは？」と疑える。ただし
「高い」の基準はタグごとに違うので、<b>実際にそのタグが付いているケースの disc 分布</b>と比べる。
「陽性ケースの下位25%より上」なら候補とし、<b>パーセンタイル</b>（＝陽性ケースの何％より該当が強いか）で順位づける。</p>
<div class="gloss"><b>パーセンタイル</b>：ある値が分布の中で下から何％の位置にあるか。
「陽性群での位置 80%」＝「実際にそのタグが付くケースの8割より、このケースの方が該当が強い」の意味。</div>

<h3>(g) 検証：ROC-AUC / Does it actually work?</h3>
<p>提案を出す前に「そもそも埋め込みがタグを言い当てられるか」を <b>ROC-AUC</b> で確認した（第2章）。</p>
<div class="gloss"><b>ROC-AUC</b>：スコア（ここではLOOの disc）で「タグ有り／無し」をどれだけ分離できるかの指標。
<b>1.0＝完全に分離、0.5＝でたらめと同じ</b>。無作為に選んだ「タグ有りケース」の点数が「タグ無しケース」より
高くなる確率、と読める。本手法は平均 {mean_auc}（0.5から大きく上）で、採点が機能していることを示す。</div>

</div></details>

<h2>2. 検証：この採点を信じてよいか <span class="en">Validation</span></h2>
<div class="ok"><b>結論：信頼できる。</b> 埋め込みは11タグ中10タグを AUC≈0.9〜0.99 で言い当て（平均{mean_auc}）、
各ケースの最高スコアのタグは <b>{recov['top1']:.0%}</b> で実際の付与タグと一致（top-3で{recov['top3']:.0%}）。
唯一低いのは横断的な<b>「公正な手続」(AUC {auc['公正な手続']})</b>で、これは「テキストから復元しにくい手続的タグ」という
plan03 の知見と整合的。→ この横断タグに関する提案は控えめに読む。</div>
<figure>{fig_auc()}<figcaption>タグ別 ROC-AUC（LOO）。高い＝そのタグは埋め込みでよく捉えられ、割合・貼り忘れ提案が信頼できる。</figcaption></figure>

<h2>3. ケース内タグ割合 <span class="en">Per-case tag share</span></h2>
<p>複数タグを持つ{n_multi}ケースについて、各タグの相対的な重み（合計1）。
1タグのみのケースは自明に1.0なので省略。<b>色＝タグ</b>。</p>
<figure>{fig_ratio_stack()}<figcaption>各棒＝1ケース。積み上げがそのケース内のタグ配分。
例：「同性パートナー遺族給付金」＝ジェンダー0.89／刑事0.11 のように主題と傍論が分かれる。</figcaption></figure>

<h4>全ケースの割合（検索可） <span class="en">All cases (searchable)</span></h4>
<p><input id="search" placeholder="ケース名・タグで絞り込み / filter…">
<label class="af"><input type="checkbox" id="tbl-active"> アクティブのみ / active only（{n_active}件）</label></p>
<div id="rt"><table><thead><tr><th>ケース / case</th><th>タグ割合 / tag share</th></tr></thead>
<tbody id="rtb"></tbody></table></div>

<h4>二部グラフ（ホバーで割合表示）<span class="en">Bipartite graph — hover a case for its tag share</span></h4>
<p class="hint">🖱 <b>ケース（丸）にカーソルを合わせると、そのケースのタグ割合</b>（と貼り忘れ候補）を表示、
<b>クリックでCALL4のページ</b>へ。タグ（四角■）にカーソルを合わせると件数を表示。
辺の太さ＝割合、ケースの色＝最も割合の高いタグ、丸の大きさ＝保有タグ数。
Hover a circle to see its tag share (edge thickness = share); click to open its CALL4 page.</p>
<div id="bip-wrap">
  <div id="bip-btns"></div>
  <label class="af" style="margin:2px 4px 8px"><input type="checkbox" id="bip-active">
    アクティブなケースのみ表示 / show active cases only（{n_active}件）</label>
  <canvas id="bip" width="1040" height="740"></canvas>
  <div id="bip-legend"></div>
</div>

<h2>4. 貼り忘れ候補 <span class="en">Missing-tag candidates</span></h2>
<p>付いていないが、テキスト上は該当が強いタグ。<b>%</b>＝「実際にそのタグが付くケースの何％より該当が強いか」。
<b>あくまで人手レビュー用の仮説</b>。上位{min(25, n_missing)}件（全{n_missing}件）。</p>
<table><thead><tr><th>ケース</th><th>提案タグ</th><th>disc</th><th>陽性群での位置</th></tr></thead>
<tbody>{missing_rows()}</tbody></table>

<h2>5. 過剰タグ候補（副産物）<span class="en">Weak / over-tag candidates</span></h2>
<p>付いてはいるが、テキスト上の該当が弱い（非該当ケースの中央値並み）タグ。誤タグor形式的タグの候補。</p>
<table><thead><tr><th>ケース</th><th>該当タグ</th><th>disc</th><th>ケース内割合</th></tr></thead>
<tbody>{over_rows()}</tbody></table>

<h2>6. 限界 <span class="en">Limitations</span></h2>
<ul>
<li>重みの根拠は<b>テキストと埋め込みモデル(bge-m3)</b>であり、法的な重要度そのものではない。</li>
<li>貼り忘れ・過剰タグは<b>仮説</b>。特に横断タグ（公正な手続）・希少タグ（沖縄・個人情報）は低信頼。</li>
<li>N=95・タグ11の小規模。プロトタイプは少数ケースの平均に過ぎず、外れ値に弱い。</li>
<li>本分析は探索であり、当事者の類型化・評価を意図しない。</li>
</ul>

<h2>7. 再現情報 <span class="en">Reproducibility</span></h2>
<p>seed=42。埋め込み：<code>plan02/features/emb.npz</code>（bge-m3, 95×1024, L2正規化・再利用）。
手法：判別的プロトタイプ＋LOO、softmax割合（τ={d['tau']}）、ROC-AUC検証。
スクリプト：<code>plan05/s0_softtag.py</code>・<code>s1_report.py</code>。結果：<code>plan05/results/soft_tags.json</code>。生成日：2026-07-24。</p>

<div id="tip"></div>
<script>
const ROWS = {ratio_table_json()};
const tb = document.getElementById('rtb');
const tblActive = document.getElementById('tbl-active');
function render(q) {{
  q = (q || '').toLowerCase();
  const actOnly = tblActive.checked;
  tb.innerHTML = '';
  ROWS.forEach(r => {{
    if (actOnly && r.st !== 'active') return;
    const hay = (r.title + ' ' + r.tags.map(t => t.t).join(' ')).toLowerCase();
    if (q && !hay.includes(q)) return;
    const segs = r.tags.map(t =>
      `<span class="rseg" style="background:${{t.c}};width:${{(t.v*100).toFixed(1)}}%" title="${{t.t}} ${{(t.v*100).toFixed(0)}}%">`
      + (t.v >= 0.15 ? (t.v*100).toFixed(0)+'%' : '') + `</span>`).join('');
    const lbl = r.tags.map(t => `${{t.t}} ${{(t.v*100).toFixed(0)}}%`).join(' ／ ');
    tb.insertAdjacentHTML('beforeend',
      `<tr><td><a href="https://www.call4.jp/info.php?type=items&id=${{r.id}}" target="_blank">${{r.title}}</a></td>`
      + `<td><div class="rbar">${{segs}}</div><div class="rlbl">${{lbl}}</div></td></tr>`);
  }});
}}
render('');
document.getElementById('search').addEventListener('input', e => render(e.target.value));
tblActive.addEventListener('change', () => render(document.getElementById('search').value));

// ---------- interactive bipartite tag<->case graph ----------
const BTAGS = {bip_tags_json};
const BCASES = {bip_cases_json};
const BEDGES = {bip_edges_json};
const BLAYOUTS = [["radial", "放射状 Radial（割合）"],
                  ["kamada_r", "Kamada–Kawai（割合）"],
                  ["force_r", "力学 Force（割合）"],
                  ["kamada", "Kamada–Kawai（等重み）"],
                  ["force", "力学 Force（等重み）"],
                  ["columns", "二列 Columns"]];
(function() {{
  const cv = document.getElementById('bip'), cx = cv.getContext('2d');
  const tip2 = document.getElementById('tip');
  const PAD = 96;                       // room for label pills at the edges
  let LAY = "radial", sx = 1, sy = 1, ox = 0, oy = 0;
  const actChk = document.getElementById('bip-active');
  const shown = c => !actChk.checked || c.st === 'active';
  // coordinate set: active-only cases re-layout via posA, all-cases via pos
  const co = n => (actChk.checked ? (n.posA || n.pos) : n.pos)[LAY];
  function fit() {{
    // fit only the currently-visible nodes (so active-only re-frames tightly)
    const vis = BTAGS.concat(BCASES.filter(shown));
    const X = vis.map(n => co(n)[0]), Y = vis.map(n => co(n)[1]);
    const xmin = Math.min(...X), xmax = Math.max(...X);
    const ymin = Math.min(...Y), ymax = Math.max(...Y);
    const s = Math.min((cv.width - 2 * PAD) / (xmax - xmin || 1),
                       (cv.height - 2 * PAD) / (ymax - ymin || 1));
    sx = sy = s;
    ox = (cv.width - s * (xmax + xmin)) / 2;
    oy = (cv.height - s * (ymax + ymin)) / 2;
  }}
  const PX = n => ox + sx * co(n)[0];
  const PY = n => oy + sy * co(n)[1];
  const caseR = c => 4 + (c.ndeg - 1) * 2.4;
  // --- stylish tag-label pills (readable on any tag color) ---
  const LABEL_FONT = 'bold 12.5px -apple-system,"Hiragino Sans","Yu Gothic",sans-serif';
  const _dim = new Map();
  function tagDims(t) {{
    if (_dim.has(t.name)) return _dim.get(t.name);
    cx.font = LABEL_FONT;
    const w = cx.measureText(t.name).width + 20;
    const h = 20 + Math.min(6, Math.sqrt(t.count));      // subtle size = count
    const o = {{w, h}}; _dim.set(t.name, o); return o;
  }}
  function rr(x, y, w, h, r) {{
    cx.beginPath();
    cx.moveTo(x + r, y);
    cx.arcTo(x + w, y, x + w, y + h, r); cx.arcTo(x + w, y + h, x, y + h, r);
    cx.arcTo(x, y + h, x, y, r); cx.arcTo(x, y, x + w, y, r); cx.closePath();
  }}
  function lum(hex) {{
    const n = parseInt(hex.slice(1), 16);
    return 0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255);
  }}
  fit();

  function draw() {{
    cx.clearRect(0, 0, cv.width, cv.height);
    // edges: color = tag color, width & alpha proportional to ratio
    BEDGES.forEach(([ti, ci, v]) => {{
      const t = BTAGS[ti], c = BCASES[ci];
      if (!shown(c)) return;
      cx.beginPath(); cx.moveTo(PX(t), PY(t)); cx.lineTo(PX(c), PY(c));
      cx.strokeStyle = t.color; cx.globalAlpha = 0.25 + 0.6 * v;
      cx.lineWidth = 0.5 + 3.5 * v; cx.stroke();
    }});
    cx.globalAlpha = 1;
    // case circles (fill = dominant tag color; size = #tags)
    BCASES.forEach(c => {{
      if (!shown(c)) return;
      cx.beginPath(); cx.arc(PX(c), PY(c), caseR(c), 0, Math.PI * 2);
      cx.fillStyle = '#8593a6'; cx.globalAlpha = 0.88; cx.fill();
      cx.globalAlpha = 1; cx.strokeStyle = '#fff'; cx.lineWidth = 1; cx.stroke();
    }});
    // tag nodes: rounded color pill with its name (readable on any color)
    BTAGS.forEach(t => {{
      const {{w, h}} = tagDims(t), x = PX(t), y = PY(t);
      const rx = x - w / 2, ry = y - h / 2;
      cx.save();
      cx.shadowColor = 'rgba(20,30,50,.25)'; cx.shadowBlur = 6; cx.shadowOffsetY = 1.5;
      cx.fillStyle = t.color; rr(rx, ry, w, h, h / 2); cx.fill();
      cx.restore();
      cx.strokeStyle = 'rgba(255,255,255,.92)'; cx.lineWidth = 1.5;
      rr(rx, ry, w, h, h / 2); cx.stroke();
      const dark = lum(t.color) > 150;                    // light pill -> dark text
      cx.font = LABEL_FONT; cx.textAlign = 'center'; cx.textBaseline = 'middle';
      cx.lineJoin = 'round';
      cx.strokeStyle = dark ? 'rgba(255,255,255,.9)' : 'rgba(0,0,0,.38)';
      cx.lineWidth = 2.6; cx.strokeText(t.name, x, y);     // contrasting halo
      cx.fillStyle = dark ? '#17202e' : '#ffffff';
      cx.fillText(t.name, x, y);
    }});
  }}
  draw();

  const lg = document.getElementById('bip-legend');
  lg.insertAdjacentHTML('beforeend',
    `<span style="color:#666">▬ 色ラベル = タグ tag（色は個別）／● ケース case（大きさ=タグ数・辺の色/太さ=タグと割合）</span>`);

  // layout switch buttons
  const bb = document.getElementById('bip-btns');
  BLAYOUTS.forEach(([key, label]) => {{
    const b = document.createElement('button');
    b.textContent = label; b.className = 'lbtn' + (key === LAY ? ' on' : '');
    b.onclick = () => {{
      LAY = key; fit(); draw();
      bb.querySelectorAll('.lbtn').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
    }};
    bb.appendChild(b);
  }});
  actChk.addEventListener('change', () => {{ fit(); draw(); }});

  function hit(ev) {{
    const r = cv.getBoundingClientRect();
    const mx = (ev.clientX - r.left) * (cv.width / r.width);
    const my = (ev.clientY - r.top) * (cv.height / r.height);
    for (let i = 0; i < BTAGS.length; i++) {{
      const t = BTAGS[i], {{w, h}} = tagDims(t);
      if (Math.abs(mx - PX(t)) <= w / 2 && Math.abs(my - PY(t)) <= h / 2)
        return {{kind: 'tag', obj: t}};
    }}
    let bestD = 1e9, best = null;
    BCASES.forEach(c => {{
      if (!shown(c)) return;
      const dd = Math.hypot(mx - PX(c), my - PY(c)), rr = caseR(c) + 3;
      if (dd <= rr && dd < bestD) {{ bestD = dd; best = c; }}
    }});
    return best ? {{kind: 'case', obj: best}} : null;
  }}

  cv.addEventListener('mousemove', ev => {{
    const h = hit(ev);
    if (!h) {{ tip2.style.display = 'none'; cv.style.cursor = 'default'; return; }}
    let html;
    if (h.kind === 'tag') {{
      html = `<b>${{h.obj.name}}</b><br>${{h.obj.count}} 件に付与 / cases`;
      cv.style.cursor = 'default';
    }} else {{
      const c = h.obj;
      const bars = c.tags.map(t =>
        `<div class="trow"><span class="tsw" style="background:${{t.c}}"></span>`
        + `<span style="flex:none;min-width:118px">${{t.t}}</span>`
        + `<span class="tbar"><span class="tbf" style="width:${{(t.v*100).toFixed(0)}}%;background:${{t.c}}"></span></span>`
        + `<span style="flex:none">${{(t.v*100).toFixed(0)}}%</span></div>`).join('');
      let miss = '';
      if (c.miss && c.miss.length) miss = `<div style="margin-top:5px;color:#f2c94c">貼り忘れ候補: `
        + c.miss.map(m => `${{m.t}}(${{(m.pct*100).toFixed(0)}}%${{m.lc?'⚠':''}})`).join('、') + `</div>`;
      html = `<b>${{c.title}}</b>${{bars}}${{miss}}`
        + `<div style="margin-top:4px;color:#9ecbff">クリックで開く / click to open</div>`;
      cv.style.cursor = 'pointer';
    }}
    tip2.innerHTML = html; tip2.style.display = 'block';
    tip2.style.left = Math.min(ev.clientX + 14, window.innerWidth - 310) + 'px';
    tip2.style.top = Math.min(ev.clientY + 12, window.innerHeight - 160) + 'px';
  }});
  cv.addEventListener('mouseleave', () => {{ tip2.style.display = 'none'; }});
  cv.addEventListener('click', ev => {{
    const h = hit(ev);
    if (h && h.kind === 'case')
      window.open('https://www.call4.jp/info.php?type=items&id=' + h.obj.id, '_blank');
  }});
}})();
</script>
</body></html>"""

out = os.path.join(REP, "call4_plan05_report.html")
open(out, "w", encoding="utf-8").write(HTML)
print(f"[s1] wrote {out}  ({len(HTML)/1024:.0f} KB)")
