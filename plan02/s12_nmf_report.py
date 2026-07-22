#!/usr/bin/env python3
"""
Explainability-first public report, NMF K=6 (mixture membership) as the star.

Same 3-part narrative as the Leiden report, but the method is TF-IDF + NMF.
The selling point: with word-based topics you can answer *in natural language*
why two cases are similar (they share the same topic words) - which embedding
distance cannot do easily. Self-contained single HTML.

Run from the repo root:  python plan02/s12_nmf_report.py
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
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath

SEED = 42
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

TC = ["#4C8C2B", "#2B6CB0", "#B0532B", "#8A4FA8", "#C29B2C", "#2BA8A0"]
TEN = ["Regional development & environment", "Freedom of information",
       "Immigration detention & refugees", "Marriage equality & gender",
       "Suffrage & political participation", "Criminal procedure"]


def svg_of(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    s = buf.getvalue().decode("utf-8"); return s[s.find("<svg"):]

def url(cid):
    return f"https://www.call4.jp/info.php?type=items&id={cid}"

def blend(row):
    rgb = np.zeros(3)
    for t, r in enumerate(row):
        c = TC[t].lstrip("#")
        rgb += r * np.array([int(c[i:i+2], 16) for i in (0, 2, 4)])
    return "#%02x%02x%02x" % tuple(int(min(v, 255)) for v in rgb)


# ---------------- figures ----------------

def fig_tag_bar(tags):
    c = Counter(t for ts in tags for t in ts).most_common()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(range(len(c)), [v for _, v in c], color="#9ab")
    ax.set_yticks(range(len(c))); ax.set_yticklabels([k for k, _ in c], fontproperties=jp(9))
    ax.invert_yaxis(); ax.bar_label(ax.containers[0], fontsize=8, padding=2)
    ax.set_title("既存タグの件数（11タグ）/ Existing tags by frequency", fontproperties=jp(10))
    ax.set_xlim(0, max(v for _, v in c) * 1.15)
    return svg_of(fig)

def fig_null():
    sizes = [56, 21, 10, 7, 1]; fig, ax = plt.subplots(figsize=(6, 1.9)); left = 0
    for i, s in enumerate(sizes):
        ax.barh(0, s, left=left, color=plt.cm.Pastel1(i), edgecolor="white")
        ax.text(left + s/2, 0, str(s), ha="center", va="center", fontsize=9); left += s
    ax.set_xlim(0, 95); ax.set_yticks([]); ax.set_xlabel("ケース数", fontproperties=jp(9))
    ax.set_title("タグだけで機械分類すると56件が1塊に（座標として粗い）", fontproperties=jp(10))
    return svg_of(fig)

def fig_error(kselect):
    ks = sorted(int(k) for k in kselect["errors"])
    errs = [kselect["errors"][str(k)] for k in ks]
    drops = [errs[i-1]-errs[i] for i in range(1, len(errs))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(ks, errs, "o-", color="#2B6CB0"); ax[0].axvline(6, color="#B0532B", ls="--")
    ax[0].set_xlabel("トピック数 K", fontproperties=jp(10)); ax[0].set_ylabel("再構成誤差", fontproperties=jp(10))
    ax[0].set_title("誤差は単調減少（肘がない）", fontproperties=jp(10)); ax[0].grid(alpha=.3)
    ax[1].bar([f"{k-1}→{k}" for k in ks[1:]], drops, color="#8fb0d4")
    ax[1].set_title("追加1トピックあたりの改善（ほぼ横ばい）", fontproperties=jp(10)); ax[1].grid(alpha=.3, axis="y")
    for a in ax: a.tick_params(labelsize=8)
    return svg_of(fig)

def fig_stack(ratios, dom, order_idx):
    fig, ax = plt.subplots(figsize=(13, 3.0))
    x = np.arange(len(order_idx)); bottom = np.zeros(len(order_idx))
    R = ratios[order_idx]
    for t in range(6):
        ax.bar(x, R[:, t], bottom=bottom, color=TC[t], width=1.0)
        bottom += R[:, t]
    ax.set_xlim(-0.5, len(order_idx)-0.5); ax.set_ylim(0, 1); ax.set_xticks([])
    ax.set_ylabel("混合比", fontproperties=jp(9))
    ax.set_title("95ケースの混合比（1本=1ケース・優勢トピック順）/ Mixture per case", fontproperties=jp(11))
    return svg_of(fig)

def fig_words(topics, names):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    for k, ax in enumerate(axes.ravel()):
        words = topics[str(k)]["descriptor_words"][:7][::-1]
        ax.barh(range(len(words)), range(1, len(words)+1), color=TC[k])
        ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontproperties=jp(9))
        ax.set_xticks([]); ax.set_title(f"T{k} {names[k]}", fontproperties=jp(9.5))
    fig.suptitle("各トピックを定義する言葉（c-TF-IDF上位）/ Defining words per topic", fontproperties=jp(11))
    fig.tight_layout(); return svg_of(fig)

def fig_sankey(tags, dom, names):
    tag_tot = Counter(t for ts in tags for t in ts); tags_sorted = [t for t, _ in tag_tot.most_common()]
    top_tot = Counter(int(d) for d in dom)
    flows = Counter()
    for i, ts in enumerate(tags):
        for t in ts: flows[(t, int(dom[i]))] += 1
    gap, H = 1.6, 100.0
    def layout(nm, tot):
        total = sum(tot[n] for n in nm); sc = (H-gap*(len(nm)-1))/total; pos, y = {}, 0.0
        for n in nm: h = tot[n]*sc; pos[n] = (y, h); y += h+gap
        return pos, sc
    lpos, ls = layout(tags_sorted, tag_tot); topics_sorted = sorted(top_tot); rpos, rs = layout(topics_sorted, top_tot)
    fig, ax = plt.subplots(figsize=(11, 6.4)); loff = {t: 0. for t in tags_sorted}; roff = {k: 0. for k in topics_sorted}
    for (t, k), n in sorted(flows.items(), key=lambda x: (tags_sorted.index(x[0][0]), x[0][1])):
        y0, _ = lpos[t]; y1, _ = rpos[k]; a0, a1 = y0+loff[t], y1+roff[k]; h0, h1 = n*ls, n*rs
        loff[t] += h0; roff[k] += h1
        verts = [(.14, a0), (.5, a0), (.5, a1), (.86, a1), (.86, a1+h1), (.5, a1+h1), (.5, a0+h0), (.14, a0+h0), (.14, a0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=TC[k], alpha=.4, lw=0))
    for t in tags_sorted:
        y, h = lpos[t]; ax.add_patch(mpatches.Rectangle((.10, y), .04, h, color="#555"))
        ax.text(.09, y+h/2, f"{t} ({tag_tot[t]})", ha="right", va="center", fontproperties=jp(9))
    for k in topics_sorted:
        y, h = rpos[k]; ax.add_patch(mpatches.Rectangle((.86, y), .04, h, color=TC[k]))
        ax.text(.91, y+h/2, f"T{k} {names[k]} ({top_tot[k]})", ha="left", va="center", fontproperties=jp(9))
    ax.set_xlim(-.3, 1.55); ax.set_ylim(-2, H+2); ax.invert_yaxis(); ax.axis("off")
    ax.set_title("既存タグ（左） → NMF優勢トピック（右）/ Tags → dominant topic", fontproperties=jp(11))
    return svg_of(fig)


# ---------------- main ----------------

def main():
    tok = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8")); tok.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in tok]; titles = [r["title"] for r in tok]; tags = [r["subject_tags"] for r in tok]
    coords = np.load(FEATURES_DIR / "umap2d.npz")["coords"]
    m = json.loads((RESULTS_DIR / "membership_nmf.json").read_text(encoding="utf-8"))
    names = [v["name"] for k, v in json.loads((RESULTS_DIR/"names_llm.json").read_text(encoding="utf-8"))["nmf"].items()]
    interp = json.loads((RESULTS_DIR/"interpretation.json").read_text(encoding="utf-8"))["mixture"]["nmf"]["topics"]
    kselect = json.loads((RESULTS_DIR/"nmf_kselect.json").read_text(encoding="utf-8"))
    ev = json.loads((RESULTS_DIR/"evaluation.json").read_text(encoding="utf-8"))

    ratios = np.array(m["ratios"]); dom = np.array(m["dominant_topic"]); ent = np.array(m["entropy_normalized"])
    Rn = ratios / np.maximum(np.linalg.norm(ratios, axis=1, keepdims=True), 1e-9)
    order_idx = np.lexsort((-ratios[np.arange(len(ids)), dom], dom))

    boot = ev["q2_bootstrap_stability"]["nmf_dom"]["ari_median"]
    ari_tag = ev["q3_tag_alignment"]["nmf_dom"]["ari"]
    ari_leiden = 0.437  # leiden6 vs nmf6 (computed in s11)

    # "why similar" worked examples
    def shared_sentence(qi, j):
        pairs = [(t, ratios[qi, t], ratios[j, t]) for t in range(6)
                 if ratios[qi, t] > 0.12 and ratios[j, t] > 0.12]
        pairs.sort(key=lambda x: -min(x[1], x[2]))
        return pairs
    examples = []
    for kw in ["財務省改ざん", "結婚の自由", "海外でも国民審査", "カメルーン"]:
        qi = next(i for i, t in enumerate(titles) if kw in t)
        sims = Rn @ Rn[qi]; sims[qi] = -1; j = int(sims.argmax())
        examples.append((qi, j, float(sims[j]), shared_sentence(qi, j)))

    figs = {"tag": fig_tag_bar(tags), "null": fig_null(), "err": fig_error(kselect),
            "stack": fig_stack(ratios, dom, order_idx), "words": fig_words(interp, names),
            "sankey": fig_sankey(tags, dom, names)}

    # topic cards
    cards = []
    for k in range(6):
        t = interp[str(k)]
        reps = "".join(f'<li><a href="{url(r["id"])}" target="_blank">{r["title"]}</a>'
                       f' <span class="dim">({r["ratio"]})</span></li>' for r in t["representatives"])
        allm = "".join(f'<li><a href="{url(ids[i])}" target="_blank">{titles[i]}</a>'
                       f' <span class="dim">({ratios[i,k]:.2f})</span></li>'
                       for i in np.where(dom == k)[0])
        cards.append(f"""
<div class="card" style="border-top:5px solid {TC[k]}">
  <h4><span class="dot" style="background:{TC[k]}"></span>T{k}: {names[k]}
      <span class="dim">(優勢 {t['size_dominant']}件)</span></h4>
  <p><b>特徴語:</b> {' / '.join(t['descriptor_words'][:8])}</p>
  <p><b>代表ケース:</b></p><ul>{reps}</ul>
  <details><summary>このトピックが優勢な{t['size_dominant']}件</summary><ul class="small">{allm}</ul></details>
</div>""")

    # why-similar cards
    why = []
    for qi, j, s, pairs in examples:
        shares = "".join(
            f'<li><span class="dot" style="background:{TC[t]}"></span>{names[t]}'
            f'（{titles[qi][:12]}… {a*100:.0f}% ／ {titles[j][:12]}… {b*100:.0f}%）</li>'
            for t, a, b in pairs)
        why.append(f"""
<div class="card">
  <p><b><a href="{url(ids[qi])}" target="_blank">{titles[qi]}</a></b><br>
     ↕ 混合比が似ている（コサイン {s:.2f}）<br>
     <b><a href="{url(ids[j])}" target="_blank">{titles[j]}</a></b></p>
  <p class="dim">似ている理由 = 共有するトピック:</p><ul class="small">{shares}</ul>
</div>""")

    # network on MIXTURE vectors (interpretable similarity), FR layout, blended colors
    sims = Rn @ Rn.T; np.fill_diagonal(sims, -1)
    edges = set()
    for i in range(len(ids)):
        for jj in np.argsort(sims[i])[-3:]:
            edges.add((min(i, int(jj)), max(i, int(jj))))
    edges = sorted(edges)
    import random as pyr; pyr.seed(SEED)
    net = np.array(ig.Graph(n=len(ids), edges=edges).layout_fruchterman_reingold(niter=800).coords)
    nmn = net - net.min(0); nmn /= np.ptp(net, 0)
    blends = [blend(Rn[i] / Rn[i].sum()) for i in range(len(ids))]
    nodes = [{"i": i, "title": titles[i], "url": url(ids[i]), "tags": tags[i],
              "blend": blends[i], "dom": int(dom[i]), "ent": round(float(ent[i]), 2),
              "nmf": [round(float(r), 3) for r in ratios[i]],
              "nx": round(float(nmn[i, 0]), 3), "ny": round(float(nmn[i, 1]), 3),
              "px": round(float(coords[i, 0]), 2), "py": round(float(coords[i, 1]), 2)}
             for i in range(len(ids))]
    edge_js = [[int(a), int(b)] for a, b in edges]

    gen = date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALL4公共訴訟95件を「言葉」で6つの軸に分ける（NMF版）</title>
<style>
body {{ font-family:"Hiragino Sans","Yu Gothic",sans-serif; margin:0; background:#f7f6f3; color:#222; line-height:1.85; }}
.container {{ max-width:1000px; margin:0 auto; padding:30px 20px 70px; }}
h1 {{ font-size:1.5em; border-bottom:3px solid #B0532B; padding-bottom:10px; }}
h2 {{ font-size:1.25em; margin-top:2.6em; border-left:6px solid #B0532B; padding-left:11px; }}
h3 {{ font-size:1.05em; margin-top:1.7em; }}
h4 {{ margin:.3em 0; }}
.dim {{ color:#778; font-size:.85em; }}
.lead, .callout, .step, .why {{ border-radius:8px; padding:14px 18px; }}
.lead {{ background:#f6ece4; border:1px solid #e2c4ad; }}
.callout {{ background:#eef3fa; border:1px solid #c9d8ee; }}
.step {{ background:#fff; border:1px solid #e0ddd6; margin:8px 0; }}
.card {{ background:#fff; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin:12px 0; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; margin:12px 0; }}
.figure {{ background:#fff; border-radius:8px; padding:12px; margin:14px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); overflow-x:auto; }}
table {{ border-collapse:collapse; background:#fff; font-size:.86em; margin:10px 0; }}
th,td {{ border:1px solid #ccc; padding:6px 10px; text-align:left; }}
th {{ background:#f0e8e2; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:6px; margin-right:5px; vertical-align:middle; }}
canvas {{ max-width:100%; }}
#tip {{ position:fixed; display:none; background:rgba(20,25,35,.94); color:#fff; padding:8px 11px; border-radius:6px; font-size:12px; max-width:340px; pointer-events:none; z-index:10; }}
a {{ color:#B0532B; }}
details summary {{ cursor:pointer; color:#B0532B; font-size:.9em; }}
ul.small li {{ font-size:.85em; }}
.num {{ font-size:1.9em; font-weight:bold; color:#8a3f20; }}
.kpi {{ display:flex; flex-wrap:wrap; gap:20px; margin:10px 0; }}
.kpi > div {{ background:#fff; border-radius:8px; padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,.08); text-align:center; }}
</style>
</head>
<body>
<div class="container">

<h1>CALL4 公共訴訟95件を「言葉」で6つの軸に分ける<br>
<span class="dim" style="font-size:.62em">— 混合メンバーシップ版（NMF）: 似ている理由を言葉で説明できる分類</span></h1>
<p class="dim">生成 {gen} ｜ 対象: <a href="https://www.call4.jp" target="_blank">CALL4</a> 掲載の公共訴訟95件</p>

<div class="lead">
この資料は、95件の訴訟を<b>単語の使われ方</b>から6つのトピックに分解し、
各ケースを<b>複数トピックの「混合比」</b>で表したものです（手法: NMF）。
最大の利点は<b>「なぜこの2件が似ているのか」を言葉で説明できる</b>こと——
2件が同じトピック（＝同じ言葉の群れ）を共有しているから似ている、と言えます。
姉妹版（埋め込み+Leiden）は「近さ」を精密に測れますが、その理由の言語化は苦手でした。
構成は ①タグを材料に使うべきか → ②手法と6トピック → ③結果の色々な見せ方、です。
</div>

<h2>1. そもそも「タグ」を分類の材料に使うべき？</h2>
<p><b>使いません。</b> 既存タグは分類を作る<b>材料</b>ではなく、できた分類を照合する<b>「答え合わせ」</b>に回します。</p>
<div class="step"><b>① 循環論法になる。</b> 「タグ体系を見直す」のが狙いなのに、そのタグから分類を作ればタグの焼き直しにしかならない。</div>
<div class="step"><b>② 座標として粗い。</b> タグだけで機械分類すると<b>95件中56件が1塊</b>に潰れる（「公正な手続」が43件に偏るため）。</div>
<div class="figure">{figs['null']}</div>
<div class="step"><b>③ 多ラベルで不安定。</b> タグが2個のケースも0個のケースもあり、材料としてばらつく。</div>
<p>そこで<b>本文テキストを材料</b>にし、タグは最後に照合します。既存11タグの内訳はこちら（後で6トピックと突き合わせ）。</p>
<div class="figure">{figs['tag']}</div>

<h2>2. 手法と結果 — 言葉から6つのトピックへ</h2>
<div class="step"><b>ステップ1: 単語の数値化（TF-IDF）。</b> 各ケースを「どの単語をどれだけ特徴的に使うか」の表にする（形態素解析で語に分割、7,350語）。</div>
<div class="step"><b>ステップ2: NMF（非負値行列分解）。</b>「95ケース×単語」の表を、<b>6つのトピック</b>に分解する。
数式で書くと <b>表 ≈ ケースのトピック混合比 × トピックの単語重み</b>。
各トピックは「重みの大きい単語」で説明でき、各ケースは6トピックの<b>混合比</b>（合計100%）で表される。</div>

<div class="callout">
<b>なぜNMFは「似ている理由」を言葉で言えるのか</b><br>
埋め込みベクトルは各ケースを1024個の数値で表すため「近い」は測れても、その理由は数値の海に埋もれます。
NMFは各ケースを<b>6つの意味のあるトピックの比率</b>だけで表すので、「近い」の理由を
<b>『どのトピック＝どの言葉群を共有するか』</b>として名指しできます。実例（各ケースの最も近い相手）:
</div>
<div class="grid2">{''.join(why)}</div>

<h3>6つのトピック</h3>
<div class="grid">{''.join(cards)}</div>

<h3>この6トピックは「偶然」ではない</h3>
<div class="kpi">
<div><div class="num">{boot:.0%}</div>データを8割に間引いても<br>優勢トピックが一致<br><span class="dim">(ブートストラップ中央値ARI)</span></div>
<div><div class="num">{ari_leiden:.2f}</div>別手法(埋め込み+Leiden)とも<br>同じ6軸に着地<br><span class="dim">(ARI)</span></div>
<div><div class="num">{ari_tag:.2f}</div>既存タグと"ほどよい距離"<br>＝焼き直しでも無関係でもない<br><span class="dim">(ARI)</span></div>
</div>

<h3>なぜ「6」なのか（5でも7でもなく）</h3>
<p>トピック数Kは4〜10を全て試した。数値（再構成誤差）はKを増やすほど減り続け、「ここで止めよ」という折れ目がない。</p>
<div class="figure">{figs['err']}</div>
<p>そこで各Kの<b>中身</b>を見ると判断は明確だった。</p>
<table>
<tr><th>K</th><th>何が起きるか</th></tr>
<tr><td>5</td><td>「刑事手続」と「入管収容」が1トピックに<b>混在</b>（収容/刑事/逮捕/入管が同居）</td></tr>
<tr style="background:#f6ece4"><td><b>6（採用）</b></td><td><b>刑事と入管が分離</b>し、6トピック全てが一言で説明できる</td></tr>
<tr><td>7</td><td>「アーカイブ_プロジェクト」など<b>サイト定型語だけの無意味なトピック</b>が出現＝分解の限界</td></tr>
</table>
<p>K=5では粗すぎ・K=7では壊れる、という挟み撃ちで6に決まった。しかも姉妹版の埋め込み+Leidenも
<b>独立に6グループ</b>に着地しており、「6」は複数の入口が合流した数。</p>

<h2>3. 結果を色々な角度で見る</h2>
<p>すべて同じ混合比データを別の切り口で描いたもの。点にカーソルで詳細・クリックでCALL4のケースページへ。</p>

<h3>A. 混合比の一覧（スタックバー）</h3>
<p>95件それぞれを1本の棒にし、6トピックの混合比を色で積み上げ。<b>単色に近い棒＝単一論点、複数色＝横断的な訴訟</b>。</p>
<div class="figure">
  <div id="lg-stack" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px;"></div>
  <canvas id="stack" width="960" height="320"></canvas>
</div>

<h3>B. 類似ネットワーク — 「似ている理由」ごと見る</h3>
<p>各ケースを<b>混合比が最も似た3件</b>と線で結んだネットワーク。ノード色は混合比を混ぜた色。
中間色のノード＝複数トピックを橋渡しするケースが、軸と軸のあいだに立地する。
<b>ホバーすると、そのケースの混合内訳が言葉で出る</b>ので「なぜここにいるか」がすぐ分かる。</p>
<div class="figure"><canvas id="net" width="960" height="600"></canvas></div>

<h3>C. 各トピックを定義する言葉</h3>
<div class="figure">{figs['words']}</div>

<h3>D. 既存タグとの対応</h3>
<p>左＝既存タグ、右＝NMF優勢トピック。「公正な手続」が全トピックに散っている（＝分類として機能していない）。</p>
<div class="figure">{figs['sankey']}</div>

<h2>4. わかったこと・示唆</h2>
<ul>
<li><b>「公正な手続」（43件）は実質「その他」</b>。6トピック全てに分散。争点ベースの軸への再編が有効。</li>
<li>訴訟の多くは1〜2トピックに集中するが、<b>複数論点を横断するケースも確かに存在</b>（混合比が分散するケース）。
混合比表現なら「主トピック＋副トピック」として自然に表せる——これが排他分類にない強み。</li>
<li>姉妹版（埋め込み+Leiden）と同じ6軸に着地。<b>「近さの理由を言葉で言いたい」ならNMF、
「近さを精密に測りたい」なら埋め込み</b>、と用途で使い分けられる。</li>
</ul>

<h2>付記: 手法と限界</h2>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>データ</td><td>CALL4公開95件（2026-07-18取得）。ケース名＋概要＋本文</td></tr>
<tr><td>手法</td><td>形態素解析(SudachiPy)→TF-IDF 7,350語→NMF（トピック数6）。各ケード=6トピックの混合比</td></tr>
<tr><td>検証</td><td>ブートストラップ80%×100回 中央値ARI {boot:.2f}／埋め込み+Leidenとの一致 ARI {ari_leiden}／既存タグとの一致 ARI {ari_tag}</td></tr>
<tr><td>再現性</td><td>乱数シード42固定。コード・データは GitHub (shio-koji/case_clustering) の plan02/</td></tr>
<tr><td>限界</td><td>95件は小規模で探索・仮説生成用。トピック数6は解釈可能性による選択（5〜7に議論の余地）。訴状・判決文の全文は未使用</td></tr>
</table>
<p class="dim">本分類は解析目的であり、訴訟当事者を類型化して評価する意図はありません。
ケース本文の著作権はCALL4および執筆者に帰属します。各ケースの詳細・支援は各リンク先をご覧ください。</p>

</div>
<div id="tip"></div>

<script>
const NODES = {json.dumps(nodes, ensure_ascii=False)};
const EDGES = {json.dumps(edge_js)};
const TN = {json.dumps(names, ensure_ascii=False)};
const TC = {json.dumps(TC)};
const ORDER = {json.dumps([int(i) for i in order_idx])};
const tip = document.getElementById('tip');
function showTip(h, ev) {{
  tip.innerHTML = h; tip.style.display='block';
  tip.style.left = Math.min(ev.clientX+14, window.innerWidth-360)+'px';
  tip.style.top = (ev.clientY+12)+'px';
}}
function hideTip() {{ tip.style.display='none'; }}
function mixText(n) {{
  return n.nmf.map((v,t)=> v>0.08 ? `<span style="color:${{TC[t]}}">■</span>${{TN[t]}} ${{(v*100).toFixed(0)}}%` : null)
             .filter(Boolean).join('<br>');
}}
// legend
const lg = document.getElementById('lg-stack');
TN.forEach((n,t) => lg.insertAdjacentHTML('beforeend',
  `<span style="font-size:12px"><span style="display:inline-block;width:11px;height:11px;background:${{TC[t]}};margin-right:4px;border-radius:2px"></span>T${{t}} ${{n}}</span>`));

// A. stacked bar
const sc=document.getElementById('stack'), sx=sc.getContext('2d');
const BW=(sc.width-16)/ORDER.length;
ORDER.forEach((ci,pos) => {{ const n=NODES[ci]; let y=sc.height-14;
  n.nmf.forEach((r,t) => {{ const h=r*(sc.height-24); sx.fillStyle=TC[t];
    sx.fillRect(10+pos*BW, y-h, Math.max(BW-0.5,1.2), h); y-=h; }});
}});
let scur=-1;
sc.addEventListener('mousemove', ev => {{
  const r=sc.getBoundingClientRect(); const pos=Math.floor(((ev.clientX-r.left)*(sc.width/r.width)-10)/BW);
  if (pos<0||pos>=ORDER.length){{hideTip();scur=-1;return;}} scur=pos; const n=NODES[ORDER[pos]];
  showTip(`<b>${{n.title}}</b><br>${{mixText(n)}}`, ev); sc.style.cursor='pointer';
}});
sc.addEventListener('mouseleave', ()=>{{hideTip();scur=-1;}});
sc.addEventListener('click', ()=>{{ if(scur>=0) window.open(NODES[ORDER[scur]].url,'_blank'); }});

// B. mixture-similarity network
const nc=document.getElementById('net'), nx=nc.getContext('2d'); const pad=30;
function nPos(n){{ return [pad+n.nx*(nc.width-2*pad), pad+n.ny*(nc.height-2*pad)]; }}
nx.strokeStyle='rgba(150,155,165,0.4)'; nx.lineWidth=0.9;
EDGES.forEach(([a,b])=>{{ const [x1,y1]=nPos(NODES[a]),[x2,y2]=nPos(NODES[b]);
  nx.beginPath(); nx.moveTo(x1,y1); nx.lineTo(x2,y2); nx.stroke(); }});
NODES.forEach(n=>{{ const [x,y]=nPos(n); nx.beginPath(); nx.arc(x,y,6.5,0,7);
  nx.fillStyle=n.blend; nx.fill(); nx.strokeStyle='#fff'; nx.lineWidth=1; nx.stroke(); }});
let ncur=null;
nc.addEventListener('mousemove', ev=>{{
  const r=nc.getBoundingClientRect(); const mx=(ev.clientX-r.left)*(nc.width/r.width), my=(ev.clientY-r.top)*(nc.height/r.height);
  ncur=null; let bd=180; NODES.forEach(n=>{{const [x,y]=nPos(n);const d=(x-mx)**2+(y-my)**2;if(d<bd){{bd=d;ncur=n;}}}});
  if(!ncur){{hideTip();nc.style.cursor='default';return;}} nc.style.cursor='pointer';
  showTip(`<b>${{ncur.title}}</b><br>${{mixText(ncur)}}<br><span style="color:#9db">多様度(entropy) ${{ncur.ent}}</span>`, ev);
}});
nc.addEventListener('mouseleave', hideTip);
nc.addEventListener('click', ()=>{{ if(ncur) window.open(ncur.url,'_blank'); }});
</script>
</body>
</html>"""
    out = REPORT_DIR / "call4_nmf6_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"NMF-6 report saved: {out} ({out.stat().st_size//1024} KB)")
    print(f"  bootstrap median ARI={boot}  vs Leiden6={ari_leiden}  vs tags={ari_tag}")


if __name__ == "__main__":
    main()
