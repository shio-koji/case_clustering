#!/usr/bin/env python3
"""plan_final_report — 団体向けの報告用HTML（plan05のソフトタグ結果を再構成）。

伝えたいこと:
  1. 既存タグ × ケースの「二部グラフ」という見せ方が良い（出発点）。
  2. 「重み付けがあった方が良い」との要望に応え、並列だったタグに重みを付けた
     （＝各ケースのタグ割合）。
  3. 重みを含む／含まない二部グラフを複数用意した。
補足Aで plan02（機械学習で新分類を作る試み）の主要結果 K=6 も紹介する。
入力: plan05/results/soft_tags.json（ソフトタグ）, plan02/results/membership_nmf.json（K=6）。
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
P2 = os.path.join(ROOT, "plan02", "results")
FEAT2 = os.path.join(ROOT, "plan02", "features")
REP = os.path.join(ROOT, "plan_final_report")
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
tag_count = {tg: sum(1 for c in case_ids if tg in ratios[c]) for tg in tags}
tag_active = {tg: sum(1 for c in case_ids if tg in ratios[c] and status_of[c] == "active")
              for tg in tags}
tags_by_count = sorted(tags, key=lambda t: -tag_count[t])

# ---- plan02 K=6 NMF (supplement A) ---------------------------------------
nmf = json.load(open(os.path.join(P2, "membership_nmf.json")))
nmf_ratios = np.array(nmf["ratios"])                 # 95 x 6
nmf_dom = np.array(nmf["dominant_topic"])
TOPIC_COL = ["#4C8C2B", "#2B6CB0", "#B0532B", "#8A4FA8", "#C29B2C", "#2BA8A0"]
TOPIC_NAME = ["地域・環境の行政訴訟", "情報公開", "入管収容・難民",
              "婚姻の平等", "選挙権・政治参加", "刑事手続"]


def topic_words(k, n=5):
    ws = nmf["topic_words"][str(k)]
    out = [w[0] if isinstance(w, (list, tuple)) else w for w in ws][:n]
    return "・".join(out)


# ---- interactive per-case ratio table + stacked bar (JSON for JS) ---------
def ratio_rows_json():
    rows = []
    for c in case_ids:
        parts = sorted(ratios[c].items(), key=lambda x: -x[1])
        rows.append({"id": c, "title": tit[c], "st": status_of[c],
                     "tags": [{"t": k, "v": round(v, 3), "c": TCOL[k]} for k, v in parts]})
    return json.dumps(rows, ensure_ascii=False)


# ---- supplement A figures (plan02 K=6) -----------------------------------
def fig_nmf_stack():
    order = sorted(range(len(case_ids)),
                   key=lambda i: (nmf_dom[i], -nmf_ratios[i, nmf_dom[i]]))
    R = nmf_ratios[order]
    n = len(order)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(n); bottom = np.zeros(n)
    for k in range(6):
        ax.bar(x, R[:, k], bottom=bottom, width=1.0, color=TOPIC_COL[k],
               label=f"T{k} {TOPIC_NAME[k]}")
        bottom += R[:, k]
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_ylabel("トピック混合比", fontproperties=jp(9))
    ax.set_xlabel("95ケース（優勢トピック順）", fontproperties=jp(9))
    ax.set_title("plan02：機械学習で見つけた6トピックの混合メンバーシップ（K=6）",
                 fontproperties=jp(11))
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return svg_of(fig)


def fig_nmf_network():
    E = np.load(os.path.join(FEAT2, "emb.npz"))["matrix"].astype(np.float64)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    k = 6
    edges = set()
    for i in range(len(E)):
        for j in np.argsort(-S[i])[:k]:
            edges.add((min(i, int(j)), max(i, int(j))))
    edges = sorted(edges)
    g = ig.Graph(n=len(E), edges=edges)
    import random as _r
    _r.seed(42)
    try:
        ig.set_random_number_generator(_r)
    except Exception:
        pass
    P = np.array(g.layout_fruchterman_reingold(niter=1200).coords)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for (i, j) in edges:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="#d3d7de", lw=0.4,
                alpha=0.6, zorder=1)
    for kk in range(6):
        m = nmf_dom == kk
        ax.scatter(P[m, 0], P[m, 1], s=42, c=TOPIC_COL[kk], edgecolors="white",
                   linewidths=0.5, zorder=2, label=f"T{kk} {TOPIC_NAME[kk]}")
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=8, title="優勢トピック")
    ax.set_title("plan02：ケースの意味的な近さネットワーク（点の色＝6トピック）",
                 fontproperties=jp(11))
    fig.tight_layout()
    return svg_of(fig)


# ---- bipartite case<->tag graph (interactive) -----------------------------
def _layouts_for(subset):
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
    kk_len = [max(0.12, 1.0 - 0.85 * w) for w in wts]
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
    full = _layouts_for(case_ids)
    active_ids = [c for c in case_ids if status_of[c] == "active"]
    active = _layouts_for(active_ids)
    LAYS = list(full.keys())
    tag_nodes = []
    for j in range(T):
        tag_nodes.append({
            "name": tags[j], "count": tag_count[tags[j]], "color": TCOL[tags[j]],
            "pos": {ly: full[ly]["tags"][j] for ly in LAYS},
            "posA": {ly: active[ly]["tags"][j] for ly in LAYS}})
    case_nodes = []
    for cid in case_ids:
        parts = sorted(ratios[cid].items(), key=lambda x: -x[1])
        node = {
            "id": cid, "title": tit[cid], "st": status_of[cid], "ndeg": len(parts),
            "tags": [{"t": k, "v": round(v, 3), "c": TCOL[k]} for k, v in parts],
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


bip_tags_json, bip_cases_json, bip_edges_json = build_bipartite_json()
taglist_json = json.dumps(tags, ensure_ascii=False)

# ---- static HTML fragments ------------------------------------------------
tag_table = "\n".join(
    f'<tr><td><span class="sw" style="background:{TCOL[t]}"></span>{t}</td>'
    f'<td>{tag_count[t]}</td><td>{tag_active[t]}</td></tr>' for t in tags_by_count)

HTML = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CALL4 ケースマップ 報告</title>
<style>
:root{{--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--accent:#2B6CB0;}}
*{{box-sizing:border-box;}}
body{{font-family:-apple-system,"Hiragino Sans","Yu Gothic",sans-serif;color:var(--fg);
max-width:1080px;margin:0 auto;padding:28px 20px 80px;line-height:1.8;}}
h1{{font-size:1.6rem;border-bottom:3px solid var(--accent);padding-bottom:10px;}}
h2{{font-size:1.25rem;margin-top:2.3em;border-left:5px solid var(--accent);padding-left:10px;}}
h4{{margin:.6em 0 .2em;}}
.lead{{background:#f5f8fc;border:1px solid var(--line);border-radius:8px;padding:16px 20px;}}
figure{{margin:1.2em 0;text-align:center;}} svg{{max-width:100%;height:auto;}}
figcaption{{color:var(--muted);font-size:.84em;margin-top:6px;}}
.gloss{{background:#f7f9fb;border-left:3px solid #b9c6d6;padding:6px 12px;margin:.5em 0;
color:#3a4a5a;font-size:.9em;border-radius:0 4px 4px 0;}}
.gloss b{{color:#243b53;}}
.note{{background:#fbfbf7;border:1px solid #e6e0c8;border-radius:8px;padding:14px 18px;font-size:.92em;}}
code{{background:#f2f2f2;padding:1px 5px;border-radius:3px;}}
.tables{{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start;}}
table{{border-collapse:collapse;font-size:.9em;}}
th,td{{border:1px solid var(--line);padding:5px 10px;text-align:left;vertical-align:top;}}
th{{background:#f2f2f2;}}
.summary td:last-child{{text-align:right;font-weight:700;color:var(--accent);}}
.tagtbl td:nth-child(n+2){{text-align:right;}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:middle;}}
#search{{padding:7px 11px;width:260px;border:1px solid var(--line);border-radius:6px;font-size:14px;}}
#rt{{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px;}}
#rt table{{width:100%;margin:0;}} #rt th{{position:sticky;top:0;}}
.rbar{{display:flex;height:16px;border-radius:3px;overflow:hidden;min-width:180px;}}
.rseg{{color:#fff;font-size:10px;text-align:center;line-height:16px;white-space:nowrap;overflow:hidden;}}
.rlbl{{font-size:.82em;color:#444;}}
.canvwrap{{border:1px solid var(--line);border-radius:8px;padding:8px;background:#fbfcfe;}}
canvas{{width:100%;height:auto;display:block;cursor:default;}}
#bip-legend,#stk-legend{{display:flex;flex-wrap:wrap;gap:6px 12px;margin:8px 4px 2px;}}
#bip-legend span,#stk-legend span{{font-size:11.5px;}}
#bip-btns{{display:flex;flex-wrap:wrap;gap:6px;margin:2px 4px 8px;}}
.lbtn{{font-size:12px;padding:4px 12px;border:1px solid #c3ccd6;background:#fff;
border-radius:15px;cursor:pointer;color:#334;}}
.lbtn.on{{background:var(--accent);color:#fff;border-color:var(--accent);}}
.af{{font-size:13px;color:#334;margin-left:2px;cursor:pointer;user-select:none;}}
.af input{{vertical-align:middle;margin-right:4px;}}
#tip{{position:fixed;display:none;background:#20293a;color:#fff;padding:9px 12px;
border-radius:6px;font-size:12px;max-width:300px;box-shadow:0 3px 12px rgba(0,0,0,.28);
z-index:50;line-height:1.5;pointer-events:none;}}
#tip .trow{{display:flex;align-items:center;gap:6px;margin:2px 0;}}
#tip .tsw{{display:inline-block;width:10px;height:10px;border-radius:2px;flex:none;}}
#tip .tbar{{flex:1;height:8px;background:#46536b;border-radius:4px;overflow:hidden;min-width:70px;}}
#tip .tbf{{display:block;height:100%;border-radius:4px;min-width:2px;}}
.hint{{color:var(--muted);font-size:.85em;margin:.3em 0 .6em;}}
</style></head><body>

<h1>CALL4 ケースマップ：既存タグ × 二部グラフによる可視化</h1>

<div class="lead">
<b>このレポートの要点</b><br>
90件強のケースをどう見せるかを色々試した結果、<b>「既存のタグ」と「ケース」を線で結んだ
ネットワーク図（＝二部グラフ）</b>が、シンプルで探索しやすい見せ方だという手応えを得ました。
さらに「タグに<b>重み（濃淡）</b>があった方が良い」という意見を受けて、
これまで<b>並列（対等）だった各ケースのタグに重みを付け</b>、
「そのケースがどのタグの話をどれくらいしているか」を割合で表しました。
本レポートでは、<b>その重みと、重みを反映した／しない二部グラフ</b>をまとめています。
</div>

<h4>データの概要</h4>
<div class="tables">
<table class="summary"><tbody>
<tr><td>総ケース数</td><td>{len(case_ids)}</td></tr>
<tr><td>既存タグの種類</td><td>{T}</td></tr>
<tr><td>アクティブなケース</td><td>{n_active}</td></tr>
</tbody></table>
<table class="tagtbl"><thead><tr><th>既存タグ</th><th>件数</th><th>うちアクティブ</th></tr></thead>
<tbody>
{tag_table}
</tbody></table>
</div>

<h2>1. やったこと（かんたんな説明）</h2>
<p>既存タグは本来「付いている／いない」の2択で、<b>どのタグがそのケースの主題かは区別できません</b>。
そこで各ケースの文章（ケース名＋概要＋本文）を手がかりに、
<b>そのケースの内容が各タグにどれくらい近いか</b>を測り、付いているタグに合計100%の<b>割合（重み）</b>を配りました。
新しい分類を機械に作らせるのではなく、<b>すでに人が付けた既存タグを土台に、重みだけ</b>を推定しています。</p>
<div class="gloss"><b>埋め込み（embedding）</b>とは、<b>文章をベクトル（数値の並び）に変換</b>する技術です。
変換したあとは、<b>ベクトル同士の近さ（向きの近さ＝コサイン類似度）を計算</b>することで、
元の文章どうしがどれくらい意味的に近いかを数値で測れます。</div>
<p><b>今回の測り方：</b>
<b>ケース側</b>は、そのケースの文章（<b>ケース名＋概要＋本文</b>）をベクトルに変換しました。
<b>タグ側</b>は、タグ名そのものではなく、<b>そのタグが実際に付いている複数ケースの文章ベクトルを平均</b>して、
「<b>そのタグらしい文章</b>」の典型像をベクトルにしました。
そのうえで<b>ケースのベクトルとタグのベクトルの近さ</b>を測り、各ケースについて、
付いているタグの間で合計100%になるように配分したものが「<b>タグ割合</b>」です。<br>
なお、タグ側の平均にそのケース自身が入っていると「<b>自分で自分との近さを測る</b>」ことになり、点が甘くなってしまいます。
これを避けるため、<b>Leave-One-Out（1つ抜き）という手法</b>を使い、
あるケースを採点するときは、<b>そのケース自身を「そのタグらしい文章」の平均から除いて</b>計算しています。
<br>（※<b>法律の専門知識は使わず、文章の意味だけを頼りに機械的に</b>計算しています。細かい計算式は省略します。）</p>

<h2>2. ケースごとのタグ割合</h2>
<p>各ケースの中で、付いているタグがどれくらいの割合を占めるか（合計100%）を色分けで示します。
たとえば「同性パートナーにも犯罪被害の遺族給付金を」訴訟は
<b>ジェンダー・セクシュアリティ 89%／刑事司法 11%</b> のように、主題と副次的な論点が数値で分かれます。<br>
<span style="color:#666;font-size:.92em">※ もともと<b>タグが1つしか付いていないケースは、比べる相手がいないので機械的に100%</b>になります
（重みが意味を持つのは、複数タグが付いたケースです）。</span></p>
<p class="hint">🖱 各棒が1ケース。<b>棒にカーソルを合わせるとケース名と内訳</b>が出て、<b>クリックでCALL4のページ</b>へ。</p>
<div class="canvwrap">
  <label class="af"><input type="checkbox" id="stk-active"> アクティブなケースのみ（{n_active}件）</label>
  <canvas id="rstack" width="1040" height="300"></canvas>
  <div id="stk-legend"></div>
</div>

<h4>一覧（検索できます）</h4>
<p><input id="search" placeholder="ケース名・タグで絞り込み…">
<label class="af"><input type="checkbox" id="tbl-active"> アクティブなケースのみ（{n_active}件）</label></p>
<div id="rt"><table><thead><tr><th>ケース</th><th>タグ割合</th></tr></thead>
<tbody id="rtb"></tbody></table></div>

<h2>3. 二部グラフ（本命の見せ方）</h2>
<p><b>二部グラフ</b>は、<b>「ケース」と「タグ」という2種類の点を、関係がある所だけ線で結んだ</b>ネットワーク図です。
どのケースがどのタグにつながるか、複数タグをまたぐケースはどれか、が一目で分かります。</p>
<div class="gloss"><b>そもそもネットワーク図（グラフ）とは？</b>
「もの（＝ノード／丸）」と「そのつながり（＝エッジ／線）」だけで関係を表す図です。
路線図・人間関係図・組織図などが身近な例で、<b>「何と何がつながっているか」を見たい時</b>に使われます。
<br><b>大事な補足：ノード（丸）を置く位置や、エッジ（線）の長さ・太さ・曲がり方は、実は「決まった正解」がありません。</b>
つながり方（誰と誰が結ばれているか）さえ保たれていれば、<b>見やすさ・伝えたさに合わせて自由に配置・デフォルメしてよい</b>ものです
（本レポートで複数の「並べ方」を切り替えられるのも、そのためです）。
なので、物理的な展示物として作る際も、<b>位置や線の見た目は自由にデザインして構いません</b>——
つながりの情報さえ守れば、それは正しいネットワーク図です。</div>
<p class="hint">🖱 <b>ケース（丸）にカーソルを合わせると、そのケースのタグ割合</b>が出ます。
<b>クリックで CALL4 の該当ページ</b>へ。タグ（色ラベル）に合わせると件数が出ます。
丸の大きさ＝そのケースが持つタグ数、線の色と太さ＝タグと割合。</p>
<p>上のボタンで<b>並べ方（レイアウト）</b>を切り替えられます。
「<b>（割合）</b>」が付くものは<b>割合を距離に反映</b>（割合が高いタグほど近くに配置）、
「<b>等重み</b>」は割合を使わず<b>つながりの形だけ</b>で配置したものです。
「アクティブなケースのみ」にチェックを入れると、<b>アクティブなケースだけで並べ直します</b>。</p>
<div class="canvwrap">
  <div id="bip-btns"></div>
  <label class="af" style="margin-bottom:8px"><input type="checkbox" id="bip-active">
    アクティブなケースのみ表示（{n_active}件）</label>
  <canvas id="bip" width="1040" height="740"></canvas>
  <div id="bip-legend"></div>
</div>

<h2>補足A：なぜ「既存タグ」を使ったのか（機械学習で分類する試みの結果）</h2>
<div class="note">
当初は<b>ケースの文章から新しい分類（タグ）を自動生成する</b>方法にかなり取り組みました。
各ケースの単語の使われ方を数値化し（<b>TF-IDF</b>）、それを<b>4〜24グループ程度</b>に分ける手法（<b>NMF</b>）です。
下図はその代表例で、<b>6グループ（K=6）</b>に分けたときの結果——各ケースが6トピックにどれくらい
またがるか（混合メンバーシップ）と、ケースどうしの意味的な近さのネットワークです。
6トピック自体はそれなりに意味が通るものになりました
（地域・環境／情報公開／入管・難民／婚姻の平等／選挙権／刑事手続）。<br>
ただ、<b>グループ数をいくつにしても「意味的にきれいに割り切れた」とまでは言えず</b>、
また<b>法律の専門家ではない立場では、その分類が妥当かどうかを判断しきれなかった</b>ことから、
自動生成の路線はいったん見送り、<b>すでに人手で丁寧に付けられた既存タグ</b>を土台にする方針に落ち着きました。
<div class="gloss" style="margin-top:8px"><b>TF-IDF</b>：文書を特徴づける単語を「珍しさ」で重み付けする古典的手法。
<b>NMF</b>：多数の特徴を少数のグループ（トピック）に分解する手法。ここでは「文章から自動でタグ群を作る」用途で使用。
<b>混合メンバーシップ</b>：1ケースを1グループに割り当てず、複数トピックへの“割合”で表す考え方（今回の重み付けと同じ発想）。</div>
</div>
<figure>{fig_nmf_stack()}<figcaption>1本の棒が1ケース。色は6トピック。多くのケースが1〜2トピックに集中しつつ、
またがるケースもあることが分かる（plan02の主要結果）。</figcaption></figure>
<figure>{fig_nmf_network()}<figcaption>ケースを「文章の意味の近さ」でつないだネットワーク。点の色は上の6トピック。
近い色がまとまって島を作っている（plan02）。</figcaption></figure>

<h2>補足B：技術メモ（再現のための情報）</h2>
<p style="font-size:.9em">
対象：CALL4掲載ケース {len(case_ids)}件（ケース名＋概要＋本文）。既存タグ {T}種。乱数シード=42固定。<br>
文章の埋め込み：多言語モデル <code>bge-m3</code>（1ケース=1024次元のベクトル、長さを1に正規化）。<br>
タグ割合：各タグについて「そのタグを持つケースの平均像」と「持たないケースの平均像」への近さの差を取り、
付いているタグの間で合計100%になるよう配分（自分自身は平均像から除外して計算）。<br>
二部グラフの配置：力学レイアウト（Fruchterman–Reingold / Kamada–Kawai）と、
タグを円周に置きケースを割合の重心に置く放射状レイアウト。「割合」版は線の割合を距離に反映。<br>
補足Aの図：plan02 の NMF（K=6）混合メンバーシップと、埋め込みの近傍ネットワーク。<br>
※本可視化は探索・整理のためのもので、当事者を類型化・評価する意図はありません。
</p>

<div id="tip"></div>
<script>
const ROWS = {ratio_rows_json()};
const TAGLIST = {taglist_json};

// ===== searchable table =====
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

const tip = document.getElementById('tip');
function tipShow(html, ev) {{
  tip.innerHTML = html; tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 310) + 'px';
  tip.style.top = Math.min(ev.clientY + 12, window.innerHeight - 160) + 'px';
}}
function caseTipHTML(c) {{
  const bars = c.tags.map(t =>
    `<div class="trow"><span class="tsw" style="background:${{t.c}}"></span>`
    + `<span style="flex:none;min-width:118px">${{t.t}}</span>`
    + `<span class="tbar"><span class="tbf" style="width:${{(t.v*100).toFixed(0)}}%;background:${{t.c}}"></span></span>`
    + `<span style="flex:none">${{(t.v*100).toFixed(0)}}%</span></div>`).join('');
  return `<b>${{c.title}}</b>${{bars}}<div style="margin-top:4px;color:#9ecbff">クリックで開く</div>`;
}}
function openCase(id) {{ window.open('https://www.call4.jp/info.php?type=items&id=' + id, '_blank'); }}

// ===== interactive stacked bar (per-case tag share) =====
(function() {{
  const cv = document.getElementById('rstack'), cx = cv.getContext('2d');
  const chk = document.getElementById('stk-active');
  const PAD = 8;
  let view = [];
  function order() {{
    view = ROWS.filter(r => !chk.checked || r.st === 'active').slice();
    view.sort((a, b) => {{
      const da = TAGLIST.indexOf(a.tags[0].t), db = TAGLIST.indexOf(b.tags[0].t);
      return da - db || b.tags[0].v - a.tags[0].v;
    }});
  }}
  function bw() {{ return (cv.width - 2 * PAD) / Math.max(view.length, 1); }}
  function draw() {{
    order();
    cx.clearRect(0, 0, cv.width, cv.height);
    const w = bw(), H = cv.height - 24;
    view.forEach((r, i) => {{
      let y = 12;
      const x = PAD + i * w;
      r.tags.forEach(t => {{
        const h = t.v * H;
        cx.fillStyle = t.c; cx.fillRect(x, y, Math.max(w - 0.5, 1), h);
        y += h;
      }});
    }});
    cx.fillStyle = '#555'; cx.font = '11px sans-serif';
    cx.fillText('← 各棒 = 1ケース（優勢タグ順）｜ ' + view.length + '件', PAD, cv.height - 4);
  }}
  draw();
  chk.addEventListener('change', draw);
  function idxAt(ev) {{
    const r = cv.getBoundingClientRect();
    const mx = (ev.clientX - r.left) * (cv.width / r.width);
    const i = Math.floor((mx - PAD) / bw());
    return (i >= 0 && i < view.length) ? i : -1;
  }}
  cv.addEventListener('mousemove', ev => {{
    const i = idxAt(ev);
    if (i < 0) {{ tip.style.display = 'none'; cv.style.cursor = 'default'; return; }}
    cv.style.cursor = 'pointer'; tipShow(caseTipHTML(view[i]), ev);
  }});
  cv.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
  cv.addEventListener('click', ev => {{ const i = idxAt(ev); if (i >= 0) openCase(view[i].id); }});
  // legend
  const lg = document.getElementById('stk-legend');
  TAGLIST.forEach(t => {{
    const c = ROWS.flatMap(r => r.tags).find(x => x.t === t);
    if (c) lg.insertAdjacentHTML('beforeend',
      `<span><span class="tsw" style="display:inline-block;border-radius:2px;background:${{c.c}};margin-right:4px"></span>${{t}}</span>`);
  }});
}})();

// ===== interactive bipartite tag<->case graph =====
const BTAGS = {bip_tags_json};
const BCASES = {bip_cases_json};
const BEDGES = {bip_edges_json};
const BLAYOUTS = [["radial", "放射状（割合）"],
                  ["kamada_r", "Kamada–Kawai（割合）"],
                  ["force_r", "力学（割合）"],
                  ["kamada", "Kamada–Kawai（等重み）"],
                  ["force", "力学（等重み）"],
                  ["columns", "二列"]];
(function() {{
  const cv = document.getElementById('bip'), cx = cv.getContext('2d');
  const PAD = 96;
  let LAY = "radial", sx = 1, sy = 1, ox = 0, oy = 0;
  const actChk = document.getElementById('bip-active');
  const shown = c => !actChk.checked || c.st === 'active';
  const co = n => (actChk.checked ? (n.posA || n.pos) : n.pos)[LAY];
  function fit() {{
    const vis = BTAGS.concat(BCASES.filter(shown));
    const X = vis.map(n => co(n)[0]), Y = vis.map(n => co(n)[1]);
    const xmin = Math.min(...X), xmax = Math.max(...X);
    const ymin = Math.min(...Y), ymax = Math.max(...Y);
    const s = Math.min((cv.width - 2 * PAD) / (xmax - xmin || 1),
                       (cv.height - 2 * PAD) / (ymax - ymin || 1));
    sx = sy = s; ox = (cv.width - s * (xmax + xmin)) / 2; oy = (cv.height - s * (ymax + ymin)) / 2;
  }}
  const PX = n => ox + sx * co(n)[0];
  const PY = n => oy + sy * co(n)[1];
  const caseR = c => 4 + (c.ndeg - 1) * 2.4;
  const LABEL_FONT = 'bold 12.5px -apple-system,"Hiragino Sans","Yu Gothic",sans-serif';
  const _dim = new Map();
  function tagDims(t) {{
    if (_dim.has(t.name)) return _dim.get(t.name);
    cx.font = LABEL_FONT;
    const w = cx.measureText(t.name).width + 20;
    const h = 20 + Math.min(6, Math.sqrt(t.count));
    const o = {{w, h}}; _dim.set(t.name, o); return o;
  }}
  function rr(x, y, w, h, r) {{
    cx.beginPath(); cx.moveTo(x + r, y);
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
    BEDGES.forEach(([ti, ci, v]) => {{
      const t = BTAGS[ti], c = BCASES[ci];
      if (!shown(c)) return;
      cx.beginPath(); cx.moveTo(PX(t), PY(t)); cx.lineTo(PX(c), PY(c));
      cx.strokeStyle = t.color; cx.globalAlpha = 0.25 + 0.6 * v;
      cx.lineWidth = 0.5 + 3.5 * v; cx.stroke();
    }});
    cx.globalAlpha = 1;
    BCASES.forEach(c => {{
      if (!shown(c)) return;
      cx.beginPath(); cx.arc(PX(c), PY(c), caseR(c), 0, Math.PI * 2);
      cx.fillStyle = '#8593a6'; cx.globalAlpha = 0.88; cx.fill();
      cx.globalAlpha = 1; cx.strokeStyle = '#fff'; cx.lineWidth = 1; cx.stroke();
    }});
    BTAGS.forEach(t => {{
      const {{w, h}} = tagDims(t), x = PX(t), y = PY(t);
      const rx = x - w / 2, ry = y - h / 2;
      cx.save();
      cx.shadowColor = 'rgba(20,30,50,.25)'; cx.shadowBlur = 6; cx.shadowOffsetY = 1.5;
      cx.fillStyle = t.color; rr(rx, ry, w, h, h / 2); cx.fill();
      cx.restore();
      cx.strokeStyle = 'rgba(255,255,255,.92)'; cx.lineWidth = 1.5;
      rr(rx, ry, w, h, h / 2); cx.stroke();
      const dark = lum(t.color) > 150;
      cx.font = LABEL_FONT; cx.textAlign = 'center'; cx.textBaseline = 'middle';
      cx.lineJoin = 'round';
      cx.strokeStyle = dark ? 'rgba(255,255,255,.9)' : 'rgba(0,0,0,.38)';
      cx.lineWidth = 2.6; cx.strokeText(t.name, x, y);
      cx.fillStyle = dark ? '#17202e' : '#ffffff';
      cx.fillText(t.name, x, y);
    }});
  }}
  draw();
  const lg = document.getElementById('bip-legend');
  lg.insertAdjacentHTML('beforeend',
    `<span style="color:#666">■ 色ラベル = タグ（色は個別・大きさ=件数）／● = ケース（大きさ=タグ数・線の色/太さ=タグと割合）</span>`);
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
      const dd = Math.hypot(mx - PX(c), my - PY(c)), rr2 = caseR(c) + 3;
      if (dd <= rr2 && dd < bestD) {{ bestD = dd; best = c; }}
    }});
    return best ? {{kind: 'case', obj: best}} : null;
  }}
  cv.addEventListener('mousemove', ev => {{
    const h = hit(ev);
    if (!h) {{ tip.style.display = 'none'; cv.style.cursor = 'default'; return; }}
    if (h.kind === 'tag') {{
      tipShow(`<b>${{h.obj.name}}</b><br>${{h.obj.count}} 件のケースに付与`, ev); cv.style.cursor = 'default';
    }} else {{
      tipShow(caseTipHTML(h.obj), ev); cv.style.cursor = 'pointer';
    }}
  }});
  cv.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
  cv.addEventListener('click', ev => {{ const h = hit(ev); if (h && h.kind === 'case') openCase(h.obj.id); }});
}})();
</script>
</body></html>"""

out = os.path.join(REP, "call4_case_map_report.html")
open(out, "w", encoding="utf-8").write(HTML)
print(f"[final] wrote {out}  ({len(HTML)/1024:.0f} KB)")
