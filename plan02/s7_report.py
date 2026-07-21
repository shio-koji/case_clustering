#!/usr/bin/env python3
"""
plan02 Stage 6: Visualization + bilingual single-file HTML report.

Figures (inline SVG via matplotlib): tag<->topic Sankey, 3 dendrograms,
ARI heatmap, bootstrap stability, topic x case heatmap.
Interactive (vanilla canvas JS): NMF mixture stacked bar, common UMAP map
with color-by buttons. Fully self-contained HTML (no external assets).

Run from the repo root:  python plan02/s7_report.py
"""

import io
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches

SEED = 42
FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")
REPORT_DIR = Path("plan02/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --- Japanese font (same asset as plan01, verified working) ---
JP_FONT_PATH = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
                "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/"
                "YuGothic-Medium.otf")
fm.fontManager.addfont(JP_FONT_PATH)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

def jp(size=9):
    return fm.FontProperties(fname=JP_FONT_PATH, size=size)

TOPIC_COLORS = ["#4C8C2B", "#2B6CB0", "#B0532B", "#8A4FA8", "#C29B2C", "#2BA8A0"]
TOPIC_EN = ["Regional development & environment", "Freedom of information",
            "Immigration detention & refugees", "Marriage equality & gender",
            "Suffrage & political participation", "Criminal procedure"]
METHOD_LABELS = {"kmeans_svd": "k-means (lex)", "kmeans_pca": "k-means (sem)",
                 "hier_svd": "hier. (lex)", "hier_pca": "hier. (sem)",
                 "leiden_svd": "Leiden (lex)", "leiden_pca": "Leiden (sem)",
                 "bertopic": "BERTopic-like", "tags_null": "tags null",
                 "nmf_dom": "NMF dom.", "lda_dom": "LDA dom."}


def svg_of(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue().decode("utf-8")
    return s[s.find("<svg"):]


def load_everything():
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    data = {
        "ids": [r["id"] for r in tokens],
        "titles": [r["title"] for r in tokens],
        "tags": [r["subject_tags"] for r in tokens],
        "status": [r["case_status"] for r in tokens],
    }
    data["nmf"] = json.loads((RESULTS_DIR / "membership_nmf.json").read_text(encoding="utf-8"))
    data["names"] = json.loads((RESULTS_DIR / "names_llm.json").read_text(encoding="utf-8"))
    data["interp"] = json.loads((RESULTS_DIR / "interpretation.json").read_text(encoding="utf-8"))
    data["eval"] = json.loads((RESULTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    data["linkages"] = json.loads((RESULTS_DIR / "linkages.json").read_text(encoding="utf-8"))
    data["s2"] = json.loads((FEATURES_DIR / "s2_meta.json").read_text(encoding="utf-8"))
    for m, fname in [("leiden_pca", "labels_leiden_pca.json"),
                     ("kmeans_pca", "labels_kmeans_pca.json"),
                     ("hier_pca", "labels_hier_pca.json"),
                     ("bertopic", "labels_bertopic_pca.json")]:
        data[f"labels_{m}"] = json.loads(
            (RESULTS_DIR / fname).read_text(encoding="utf-8"))["labels"]
    return data


def umap_2d():
    cache = FEATURES_DIR / "umap2d.npz"
    if cache.exists():
        return np.load(cache)["coords"]
    import umap as umap_lib
    emb = np.load(FEATURES_DIR / "emb.npz")["matrix"]
    coords = umap_lib.UMAP(n_components=2, n_neighbors=10, min_dist=0.15,
                           metric="cosine", random_state=SEED).fit_transform(emb)
    np.savez_compressed(cache, coords=coords)
    return coords


# ---------------- static figures ----------------

def fig_sankey(data):
    """Tags (left) -> NMF dominant topics (right), ribbon width = case count."""
    dom = np.array(data["nmf"]["dominant_topic"])
    flows = {}
    tag_tot, top_tot = {}, {}
    for i, tags in enumerate(data["tags"]):
        for t in tags:
            flows[(t, int(dom[i]))] = flows.get((t, int(dom[i])), 0) + 1
            tag_tot[t] = tag_tot.get(t, 0) + 1
            top_tot[int(dom[i])] = top_tot.get(int(dom[i]), 0) + 1
    tags_sorted = sorted(tag_tot, key=lambda t: -tag_tot[t])
    topics_sorted = sorted(top_tot)

    gap, H = 2.0, 100.0
    def layout(names, totals):
        total = sum(totals[n] for n in names)
        scale = (H - gap * (len(names) - 1)) / total
        pos, y = {}, 0.0
        for n in names:
            h = totals[n] * scale
            pos[n] = (y, h)
            y += h + gap
        return pos, scale

    lpos, lscale = layout(tags_sorted, tag_tot)
    rpos, rscale = layout(topics_sorted, top_tot)

    fig, ax = plt.subplots(figsize=(11.5, 7))
    loff = {t: 0.0 for t in tags_sorted}
    roff = {k: 0.0 for k in topics_sorted}
    for (t, k), n in sorted(flows.items(), key=lambda x: (tags_sorted.index(x[0][0]), x[0][1])):
        y0, _ = lpos[t]; y1, _ = rpos[k]
        a0, a1 = y0 + loff[t], y1 + roff[k]
        h0, h1 = n * lscale, n * rscale
        loff[t] += h0; roff[k] += h1
        verts = [(0.14, a0), (0.5, a0), (0.5, a1), (0.86, a1),
                 (0.86, a1 + h1), (0.5, a1 + h1), (0.5, a0 + h0), (0.14, a0 + h0),
                 (0.14, a0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MplPath(verts, codes),
                                        facecolor=TOPIC_COLORS[k], alpha=0.42, lw=0))
    for t in tags_sorted:
        y, h = lpos[t]
        ax.add_patch(mpatches.Rectangle((0.10, y), 0.04, h, color="#555"))
        ax.text(0.09, y + h / 2, f"{t} ({tag_tot[t]})", ha="right", va="center",
                fontproperties=jp(9))
    names = data["names"]["nmf"]
    for k in topics_sorted:
        y, h = rpos[k]
        ax.add_patch(mpatches.Rectangle((0.86, y), 0.04, h, color=TOPIC_COLORS[k]))
        ax.text(0.91, y + h / 2, f"T{k} {names[str(k)]['name']} ({top_tot[k]})",
                ha="left", va="center", fontproperties=jp(9))
    ax.set_xlim(-0.28, 1.5); ax.set_ylim(-2, H + 2)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_title("既存タグ → NMFトピック（優勢トピック）の対応 / Existing tags → NMF dominant topics",
                 fontproperties=jp(11))
    return svg_of(fig)


def fig_dendrograms(data):
    from scipy.cluster.hierarchy import dendrogram
    out = {}
    titles = {"pca": "意味空間の階層 / semantic space", "svd": "語彙空間の階層 / lexical space",
              "tags_null": "既存タグのみ（null model）/ tags-only null model"}
    for key in ["pca", "svd", "tags_null"]:
        Z = np.array(data["linkages"]["linkages"][key])
        fig, ax = plt.subplots(figsize=(14, 3.4))
        dendrogram(Z, ax=ax, labels=[i[-3:] for i in data["ids"]], leaf_rotation=90,
                   leaf_font_size=5.5, color_threshold=Z[-4, 2])
        ax.set_title(titles[key] + "（cosine/Jaccard + average linkage, 赤破線=5クラスタ相当）",
                     fontproperties=jp(10))
        ax.axhline(Z[-4, 2], color="red", ls="--", lw=0.7)
        out[key] = svg_of(fig)
    return out


def fig_ari_heatmap(data):
    ev = data["eval"]["q2_agreement"]
    order = ["kmeans_svd", "hier_svd", "leiden_svd", "nmf_dom", "lda_dom",
             "kmeans_pca", "hier_pca", "leiden_pca", "bertopic", "tags_null"]
    n = len(order)
    M = np.array([[ev["ari"][a][b] for b in order] for a in order])
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(M, cmap="YlGnBu", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([METHOD_LABELS[m] for m in order], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([METHOD_LABELS[m] for m in order], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.6, color="white" if M[i, j] > 0.55 else "#222")
    ax.set_title("手法間一致度（ARI）/ Method agreement (ARI)", fontproperties=jp(11))
    fig.colorbar(im, shrink=0.8)
    return svg_of(fig)


def fig_bootstrap(data):
    boot = data["eval"]["q2_bootstrap_stability"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(boot)
    ax.boxplot([boot[m]["distribution"] for m in names], tick_labels=[METHOD_LABELS[m] for m in names],
               showfliers=False, widths=0.5)
    for i, m in enumerate(names):
        d = boot[m]["distribution"]
        ax.scatter(np.random.default_rng(0).normal(i + 1, 0.06, len(d)), d, s=4, alpha=0.3)
    ax.set_ylabel("ARI vs full-data labels")
    ax.set_title("ブートストラップ安定性（80%×100回）/ Bootstrap stability", fontproperties=jp(11))
    ax.grid(alpha=0.3, axis="y")
    return svg_of(fig)


def fig_topic_case_heatmap(data, order):
    R = np.array(data["nmf"]["ratios"])[order].T  # 6 x 95
    names = data["names"]["nmf"]
    fig, ax = plt.subplots(figsize=(14, 2.6))
    ax.imshow(R, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"T{k} {names[str(k)]['name']}" for k in range(6)],
                       fontproperties=jp(8))
    dom_sorted = np.array(data["nmf"]["dominant_topic"])[order]
    for b in np.where(np.diff(dom_sorted) != 0)[0]:
        ax.axvline(b + 0.5, color=("#c33"), lw=0.7)
    ax.set_xticks([])
    ax.set_xlabel("95 cases (sorted by dominant topic) / 95ケース（優勢トピック順）",
                  fontproperties=jp(9))
    ax.set_title("トピック×ケース 混合比ヒートマップ / Topic × case mixture heatmap",
                 fontproperties=jp(11))
    return svg_of(fig)


# ---------------- HTML assembly ----------------

def build_cases_json(data, coords):
    dom = data["nmf"]["dominant_topic"]
    ratios = data["nmf"]["ratios"]
    ent = data["nmf"]["entropy_normalized"]
    first_tags = [(t[0] if t else "no_tag") for t in data["tags"]]
    cases = []
    for i, cid in enumerate(data["ids"]):
        cases.append({
            "id": cid, "title": data["titles"][i],
            "url": f"https://www.call4.jp/info.php?type=items&id={cid}",
            "tags": data["tags"][i], "ftag": first_tags[i],
            "status": data["status"][i],
            "x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3),
            "nmf": [round(float(r), 3) for r in ratios[i]],
            "dom": int(dom[i]), "ent": round(float(ent[i]), 3),
            "leiden": int(data["labels_leiden_pca"][i]),
            "kmeans": int(data["labels_kmeans_pca"][i]),
            "hier": int(data["labels_hier_pca"][i]),
            "bertopic": int(data["labels_bertopic"][i]),
        })
    return cases


def topic_cards_html(data):
    names = data["names"]["nmf"]
    interp = data["interp"]["mixture"]["nmf"]["topics"]
    cards = []
    for k in range(6):
        t = interp[str(k)]
        reps = "".join(
            f'<li><a href="https://www.call4.jp/info.php?type=items&id={r["id"]}" '
            f'target="_blank">{r["title"]}</a> <span class="dim">({r["ratio"]})</span></li>'
            for r in t["representatives"])
        cards.append(f"""
<div class="topic-card" style="border-top: 4px solid {TOPIC_COLORS[k]}">
  <h4>T{k}: {names[str(k)]['name']}<br><span class="en">{TOPIC_EN[k]}</span></h4>
  <p class="dim">優勢 {t['size_dominant']}件 / dominant in {t['size_dominant']} cases</p>
  <p><b>特徴語 / terms:</b> {' / '.join(t['descriptor_words'][:8])}</p>
  <p><b>代表 / representatives:</b></p><ul>{reps}</ul>
</div>""")
    return "\n".join(cards)


def validity_table_html(data):
    q1 = data["eval"]["q1_internal_validity"]
    order = ["leiden_pca", "kmeans_pca", "hier_pca", "bertopic",
             "leiden_svd", "kmeans_svd", "hier_svd", "nmf_dom", "lda_dom"]
    align = data["eval"]["q3_tag_alignment"]
    boot = data["eval"]["q2_bootstrap_stability"]
    rows = []
    for m in order:
        r = q1[m]
        b = boot.get(m)
        rows.append(
            f"<tr><td>{METHOD_LABELS[m]}</td><td>{r['space']}</td><td>{r['n_clusters']}"
            f"{' (+'+str(r['n_outliers'])+' outl.)' if r['n_outliers'] else ''}</td>"
            f"<td>{r['silhouette']}</td><td>{r.get('one_sentence_pass','—')}</td>"
            f"<td>{align[m]['ari']:+.3f}</td>"
            f"<td>{b['ari_median'] if b else '—'}</td></tr>")
    return ("<table><tr><th>手法 / method</th><th>空間</th><th>k</th>"
            "<th>silhouette†</th><th>一文テスト / one-sentence</th>"
            "<th>ARI vs tags</th><th>安定性 (median ARI)</th></tr>"
            + "".join(rows) + "</table>"
            "<p class='dim'>† silhouetteは同一空間内でのみ比較可（lex空間とsem空間の値の直接比較は不可）"
            " / comparable only within the same space.</p>")


def merges_splits_html(data):
    mg = data["eval"]["q3_merges"]["nmf_dom"]
    sp = data["eval"]["q3_splits"]["nmf_dom"]
    names = data["names"]["nmf"]
    mrows = "".join(f"<tr><td>T{c} {names[str(c)]['name']}</td><td>{'、'.join(ts)}</td></tr>"
                    for c, ts in mg.items())
    srows = "".join(f"<tr><td>{t}</td><td>{v['n_cases']}件</td><td>{v['n_clusters_spread']}トピック</td>"
                    f"<td>{v['top_cluster_share']:.0%}</td></tr>" for t, v in sp.items())
    return f"""
<div class="two-col">
<div><h4>統合 / merges（複数タグが1トピックに融合）</h4>
<table><tr><th>トピック</th><th>融合しているタグ（各3件以上）</th></tr>{mrows}</table></div>
<div><h4>分裂 / splits（1タグが複数トピックに分散）</h4>
<table><tr><th>タグ</th><th>件数</th><th>分散先</th><th>最大集中率</th></tr>{srows}</table></div>
</div>"""


def boundary_html(data):
    b = data["eval"]["q3_boundary_cases"]
    items = "".join(f"<li><b>{x['title']}</b></li>" for x in b["in_both_lists"])
    return f"""<p>NMFの混合エントロピー上位 かつ 手法間で所属が最も揺れる、の両方に該当した5件。
いずれも複数論点を本質的に横断する訴訟で、単一タグでは表現しきれないことの実例。/
Cases flagged by BOTH high NMF mixture entropy and cross-method instability —
genuinely multi-issue lawsuits.</p><ul>{items}</ul>"""


def generate(data, coords):
    order = np.lexsort((-np.array(data["nmf"]["ratios"])[
        np.arange(95), data["nmf"]["dominant_topic"]],
        np.array(data["nmf"]["dominant_topic"])))
    sankey = fig_sankey(data)
    dendros = fig_dendrograms(data)
    ariheat = fig_ari_heatmap(data)
    bootfig = fig_bootstrap(data)
    tc_heat = fig_topic_case_heatmap(data, order)
    cases_json = json.dumps(build_cases_json(data, coords), ensure_ascii=False)
    order_json = json.dumps([int(i) for i in order])
    topic_names_json = json.dumps(
        [data["names"]["nmf"][str(k)]["name"] for k in range(6)], ensure_ascii=False)
    topic_colors_json = json.dumps(TOPIC_COLORS)
    gen = date.today().isoformat()
    s2 = data["s2"]

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALL4 ケース分類レポート v2 (plan02)</title>
<style>
body {{ font-family: "Hiragino Sans", "Yu Gothic", sans-serif; margin: 0; background: #f6f5f2; color: #222; line-height: 1.75; }}
.container {{ max-width: 1080px; margin: 0 auto; padding: 28px 20px 80px; }}
h1 {{ font-size: 1.5em; border-bottom: 3px solid #2B6CB0; padding-bottom: 8px; }}
h2 {{ font-size: 1.22em; margin-top: 2.2em; border-left: 5px solid #2B6CB0; padding-left: 10px; }}
h3 {{ font-size: 1.05em; margin-top: 1.6em; }}
h4 {{ margin: 0.4em 0; }}
.en {{ color: #667; font-size: 0.85em; font-weight: normal; }}
.dim {{ color: #778; font-size: 0.85em; }}
.summary {{ background: #eef3fa; border: 1px solid #c9d8ee; border-radius: 8px; padding: 14px 18px; }}
table {{ border-collapse: collapse; background: #fff; font-size: 0.86em; margin: 10px 0; }}
th, td {{ border: 1px solid #ccc; padding: 5px 9px; text-align: left; }}
th {{ background: #e8edf5; }}
.figure {{ background: #fff; border-radius: 8px; padding: 12px; margin: 14px 0; box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow-x: auto; }}
.topic-card {{ background: #fff; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
.topic-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 14px; }}
.two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }}
.btn {{ background: #2B6CB0; color: #fff; border: 0; border-radius: 4px; padding: 5px 12px; margin: 3px; cursor: pointer; font-size: 0.84em; font-family: inherit; }}
.btn.active {{ background: #14385f; font-weight: bold; }}
canvas {{ max-width: 100%; }}
#tip {{ position: fixed; display: none; background: rgba(20,25,35,.94); color: #fff; padding: 8px 11px; border-radius: 6px; font-size: 12px; max-width: 340px; pointer-events: none; z-index: 10; }}
a {{ color: #2B6CB0; }}
details {{ margin: 8px 0; }}
li {{ margin: 2px 0; }}
</style>
</head>
<body>
<div class="container">

<h1>CALL4 公共訴訟ケース 分類・可視化レポート v2<br>
<span class="en">CALL4 Public-Interest Litigation — Classification & Visualization Report v2 (plan02)</span></h1>
<p class="dim">生成 {gen} ｜ N=95 ｜ 埋め込み: BAAI/bge-m3（全文・8192tok）｜ 形態素: SudachiPy モードC ｜ seed=42</p>

<div class="summary">
<b>要旨 / Summary</b><br>
95件のケース本文を、語彙（TF-IDF→SVD 68d）と意味（bge-m3→PCA 48d）の2表現で数値化し、
排他クラスタリング5手法と混合メンバーシップ（NMF/LDA, K=6）を比較した。
<b>手法・表現をまたいで同じ6軸構造</b>（地域環境／情報公開／入管・難民／家族・ジェンダー／選挙・政治参加／刑事手続、＋労働）が再現し、
Leiden（意味空間）はブートストラップ中央値ARI 0.73と頑健。
既存タグとの対応はNMFで ARI 0.335（参照的一致であって焼き直しではない）。
タグ「公正な手続」は全トピックに分散しており（最大集中40%）、実質「その他」化している——タグ体系再考の最有力ポイント。<br>
<span class="en">Two text representations (lexical TF-IDF and semantic bge-m3), five hard-clustering
methods and mixture membership (NMF K=6) reproduce the same six-axis structure. The
mega-tag "fair procedure" disperses across all topics — the prime candidate for tag reform.</span>
</div>

<h2>1. データと手法 <span class="en">Data & methods</span></h2>
<p>入力はケース名＋概要＋本文（アップデート除外、NFKC正規化）。形態素解析（SudachiPy 長単位）
→ 名詞・動詞・形容詞の基本形、ストップワード22語（承認済み）。</p>
<table>
<tr><th>段階</th><th>内容</th></tr>
<tr><td>表現</td><td>①タグone-hot（null model）②TF-IDF 7,350語（1+2gram）③bge-m3 1024d（全文）</td></tr>
<tr><td>圧縮</td><td>TF-IDF→SVD <b>{s2['tfidf_svd']['dim']}d</b>（寄与率{s2['tfidf_svd']['cumulative_variance']:.0%}）、
埋め込み→PCA <b>{s2['emb_pca']['dim']}d</b>（{s2['emb_pca']['cumulative_variance']:.0%}）。
1024d生空間より圧縮後の方がシルエット・タグARIとも一貫して良いことを実験で確認</td></tr>
<tr><td>構造発見</td><td>k-means / 階層(cosine+average) / KNN+Leiden ×2空間、BERTopic相当、NMF・LDA（K=6凍結）</td></tr>
<tr><td>解釈</td><td>全手法共通: c-TF-IDF特徴語＋medoid代表。命名に既存タグは不使用（循環回避）</td></tr>
</table>

<h2>2. 頑健な6軸構造（NMF K=6） <span class="en">The six robust axes</span></h2>
<p>トピック数K=6は「誤差カーブに肘がない中、全トピックが一文で説明でき、刑事手続と入管収容が分離する最小のK」として採用（K=7から寄せ集めトピックが発生）。</p>
<div class="topic-grid">
{topic_cards_html(data)}
</div>

<h2>3. 混合メンバーシップ <span class="en">Mixture membership</span></h2>
<p>各ケースは6トピックへの<b>混合比</b>で表される（合計1）。棒にカーソルを載せると内訳が見える。
複数色が混ざるケース＝複数論点を横断する訴訟。/ Each case is a mixture over six topics; hover for details.</p>
<div class="figure">
  <div id="legend-stack" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px;"></div>
  <canvas id="stack" width="1020" height="330"></canvas>
</div>
<div class="figure">{tc_heat}</div>

<h2>4. ケースマップ <span class="en">Case map (UMAP)</span></h2>
<p>座標は意味埋め込み由来のUMAP 2D（<b>可視化専用</b>で距離に厳密な意味はない）。色は下のボタンで切替。点にカーソルで詳細、クリックでCALL4のケースページへ。</p>
<div class="figure">
  <div id="btns"></div>
  <div id="legend-map" style="display:flex;flex-wrap:wrap;gap:10px;margin:6px 0;"></div>
  <canvas id="map" width="1020" height="560"></canvas>
</div>

<h2>5. 手法比較 <span class="en">Method comparison</span></h2>
{validity_table_html(data)}
<div class="figure">{ariheat}</div>
<p>意味空間の3手法（hier↔Leiden ARI 0.84、k-means↔Leiden 0.69）が強いブロックを形成し、
語彙空間のLeiden・NMFとも0.4〜0.5で一致。<b>別々の情報源が同じ構造を支持している</b>ことが頑健性の根拠。
一方 LDA と 語彙×階層 は崩壊（巨大クラスタ＋断片）しており、その事実自体を手法知見として記録する。</p>
<div class="figure">{bootfig}</div>
<details><summary><b>デンドログラム（3系統）/ Dendrograms</b></summary>
<div class="figure">{dendros['pca']}</div>
<div class="figure">{dendros['svd']}</div>
<div class="figure">{dendros['tags_null']}</div>
</details>

<h2>6. 既存タグ体系との突き合わせ <span class="en">Tag critique</span></h2>
<p>既存タグは「正解」ではなく参照。データ駆動構造との ARI は最大でも0.335（NMF）で、
タグの焼き直しではない。下のSankeyで「タグ→トピック」の流れを見る。</p>
<div class="figure">{sankey}</div>
{merges_splits_html(data)}
<h3>境界ケース <span class="en">Boundary cases</span></h3>
{boundary_html(data)}
<h3>タグ体系への示唆 <span class="en">Implications</span></h3>
<ul>
<li><b>「公正な手続」（43件）は実質「その他」</b>。全トピックに分散し最大集中も40%。争点ベース（情報公開・刑事手続・行政の説明責任など）への分解が有力。</li>
<li><b>「労働・生活者の権利」軸が繰り返し自然発生</b>（教員働き方・児相・労災・遺族年金）。現行「働き方」より広い独立軸として立てる価値。</li>
<li><b>刑事手続と入管収容は近接しつつ別軸</b>。意味空間では「身体拘束」として融合する手法もあり、上位概念（身体の自由）＋下位2軸の階層構造が実態に合う。</li>
<li>境界5ケースが示すように、<b>単一タグの排他割当では表現しきれない訴訟が実在</b>する。主タグ＋副タグ（または混合比）の運用が実態に忠実。</li>
</ul>

<h2>7. 限界 <span class="en">Limitations</span></h2>
<ul>
<li>N=95は小さく、シルエットは全手法で低め（最大0.13）。「くっきりした島」ではなく「ゆるやかな地形」であり、本結果は<b>探索・仮説生成</b>のためのもの。</li>
<li>K=6は解釈性基準の選択であり、誤差基準では一意に決まらなかった（K=5〜7で議論の余地）。</li>
<li>入力はケースページ本文まで。訴状・判決文の全文は未使用（次期拡張）。</li>
<li>埋め込みはbge-m3の1モデルのみ。モデル選択への依存性は未検証。</li>
</ul>

<h2>8. 再現情報 <span class="en">Reproducibility</span></h2>
<table>
<tr><th>項目</th><th>値</th></tr>
<tr><td>パイプライン</td><td>plan02/s0〜s7（GitHub: shio-koji/case_clustering）</td></tr>
<tr><td>データ取得日</td><td>2026-07-18（CALL4 MCP API）</td></tr>
<tr><td>形態素</td><td>SudachiPy 0.6.11 / sudachidict_core 20260428 / モードC / 名詞・動詞・形容詞 / ストップワード22語</td></tr>
<tr><td>埋め込み</td><td>BAAI/bge-m3（1024d, max 8192 tokens, 正規化）</td></tr>
<tr><td>圧縮</td><td>SVD {s2['tfidf_svd']['dim']}d ・ PCA {s2['emb_pca']['dim']}d（寄与率80%基準）＋再L2正規化</td></tr>
<tr><td>クラスタリング</td><td>k-means(n_init=50) / cosine+average / Leiden(k=10近傍, res採用値は結果JSON) / UMAP5d+HDBSCAN / NMF・LDA K=6</td></tr>
<tr><td>乱数シード</td><td>42（全段）</td></tr>
<tr><td>ブートストラップ</td><td>80%サブサンプル×100回</td></tr>
</table>
<p class="dim">本分類は解析目的であり、当事者を類型化して評価する意図はありません。ケース本文の著作権はCALL4および執筆者に帰属します。/
This classification is for analytical purposes only.</p>

</div>
<div id="tip"></div>

<script>
const CASES = {cases_json};
const ORDER = {order_json};
const TNAMES = {topic_names_json};
const TCOLORS = {topic_colors_json};
const tip = document.getElementById('tip');

function showTip(html, ev) {{
  tip.innerHTML = html; tip.style.display = 'block';
  const x = Math.min(ev.clientX + 14, window.innerWidth - 360);
  tip.style.left = x + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
}}
function hideTip() {{ tip.style.display = 'none'; }}

// ---------- stacked mixture bar ----------
const sc = document.getElementById('stack'), sctx = sc.getContext('2d');
const SW = sc.width, SH = sc.height, BW = (SW - 20) / CASES.length;
function drawStack() {{
  sctx.clearRect(0, 0, SW, SH);
  ORDER.forEach((ci, pos) => {{
    const c = CASES[ci]; let y = SH - 18;
    c.nmf.forEach((r, t) => {{
      const h = r * (SH - 30);
      sctx.fillStyle = TCOLORS[t];
      sctx.fillRect(12 + pos * BW, y - h, Math.max(BW - 1, 1.5), h);
      y -= h;
    }});
  }});
  sctx.fillStyle = '#555'; sctx.font = '11px sans-serif';
  sctx.fillText('← 各棒 = 1ケース（優勢トピック順）', 12, SH - 4);
}}
const lg = document.getElementById('legend-stack');
TNAMES.forEach((n, t) => {{
  lg.insertAdjacentHTML('beforeend',
    `<span style="font-size:12px"><span style="display:inline-block;width:11px;height:11px;background:${{TCOLORS[t]}};margin-right:4px;border-radius:2px"></span>T${{t}} ${{n}}</span>`);
}});
sc.addEventListener('mousemove', ev => {{
  const r = sc.getBoundingClientRect();
  const pos = Math.floor((ev.clientX - r.left) * (sc.width / r.width) - 12) / BW | 0;
  if (pos < 0 || pos >= ORDER.length) {{ hideTip(); return; }}
  const c = CASES[ORDER[pos]];
  const mix = c.nmf.map((v, t) => v > 0.08 ? `T${{t}} ${{TNAMES[t]}}: ${{(v*100).toFixed(0)}}%` : null)
                   .filter(Boolean).join('<br>');
  showTip(`<b>${{c.title}}</b><br>${{mix}}<br><span style="color:#9db">entropy ${{c.ent}} ｜ tags: ${{c.tags.join('・')||'—'}}</span>`, ev);
}});
sc.addEventListener('mouseleave', hideTip);
drawStack();

// ---------- UMAP map ----------
const mc = document.getElementById('map'), mctx = mc.getContext('2d');
const PAL = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#17becf','#7f7f7f','#bcbd22','#e377c2','#8c564b'];
const MODES = {{
  dom:    {{label: 'NMF優勢トピック', get: c => c.dom, name: t => 'T'+t+' '+TNAMES[t], col: t => TCOLORS[t]}},
  leiden: {{label: 'Leiden (意味空間)', get: c => c.leiden, name: v => 'C'+v, col: v => PAL[v % PAL.length]}},
  kmeans: {{label: 'k-means (意味空間)', get: c => c.kmeans, name: v => 'C'+v, col: v => PAL[v % PAL.length]}},
  bertopic: {{label: 'BERTopic相当', get: c => c.bertopic, name: v => v < 0 ? '外れ値' : 'T'+v, col: v => v < 0 ? '#999' : PAL[v % PAL.length]}},
  ftag:   {{label: '既存タグ(第1)', get: c => c.ftag, name: v => v, col: (v, i) => PAL[i % PAL.length]}},
  status: {{label: 'ステータス', get: c => c.status, name: v => v, col: (v, i) => PAL[i % PAL.length]}},
}};
let mode = 'dom';
const xs = CASES.map(c => c.x), ys = CASES.map(c => c.y);
const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
const px = c => 30 + (c.x - xmin) / (xmax - xmin) * (mc.width - 60);
const py = c => 25 + (c.y - ymin) / (ymax - ymin) * (mc.height - 50);
function drawMap() {{
  mctx.clearRect(0, 0, mc.width, mc.height);
  const M = MODES[mode];
  const vals = [...new Set(CASES.map(M.get))].sort((a,b) => (a>b)-(a<b));
  CASES.forEach(c => {{
    const v = M.get(c), i = vals.indexOf(v);
    mctx.beginPath(); mctx.arc(px(c), py(c), 6, 0, Math.PI*2);
    mctx.fillStyle = M.col(v, i); mctx.globalAlpha = 0.85; mctx.fill();
    mctx.globalAlpha = 1; mctx.strokeStyle = '#fff'; mctx.lineWidth = 1; mctx.stroke();
  }});
  const lm = document.getElementById('legend-map'); lm.innerHTML = '';
  vals.forEach((v, i) => {{
    lm.insertAdjacentHTML('beforeend',
      `<span style="font-size:12px"><span style="display:inline-block;width:11px;height:11px;background:${{M.col(v,i)}};margin-right:4px;border-radius:6px"></span>${{M.name(v)}}</span>`);
  }});
}}
const btns = document.getElementById('btns');
Object.entries(MODES).forEach(([k, m], i) => {{
  btns.insertAdjacentHTML('beforeend',
    `<button class="btn ${{i===0?'active':''}}" data-k="${{k}}">${{m.label}}</button>`);
}});
btns.addEventListener('click', ev => {{
  if (ev.target.tagName !== 'BUTTON') return;
  mode = ev.target.dataset.k;
  btns.querySelectorAll('.btn').forEach(b => b.classList.toggle('active', b === ev.target));
  drawMap();
}});
function nearest(ev) {{
  const r = mc.getBoundingClientRect();
  const mx = (ev.clientX - r.left) * (mc.width / r.width);
  const my = (ev.clientY - r.top) * (mc.height / r.height);
  let best = null, bd = 144;
  CASES.forEach(c => {{
    const d = (px(c)-mx)**2 + (py(c)-my)**2;
    if (d < bd) {{ bd = d; best = c; }}
  }});
  return best;
}}
mc.addEventListener('mousemove', ev => {{
  const c = nearest(ev);
  if (!c) {{ hideTip(); mc.style.cursor = 'default'; return; }}
  mc.style.cursor = 'pointer';
  const mix = c.nmf.map((v, t) => v > 0.15 ? TNAMES[t] + ' ' + (v*100).toFixed(0) + '%' : null)
                   .filter(Boolean).join(' / ');
  showTip(`<b>${{c.title}}</b><br>NMF: ${{mix}}<br><span style="color:#9db">tags: ${{c.tags.join('・')||'—'}} ｜ ${{c.status}}</span>`, ev);
}});
mc.addEventListener('mouseleave', hideTip);
mc.addEventListener('click', ev => {{
  const c = nearest(ev);
  if (c) window.open(c.url, '_blank');
}});
drawMap();
</script>
</body>
</html>"""
    out = REPORT_DIR / "call4_plan02_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report saved: {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    data = load_everything()
    coords = umap_2d()
    generate(data, coords)
