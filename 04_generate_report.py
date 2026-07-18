#!/usr/bin/env python3
"""
Step 4: Generate self-contained Japanese HTML report with inline SVG/Canvas figures.
"""

import json
import pickle
import numpy as np
from pathlib import Path
import io
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

# Japanese font setup
JP_FONT_PATH = "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/YuGothic-Medium.otf"
jp_prop = fm.FontProperties(fname=JP_FONT_PATH)
matplotlib.rcParams["font.family"] = "sans-serif"
# Add to font manager
fm.fontManager.addfont(JP_FONT_PATH)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def jp_font():
    return fm.FontProperties(fname=JP_FONT_PATH, size=8)

def jp_font_s(size=8):
    return fm.FontProperties(fname=JP_FONT_PATH, size=size)


# Load data
results = json.loads((RESULTS_DIR / "analysis_results.json").read_text(encoding="utf-8"))
corpus = json.loads((CACHE_DIR / "corpus_clean.json").read_text(encoding="utf-8"))

ids = results["ids"]
titles = results["titles"]
statuses = results["statuses"]
subject_tags_list = results["subject_tags_list"]
case_clusters = results["case_clusters"]
best_k = results["best_k"]

id_to_idx = {cid: i for i, cid in enumerate(ids)}

KM_NAMES = {
    0: "ジェンダー・市民権・選挙権",
    1: "刑事司法・入管・個人の自由",
    2: "情報公開・司法の独立",
    3: "環境・地域・行政の責任",
}
HIER_NAMES = {
    0: "ジェンダー・セクシュアリティ",
    1: "入管・外国人の権利",
    2: "刑事司法・被疑者の権利",
    3: "情報公開・行政の透明性",
    4: "環境・災害・地域インフラ",
    5: "選挙・政治参加",
}
CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
STATUS_COLORS = {"active": "#2196F3", "archived": "#9E9E9E", "closed": "#F44336", "no_donation": "#FF9800"}
TAG_COLORS = {
    "ジェンダー・セクシュアリティ": "#E91E63",
    "公正な手続": "#9C27B0",
    "外国にルーツを持つ人々": "#673AB7",
    "政治参加・表現の自由": "#3F51B5",
    "刑事司法": "#2196F3",
    "情報公開": "#00BCD4",
    "環境・災害": "#4CAF50",
    "働き方": "#8BC34A",
    "医療・福祉・障がい": "#FF9800",
    "沖縄": "#FF5722",
    "個人情報・プライバシー": "#795548",
}


def svg_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    content = buf.read().decode("utf-8")
    start = content.find("<svg")
    return content[start:]


def build_dendrogram_svg():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    with open(CACHE_DIR / "e5_embeddings.pkl", "rb") as f:
        emb_e5 = pickle.load(f)["embeddings"]
    cos_dist = pdist(emb_e5, metric="cosine")
    Z = linkage(cos_dist, method="ward")

    fig, ax = plt.subplots(figsize=(16, 4.5))
    # Use short ID labels (not Japanese titles) to avoid font issues in axis
    short_labels = [f"{ids[i]}" for i in range(len(ids))]
    dendrogram(
        Z, ax=ax,
        labels=short_labels,
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=Z[-6, 2],
    )
    ax.set_title("階層クラスタリング デンドログラム（Ward法、コサイン距離、multilingual-E5埋め込み）",
                 fontproperties=jp_font_s(10))
    ax.axhline(y=Z[-6, 2], color="red", linestyle="--", alpha=0.6, label="k=6 カット")
    ax.axhline(y=Z[-4, 2], color="orange", linestyle="--", alpha=0.6, label="k=4 カット")
    ax.legend(prop=jp_font_s(8))
    ax.set_ylabel("距離", fontproperties=jp_font_s(9))
    ax.set_xlabel("ケースID", fontproperties=jp_font_s(9))
    fig.tight_layout()
    return svg_from_fig(fig)


def build_silhouette_svg():
    metrics = results["metrics_summary"]["kmeans_sweep"]
    ks = list(range(4, 9))
    sils = [metrics[str(k)]["silhouette"] for k in ks]
    dbs = [metrics[str(k)]["db"] for k in ks]
    best = results["best_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    colors = ["#e41a1c" if k == best else "#377eb8" for k in ks]
    ax1.bar(ks, sils, color=colors)
    ax1.set_xlabel("k", fontsize=9)
    ax1.set_ylabel("シルエット係数（↑）", fontproperties=jp_font_s(9))
    ax1.set_title("k別シルエット係数", fontproperties=jp_font_s(10))
    ax1.set_xticks(ks)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    ax2.bar(ks, dbs, color=colors)
    ax2.set_xlabel("k", fontsize=9)
    ax2.set_ylabel("Davies-Bouldin値（↓）", fontproperties=jp_font_s(9))
    ax2.set_title("k別 Davies-Bouldin", fontproperties=jp_font_s(10))
    ax2.set_xticks(ks)

    fig.tight_layout()
    return svg_from_fig(fig)


def build_crosstab_svg(cross_tab, cluster_names, title_str):
    all_tags_here = results["all_tags"]
    # normalize keys to int for sorting, but access via str keys
    clusters = sorted(int(k) for k in cross_tab.keys())
    n_clusters = len(clusters)
    n_tags = len(all_tags_here)

    mat = np.zeros((n_clusters, n_tags))
    for i, c in enumerate(clusters):
        row = cross_tab.get(str(c)) or cross_tab.get(c, {})
        for j, t in enumerate(all_tags_here):
            mat[i, j] = row.get(t, 0)

    fig, ax = plt.subplots(figsize=(max(10, n_tags * 1.0), max(3.5, n_clusters * 0.75)))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n_tags))
    ax.set_xticklabels(all_tags_here, rotation=45, ha="right",
                       fontproperties=jp_font_s(8))
    ax.set_yticks(range(n_clusters))
    cluster_labels = [f"C{c}: {cluster_names.get(c, '')}" for c in clusters]
    ax.set_yticklabels(cluster_labels, fontproperties=jp_font_s(8))
    for i in range(n_clusters):
        for j in range(n_tags):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() * 0.5 else "black", fontsize=9)
    ax.set_title(title_str, fontproperties=jp_font_s(11))
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    fig.tight_layout()
    return svg_from_fig(fig)


def render_cluster_table(labels_key, names, terms_key, reps_key, cross_key):
    html = ""
    cross = results[cross_key]
    terms = results[terms_key]
    reps = results[reps_key]
    cluster_ids = sorted(int(k) for k in cross.keys())

    for c in cluster_ids:
        name = names.get(c, f"クラスタ {c}")
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        members = [ids[i] for i, lbl in enumerate(results[labels_key]) if lbl == c]
        n = len(members)
        top_terms = [t[0] for t in terms.get(str(c), [])[:8]]
        rep_idxs = reps.get(str(c), [])[:3]
        rep_cases = [(ids[i], titles[i]) for i in rep_idxs]
        tag_counts = cross[str(c)]
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:4]
        top_tags = [(t, cnt) for t, cnt in top_tags if cnt > 0]

        html += f"""
<div class="cluster-card" style="border-left: 5px solid {color};">
  <h4 style="color:{color}">C{c}: {name}</h4>
  <p><b>件数:</b> {n}件</p>
  <p><b>特徴語（c-TF-IDF 上位8語）:</b><br>
    {' '.join(f'<span class="term">{t}</span>' for t in top_terms)}
  </p>
  <p><b>代表ケース</b>（重心に最も近い上位3件）:</p>
  <ul>"""
        for cid, title in rep_cases:
            url = f"https://www.call4.jp/info.php?type=items&id={cid}"
            html += f'<li><a href="{url}" target="_blank">{title}</a> <code>[{cid}]</code></li>'
        html += "</ul>"
        if top_tags:
            html += "<p><b>既存タグ分布:</b> "
            html += ", ".join(f"{t}: {cnt}件" for t, cnt in top_tags)
            html += "</p>"
        html += "</div>"
    return html


def render_similarity_table():
    html = ""
    for ex in results["similarity_examples"]:
        q_title = ex["query_title"]
        q_id = ex["query_id"]
        q_cluster = ex["query_cluster_km"]
        url = f"https://www.call4.jp/info.php?type=items&id={q_id}"
        html += f"""
<div class="sim-card">
  <p><b>クエリ:</b> <a href="{url}" target="_blank">{q_title}</a>
     <span class="cluster-badge" style="background:{CLUSTER_COLORS[q_cluster % len(CLUSTER_COLORS)]}">C{q_cluster}: {KM_NAMES.get(q_cluster,'')}</span>
  </p>
  <table class="sim-table">
    <tr><th>類似度</th><th>ケース</th><th>クラスタ</th></tr>"""
        for item in ex["similar"]:
            ic = item["cluster_km"]
            iurl = f"https://www.call4.jp/info.php?type=items&id={item['id']}"
            html += f"""
    <tr>
      <td><b>{item['sim']:.3f}</b></td>
      <td><a href="{iurl}" target="_blank">{item['title']}</a></td>
      <td><span class="cluster-badge" style="background:{CLUSTER_COLORS[ic % len(CLUSTER_COLORS)]}">C{ic}: {KM_NAMES.get(ic,'')}</span></td>
    </tr>"""
        html += "</table></div>"
    return html


def build_scatter_data(coord_key_x, coord_key_y, color_by="cluster_km"):
    points = []
    for cid in ids:
        cc = case_clusters[cid]
        idx = id_to_idx[cid]
        if color_by == "cluster_km":
            cluster = cc["kmeans_best"]
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
            label = KM_NAMES.get(cluster, f"C{cluster}")
        elif color_by == "cluster_hier":
            cluster = cc["hierarchical_k6"]
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
            label = HIER_NAMES.get(cluster, f"C{cluster}")
        elif color_by == "status":
            cluster = statuses[idx]
            color = STATUS_COLORS.get(cluster, "#666")
            label = cluster
        elif color_by == "tag":
            tags = subject_tags_list[idx]
            cluster = tags[0] if tags else "タグなし"
            color = TAG_COLORS.get(cluster, "#666666")
            label = cluster
        points.append({
            "x": cc[coord_key_x], "y": cc[coord_key_y],
            "id": cid, "title": titles[idx],
            "color": color, "label": label,
            "status": statuses[idx],
            "tags": "、".join(subject_tags_list[idx]) if subject_tags_list[idx] else "なし",
        })
    return points


def generate_html():
    print("デンドログラム生成中...")
    dendro_svg = build_dendrogram_svg()
    print("シルエット係数グラフ生成中...")
    sil_svg = build_silhouette_svg()
    print("クロス集計ヒートマップ生成中...")
    cross_km_svg = build_crosstab_svg(
        results["cross_kmeans"],
        KM_NAMES,
        "k-means（k=4）× 既存タグ クロス集計"
    )
    cross_hier_svg = build_crosstab_svg(
        results["cross_hier"],
        HIER_NAMES,
        "階層クラスタリング（k=6）× 既存タグ クロス集計"
    )

    scatter_km = build_scatter_data("umap_e5_x", "umap_e5_y", "cluster_km")
    scatter_hier = build_scatter_data("umap_e5_x", "umap_e5_y", "cluster_hier")
    scatter_status = build_scatter_data("umap_e5_x", "umap_e5_y", "status")
    scatter_tag = build_scatter_data("umap_e5_x", "umap_e5_y", "tag")

    cluster_table_km = render_cluster_table(
        "kmeans_best_labels", KM_NAMES,
        "terms_kmeans", "reps_kmeans", "cross_kmeans"
    )
    cluster_table_hier = render_cluster_table(
        "hier_k6_labels", HIER_NAMES,
        "terms_hier", "reps_hier", "cross_hier"
    )
    sim_table = render_similarity_table()

    val = results["validation"]
    n_cases = results["n_cases"]
    gen_date = results["generated_at"][:10]

    scatter_km_json = json.dumps(scatter_km, ensure_ascii=False)
    scatter_hier_json = json.dumps(scatter_hier, ensure_ascii=False)
    scatter_status_json = json.dumps(scatter_status, ensure_ascii=False)
    scatter_tag_json = json.dumps(scatter_tag, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALL4 公共訴訟クラスタリング解析レポート</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Hiragino Sans", "YuGothic", "Noto Sans JP", sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #222; line-height: 1.75; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 28px 24px; }}
  h1 {{ font-size: 1.55em; border-bottom: 3px solid #1565C0; padding-bottom: 8px; margin-bottom: 6px; }}
  h2 {{ font-size: 1.2em; border-left: 5px solid #1976D2; padding: 8px 14px; margin-top: 44px; background: #E3F2FD; border-radius: 0 4px 4px 0; }}
  h3 {{ font-size: 1.05em; margin-top: 24px; color: #1565C0; }}
  h4 {{ font-size: 1.0em; margin: 10px 0 6px; }}
  .subtitle {{ color: #555; font-size: 0.88em; margin-bottom: 20px; }}
  .note {{ background: #FFF9C4; border-left: 4px solid #F9A825; padding: 10px 14px; border-radius: 0 4px 4px 0; margin: 12px 0; font-size: 0.9em; }}
  .info {{ background: #E8F5E9; border-left: 4px solid #388E3C; padding: 10px 14px; border-radius: 0 4px 4px 0; margin: 12px 0; }}
  .warn {{ background: #FFF3E0; border-left: 4px solid #E65100; padding: 10px 14px; border-radius: 0 4px 4px 0; margin: 12px 0; font-size: 0.88em; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 14px 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.12); text-align: center; }}
  .stat-card .num {{ font-size: 2em; font-weight: bold; color: #1565C0; }}
  .stat-card .lbl {{ font-size: 0.82em; color: #666; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.87em; }}
  th {{ background: #1565C0; color: white; padding: 7px 10px; text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #f5f5f5; }}
  .cluster-card {{ background: white; border-radius: 6px; padding: 16px 20px; margin: 10px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .term {{ display: inline-block; background: #E3F2FD; color: #0D47A1; border-radius: 3px; padding: 2px 6px; margin: 2px; font-size: 0.82em; font-family: monospace; }}
  .cluster-badge {{ display: inline-block; color: white; border-radius: 3px; padding: 2px 8px; font-size: 0.78em; margin-left: 6px; }}
  .sim-card {{ background: white; border-radius: 6px; padding: 12px 16px; margin: 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .sim-table td, .sim-table th {{ padding: 4px 8px; }}
  #scatter-container {{ background: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin: 12px 0; }}
  .scatter-btn {{ background: #1976D2; color: white; border: none; padding: 5px 12px; border-radius: 4px; cursor: pointer; margin: 3px; font-size: 0.82em; font-family: inherit; }}
  .scatter-btn.active {{ background: #0D47A1; font-weight: bold; }}
  .scatter-btn:hover {{ background: #1565C0; }}
  .legend-item {{ display: inline-flex; align-items: center; margin: 3px 7px; font-size: 0.8em; }}
  .legend-dot {{ width: 11px; height: 11px; border-radius: 50%; margin-right: 5px; flex-shrink: 0; }}
  .toc {{ background: white; border-radius: 8px; padding: 14px 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 26px; }}
  .toc ol {{ margin: 4px 0; padding-left: 20px; }}
  .toc li {{ margin: 3px 0; }}
  .toc a {{ color: #1565C0; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  .footer {{ margin-top: 50px; padding-top: 16px; border-top: 1px solid #ddd; font-size: 0.8em; color: #777; }}
  svg {{ max-width: 100%; height: auto; }}
  @media (max-width: 700px) {{ .grid-2, .grid-4 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">

<h1>CALL4 公共訴訟ケース クラスタリング解析レポート</h1>
<p class="subtitle">生成日: {gen_date} ｜ データ取得日: 2026-07-13 ｜ モデル: intfloat/multilingual-e5-small ｜ seed: 42</p>

<div class="note">
<b>分類は探索目的です。</b>このレポートの分類は、ケースのテキストから潜在的な構造を探索するためのものです。既存のタグ体系の代替、またはケースの質・重要性の評価を意図するものではありません。掲載ケースには実在の当事者が関わっており、データ利用は解析目的のみです。
</div>

<div class="toc">
<b>目次</b>
<ol>
  <li><a href="#overview">データ概要</a></li>
  <li><a href="#methods">手法解説</a></li>
  <li><a href="#umap">UMAP可視化</a></li>
  <li><a href="#kmeans">k-meansクラスタリング</a></li>
  <li><a href="#hierarchical">階層クラスタリング</a></li>
  <li><a href="#hdbscan">HDBSCAN（参考）</a></li>
  <li><a href="#validation">バリデーション（ARI / NMI）</a></li>
  <li><a href="#similarity">類似ケース検索</a></li>
  <li><a href="#insights">解釈と示唆</a></li>
  <li><a href="#limits">限界と次の一手</a></li>
</ol>
</div>

<!-- ========== 1. データ概要 ========== -->
<h2 id="overview">1. データ概要</h2>

<div class="grid-4">
  <div class="stat-card"><div class="num">{n_cases}</div><div class="lbl">取得ケース数</div></div>
  <div class="stat-card"><div class="num">49</div><div class="lbl">進行中（active）</div></div>
  <div class="stat-card"><div class="num">41</div><div class="lbl">アーカイブ済</div></div>
  <div class="stat-card"><div class="num">11</div><div class="lbl">ユニーク主題タグ数</div></div>
</div>

<div class="grid-2">
<div>
<h3>主題タグ分布</h3>
<table>
<tr><th>タグ</th><th>件数</th></tr>
<tr><td>公正な手続</td><td>43</td></tr>
<tr><td>政治参加・表現の自由</td><td>17</td></tr>
<tr><td>外国にルーツを持つ人々</td><td>16</td></tr>
<tr><td>環境・災害</td><td>14</td></tr>
<tr><td>刑事司法</td><td>14</td></tr>
<tr><td>ジェンダー・セクシュアリティ</td><td>13</td></tr>
<tr><td>情報公開</td><td>11</td></tr>
<tr><td>働き方</td><td>10</td></tr>
<tr><td>医療・福祉・障がい</td><td>9</td></tr>
<tr><td>沖縄</td><td>4</td></tr>
<tr><td>個人情報・プライバシー</td><td>1</td></tr>
</table>
</div>
<div>
<h3>テキスト統計</h3>
<div class="info">
<p><b>使用フィールド:</b><br>title + description + contents（本文全文）+ 最新3件アップデート</p>
<p><b>テキスト長（combined_text）:</b><br>
最小: 530文字 ／ 中央値: 5,674文字 ／ 最大: 11,881文字</p>
<p><b>本文（contents）あり:</b> 95件 / 95件</p>
<p><b>アップデートあり:</b> 84件 / 95件</p>
<p><b>埋め込み入力長上限:</b> 2,048文字（E5モデルのトークン制約に対応。長文ケースは後半情報が一部欠落）</p>
</div>
</div>
</div>

<!-- ========== 2. 手法解説 ========== -->
<h2 id="methods">2. 手法解説</h2>

<h3>2.1 埋め込み手法</h3>
<p><b>採用モデル: <code>intfloat/multilingual-e5-small</code>（次元数: 384）</b></p>
<p>日本語テキストで意味的な質を確保するため、多言語対応の文埋め込みモデルを採用しました。E5系モデルはコサイン類似度を前提として訓練されており、短文・長文いずれもロバストに扱えます。ローカル実行のため再現性が高く、API課金不要です。</p>

<div class="note">
<b>バイオインフォアナロジー：</b> 埋め込み行列（N=95 × 384次元）は、scRNA-seqにおける正規化済み発現行列（細胞 × 遺伝子）に相当します。TF-IDFベースライン ≈ 生カウント行列、多言語E5 ≈ バッチ補正・正規化済みの潜在表現、くらいの位置づけです。
</div>

<p><b>TF-IDFベースライン:</b> 日本語は分かち書き不要の文字nグラム（2〜4文字）でTF-IDF行列を構築し、SVDで50次元に削減。E5との比較対照として用いました。</p>

<h3>2.2 前処理・次元削減</h3>
<ul>
  <li>埋め込みベクトルはL2正規化済み（コサイン類似度と整合）</li>
  <li>可視化用UMAP: <code>n_neighbors=10</code>（N=95の小規模データ向けにデフォルト15より小さく設定）、<code>metric=cosine</code></li>
  <li><b>クラスタリングは高次元埋め込み（384次元）上で実施</b>。UMAPは可視化専用</li>
</ul>

<div class="warn">
<b>注意：</b> UMAPの2次元座標に過剰な意味を持たせないでください。UMAP座標間の距離は大域的に保存されず、近傍構造（局所的類似性）のみを反映します。scRNA-seqでのUMAP可視化と全く同じ制約です。
</div>

<h3>2.3 クラスタリング手法の比較</h3>
<table>
<tr><th>手法</th><th>概要</th><th>前提</th><th>本データでの結果</th></tr>
<tr>
  <td><b>k-means</b></td>
  <td>重心を繰り返し更新してk個のクラスタに割り当て</td>
  <td>球状クラスタ・等分散・k事前指定</td>
  <td>k=4が最良（シルエット係数）。95件を4クラスタに分割。</td>
</tr>
<tr>
  <td><b>Ward階層クラスタリング</b></td>
  <td>クラスタ内分散の増加を最小化しながら順次統合。デンドログラムで入れ子構造を可視化。</td>
  <td>距離行列のみ（分布仮定なし）</td>
  <td>k=6でシルエット係数最良。「公正な手続」クラスタが細分化。</td>
</tr>
<tr>
  <td><b>HDBSCAN</b></td>
  <td>密度ベース。ノイズ点（-1）を許容、k不要。</td>
  <td>密度で定義されるクラスタ・ノイズの存在</td>
  <td>2クラスタ + 71ノイズ点。N=95では密度推定が不安定。</td>
</tr>
<tr>
  <td><b>TF-IDF k-means（ベースライン）</b></td>
  <td>文字nグラムTF-IDF + SVD50次元 + k-means(k=6)</td>
  <td>語彙的類似性</td>
  <td>E5と低一致（ARI=0.06）→意味的埋め込みは異なる構造を検出</td>
</tr>
</table>

<!-- ========== 3. UMAP可視化 ========== -->
<h2 id="umap">3. UMAP可視化</h2>

<div id="scatter-container">
  <div style="margin-bottom:8px;">
    <button class="scatter-btn active" onclick="showScatter('km', this)">k-means k={best_k}</button>
    <button class="scatter-btn" onclick="showScatter('hier', this)">階層クラスタリング k=6</button>
    <button class="scatter-btn" onclick="showScatter('status', this)">ステータス別</button>
    <button class="scatter-btn" onclick="showScatter('tag', this)">既存タグ別</button>
  </div>
  <div id="scatter-legend" style="margin: 6px 0; min-height:24px; flex-wrap:wrap; display:flex;"></div>
  <canvas id="scatter-canvas" width="1000" height="520"
    style="width:100%; max-width:1000px; border:1px solid #eee; border-radius:4px; cursor:crosshair; display:block;"></canvas>
  <div id="scatter-tooltip"
    style="position:fixed; background:rgba(30,30,30,0.9); color:white; padding:8px 12px; border-radius:6px;
           font-size:0.8em; pointer-events:none; display:none; max-width:320px; z-index:1000; line-height:1.5;"></div>
</div>

<script>
const SCATTER = {{
  km: {scatter_km_json},
  hier: {scatter_hier_json},
  status: {scatter_status_json},
  tag: {scatter_tag_json}
}};
let curMode = 'km';

function getCoords(pts) {{
  const canvas = document.getElementById('scatter-canvas');
  const W = canvas.width, H = canvas.height, pad = 45;
  const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  const ymin=Math.min(...ys), ymax=Math.max(...ys);
  const xr=xmax-xmin||1, yr=ymax-ymin||1;
  return pts.map(p=>({{...p,
    cx: pad+(p.x-xmin)/xr*(W-2*pad),
    cy: H-pad-(p.y-ymin)/yr*(H-2*pad)
  }}));
}}

function draw(mode) {{
  const canvas=document.getElementById('scatter-canvas');
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#fafafa'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const pts=getCoords(SCATTER[mode]);
  pts.forEach(p=>{{
    ctx.beginPath(); ctx.arc(p.cx,p.cy,6,0,Math.PI*2);
    ctx.fillStyle=p.color; ctx.fill();
    ctx.strokeStyle='rgba(255,255,255,0.8)'; ctx.lineWidth=1.5; ctx.stroke();
  }});
  // legend
  const seen={{}};
  SCATTER[mode].forEach(p=>{{if(!seen[p.label]) seen[p.label]=p.color;}});
  const leg=document.getElementById('scatter-legend');
  leg.innerHTML=Object.entries(seen).map(([l,c])=>
    `<span class="legend-item"><span class="legend-dot" style="background:${{c}}"></span>${{l}}</span>`
  ).join('');
}}

function showScatter(mode, btn) {{
  curMode=mode;
  document.querySelectorAll('.scatter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  draw(mode);
}}

const canvas=document.getElementById('scatter-canvas');
canvas.addEventListener('mousemove', function(e) {{
  const r=this.getBoundingClientRect();
  const sx=this.width/r.width, sy=this.height/r.height;
  const mx=(e.clientX-r.left)*sx, my=(e.clientY-r.top)*sy;
  const pts=getCoords(SCATTER[curMode]);
  let near=null, nd=16;
  pts.forEach(p=>{{ const d=Math.hypot(p.cx-mx,p.cy-my); if(d<nd){{nd=d;near=p;}} }});
  const tt=document.getElementById('scatter-tooltip');
  if(near) {{
    tt.style.display='block';
    tt.style.left=(e.clientX+16)+'px'; tt.style.top=(e.clientY-12)+'px';
    tt.innerHTML=`<b>${{near.title}}</b><br>ID: ${{near.id}}<br>タグ: ${{near.tags}}<br>ステータス: ${{near.status}}<br>分類: ${{near.label}}`;
  }} else tt.style.display='none';
}});
canvas.addEventListener('mouseleave',()=>document.getElementById('scatter-tooltip').style.display='none');
canvas.addEventListener('click', function(e) {{
  const r=this.getBoundingClientRect();
  const sx=this.width/r.width, sy=this.height/r.height;
  const mx=(e.clientX-r.left)*sx, my=(e.clientY-r.top)*sy;
  const pts=getCoords(SCATTER[curMode]);
  let near=null, nd=14;
  pts.forEach(p=>{{ const d=Math.hypot(p.cx-mx,p.cy-my); if(d<nd){{nd=d;near=p;}} }});
  if(near) window.open(`https://www.call4.jp/info.php?type=items&id=${{near.id}}`,'_blank');
}});
draw('km');
</script>
<p style="font-size:0.82em; color:#666;">※ 点をクリックするとCALL4のケースページが開きます。マウスオーバーでケース情報を表示。</p>

<!-- ========== 4. k-means ========== -->
<h2 id="kmeans">4. k-meansクラスタリング（E5埋め込み、k={best_k}）</h2>

<h3>4.1 k選択</h3>
{sil_svg}

<p>シルエット係数でk=4が最良（silhouette={results["metrics_summary"]["kmeans_sweep"]["4"]["silhouette"]:.4f}）。全体的に値が低い（0.02〜0.04）のはN=95の小規模データと、日本語訴訟テキスト間の意味的重複（「公正な手続」が多くのケースに横断する）に起因します。「指標が最良でも解釈不能なら不採用」という原則のもと、k=4を採用し、k=6（階層クラスタリング）との比較も行います。</p>

<h3>4.2 クラスタ詳細（k-means k=4）</h3>
{cluster_table_km}

<h3>4.3 クラスタ × 既存タグ クロス集計</h3>
{cross_km_svg}

<!-- ========== 5. 階層クラスタリング ========== -->
<h2 id="hierarchical">5. 階層クラスタリング（Ward法、k=6）</h2>

<h3>5.1 デンドログラム</h3>
{dendro_svg}
<p style="font-size:0.85em;">赤点線: k=6カット位置 ／ オレンジ点線: k=4カット位置。横軸はケースID。縦軸は統合時のWard距離。</p>

<p>k=6で切断することで、k-meansでは1クラスタだった「公正な手続」関連が細分化され、入管・刑事司法・情報公開・選挙権という4方向に分離します。デンドログラムを見ると、ジェンダー・セクシュアリティ系ケースと選挙権系ケースが比較的早く合流する（=高い類似度で関連する）ことが分かります。</p>

<h3>5.2 クラスタ詳細（階層 k=6）</h3>
{cluster_table_hier}

<h3>5.3 クラスタ × 既存タグ クロス集計</h3>
{cross_hier_svg}

<!-- ========== 6. HDBSCAN ========== -->
<h2 id="hdbscan">6. HDBSCAN（参考）</h2>

<p>HDBSCAN（<code>min_cluster_size=5, min_samples=3</code>）の結果: <b>2クラスタ + 71ノイズ点（-1）</b>。N=95の小規模データでは密度推定が不安定になり、ほとんどの点がノイズ扱いとなりました。</p>

<div class="warn">
<b>教訓：</b> HDBSCANはN=100規模でも動作しますが、本データのように「公正な手続」など広範な概念が多数ケースに横断する場合、密度的に分離したクラスタが形成されにくくノイズ点が多くなります。明確な密度差があるデータ（N≥500など）でより効果的です。scRNA-seqで言えば、細胞集団が均質に混在していてLeiden/Louvainがコミュニティを見つけられない状況に近いです。
</div>

<!-- ========== 7. バリデーション ========== -->
<h2 id="validation">7. バリデーション（ARI / NMI）</h2>

<h3>7.1 手法間・既存タグとの比較</h3>
<table>
<tr><th>比較</th><th>ARI</th><th>NMI</th><th>解釈</th></tr>
<tr><td>k-means（k=4, E5）vs 既存タグ</td><td>{val["km_vs_tags"]["ari"]:.3f}</td><td>{val["km_vs_tags"]["nmi"]:.3f}</td><td>低い一致→新分類軸の候補。タグの焼き直しではない。</td></tr>
<tr><td>階層（k=6, E5）vs 既存タグ</td><td>{val["hier_vs_tags"]["ari"]:.3f}</td><td>{val["hier_vs_tags"]["nmi"]:.3f}</td><td>やや高い→部分的対応あり（ジェンダー・情報公開で対応）</td></tr>
<tr><td>HDBSCAN vs 既存タグ</td><td>{val["hdb_vs_tags"]["ari"]:.3f}</td><td>{val["hdb_vs_tags"]["nmi"]:.3f}</td><td>2クラスタのみのため参考値</td></tr>
<tr><td>k-means vs 階層クラスタリング</td><td>{val["km_vs_hier"]["ari"]:.3f}</td><td>{val["km_vs_hier"]["nmi"]:.3f}</td><td>中程度の一致→同じ大域構造を共有。細かい割り当てで差異。</td></tr>
<tr><td>k-means E5 vs k-means TF-IDF</td><td>{val["km_vs_tfidf"]["ari"]:.3f}</td><td>{val["km_vs_tfidf"]["nmi"]:.3f}</td><td>非常に低い→意味的埋め込みとTF-IDFは全く異なる構造を検出</td></tr>
<tr><td>k-means vs HDBSCAN</td><td>{val["km_vs_hdb"]["ari"]:.3f}</td><td>{val["km_vs_hdb"]["nmi"]:.3f}</td><td>ほぼ無相関（HDBSCANが2クラスタのため）</td></tr>
</table>

<div class="info">
<b>解釈指針（バイオインフォアナロジー）：</b>
<ul>
  <li><b>クラスタ vs 既存タグ ARI=0.07〜0.13（低〜中低）：</b>「一致しすぎ＝タグの焼き直し、乖離しすぎ＝ノイズor新発見」の原則に基づくと、本解析は適度な乖離を示しており、<b>既存タグとは異なる、テキストから浮かび上がる新しい構造</b>が存在することを示唆します。scRNA-seqで言えば、マーカー遺伝子ベースのアノテーションとunsupervisedクラスタが完全一致しない状況で、新規細胞型候補が存在することに近い。</li>
  <li><b>E5 vs TF-IDF ARI=0.06（非常に低い）：</b>文意味埋め込みと表層形（文字nグラム）は全く異なる構造を検出。TF-IDF ≈ 語彙的類似（同じ単語を使うか）、E5 ≈ 意味的類似（意味が近いか）。</li>
  <li><b>k-means vs 階層 ARI=0.26（中程度）：</b>2手法が粗い大域構造は共有。細かい割り当てで差異あり。</li>
</ul>
</div>

<!-- ========== 8. 類似ケース検索 ========== -->
<h2 id="similarity">8. 類似ケース検索</h2>

<p>コサイン類似度（E5埋め込み空間）による近傍上位4件の例。</p>

{sim_table}

<div class="info">
<b>直感的妥当性チェック：</b>
<ul>
  <li>「財務省改ざん事件」→「財務省補正要求訴訟」(sim=0.929)：同一省庁・同一争点。</li>
  <li>「海外国民審査訴訟」→「在外選挙権訴訟」(sim=0.929)：在外日本人の権利行使という同一軸。</li>
  <li>「ジャーナリスト渡航」→「入管収容訴訟」(sim=0.901)：移動の自由・入管制度という共通テーマ。</li>
</ul>
これらの結果は直感と高く一致しており、E5埋め込みが意味的類似性を適切に捉えていることを示します。
</div>

<!-- ========== 9. 解釈と示唆 ========== -->
<h2 id="insights">9. 解釈と示唆</h2>

<h3>9.1 新分類軸の候補</h3>
<table>
<tr><th>新分類軸候補</th><th>既存タグとの関係</th><th>具体ケース例</th></tr>
<tr>
  <td><b>国家による身体的自由の侵害</b></td>
  <td>刑事司法・外国にルーツを持つ人々・公正な手続にまたがる</td>
  <td>入管収容死亡、取調べ中死亡、優生保護法、人質司法、刑務所医療、黙秘権</td>
</tr>
<tr>
  <td><b>書類・記録の透明性</b></td>
  <td>情報公開・政治参加にまたがる</td>
  <td>財務省改ざん、国葬文書、学術会議任命拒否理由、イラク戦争検証、日米合同委員会</td>
</tr>
<tr>
  <td><b>個人の法的地位・アイデンティティ</b></td>
  <td>ジェンダー・外国にルーツ・公正な手続にまたがる</td>
  <td>同性婚、性別変更手術要件、国籍法、難民認定、婚外子、赤ちゃん取り違え</td>
</tr>
<tr>
  <td><b>地域住民 vs 大規模開発・インフラ</b></td>
  <td>環境・沖縄・公正な手続にまたがる</td>
  <td>神宮外苑、リニア、馬毛島、羽田新ルート、石炭火力、野村ダム・鬼怒川水害</td>
</tr>
</table>

<h3>9.2 既存タグ体系への示唆</h3>
<ul>
  <li>「<b>公正な手続</b>」タグが43件（全体の45%）に付与されており、実質的な「その他」タグになっています。テキスト上の類似性からは、このタグ内に少なくとも4〜5の異なるサブテーマが存在します。</li>
  <li>k-means C1（刑事司法・入管・個人の自由）は既存タグ「刑事司法」「外国にルーツ」「公正な手続」にまたがります。これらを統合する「<b>身体的自由の侵害</b>」という新軸が有望です。</li>
  <li>k-means C2（情報公開・司法の独立）は既存タグ「情報公開」「政治参加・表現の自由」にまたがります。「<b>行政の透明性</b>」という統合軸が考えられます。</li>
  <li>階層クラスタリングで、「ジェンダー・セクシュアリティ」と「選挙権・政治参加」が比較的近い枝に位置することは興味深い発見です。どちらも「<b>制度に排除されてきた集団が法的承認を求める</b>」という共通構造を持つ可能性があります。</li>
</ul>

<!-- ========== 10. 限界と次の一手 ========== -->
<h2 id="limits">10. 限界と次の一手</h2>

<h3>10.1 現在の限界</h3>
<ul>
  <li><b>N=95の小ささ：</b>シルエット係数が全体的に低く（0.02〜0.04）、クラスタの統計的安定性に限界があります。「統計的に有意なクラスタ」とは主張できず、<b>探索・仮説生成のツール</b>として扱ってください。</li>
  <li><b>多ラベル性の無視：</b>ARI/NMI計算では各ケースの第1タグのみを使用。多ラベルメトリクス（Jaccard類似度ベースなど）への拡張が有益です。</li>
  <li><b>埋め込みの切り詰め：</b>multilingual-E5-smallへの入力を先頭2,048文字に制限。長文ケース（最大11,881文字）では後半情報が欠落します。<code>multilingual-e5-large</code>や長文対応モデルとの比較が望ましいです。</li>
  <li><b>テキスト種別の混在：</b>概要・本文・アップデートを単純連結。各フィールドへの重み付けや、本文のみ・アップデートのみでの別途解析が有益です。</li>
  <li><b>HDBSCAN不適合：</b>本データではほとんどノイズ扱いとなり、解釈に限界があります。</li>
</ul>

<h3>10.2 次の一手</h3>
<ol>
  <li>ブートストラップ安定性評価（n=100回サブサンプリング → ARI分布）</li>
  <li><code>multilingual-e5-large</code>またはOpenAI Embeddings APIとの比較</li>
  <li>タグ体系の再設計提案（「公正な手続」の細分化、「身体的自由の侵害」等の新タグ案）</li>
  <li>類似ケース検索のインタラクティブWebUI化</li>
  <li>提訴年別の話題傾向変化の時系列分析</li>
</ol>

<h3>10.3 再現情報</h3>
<table>
<tr><th>項目</th><th>値</th></tr>
<tr><td>埋め込みモデル</td><td>intfloat/multilingual-e5-small</td></tr>
<tr><td>乱数シード</td><td>42</td></tr>
<tr><td>TF-IDF</td><td>analyzer=char_wb, ngram=(2,4), min_df=2, max_features=20000, sublinear_tf=True</td></tr>
<tr><td>SVD</td><td>n_components=50</td></tr>
<tr><td>UMAP</td><td>n_neighbors=10, min_dist=0.1, metric=cosine</td></tr>
<tr><td>k-means</td><td>k=4（シルエット係数最良）, n_init=10</td></tr>
<tr><td>Ward階層クラスタリング</td><td>コサイン距離, k=6</td></tr>
<tr><td>HDBSCAN</td><td>min_cluster_size=5, min_samples=3, metric=euclidean, eom</td></tr>
<tr><td>データ取得日</td><td>2026-07-13</td></tr>
<tr><td>Python</td><td>3.9（venv）</td></tr>
<tr><td>主要ライブラリ</td><td>scikit-learn, sentence-transformers, umap-learn, hdbscan, scipy, matplotlib</td></tr>
</table>

<div class="footer">
<p>CALL4 公共訴訟ケース クラスタリング解析レポート ｜ 生成: {gen_date} ｜ 探索的解析（仮説生成目的）</p>
<p>データソース: <a href="https://www.call4.jp" target="_blank">CALL4 (call4.jp)</a> — 非営利訴訟支援プラットフォーム<br>
本レポートは探索目的の解析であり、当事者評価・法的判断を意図するものではありません。</p>
</div>

</div><!-- /container -->
</body>
</html>"""
    return html


print("HTMLレポートを生成中...")
html = generate_html()
out_path = RESULTS_DIR / "call4_clustering_report.html"
out_path.write_text(html, encoding="utf-8")
size_kb = out_path.stat().st_size // 1024
print(f"レポート保存完了: {out_path} ({size_kb} KB)")
