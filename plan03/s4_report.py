#!/usr/bin/env python3
"""plan03 Stage 4 — Single self-contained bilingual HTML report (tag-only track).

Static inline-SVG figures (matplotlib): tag distribution + co-occurrence,
MCA biplot, tag co-occurrence network, NMF-tag mixture stacked bar, spectral
co-cluster block heatmap, plan03<->plan02 ARI heatmap.
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
FEAT = os.path.join(ROOT, "plan03", "features")
RES = os.path.join(ROOT, "plan03", "results")
REP = os.path.join(ROOT, "plan03", "report")
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


PAL = ["#4C8C2B", "#2B6CB0", "#B0532B", "#8A4FA8", "#C29B2C", "#2BA8A0",
       "#C0392B", "#7F8C8D"]

# ---- load ----------------------------------------------------------------
sz = np.load(os.path.join(FEAT, "tag_sim.npz"), allow_pickle=True)
M = sz["matrix"].astype(float)
tags = [str(t) for t in sz["columns"]]
mca = np.load(os.path.join(FEAT, "mca.npz"), allow_pickle=True)
s0 = json.load(open(os.path.join(FEAT, "s0_meta.json")))
ev = json.load(open(os.path.join(RES, "evaluation.json")))
nmf = json.load(open(os.path.join(RES, "membership_nmf_tags.json")))
comm = json.load(open(os.path.join(RES, "tag_communities.json")))
interp = json.load(open(os.path.join(RES, "interpretation.json")))
cocl = json.load(open(os.path.join(RES, "labels_cocluster_k6.json")))
titles = s0["titles"]
case_ids = s0["case_ids"]
N, T = M.shape
counts = {d["tag"]: d["count"] for d in s0["tags_by_count"]}
tag_order = [d["tag"] for d in s0["tags_by_count"]]


def url(cid):
    return f"https://www.call4.jp/info.php?type=items&id={cid}"


# ==========================================================================
# Fig 1. tag distribution bar + co-occurrence heatmap
# ==========================================================================
def fig_dist():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(tag_order))
    a1.barh(y, [counts[t] for t in tag_order], color="#2B6CB0")
    a1.set_yticks(y); a1.set_yticklabels(tag_order, fontproperties=jp(9))
    a1.invert_yaxis(); a1.set_xlabel("件数 / cases", fontproperties=jp(9))
    a1.set_title("タグ分布 Tag distribution (N=95)", fontproperties=jp(10))
    for i, t in enumerate(tag_order):
        a1.text(counts[t] + 0.4, i, str(counts[t]), va="center", fontproperties=jp(8))

    cooc = np.array([[comm["cooccurrence"][i][j] for j in range(T)] for i in range(T)])
    idx = [tags.index(t) for t in tag_order]
    C = cooc[np.ix_(idx, idx)].astype(float)
    Cd = C.copy()
    np.fill_diagonal(Cd, np.nan)
    im = a2.imshow(Cd, cmap="YlOrRd")
    a2.set_xticks(range(T)); a2.set_yticks(range(T))
    a2.set_xticklabels(tag_order, fontproperties=jp(7), rotation=90)
    a2.set_yticklabels(tag_order, fontproperties=jp(7))
    a2.set_title("タグ共起 Tag co-occurrence", fontproperties=jp(10))
    for i in range(T):
        for j in range(T):
            if i != j and C[i, j] > 0:
                a2.text(j, i, int(C[i, j]), ha="center", va="center",
                        fontsize=6, color="#333")
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return svg_of(fig)


# ==========================================================================
# Fig 2. MCA biplot (cases colored by NMF-tag archetype + tag categories)
# ==========================================================================
def fig_mca():
    rc = mca["row_coords"]
    cc = mca["col_coords"]
    col_labels = [str(x) for x in mca["col_labels"]]
    inertia = mca["inertia"]
    dom = np.asarray(nmf["dominant_topic"])
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for k in range(6):
        m = dom == k
        ax.scatter(rc[m, 0], rc[m, 1], s=26, c=PAL[k], alpha=0.7,
                   edgecolors="white", linewidths=0.4, label=f"A{k}")
    # overlay present-tag categories
    for j, lab in enumerate(col_labels):
        if lab.endswith("=1"):
            name = lab[:-2]
            ax.annotate(name, (cc[j, 0], cc[j, 1]), fontproperties=jp(9),
                        color="#111", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#fffbe6",
                                  ec="#C29B2C", lw=0.6, alpha=0.9))
    ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
    ax.set_xlabel(f"MCA dim1 ({inertia[0]*100:.1f}%)", fontproperties=jp(10))
    ax.set_ylabel(f"MCA dim2 ({inertia[1]*100:.1f}%)", fontproperties=jp(10))
    ax.set_title("MCAバイプロット：ケース（点=NMFアーキタイプ色）＋タグ（枠）\n"
                 "MCA biplot: cases (color=NMF archetype) + tags",
                 fontproperties=jp(11))
    ax.legend(loc="upper right", fontsize=8, title="NMF archetype")
    fig.tight_layout()
    return svg_of(fig)


# ==========================================================================
# Fig 3. tag co-occurrence network (nodes colored by meta-tag community)
# ==========================================================================
def fig_network():
    cooc = np.array([[comm["cooccurrence"][i][j] for j in range(T)] for i in range(T)])
    mem = comm["membership"]
    edges, w = [], []
    for i in range(T):
        for j in range(i + 1, T):
            if cooc[i, j] > 0:
                edges.append((i, j)); w.append(cooc[i, j])
    g = ig.Graph(n=T, edges=edges)
    import random as _r
    _r.seed(42)
    try:
        ig.set_random_number_generator(_r)
    except Exception:
        pass
    lay = g.layout_fruchterman_reingold(weights=w, niter=800)
    coords = np.array(lay.coords)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    wmax = max(w)
    for (i, j), ww in zip(edges, w):
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color="#999", lw=0.5 + 3.5 * ww / wmax, alpha=0.5, zorder=1)
    sizes = np.array([counts[tags[i]] for i in range(T)])
    ax.scatter(coords[:, 0], coords[:, 1], s=60 + sizes * 18,
               c=[PAL[mem[i]] for i in range(T)], edgecolors="white",
               linewidths=1.2, zorder=2)
    for i in range(T):
        ax.annotate(tags[i], (coords[i, 0], coords[i, 1]), fontproperties=jp(9),
                    ha="center", va="center", zorder=3)
    ax.set_axis_off()
    ax.set_title("タグ共起ネットワーク（色＝共起コミュニティ／メタタグ, 太さ＝共起数）\n"
                 "Tag co-occurrence network (color = meta-tag community)",
                 fontproperties=jp(11))
    fig.tight_layout()
    return svg_of(fig)


# ==========================================================================
# Fig 4. NMF-tag mixture stacked bar (95 cases sorted by dominant archetype)
# ==========================================================================
def fig_stack():
    R = np.array(nmf["ratios"])
    dom = np.asarray(nmf["dominant_topic"])
    order = sorted(range(N), key=lambda i: (dom[i], -R[i, dom[i]]))
    Rs = R[order]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    bottom = np.zeros(N)
    x = np.arange(N)
    for k in range(6):
        ax.bar(x, Rs[:, k], bottom=bottom, width=1.0, color=PAL[k], label=f"A{k}")
        bottom += Rs[:, k]
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(0, 1)
    ax.set_xlabel("95ケース（優勢アーキタイプ順） / cases sorted by dominant archetype",
                  fontproperties=jp(9))
    ax.set_ylabel("混合比 / ratio", fontproperties=jp(9))
    ax.set_title("NMF（タグ）混合メンバーシップ NMF-on-tags mixture", fontproperties=jp(11))
    ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return svg_of(fig)


# ==========================================================================
# Fig 5. spectral co-cluster block heatmap (cases x tags reordered)
# ==========================================================================
def fig_cocluster():
    row_lab = np.asarray(cocl["labels"])
    col_lab = np.asarray(cocl["tag_labels"])
    r_order = np.argsort(row_lab)
    c_order = np.argsort(col_lab)
    B = M[np.ix_(r_order, c_order)]
    fig, ax = plt.subplots(figsize=(6.2, 8))
    ax.imshow(B, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(T))
    ax.set_xticklabels([tags[c] for c in c_order], fontproperties=jp(8), rotation=90)
    ax.set_yticks([]); ax.set_ylabel("95 cases (block-ordered)", fontproperties=jp(9))
    # block separators
    for b in np.where(np.diff(np.sort(row_lab)))[0]:
        ax.axhline(b + 0.5, color="#C0392B", lw=0.8)
    for b in np.where(np.diff(np.sort(col_lab)))[0]:
        ax.axvline(b + 0.5, color="#C0392B", lw=0.8)
    ax.set_title("スペクトル共クラスタ Spectral co-clustering\n(cases×tags blocks)",
                 fontproperties=jp(10))
    fig.tight_layout()
    return svg_of(fig)


# ==========================================================================
# Fig 6. plan03 <-> plan02 ARI heatmap
# ==========================================================================
def fig_ari():
    tag_methods = list(ev["Q3_vs_plan02"].keys())
    p2 = list(next(iter(ev["Q3_vs_plan02"].values())).keys())
    A = np.array([[ev["Q3_vs_plan02"][tm][pm]["ARI"] for pm in p2]
                  for tm in tag_methods])
    fig, ax = plt.subplots(figsize=(5.2, 6))
    im = ax.imshow(A, cmap="RdYlGn", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(p2))); ax.set_xticklabels(p2, fontproperties=jp(9), rotation=20, ha="right")
    ax.set_yticks(range(len(tag_methods)))
    ax.set_yticklabels(tag_methods, fontproperties=jp(9))
    for i in range(len(tag_methods)):
        for j in range(len(p2)):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="#111")
    ax.set_title("タグ手法 × plan02テキスト手法 ARI\nTag-only vs text-driven (ARI)",
                 fontproperties=jp(10))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return svg_of(fig)


# ==========================================================================
# HTML assembly
# ==========================================================================
def archetype_cards():
    parts = []
    for g in interp["mixture"]["nmf_tags"]:
        atags = "、".join(t["tag"] for t in g["archetype_tags"])
        cases = "".join(
            f'<li><a href="{url(c["id"])}" target="_blank">{c["title"]}</a> '
            f'<span class="r">({c["ratio"]})</span></li>' for c in g["top_cases"])
        parts.append(
            f'<div class="card"><div class="chip" style="background:{PAL[g["archetype"]]}">'
            f'A{g["archetype"]}</div><h4>{g["name"]}</h4>'
            f'<p class="tags">タグ荷重 / loading: {atags}</p>'
            f'<p class="meta">優勢ケース数 dominant n = {g["n_dominant"]}</p>'
            f'<ul class="cases">{cases}</ul></div>')
    return "\n".join(parts)


def validity_rows():
    rows = []
    for name, v in ev["Q1_internal"].items():
        sil = v["silhouette_jaccard"]
        rows.append(f"<tr><td>{name}</td><td>{v['K']}</td>"
                    f"<td>{sil if sil is not None else '—'}</td>"
                    f"<td>{v['largest_cluster_frac']:.0%}</td></tr>")
    return "\n".join(rows)


def splitting_rows():
    rows = []
    for d in ev["Q4_splitting_tags"]:
        rows.append(f"<tr><td>{d['tag']}</td><td>{d['n']}</td>"
                    f"<td>{d['max_topic_frac']:.0%}</td>"
                    f"<td>{d['n_topics_touched']} / 6</td></tr>")
    return "\n".join(rows)


def meta_tag_items():
    out = []
    for c, grp in ev["Q4_meta_tags"].items():
        col = PAL[int(c)]
        chips = " ".join(f'<span class="tg">{t}</span>' for t in grp)
        out.append(f'<li><span class="dot" style="background:{col}"></span>{chips}</li>')
    return "\n".join(out)


best = ev["Q3_best_recovery"]
ari_leiden = best["text-Leiden(K=5)"][0]
ari_nmf = best["text-NMF(K=6)"][0]

# ---- closing essay (plain string; no braces so it can sit in the f-string) --
ESSAY = """
<h2>9. 考察：タグを用いた分類は適切か
<span class="en">Discussion — is tag-based classification appropriate?</span></h2>

<div class="essay">

<div class="verdict">
<b>先に結論 / Verdict first.</b>
既存タグだけを使った分類は、<b>目的次第で「適切」にも「不適切」にもなる</b>。
おおまかには次のとおり：
<div class="tldr">
<div class="g">◎ 適切</div><div>ベースライン（基準線）／タグ体系そのものの点検・批評／
テキスト由来構造の裏取り（三角測量）／透明で再現しやすい粗い俯瞰</div>
<div class="n">✕ 不適切</div><div>「新しい分類軸の発見」の主役／
細かい類似ケース推薦／「統計的に有意なクラスタがある」という主張</div>
<div class="m">△ 条件つき</div><div>混合メンバーシップとして使う分には可（多重ラベルを壊さないため）。
ただし解像度の天井は残る</div>
</div>
本 plan03 は<b>◎の用途に限定して</b>タグ分類を使っており、その範囲では方法論的に妥当である。
以下、なぜそう言えるのかを順に説明する。
</div>

<h3>9.1 「適切か」は問いの目的で決まる / Appropriateness depends on the goal</h3>
<p>本プロジェクトの目的は4つあった——
(1) 新しい分類軸の発見、(2) 類似ケースの推薦・検索、(3) 既存タグ体系の見直し、(4) 対外可視化。
「タグ分類が適切か」を一括では答えられないのは、<b>この4つで答えが逆になる</b>からである。
タグを入力にした分類は (3)(4) と (1) の“裏取り”には向くが、(1) の主役や (2) の精密化には向かない。
理由はこの後の (9.2) と (9.3) に尽きる。</p>

<h3>9.2 最大の落とし穴：循環論法 / The core pitfall: circularity</h3>
<p><b>循環論法（circular reasoning）</b>とは、証明したい結論を前提として使ってしまう論法のこと。
タグ分類にはこれが構造的に潜む。<b>タグを特徴量にしてケースを分け、その結果から「タグはこう分かれる」と言う</b>のは、
入力（タグ）と結論（タグの構造）が同じものを指しており、ほとんど同義反復に近い。
タグでクラスタリングして得られるのは<b>「タグがどう一緒に使われているか（共起構造）」</b>だけで、
<b>「そのタグ体系が良いか／取りこぼしがないか」までは原理的に言えない</b>。
物差しそのものを、その物差しで測っているからである。</p>
<p>だからこそ本レポートは、タグ分類の結果を「新発見」とは呼ばず、
<b>タグの使われ方の記述（6.1 背景タグ・6.2 冗長タグ）</b>と、
<b>外部の物差し（テキスト由来の構造）との突き合わせ（第5章）</b>に用途を絞っている。
「タグ体系の批評」が成立するのは、批評の基準を<b>タグの外</b>——テキスト——に置いているときだけである。</p>

<h3>9.3 タグは「測定量」ではなく「人手ラベル」 / Tags are labels, not measurements</h3>
<p>ここが最も本質的な点。機械学習では特徴量に2種類ある。
一つは対象から<b>測って得た量（measurement）</b>、もう一つは人が<b>目的をもって貼ったラベル（label）</b>。
テキスト（概要・本文）は前者に近く、タグは明確に後者である。
<b>ラベルを入力にしてクラスタリングすると、元のラベル付け規則をなぞり直すだけになりやすい</b>
（機械学習でいう<b>ラベル・リーク／label leakage</b>＝答えの一部が入力に混ざっている状態に近い）。</p>
<p class="gloss">バイオインフォの対応：これは<b>発現行列でクラスタリングする（de novo）</b>のと、
<b>すでに付いている細胞型アノテーションでクラスタリングする</b>ことの違いに等しい。
後者は「アノテーションの再確認」にはなるが、「新しい細胞集団の発見」にはならない。
scRNA-seq で marker 遺伝子の手動ゲーティング結果を再クラスタリングしても、
新しい亜集団は出てこないのと同じ理屈である。タグは<b>事前知識（prior）であって真実（ground truth）ではない</b>。</p>

<h3>9.4 情報量の天井 —— 11タグ・37組合せという自由度 / The information ceiling</h3>
<p><b>自由度（degrees of freedom）</b>とは、データが取りうる状態の“広さ”のこと。
今回の特徴は11個の二値（0/1）タグだけで、実際に現れた<b>ユニークな組合せは37通り</b>しかない。
つまり95件は<b>タグ空間では最大37点にしか散らばれない</b>——距離ベースの手法が区別できる点は原理的にここが上限。
さらに<b>68%のケースペアはタグを1つも共有しない</b>（＝Jaccard類似度がゼロ）。
これは「まばら（sparse）で、かつ多くのケースが互いに“無関係”に見える」空間で、
<b>そもそも滑らかな地形やグラデーションが存在しにくい</b>。
テキスト（数千語→数十次元の連続空間）と比べて、情報の器が桁違いに小さい。</p>
<p>この天井は手法をどれだけ工夫しても超えられない。plan03 で複数手法が
だいたい同じ粗い塊に落ち着いたのは、手法が優秀だからではなく、
<b>データがそれ以上の分解能を持っていない</b>ことの反映である。</p>

<h3>9.5 タグ特有の品質問題 / Quality issues specific to these tags</h3>
<p>上の一般論に加え、CALL4 のタグには実務的なクセがある。いずれも分類の“素性”を悪くする。</p>
<ul>
<li><b>支配タグ（背景化）</b>：「公正な手続」は95件中43件（45%）に付き、テキスト6トピック<b>すべて</b>に出現する。
全体に薄く広がる語は<b>ケースを分ける力（識別力）を持たない</b>。
バイオでいえば全細胞で一様に光るマーカーのようなもので、ゲーティングに使えない。</li>
<li><b>粒度の不揃い</b>：一方で「個人情報・プライバシー」は1件、「沖縄」は4件。
これらは<b>単独ではクラスタを構成できない</b>（統計的に支えるサンプルがない）。
1つの体系の中に、広すぎるタグと狭すぎるタグが同居している。</li>
<li><b>多重ラベル（multi-label）の圧殺</b>：1ケースは平均1.6個のタグを持つ“かけもち”である。
にもかかわらず k-means や階層のような<b>排他的（各ケースを1グループに）</b>な手法に通すと、
かけもちの情報が捨てられ、無理な単一割り当てが起きる。
この点で<b>混合メンバーシップ（NMF・LCA）の方がタグの性質に忠実</b>——(9.6) で詳述。</li>
<li><b>目的バイアス</b>：タグは統計的な特徴量として設計されたのではなく、
<b>支援・検索・アドボカシーのために人が付けた</b>ものである。
「公正な手続」の遍在は、CALL4が“公共訴訟”という枠組みを重視していることの反映かもしれず、
必ずしも法分野の実体的な違いを表さない。ラベルは<b>付けた人の視点</b>を含む。</li>
</ul>

<h3>9.6 タグ分類の“中でも”手法に優劣がある / Even within tag-only, methods differ</h3>
<p>「タグ分類」と一括りにできない。同じタグ行列でも、手法の前提がデータと合うかで結果の質が変わる。</p>
<ul>
<li><b>距離の選択が決定的</b>：共不在（両方“タグ無し”）を似ていると数える距離（Hamming等）を使うと、
支配タグ経由で無関係なケースが束になる。<b>共有の有無だけを見る Jaccard</b>を選んで初めて意味が出る。</li>
<li><b>排他 vs 混合</b>：<b>Jaccard階層（排他）は「公正な手続」の巨大塊（約6割）に退化</b>した
（第4章の表）。多重ラベルを1つに潰す排他手法の弱点が、支配タグと相まって最悪化した例である。
対して<b>混合系（NMF・LCA）はかけもちを比率で表せる</b>ので、同じデータでもバランス良く分かれた。</li>
<li><b>内部指標の“見かけの良さ”に注意</b>：タグ空間の<b>シルエット係数</b>
（各ケースが自分のグループに収まる度合いの点数）は 0.3〜0.4 と、テキスト空間の 0.12 前後より高く出た。
だがこれは分類が“優れている”からではなく、<b>離散で同一点が多い空間では機械的に高く出る</b>ためで、
<b>異なる空間の値を並べて比べてはいけない</b>（距離の物差しが違う）。数字の大小に釣られない。</li>
<li><b>モデル選択の示唆</b>：LCA を情報量規準<b>BIC</b>
（あてはまりの良さと複雑さのバランスで最適なクラス数を選ぶ基準）で選ぶと、<b>最適は K=2</b> だった。
これは「タグ情報はそもそも“2つに大別”程度しか強く支持しない」というデータからの率直な返答である。
本レポートの K=6 は<b>plan02 と比較するための便宜的な固定値</b>であって、データが要求した数ではない。</li>
</ul>

<h3>9.7 それでもタグ分類が“適切”に働く場面 / When it genuinely helps</h3>
<p>ここまで限界を並べたが、タグ分類は無価値ではない。次の使い方では<b>むしろ最適</b>である。</p>
<ul>
<li><b>ヌルモデル（null model／基準線）</b>：「人手タグだけならここまでしか分かれない」を示すことで、
テキスト解析の付加価値を定量化できる。plan02・plan03 はこれを実行した。</li>
<li><b>タグ体系の点検</b>：共起・冗長（メタタグ）・支配タグの“その他”化・孤立タグを、
<b>数字で</b>指摘できる。運用改善（統合・分割・新設）の具体的な出発点になる。</li>
<li><b>三角測量（triangulation）</b>：独立な情報源（タグ と テキスト）が
<b>部分的にでも一致</b>すれば、その構造は本物らしい。第5章の ARI 0.3〜0.4 がまさにこれ。</li>
<li><b>透明性・再現性</b>：タグは人が読める。ブラックボックスの埋め込みと違い、
「なぜこのグループか」を即答できる。対外説明や合意形成の場では、この明快さが埋め込みの精度に勝ることがある。</li>
</ul>

<h3>9.8 本プロジェクトでの正しい位置づけ / The right framing here</h3>
<p>結局のところ、最も健全な設計は<b>役割分担</b>である：</p>
<div class="verdict">
<b>テキスト＝測定量（何が書かれているか）／タグ＝参照ラベル（人がどう分類したか）。</b><br>
価値は<b>タグ単独のクラスタリングそのものではなく、両者の“ズレ”を読むこと</b>にある。<br>
今回それは <b>ARI ≈ 0.3〜0.4</b>（タグはテキスト構造の約1/3を説明する「参照的一致」）として現れた——
<b>焼き直しでも無関係でもない</b>。この“ほどよいズレ”こそが、
「タグはおおむね妥当だが、公正な手続の過剰使用・複数タグの分散など<b>手を入れる余地がある</b>」という
本プロジェクトの中心的な示唆を支えている。
</div>
<p>したがって問いへの最終的な答えは——
<b>タグを主役にした“発見”のための分類は不適切</b>（循環と情報の天井のため）だが、
<b>基準線・体系批評・裏取りのための分類としては適切であり、本 plan03 はその範囲を守っている。</b>
タグは真実ではなく、よくできた（しかし完璧ではない）人手の見立てである。
その見立てを<b>点検し、テキストと突き合わせる道具</b>として使う限り、タグ分類は十分に妥当な選択だと考える。</p>

</div>
"""

HTML = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CALL4 タグ単独分類レポート (plan03)</title>
<style>
:root{{--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--bg:#fff;--accent:#2B6CB0;}}
*{{box-sizing:border-box;}}
body{{font-family:-apple-system,"Hiragino Sans","Yu Gothic",sans-serif;
color:var(--fg);max-width:1080px;margin:0 auto;padding:28px 20px 80px;line-height:1.7;}}
h1{{font-size:1.7rem;border-bottom:3px solid var(--accent);padding-bottom:10px;}}
h2{{font-size:1.28rem;margin-top:2.4em;border-left:5px solid var(--accent);padding-left:10px;}}
h4{{margin:.4em 0;}}
.en{{color:var(--muted);font-weight:400;font-size:.72em;}}
.lead{{background:#f5f8fc;border:1px solid var(--line);border-radius:8px;padding:16px 20px;}}
.warn{{background:#fff8e6;border:1px solid #e6c34c;border-radius:8px;padding:12px 18px;font-size:.92em;}}
figure{{margin:1.2em 0;text-align:center;}} svg{{max-width:100%;height:auto;}}
figcaption{{color:var(--muted);font-size:.84em;margin-top:6px;}}
table{{border-collapse:collapse;width:100%;font-size:.9em;margin:1em 0;}}
th,td{{border:1px solid var(--line);padding:6px 9px;text-align:left;}}
th{{background:#f2f2f2;}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px 14px;}}
.chip{{display:inline-block;color:#fff;font-weight:700;border-radius:5px;padding:1px 9px;font-size:.8em;}}
.card .tags{{font-size:.82em;color:#444;}} .card .meta{{font-size:.8em;color:var(--muted);}}
.cases{{margin:.3em 0 0;padding-left:18px;font-size:.84em;}} .cases .r{{color:var(--muted);}}
.metatags{{list-style:none;padding:0;}} .metatags li{{margin:.35em 0;}}
.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;vertical-align:middle;}}
.tg{{display:inline-block;background:#eef2f7;border:1px solid #d6dee7;border-radius:4px;
padding:1px 7px;margin:2px;font-size:.82em;}}
.big{{font-size:2.1rem;font-weight:800;color:var(--accent);}}
.kpi{{display:flex;gap:24px;flex-wrap:wrap;margin:1em 0;}}
.kpi div{{background:#f5f8fc;border:1px solid var(--line);border-radius:8px;padding:12px 18px;min-width:180px;}}
code{{background:#f2f2f2;padding:1px 5px;border-radius:3px;}}
.essay{{font-size:.98em;}}
.essay h3{{font-size:1.06rem;margin-top:1.7em;color:#243b53;}}
.essay p{{margin:.7em 0;}}
.essay b{{color:#1a2a3a;}}
.gloss{{color:var(--muted);font-size:.9em;}}
.verdict{{background:#eef6ef;border:1px solid #9cc1a4;border-left:6px solid #4C8C2B;
border-radius:8px;padding:16px 20px;margin:1.2em 0;}}
.verdict b{{color:#2f6b34;}}
.tldr{{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin:.8em 0;font-size:.95em;}}
.tldr .g{{font-weight:700;color:#4C8C2B;}} .tldr .n{{font-weight:700;color:#C0392B;}}
.tldr .m{{font-weight:700;color:#C29B2C;}}
</style></head><body>

<h1>CALL4 公共訴訟ケース：既存タグ単独での分類<br>
<span class="en">Classifying cases using existing tags only — plan03</span></h1>

<div class="lead">
<b>これは何か / What.</b> plan02 が<b>テキスト</b>（概要・本文）を特徴量にしたのに対し、本レポートは
<b>既存の主題タグ（{T}種）だけ</b>を特徴量に、同じ95ケースを複数手法で分類した結果である。
目的は新構造の発見ではなく <b>(1) 誠実なベースライン、(2) タグ体系そのものの批評、(3) タグ由来の混合可視化</b>。
<br><span class="en">Using only the {T} human-assigned subject tags (not text) as features, we cluster the
same 95 cases with several categorical methods — to benchmark, to critique the tag system, and to visualize mixture.</span>
</div>

<div class="kpi">
<div><div class="big">95→37</div>95ケースが取りうる<b>ユニークなタグ組合せは37通り</b>だけ<br>
<span class="en">only 37 unique tag combinations</span></div>
<div><div class="big">{ev['resolution_ceiling']['frac_zero_similarity_pairs']:.0%}</div>
タグを1つも共有しないケースペアの割合<br><span class="en">case-pairs sharing zero tags</span></div>
<div><div class="big">{ari_nmf['ARI']:.2f}</div>タグNMFが再現するテキストNMF構造 (ARI)<br>
<span class="en">tag-NMF recovers text-NMF</span></div>
</div>

<div class="warn"><b>情報の天井 / Resolution ceiling.</b>
特徴は{T}タグのみ・平均1.6タグ/ケース。距離ベース手法が区別できる点は最大37。
「公正な手続」は43/95件(45%)に付く<b>背景タグ</b>で、共不在を無視する<b>Jaccard距離</b>を使わないと
大半のペアが似て見える。<b>タグ空間のシルエットはテキスト空間の値と直接比較できない</b>（距離の物差しが違う）。
</div>

<h2>1. データと手法 <span class="en">Data & methods</span></h2>
<p>特徴行列は <code>tags.npz</code>（95×{T}の二値 multi-hot、plan02と共有）。距離は原則 <b>Jaccard</b>。
バイオ対応：これは発現行列ではなく <b>{T}マーカーのフローサイトメトリー・パネル（二値・multi-positive）</b>で、
「表面マーカーだけで細胞を分ける」問題に近い。連続用の手法ではなく二値・カテゴリ専用手法を選んだ。</p>
<figure>{fig_dist()}<figcaption>タグ分布と共起。「公正な手続」が突出し、多くのタグは十数件規模。
Tag frequency and co-occurrence.</figcaption></figure>
<table><tr><th>系統 / family</th><th>手法 / method</th><th>出力 / output</th></tr>
<tr><td>潜在変数モデル</td><td>LCA / ベルヌーイ混合 (EM, BIC)</td><td>ソフト所属＋硬ラベル</td></tr>
<tr><td>行列分解（混合）</td><td>NMF on タグ (K=6)</td><td>タグ・アーキタイプ混合比</td></tr>
<tr><td>カテゴリ次元圧縮</td><td>MCA</td><td>ケース＋タグ同一空間</td></tr>
<tr><td>グラフ</td><td>タグ共起コミュニティ＋ケース類似Leiden</td><td>メタタグ／グラフ分割</td></tr>
<tr><td>距離＋共クラスタ</td><td>Jaccard階層＋スペクトル共クラスタ</td><td>入れ子＋ブロック</td></tr></table>

<h2>2. タグ空間の地図 <span class="en">Map of the tag space</span></h2>
<figure>{fig_mca()}<figcaption>MCAバイプロット。近いタグ＝一緒に付きやすい。点はケース（NMFアーキタイプで着色）。
MCA biplot; nearby tags co-occur.</figcaption></figure>
<figure>{fig_network()}<figcaption>タグ共起ネットワーク。色は共起コミュニティ（メタタグ）、線の太さ＝共起数、丸の大きさ＝件数。
Tag co-occurrence network.</figcaption></figure>

<h2>3. 混合メンバーシップ（タグ由来） <span class="en">Mixture membership from tags</span></h2>
<p>NMFをタグ行列に適用すると、各ケースを<b>{6}個のタグ・アーキタイプの混合比</b>で表せる
（plan02のテキストNMFと同じ土俵で比較可能）。</p>
<figure>{fig_stack()}<figcaption>95ケースのタグ由来混合比。plan02のテキスト版スタックバーの対応物。</figcaption></figure>
<div class="cards">{archetype_cards()}</div>

<h2>4. 手法比較（タグ空間の中で） <span class="en">Method comparison (within tag space)</span></h2>
<p>Jaccard距離上のシルエットと最大クラスタ占有率。<b>距離ベースの階層は「公正な手続」の巨大塊に退化</b>
（最大クラスタが約6割）。混合・グラフ系はよりバランスが良い。
※このシルエットは<b>タグ空間内でのみ</b>比較可能（plan02の埋め込み空間0.12前後とは物差しが違う）。</p>
<table><tr><th>手法 / method</th><th>K</th><th>シルエット(Jaccard)</th><th>最大クラスタ占有</th></tr>
{validity_rows()}</table>

<h2>5. plan02（テキスト）との比較 —— 本トラックの主眼<br>
<span class="en">5. vs plan02 (text-driven) — the main question</span></h2>
<p>「<b>タグだけの分類は、テキスト由来の構造をどれだけ再現するか？</b>」。
最も一致したのは <b>タグNMF ↔ テキストNMF で ARI={ari_nmf['ARI']:.2f}</b>、
テキストLeiden(K=5)に対しては最良でも <b>ARI={ari_leiden['ARI']:.2f}（{ari_leiden['method']}）</b>。</p>
<figure>{fig_ari()}<figcaption>タグ手法×plan02テキスト手法のARI。0.3〜0.4が上限。</figcaption></figure>
<div class="warn"><b>解釈 / Reading.</b> ARI 0.3〜0.4 は「<b>参照的一致</b>」——
タグはテキスト構造の<b>おおよそ1/3</b>を説明するが、焼き直しでも無関係でもない。
plan02でテキストNMFがタグと最整合だった (ARI 0.335) のと整合的で、
<b>タグとテキストは部分的に重なるが、互いを再現しない別々の情報源</b>である。</div>

<h2>6. タグ体系への批評 <span class="en">Critique of the tag system</span></h2>
<h4>6.1 背景タグ・分裂タグ / cross-cutting & splitting tags</h4>
<p>各タグのケースが、plan02のテキスト6トピックにどう散るか。
<b>「公正な手続」は6トピック全てに出現する背景タグ</b>で、実質「その他」化している。
「働き方」「政治参加・表現の自由」等も複数トピックに分散し、単独では分類軸として弱い。</p>
<table><tr><th>タグ / tag</th><th>件数</th><th>最大トピック占有</th><th>到達トピック数</th></tr>
{splitting_rows()}</table>
<h4>6.2 冗長タグ（メタタグ）/ redundant tag groups</h4>
<p>共起コミュニティ検出が束ねた<b>いつも一緒に付くタグ群</b>。統合の候補。</p>
<ul class="metatags">{meta_tag_items()}</ul>
<h4>6.3 ブロック構造 / block structure</h4>
<figure>{fig_cocluster()}<figcaption>スペクトル共クラスタリングによるケース群×タグ群のブロック。</figcaption></figure>

<h2>7. 限界 <span class="en">Limitations</span></h2>
<ul>
<li><b>解像度の天井</b>：{T}タグ・37組合せ。新しい微細構造は原理的に出ない。本分析はベースライン／批評ツール。</li>
<li><b>シルエットの見かけの高さ</b>：離散・重複点が多いためタグ空間のシルエットは高く出る。テキスト空間と比較しない。</li>
<li><b>LCAのBIC最適はK=2</b>：タグ情報はそもそも多クラスを強く支持しない。K=6は比較のため固定した便宜値。</li>
<li><b>リフト命名の偏り</b>：命名に使うタグ濃縮(lift)は希少タグ（沖縄n=4等）を過大評価しやすい。名前は目安。</li>
<li><b>因果・評価ではない</b>：当事者の類型化・評価を意図しない探索的解析。</li>
</ul>

<h2>8. 再現情報 <span class="en">Reproducibility</span></h2>
<p>seed=42固定。特徴：<code>plan02/features/tags.npz</code>（95×{T}）。
手法：自前EM（ベルヌーイ混合）・sklearn NMF/SpectralCoclustering・自前MCA(SVD)・
scipy階層(average, Jaccard)・igraph+leidenalg。スクリプト：<code>plan03/s0–s4</code>。
結果JSON：<code>plan03/results/</code>。生成日：2026-07-22。</p>

{ESSAY}

<p style="color:var(--muted);font-size:.82em;margin-top:2em">
本レポートは探索的データ解析であり、既存タグ体系や当事者の評価を目的としない。
This is exploratory analysis; it is not an evaluation of the tag system or of the parties involved.</p>

</body></html>"""

out = os.path.join(REP, "call4_plan03_report.html")
open(out, "w", encoding="utf-8").write(HTML)
print(f"[s4] wrote {out}  ({len(HTML)/1024:.0f} KB)")
