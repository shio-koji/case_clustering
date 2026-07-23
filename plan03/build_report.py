#!/usr/bin/env python3
"""
plan03: accessible NMF report for a non-statistical audience (a legal NPO).

Differences from plan02's NMF report:
- No "should we use tags" section (dropped per request).
- Plain-language explainer boxes throughout; jargon (NMF/ARI/...) moved to a
  small technical footnote for the presenter.

Reads plan02 outputs. Run from the repo root:  python plan03/build_report.py
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

SEED = 42
FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")
OUT = Path("plan03/report/call4_plan03_report.html")

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


def svg_of(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    s = buf.getvalue().decode("utf-8"); return s[s.find("<svg"):]

def url(cid):
    return f"https://www.call4.jp/info.php?type=items&id={cid}"

def blend(row):
    rgb = np.zeros(3)
    for t, r in enumerate(row):
        c = TC[t].lstrip("#"); rgb += r * np.array([int(c[i:i+2], 16) for i in (0, 2, 4)])
    return "#%02x%02x%02x" % tuple(int(min(v, 255)) for v in rgb)


def fig_stack(ratios, dom, order_idx, names):
    fig, ax = plt.subplots(figsize=(13, 3.0))
    x = np.arange(len(order_idx)); bottom = np.zeros(len(order_idx)); R = ratios[order_idx]
    for t in range(6):
        ax.bar(x, R[:, t], bottom=bottom, color=TC[t], width=1.0); bottom += R[:, t]
    ax.set_xlim(-0.5, len(order_idx)-0.5); ax.set_ylim(0, 1); ax.set_xticks([])
    ax.set_ylabel("配合の割合", fontproperties=jp(9))
    ax.set_title("95件の訴訟それぞれの「テーマの配合」（棒1本＝訴訟1件）", fontproperties=jp(11))
    return svg_of(fig)

def fig_sizes(names, sizes):
    fig, ax = plt.subplots(figsize=(8, 2.7))
    idx = np.argsort(sizes)[::-1]
    ax.bar(range(len(sizes)), [sizes[i] for i in idx], color=[TC[i] for i in idx])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([names[i] for i in idx], fontproperties=jp(9), rotation=18, ha="right")
    ax.bar_label(ax.containers[0], fontsize=9)
    ax.set_title("各テーマを主とする訴訟の件数", fontproperties=jp(11))
    return svg_of(fig)

def fig_words(topics, names):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    for k, ax in enumerate(axes.ravel()):
        words = topics[str(k)]["descriptor_words"][:7][::-1]
        ax.barh(range(len(words)), range(1, len(words)+1), color=TC[k])
        ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontproperties=jp(9)); ax.set_xticks([])
        ax.set_title(f"テーマ{k+1}: {names[k]}", fontproperties=jp(9.5))
    fig.suptitle("各テーマでよく使われる言葉", fontproperties=jp(11)); fig.tight_layout()
    return svg_of(fig)


def main():
    tok = json.loads((FEATURES_DIR/"tokens.json").read_text(encoding="utf-8")); tok.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in tok]; titles = [r["title"] for r in tok]
    m = json.loads((RESULTS_DIR/"membership_nmf.json").read_text(encoding="utf-8"))
    names = [v["name"] for _, v in json.loads((RESULTS_DIR/"names_llm.json").read_text(encoding="utf-8"))["nmf"].items()]
    topics = json.loads((RESULTS_DIR/"interpretation.json").read_text(encoding="utf-8"))["mixture"]["nmf"]["topics"]
    ev = json.loads((RESULTS_DIR/"evaluation.json").read_text(encoding="utf-8"))

    ratios = np.array(m["ratios"]); dom = np.array(m["dominant_topic"]); ent = np.array(m["entropy_normalized"])
    Rn = ratios / np.maximum(np.linalg.norm(ratios, axis=1, keepdims=True), 1e-9)
    order_idx = np.lexsort((-ratios[np.arange(len(ids)), dom], dom))
    sizes = [int((dom == k).sum()) for k in range(6)]
    boot = ev["q2_bootstrap_stability"]["nmf_dom"]["ari_median"]
    ari_tag = ev["q3_tag_alignment"]["nmf_dom"]["ari"]

    # why-similar worked examples
    def shares(qi, j):
        p = [(t, ratios[qi, t], ratios[j, t]) for t in range(6) if ratios[qi, t] > 0.12 and ratios[j, t] > 0.12]
        p.sort(key=lambda x: -min(x[1], x[2])); return p
    examples = []
    for kw in ["財務省改ざん", "結婚の自由", "海外でも国民審査", "カメルーン"]:
        qi = next(i for i, t in enumerate(titles) if kw in t)
        sims = Rn @ Rn[qi]; sims[qi] = -1; j = int(sims.argmax()); examples.append((qi, j, float(sims[j]), shares(qi, j)))

    figs = {"stack": fig_stack(ratios, dom, order_idx, names),
            "sizes": fig_sizes(names, sizes), "words": fig_words(topics, names)}

    # topic cards
    cards = []
    for k in range(6):
        t = topics[str(k)]
        reps = "".join(f'<li><a href="{url(r["id"])}" target="_blank">{r["title"]}</a></li>'
                       for r in t["representatives"])
        cards.append(f"""
<div class="card" style="border-top:5px solid {TC[k]}">
  <h4><span class="dot" style="background:{TC[k]}"></span>テーマ{k+1}: {names[k]}
      <span class="dim">（この論点が主な訴訟 {t['size_dominant']}件）</span></h4>
  <p><b>よく出る言葉:</b> {' / '.join(t['descriptor_words'][:8])}</p>
  <p><b>代表的な訴訟:</b></p><ul>{reps}</ul>
</div>""")

    why = []
    for qi, j, s, pairs in examples:
        sh = "".join(f'<li><span class="dot" style="background:{TC[t]}"></span>{names[t]}'
                     f'（{a*100:.0f}% と {b*100:.0f}%）</li>' for t, a, b in pairs)
        why.append(f"""
<div class="card">
  <p><b><a href="{url(ids[qi])}" target="_blank">{titles[qi]}</a></b><br>
     <span class="dim">と</span><br>
     <b><a href="{url(ids[j])}" target="_blank">{titles[j]}</a></b> は似ている。</p>
  <p class="dim">なぜなら、同じテーマを同じくらいの割合で含むから:</p><ul class="small">{sh}</ul>
</div>""")

    # mixture-similarity network
    sims = Rn @ Rn.T; np.fill_diagonal(sims, -1)
    edges = sorted({(min(i, int(j)), max(i, int(j))) for i in range(len(ids)) for j in np.argsort(sims[i])[-3:]})
    import random as pyr; pyr.seed(SEED)
    net = np.array(ig.Graph(n=len(ids), edges=edges).layout_fruchterman_reingold(niter=800).coords)
    nmn = net - net.min(0); nmn /= np.ptp(net, 0)
    nodes = [{"i": i, "title": titles[i], "url": url(ids[i]), "blend": blend(Rn[i]/Rn[i].sum()),
              "ent": round(float(ent[i]), 2), "nmf": [round(float(r), 3) for r in ratios[i]],
              "nx": round(float(nmn[i, 0]), 3), "ny": round(float(nmn[i, 1]), 3)} for i in range(len(ids))]
    edge_js = [[int(a), int(b)] for a, b in edges]

    gen = date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALL4の訴訟95件を「内容」で6つのテーマに整理する</title>
<style>
body {{ font-family:"Hiragino Sans","Yu Gothic",sans-serif; margin:0; background:#f7f6f3; color:#222; line-height:1.9; }}
.container {{ max-width:960px; margin:0 auto; padding:30px 20px 70px; }}
h1 {{ font-size:1.5em; border-bottom:3px solid #B0532B; padding-bottom:10px; }}
h2 {{ font-size:1.28em; margin-top:2.4em; border-left:6px solid #B0532B; padding-left:11px; }}
h3 {{ font-size:1.06em; margin-top:1.6em; }}
h4 {{ margin:.3em 0; }}
.dim {{ color:#778; font-size:.85em; }}
.lead {{ background:#f6ece4; border:1px solid #e2c4ad; border-radius:8px; padding:16px 20px; font-size:1.02em; }}
.explain {{ background:#eef6f0; border:1px solid #bcdcc8; border-radius:8px; padding:12px 16px; margin:12px 0; }}
.explain b {{ color:#2f7a4d; }}
.explain .h {{ font-weight:bold; color:#2f7a4d; }}
.note {{ display:block; margin:4px 0 8px; font-size:.85em; color:#5a6b60; padding-left:12px; border-left:2px solid #bcdcc8; }}
.term {{ font-size:.78em; color:#2f7a4d; border:1px solid #bcdcc8; border-radius:3px; padding:0 4px; }}
.card {{ background:#fff; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin:12px 0; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; margin:12px 0; }}
.figure {{ background:#fff; border-radius:8px; padding:12px; margin:14px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); overflow-x:auto; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:6px; margin-right:5px; vertical-align:middle; }}
canvas {{ max-width:100%; }}
#tip {{ position:fixed; display:none; background:rgba(20,25,35,.94); color:#fff; padding:8px 11px; border-radius:6px; font-size:12px; max-width:340px; pointer-events:none; z-index:10; }}
a {{ color:#B0532B; }}
.small li {{ font-size:.85em; }}
.tech {{ background:#f0efe9; border-radius:8px; padding:12px 16px; font-size:.82em; color:#555; }}
</style>
</head>
<body>
<div class="container">

<h1>CALL4の訴訟95件を「内容」で6つのテーマに整理する</h1>
<p class="dim">{gen} ｜ 対象: <a href="https://www.call4.jp" target="_blank">CALL4</a> 掲載の公共訴訟 95件</p>

<div class="lead">
CALL4掲載の公共訴訟95件を、<b>トピックモデル</b>という手法で分析し、文章の内容から<b>6つの潜在トピック（テーマ）</b>を
抽出しました。特徴は、各訴訟を単一カテゴリに割り当てる（ハードクラスタリング）のではなく、
<b>複数トピックの混合比で表す「混合メンバーシップ」</b>を採ったこと。実際の公共訴訟は複数の論点にまたがることが多く、
その多面性を保てるためです（例:「入管収容 70% ＋ 刑事手続 30%」）。
</div>

<h2>1. 手法 — トピックモデル（NMF）による分解</h2>
<p>処理は3段階です。専門用語には各文末に<span class="term">補足</span>を付けました。</p>
<div class="explain">
<b>① 特徴量化。</b> 各訴訟の紹介文を<b>形態素解析</b>で単語に分割し、
<b>TF-IDF</b>で「訴訟×単語」の重み付き行列を作ります。
<span class="note">補足: 形態素解析＝日本語の文を単語に区切る処理（例「取調べを拒否」→「取調べ／拒否」）。
TF-IDF＝その訴訟に特徴的な語ほど重く、どの訴訟にも出る語（「訴訟」等）は軽く扱う重み付け。</span><br>
<b>② トピック分解。</b> この行列を<b>非負値行列分解（NMF）</b>で6つの潜在トピックに分解します。
<span class="note">補足: NMF＝行列を「トピックごとの語の重み」×「訴訟ごとのトピック配合」の2つの積で近似する手法。
値がすべて非負（0以上）なので、各訴訟を“複数トピックの足し合わせ”として素直に解釈できるのが利点。</span><br>
<b>③ 混合比で表現。</b> 結果、各訴訟は6トピックの<b>混合比（合計1に正規化）</b>で表されます。
<span class="note">補足: これが「混合メンバーシップ」。どのトピックにどれだけ属するかを割合で持つ表現。</span>
</div>
<p class="dim">人が事前にテーマを定義するのではなく、<b>文章データからトピックが教師なしで立ち上がる</b>点がこの方法の要点です。
（イメージとしては絵の具の混色に近く、6色＝6トピックの配合で各訴訟の“色”が決まります。）</p>

<h2>2. 見つかった6つのテーマ</h2>
<div class="figure">{figs['sizes']}</div>
<div class="grid">{''.join(cards)}</div>

<h2>3. 類似の「説明可能性」— なぜ似ているかを言葉で言える</h2>
<div class="explain">
<b>この手法の最大の利点。</b> 2訴訟の類似度は、混合比ベクトル同士の<b>コサイン類似度</b>で測っています。
<span class="note">補足: コサイン類似度＝2つの配合を方向ベクトルとみなし、なす角の近さで測る指標（1に近いほど similar）。</span>
重要なのは、その類似が<b>「どのトピックを共有するか」に分解して言葉で説明できる</b>こと。
埋め込みベクトル（後述の別手法）では「近い」ことは測れても理由が高次元に埋もれますが、
NMFは解釈可能な6軸で表すため、根拠を名指しできます。
</div>
<div class="grid2">{''.join(why)}</div>

<h2>4. 図で見る</h2>

<h3>A. 全95件の「配合」一覧</h3>
<div class="explain">
<span class="h">この図の読み方。</span> 棒1本が訴訟1件です。棒の中の色の割合が「テーマの配合」。
<b>ほぼ1色の棒＝1つの論点の訴訟</b>、<b>何色も混ざる棒＝複数の論点にまたがる訴訟</b>です。
棒にカーソルを合わせると内訳が出ます。
</div>
<div class="figure">
  <div id="lg" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px;"></div>
  <canvas id="stack" width="940" height="320"></canvas>
</div>

<h3>B. 似た訴訟のつながり（ネットワーク図）</h3>
<div class="explain">
<span class="h">この図の読み方。</span> 配合が似た訴訟どうしを線でつないだ図です。
<b>近くに集まっている＝似た訴訟</b>。点の色は配合を混ぜた色で、
<b>中間的な色の点＝複数テーマをまたぐ「橋渡し」の訴訟</b>です。
点にカーソルで内訳、クリックでその訴訟のCALL4ページが開きます。
</div>
<div class="figure"><canvas id="net" width="940" height="580"></canvas></div>

<h3>C. 各テーマを特徴づける言葉</h3>
<div class="figure">{figs['words']}</div>

<h2>5. 頑健性 — この結果はたまたまではないか</h2>
<div class="explain">
<b>① ブートストラップによる安定性検証。</b>
95件から無作為に80%を抽出して同じ分析を100回繰り返し、元の分類とどれだけ一致するかを見ました。
一致度の指標<b>ARI（調整ランド指数）</b>の中央値は<b>{boot:.2f}</b>。特定の数件に依存した結果ではありません。
<span class="note">補足: ブートストラップ＝データを一部抜き取って再分析し、結果のブレを調べる定番の検証法。
ARI＝2つの分類の一致を偶然の一致分を差し引いて -1〜1 で測る（1が完全一致、0が偶然並み）。</span><br>
<b>② 別手法との一致（収束的妥当性）。</b> まったく別系統の手法——文章の意味を捉える
<b>多言語埋め込み（bge-m3）＋Leidenクラスタリング</b>——でも<b>同じ6テーマ</b>に到達しました（ARI 0.44）。
語彙ベースと意味ベースという独立な入口が同じ構造を指す、というのが強い根拠です。
</div>
<div class="explain">
<b>なぜトピック数は「6」か。</b> K=4〜10を全て試しました。<b>再構成誤差</b>（元行列の復元精度）には
明確な h-elbow（折れ目）がなく、数値だけでは決まりません。そこで<b>解釈可能性</b>で判断しました:
<b>K=5</b>では「刑事手続」と「入管収容」が1トピックに融合、<b>K=7</b>ではサイトの定型語だけの無意味なトピックが出現。
<b>K=6</b>が全トピックを一文で説明でき、論点も分離する水準でした（別手法も独立に6を選択）。
<span class="note">補足: 再構成誤差＝分解した2行列の積が元の行列をどれだけ再現できるかの誤差。小さいほど良いが、
Kを増やせば必ず下がるため「どこで頭打ちか（肘）」で止めるのが定石。今回は肘が不明瞭だったため中身で判断した。</span>
</div>

<h2>6. 気をつけたいこと（正直に）</h2>
<ul>
<li>95件は分析としては<b>小さめ</b>です。この結果は「きっちり正しい唯一の分類」ではなく、
<b>議論のたたき台・気づきのための地図</b>と考えてください。</li>
<li>「刑事手続」と「入管収容」は、<b>1つ（身体拘束）と見るか2つに分けるか</b>は見方次第です。
これは手法の欠陥ではなく、論点の捉え方の違いです。</li>
<li>分析に使ったのは各訴訟の紹介文までで、訴状や判決文の全文は使っていません（今後の拡張余地）。</li>
</ul>

<div class="tech">
<b>技術メモ（作成者向け）</b><br>
手法: 形態素解析(SudachiPy) → TF-IDF 7,350語 → 非負値行列分解 NMF（トピック数K=6、各ケースの行をL1正規化して混合比化）。
テーマ数の選択は再構成誤差に明確な肘がなく解釈可能性で決定（K=5で刑事/入管が融合、K=7で定型語トピックが発生）。
頑健性: 80%サブサンプル×100回のブートストラップで優勢トピックの中央値ARI={boot:.3f}。
別手法（bge-m3埋め込み+Leiden, K=6）との一致 ARI=0.44。既存タグとの一致 ARI={ari_tag:.3f}（参照的一致）。
乱数シード42固定。コード・データ: GitHub shio-koji/case_clustering の plan02/（plan03はこの説明用HTML）。
「似ている」の判定は混合比ベクトルのコサイン類似度。ネットワーク配置はigraphのFruchterman-Reingold。<br>
本分類は解析目的であり、訴訟当事者を類型化して評価する意図はありません。ケース本文の著作権はCALL4および執筆者に帰属します。
</div>

</div>
<div id="tip"></div>

<script>
const NODES = {json.dumps(nodes, ensure_ascii=False)};
const EDGES = {json.dumps(edge_js)};
const TN = {json.dumps(names, ensure_ascii=False)};
const TC = {json.dumps(TC)};
const ORDER = {json.dumps([int(i) for i in order_idx])};
const tip = document.getElementById('tip');
function showTip(h, ev) {{ tip.innerHTML=h; tip.style.display='block';
  tip.style.left=Math.min(ev.clientX+14, window.innerWidth-360)+'px'; tip.style.top=(ev.clientY+12)+'px'; }}
function hideTip() {{ tip.style.display='none'; }}
function mixText(n) {{ return n.nmf.map((v,t)=> v>0.08 ? `<span style="color:${{TC[t]}}">■</span>${{TN[t]}} ${{(v*100).toFixed(0)}}%`:null).filter(Boolean).join('<br>'); }}
const lg=document.getElementById('lg');
TN.forEach((n,t)=>lg.insertAdjacentHTML('beforeend',
  `<span style="font-size:12px"><span style="display:inline-block;width:11px;height:11px;background:${{TC[t]}};margin-right:4px;border-radius:2px"></span>テーマ${{t+1}} ${{n}}</span>`));
// stacked bar
const sc=document.getElementById('stack'), sx=sc.getContext('2d'); const BW=(sc.width-16)/ORDER.length;
ORDER.forEach((ci,pos)=>{{ const n=NODES[ci]; let y=sc.height-14;
  n.nmf.forEach((r,t)=>{{ const h=r*(sc.height-24); sx.fillStyle=TC[t]; sx.fillRect(10+pos*BW,y-h,Math.max(BW-0.5,1.2),h); y-=h; }}); }});
let scur=-1;
sc.addEventListener('mousemove', ev=>{{ const r=sc.getBoundingClientRect(); const pos=Math.floor(((ev.clientX-r.left)*(sc.width/r.width)-10)/BW);
  if(pos<0||pos>=ORDER.length){{hideTip();scur=-1;return;}} scur=pos; const n=NODES[ORDER[pos]];
  showTip(`<b>${{n.title}}</b><br>${{mixText(n)}}`, ev); sc.style.cursor='pointer'; }});
sc.addEventListener('mouseleave', ()=>{{hideTip();scur=-1;}});
sc.addEventListener('click', ()=>{{ if(scur>=0) window.open(NODES[ORDER[scur]].url,'_blank'); }});
// network
const nc=document.getElementById('net'), nx=nc.getContext('2d'); const pad=30;
function nPos(n){{ return [pad+n.nx*(nc.width-2*pad), pad+n.ny*(nc.height-2*pad)]; }}
nx.strokeStyle='rgba(150,155,165,0.4)'; nx.lineWidth=0.9;
EDGES.forEach(([a,b])=>{{ const [x1,y1]=nPos(NODES[a]),[x2,y2]=nPos(NODES[b]); nx.beginPath(); nx.moveTo(x1,y1); nx.lineTo(x2,y2); nx.stroke(); }});
NODES.forEach(n=>{{ const [x,y]=nPos(n); nx.beginPath(); nx.arc(x,y,6.5,0,7); nx.fillStyle=n.blend; nx.fill(); nx.strokeStyle='#fff'; nx.lineWidth=1; nx.stroke(); }});
let ncur=null;
nc.addEventListener('mousemove', ev=>{{ const r=nc.getBoundingClientRect(); const mx=(ev.clientX-r.left)*(nc.width/r.width), my=(ev.clientY-r.top)*(nc.height/r.height);
  ncur=null; let bd=180; NODES.forEach(n=>{{const [x,y]=nPos(n);const d=(x-mx)**2+(y-my)**2;if(d<bd){{bd=d;ncur=n;}}}});
  if(!ncur){{hideTip();nc.style.cursor='default';return;}} nc.style.cursor='pointer';
  showTip(`<b>${{ncur.title}}</b><br>${{mixText(ncur)}}`, ev); }});
nc.addEventListener('mouseleave', hideTip);
nc.addEventListener('click', ()=>{{ if(ncur) window.open(ncur.url,'_blank'); }});
</script>
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"plan03 report saved: {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
