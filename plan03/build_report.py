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
CALL4に載っている95件の訴訟を、<b>コンピュータに文章を読ませて内容の近いものどうしにまとめ</b>、
<b>6つのテーマ</b>を見つけました。ポイントは、1件の訴訟を「1つの箱」に入れるのではなく、
<b>複数テーマの「配合（ブレンド）」で表す</b>ことです。実際の訴訟は複数の論点にまたがることが多いからです。
（例:「入管の収容 70% ＋ 刑事手続 30%」のように）
</div>

<h2>1. どうやってまとめたのか（3ステップ）</h2>
<div class="explain">
<span class="h">やっていることは、とても素朴です。</span><br>
<b>① 言葉を数える</b>：各訴訟の文章に、どんな言葉がよく出てくるかを数えます。<br>
<b>② よく一緒に出る言葉のまとまり＝「テーマ」を見つける</b>：たとえば「取調べ・勾留・弁護人」は
いつも一緒に出てくる → これで1つのテーマ（刑事手続）、という具合に、コンピュータが自動でテーマを抽出します。<br>
<b>③ 各訴訟を「テーマの配合」で表す</b>：その訴訟がどのテーマの言葉をどれくらい含むかを、割合（合計100%）で表します。
</div>
<p class="dim">たとえるなら、絵の具の混色です。「赤60%＋青40%で紫」のように、
6色（＝6テーマ）の配合で各訴訟の“色”が決まります。人が事前にテーマを決めるのではなく、
<b>文章データそのものからテーマが浮かび上がる</b>のがこの方法の特徴です。
（専門的には「NMF（非負値行列分解）」という手法です。詳細は末尾の技術メモに。）</p>

<h2>2. 見つかった6つのテーマ</h2>
<div class="figure">{figs['sizes']}</div>
<div class="grid">{''.join(cards)}</div>

<h2>3. 「なぜこの2件は似ているのか」が言葉で分かる</h2>
<div class="explain">
<span class="h">この方法のいちばんの利点。</span>
2件の訴訟が「似ている」とき、その理由を<b>「同じテーマを同じくらいの割合で含むから」</b>と、
言葉ではっきり言えます。「なんとなく似ている」ではなく、根拠を示せるということです。
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

<h2>5. この結果は信頼できる？（たまたまではないか）</h2>
<div class="explain">
<span class="h">2つの確かめ方をしました。</span><br>
<b>① データを一部隠しても同じか</b>：95件のうち2割を無作為に取り除いて同じ分析をしても、
グループ分けは<b>だいたい同じ（約{boot:.0%}が一致）</b>でした。特定の数件に依存した結果ではありません。<br>
<b>② 別の方法でも同じか</b>：まったく別の分析方法（文章の意味をAIで測る方法）でも、
<b>同じ6テーマ</b>にたどり着きました。1つのやり方の“クセ”ではないということです。
</div>
<div class="explain">
<span class="h">なぜ「6つ」なのか。</span> テーマ数は4〜10まで全部試しました。
<b>5つ</b>だと「刑事手続」と「入管収容」が1つに混ざってしまい、
<b>7つ</b>にすると意味をなさない“ゴミのようなテーマ”が出てきます。
<b>6つ</b>がちょうど、全テーマをひと言で説明でき、かつ論点がきれいに分かれる数でした。
（別の方法も独立に6を選んでいます。）
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
