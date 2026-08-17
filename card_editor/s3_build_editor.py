#!/usr/bin/env python3
"""card_editor/s3 — ノード配置エディタ（単一HTML）を書き出す。

出力は out/editor.html の1ファイルだけ。カード画像はbase64で埋め込むので、
相手はダブルクリックしてブラウザで開くだけで使える（サーバ・解凍・インストール不要）。
data: URI なのでCanvasが汚染されず、PNG書き出しも動く。

座標系は最初から mm。紙は A0×2（既定 1682x1189mm）で、
A0×2縦（841x2378。914mmロールなら継ぎ目なしで刷れる可能性がある形）と
A0単票にも切り替えられる。

叩き台のレイアウトは igraph で作ってから正規化して埋め込む。
エディタ側は「紙の使える範囲」に写して使う。
"""
import base64

import json
import math
import os
import random

import igraph as ig
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# ---- 紙 -------------------------------------------------------------------
# seam: 継ぎ目の位置。'v' は x=値 の縦線、'h' は y=値 の横線、None は継ぎ目なし
SHEETS = [
    {"key": "A0x2H", "label": "A0×2 横並び 1682×1189",
     "w": 1682, "h": 1189, "seam": ["v", 841],
     "note": "A0を2枚横に並べる。継ぎ目は中央の縦線。"},
    {"key": "A0x2V", "label": "A0×2 縦長 841×2378",
     "w": 841, "h": 2378, "seam": ["h", 1189],
     "note": "914mm幅ロールなら継ぎ目なしで1枚に刷れる可能性がある形。印刷所に確認を。"},
    {"key": "A0", "label": "A0 単票 841×1189",
     "w": 841, "h": 1189, "seam": None,
     "note": "88枚を載せるとカードは53〜58mm角が上限。文字が小さくなる。"},
]
MARGIN_MM = 25.0        # 印刷余白（この内側に置く）

# ---- GUI用タグ配色 -------------------------------------------------------
# 参照色見本の「濃色 + 淡色」のペア。
# 個人情報・プライバシーと情報公開は、指定により色見本と入れ替える。
# このパレットはGUIのタグノードにだけ使い、カード画像の色は変えない。
GUI_TAG_PALETTE = {
    "個人情報・プライバシー": ("#22504e", "#bac9c8"),
    "医療・福祉・障がい": ("#fe7389", "#ffdbe0"),
    "ジェンダー・セクシュアリティ": ("#ff9423", "#ffdccb"),
    "刑事司法": ("#9f6e34", "#e2d3c2"),
    "環境・災害": ("#2e9d7e", "#b6ddd2"),
    "働き方": ("#99b73d", "#e0e9c5"),
    "公正な手続": ("#033064", "#b3c1d0"),
    "沖縄": ("#3970cb", "#bacded"),
    "外国にルーツを持つ人々": ("#6b5498", "#d4cde1"),
    "政治参加・表現の自由": ("#3daac8", "#c5e5ee"),
    "情報公開": ("#ff4709", "#ffcfbf"),
}


def load():
    cards = json.load(open(os.path.join(OUT, "cards.json"), encoding="utf-8"))
    geo = json.load(open(os.path.join(OUT, "card_geometry.json"), encoding="utf-8"))
    for c in cards["cases"]:
        p = os.path.join(OUT, "cards_preview", f"{c['id']}.webp")
        c["img"] = "data:image/webp;base64," + \
            base64.b64encode(open(p, "rb").read()).decode("ascii")
    return cards, geo


def layouts(cases, tags):
    """叩き台レイアウトを [0,1]^2 に正規化して返す。
    ノード番号は 0..T-1 がタグ、T.. がケース（cases の順）。"""
    T, m = len(tags), len(cases)
    tidx = {t: j for j, t in enumerate(tags)}
    edges, wts = [], []
    for k, c in enumerate(cases):
        for t in c["tags"]:
            edges.append((tidx[t["t"]], T + k))
            wts.append(t["v"])
    g = ig.Graph(n=T + m, edges=edges)
    r = random.Random(42)
    try:
        ig.set_random_number_generator(r)
    except Exception:
        pass

    P = {}
    P["force"] = np.array(g.layout_fruchterman_reingold(niter=2000, weights=wts).coords)
    try:
        P["kamada"] = np.array(
            g.layout_kamada_kawai(weights=[max(0.12, 1.0 - 0.85 * w) for w in wts]).coords)
    except Exception:
        P["kamada"] = P["force"]

    # 放射状：タグを件数順に円周に置き、ケースは自分のタグの重心へ
    tcount = [sum(1 for c in cases if any(x["t"] == tags[j] for x in c["tags"]))
              for j in range(T)]
    torder = list(np.argsort(-np.array(tcount)))
    tagpos = np.zeros((T, 2))
    for k, j in enumerate(torder):
        a = 2 * math.pi * k / T - math.pi / 2
        tagpos[j] = [math.cos(a), math.sin(a)]
    rad = np.zeros((T + m, 2))
    rad[:T] = tagpos * 1.18
    # タグが1つだけのケース（88件中44件）は重心が完全に一致してしまう。
    # 同じ点に固まったまま渡すと、あとの重なり解消が対称な行き詰まりに入って
    # ほどけない（反復を3倍にしても10組残る）。優勢タグごとの通し番号で
    # 渦巻き状にずらしてから渡す。
    seq = {}
    for k, c in enumerate(cases):
        bc = np.zeros(2)
        for t in c["tags"]:
            bc += t["v"] * tagpos[tidx[t["t"]]]
        dom = max(c["tags"], key=lambda x: x["v"])["t"]
        i = seq.get(dom, 0)
        seq[dom] = i + 1
        rr = 0.04 + 0.045 * math.sqrt(i)
        ang = i * 2.399963229                # 黄金角
        rad[T + k] = bc * 0.80 + rr * np.array([math.cos(ang), math.sin(ang)])
    P["radial"] = rad

    # 房状：優勢タグごとに固まりを作り、固まりの中は同心円で詰める
    grp = {}
    for k, c in enumerate(cases):
        grp.setdefault(max(c["tags"], key=lambda x: x["v"])["t"], []).append(k)
    clu = np.zeros((T + m, 2))
    # 環の半径は、房の中の詰まり具合（最大 0.10+0.055*sqrt(n)）より
    # 隣の房との間隔が広くなるように取る。近すぎると房同士が食い合う。
    ring = 1.45
    for gi, j in enumerate(torder):
        a = 2 * math.pi * gi / T - math.pi / 2
        ctr = np.array([math.cos(a), math.sin(a)]) * ring
        clu[j] = ctr
        mem = grp.get(tags[j], [])
        for i, k in enumerate(mem):
            rr = 0.10 + 0.055 * math.sqrt(i)
            aa = i * 2.399963229
            clu[T + k] = ctr + rr * np.array([math.cos(aa), math.sin(aa)])
    P["cluster"] = clu

    # 格子：優勢タグ順に並べるだけ。重なりゼロが保証される出発点
    grid = np.zeros((T + m, 2))
    cols = max(1, int(round(math.sqrt(m * 1.4))))
    order = sorted(range(m), key=lambda k: (
        torder.index(tidx[max(cases[k]["tags"], key=lambda x: x["v"])["t"]]),
        -max(x["v"] for x in cases[k]["tags"])))
    for rank, k in enumerate(order):
        grid[T + k] = [rank % cols, rank // cols]
    grid[:, 1] *= 1.0
    for gi, j in enumerate(torder):
        grid[j] = [cols + 1.6, gi * (max(1, (m // cols)) / max(T - 1, 1))]
    P["grid"] = grid

    # 格子だけはマスが正方形でないと意味がないのでアスペクトを保つ。
    # ほかは紙の形いっぱいに伸ばしてよい（グラフのレイアウトに固有の縦横比は無いため）。
    KEEP = {"grid"}

    out = {}
    for name, A in P.items():
        A = np.array(A, dtype=float)
        lo, hi = A.min(axis=0), A.max(axis=0)
        if name in KEEP:
            span = max(float((hi - lo).max()), 1e-9)
            N = (A - lo) / span
            N += (1.0 - (hi - lo) / span) / 2.0      # 正方形の中で中央寄せ
        else:
            span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
            N = (A - lo) / span
        out[name] = {
            "aspect": "keep" if name in KEEP else "fill",
            "tags": [[round(float(N[j, 0]), 5), round(float(N[j, 1]), 5)] for j in range(T)],
            "cases": [[round(float(N[T + k, 0]), 5), round(float(N[T + k, 1]), 5)]
                      for k in range(m)]}
    return out


TEMPLATE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<!-- GitHub Pages はプライベートリポジトリでも配信ファイルが公開になるので、
     少なくとも検索に載らないようにしておく（権利の確認が済むまでの措置） -->
<meta name="robots" content="noindex,nofollow,noarchive">
<title>CALL4 ケースマップ 配置エディタ</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:13px/1.6 -apple-system,"Hiragino Sans","Yu Gothic",sans-serif;
color:#1a1a1a;background:#eceff1;height:100vh;display:flex;flex-direction:column;
overflow:hidden}
#warn{background:#fff4e5;border-bottom:2px solid #e8a33d;padding:9px 14px;font-size:12.2px}
#warn h3{margin:0 0 4px;font-size:12.5px;color:#8a4b00;display:flex;gap:8px;
align-items:center;cursor:pointer;user-select:none}
#warn ul{margin:2px 0 0 18px;padding:0}
#warn li{margin:1px 0;color:#5d4200}
#warn li b{color:#b32d00}
#warn.closed ul,#warn.closed .wsub{display:none}
.wsub{color:#7a5a20;margin:5px 0 0 2px;font-size:11.5px}
#bar{background:#37474f;color:#fff;padding:6px 10px;display:flex;flex-wrap:wrap;
gap:6px 12px;align-items:center;font-size:12px}
#bar .grp{display:flex;gap:5px;align-items:center;padding-right:12px;
border-right:1px solid rgba(255,255,255,.18)}
#bar .grp:last-child{border:0}
#bar label{color:#cfd8dc;font-size:11.5px}
button{font:inherit;font-size:11.5px;padding:3px 9px;border:1px solid #78909c;
background:#546e7a;color:#fff;border-radius:4px;cursor:pointer}
button:hover{background:#607d8b}
button.on{background:#ffb300;color:#263238;border-color:#ffca28;font-weight:bold}
button.warnbtn{background:#c62828;border-color:#ef5350}
select,input[type=number]{font:inherit;font-size:11.5px;padding:2px 4px;border-radius:4px;
border:1px solid #90a4ae;background:#fff;color:#1a1a1a}
input[type=range]{width:120px;vertical-align:middle}
#main{flex:1;display:flex;min-height:0}
#cvwrap{flex:1;position:relative;min-width:0;background:#cfd8dc}
canvas{display:block;width:100%;height:100%}
#side{width:290px;background:#fafafa;border-left:1px solid #cfd8dc;overflow-y:auto;
padding:10px 12px;font-size:12px}
#side h4{margin:12px 0 5px;font-size:12px;color:#37474f;border-bottom:1px solid #cfd8dc;
padding-bottom:3px}
#side h4:first-child{margin-top:0}
.chk{display:flex;flex-wrap:wrap;gap:2px 10px}
.chk label{display:flex;gap:4px;align-items:center;color:#37474f;font-size:11.5px}
#live div{margin:2px 0}
#live .bad{color:#b32d00;font-weight:bold}
#live .ok{color:#2e7d32}
#live .num{font-variant-numeric:tabular-nums}
#sel{font-size:11.5px;color:#455a64}
#sel img{width:100%;border:1px solid #cfd8dc;margin:4px 0}
#status{background:#263238;color:#b0bec5;padding:3px 10px;font-size:11px;
display:flex;gap:16px;font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:rgba(38,50,56,.96);color:#fff;
padding:6px 9px;border-radius:5px;font-size:11.5px;max-width:280px;display:none;z-index:9}
.hint{color:#78909c;font-size:11px;line-height:1.5}
/* テキストで受け渡すためのパネル。ブラウザに埋め込まれた状態だと
   ダウンロードとファイル選択が塞がれることがあるため、その逃げ道 */
#ov{display:none;position:fixed;inset:0;background:rgba(38,50,56,.55);z-index:20;
align-items:center;justify-content:center}
.ovbox{background:#fff;border-radius:8px;padding:18px 20px;width:min(760px,92vw);
max-height:86vh;display:flex;flex-direction:column;gap:9px;
box-shadow:0 12px 40px rgba(20,30,50,.35)}
.ovbox h3{margin:0;font-size:14px;color:#263238}
.ovbox p{margin:0;font-size:12px;color:#546e7a}
#ovta{flex:1;min-height:240px;font:11px/1.45 ui-monospace,Menlo,monospace;
border:1px solid #b0bec5;border-radius:5px;padding:8px;resize:vertical;
color:#263238;background:#fafafa}
.ovbtns{display:flex;gap:8px;align-items:center}
.ovbtns button{background:#37474f}
.ovbtns button:hover{background:#546e7a}
#ovmsg{font-size:11.5px;color:#2e7d32}
</style></head><body>

<div id="warn">
<h3><span id="wtog">▼</span>この図を紙にするまでに未解決の点（クリックで折りたたみ）</h3>
<ul>
<li><b>タグの割合は暫定値です。</b>タグ確認（フェーズ1）が終わると各カードの帯が変わります。
位置はケースIDで保存するので配置作業は無駄になりませんが、カード画像は作り直しになります。</li>
<li><b>タグ11色は画面用のRGB色です。</b>CMYK印刷では色味や色の差が変わるため、
画面で見分けられたタグが紙では混ざります。
入稿前に印刷用の色を選び直し、実機で色校正を1回とる必要があります。</li>
<li><b>入稿用PDFはまだありません。</b>この画面はCanvasなので、そのまま拡大すると
文字がボケます。本番は文字と線をベクタで組み直し、写真だけを埋め込みます
（<code>card_geometry.json</code> に同じ寸法を残してあります）。</li>
<li><b>実寸の試し刷りをまだしていません。</b><code>out/proof_A4.pdf</code> をA4等倍で刷って、
カードの文字が読めるサイズを手元で確かめてください。ここの数値はその後に決まります。</li>
<li><b>継ぎ目なしで刷れるか未確認です。</b>「A0×2 縦長」(841×2378mm) は914mm幅ロールに
収まるので1枚で刷れる可能性があります。刷れるなら継ぎ目の制約が全部消えます。</li>
<li><b>画像の権利。</b>CALL4掲載のサムネイルを含む図です。公開物にするなら
CALL4側の了解が前提になります。</li>
</ul>
<div class="wsub">下の「点検」欄は、いまの配置に対する自動チェックです（重なり・紙外・継ぎ目・解像度・文字サイズ）。</div>
</div>

<div id="bar">
<div class="grp">
<label>紙</label><select id="sheet"></select>
</div>
<div class="grp">
<label>叩き台</label>
<button data-lay="cluster">房状</button>
<button data-lay="radial">放射状</button>
<button data-lay="kamada">Kamada</button>
<button data-lay="force">力学</button>
<button data-lay="grid">格子</button>
</div>
<div class="grp">
<label>カード</label><input type="range" id="csz" min="20" max="140" step="1">
<input type="number" id="cszn" min="20" max="140" step="1" style="width:56px">mm
<label><input type="checkbox" id="selonly">選択のみ</label>
</div>
<div class="grp">
<label>タグ文字</label><input type="range" id="tsz" min="10" max="90" step="1" style="width:80px">
<span id="tszn" style="color:#cfd8dc"></span>pt
</div>
<div class="grp">
<button id="relax">整える（力学）</button><button id="stop">停止</button>
<button id="spread">重なりだけ解く</button>
</div>
<div class="grp">
<button id="pin">ピン留め</button><button id="unpin">解除</button>
<button id="selall">全選択</button><button id="selnone">選択解除</button>
</div>
<div class="grp">
<button id="undo">↶</button><button id="redo">↷</button>
</div>
<div class="grp">
<button id="save">保存(JSON)</button>
<button id="load">読込</button>
<button id="text">テキストで受け渡し</button>
<button id="png">PNG書き出し</button>
<input type="file" id="file" accept=".json" style="display:none">
</div>
</div>

<div id="ov">
<div class="ovbox">
<h3>テキストで受け渡し</h3>
<p id="ovnote"></p>
<textarea id="ovta" spellcheck="false"></textarea>
<div class="ovbtns">
<button id="ovcopy">全部コピー</button>
<button id="ovload">この内容を読み込む</button>
<button id="ovclose">閉じる</button>
<span id="ovmsg"></span>
</div>
</div>
</div>

<div id="main">
<div id="cvwrap"><canvas id="cv"></canvas></div>
<div id="side">
<h4>点検（いまの配置）</h4>
<div id="live"></div>
<h4>表示</h4>
<div class="chk">
<label><input type="checkbox" id="o_img" checked>カード画像</label>
<label><input type="checkbox" id="o_edge" checked>エッジ</label>
<label><input type="checkbox" id="o_arch" checked>アーカイブ</label>
<label><input type="checkbox" id="o_tag" checked>タグ</label>
<label><input type="checkbox" id="o_grid">グリッド</label>
<label><input type="checkbox" id="o_seam" checked>継ぎ目</label>
<label><input type="checkbox" id="o_marg" checked>余白線</label>
<label><input type="checkbox" id="o_snap">5mm吸着</label>
<label><input type="checkbox" id="o_over" checked>重なりを赤枠</label>
<label><input type="checkbox" id="o_num">番号</label>
</div>
<h4>選択中</h4>
<div id="sel">なし</div>
<h4>操作</h4>
<div class="hint">
ドラッグ＝移動／Shift+クリック＝追加選択／背景ドラッグ＝範囲選択<br>
何もない所をドラッグ（またはスペース＋ドラッグ）＝画面移動<br>
ホイール＝拡大縮小　Cmd+Z / Shift+Cmd+Z＝取り消し<br>
P＝ピン留め切替　F＝紙全体を表示　Delete＝（削除はできません）<br>
1枚だけ選ぶと右下に取っ手が出てサイズを変えられます
</div>
</div>
</div>
<div id="status">
<span id="st1"></span><span id="st2"></span><span id="st3"></span><span id="st4"></span>
</div>
<div class="tip" id="tip"></div>

<script>
const SHEETS=@@SHEETS@@, MARGIN=@@MARGIN@@;
const CASES=@@CASES@@, TAGS=@@TAGS@@, EDGES=@@EDGES@@, LAYOUTS=@@LAYOUTS@@;
const MASTER_MM=@@MASTER_MM@@, GEO=@@GEO@@;
const PT=25.4/72;

// ---- 状態 ---------------------------------------------------------------
let sheet=SHEETS[0], LAY='cluster';
let cardMM=75, tagPT=30;
const cpos=CASES.map(()=>({x:0,y:0,s:75,pin:false}));   // カード中心(mm)
const tpos=TAGS.map(()=>({x:0,y:0,pin:false}));
let sel=new Set(), selKind='case';
const view={k:0.3,px:0,py:0};
const opts={};
['img','edge','arch','tag','grid','seam','marg','snap','over','num'].forEach(n=>{
  const el=document.getElementById('o_'+n);
  opts[n]=el.checked;
  el.addEventListener('change',()=>{opts[n]=el.checked; draw(); check();});
});

const IMG=CASES.map(c=>{const i=new Image(); i.src=c.img; return i;});
let imgReady=0;
IMG.forEach(i=>i.onload=()=>{if(++imgReady===IMG.length) draw();});

const cv=document.getElementById('cv');
// cx は let にしてある。PNG書き出しのときに一時的に別Canvasへ差し替えて
// draw() をそのまま再利用するため（描画コードを二重に持たない）。
let cx=cv.getContext('2d');
const tip=document.getElementById('tip');

// ---- 座標変換 -----------------------------------------------------------
const SX=mm=>(mm-view.px)*view.k, SY=mm=>(mm-view.py)*view.k;
const MX=px=>px/view.k+view.px, MY=px=>px/view.k+view.py;

function resize(){
  const r=cv.parentNode.getBoundingClientRect(), d=window.devicePixelRatio||1;
  cv.width=Math.round(r.width*d); cv.height=Math.round(r.height*d);
  cx.setTransform(d,0,0,d,0,0);
  CW=r.width; CH=r.height;
}
let CW=0,CH=0;
function fit(){
  const k=Math.min(CW/sheet.w,CH/sheet.h)*0.94;
  view.k=k;
  view.px=sheet.w/2-CW/(2*k); view.py=sheet.h/2-CH/(2*k);
}

// ---- 叩き台の配置 -------------------------------------------------------
// 位置だけ書き換える。カード個別のサイズ（1枚だけ大きくした等）は保つ。
function applyLayout(name,resetSize){
  LAY=name;
  const L=LAYOUTS[name];
  const pad=MARGIN+cardMM/2;
  const w=Math.max(1,sheet.w-2*pad), h=Math.max(1,sheet.h-2*pad);
  let sx,sy,ox,oy;
  if(L.aspect==='keep'){                 // 格子：マスを正方形に保って中央寄せ
    sx=sy=Math.min(w,h); ox=pad+(w-sx)/2; oy=pad+(h-sy)/2;
  }else{                                 // ほかは紙いっぱいに伸ばす
    sx=w; sy=h; ox=pad; oy=pad;
  }
  L.cases.forEach((p,i)=>{
    cpos[i].x=ox+p[0]*sx; cpos[i].y=oy+p[1]*sy;
    if(resetSize) cpos[i].s=cardMM;
  });
  L.tags.forEach((p,j)=>{tpos[j].x=ox+p[0]*sx; tpos[j].y=oy+p[1]*sy;});
  document.querySelectorAll('[data-lay]').forEach(b=>
    b.classList.toggle('on',b.dataset.lay===name));
}

// ---- 履歴 ---------------------------------------------------------------
const hist=[]; let hi=-1;
function snap(){
  const s=JSON.stringify({c:cpos,t:tpos,cardMM,tagPT});
  if(hi>=0&&hist[hi]===s) return;
  hist.splice(hi+1); hist.push(s); if(hist.length>80) hist.shift(); hi=hist.length-1;
}
function restore(s){
  const o=JSON.parse(s);
  o.c.forEach((p,i)=>Object.assign(cpos[i],p));
  o.t.forEach((p,j)=>Object.assign(tpos[j],p));
  cardMM=o.cardMM; tagPT=o.tagPT; syncInputs(); draw(); check();
}
document.getElementById('undo').onclick=()=>{if(hi>0) restore(hist[--hi]);};
document.getElementById('redo').onclick=()=>{if(hi<hist.length-1) restore(hist[++hi]);};

// ---- タグラベルの寸法（mm） ---------------------------------------------
function tagBox(j){
  const t=TAGS[j];
  cx.save(); cx.font=`bold ${tagPT*PT*view.k}px -apple-system,"Hiragino Sans",sans-serif`;
  const textW=cx.measureText(t.name).width/view.k;
  cx.restore();
  const h=tagPT*PT*1.75;
  // 淡色のピルの左に濃色のドットを置く余白を含める。
  // 文字をドットから離すと、長い日本語ラベルでも窮屈に見えない。
  return {w:textW+tagPT*PT*2.25, h:h, accentW:h*0.90};
}
const cardBox=i=>({w:cpos[i].s,h:cpos[i].s});

// ---- 描画 ---------------------------------------------------------------
function draw(){
  cx.clearRect(0,0,CW,CH);
  // 紙
  cx.fillStyle='#b0bec5'; cx.fillRect(0,0,CW,CH);
  cx.fillStyle='#fff';
  cx.fillRect(SX(0),SY(0),sheet.w*view.k,sheet.h*view.k);
  cx.strokeStyle='#546e7a'; cx.lineWidth=1;
  cx.strokeRect(SX(0),SY(0),sheet.w*view.k,sheet.h*view.k);

  if(opts.grid){
    cx.strokeStyle='#eceff1'; cx.lineWidth=1; cx.beginPath();
    for(let x=0;x<=sheet.w;x+=50){cx.moveTo(SX(x),SY(0));cx.lineTo(SX(x),SY(sheet.h));}
    for(let y=0;y<=sheet.h;y+=50){cx.moveTo(SX(0),SY(y));cx.lineTo(SX(sheet.w),SY(y));}
    cx.stroke();
  }
  if(opts.marg){
    cx.strokeStyle='#90a4ae'; cx.setLineDash([6,4]); cx.lineWidth=1;
    cx.strokeRect(SX(MARGIN),SY(MARGIN),(sheet.w-2*MARGIN)*view.k,(sheet.h-2*MARGIN)*view.k);
    cx.setLineDash([]);
  }
  if(opts.seam&&sheet.seam){
    const [d,v]=sheet.seam;
    cx.strokeStyle='#e53935'; cx.lineWidth=1.5; cx.setLineDash([10,6]); cx.beginPath();
    if(d==='v'){cx.moveTo(SX(v),SY(0));cx.lineTo(SX(v),SY(sheet.h));}
    else{cx.moveTo(SX(0),SY(v));cx.lineTo(SX(sheet.w),SY(v));}
    cx.stroke(); cx.setLineDash([]);
    cx.fillStyle='#e53935'; cx.font='11px sans-serif';
    if(d==='v') cx.fillText('継ぎ目',SX(v)+4,SY(0)+14);
    else cx.fillText('継ぎ目',SX(0)+4,SY(v)-4);
  }

  // エッジ
  if(opts.edge){
    EDGES.forEach(([tj,ci,v])=>{
      if(!opts.arch&&CASES[ci].status==='archived') return;
      cx.beginPath();
      cx.moveTo(SX(tpos[tj].x),SY(tpos[tj].y));
      cx.lineTo(SX(cpos[ci].x),SY(cpos[ci].y));
      cx.strokeStyle=TAGS[tj].color;
      cx.globalAlpha=0.18+0.5*v;
      cx.lineWidth=Math.max(0.6,(0.4+2.6*v)*view.k);
      cx.stroke();
    });
    cx.globalAlpha=1;
  }

  // カード
  CASES.forEach((c,i)=>{
    if(!opts.arch&&c.status==='archived') return;
    const p=cpos[i], s=p.s*view.k, x=SX(p.x)-s/2, y=SY(p.y)-s/2;
    if(x+s<0||y+s<0||x>CW||y>CH) return;
    if(!opts.img||s<22){
      const domi=c.tags[0];
      cx.beginPath(); cx.arc(SX(p.x),SY(p.y),Math.max(2.5,s*0.16),0,6.2832);
      cx.fillStyle=domi?domi.c:'#78909c'; cx.globalAlpha=c.status==='archived'?0.45:0.9;
      cx.fill(); cx.globalAlpha=1;
      cx.strokeStyle='#fff'; cx.lineWidth=1; cx.stroke();
    }else{
      if(IMG[i].complete) cx.drawImage(IMG[i],x,y,s,s);
      else{cx.fillStyle='#eceff1';cx.fillRect(x,y,s,s);}
    }
    if(p.pin){
      cx.beginPath(); cx.arc(x+s-4,y+4,3.2,0,6.2832);
      cx.fillStyle='#e53935'; cx.fill();
    }
    if(opts.num&&s>14){
      cx.font='bold 10px sans-serif'; cx.fillStyle='rgba(0,0,0,.62)';
      cx.fillRect(x,y+s-12,20,12);
      cx.fillStyle='#fff'; cx.fillText(c.no,x+3,y+s-3);
    }
    if(selKind==='case'&&sel.has(i)){
      cx.strokeStyle='#1e88e5'; cx.lineWidth=2.5; cx.strokeRect(x,y,s,s);
      if(sel.size===1){
        cx.fillStyle='#1e88e5'; cx.fillRect(x+s-6,y+s-6,12,12);
      }
    }
    if(opts.over&&bad.cards.has(i)){
      cx.strokeStyle='#e53935'; cx.lineWidth=2; cx.setLineDash([4,3]);
      cx.strokeRect(x-1,y-1,s+2,s+2); cx.setLineDash([]);
    }
  });

  // タグラベル
  if(opts.tag){
    TAGS.forEach((t,j)=>{
      const b=tagBox(j), p=tpos[j];
      const w=b.w*view.k, h=b.h*view.k, x=SX(p.x)-w/2, y=SY(p.y)-h/2;
      cx.save();
      cx.shadowColor='rgba(20,30,50,.16)'; cx.shadowBlur=5; cx.shadowOffsetY=1.5;
      rr(x,y,w,h,h/2); cx.fillStyle=t.lightColor; cx.fill();
      cx.restore();
      cx.save();
      cx.strokeStyle=t.color; cx.globalAlpha=0.55; cx.lineWidth=Math.max(1,h*0.045);
      rr(x,y,w,h,h/2); cx.stroke();
      cx.restore();
      const dotX=x+b.accentW*view.k*0.50;
      cx.beginPath(); cx.arc(dotX,y+h/2,h*0.19,0,6.2832);
      cx.fillStyle=t.color; cx.fill();
      cx.font=`bold ${tagPT*PT*view.k}px -apple-system,"Hiragino Sans",sans-serif`;
      cx.textAlign='center'; cx.textBaseline='middle';
      const textX=x+b.accentW*view.k+(w-b.accentW*view.k)/2;
      cx.fillStyle='#17202e';
      cx.fillText(t.name,textX,SY(p.y));
      cx.textAlign='left'; cx.textBaseline='alphabetic';
      if(p.pin){cx.beginPath();cx.arc(x+w-3,y+3,3.2,0,6.2832);cx.fillStyle='#e53935';cx.fill();}
      if(selKind==='tag'&&sel.has(j)){
        cx.strokeStyle='#1e88e5'; cx.lineWidth=2.5; rr(x,y,w,h,h/2); cx.stroke();
      }
    });
  }
  if(marq){
    cx.strokeStyle='#1e88e5'; cx.fillStyle='rgba(30,136,229,.12)';
    cx.lineWidth=1; cx.setLineDash([4,3]);
    cx.fillRect(marq.x,marq.y,marq.w,marq.h); cx.strokeRect(marq.x,marq.y,marq.w,marq.h);
    cx.setLineDash([]);
  }
  status();
}
function rr(x,y,w,h,r){
  cx.beginPath(); cx.moveTo(x+r,y);
  cx.arcTo(x+w,y,x+w,y+h,r); cx.arcTo(x+w,y+h,x,y+h,r);
  cx.arcTo(x,y+h,x,y,r); cx.arcTo(x,y,x+w,y,r); cx.closePath();
}
// ---- 点検 ---------------------------------------------------------------
const bad={cards:new Set()};
function check(){
  bad.cards.clear();
  const vis=CASES.map((c,i)=>i).filter(i=>opts.arch||CASES[i].status!=='archived');
  // 重なり
  let ov=0;
  for(let a=0;a<vis.length;a++)for(let b=a+1;b<vis.length;b++){
    const i=vis[a],j=vis[b],pi=cpos[i],pj=cpos[j];
    if(Math.abs(pi.x-pj.x)<(pi.s+pj.s)/2&&Math.abs(pi.y-pj.y)<(pi.s+pj.s)/2){
      ov++; bad.cards.add(i); bad.cards.add(j);
    }
  }
  // 紙外・継ぎ目
  let out=0,seam=0;
  vis.forEach(i=>{
    const p=cpos[i],h=p.s/2;
    if(p.x-h<MARGIN||p.y-h<MARGIN||p.x+h>sheet.w-MARGIN||p.y+h>sheet.h-MARGIN){
      out++; bad.cards.add(i);
    }
    if(sheet.seam){
      const [d,v]=sheet.seam, c=d==='v'?p.x:p.y;
      if(c-h<v&&c+h>v){seam++; bad.cards.add(i);}
    }
  });
  // 解像度：配置サイズで300dpi / 200dpi を割るもの
  const d300=vis.filter(i=>cpos[i].s>CASES[i].max_mm_300dpi).length;
  const d200=vis.filter(i=>cpos[i].s>CASES[i].max_mm_300dpi*1.5).length;
  // 文字サイズ（版下 MASTER_MM のときの pt を配置サイズに比例させる）
  const sizes=vis.map(i=>cpos[i].s), smin=Math.min(...sizes), smax=Math.max(...sizes);
  const ptOf=(basePt,s)=>basePt*s/MASTER_MM;
  const titlePt=ptOf(GEO.layout_mm.title_pt_max,smin);
  const legPt=ptOf(GEO.layout_mm.legend_pt,smin);
  // 面積
  const area=vis.reduce((a,i)=>a+cpos[i].s*cpos[i].s,0)/(sheet.w*sheet.h)*100;
  // タグラベルがカードに隠れている
  let hid=0;
  TAGS.forEach((t,j)=>{
    const b=tagBox(j),p=tpos[j];
    if(vis.some(i=>Math.abs(cpos[i].x-p.x)<(cpos[i].s+b.w)/2&&
                   Math.abs(cpos[i].y-p.y)<(cpos[i].s+b.h)/2)) hid++;
  });

  const L=document.getElementById('live');
  const row=(bad,label,val)=>`<div class="${bad?'bad':'ok'}">${bad?'⚠':'✓'} ${label}
    <span class="num">${val}</span></div>`;
  L.innerHTML=
    row(ov>0,'カードの重なり',ov+' 組')+
    row(out>0,'余白線の外に出ている',out+' 枚')+
    (sheet.seam?row(seam>0,'継ぎ目を跨いでいる',seam+' 枚'):
      '<div class="ok">✓ 継ぎ目なし（1枚刷り前提）</div>')+
    row(hid>0,'タグがカードに隠れている',hid+' 個')+
    row(d300>0,'300dpiを割るカード',d300+' 枚')+
    row(d200>0,'200dpiを割るカード',d200+' 枚')+
    row(area>35,'カードが占める紙面積',area.toFixed(1)+' %')+
    `<div style="margin-top:6px;padding-top:5px;border-top:1px solid #cfd8dc">
     いまのカードサイズ <b>${smin.toFixed(0)}${smax>smin?'–'+smax.toFixed(0):''}mm</b> で<br>
     タイトル <b class="${titlePt<8?'bad':''}">${titlePt.toFixed(1)}pt</b> ／
     凡例 <b class="${legPt<6?'bad':''}">${legPt.toFixed(1)}pt</b>
     <div class="hint">紙のポスターで確実に読めるのは概ね8pt以上。
     6pt未満は寄っても厳しいので、そこは実寸試し刷りで判断してください。</div></div>`;
  draw2();
}
let drawPending=false;
function draw2(){ if(drawPending) return; drawPending=true;
  requestAnimationFrame(()=>{drawPending=false; draw();}); }

function status(){
  document.getElementById('st1').textContent=
    `紙 ${sheet.w}×${sheet.h}mm`;
  document.getElementById('st2').textContent=
    `倍率 ${(view.k*100).toFixed(1)}%`;
  document.getElementById('st3').textContent=
    `選択 ${sel.size} (${selKind==='case'?'カード':'タグ'})`;
  document.getElementById('st4').textContent=
    `ピン ${cpos.filter(p=>p.pin).length+tpos.filter(p=>p.pin).length}`;
}

// ---- 当たり判定 ---------------------------------------------------------
function hit(mx,my){
  if(opts.tag) for(let j=TAGS.length-1;j>=0;j--){
    const b=tagBox(j),p=tpos[j];
    if(Math.abs(mx-p.x)<=b.w/2&&Math.abs(my-p.y)<=b.h/2) return {kind:'tag',i:j};
  }
  for(let i=CASES.length-1;i>=0;i--){
    if(!opts.arch&&CASES[i].status==='archived') continue;
    const p=cpos[i];
    if(Math.abs(mx-p.x)<=p.s/2&&Math.abs(my-p.y)<=p.s/2) return {kind:'case',i};
  }
  return null;
}
function handleAt(mx,my){
  if(selKind!=='case'||sel.size!==1) return false;
  const i=[...sel][0],p=cpos[i],h=p.s/2, g=8/view.k;
  return Math.abs(mx-(p.x+h))<g&&Math.abs(my-(p.y+h))<g;
}

// ---- マウス -------------------------------------------------------------
let drag=null, marq=null, space=false;
cv.addEventListener('mousedown',e=>{
  const r=cv.getBoundingClientRect();
  const mx=MX(e.clientX-r.left), my=MY(e.clientY-r.top);
  if(space||e.button===1){drag={mode:'pan',sx:e.clientX,sy:e.clientY,px:view.px,py:view.py};return;}
  if(handleAt(mx,my)){
    const i=[...sel][0];
    drag={mode:'resize',i,s0:cpos[i].s,mx,my}; snap(); return;
  }
  const h=hit(mx,my);
  if(h){
    if(selKind!==h.kind){sel.clear(); selKind=h.kind;}
    if(e.shiftKey||e.metaKey){ sel.has(h.i)?sel.delete(h.i):sel.add(h.i); }
    else if(!sel.has(h.i)){ sel.clear(); sel.add(h.i); }
    snap();
    const arr=h.kind==='case'?cpos:tpos;
    drag={mode:'move',kind:h.kind,mx,my,
          start:[...sel].map(i=>({i,x:arr[i].x,y:arr[i].y}))};
    showSel();
  }else{
    drag={mode:'marq',sx:e.clientX-r.left,sy:e.clientY-r.top,add:e.shiftKey};
    if(!e.shiftKey){sel.clear(); showSel();}
  }
  draw();
});
window.addEventListener('mousemove',e=>{
  const r=cv.getBoundingClientRect();
  const mx=MX(e.clientX-r.left), my=MY(e.clientY-r.top);
  if(!drag){
    const h=hit(mx,my);
    cv.style.cursor=handleAt(mx,my)?'nwse-resize':(h?'move':(space?'grab':'default'));
    if(h) tipShow(h,e); else tip.style.display='none';
    return;
  }
  tip.style.display='none';
  if(drag.mode==='pan'){
    view.px=drag.px-(e.clientX-drag.sx)/view.k;
    view.py=drag.py-(e.clientY-drag.sy)/view.k;
    draw2(); return;
  }
  if(drag.mode==='resize'){
    const p=cpos[drag.i];
    const s=Math.max(20,Math.min(140,drag.s0+((mx-drag.mx)+(my-drag.my))));
    p.s=opts.snap?Math.round(s):Math.round(s*10)/10;
    check(); return;
  }
  if(drag.mode==='move'){
    const arr=drag.kind==='case'?cpos:tpos;
    let dx=mx-drag.mx, dy=my-drag.my;
    drag.start.forEach(s=>{
      let nx=s.x+dx, ny=s.y+dy;
      if(opts.snap){nx=Math.round(nx/5)*5; ny=Math.round(ny/5)*5;}
      arr[s.i].x=nx; arr[s.i].y=ny;
    });
    check(); return;
  }
  if(drag.mode==='marq'){
    const x=e.clientX-r.left,y=e.clientY-r.top;
    marq={x:Math.min(x,drag.sx),y:Math.min(y,drag.sy),
          w:Math.abs(x-drag.sx),h:Math.abs(y-drag.sy)};
    draw2();
  }
});
window.addEventListener('mouseup',()=>{
  const moved=drag&&(drag.mode==='move'||drag.mode==='resize');
  if(drag&&drag.mode==='marq'&&marq){
    const x0=MX(marq.x),y0=MY(marq.y),x1=MX(marq.x+marq.w),y1=MY(marq.y+marq.h);
    if(!drag.add){sel.clear();}
    selKind='case';
    CASES.forEach((c,i)=>{
      if(!opts.arch&&c.status==='archived') return;
      const p=cpos[i];
      if(p.x>x0&&p.x<x1&&p.y>y0&&p.y<y1) sel.add(i);
    });
    showSel();
  }
  marq=null; drag=null; draw(); check();
  if(moved) snap();       // 動かし終わった状態を履歴に積む（取り消しの単位を1操作にする）
});
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect(), sx=e.clientX-r.left, sy=e.clientY-r.top;
  const mmx=MX(sx), mmy=MY(sy);
  const f=Math.exp(-e.deltaY*0.0016);
  view.k=Math.max(0.05,Math.min(8,view.k*f));
  view.px=mmx-sx/view.k; view.py=mmy-sy/view.k;
  draw2();
},{passive:false});
cv.addEventListener('dblclick',e=>{
  const r=cv.getBoundingClientRect();
  const h=hit(MX(e.clientX-r.left),MY(e.clientY-r.top));
  if(h&&h.kind==='case') window.open(CASES[h.i].url,'_blank');
});
window.addEventListener('keydown',e=>{
  if(e.code==='Space'){space=true; cv.style.cursor='grab'; e.preventDefault();}
  if(e.key==='f'||e.key==='F'){fit(); draw();}
  if(e.key==='p'||e.key==='P'){togglePin();}
  if((e.metaKey||e.ctrlKey)&&e.key==='z'){
    e.preventDefault();
    if(e.shiftKey){if(hi<hist.length-1) restore(hist[++hi]);}
    else{if(hi>0) restore(hist[--hi]);}
  }
});
window.addEventListener('keyup',e=>{if(e.code==='Space'){space=false;cv.style.cursor='default';}});

function tipShow(h,e){
  if(h.kind==='tag'){
    tip.innerHTML=`<b>${TAGS[h.i].name}</b><br>${TAGS[h.i].count} 件に付与`;
  }else{
    const c=CASES[h.i];
    tip.innerHTML=`<b>No.${c.no} ${c.title}</b><br>`+
      c.tags.map(t=>`${t.t} ${Math.round(t.v*100)}%`).join(' ／ ')+
      `<br>配置 ${cpos[h.i].s.toFixed(1)}mm ／ 300dpi上限 ${c.max_mm_300dpi}mm`+
      `<br><span style="color:#90a4ae">ダブルクリックでCALL4のページ</span>`;
  }
  tip.style.display='block';
  tip.style.left=Math.min(window.innerWidth-300,e.clientX+14)+'px';
  tip.style.top=(e.clientY+14)+'px';
}
function showSel(){
  const d=document.getElementById('sel');
  if(!sel.size){d.textContent='なし'; return;}
  if(selKind==='tag'){
    d.innerHTML=[...sel].map(j=>`<div>${TAGS[j].name}（${TAGS[j].count}件）</div>`).join('');
    return;
  }
  if(sel.size===1){
    const i=[...sel][0],c=CASES[i],p=cpos[i];
    d.innerHTML=`<img src="${c.img}"><b>No.${c.no}</b> ${c.title}<br>
      ${c.status==='archived'?'アーカイブ':'進行中'}／${p.s.toFixed(1)}mm角<br>
      ${c.tags.map(t=>`<span style="color:${t.c}">■</span>${t.t} ${Math.round(t.v*100)}%`).join('<br>')}
      ${c.max_mm_300dpi<p.s?`<div style="color:#b32d00">この配置では300dpi未満（上限 ${c.max_mm_300dpi}mm）</div>`:''}
      <div style="margin-top:4px"><a href="${c.url}" target="_blank">CALL4のページ</a></div>`;
  }else{
    d.innerHTML=`カード ${sel.size} 枚を選択中`;
  }
}

// ---- 力学で整える -------------------------------------------------------
let anim=null;
function relaxStep(iters,collideOnly){
  const N=CASES.length, T=TAGS.length;
  const fx=new Float64Array(N+T), fy=new Float64Array(N+T);
  const gx=i=>i<N?cpos[i].x:tpos[i-N].x, gy=i=>i<N?cpos[i].y:tpos[i-N].y;
  const bw=i=>i<N?cpos[i].s:tagBox(i-N).w, bh=i=>i<N?cpos[i].s:tagBox(i-N).h;
  const pinned=i=>i<N?cpos[i].pin:tpos[i-N].pin;
  const live=i=>i<N?(opts.arch||CASES[i].status!=='archived'):true;
  for(let it=0;it<iters;it++){
    fx.fill(0); fy.fill(0);
    if(!collideOnly){
      // エッジのばね。割合が大きいほど強く、近い距離で釣り合う
      EDGES.forEach(([tj,ci,v])=>{
        if(!live(ci)) return;
        const a=ci,b=N+tj;
        let dx=gx(b)-gx(a), dy=gy(b)-gy(a);
        const d=Math.hypot(dx,dy)||1e-6;
        const L=(bw(a)+bw(b))/2*1.35;
        const f=0.020*(0.35+v)*(d-L);
        dx/=d; dy/=d;
        fx[a]+=f*dx; fy[a]+=f*dy; fx[b]-=f*dx; fy[b]-=f*dy;
      });
      // 全体をゆるく中央へ
      const cxm=sheet.w/2, cym=sheet.h/2;
      for(let i=0;i<N+T;i++){
        if(!live(i)) continue;
        fx[i]+=(cxm-gx(i))*0.0012; fy[i]+=(cym-gy(i))*0.0012;
      }
    }
    // 矩形の重なりを解く（カードなので円ではなく箱で押し合う）
    const GAP=collideOnly?3:6;
    for(let a=0;a<N+T;a++){
      if(!live(a)) continue;
      for(let b=a+1;b<N+T;b++){
        if(!live(b)) continue;
        const ox=(bw(a)+bw(b))/2+GAP-Math.abs(gx(a)-gx(b));
        if(ox<=0) continue;
        const oy=(bh(a)+bh(b))/2+GAP-Math.abs(gy(a)-gy(b));
        if(oy<=0) continue;
        // 抜け出しやすい軸に押し出す
        if(ox<oy){
          const s=(gx(a)<gx(b)?-1:1)*ox*0.5;
          fx[a]+=s; fx[b]-=s;
        }else{
          const s=(gy(a)<gy(b)?-1:1)*oy*0.5;
          fy[a]+=s; fy[b]-=s;
        }
      }
    }
    // 反映（ピンは動かさない・余白の内側に留める・継ぎ目を跨がせない）
    for(let i=0;i<N+T;i++){
      if(pinned(i)||!live(i)) continue;
      const damp=collideOnly?0.45:0.35;
      let nx=gx(i)+Math.max(-14,Math.min(14,fx[i]))*damp;
      let ny=gy(i)+Math.max(-14,Math.min(14,fy[i]))*damp;
      const hw=bw(i)/2, hh=bh(i)/2;
      nx=Math.max(MARGIN+hw,Math.min(sheet.w-MARGIN-hw,nx));
      ny=Math.max(MARGIN+hh,Math.min(sheet.h-MARGIN-hh,ny));
      if(sheet.seam){
        // 継ぎ目に載ったら近い側へ寄せる。紙を分けて刷る前提だと
        // 跨いだカードは物理的に貼れない（切って貼ることになる）ため。
        const [dir,v]=sheet.seam;
        if(dir==='v'&&nx-hw<v&&nx+hw>v)
          nx=(nx<v)?Math.min(nx,v-hw-1):Math.max(nx,v+hw+1);
        if(dir==='h'&&ny-hh<v&&ny+hh>v)
          ny=(ny<v)?Math.min(ny,v-hh-1):Math.max(ny,v+hh+1);
      }
      if(i<N){cpos[i].x=nx;cpos[i].y=ny;}else{tpos[i-N].x=nx;tpos[i-N].y=ny;}
    }
  }
  // 残っているカード同士の重なり数を返す（0になったら回すのをやめる）
  let rest=0;
  for(let a=0;a<N;a++){
    if(!live(a)) continue;
    for(let b=a+1;b<N;b++){
      if(!live(b)) continue;
      if(Math.abs(cpos[a].x-cpos[b].x)<(cpos[a].s+cpos[b].s)/2&&
         Math.abs(cpos[a].y-cpos[b].y)<(cpos[a].s+cpos[b].s)/2) rest++;
    }
  }
  return rest;
}
// 「重なりだけ解く」は見せる意味がないので、一気に回して収束したら終わり。
// 毎フレーム88枚を描き直すと、収束前に何秒も待たされることになる。
function spread(){
  stop();
  snap();
  for(let i=0;i<1500;i+=15){
    if(relaxStep(15,true)===0) break;
  }
  check(); snap();
}
// 「整える（力学）」は動いていく過程に意味があるのでアニメーションさせる。
function run(){
  stop(); snap();
  let left=900;
  const tick=()=>{
    relaxStep(10,false);
    left-=10; check();
    if(left>0) anim=requestAnimationFrame(tick); else {anim=null; snap();}
  };
  anim=requestAnimationFrame(tick);
}
function stop(){ if(anim){cancelAnimationFrame(anim); anim=null; snap();} }
document.getElementById('relax').onclick=run;
document.getElementById('spread').onclick=spread;
document.getElementById('stop').onclick=stop;

function togglePin(){
  if(!sel.size) return;
  snap();
  const arr=selKind==='case'?cpos:tpos;
  const any=[...sel].some(i=>!arr[i].pin);
  sel.forEach(i=>arr[i].pin=any);
  draw(); status(); snap();
}
document.getElementById('pin').onclick=togglePin;
document.getElementById('unpin').onclick=()=>{
  snap(); cpos.forEach(p=>p.pin=false); tpos.forEach(p=>p.pin=false);
  draw(); status(); snap();};
document.getElementById('selall').onclick=()=>{
  selKind='case'; sel=new Set(CASES.map((c,i)=>i)
    .filter(i=>opts.arch||CASES[i].status!=='archived')); showSel(); draw();};
document.getElementById('selnone').onclick=()=>{sel.clear(); showSel(); draw();};

// ---- 入力 ---------------------------------------------------------------
const csz=document.getElementById('csz'), cszn=document.getElementById('cszn');
const tsz=document.getElementById('tsz'), tszn=document.getElementById('tszn');
function syncInputs(){
  csz.value=cszn.value=Math.round(cardMM);
  tsz.value=tagPT; tszn.textContent=tagPT;
}
// スライダーは input が連続で飛んでくるので、履歴は
// つまむ前(mousedown) と 離したあと(change) の2点だけ積む。
function setCard(v){
  cardMM=+v;
  const target=(document.getElementById('selonly').checked&&selKind==='case'&&sel.size)
    ? [...sel] : CASES.map((c,i)=>i);
  target.forEach(i=>cpos[i].s=cardMM);
  syncInputs(); check(); showSel();
}
csz.addEventListener('mousedown',()=>snap());
csz.addEventListener('input',e=>setCard(e.target.value));
csz.addEventListener('change',()=>snap());
cszn.addEventListener('change',e=>{snap(); setCard(e.target.value); snap();});
tsz.addEventListener('mousedown',()=>snap());
tsz.addEventListener('input',e=>{tagPT=+e.target.value; syncInputs(); check();});
tsz.addEventListener('change',()=>snap());

const sheetSel=document.getElementById('sheet');
SHEETS.forEach((s,i)=>{
  const o=document.createElement('option'); o.value=i; o.textContent=s.label;
  sheetSel.appendChild(o);
});
sheetSel.addEventListener('change',e=>{
  snap(); sheet=SHEETS[+e.target.value];
  // 紙が変わるとカードの上限も変わるので、既定サイズに戻す
  cardMM=sheet.key==='A0'?55:75; syncInputs();
  applyLayout(LAY,true); fit(); check(); spread();
});
// 叩き台を当てたあとは、そのまま重なりだけ解いておく。
// 生の力学レイアウトは必ず重なるので、押した直後から触れる状態にしておく。
document.querySelectorAll('[data-lay]').forEach(b=>b.onclick=()=>{
  snap(); applyLayout(b.dataset.lay,false); check(); spread();});

document.getElementById('wtog').parentNode.onclick=()=>{
  const w=document.getElementById('warn');
  w.classList.toggle('closed');
  document.getElementById('wtog').textContent=w.classList.contains('closed')?'▶':'▼';
  resize(); draw();
};

// ---- 保存 / 読込 / PNG --------------------------------------------------
function dl(name,blob){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),4000);
}
function layoutObj(){
  const o={
    format:'call4-card-layout', version:1,
    sheet:sheet.key, sheet_mm:[sheet.w,sheet.h], margin_mm:MARGIN,
    layout_base:LAY, card_mm_default:cardMM, tag_pt:tagPT,
    note:'座標はカード中心のmm。原点は紙の左上。',
    cards:{}, tags:{}
  };
  CASES.forEach((c,i)=>o.cards[c.id]={
    x:+cpos[i].x.toFixed(2), y:+cpos[i].y.toFixed(2),
    s:+cpos[i].s.toFixed(2), pin:cpos[i].pin});
  TAGS.forEach((t,j)=>o.tags[t.name]={
    x:+tpos[j].x.toFixed(2), y:+tpos[j].y.toFixed(2), pin:tpos[j].pin});
  return o;
}
// 座標はケースID・タグ名で引くので、ケースが増減しても壊れない。
// 見つからなかったものは叩き台の位置に残し、件数を返す。
function applySaved(o){
  const si=SHEETS.findIndex(s=>s.key===o.sheet);
  if(si>=0){sheet=SHEETS[si]; sheetSel.value=si;}
  if(o.tag_pt) tagPT=o.tag_pt;
  if(o.card_mm_default) cardMM=o.card_mm_default;
  let miss=0;
  CASES.forEach((c,i)=>{
    const p=o.cards&&o.cards[c.id];
    if(p){cpos[i].x=p.x;cpos[i].y=p.y;cpos[i].s=p.s;cpos[i].pin=!!p.pin;}
    else miss++;
  });
  TAGS.forEach((t,j)=>{
    const p=o.tags&&o.tags[t.name];
    if(p){tpos[j].x=p.x;tpos[j].y=p.y;tpos[j].pin=!!p.pin;}
  });
  syncInputs(); snap(); fit(); check();
  return miss;
}
document.getElementById('save').onclick=()=>{
  dl('layout_manual.json',
     new Blob([JSON.stringify(layoutObj(),null,1)],{type:'application/json'}));
};
document.getElementById('load').onclick=()=>document.getElementById('file').click();
document.getElementById('file').addEventListener('change',e=>{
  const f=e.target.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=()=>{
    try{
      const miss=applySaved(JSON.parse(r.result));
      if(miss) alert(`読み込みました。${miss} 件は座標が入っていなかったので`
        +`叩き台の位置のままです（ケースが増えた場合など）。`);
    }catch(err){ alert('読み込めませんでした: '+err.message); }
  };
  r.readAsText(f);
});

// ---- テキストでの受け渡し ----------------------------------------------
// リンクで共有した場合、ページは枠の中で動くので
// ダウンロードとファイル選択が使えないことがある。そのときはここを使う。
const OV=document.getElementById('ov'), OVTA=document.getElementById('ovta');
const OVMSG=document.getElementById('ovmsg');
const embedded=(()=>{try{return window.self!==window.top;}catch(e){return true;}})();
document.getElementById('ovnote').textContent=embedded
  ? '下の中身が、いまの配置です。全部コピーして徳永に送ってください。'
    +'（このページは枠の中で動いているので「保存(JSON)」のダウンロードが'
    +'効かないことがあります）逆に、受け取った内容をここに貼って読み込むこともできます。'
  : '下の中身が、いまの配置です。コピーして貼り付ける形でも渡せます。'
    +'受け取った内容をここに貼って読み込むこともできます。';
function showText(){
  OVTA.value=JSON.stringify(layoutObj(),null,1);
  OVMSG.textContent=''; OV.style.display='flex';
  OVTA.focus(); OVTA.setSelectionRange(0,0);
}
document.getElementById('text').onclick=showText;
document.getElementById('ovclose').onclick=()=>{OV.style.display='none';};
document.getElementById('ovcopy').onclick=async()=>{
  OVTA.select();
  try{
    await navigator.clipboard.writeText(OVTA.value);
    OVMSG.textContent='コピーしました';
  }catch(e){
    // クリップボードAPIが塞がれている場合。全選択だけしておく
    OVMSG.textContent='選択しました。Cmd+C（Ctrl+C）でコピーしてください';
  }
};
document.getElementById('ovload').onclick=()=>{
  try{
    const miss=applySaved(JSON.parse(OVTA.value));
    OV.style.display='none';
    if(miss) alert(`読み込みました。${miss} 件は座標が入っていなかったので`
      +`叩き台の位置のままです。`);
  }catch(err){ OVMSG.style.color='#b32d00';
    OVMSG.textContent='読み込めませんでした: '+err.message; }
};
OV.addEventListener('click',e=>{ if(e.target===OV) OV.style.display='none'; });
document.getElementById('png').onclick=()=>{
  // 紙全体を 2px/mm で書き出す。画面と同じ draw() を別Canvasに向けて呼ぶだけ。
  // あくまで確認用。入稿には使えません（文字がラスタになるため）。
  const S=2, w=Math.round(sheet.w*S), h=Math.round(sheet.h*S);
  const o=document.createElement('canvas'); o.width=w; o.height=h;
  const keep={cx,CW,CH,k:view.k,px:view.px,py:view.py}, keepSel=sel;
  try{
    cx=o.getContext('2d');
    CW=w; CH=h; view.k=S; view.px=0; view.py=0;
    sel=new Set();                   // 選択枠は書き出さない
    const m=marq; marq=null;
    draw();
    marq=m;
  }finally{
    cx=keep.cx; CW=keep.CW; CH=keep.CH;
    view.k=keep.k; view.px=keep.px; view.py=keep.py; sel=keepSel;
    draw();
  }
  o.toBlob(b=>dl('layout_preview.png',b),'image/png');
};

// ---- 起動 ---------------------------------------------------------------
window.addEventListener('resize',()=>{resize(); draw();});
resize(); applyLayout('cluster',true); fit(); syncInputs(); snap(); check(); spread();
</script></body></html>"""


def main():
    cards, geo = load()
    cases, tags = cards["cases"], cards["tags"]
    missing_colors = [t for t in tags if t not in GUI_TAG_PALETTE]
    if missing_colors:
        raise ValueError(f"GUI_TAG_PALETTE に未定義のタグがあります: {missing_colors}")
    tag_nodes = [{"name": t,
                  "color": GUI_TAG_PALETTE[t][0],
                  "lightColor": GUI_TAG_PALETTE[t][1],
                  "count": sum(1 for c in cases if any(x["t"] == t for x in c["tags"]))}
                 for t in tags]
    tidx = {t: j for j, t in enumerate(tags)}
    edges = [[tidx[x["t"]], i, x["v"]]
             for i, c in enumerate(cases) for x in c["tags"]]
    lays = layouts(cases, tags)

    slim = [{k: c[k] for k in ("id", "no", "title", "status", "url", "tags",
                               "max_mm_300dpi", "img")} for c in cases]

    html = TEMPLATE
    for tok, val in [
        ("@@SHEETS@@", json.dumps(SHEETS, ensure_ascii=False)),
        ("@@MARGIN@@", json.dumps(MARGIN_MM)),
        ("@@CASES@@", json.dumps(slim, ensure_ascii=False)),
        ("@@TAGS@@", json.dumps(tag_nodes, ensure_ascii=False)),
        ("@@EDGES@@", json.dumps(edges)),
        ("@@LAYOUTS@@", json.dumps(lays)),
        ("@@MASTER_MM@@", json.dumps(cards["master_mm"])),
        ("@@GEO@@", json.dumps({"layout_mm": geo["layout_mm"]}, ensure_ascii=False)),
    ]:
        html = html.replace(tok, val)

    p = os.path.join(OUT, "editor.html")
    open(p, "w", encoding="utf-8").write(html)
    print(f"[s3] wrote {p}  ({len(html.encode('utf-8')) / 1024 / 1024:.2f} MB)")

    # 埋め込み用（リンク共有）。<!doctype>/<html>/<head>/<body> を持たない断片にする。
    # 共有先が同じ骨組みを外から付けるため、二重になると壊れる。
    frag = html[html.index("<title>"):]
    for a, b in (("</style></head><body>", "</style>"), ("</body></html>", "")):
        frag = frag.replace(a, b)
    q = os.path.join(OUT, "editor_embed.html")
    open(q, "w", encoding="utf-8").write(frag)
    print(f"[s3] wrote {q}  ({len(frag.encode('utf-8')) / 1024 / 1024:.2f} MB) 埋め込み用")
    print(f"[s3] カード {len(cases)} / タグ {len(tags)} / エッジ {len(edges)}")
    print(f"[s3] 紙の既定: {SHEETS[0]['label']}（余白 {MARGIN_MM}mm）")


if __name__ == "__main__":
    main()
