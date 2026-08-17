#!/usr/bin/env python3
"""card_editor/s3 — ノード配置エディタ（単一HTML）を書き出す。

出力は out/editor.html の1ファイルだけ。カード画像はbase64で埋め込むので、
相手はダブルクリックしてブラウザで開くだけで使える（サーバ・解凍・インストール不要）。
data: URI なのでCanvasが汚染されず、PNG書き出しも動く。

座標系は最初から mm。紙は幅914mm固定のロール紙で、
縦の長さはGUI上で500〜10,000mmの範囲で調整できる。

叩き台のレイアウトは igraph で作ってから正規化して埋め込む。
エディタ側は「紙の使える範囲」に写して使う。
"""
import base64
import hashlib

import json
import math
import os
import random

import igraph as ig
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# ---- ロール紙 --------------------------------------------------------------
ROLL_SPEC = {
    "key": "roll_914",
    "w": 914,
    "default_h": 2378,
    "min_h": 500,
    "max_h": 10000,
    "step": 10,
}
MARGIN_MM = 25.0        # 印刷余白（この内側に置く）

# ---- GUI用タグ配色 -------------------------------------------------------
# 参照色見本の「濃色 + 淡色」のペア。
# 個人情報・プライバシーと情報公開は、指定により色見本と入れ替える。
# GUIのタグノード用。カード画像は s2_make_cards.py で同じ濃色を使う。
GUI_TAG_PALETTE = {
    "個人情報・プライバシー": ("#22504e", "#bac9c8"),
    "医療・福祉・障がい": ("#ff4709", "#ffcfbf"),
    "ジェンダー・セクシュアリティ": ("#ff9423", "#ffdccb"),
    "刑事司法": ("#9f6e34", "#e2d3c2"),
    "環境・災害": ("#2e9d7e", "#b6ddd2"),
    "働き方": ("#99b73d", "#e0e9c5"),
    "公正な手続": ("#fe7389", "#ffdbe0"),
    "沖縄": ("#3970cb", "#bacded"),
    "外国にルーツを持つ人々": ("#6b5498", "#d4cde1"),
    "政治参加・表現の自由": ("#3daac8", "#c5e5ee"),
    "情報公開": ("#033064", "#b3c1d0"),
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
.modal{display:none;position:fixed;inset:0;background:rgba(38,50,56,.55);z-index:21;
align-items:center;justify-content:center}
.exportopts{display:flex;flex-wrap:wrap;gap:12px 24px;padding:8px 0}
.exportopts label{display:flex;align-items:center;gap:8px;color:#37474f}
.exportinfo{background:#f5f7f8;border:1px solid #cfd8dc;border-radius:5px;padding:9px 11px;
font-size:11.5px;color:#455a64;min-height:42px}
.exportinfo.bad{background:#fff3f1;border-color:#ef9a9a;color:#b32d00}
#exmsg{font-size:11.5px;color:#2e7d32}
</style></head><body>

<div id="warn">
<h3><span id="wtog">▼</span>この図を紙にするまでに未解決の点（クリックで折りたたみ）</h3>
<ul>
<li><b>タグの割合は暫定値です。</b>タグ確認（フェーズ1）が終わると各カードの帯が変わります。
位置はケースIDで保存するので配置作業は無駄になりませんが、カード画像は作り直しになります。</li>
<li><b>タグ11色は画面用のRGB色です。</b>CMYK印刷では色味や色の差が変わるため、
画面で見分けられたタグが紙では混ざります。
入稿前に印刷用の色を選び直し、実機で色校正を1回とる必要があります。</li>
<li><b>入稿前にIllustratorで最終確認してください。</b>「書き出し」から、文字・線・タグを
ベクタのまま編集できる透明SVGを出せます。カード部分だけは高解像度JPEGを埋め込みます。</li>
<li><b>実寸の試し刷りをまだしていません。</b><code>out/proof_A4.pdf</code> をA4等倍で刷って、
カードの文字が読めるサイズを手元で確かめてください。ここの数値はその後に決まります。</li>
<li><b>ロール紙の長さは914mm幅に対して調整できます。</b>914×2378mmを初期値とし、
上部の「長さ」欄で500〜10,000mmの範囲で変更できます。</li>
<li><b>画像の権利。</b>CALL4掲載のサムネイルを含む図です。公開物にするなら
CALL4側の了解が前提になります。</li>
</ul>
<div class="wsub">下の「点検」欄は、いまの配置に対する自動チェックです（重なり・紙外・解像度・文字サイズ）。</div>
</div>

<div id="bar">
<div class="grp">
<label>ロール幅</label><strong>914mm</strong>
<label>長さ</label><input type="number" id="rolllen" min="500" max="10000" step="10" style="width:76px">mm
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
<button id="open">開く</button>
<button id="save">上書き保存</button>
<button id="saveas">別名で保存</button>
<button id="recover" style="display:none">自動保存から復元</button>
<button id="text">テキストで受け渡し</button>
<button id="export">書き出し</button>
<input type="file" id="file" accept=".json" style="display:none">
<span id="savest" style="color:#cfd8dc;font-size:11px"></span>
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

<div id="exov" class="modal">
<div class="ovbox" style="width:min(620px,92vw)">
<h3>Illustrator / Photoshop向け書き出し</h3>
<p>背景は透明です。SVGは線・タグ・文字をベクタのまま、カードだけを高解像度JPEGで埋め込みます。</p>
<div class="exportopts">
<label>範囲
<select id="exscope"><option value="graph">グラフ部分（推奨）</option><option value="paper">ロール紙全体</option></select>
</label>
<label>PNG解像度
<select id="exdpi"><option value="150">150 dpi（推奨）</option><option value="300">300 dpi</option></select>
</label>
</div>
<div id="exinfo" class="exportinfo"></div>
<div class="ovbtns">
<button id="exsvg">SVG（Illustrator）</button>
<button id="expng">透明PNG（Photoshop）</button>
<button id="exclose">閉じる</button>
<span id="exmsg"></span>
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
const ROLL=@@ROLL@@, MARGIN=@@MARGIN@@;
const CASES=@@CASES@@, TAGS=@@TAGS@@, EDGES=@@EDGES@@, LAYOUTS=@@LAYOUTS@@;
const MASTER_MM=@@MASTER_MM@@, GEO=@@GEO@@;
const PT=25.4/72;

// ---- 状態 ---------------------------------------------------------------
const sheet={key:ROLL.key,w:ROLL.w,h:ROLL.default_h};
let LAY='cluster';
let cardMM=75, tagPT=30;
const cpos=CASES.map(()=>({x:0,y:0,s:75,pin:false}));   // カード中心(mm)
const tpos=TAGS.map(()=>({x:0,y:0,pin:false}));
let sel=new Set(), selKind='case';
const view={k:0.3,px:0,py:0};
const opts={};
['img','edge','arch','tag','grid','marg','snap','over','num'].forEach(n=>{
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
  const s=JSON.stringify({c:cpos,t:tpos,cardMM,tagPT,sheetH:sheet.h});
  if(hi>=0&&hist[hi]===s) return;
  hist.splice(hi+1); hist.push(s); if(hist.length>80) hist.shift(); hi=hist.length-1;
  scheduleAutoSave();
}
function restore(s){
  const o=JSON.parse(s);
  o.c.forEach((p,i)=>Object.assign(cpos[i],p));
  o.t.forEach((p,j)=>Object.assign(tpos[j],p));
  cardMM=o.cardMM; tagPT=o.tagPT;
  if(o.sheetH) sheet.h=o.sheetH;
  syncInputs(); fit(); draw(); check(); scheduleAutoSave();
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
  // ロール紙の印刷余白外
  let out=0;
  vis.forEach(i=>{
    const p=cpos[i],h=p.s/2;
    if(p.x-h<MARGIN||p.y-h<MARGIN||p.x+h>sheet.w-MARGIN||p.y+h>sheet.h-MARGIN){
      out++; bad.cards.add(i);
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
    '<div class="ok">✓ ロール紙（継ぎ目なし）</div>'+
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
    `ロール紙 ${sheet.w}×${sheet.h}mm`;
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
    // 反映（ピンは動かさない・余白の内側に留める）
    for(let i=0;i<N+T;i++){
      if(pinned(i)||!live(i)) continue;
      const damp=collideOnly?0.45:0.35;
      let nx=gx(i)+Math.max(-14,Math.min(14,fx[i]))*damp;
      let ny=gy(i)+Math.max(-14,Math.min(14,fy[i]))*damp;
      const hw=bw(i)/2, hh=bh(i)/2;
      nx=Math.max(MARGIN+hw,Math.min(sheet.w-MARGIN-hw,nx));
      ny=Math.max(MARGIN+hh,Math.min(sheet.h-MARGIN-hh,ny));
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
const rolllen=document.getElementById('rolllen');
function syncInputs(){
  csz.value=cszn.value=Math.round(cardMM);
  tsz.value=tagPT; tszn.textContent=tagPT;
  rolllen.value=Math.round(sheet.h);
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

function setRollLength(v,scalePositions){
  let next=Number(v);
  if(!Number.isFinite(next)) next=ROLL.default_h;
  next=Math.round(next/ROLL.step)*ROLL.step;
  next=Math.max(ROLL.min_h,Math.min(ROLL.max_h,next));
  const old=sheet.h;
  if(next===old){syncInputs(); return;}
  if(scalePositions&&old>2*MARGIN){
    const ratio=(next-2*MARGIN)/(old-2*MARGIN);
    cpos.forEach(p=>p.y=MARGIN+(p.y-MARGIN)*ratio);
    tpos.forEach(p=>p.y=MARGIN+(p.y-MARGIN)*ratio);
  }
  sheet.h=next;
  syncInputs(); fit(); check();
}
rolllen.addEventListener('change',e=>{
  snap(); setRollLength(e.target.value,true); snap();
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

// ---- 保存 / 読込 --------------------------------------------------------
function dl(name,blob){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),4000);
}
const AUTOSAVE_KEY='call4-card-layout-autosave-v2';
let layoutHandle=null, autoTimer=null, savedState='';
const saveStatus=document.getElementById('savest');
function layoutObj(){
  const o={
    format:'call4-card-layout', version:2,
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
function layoutJSON(){return JSON.stringify(layoutObj(),null,1);}
function setSaveStatus(msg){saveStatus.textContent=msg;}
function scheduleAutoSave(){
  clearTimeout(autoTimer);
  autoTimer=setTimeout(()=>{
    try{
      const s=layoutJSON();
      localStorage.setItem(AUTOSAVE_KEY,s);
      document.getElementById('recover').style.display='inline-block';
      setSaveStatus(s===savedState?'保存済み':'未保存・自動バックアップ済み');
    }catch(e){setSaveStatus('自動バックアップ不可');}
  },350);
}
// 座標はケースID・タグ名で引くので、ケースが増減しても壊れない。
// 見つからなかったものは叩き台の位置に残し、件数を返す。
function applySaved(o){
  // version 1のA0配置も sheet_mm の長さを引き継いで読み込める。
  // 幅は現行仕様の914mmに固定し、旧幅から横座標を比例変換する。
  if(o.sheet_mm&&o.sheet_mm[1]) setRollLength(o.sheet_mm[1],false);
  const sourceW=Number(o.sheet_mm&&o.sheet_mm[0])||sheet.w;
  const xScale=sourceW>2*MARGIN?(sheet.w-2*MARGIN)/(sourceW-2*MARGIN):1;
  const mapX=x=>MARGIN+(x-MARGIN)*xScale;
  if(o.tag_pt) tagPT=o.tag_pt;
  if(o.card_mm_default) cardMM=o.card_mm_default;
  let miss=0;
  CASES.forEach((c,i)=>{
    const p=o.cards&&o.cards[c.id];
    if(p){cpos[i].x=mapX(p.x);cpos[i].y=p.y;cpos[i].s=p.s;cpos[i].pin=!!p.pin;}
    else miss++;
  });
  TAGS.forEach((t,j)=>{
    const p=o.tags&&o.tags[t.name];
    if(p){tpos[j].x=mapX(p.x);tpos[j].y=p.y;tpos[j].pin=!!p.pin;}
  });
  syncInputs(); snap(); fit(); check(); scheduleAutoSave();
  return miss;
}
function afterLoaded(name){
  savedState=layoutJSON();
  setSaveStatus(name?`${name} を開きました`:'読み込みました');
}
function loadLayoutText(text,name){
  const miss=applySaved(JSON.parse(text));
  afterLoaded(name);
  if(miss) alert(`読み込みました。${miss} 件は座標が入っていなかったので`
    +`叩き台の位置のままです（ケースが増えた場合など）。`);
}
async function writeHandle(handle,text){
  const writable=await handle.createWritable();
  await writable.write(new Blob([text],{type:'application/json'}));
  await writable.close();
}
async function saveAs(){
  const text=layoutJSON();
  if('showSaveFilePicker' in window){
    const handle=await window.showSaveFilePicker({
      suggestedName:'layout_manual.json',
      types:[{description:'CALL4配置データ',accept:{'application/json':['.json']}}]
    });
    await writeHandle(handle,text); layoutHandle=handle;
  }else{
    dl('layout_manual.json',new Blob([text],{type:'application/json'}));
  }
  savedState=text; setSaveStatus(layoutHandle?'上書き保存しました':'ダウンロードしました');
}
document.getElementById('save').onclick=async()=>{
  try{
    if(!layoutHandle){await saveAs();return;}
    const text=layoutJSON(); await writeHandle(layoutHandle,text);
    savedState=text; setSaveStatus('上書き保存しました');
  }catch(e){if(e.name!=='AbortError') alert('保存できませんでした: '+e.message);}
};
document.getElementById('saveas').onclick=async()=>{
  try{await saveAs();}catch(e){if(e.name!=='AbortError') alert('保存できませんでした: '+e.message);}
};
document.getElementById('open').onclick=async()=>{
  if(!('showOpenFilePicker' in window)){
    document.getElementById('file').click(); return;
  }
  try{
    const [handle]=await window.showOpenFilePicker({
      multiple:false,types:[{description:'CALL4配置データ',accept:{'application/json':['.json']}}]
    });
    const f=await handle.getFile(); loadLayoutText(await f.text(),f.name); layoutHandle=handle;
  }catch(e){if(e.name!=='AbortError') alert('読み込めませんでした: '+e.message);}
};
document.getElementById('file').addEventListener('change',e=>{
  const f=e.target.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=()=>{
    try{loadLayoutText(r.result,f.name); layoutHandle=null;}
    catch(err){alert('読み込めませんでした: '+err.message);}
  };
  r.readAsText(f);
  e.target.value='';
});
document.getElementById('recover').onclick=()=>{
  try{
    const text=localStorage.getItem(AUTOSAVE_KEY);
    if(!text) return;
    applySaved(JSON.parse(text)); layoutHandle=null;
    setSaveStatus('自動保存を復元しました（未保存）');
  }catch(e){alert('自動保存を復元できませんでした: '+e.message);}
};

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
    OV.style.display='none'; layoutHandle=null;
    if(miss) alert(`読み込みました。${miss} 件は座標が入っていなかったので`
      +`叩き台の位置のままです。`);
  }catch(err){ OVMSG.style.color='#b32d00';
    OVMSG.textContent='読み込めませんでした: '+err.message; }
};
OV.addEventListener('click',e=>{ if(e.target===OV) OV.style.display='none'; });

// ---- Illustrator / Photoshop向け書き出し -------------------------------
const EXOV=document.getElementById('exov'), EXMSG=document.getElementById('exmsg');
const EXSCOPE=document.getElementById('exscope'), EXDPI=document.getElementById('exdpi');
const EXINFO=document.getElementById('exinfo'), EXPNG=document.getElementById('expng');
const masterBlobs=new Map(), masterData=new Map(), masterImgs=new Map();
const visibleCases=()=>CASES.map((c,i)=>i).filter(i=>opts.arch||c.status!=='archived');
function exportBounds(){
  if(EXSCOPE.value==='paper') return {x:0,y:0,w:sheet.w,h:sheet.h};
  let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  visibleCases().forEach(i=>{
    const p=cpos[i],r=p.s/2;
    x0=Math.min(x0,p.x-r); y0=Math.min(y0,p.y-r);
    x1=Math.max(x1,p.x+r); y1=Math.max(y1,p.y+r);
  });
  if(opts.tag) TAGS.forEach((t,j)=>{
    const p=tpos[j],b=tagBox(j);
    x0=Math.min(x0,p.x-b.w/2); y0=Math.min(y0,p.y-b.h/2);
    x1=Math.max(x1,p.x+b.w/2); y1=Math.max(y1,p.y+b.h/2);
  });
  if(!Number.isFinite(x0)) return {x:0,y:0,w:sheet.w,h:sheet.h};
  const pad=10;
  return {x:x0-pad,y:y0-pad,w:x1-x0+2*pad,h:y1-y0+2*pad};
}
function exportEstimate(){
  const b=exportBounds(),dpi=+EXDPI.value,s=dpi/25.4;
  const w=Math.ceil(b.w*s),h=Math.ceil(b.h*s),pixels=w*h;
  const tooBig=pixels>120000000||w>32767||h>32767;
  EXINFO.classList.toggle('bad',tooBig);
  EXINFO.innerHTML=`実寸 <b>${b.w.toFixed(1)} × ${b.h.toFixed(1)}mm</b><br>`+
    `透明PNG: <b>${w.toLocaleString()} × ${h.toLocaleString()}px</b>`+
    `（処理時メモリ目安 ${(pixels*4/1024/1024).toFixed(0)}MB）`+
    (tooBig?'<br><b>この組合せはブラウザの上限を超えます。範囲を「グラフ部分」にするか150dpiを選んでください。</b>':'');
  EXPNG.disabled=tooBig;
}
document.getElementById('export').onclick=()=>{
  EXMSG.textContent=''; EXOV.style.display='flex'; exportEstimate();
};
document.getElementById('exclose').onclick=()=>EXOV.style.display='none';
EXOV.addEventListener('click',e=>{if(e.target===EXOV) EXOV.style.display='none';});
EXSCOPE.onchange=EXDPI.onchange=exportEstimate;

async function masterBlob(i){
  if(masterBlobs.has(i)) return masterBlobs.get(i);
  let blob;
  try{
    const r=await fetch(CASES[i].master,{cache:'no-cache'});
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    blob=await r.blob();
  }catch(e){
    // 単一HTMLを file:// で開いた場合は外部JPEGを読めないため、
    // 埋め込み済みの軽量プレビューへ退避する。
    blob=await (await fetch(CASES[i].img)).blob();
  }
  masterBlobs.set(i,blob); return blob;
}
function blobDataURL(blob){return new Promise((resolve,reject)=>{
  const r=new FileReader(); r.onload=()=>resolve(r.result); r.onerror=()=>reject(r.error);
  r.readAsDataURL(blob);
});}
async function masterDataURL(i){
  if(!masterData.has(i)) masterData.set(i,await blobDataURL(await masterBlob(i)));
  return masterData.get(i);
}
async function masterImage(i){
  if(masterImgs.has(i)) return masterImgs.get(i);
  const src=await masterDataURL(i);
  const im=await new Promise((resolve,reject)=>{
    const x=new Image(); x.onload=()=>resolve(x); x.onerror=()=>reject(new Error('画像を展開できません'));
    x.src=src;
  });
  masterImgs.set(i,im); return im;
}
async function exportAssets(kind){
  const ids=visibleCases();
  EXMSG.style.color='#455a64'; EXMSG.textContent=`高解像度カードを準備中… 0/${ids.length}`;
  let done=0;
  const values=await Promise.all(ids.map(async i=>{
    const value=kind==='data'?await masterDataURL(i):await masterImage(i);
    done++; EXMSG.textContent=`高解像度カードを準備中… ${done}/${ids.length}`;
    return [i,value];
  }));
  return new Map(values);
}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const safeId=s=>String(s).replace(/[^A-Za-z0-9_.-]+/g,'-');
async function makeSVG(){
  const b=exportBounds(),images=await exportAssets('data');
  const tx=-b.x,ty=-b.y,lines=[],cards=[],tags=[];
  if(opts.edge) EDGES.forEach(([tj,ci,v])=>{
    if(!images.has(ci)) return;
    lines.push(`<g id="edge-${tj}-${safeId(CASES[ci].id)}"><line x1="${tpos[tj].x}" y1="${tpos[tj].y}" x2="${cpos[ci].x}" y2="${cpos[ci].y}" stroke="${TAGS[tj].color}" stroke-opacity="${(0.18+0.5*v).toFixed(3)}" stroke-width="${(0.4+2.6*v).toFixed(3)}"/></g>`);
  });
  for(const [i,data] of images){
    const c=CASES[i],p=cpos[i],x=p.x-p.s/2,y=p.y-p.s/2;
    cards.push(`<g id="case-${safeId(c.id)}" data-case-id="${esc(c.id)}"><title>${esc(c.title)}</title><image x="${x}" y="${y}" width="${p.s}" height="${p.s}" href="${data}"/></g>`);
  }
  if(opts.tag) TAGS.forEach((t,j)=>{
    const b0=tagBox(j),p=tpos[j],x=p.x-b0.w/2,y=p.y-b0.h/2;
    const textX=x+b0.accentW+(b0.w-b0.accentW)/2;
    tags.push(`<g id="tag-${j}" data-tag="${esc(t.name)}"><rect x="${x}" y="${y}" width="${b0.w}" height="${b0.h}" rx="${b0.h/2}" fill="${t.lightColor}" stroke="${t.color}" stroke-opacity=".55" stroke-width="${b0.h*.045}"/><circle cx="${x+b0.accentW*.5}" cy="${p.y}" r="${b0.h*.19}" fill="${t.color}"/><text x="${textX}" y="${p.y}" text-anchor="middle" dominant-baseline="central" font-family="Hiragino Sans, Yu Gothic, sans-serif" font-size="${tagPT*PT}" font-weight="700" fill="#17202e">${esc(t.name)}</text></g>`);
  });
  return `<?xml version="1.0" encoding="UTF-8"?>\n`+
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${b.w}mm" height="${b.h}mm" viewBox="0 0 ${b.w} ${b.h}">`+
    `<title>CALL4 ケースマップ</title><desc>背景透明。単位mm。ケース、タグ、エッジを個別グループ化。</desc>`+
    `<g transform="translate(${tx} ${ty})"><g id="edges">${lines.join('')}</g><g id="cases">${cards.join('')}</g><g id="tags">${tags.join('')}</g></g></svg>`;
}
document.getElementById('exsvg').onclick=async()=>{
  try{
    const svg=await makeSVG();
    dl('call4_case_map.svg',new Blob([svg],{type:'image/svg+xml;charset=utf-8'}));
    EXMSG.style.color='#2e7d32'; EXMSG.textContent='SVGを書き出しました';
  }catch(e){EXMSG.style.color='#b32d00'; EXMSG.textContent='書き出せませんでした: '+e.message;}
};
function roundPath(g,x,y,w,h,r){
  g.beginPath();g.moveTo(x+r,y);g.arcTo(x+w,y,x+w,y+h,r);g.arcTo(x+w,y+h,x,y+h,r);
  g.arcTo(x,y+h,x,y,r);g.arcTo(x,y,x+w,y,r);g.closePath();
}
async function makePNG(){
  const b=exportBounds(),dpi=+EXDPI.value,s=dpi/25.4,images=await exportAssets('image');
  const o=document.createElement('canvas');
  o.width=Math.ceil(b.w*s);o.height=Math.ceil(b.h*s);
  const g=o.getContext('2d');g.setTransform(s,0,0,s,-b.x*s,-b.y*s);
  if(opts.edge) EDGES.forEach(([tj,ci,v])=>{
    if(!images.has(ci)) return;
    g.beginPath();g.moveTo(tpos[tj].x,tpos[tj].y);g.lineTo(cpos[ci].x,cpos[ci].y);
    g.strokeStyle=TAGS[tj].color;g.globalAlpha=0.18+0.5*v;g.lineWidth=0.4+2.6*v;g.stroke();
  });
  g.globalAlpha=1;
  images.forEach((im,i)=>{const p=cpos[i];g.drawImage(im,p.x-p.s/2,p.y-p.s/2,p.s,p.s);});
  if(opts.tag) TAGS.forEach((t,j)=>{
    const b0=tagBox(j),p=tpos[j],x=p.x-b0.w/2,y=p.y-b0.h/2;
    g.save();g.shadowColor='rgba(20,30,50,.16)';g.shadowBlur=1.8;g.shadowOffsetY=.5;
    roundPath(g,x,y,b0.w,b0.h,b0.h/2);g.fillStyle=t.lightColor;g.fill();g.restore();
    roundPath(g,x,y,b0.w,b0.h,b0.h/2);g.strokeStyle=t.color;g.globalAlpha=.55;
    g.lineWidth=b0.h*.045;g.stroke();g.globalAlpha=1;
    g.beginPath();g.arc(x+b0.accentW*.5,p.y,b0.h*.19,0,Math.PI*2);g.fillStyle=t.color;g.fill();
    g.font=`700 ${tagPT*PT}px -apple-system,"Hiragino Sans","Yu Gothic",sans-serif`;
    g.textAlign='center';g.textBaseline='middle';g.fillStyle='#17202e';
    g.fillText(t.name,x+b0.accentW+(b0.w-b0.accentW)/2,p.y);
  });
  return new Promise((resolve,reject)=>o.toBlob(b=>b?resolve(b):reject(new Error('PNGの作成に失敗しました')),'image/png'));
}
document.getElementById('expng').onclick=async()=>{
  try{
    const blob=await makePNG();dl(`call4_case_map_${EXDPI.value}dpi.png`,blob);
    EXMSG.style.color='#2e7d32';EXMSG.textContent='透明PNGを書き出しました';
  }catch(e){EXMSG.style.color='#b32d00';EXMSG.textContent='書き出せませんでした: '+e.message;}
};

// ---- 起動 ---------------------------------------------------------------
window.addEventListener('resize',()=>{resize(); draw();});
resize(); applyLayout('cluster',true); fit(); syncInputs(); snap(); check(); spread();
savedState=layoutJSON();
try{if(localStorage.getItem(AUTOSAVE_KEY)) document.getElementById('recover').style.display='inline-block';}catch(e){}
setSaveStatus('新規・自動バックアップ有効');
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

    slim = []
    for c in cases:
        master_path = os.path.join(OUT, "cards", f"{c['id']}.jpg")
        with open(master_path, "rb") as f:
            master_version = hashlib.sha256(f.read()).hexdigest()[:12]
        slim.append({
            **{k: c[k] for k in ("id", "no", "title", "status", "url", "tags",
                                 "max_mm_300dpi", "img")},
            # 内容が変わるたびURLも変え、ブラウザ/CDNの旧画像キャッシュを避ける。
            "master": f"cards/{c['id']}.jpg?v={master_version}",
        })

    html = TEMPLATE
    for tok, val in [
        ("@@ROLL@@", json.dumps(ROLL_SPEC, ensure_ascii=False)),
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
    print(f"[s3] ロール紙: 幅 {ROLL_SPEC['w']}mm / "
          f"既定長 {ROLL_SPEC['default_h']}mm（余白 {MARGIN_MM}mm）")


if __name__ == "__main__":
    main()
