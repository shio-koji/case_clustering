#!/usr/bin/env python3
"""
plan02 addendum: winners-only summary HTML.

Only the two adopted methods:
- Leiden on the semantic space (exclusive partition, the "map")
- NMF K=6 mixture membership (the "tag-redesign basis")
plus their correspondence. Self-contained single file.

Run from the repo root:  python plan02/s8_winners_summary.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

FEATURES_DIR = Path("plan02/features")
RESULTS_DIR = Path("plan02/results")
REPORT_DIR = Path("plan02/report")

TOPIC_COLORS = ["#4C8C2B", "#2B6CB0", "#B0532B", "#8A4FA8", "#C29B2C", "#2BA8A0"]
LEIDEN_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def url(cid):
    return f"https://www.call4.jp/info.php?type=items&id={cid}"


def main():
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in tokens]
    titles = [r["title"] for r in tokens]
    tags = [r["subject_tags"] for r in tokens]
    coords = np.load(FEATURES_DIR / "umap2d.npz")["coords"]

    nmf = json.loads((RESULTS_DIR / "membership_nmf.json").read_text(encoding="utf-8"))
    names = json.loads((RESULTS_DIR / "names_llm.json").read_text(encoding="utf-8"))
    interp = json.loads((RESULTS_DIR / "interpretation.json").read_text(encoding="utf-8"))
    leiden = np.array(json.loads((RESULTS_DIR / "labels_leiden_pca.json").read_text())["labels"])
    ev = json.loads((RESULTS_DIR / "evaluation.json").read_text(encoding="utf-8"))

    lnames = {int(k): v["name"] for k, v in names["leiden_pca"].items() if not k.startswith("_")}
    tnames = {int(k): v["name"] for k, v in names["nmf"].items()}
    dom = np.array(nmf["dominant_topic"])

    # --- Leiden cluster cards with full member lists ---
    lcards = []
    lclusters = interp["hard"]["leiden_pca"]["clusters"]
    for c in sorted(lclusters, key=int):
        cl = lclusters[c]
        members = "".join(
            f'<li><a href="{url(m["id"])}" target="_blank">{m["title"]}</a></li>'
            for m in cl["members"])
        lcards.append(f"""
<div class="card" style="border-top:4px solid {LEIDEN_COLORS[int(c)]}">
  <h4>C{c}: {lnames[int(c)]} <span class="dim">({cl['size']}件)</span></h4>
  <p><b>特徴語:</b> {' / '.join(cl['descriptor_words'][:8])}</p>
  <p><b>代表（medoid）:</b> <a href="{url(cl['medoid']['id'])}" target="_blank">{cl['medoid']['title']}</a></p>
  <details><summary>所属ケース一覧（{cl['size']}件）</summary><ul>{members}</ul></details>
</div>""")

    # --- NMF topic cards ---
    tcards = []
    ttopics = interp["mixture"]["nmf"]["topics"]
    for k in range(6):
        t = ttopics[str(k)]
        reps = "".join(
            f'<li><a href="{url(r["id"])}" target="_blank">{r["title"]}</a>'
            f' <span class="dim">(比率 {r["ratio"]})</span></li>'
            for r in t["representatives"])
        members_idx = np.where(dom == k)[0]
        members = "".join(
            f'<li><a href="{url(ids[i])}" target="_blank">{titles[i]}</a>'
            f' <span class="dim">({np.array(nmf["ratios"])[i, k]:.2f})</span></li>'
            for i in members_idx)
        tcards.append(f"""
<div class="card" style="border-top:4px solid {TOPIC_COLORS[k]}">
  <h4>T{k}: {tnames[k]} <span class="dim">(優勢 {t['size_dominant']}件)</span></h4>
  <p><b>特徴語:</b> {' / '.join(t['descriptor_words'][:8])}</p>
  <p><b>代表（比率上位）:</b></p><ul>{reps}</ul>
  <details><summary>優勢ケース一覧（{t['size_dominant']}件、括弧=このトピックの比率）</summary><ul>{members}</ul></details>
</div>""")

    # --- correspondence cross-tab ---
    tab_rows = []
    for c in range(5):
        cells = "".join(
            f'<td style="background:rgba(43,108,176,{min(int(((leiden==c)&(dom==k)).sum())/12,1)*0.55});">'
            f'{int(((leiden==c)&(dom==k)).sum()) or ""}</td>'
            for k in range(6))
        tab_rows.append(f"<tr><th>C{c} {lnames[c]}</th>{cells}</tr>")
    header = "".join(f"<th>T{k}<br>{tnames[k]}</th>" for k in range(6))
    crosstab = (f"<table class='xtab'><tr><th>Leiden ＼ NMF優勢</th>{header}</tr>"
                + "".join(tab_rows) + "</table>")

    # --- JS data ---
    cases = [{"id": ids[i], "title": titles[i], "url": url(ids[i]),
              "tags": tags[i],
              "x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3),
              "leiden": int(leiden[i]), "dom": int(dom[i]),
              "nmf": [round(float(r), 3) for r in nmf["ratios"][i]],
              "ent": round(float(nmf["entropy_normalized"][i]), 3)}
             for i in range(len(ids))]
    order = np.lexsort((-np.array(nmf["ratios"])[np.arange(len(ids)), dom], dom))

    boot = ev["q2_bootstrap_stability"]
    q1 = ev["q1_internal_validity"]
    align = ev["q3_tag_alignment"]
    ari_between = ev["q2_agreement"]["ari"]["leiden_pca"]["nmf_dom"]

    gen = date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALL4 分類サマリ — 採用2手法</title>
<style>
body {{ font-family: "Hiragino Sans", "Yu Gothic", sans-serif; margin:0; background:#f6f5f2; color:#222; line-height:1.75; }}
.container {{ max-width:1060px; margin:0 auto; padding:26px 20px 70px; }}
h1 {{ font-size:1.4em; border-bottom:3px solid #2B6CB0; padding-bottom:8px; }}
h2 {{ font-size:1.2em; margin-top:2em; border-left:5px solid #2B6CB0; padding-left:10px; }}
h4 {{ margin:0.3em 0; }}
.dim {{ color:#778; font-size:0.85em; }}
.summary {{ background:#eef3fa; border:1px solid #c9d8ee; border-radius:8px; padding:13px 17px; }}
.card {{ background:#fff; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; margin:12px 0; }}
.figure {{ background:#fff; border-radius:8px; padding:12px; margin:14px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); overflow-x:auto; }}
table {{ border-collapse:collapse; background:#fff; font-size:0.85em; margin:10px 0; }}
th,td {{ border:1px solid #ccc; padding:5px 9px; }}
th {{ background:#e8edf5; }}
.xtab td {{ text-align:center; min-width:52px; }}
canvas {{ max-width:100%; }}
#tip {{ position:fixed; display:none; background:rgba(20,25,35,.94); color:#fff; padding:8px 11px; border-radius:6px; font-size:12px; max-width:330px; pointer-events:none; z-index:10; }}
a {{ color:#2B6CB0; }}
details summary {{ cursor:pointer; color:#2B6CB0; }}
li {{ margin:2px 0; }}
</style>
</head>
<body>
<div class="container">

<h1>CALL4 公共訴訟 分類結果サマリ — 採用した2手法だけ</h1>
<p class="dim">生成 {gen} ｜ N=95 ｜ 全手法比較の詳細版は call4_plan02_report.html を参照</p>

<div class="summary">
手法比較（9手法）の結果、採用したのは次の2つ。<br>
<b>① Leiden（意味空間）</b> — 1ケース=1グループの<b>排他分割（5グループ）</b>。地図・概観用。
シルエット0.126（比較中最良）、ブートストラップ安定性 中央値ARI <b>{boot['leiden_pca']['ari_median']}</b>（最安定）。<br>
<b>② NMF K=6</b> — 各ケースを6トピックへの<b>混合比</b>で表す。多ラベル・タグ再設計用。
「一文で説明できるか」テスト唯一の全通過（6/6）、既存タグとの整合 ARI {align['nmf_dom']['ari']:+.3f}（参照的一致）。<br>
両者の一致は ARI {ari_between:+.3f}。<b>語彙（NMF）と意味（Leiden）という別々の情報源が同じ骨格を出した</b>ことが、この区分の信頼性の根拠。
</div>

<h2>① Leiden（意味空間）による5グループ</h2>
<div class="figure">
  <div id="legend-map" style="display:flex;flex-wrap:wrap;gap:10px;margin:4px 0;"></div>
  <canvas id="map" width="1000" height="520"></canvas>
  <p class="dim">UMAP 2D（可視化専用の座標）。ホバーで詳細、クリックでCALL4のケースページ。</p>
</div>
<div class="grid">
{''.join(lcards)}
</div>

<h2>② NMF K=6 による混合メンバーシップ</h2>
<div class="figure">
  <div id="legend-stack" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px;"></div>
  <canvas id="stack" width="1000" height="320"></canvas>
  <p class="dim">各棒=1ケース（優勢トピック順）。複数色が混ざる棒＝複数論点を横断する訴訟。ホバーで内訳。</p>
</div>
<div class="grid">
{''.join(tcards)}
</div>

<h2>③ 2手法の対応関係</h2>
<p>行=Leidenグループ、列=NMF優勢トピック、セル=ケース数。対角的な対応（例: C2↔T0、C3↔T1）が
「別の情報源から同じ構造」の中身。C0（刑事・身体拘束）がT2（入管）とT5（刑事）の両方を受けている＝
意味空間では「身体拘束」として融合し、語彙では2軸に分かれる、という関係も読み取れる。</p>
<div class="figure">{crosstab}</div>

<p class="dim">再現情報: bge-m3全文埋め込み→PCA48d上のLeiden（近傍10, resolution採用値はlabels_leiden_pca.json）／
形態素TF-IDF 7,350語上のNMF K=6。seed=42。詳細は plan02.md・GitHub参照。
本分類は解析目的であり、当事者を類型化して評価する意図はありません。</p>

</div>
<div id="tip"></div>

<script>
const CASES = {json.dumps(cases, ensure_ascii=False)};
const ORDER = {json.dumps([int(i) for i in order])};
const TNAMES = {json.dumps([tnames[k] for k in range(6)], ensure_ascii=False)};
const TCOLORS = {json.dumps(TOPIC_COLORS)};
const LNAMES = {json.dumps([lnames[c] for c in range(5)], ensure_ascii=False)};
const LCOLORS = {json.dumps(LEIDEN_COLORS)};
const tip = document.getElementById('tip');
function showTip(html, ev) {{
  tip.innerHTML = html; tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 350) + 'px';
  tip.style.top = (ev.clientY + 12) + 'px';
}}
function hideTip() {{ tip.style.display = 'none'; }}

// map
const mc = document.getElementById('map'), mctx = mc.getContext('2d');
const xs = CASES.map(c=>c.x), ys = CASES.map(c=>c.y);
const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
const px = c => 30 + (c.x-xmin)/(xmax-xmin)*(mc.width-60);
const py = c => 25 + (c.y-ymin)/(ymax-ymin)*(mc.height-50);
CASES.forEach(c => {{
  mctx.beginPath(); mctx.arc(px(c), py(c), 6.5, 0, Math.PI*2);
  mctx.fillStyle = LCOLORS[c.leiden]; mctx.globalAlpha = 0.85; mctx.fill();
  mctx.globalAlpha = 1; mctx.strokeStyle = '#fff'; mctx.stroke();
}});
const lm = document.getElementById('legend-map');
LNAMES.forEach((n,c) => lm.insertAdjacentHTML('beforeend',
  `<span style="font-size:12px"><span style="display:inline-block;width:11px;height:11px;background:${{LCOLORS[c]}};margin-right:4px;border-radius:6px"></span>C${{c}} ${{n}}</span>`));
function nearest(ev) {{
  const r = mc.getBoundingClientRect();
  const mx = (ev.clientX-r.left)*(mc.width/r.width), my = (ev.clientY-r.top)*(mc.height/r.height);
  let best=null, bd=150;
  CASES.forEach(c => {{ const d=(px(c)-mx)**2+(py(c)-my)**2; if (d<bd) {{bd=d; best=c;}} }});
  return best;
}}
mc.addEventListener('mousemove', ev => {{
  const c = nearest(ev);
  if (!c) {{ hideTip(); mc.style.cursor='default'; return; }}
  mc.style.cursor = 'pointer';
  const mix = c.nmf.map((v,t)=> v>0.15 ? TNAMES[t]+' '+(v*100).toFixed(0)+'%' : null).filter(Boolean).join(' / ');
  showTip(`<b>${{c.title}}</b><br>Leiden: C${{c.leiden}} ${{LNAMES[c.leiden]}}<br>NMF: ${{mix}}<br><span style="color:#9db">tags: ${{c.tags.join('・')||'—'}}</span>`, ev);
}});
mc.addEventListener('mouseleave', hideTip);
mc.addEventListener('click', ev => {{ const c = nearest(ev); if (c) window.open(c.url, '_blank'); }});

// stacked bar
const sc = document.getElementById('stack'), sctx = sc.getContext('2d');
const BW = (sc.width - 20) / CASES.length;
ORDER.forEach((ci, pos) => {{
  const c = CASES[ci]; let y = sc.height - 16;
  c.nmf.forEach((r, t) => {{
    const h = r * (sc.height - 28);
    sctx.fillStyle = TCOLORS[t];
    sctx.fillRect(12 + pos*BW, y-h, Math.max(BW-1, 1.5), h);
    y -= h;
  }});
}});
const lg = document.getElementById('legend-stack');
TNAMES.forEach((n,t) => lg.insertAdjacentHTML('beforeend',
  `<span style="font-size:12px"><span style="display:inline-block;width:11px;height:11px;background:${{TCOLORS[t]}};margin-right:4px;border-radius:2px"></span>T${{t}} ${{n}}</span>`));
sc.addEventListener('mousemove', ev => {{
  const r = sc.getBoundingClientRect();
  const pos = Math.floor(((ev.clientX-r.left)*(sc.width/r.width) - 12) / BW);
  if (pos < 0 || pos >= ORDER.length) {{ hideTip(); return; }}
  const c = CASES[ORDER[pos]];
  const mix = c.nmf.map((v,t)=> v>0.08 ? `T${{t}} ${{TNAMES[t]}}: ${{(v*100).toFixed(0)}}%` : null).filter(Boolean).join('<br>');
  showTip(`<b>${{c.title}}</b><br>${{mix}}<br><span style="color:#9db">entropy ${{c.ent}}</span>`, ev);
}});
sc.addEventListener('mouseleave', hideTip);
</script>
</body>
</html>"""
    out = REPORT_DIR / "call4_winners_summary.html"
    out.write_text(html, encoding="utf-8")
    print(f"Winners summary saved: {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
