#!/usr/bin/env python3
"""
plan02 public report: the story in three acts.

1. All methods tried, organized very concisely (verdict table)
2. The data-based rationale for K=6 (error curve + K=5/6/7 topic words)
3. Conclusion: the mixture-membership result only

Self-contained single HTML for sharing.
Run from the repo root:  python plan02/s9_public_report.py
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

TOPIC_COLORS = ["#4C8C2B", "#2B6CB0", "#B0532B", "#8A4FA8", "#C29B2C", "#2BA8A0"]


def url(cid):
    return f"https://www.call4.jp/info.php?type=items&id={cid}"


def svg_of(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue().decode("utf-8")
    return s[s.find("<svg"):]


def fig_error_curve(kselect):
    ks = sorted(int(k) for k in kselect["errors"])
    errs = [kselect["errors"][str(k)] for k in ks]
    drops = [errs[i - 1] - errs[i] for i in range(1, len(errs))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(ks, errs, "o-", color="#2B6CB0")
    axes[0].axvline(6, color="#B0532B", ls="--", lw=1)
    axes[0].set_xlabel("トピック数 K", fontproperties=jp(10))
    axes[0].set_ylabel("再構成誤差", fontproperties=jp(10))
    axes[0].set_title("誤差はKとともに単調減少（明確な折れ＝肘がない）", fontproperties=jp(10))
    axes[0].grid(alpha=0.3)
    axes[1].bar([f"{k-1}→{k}" for k in ks[1:]], drops, color="#8fb0d4")
    axes[1].set_ylabel("誤差の改善幅", fontproperties=jp(10))
    axes[1].set_title("1トピック追加ごとの改善幅（ほぼ横並び）", fontproperties=jp(10))
    axes[1].grid(alpha=0.3, axis="y")
    for ax in axes:
        ax.tick_params(labelsize=8)
    return svg_of(fig)


def k_compare_table(kselect):
    tw = kselect["topic_words"]

    def cell(words, note="", style=""):
        n = f'<div class="note">{note}</div>' if note else ""
        return f'<td style="{style}">{" / ".join(words[:5])}{n}</td>'

    bad = "background:#fdecea;"
    good = "background:#edf7ed;"
    rows = []
    # align rows loosely by theme for readability
    themes = [
        ("地域・環境", tw["5"]["0"], tw["6"]["0"], tw["7"]["0"], "", "", ""),
        ("情報公開", tw["5"]["1"], tw["6"]["1"], tw["7"]["1"], "", "", ""),
        ("刑事手続", None, tw["6"]["5"], tw["7"]["2"], "", "", ""),
        ("入管・難民", None, tw["6"]["2"], tw["7"]["5"], "", "", ""),
        ("（刑事＋入管の混在）", tw["5"]["2"], None, None, "混在＝Kが小さすぎる兆候", "", ""),
        ("家族・ジェンダー", tw["5"]["3"], tw["6"]["3"], tw["7"]["3"], "", "", ""),
        ("選挙・政治参加", tw["5"]["4"], tw["6"]["4"], tw["7"]["4"], "", "", ""),
        ("（定型語の寄せ集め）", None, None, tw["7"]["6"], "", "", "サイト定型語が混入＝Kが大きすぎる兆候"),
    ]
    for theme, k5, k6, k7, n5, n6, n7 in themes:
        c5 = cell(k5, n5, bad if "混在" in theme else "") if k5 else "<td class='dim'>—</td>"
        c6 = cell(k6, n6, good) if k6 else "<td class='dim'>—</td>"
        c7 = cell(k7, n7, bad if "寄せ集め" in theme else "") if k7 else "<td class='dim'>—</td>"
        rows.append(f"<tr><th>{theme}</th>{c5}{c6}{c7}</tr>")
    return ("<table class='ktab'><tr><th>テーマ</th><th>K=5</th>"
            "<th>K=6（採用）</th><th>K=7</th></tr>" + "".join(rows) + "</table>")


def main():
    tokens = json.loads((FEATURES_DIR / "tokens.json").read_text(encoding="utf-8"))
    tokens.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in tokens]
    titles = [r["title"] for r in tokens]
    tags = [r["subject_tags"] for r in tokens]

    kselect = json.loads((RESULTS_DIR / "nmf_kselect.json").read_text(encoding="utf-8"))
    nmf = json.loads((RESULTS_DIR / "membership_nmf.json").read_text(encoding="utf-8"))
    names = json.loads((RESULTS_DIR / "names_llm.json").read_text(encoding="utf-8"))
    interp = json.loads((RESULTS_DIR / "interpretation.json").read_text(encoding="utf-8"))
    ev = json.loads((RESULTS_DIR / "evaluation.json").read_text(encoding="utf-8"))

    tnames = {int(k): v["name"] for k, v in names["nmf"].items()}
    dom = np.array(nmf["dominant_topic"])
    ratios = np.array(nmf["ratios"])
    order = np.lexsort((-ratios[np.arange(len(ids)), dom], dom))

    err_svg = fig_error_curve(kselect)
    ktable = k_compare_table(kselect)

    # topic cards (conclusion)
    cards = []
    for k in range(6):
        t = interp["mixture"]["nmf"]["topics"][str(k)]
        reps = "".join(
            f'<li><a href="{url(r["id"])}" target="_blank">{r["title"]}</a></li>'
            for r in t["representatives"])
        cards.append(f"""
<div class="card" style="border-top:4px solid {TOPIC_COLORS[k]}">
  <h4>T{k}: {tnames[k]}</h4>
  <p class="dim">このトピックが最も強いケース: {t['size_dominant']}件</p>
  <p><b>特徴語:</b> {' / '.join(t['descriptor_words'][:7])}</p>
  <p><b>代表的なケース:</b></p><ul>{reps}</ul>
</div>""")

    # multi-issue callout (boundary cases with their mixtures)
    id2i = {c: i for i, c in enumerate(ids)}
    multi = []
    for b in ev["q3_boundary_cases"]["in_both_lists"]:
        i = id2i[b["id"]]
        mix = " ＋ ".join(f"{tnames[t]} {ratios[i, t]*100:.0f}%"
                          for t in np.argsort(-ratios[i])[:3] if ratios[i, t] > 0.1)
        multi.append(f'<li><a href="{url(b["id"])}" target="_blank">{b["title"]}</a>'
                     f'<br><span class="dim">{mix}</span></li>')

    cases = [{"title": titles[i], "url": url(ids[i]), "tags": tags[i],
              "nmf": [round(float(r), 3) for r in ratios[i]],
              "ent": round(float(nmf["entropy_normalized"][i]), 3)}
             for i in range(len(ids))]

    gen = date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CALL4公共訴訟95件の分類 — データが示した6つの軸</title>
<style>
body {{ font-family:"Hiragino Sans","Yu Gothic",sans-serif; margin:0; background:#f7f6f3; color:#222; line-height:1.8; }}
.container {{ max-width:1000px; margin:0 auto; padding:30px 20px 70px; }}
h1 {{ font-size:1.45em; border-bottom:3px solid #2B6CB0; padding-bottom:10px; }}
h2 {{ font-size:1.22em; margin-top:2.4em; border-left:5px solid #2B6CB0; padding-left:10px; }}
h4 {{ margin:0.3em 0; }}
.dim {{ color:#778; font-size:0.86em; }}
.lead {{ background:#eef3fa; border:1px solid #c9d8ee; border-radius:8px; padding:14px 18px; }}
.card {{ background:#fff; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin:12px 0; }}
.figure {{ background:#fff; border-radius:8px; padding:12px; margin:14px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); overflow-x:auto; }}
table {{ border-collapse:collapse; background:#fff; font-size:0.85em; margin:12px 0; }}
th,td {{ border:1px solid #ccc; padding:6px 10px; text-align:left; vertical-align:top; }}
th {{ background:#e8edf5; }}
.ktab td {{ font-size:0.92em; }}
.ktab .note {{ color:#a33; font-size:0.85em; margin-top:2px; }}
.verdict td:last-child {{ text-align:center; font-size:1.05em; }}
canvas {{ max-width:100%; }}
#tip {{ position:fixed; display:none; background:rgba(20,25,35,.94); color:#fff; padding:8px 11px; border-radius:6px; font-size:12px; max-width:330px; pointer-events:none; z-index:10; }}
a {{ color:#2B6CB0; }}
.callout {{ background:#fbf6e9; border:1px solid #e6d9b0; border-radius:8px; padding:12px 16px; }}
li {{ margin:3px 0; }}
</style>
</head>
<body>
<div class="container">

<h1>CALL4 公共訴訟95件の分類<br>— データが示した6つの軸と「混合」という見方</h1>
<p class="dim">生成 {gen} ｜ 対象: CALL4掲載の公共訴訟95件（ケース名・概要・本文）｜ 手法詳細は文末</p>

<div class="lead">
公共訴訟プラットフォーム <a href="https://www.call4.jp" target="_blank">CALL4</a> の95ケースの本文を、
機械学習で「テーマの近さ」に基づき分類し直しました。複数の手法を試して突き合わせた結果、
<b>手法が変わっても繰り返し現れる6つの軸</b>が見つかり、さらに各ケースは1つの軸に収まるのではなく
<b>複数の軸の「混合」</b>として表すのが実態に合うことがわかりました。
このページでは ①試した手法の全体像 → ②「6」という数の根拠 → ③結論（混合の見取り図）の順に示します。
</div>

<h2>1. 試した手法の全体像</h2>
<p>テキストの数値化は2系統——<b>語彙</b>（どんな単語を使うか: TF-IDF）と<b>意味</b>（文章全体の内容: 多言語埋め込みbge-m3）——を用意し、
その上で分類アルゴリズムを一通り走らせて比較しました（次元圧縮後に実行。全て乱数固定・同一入力）。</p>
<table class="verdict">
<tr><th>数値化</th><th>分類法</th><th>結果ひとこと</th><th>判定</th></tr>
<tr><td rowspan="4">意味<br>(bge-m3)</td><td>Leiden（グラフ）</td><td>まとまり最良・データを8割に間引いても7割一致する安定性</td><td><b>◎ 採用</b>（排他分割）</td></tr>
<tr><td>k-means</td><td>良好。Leidenとほぼ同じ分割</td><td>○</td></tr>
<tr><td>階層型</td><td>良好。Leidenと一致度0.84</td><td>○</td></tr>
<tr><td>BERTopic相当</td><td>62件が1つの巨大クラスタに</td><td>△</td></tr>
<tr><td rowspan="4">語彙<br>(TF-IDF)</td><td><b>NMF（混合）</b></td><td>6トピック全てが一言で説明可能（唯一の全通過）</td><td><b>◎ 採用</b>（混合表現）</td></tr>
<tr><td>Leiden</td><td>良好。意味側の結果と一致度0.5</td><td>○</td></tr>
<tr><td>k-means</td><td>1つの混成クラスタ（31件）が発生</td><td>△</td></tr>
<tr><td>階層型</td><td>崩壊（79件が1塊＋断片）</td><td>✗</td></tr>
<tr><td>語彙</td><td>LDA（混合）</td><td>6中3トピックが解釈不能。NMFに完敗</td><td>✗</td></tr>
<tr><td>既存タグのみ</td><td>階層型（基準線）</td><td>56件が1塊——人手タグだけでは構造が出ない</td><td>基準線</td></tr>
</table>
<p><b>重要な点</b>: 成績の良い手法たち（意味×Leiden／語彙×NMFなど）は、<b>互いに独立した情報源から
ほぼ同じ骨格</b>——地域環境・情報公開・入管難民・家族ジェンダー・選挙・刑事手続——を出しました。
1つの手法の偶然ではなく、複数の目で同じ構造が見えたことが、以下の結果の信頼性の土台です。</p>

<h2>2. なぜ「6つ」なのか — データによる根拠</h2>
<p>トピック数Kは4〜10を全て試しました。数値（再構成誤差）は判断材料になりませんでした——
Kを増やすほど誤差は機械的に減り続け、「ここで止めるべき」という折れ目（肘）が存在しないからです。</p>
<div class="figure">{err_svg}</div>
<p>そこで各Kの<b>トピックの中身</b>を比較すると、判断は明確になります。</p>
<div class="figure">{ktable}</div>
<ul>
<li><b>K=5では「刑事手続」と「入管収容」が1つに混ざる</b>（収容/刑事/逮捕/入管が同居）。
法的にも実務的にも別物であり、分けられるべき区別。</li>
<li><b>K=6で初めてこの2つが分離</b>し、6トピック全てが一言で説明できる状態になる。</li>
<li><b>K=7では「アーカイブ_プロジェクト」などサイトの定型語で構成された無意味なトピックが出現</b>。
内容の分解が限界に達し、ノイズを拾い始めた兆候。</li>
</ul>
<p>補強証拠として、トピック数を<b>自動で</b>決める別手法も同じスケールを支持しています
（グラフ法Leidenは5グループを自動選択、密度法は4グループ＋外れ値5件）。
つまり「6」は恣意的な設定ではなく、<b>K=5では粗すぎ・K=7では壊れる、という挟み撃ちで決まった値</b>です。</p>

<h2>3. 結論 — 95件は「6軸の混合」で表せる</h2>
<p>採用した最終結果は、各ケースを6軸への<b>混合比</b>（合計100%）で表すものです。
下の図は95件それぞれを1本の棒にし、混合比を色で積み上げたもの。
<b>単色に近い棒＝単一論点の訴訟、複数色の棒＝複数の論点を横断する訴訟</b>です。棒にカーソルを載せると内訳が見えます。</p>
<div class="figure">
  <div id="legend" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px;"></div>
  <canvas id="stack" width="960" height="330"></canvas>
</div>

<div class="grid">
{''.join(cards)}
</div>

<h3>複数の論点を横断するケース（混合表現だから見えるもの）</h3>
<div class="callout">
<p>次の5件は「混合の偏りが小さい」「どの分類手法でも所属が安定しない」の両方に該当した、
<b>本質的に複数論点にまたがる訴訟</b>です。1つのカテゴリに押し込む分類では、この性質は失われます。</p>
<ul>{''.join(multi)}</ul>
</div>

<h3>この結果が示唆すること</h3>
<ul>
<li>既存のタグ「公正な手続」（43件が該当）は6軸の全てに分散しており、分類としては機能していない。
争点ベースの軸への再編が有効。</li>
<li>訴訟の多くは1〜2軸に集中する一方、境界ケースが確かに存在する。
<b>主タグ＋副タグ（または比率）</b>という運用が実態に忠実。</li>
<li>「労働・生活者の権利」は今回6軸に含めなかったが、複数の手法で独立グループとして繰り返し出現しており、
7つ目の軸の有力候補。</li>
</ul>

<h2>付記: 手法と限界</h2>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>データ</td><td>CALL4公開ケース95件（2026-07-18取得）。ケース名＋概要＋本文</td></tr>
<tr><td>数値化</td><td>形態素解析（SudachiPy）＋TF-IDF 7,350語／多言語埋め込みbge-m3（全文）</td></tr>
<tr><td>採用手法</td><td>混合: NMF K=6（TF-IDF）／排他検証: Leiden他8手法との比較・ブートストラップ安定性</td></tr>
<tr><td>再現性</td><td>乱数シード42固定。コード・中間データはGitHubリポジトリに保存</td></tr>
<tr><td>限界</td><td>95件は統計的に小規模で、本結果は探索・仮説生成のためのもの。訴状・判決文の全文は未使用。
トピック数6は解釈可能性に基づく選択で、5〜7に議論の余地は残る</td></tr>
</table>
<p class="dim">本分類は解析目的であり、訴訟当事者を類型化して評価する意図はありません。
ケース本文の著作権はCALL4および執筆者に帰属します。各ケースの詳細・支援は各リンク先（CALL4）をご覧ください。</p>

</div>
<div id="tip"></div>

<script>
const CASES = {json.dumps(cases, ensure_ascii=False)};
const ORDER = {json.dumps([int(i) for i in order])};
const TNAMES = {json.dumps([tnames[k] for k in range(6)], ensure_ascii=False)};
const TCOLORS = {json.dumps(TOPIC_COLORS)};
const tip = document.getElementById('tip');
const sc = document.getElementById('stack'), ctx = sc.getContext('2d');
const BW = (sc.width - 20) / CASES.length;
ORDER.forEach((ci, pos) => {{
  const c = CASES[ci]; let y = sc.height - 16;
  c.nmf.forEach((r, t) => {{
    const h = r * (sc.height - 28);
    ctx.fillStyle = TCOLORS[t];
    ctx.fillRect(12 + pos*BW, y-h, Math.max(BW-1, 1.5), h);
    y -= h;
  }});
}});
ctx.fillStyle = '#555'; ctx.font = '11px sans-serif';
ctx.fillText('← 1本 = 1ケース（優勢な軸ごとに整列）', 12, sc.height - 3);
const lg = document.getElementById('legend');
TNAMES.forEach((n, t) => lg.insertAdjacentHTML('beforeend',
  `<span style="font-size:12px"><span style="display:inline-block;width:11px;height:11px;background:${{TCOLORS[t]}};margin-right:4px;border-radius:2px"></span>T${{t}} ${{n}}</span>`));
let cur = -1;
sc.addEventListener('mousemove', ev => {{
  const r = sc.getBoundingClientRect();
  const pos = Math.floor(((ev.clientX-r.left)*(sc.width/r.width) - 12) / BW);
  if (pos < 0 || pos >= ORDER.length) {{ tip.style.display='none'; cur=-1; return; }}
  cur = pos;
  const c = CASES[ORDER[pos]];
  const mix = c.nmf.map((v,t)=> v>0.08 ? `T${{t}} ${{TNAMES[t]}}: ${{(v*100).toFixed(0)}}%` : null).filter(Boolean).join('<br>');
  tip.innerHTML = `<b>${{c.title}}</b><br>${{mix}}<br><span style="color:#9db">クリックでCALL4のページへ</span>`;
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 350) + 'px';
  tip.style.top = (ev.clientY + 12) + 'px';
  sc.style.cursor = 'pointer';
}});
sc.addEventListener('mouseleave', () => {{ tip.style.display='none'; cur=-1; }});
sc.addEventListener('click', () => {{ if (cur >= 0) window.open(CASES[ORDER[cur]].url, '_blank'); }});
</script>
</body>
</html>"""
    out = REPORT_DIR / "call4_public_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Public report saved: {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
