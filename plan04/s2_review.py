#!/usr/bin/env python3
"""
plan04 Stage 2: K-sweep review HTML — pick a semantically satisfying K.

Renders (from plan04/results/nmf_sweep.json):
  - a genealogy alluvial (columns = K=4..14, ribbons = each topic's parent at
    K-1; colored by its K=4 root lineage) to see how topics split as K grows
  - per-K topic cards (top words, dominant size, tiny / generic-word flags),
    switchable by K
  - remaining generic-word candidates for a possible next stopword pass

Run from the repo root:  python plan04/s2_review.py
"""

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("plan04/results")
OUT = Path("plan04/report/call4_plan04_ksweep_review.html")
OUT.parent.mkdir(parents=True, exist_ok=True)

JP_FONT_PATH = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
                "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/"
                "YuGothic-Medium.otf")
fm.fontManager.addfont(JP_FONT_PATH)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

def jp(size=9):
    return fm.FontProperties(fname=JP_FONT_PATH, size=size)

ROOT_COLORS = ["#C0392B", "#27AE60", "#2E86C1", "#8E44AD", "#E67E22", "#16A085",
               "#D4AC0D", "#7F8C8D"]


def svg_of(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    s = buf.getvalue().decode("utf-8"); return s[s.find("<svg"):]


def fig_alluvial(sweep, kmin, kmax):
    Ks = list(range(kmin, kmax + 1))
    # y position of each topic within its K column (evenly spaced, centered)
    def ypos(K):
        n = len(sweep[str(K)]["topics"])
        return {t: (t - (n - 1) / 2) for t in range(n)}
    yp = {K: ypos(K) for K in Ks}
    # root lineage of each (K, t): follow parent links back to kmin
    root = {(kmin, t): t for t in range(len(sweep[str(kmin)]["topics"]))}
    for K in Ks[1:]:
        gen = {g["topic"]: g["from_prev_topic"] for g in sweep[str(K)]["genealogy_from_prevK"]}
        for t in range(len(sweep[str(K)]["topics"])):
            root[(K, t)] = root[(K - 1, gen[t])]

    fig, ax = plt.subplots(figsize=(19, 9))
    # ribbons parent(K-1) -> child(K)
    for K in Ks[1:]:
        gen = {g["topic"]: g["from_prev_topic"] for g in sweep[str(K)]["genealogy_from_prevK"]}
        for t in range(len(sweep[str(K)]["topics"])):
            p = gen[t]
            xs = np.linspace(K - 1, K, 30)
            y0, y1 = yp[K - 1][p], yp[K][t]
            ys = y0 + (y1 - y0) * (1 - np.cos(np.linspace(0, np.pi, 30))) / 2
            ax.plot(xs, ys, color=ROOT_COLORS[root[(K, t)] % len(ROOT_COLORS)],
                    alpha=0.4, lw=1.2, zorder=1)
    # every node labeled with its #1 (highest-weight) word
    for K in Ks:
        for t, tp in enumerate(sweep[str(K)]["topics"]):
            col = ROOT_COLORS[root[(K, t)] % len(ROOT_COLORS)]
            ax.text(K, yp[K][t], f"{tp['top_words'][0]}\n({tp['dominant_size']})",
                    ha="center", va="center", fontproperties=jp(8), color=col, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col, lw=1.1))
    ax.set_xticks(Ks); ax.set_xlabel("トピック数 K", fontproperties=jp(11))
    ax.set_yticks([]); ax.set_xlim(kmin - 0.6, kmax + 0.6)
    ax.set_ylim(-(kmax) / 2 - 1, kmax / 2 + 1)
    ax.set_title("トピックの系譜: 各ノード＝そのトピックの最重み語（括弧=主となるケース数）／色＝K=4の起源",
                 fontproperties=jp(12))
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    return svg_of(fig)


def main():
    d = json.loads((RESULTS_DIR / "nmf_sweep.json").read_text(encoding="utf-8"))
    sweep, kmin, kmax = d["sweep"], d["kmin"], d["kmax"]
    alluvial = fig_alluvial(sweep, kmin, kmax)

    # per-K topic cards data
    kdata = {}
    for K in range(kmin, kmax + 1):
        cards = []
        for i, t in enumerate(sweep[str(K)]["topics"]):
            flags = []
            if t["tiny"]:
                flags.append('<span class="flag tiny">極小</span>')
            if t["generic_words"]:
                flags.append(f'<span class="flag gen">一般語: {"/".join(t["generic_words"])}</span>')
            cards.append(
                f'<div class="tcard"><div class="th">T{i} '
                f'<span class="dim">(優勢{t["dominant_size"]}件)</span> {"".join(flags)}</div>'
                f'<div class="tw">{" / ".join(t["top_words"][:8])}</div></div>')
        kdata[K] = {"err": sweep[str(K)]["reconstruction_err"], "cards": "".join(cards)}

    btns = "".join(f'<button class="kb" data-k="{K}">{K}</button>' for K in range(kmin, kmax + 1))
    gen = "".join(
        f'<li><b>{g["word"]}</b> <span class="dim">(DF {g["df_ratio"]:.0%}, K={g["appears_in_K"]})</span></li>'
        for g in d["generic_candidates"])

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>plan04 NMF K-sweep レビュー</title>
<style>
body {{ font-family:"Hiragino Sans","Yu Gothic",sans-serif; margin:0; background:#f7f6f3; color:#222; line-height:1.8; }}
.container {{ max-width:1040px; margin:0 auto; padding:28px 20px 70px; }}
h1 {{ font-size:1.4em; border-bottom:3px solid #2E86C1; padding-bottom:9px; }}
h2 {{ font-size:1.18em; margin-top:2em; border-left:5px solid #2E86C1; padding-left:10px; }}
.dim {{ color:#778; font-size:.85em; }}
.note {{ background:#eef3fa; border:1px solid #c9d8ee; border-radius:8px; padding:12px 16px; }}
.figure {{ background:#fff; border-radius:8px; padding:12px; margin:14px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); overflow-x:auto; }}
.kb {{ background:#e8edf5; border:1px solid #bcd; border-radius:5px; padding:5px 11px; margin:2px; cursor:pointer; font-size:.9em; font-family:inherit; }}
.kb.active {{ background:#2E86C1; color:#fff; font-weight:bold; }}
.tcard {{ background:#fff; border:1px solid #e3e0d8; border-radius:7px; padding:8px 12px; margin:6px 0; }}
.th {{ font-weight:bold; font-size:.92em; }}
.tw {{ color:#444; font-size:.92em; }}
.flag {{ font-size:.72em; padding:1px 6px; border-radius:8px; margin-left:6px; font-weight:normal; }}
.flag.tiny {{ background:#fde8e8; color:#a33; }}
.flag.gen {{ background:#fdf3e0; color:#96631a; }}
</style></head>
<body><div class="container">

<h1>plan04: NMF トピック数スイープのレビュー（K=4〜14）</h1>
<p class="dim">N=95 ／ 語彙 {d['vocab_size']} ／ ストップワード {d['n_stopwords']}語 ／ seed=42</p>

<div class="note">
<b>目的.</b> TF-IDF＋NMFで、ケースと語彙をソフト分類。K=4〜14を試し、
<b>意味的にしっくりくるトピック数</b>を選ぶための材料です。再構成誤差には明確な折れ目（肘）が無いため、
<b>「各トピックが一文で言えるか」「過分割していないか」で判断</b>します。
</div>

<h2>1. トピックの系譜（Kを増やすと何が枝分かれするか）</h2>
<p>左（K=4）から右（K=14）へ、各トピックが親からどう分岐するか。<b>各ノードのラベルは、そのトピックで
最も重みの大きい第1位語</b>（NMFのH行列の最大重み語）で、括弧内はそのトピックを主とするケース数。色は起源（K=4の4トピック）。
分岐が意味を持つうちは細分化が有効、同じ語の重複や極小トピックが増え始めたら過分割のサインです。</p>
<p class="dim">※ラベルは<b>機械的な第1位語であり、LLMによる要約ではありません</b>。
LLMに要約ラベルを付けさせることも可能です（その場合は根拠の特徴語を併記）。</p>
<div class="figure">{alluvial}</div>

<h2>2. 各Kのトピック一覧</h2>
<p>ボタンでKを切替。<span class="flag gen">一般語</span>フラグはストップワード追加の候補、
<span class="flag tiny">極小</span>は過分割の兆候。</p>
<div id="kbtns">{btns}</div>
<p class="dim" id="kmeta"></p>
<div id="kcards"></div>

<h2>3. 残っている一般語の候補（次のストップワード検討用）</h2>
<ul>{gen}</ul>
<p class="dim">※「行政」は論点語のため意図的に残置。地名（東京）・文書種別（意見書）も論点性があり残す判断。</p>

</div>
<script>
const KDATA = {json.dumps(kdata, ensure_ascii=False)};
const btns=document.getElementById('kbtns'), cards=document.getElementById('kcards'), meta=document.getElementById('kmeta');
function show(K){{
  cards.innerHTML=KDATA[K].cards;
  meta.textContent=`K=${{K}} ／ 再構成誤差 ${{KDATA[K].err}} ／ ${{KDATA[K].cards.match(/tcard/g).length}}トピック`;
  btns.querySelectorAll('.kb').forEach(b=>b.classList.toggle('active', b.dataset.k==K));
}}
btns.addEventListener('click', e=>{{ if(e.target.dataset.k) show(e.target.dataset.k); }});
show('11');
</script>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"review saved: {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
