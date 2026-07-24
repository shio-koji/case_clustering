#!/usr/bin/env python3
"""
plan04 Report A: why K=6 (or K=12) — the K-selection rationale, accessibly.

Reads plan04/results/kmetrics.json (+ nmf_sweep.json for the error curve),
writes a concise HTML that explains, with light glosses on the jargon, why
K=6 is the robust choice and K=12 the finer-exploration alternative.

Run from the repo root:  python plan04/s4_kchoice_report.py
"""

import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

OUT = Path("plan04/report/call4_plan04_kchoice.html")
JP = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
      "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/YuGothic-Medium.otf")
fm.fontManager.addfont(JP)
plt.rcParams["font.sans-serif"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"; plt.rcParams["axes.unicode_minus"] = False
def jp(s=10): return fm.FontProperties(fname=JP, size=s)


def fig_metrics(m, err_Ks, err):
    Ks = m["Ks"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    def panel(a, y, title, color, kind):
        a.plot(Ks, y, "o-", color=color); a.set_xticks(Ks)
        a.set_title(title, fontproperties=jp(10)); a.grid(alpha=.3); a.set_xlabel("K", fontproperties=jp(9))
        if kind in ("peak", "peak5"):
            cand = [(K, v) for K, v in zip(Ks, y) if kind == "peak" or K >= 5]
            b = max(cand, key=lambda kv: kv[1])[0]
            a.axvline(b, color=color, ls="--", lw=1.3)
            a.text(b, max(y), f" K={b}", color=color, fontproperties=jp(9), va="top")
            if kind == "peak5":
                a.text(0.03, 0.05, "※K=4は自明に粗い解のため除外", transform=a.transAxes,
                       fontproperties=jp(8), color="#888")
        elif kind == "mono":
            a.text(.5, .06, "単調＝K上限は決められない", transform=a.transAxes, ha="center",
                   fontproperties=jp(8), color="#888")
    panel(ax[0, 0], err, "再構成誤差（エルボー無し＝ほぼ直線）", "#555", "mono")
    ax[0, 0].set_xticks(err_Ks)
    panel(ax[0, 1], m["stability"], "安定性 stability（ピーク＝良い）", "#27AE60", "peak5")
    panel(ax[0, 2], m["tag_ari"], "既存タグとの一致 ARI（ピーク＝良い）", "#8E44AD", "peak")
    panel(ax[1, 0], m["coherence"], "整合性 coherence", "#2E86C1", "mono")
    panel(ax[1, 1], m["redundancy"], "冗長性 redundancy", "#C0392B", "mono")
    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.5,
                  "タグ一致は K=6 が明確なピーク。\n安定性は自明に粗い K=4 を除くと K=6。\n"
                  "単調な指標は上限判断に使えない。\n\n→ 頑健性なら K=6\n→ 細かい探索なら K=11–12",
                  fontproperties=jp(11), va="center")
    fig.suptitle("K を選ぶための指標（K=4〜14）", fontproperties=jp(13)); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="svg", bbox_inches="tight"); plt.close(fig)
    s = buf.getvalue().decode("utf-8"); return s[s.find("<svg"):]


def main():
    m = json.loads(Path("plan04/results/kmetrics.json").read_text(encoding="utf-8"))
    sw = json.loads(Path("plan04/results/nmf_sweep.json").read_text(encoding="utf-8"))
    eKs = list(range(sw["kmin"], sw["kmax"] + 1))
    err = [sw["sweep"][str(K)]["reconstruction_err"] for K in eKs]
    svg = fig_metrics(m, eKs, err)
    st6 = m["stability"][m["Ks"].index(6)]; tg6 = m["tag_ari"][m["Ks"].index(6)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>plan04: トピック数Kの決め方</title>
<style>
body{{font-family:"Hiragino Sans",sans-serif;margin:0;background:#f7f6f3;color:#222;line-height:1.85}}
.c{{max-width:960px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:1.4em;border-bottom:3px solid #2E86C1;padding-bottom:9px}}
h2{{font-size:1.18em;margin-top:1.9em;border-left:5px solid #2E86C1;padding-left:10px}}
.dim{{color:#778;font-size:.85em}}
.note{{background:#eef3fa;border:1px solid #c9d8ee;border-radius:8px;padding:13px 17px}}
.term{{background:#eef6f0;border:1px solid #bcdcc8;border-radius:8px;padding:10px 15px;margin:10px 0;font-size:.92em}}
.term b{{color:#2f7a4d}}
.fig{{background:#fff;border-radius:8px;padding:12px;margin:14px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow-x:auto}}
table{{border-collapse:collapse;background:#fff;font-size:.9em;margin:8px 0}}th,td{{border:1px solid #ccc;padding:5px 10px;text-align:center}}th{{background:#e8edf5}}
.pick{{background:#eaf7ee;border:1px solid #bce0c8;border-radius:8px;padding:13px 17px}}
</style></head><body><div class="c">

<h1>トピック数 K をどう決めたか</h1>
<p class="dim">plan04 ／ TF-IDF＋NMF ／ N=95 ／ 候補 K=4〜14</p>

<div class="note"><b>要点.</b> 「いくつのトピックに分けるか（K）」を機械的に一発で決める魔法の数字はありません。
そこで複数の指標を見比べ、<b>安定性</b>と<b>既存タグとの一致</b>という2つが揃って指す
<b>K=6</b> を採用しました（より細かく探索したい場合の候補は K=11〜12）。</div>

<h2>1. まず「エルボー法」は使えなかった</h2>
<div class="term"><b>再構成誤差 / エルボー法：</b>NMFは元データをどれだけ復元できたかの「誤差」を出す。
Kを増やすほど誤差は必ず下がるので、普通は<b>下がり方が急に緩やかになる折れ点（エルボー＝肘）</b>でKを決める。</div>
<p>今回はその折れ点がありません。誤差の下がり方は毎回ほぼ一定（約1%ずつ）で、カーブはほぼ直線。
→ <b>誤差ではKを決められない</b>ので、別の指標を使いました。</p>

<h2>2. Kを決められた2つの指標 → K=6</h2>
<div class="term"><b>安定性（stability）：</b>同じKで計算を何度も繰り返し（乱数の初期値を変えて{m['n_seeds']}回）、
毎回だいたい同じ分類になるかを測る。値は分類の一致度
<b>ARI（{{-1〜1}}、1で完全一致、0で偶然並み）</b>で表す。<b>高い＝結果が偶然に左右されず信頼できる</b>。</div>
<div class="term"><b>既存タグとの一致：</b>データ駆動のトピックが、CALL4の人手タグ（11種）とどれだけ整合するか（同じくARI）。
高すぎ＝タグの焼き直し、低すぎ＝無関係、<b>ほどよく一致するK</b>が「意味のある粒度」の目安。</div>
<p><b>タグ一致は K=6 が明確なピーク</b>（{tg6:.2f}）、<b>安定性</b>は自明に粗い K=4 を除けば <b>K=6 が最良</b>（{st6:.2f}）。
偶然の一致ではなく、<b>別々の観点がともに K=6 を指した</b>のが採用の決め手です。</p>

<h2>3. 使えなかった（＝上限を決められない）指標</h2>
<div class="term"><b>整合性 coherence：</b>各トピックの上位語が実際に一緒に出てくるか。
<b>冗長性 redundancy：</b>トピック同士がどれだけ似てしまっているか。</div>
<p>この2つは<b>Kを増やすほど良くなり続けます</b>（トピックが細かく専門特化するため）。
「大きいほど良い」形なので<b>止め所を示さない</b>＝Kの上限決めには使えません。正直にそのまま記します。</p>

<div class="fig">{svg}</div>

<h2>4. 結論</h2>
<table><tr><th>K</th><th>安定性</th><th>タグ一致</th><th>整合性</th><th>冗長性</th></tr>
{''.join(f"<tr{' style=\"background:#eaf7ee\"' if K==6 else ''}><td>{K}</td><td>{m['stability'][i]:.2f}</td><td>{m['tag_ari'][i]:.2f}</td><td>{m['coherence'][i]:.0f}</td><td>{m['redundancy'][i]:.2f}</td></tr>" for i,K in enumerate(m['Ks']))}
</table>
<div class="pick">
<b>採用：K=6。</b> 安定性・タグ一致の両方がピークで最も頑健。plan02が別経路でたどり着いた数とも一致。<br>
<b>細かく探索する場合：K=11〜12。</b> 安定性が谷から少し持ち直し、整合性も高く、
「国籍」「収容処遇」「水害」などタグでは見えない粒度が立ち上がる（K=14は安定性最低＋断片化で非推奨）。<br>
<span class="dim">※中間のK=7〜10は安定性が谷でタグ一致も低下＝積極的に選ぶ理由が乏しい。
最終判断は各トピックが一文で言えるか（系譜図・各K一覧）と併せて。</span>
</div>
</div></body></html>""", encoding="utf-8")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
