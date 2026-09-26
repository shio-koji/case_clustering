#!/usr/bin/env python3
"""フェーズ2 Stage 4 — 確定したタグと割合のレポートを作る（単一自己完結HTML）。

入力:
  tag_ratio/out/cases_reviewed.json   … 確定タグ入りのケース情報
  tag_ratio/out/soft_tags.json        … 再計算した割合（計算値）
  tag_ratio/out/soft_tags_final.json  … 人の確認を経た確定版
  tag_review/out/reviewed_tags.json   … フェーズ1のタグ変更の記録
  plan05/results/soft_tags.json       … 旧マップの割合（色の変化の基準）
出力:
  tag_ratio/report/call4_tag_ratio_report.html

plan05 のレポートが「機械が何を出したか」の記録なのに対して、こちらは
**人が何を決めたか**の記録。読み手はCALL4側と、あとから経緯を追う人。
体裁は plan05/s1_report.py に合わせてある。

作成: 2026-08-21。
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTDIR = HERE / "out"
REPORT = HERE / "report"
sys.path.insert(0, str(HERE))
from s2_make_ratio_sheet import to_percent  # noqa: E402

# カード・2部グラフと同じ濃色（card_editor/s2_make_cards.py の TAG_COLORS）
TCOL = {
    "個人情報・プライバシー": "#22504e",
    "医療・福祉・障がい": "#ff4709",
    "ジェンダー・セクシュアリティ": "#ff9423",
    "刑事司法": "#9f6e34",
    "環境・災害": "#2e9d7e",
    "働き方": "#99b73d",
    "公正な手続": "#fe7389",
    "沖縄": "#3970cb",
    "外国にルーツを持つ人々": "#6b5498",
    "政治参加・表現の自由": "#3daac8",
    "情報公開": "#033064",
}
CASE_URL = "https://www.call4.jp/info.php?type=items&id={}"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def chip(tag, pct=None):
    body = esc(tag) + (f" {pct}%" if pct is not None else "")
    return f'<span class="tg" style="background:{TCOL[tag]}">{body}</span>'


def ratio_bar(pairs, width=200):
    """割合の積み上げバー（カードの帯と同じ並び・同じ色）。"""
    segs = "".join(
        f'<span class="rseg" style="width:{p}%;background:{TCOL[t]}" '
        f'title="{esc(t)} {p}%">{p if p >= 9 else ""}</span>'
        for t, p in pairs)
    return f'<span class="rbar" style="min-width:{width}px">{segs}</span>'


def main():
    cases = json.loads((OUTDIR / "cases_reviewed.json").read_text(encoding="utf-8"))
    soft = json.loads((OUTDIR / "soft_tags.json").read_text(encoding="utf-8"))
    final_path = OUTDIR / "soft_tags_final.json"
    if not final_path.exists():
        raise SystemExit(f"{final_path} がありません。s3_ingest_ratios.py を先に実行してください。")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    reviewed = json.loads(
        (ROOT / "tag_review" / "out" / "reviewed_tags.json").read_text(encoding="utf-8"))
    old_path = ROOT / "plan05" / "results" / "soft_tags.json"
    old_ratios = (json.loads(old_path.read_text(encoding="utf-8"))["ratios"]
                  if old_path.exists() else {})

    TAGS = cases["tags"]
    by_id = {c["id"]: c for c in cases["cases"]}
    order = sorted(cases["cases"], key=lambda c: c["no"])
    calc = soft["ratios"]
    conf = final["ratios"]
    auc = soft["validation"]["per_tag_auc"]
    counts = {t: sum(1 for c in cases["cases"] if t in c["tags"]) for t in TAGS}

    def pct(d):
        return dict(to_percent(sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))))

    conf_pct = {cid: pct(v) for cid, v in conf.items()}
    calc_pct = {cid: pct(v) for cid, v in calc.items()}

    # 主タグ（カードの色）が旧マップから変わったケース
    flips = []
    for cid, v in conf_pct.items():
        o = old_ratios.get(cid) or {}
        if not o:
            continue
        a, b = max(o, key=o.get), max(v, key=v.get)
        if a != b:
            flips.append((by_id[cid]["no"], cid, a, b))
    flips.sort()

    # 順番を人が変えたケース
    changes = {c["case_id"]: c for c in final.get("ratio_changes", [])}

    # ---- 冒頭のKPI ---------------------------------------------------------
    n_tag_changed = len(reviewed["changes"])
    kpi = f"""
<div class="kpi">
  <div><div class="big">{len(conf)}</div>ケース（音信不通7件を除く）</div>
  <div><div class="big">{n_tag_changed}</div>フェーズ1でタグが変わった</div>
  <div><div class="big">{len(changes)}</div>フェーズ2で順番が変わった</div>
  <div><div class="big">{len(flips)}</div>カードの色が変わった</div>
</div>"""

    # ---- タグ別サマリ -----------------------------------------------------
    rows = []
    for t in TAGS:
        w_conf = sum(conf[cid].get(t, 0) for cid in conf)
        w_calc = sum(calc[cid].get(t, 0) for cid in calc)
        w_old = sum((old_ratios.get(cid) or {}).get(t, 0) for cid in conf)
        before = reviewed["tag_counts_before"][t]
        a = auc.get(t)
        avg = w_conf / counts[t] * 100 if counts[t] else 0
        low = ' <span class="lc">低信頼</span>' if counts[t] < 3 else ""
        rows.append(
            f"<tr><td>{chip(t)}{low}</td>"
            f"<td style='text-align:right'>{before} → <b>{counts[t]}</b></td>"
            f"<td style='text-align:right'>{w_old:.2f}</td>"
            f"<td style='text-align:right'>{w_calc:.2f}</td>"
            f"<td style='text-align:right'><b>{w_conf:.2f}</b></td>"
            f"<td style='text-align:right'>{w_conf - w_calc:+.2f}</td>"
            f"<td style='text-align:right'>{avg:.0f}%</td>"
            f"<td style='text-align:right'>{'—' if a is None else f'{a:.3f}'}</td></tr>")
    tag_table = "\n".join(rows)

    # ---- フェーズ1のタグ変更 ----------------------------------------------
    t1 = []
    for ch in sorted(reviewed["changes"], key=lambda c: by_id[c["case_id"]]["no"]):
        cid = ch["case_id"]
        add = "".join(f'＋{chip(t)} ' for t in ch["added"])
        rem = "".join(f'−{chip(t)} ' for t in ch["removed"])
        t1.append(
            f'<tr><td>{by_id[cid]["no"]}</td>'
            f'<td><a href="{CASE_URL.format(cid)}" target="_blank">{esc(ch["title"])}</a></td>'
            f'<td>{add}{rem}</td><td>{esc(ch["comment"][:110])}</td></tr>')
    tag_changes = "\n".join(t1)

    # ---- フェーズ2の順番変更 ----------------------------------------------
    t2 = []
    for cid, ch in sorted(changes.items(), key=lambda kv: by_id[kv[0]]["no"]):
        before = sorted(ch["before"].items(), key=lambda kv: -(kv[1] or 0))
        after = sorted(ch["after"].items(), key=lambda kv: -kv[1])
        flip = "" if max(ch["before"], key=lambda t: ch["before"][t] or 0) == after[0][0] \
            else ' <span class="lc">色が変わる</span>'
        t2.append(
            f'<tr><td>{by_id[cid]["no"]}</td>'
            f'<td><a href="{CASE_URL.format(cid)}" target="_blank">'
            f'{esc(by_id[cid]["title"])}</a>{flip}</td>'
            f'<td>{ratio_bar([(t, v) for t, v in before if v], 150)}</td>'
            f'<td>{ratio_bar(after, 150)}</td>'
            f'<td>{esc(ch["comment"][:150]) or "—"}</td></tr>')
    order_changes = "\n".join(t2)

    # ---- 全88件の一覧 -----------------------------------------------------
    t3 = []
    for c in order:
        cid = c["id"]
        pairs = sorted(conf_pct[cid].items(), key=lambda kv: -kv[1])
        src = ("人が変更" if cid in changes else "計算どおり")
        t3.append(
            f'<tr><td>{c["no"]}</td>'
            f'<td><a href="{CASE_URL.format(cid)}" target="_blank">'
            f'{esc(c["title"])}</a></td>'
            f'<td>{"アーカイブ" if c["status"] == "archived" else "進行中"}</td>'
            f'<td>{ratio_bar(pairs)}</td>'
            f'<td>{"".join(chip(t, p) + " " for t, p in pairs)}</td>'
            f'<td>{src}</td></tr>')
    all_rows = "\n".join(t3)

    flip_rows = "\n".join(
        f'<tr><td>{no}</td>'
        f'<td><a href="{CASE_URL.format(cid)}" target="_blank">'
        f'{esc(by_id[cid]["title"])}</a></td>'
        f'<td>{chip(a)}</td><td>{chip(b)}</td></tr>' for no, cid, a, b in flips)

    # 押し広げが実際に効いた件数（順番を変えた × 計算値の差が10pt未満）
    from s2_make_ratio_sheet import min_gap_ladder
    n_widened = 0
    for cid in changes:
        mags = [p for _, p in to_percent(
            sorted(calc[cid].items(), key=lambda kv: (-kv[1], kv[0])))]
        if min_gap_ladder(mags) != mags:
            n_widened += 1

    # 最低割合の底上げが効いたケース
    floored = final.get("floored_cases", [])
    n_floored = len(floored)
    floor_rows = ""
    if floored:
        body = "\n".join(
            f'<tr><td>{by_id[f["case_id"]]["no"]}</td>'
            f'<td><a href="{CASE_URL.format(f["case_id"])}" target="_blank">'
            f'{esc(by_id[f["case_id"]]["title"])}</a></td>'
            f'<td>{"".join(chip(t, v) + " " for t, v in sorted(f["before"].items(), key=lambda kv: -kv[1]))}</td>'
            f'<td>{"".join(chip(t, v) + " " for t, v in sorted(f["after"].items(), key=lambda kv: -kv[1]))}</td>'
            f'</tr>' for f in floored)
        floor_rows = ("<table><tr><th>No</th><th>ケース</th><th>底上げ前</th>"
                      f"<th>確定</th></tr>{body}</table>")

    dist = Counter(len(v) for v in conf.values())
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>CALL4 ケースマップ：確定したタグと割合</title>
<style>
:root{{--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--accent:#2B6CB0;}}
*{{box-sizing:border-box;}}
body{{font-family:-apple-system,"Hiragino Sans","Yu Gothic",sans-serif;color:var(--fg);
max-width:1080px;margin:0 auto;padding:28px 20px 80px;line-height:1.7;}}
h1{{font-size:1.65rem;border-bottom:3px solid var(--accent);padding-bottom:10px;}}
h2{{font-size:1.26rem;margin-top:2.3em;border-left:5px solid var(--accent);padding-left:10px;}}
h3{{font-size:1.05rem;margin-top:1.6em;}}
.en{{color:var(--muted);font-weight:400;font-size:.72em;}}
.lead{{background:#f5f8fc;border:1px solid var(--line);border-radius:8px;padding:16px 20px;}}
.warn{{background:#fff8e6;border:1px solid #e6c34c;border-radius:8px;padding:12px 18px;font-size:.92em;}}
.ok{{background:#eef6ef;border:1px solid #9cc1a4;border-left:6px solid #4C8C2B;
border-radius:8px;padding:12px 18px;}}
table{{border-collapse:collapse;width:100%;font-size:.9em;margin:1em 0;}}
th,td{{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top;}}
th{{background:#f2f2f2;}}
.kpi{{display:flex;gap:20px;flex-wrap:wrap;margin:1em 0;}}
.kpi div{{background:#f5f8fc;border:1px solid var(--line);border-radius:8px;
padding:12px 18px;min-width:150px;}}
.big{{font-size:2rem;font-weight:800;color:var(--accent);}}
.tg{{display:inline-block;color:#fff;border-radius:4px;padding:1px 7px;
font-size:.85em;font-weight:600;white-space:nowrap;}}
.lc{{color:#b0532b;font-size:.78em;margin-left:6px;}}
code{{background:#f2f2f2;padding:1px 5px;border-radius:3px;}}
.rbar{{display:inline-flex;height:16px;border-radius:3px;overflow:hidden;vertical-align:middle;}}
.rseg{{color:#fff;font-size:10px;text-align:center;line-height:16px;
white-space:nowrap;overflow:hidden;}}
#search{{padding:7px 11px;width:280px;border:1px solid var(--line);
border-radius:6px;font-size:14px;}}
#rt{{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:8px;}}
#rt table{{margin:0;}} #rt th{{position:sticky;top:0;z-index:1;}}
a{{color:var(--accent);}}
footer{{margin-top:3em;color:var(--muted);font-size:.85em;
border-top:1px solid var(--line);padding-top:12px;}}
</style></head><body>

<h1>CALL4 ケースマップ：確定したタグと割合<br>
<span class="en">Confirmed tags and per-case tag shares</span></h1>

<div class="lead">
<p>紙のケースマップ（1600mm幅ロール）に載せる<b>{len(conf)}件</b>のケースについて、
<b>どのタグが付くか</b>と<b>そのケースの中でどのタグが主役か</b>を確定させた記録です。
音信不通の7件を除いた88件が対象で、アーカイブ{cases['n_archived']}件・
進行中{len(conf) - cases['n_archived']}件。</p>
<p>機械が出した推定を人が確認して修正する形を2段階に分けました。
<b>フェーズ1</b>でタグそのものを目視確認し、<b>フェーズ2</b>でケース内の順番を確認しています。
このレポートは「機械が何を出したか」ではなく<b>人が何を決めたか</b>の記録です
（機械側の記録は <code>plan05/report/call4_plan05_report.html</code>）。</p>
</div>

{kpi}

<h2>1. 何をどう決めたか <span class="en">Process</span></h2>

<p>タグは11種類で、1ケースにつく数は原則3個までです。割合は
<b>そのケースの中での配分</b>（合計100%）で、紙の上では
カード下部の帯の幅と、タグとカードを結ぶ線の太さになります。</p>

<h3>フェーズ1：タグの目視確認</h3>
<p>CALL4が付けていたタグを1ケースずつ確認し、{n_tag_changed}件で変更しました。
<code>公正な手続</code> が {reviewed['tag_counts_before']['公正な手続']}件 →
{counts['公正な手続']}件と最も動いています。</p>

<h3>再計算</h3>
<p>確定したタグで割合を計算し直しました。方法は plan05 と同じで、変えたのは入力だけです。
ケース本文（タイトル＋説明＋本文）を多言語モデル <code>bge-m3</code> で1024次元に数値化し、
タグごとに「そのタグを持つケースの重心」と「持たないケースの重心」を作って、その差で採点します
（<code>disc = cos(本文, 陽性重心) − cos(本文, 陰性重心)</code>）。
自分自身は陽性重心から除いて水増しを防ぎます（Leave-One-Out）。
付与タグの中で softmax して合計1にしたものが割合です（温度 τ={soft['tau']}）。</p>
<p>母集団は plan05 の95件から<b>88件</b>に変えました。除外した7件は目視確認を
受けていないので、確定タグと未確認タグが混ざった重心になってしまうためです。</p>

<h3>フェーズ2：順番の確認</h3>
<p>数値を直接入力してもらうのは現実的でないので、
<b>「どのタグが主役か」の順番だけ</b>を選んでもらいました（プルダウン1つ）。
選ばれた順番の1位に計算値の最大、2位に2番目…を割り当てます。つまり
<b>差の大きさは本文から出た値を使い、どちらが上かだけを人が決める</b>形です。</p>
<div class="warn">
計算値がほぼ同率（50:50 や 34:33:33）のケースでは、順番を入れ替えても数字が動かず
人の判断が紙に出ません。そこで<b>既定と違う順番を選んだ場合だけ、隣り合う差を
最低10ポイントまで押し広げる</b>ようにしました。既に10ポイント以上ある差はそのまま残すので、
計算の濃淡は壊れません。順番を変えた{len(changes)}件のうち、
実際に押し広げが効いたのは<b>{n_widened}件</b>です。
</div>

<h3>最低割合の底上げ</h3>
<p>付与されたタグの割合には<b>10%の下限</b>を設けています。10%を割った分は、
10%を超えている側から<b>超過分に比例して</b>削って合計100%を保ちます
（超過している値がどれも同じ倍率で縮む形なので、順位と相対関係は崩れません）。
該当は{n_floored}件です。</p>
<div class="ok">
<b>なぜ底上げするか</b> — タグは<b>ケース主催者の意思</b>で付いています。
本文から読み取れなくても、付けると決められたタグは紙の上で読める太さで描かなければ、
そのケースの趣旨を落としてしまいます。1%はカードの帯で0.84mm——罫線より細く、
「1%」という数字そのものが入りません。
機械の推定より人の付与を尊重する、という判断です。</p>
<p>逆に言えば、<b>底上げした分は紙の上で実際より重く見えます</b>。
機械は「本文にほぼ出てこない」と判定しているので、そこは割り引いて読んでください。
</div>
{floor_rows}

<h2>2. タグ別サマリ <span class="en">Per-tag summary</span></h2>
<p>「重み合計」は各ケースの割合を足した値で、<b>そのタグが全体の何ケース分に相当するか</b>を表します。
付与件数が同じでも、割合が小さければ重みは減ります。紙の上ではタグから伸びる線の総量です。</p>
<table>
<tr><th>タグ</th><th>件数（生→確定）</th><th>旧マップ</th><th>計算値</th><th>確定</th>
<th>人の判断による差</th><th>平均割合</th><th>AUC</th></tr>
{tag_table}
</table>
<p class="en" style="font-size:.85em">AUC は「そのタグが付くケースをテキストから見分けられるか」の指標
（0.5＝偶然と同じ、1.0＝完全に分離）。平均AUC {soft['validation']['mean_auc_reliable']}。
<code>沖縄</code>（{counts['沖縄']}件）と <code>個人情報・プライバシー</code>
（{counts['個人情報・プライバシー']}件）は件数が少なく、テキストからは判別できていません。
この2つのタグの割合は人の判断をそのまま採用しています。</p>

<h2>3. フェーズ1：タグの変更 <span class="en">Tag changes</span></h2>
<table>
<tr><th>No</th><th>ケース</th><th>変更</th><th>理由（確認者のコメント）</th></tr>
{tag_changes}
</table>

<h2>4. フェーズ2：順番の変更 <span class="en">Order changes</span></h2>
<p>左が計算どおりの割合、右が人の判断を反映した確定値です。</p>
<table>
<tr><th>No</th><th>ケース</th><th>計算値</th><th>確定</th><th>理由（確認者のコメント）</th></tr>
{order_changes}
</table>
<div class="ok">
<b>コメントから読み取れる方針</b>
<ul>
<li><b>環境優位の原則</b> — 環境に関わるケースは原則として環境を上に置く方針。
ただし例外も明示されており、
新焼却炉のケースは「覚書反故に力点があるので、例外的に手続優位で」と手続を上に置いています。</li>
<li><b>類似ケースとの統一</b> — 入管のケース同士、レイシャルプロファイリングのケース同士で
主タグを揃える判断がされています。</li>
<li><b>フェーズ1の指摘の撤回</b> — 3件は、フェーズ1で自分が書いたメモを読み返して
覆したものです（ケースページを読み返して判断を変えた、と理由が添えられている）。
機械の推定とも当初のメモとも違う結論で、人の判断を最終としています。</li>
</ul>
</div>

<h2>5. カードの色が変わったケース <span class="en">Colour flips</span></h2>
<p>カードの枠色と紙面での位置づけは、いちばん割合の大きいタグ（主タグ）で決まります。
以前のマップ（生タグ・95件で計算した plan05 の値）から主タグが変わったのは{len(flips)}件です。</p>
<table>
<tr><th>No</th><th>ケース</th><th>旧マップ</th><th>確定</th></tr>
{flip_rows}
</table>

<h2>6. 確定した全{len(conf)}件 <span class="en">All cases</span></h2>
<p>タグ数の分布：{" / ".join(f"{k}タグ {v}件" for k, v in sorted(dist.items()))}。
1タグのケースは100%固定です。</p>
<input id="search" type="search" placeholder="ケース名・タグで絞り込み">
<div id="rt">
<table id="tbl">
<tr><th>No</th><th>ケース</th><th>状況</th><th>割合</th><th>内訳</th><th>由来</th></tr>
{all_rows}
</table>
</div>

<h2>7. 限界 <span class="en">Limitations</span></h2>
<ul>
<li><b>割合の根拠はテキストと埋め込みモデルです。</b>法的な重要度や社会的な影響の
大きさではありません。「本文がどのタグの向きに寄っているか」を測ったものです。</li>
<li><b>順番は人が決めましたが、差の大きさは機械の値です。</b>
「A が B より大きい」は人の判断ですが、「61:39 なのか 55:45 なのか」は
テキストから出た値です。この配分自体に人の承認は取っていません。</li>
<li><b>件数の少ないタグは判別できていません。</b><code>沖縄</code> は AUC 0.48 で
偶然と変わらず、<code>個人情報・プライバシー</code> は1件のため計算すらできません。</li>
<li><b>N=88・タグ11の小規模なデータです。</b>統計的な一般化には向きません。</li>
<li><b>コサイン類似度の差が極端に出たときの扱いは、まだ改善の余地があります。</b>
いまは softmax の温度（τ={soft['tau']}）をデータのばらつきから自動で決めているため、
本文の寄りが強いケースでは 98:1:1 のような極端な配分になり、
最低割合の底上げで手当てしています。将来的には、類似度の差を割合へ写す段で
差を圧縮する（τを意図的に緩める、または別の関数を挟む）ことで、
底上げに頼らずに済むはずです。今回は手を入れていません。</li>
<li><b>確認者名とCALL4側の最終判断は記録されていません。</b>
シートの該当列は空のままです。</li>
</ul>

<h2>8. 再現情報 <span class="en">Reproducibility</span></h2>
<table>
<tr><th>項目</th><th>内容</th></tr>
<tr><td>対象</td><td>{len(conf)}件（CALL4掲載95件のうち音信不通7件を除く）</td></tr>
<tr><td>埋め込み</td><td>BAAI/bge-m3（1024次元・最大8192トークン・L2正規化）</td></tr>
<tr><td>テキスト構成</td><td>タイトル＋掲載説明文＋本文（更新情報は含めない・NFKC正規化）</td></tr>
<tr><td>採点</td><td>判別的プロトタイプ（陽性重心−陰性重心）＋Leave-One-Out</td></tr>
<tr><td>割合</td><td>付与タグ内 softmax（τ={soft['tau']}）→ 人が選んだ順位へ割り当て
→ 最大剰余法で合計100の整数%（最低1%）</td></tr>
<tr><td>押し広げ</td><td>既定と違う順番を選んだ場合、隣り合う差を最低10ポイントに</td></tr>
<tr><td>スクリプト</td><td><code>tag_ratio/s0_cases.py</code> →
<code>s1_ratios.py</code> → <code>s2_make_ratio_sheet.py</code> →
<code>s3_ingest_ratios.py</code> → <code>s4_report.py</code>（seed=42）</td></tr>
<tr><td>確定データ</td><td><code>tag_ratio/out/soft_tags_final.json</code></td></tr>
<tr><td>配置エディタ</td><td><a href="https://shio-koji.github.io/case_clustering/"
target="_blank">https://shio-koji.github.io/case_clustering/</a></td></tr>
</table>

<footer>生成 {gen} ／ CALL4 ケースマップ フェーズ2 ／
このレポートはケース本文へのリンクを含みます。掲載画像・本文の権利はCALL4および各ケースの
関係者に帰属します。</footer>

<script>
document.getElementById('search').addEventListener('input', function (e) {{
  var q = e.target.value.trim();
  var rows = document.querySelectorAll('#tbl tr');
  for (var i = 1; i < rows.length; i++) {{
    rows[i].style.display = (!q || rows[i].textContent.indexOf(q) >= 0) ? '' : 'none';
  }}
}});
</script>
</body></html>"""

    REPORT.mkdir(parents=True, exist_ok=True)
    dest = REPORT / "call4_tag_ratio_report.html"
    dest.write_text(html, encoding="utf-8")
    print(f"[s4] 出力: {dest}  ({len(html) / 1024:.0f} KB)")
    print(f"[s4] {len(conf)}件／タグ変更 {n_tag_changed}件／順番変更 {len(changes)}件"
          f"／色が変わった {len(flips)}件")


if __name__ == "__main__":
    main()
