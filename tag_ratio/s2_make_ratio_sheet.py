#!/usr/bin/env python3
"""フェーズ2 Stage 2 — タグ割合の確認用スプレッドシート取り込みCSVを生成する。

入力:
  tag_ratio/out/cases_reviewed.json … s0_cases.py（確定タグ入りのケース情報）
  tag_ratio/out/soft_tags.json      … s1_ratios.py（再計算した割合）
出力:
  tag_ratio/out/ratio_review.csv    … Googleスプレッドシートにインポートする1枚
  tag_ratio/out/ratio_baseline.json … 計算値の控え（読み戻し時の差分判定に使う）

## 入力は「順番」だけ

数値を打たせない。人にお願いするのは**どのタグが主役かの順番**（プルダウン1つ）で、
割合の数字はそこから機械的に決める:

  - 選んだ順番の1位に計算値の最大、2位に2番目…を割り当て直す
    （スロットは計算値の大きい順に並べてあるので、順位k位＝計算k番目の値）
  - ただし**人が計算と違う順番を選んだ場合は、隣り合う差を最低 MIN_GAP_PT まで
    押し広げる**。計算値が同値（50:50 や 34:33:33）のケースで順番を変えても
    数字が動かないと、人が決めたことが紙に出ないため。
    差が既に MIN_GAP_PT 以上ある差はそのまま残す（計算の濃淡を壊さない）。
  - 「どれも同じくらい」を選んだら均等割り（端数は先頭のスロットへ）
  - 計算どおりの順番のままなら、押し広げをせず計算値をそのまま使う。
    人が何も判断していないのに数字を作らないため。

こうしているのは、**差の大きさは本文から出た値を活かし、向きだけ人が決める**形に
したいため。順番を変えなければ数字は1つも動かないので、「触っていない」ことが
そのまま担保される。固定の比（60:40など）を当てる方式にすると、
46件ある2タグのケースが全部同じ比になり、ケースごとの濃淡が消える。

差の大きさ自体を変えたい場合はコメント欄に書いてもらう。ここで数値も触れるように
すると「順番だけ決めればよい」という単純さが失われる。

## 列の並び

タグ・計算値・割合を3列ずつ組にして隣接させてある（`タグk / 計算k / 割合k`）。
シート側の数式が「タグの範囲で SUMIF して2列右を合計する」形で
そのタグの割合を取っているので、この2列のずれは崩さないこと。

その他:
  - 1ケース1行。タグを縦に展開する long 形式は、1ケースの配分を読むのに
    複数行を追う必要があり、順番の判断に向かない。
  - タグが1つだけのケース（88件中35件）は100%固定なので、プルダウンを置かない。
  - 検索用にタグを1セルに連結した列を置く（特定タグのケースだけ絞って見比べる用）。

作成: 2026-08-20。
"""

import csv
import json
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_notes import check  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "out"
MAX_SLOTS = 4          # タグ数の上限（原則3・例外4）
SEP = ", "
ORDER_SEP = " ＞ "     # プルダウンの選択肢で順位を区切る記号
EVEN = "どれも同じくらい"
PCT_FLOOR = 1          # 丸めの下限（0%を作らないため。表示上の下限は MIN_SHARE_PT）
MIN_SHARE_PT = 10      # 付与タグの最低割合（%）
MIN_GAP_PT = 10        # 人が順番を変えたときに確保する、隣り合う割合の最小差（%ポイント）
TIE_PT = 2             # これ未満の差しかないケースは、既定を「どれも同じくらい」にする

HEADER = (
    ["No", "case_id", "状況", "ケース名", "概要", "タグ数", "タグ（検索用）",
     "フェーズ1の指摘", "順番"]
    + [c for k in range(1, MAX_SLOTS + 1) for c in (f"タグ{k}", f"計算{k}", f"割合{k}")]
    + ["合計", "指摘と一致?", "色の変化", "変更あり?", "コメント・理由", "確認者",
       "CALL4最終判断", "指摘の読み取り", "旧マップの主タグ", "既定の順番"]
    + [f"調整{k}" for k in range(1, MAX_SLOTS + 1)]
    + ["URL"]
)

# 確認者が見ていたスクリーンショットのもとになった割合。
# 「ポスターの色が変わるか」の基準はこれ（生タグ・95件で計算した plan05 の値）。
OLD_RATIOS = HERE.parent / "plan05" / "results" / "soft_tags.json"


def to_percent(pairs):
    """(タグ, 割合float) の列を、合計ちょうど100の整数%にする。

    最大剰余法で丸めたあと、0%になったタグを1%に持ち上げて、
    その分を最も大きいスロットから引く。付与されているタグが
    バーから消えないようにするため（0%はカードに描けない）。
    """
    if not pairs:
        return []
    raw = [v * 100 for _, v in pairs]
    floors = [int(x) for x in raw]
    rest = 100 - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - floors[i]))
    for i in order[:max(rest, 0)]:
        floors[i] += 1

    for i, v in enumerate(floors):
        if v < PCT_FLOOR:
            need = PCT_FLOOR - v
            donor = max(range(len(floors)), key=lambda j: floors[j])
            if floors[donor] - need < PCT_FLOOR:
                continue          # 押し込めない（4タグが極端に偏った場合のみ）
            floors[donor] -= need
            floors[i] = PCT_FLOOR
    assert sum(floors) == 100, floors
    floors = apply_min_share(floors)
    return [(t, p) for (t, _), p in zip(pairs, floors)]


def even_split(n, total=100):
    """均等割り。端数は先頭から1つずつ配る（3タグなら 34/33/33）。"""
    base, rest = divmod(total, n)
    return [base + (1 if i < rest else 0) for i in range(n)]


def apply_min_share(values, floor=MIN_SHARE_PT):
    """floor 未満の割合を floor まで底上げし、合計100を保つ。

    引く先は floor を超えている側で、**超過分に比例**して削る。式にすると
    超過している値はどれも同じ倍率で縮む:

        新しい値 = floor + (元の値 − floor) × (1 − 不足分の合計 / 超過分の合計)

    これで「必要な分しか動かさない」「2位以下の相対関係を保つ」の両方が満たされ、
    降順も崩れない（倍率が共通なので順序は保存される）。

    なぜ底上げするか:
      タグはケース主催者の意思で付いている。本文から読み取れなくても、
      付いていると決めたタグは紙の上で読める太さで描く必要がある
      （1%はカードの帯で0.84mm＝数字も入らない）。
      機械の推定より人の付与を尊重する、という判断。

    floor×個数が100を超える場合（5個以上）は成立しないので均等割りにする。
    """
    n = len(values)
    if n <= 1:
        return [100] * n
    if all(v >= floor for v in values):
        return list(values)
    if floor * n > 100:
        return even_split(n)

    raised = [max(v, floor) for v in values]
    deficit = sum(raised) - 100                      # 底上げで増えた分
    excess = [max(v - floor, 0) for v in raised]     # 削れる余地
    total_excess = sum(excess)
    if total_excess <= 0 or deficit <= 0:
        return even_split(n)
    scale = 1 - deficit / total_excess
    w = [floor + e * scale for e in excess]

    floors = [int(x) for x in w]
    rest = 100 - sum(floors)
    order = sorted(range(n), key=lambda i: -(w[i] - floors[i]))
    for i in order[:rest]:
        floors[i] += 1
    assert sum(floors) == 100, floors
    assert min(floors) >= floor, floors
    return floors


def min_gap_ladder(magnitudes, min_gap=MIN_GAP_PT):
    """隣り合う差を最低 min_gap まで押し広げ、合計100を保った整数列を返す。

    既に min_gap 以上ある差はそのまま残す（計算値の濃淡を壊さない）。
    差を広げると合計が増えるので、全体を同じ量だけ下げて100に戻す
    ＝「他の差を保ったまま、足りない差だけを開く」最小限の操作。

        G_k = max(元の差_k, min_gap)          （k = 1..n-1、上から数えた差）
        Σ w_k = n·w_n + Σ k·G_k = 100  →  w_n = (100 - Σ k·G_k) / n
        w_k = w_(k+1) + G_k

    G_k の係数は k（上ほど何度も足されるので、下の差ほど重い）。
    ここを逆順にすると、差が既に十分あるケースで合計が100を外れる。

    端数は最大剰余法で配る（合計ちょうど100）。
    """
    n = len(magnitudes)
    if n <= 1:
        return [100] * n
    gaps = [max(magnitudes[k] - magnitudes[k + 1], min_gap) for k in range(n - 1)]
    tail = sum((i + 1) * gaps[i] for i in range(n - 1))
    w = [0.0] * n
    w[n - 1] = (100 - tail) / n
    for k in range(n - 2, -1, -1):
        w[k] = w[k + 1] + gaps[k]
    if w[n - 1] < PCT_FLOOR:
        # min_gap が大きすぎて下位が潰れる場合は押し広げをあきらめる
        # （4タグで極端に開いたときだけ。黙って1%未満を作らない）
        return list(magnitudes)
    floors = [int(x) for x in w]
    rest = 100 - sum(floors)
    order = sorted(range(n), key=lambda i: -(w[i] - floors[i]))
    for i in order[:rest]:
        floors[i] += 1
    assert sum(floors) == 100, floors
    return floors


def default_order(slot_tags, magnitudes, tie=TIE_PT):
    """既定の選択肢。計算値がほぼ同率なら「どれも同じくらい」にする。

    差が tie 未満しか無いケースは、機械の側に「どれが主役か」の情報が無い。
    そこに計算の並び順を既定として置くと、根拠のない順位を主張することになる。
    """
    if len(slot_tags) < 2:
        return ""
    if all(magnitudes[k] - magnitudes[k + 1] < tie for k in range(len(magnitudes) - 1)):
        return EVEN
    return ORDER_SEP.join(slot_tags)


def resolve_ratio(order_label, slot_tags, magnitudes):
    """選んだ順番から割合を決める。シート側の数式と同じ規則。

    order_label … プルダウンの値。"A ＞ B" 形式 / EVEN / 空（1タグ）
    slot_tags   … スロット順（＝計算値の降順）のタグ名
    magnitudes  … 同じ並びの計算値（整数%・降順）
    返り値      … {タグ: 整数%}

    押し広げるかどうかの基準は**既定の選択肢と違うかどうか**。スロット順と
    比べてはいけない: 計算がほぼ同率の行は既定が EVEN なので、スロット順を
    選ぶこと自体が人の判断であり、押し広げないと判断が数字に出ない。
    """
    if not order_label:
        return dict(zip(slot_tags, apply_min_share(magnitudes)))
    default = default_order(slot_tags, magnitudes)
    if order_label == EVEN:
        return dict(zip(slot_tags, even_split(len(slot_tags))))
    order = [t.strip() for t in order_label.split("＞") if t.strip()]
    if sorted(order) != sorted(slot_tags):
        raise ValueError(f"順番のタグ集合が合いません: {order} / {slot_tags}")
    if order_label == default:
        # 既定のまま。人は何も動かしていないので計算値をそのまま使う
        return dict(zip(slot_tags, apply_min_share(magnitudes)))
    # 押し広げたあとにも底上げを掛ける（押し広げは下位を薄くしうるため）。
    # 底上げが効くと最小差10ptを満たせない場合があるが、そのときは底上げを優先する。
    adjusted = apply_min_share(min_gap_ladder(magnitudes))
    return {t: adjusted[i] for i, t in enumerate(order)}


def order_options(slot_tags):
    """プルダウンに並べる順番の選択肢。計算どおりの並びを先頭に置く。"""
    if len(slot_tags) < 2:
        return []
    opts = [ORDER_SEP.join(slot_tags)]
    for p in permutations(slot_tags):
        s = ORDER_SEP.join(p)
        if s not in opts:
            opts.append(s)
    return opts + [EVEN]


def main():
    cases = json.loads((OUTDIR / "cases_reviewed.json").read_text(encoding="utf-8"))
    soft = json.loads((OUTDIR / "soft_tags.json").read_text(encoding="utf-8"))
    ratios = soft["ratios"]
    old_ratios = (json.loads(OLD_RATIOS.read_text(encoding="utf-8"))["ratios"]
                  if OLD_RATIOS.exists() else {})

    rows, baseline, over_slots, verdicts, flips = [], {}, [], [], []
    widened, evens_by_default = [], []
    for c in sorted(cases["cases"], key=lambda c: c["no"]):
        cid = c["id"]
        if set(ratios[cid]) != set(c["tags"]):
            raise SystemExit(f"{cid}: soft_tags.json のタグとケース情報が一致しません")
        pairs = sorted(ratios[cid].items(), key=lambda kv: (-kv[1], kv[0]))
        if len(pairs) > MAX_SLOTS:
            over_slots.append(cid)
            continue
        pct = to_percent(pairs)
        baseline[cid] = {t: p for t, p in pct}
        slot_tags = [t for t, _ in pct]
        mags = [p for _, p in pct]
        # 1タグのケースは順番を選ぶ余地が無いので、プルダウンも既定値も置かない。
        # ほぼ同率のケースは既定を「どれも同じくらい」にする（根拠のない順位を出さない）
        default = default_order(slot_tags, mags)
        if default == EVEN:
            evens_by_default.append(cid)
        # 初期表示の割合は、既定の選択肢から決めた値（＝押し広げは起きない）
        initial = resolve_ratio(default, slot_tags, mags)
        # 押し広げ後の値。人が順番を変えたときにシートの数式が参照する
        adjusted = apply_min_share(min_gap_ladder(mags))
        if adjusted != mags:
            widened.append((c["no"], cid, c["title"], mags, adjusted))

        slots = []
        for k in range(MAX_SLOTS):
            if k < len(pct):
                t = slot_tags[k]
                slots += [t, mags[k], initial[t]]   # タグ / 計算値 / 割合（初期値）
            else:
                slots += ["", "", ""]

        ph = c.get("phase1") or {}
        note = ph.get("note", "")
        if note and ph.get("unsettled"):
            note = "【未決着】" + note
        if ph.get("stale_refs"):
            note = f"{note}（指摘中の「{'・'.join(ph['stale_refs'])}」はタグから外れました）"
        # 「指摘の読み取り」列はシート側で照合の数式を組むための作業列。
        # A>B>C なら ">" 区切り、「Aが最大」なら MAX: を付ける。
        if ph.get("chain"):
            reading = ">".join(ph["chain"])
        elif ph.get("max_tag"):
            reading = "MAX:" + ph["max_tag"]
        else:
            reading = ""
        verdicts.append(check(ph, ratios[cid]) if reading else "")

        old_r = old_ratios.get(cid) or {}
        old_top = max(old_r, key=old_r.get) if old_r else ""
        if old_top and old_top != slot_tags[0]:
            flips.append((c["no"], cid, c["title"], old_top, slot_tags[0],
                          c["tags"] != c["tags_before"]))

        adj_cells = [adjusted[k] if k < len(adjusted) else "" for k in range(MAX_SLOTS)]
        rows.append(
            [c["no"], cid, c["status_label"], c["title"], c["description"],
             len(pct), SEP.join(slot_tags), note, default]
            + slots
            + ["", "", "", "", "", "", "", reading, old_top, default]
            + adj_cells
            + [c["url"]]
        )

    if over_slots:
        raise SystemExit(f"タグが{MAX_SLOTS}個を超えるケースがあります: {over_slots}\n"
                         f"  MAX_SLOTS と setup_ratio.gs の SLOTS を増やしてください。")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    # BOMは付けない（付けるとA1が "﻿No" になり setup_ratio.gs の見出し検出が落ちる）
    with (OUTDIR / "ratio_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    (OUTDIR / "ratio_baseline.json").write_text(
        json.dumps({"unit": "percent", "sum": 100, "order_sep": ORDER_SEP,
                    "even_label": EVEN, "ratios": baseline},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    vc = Counter(v for v in verdicts if v)
    print(f"[s2] フェーズ1の指摘との照合: "
          f"一致 {vc['◯']} / ほぼ同率 {vc['≒']} / 不一致 {vc['×']}")
    if vc["×"] or vc["≒"]:
        print("[s2] 計算が指摘と食い違うケース（シートでも赤くなります）:")
        for r, v in zip(rows, verdicts):
            if v in ("×", "≒"):
                print(f"     {v} No{r[0]:>3} {r[1]} {r[3][:24]}  指摘: {r[7][:40]}")
    if flips:
        print(f"[s2] カードの色（主タグ）が旧マップから変わるケース {len(flips)}件:")
        for no, cid, title, a, b, tag_changed in flips:
            print(f"     No{no:>3} {cid} {title[:24]:26} {a} → {b}"
                  f"{'  ※タグ自体も変更' if tag_changed else ''}")
    if evens_by_default:
        print(f"[s2] 計算がほぼ同率で既定を「{EVEN}」にしたケース {len(evens_by_default)}件: "
              + " ".join(evens_by_default))
    if widened:
        print(f"[s2] 順番を変えたときに差を{MIN_GAP_PT}ptまで押し広げるケース {len(widened)}件:")
        for no, cid, title, before, after in widened:
            print(f"     No{no:>3} {cid} {title[:22]:24} {before} → {after}")
    multi = [r for r in rows if r[5] > 1]
    dist = {}
    for r in rows:
        dist[r[5]] = dist.get(r[5], 0) + 1
    print(f"[s2] {len(rows)}件 / 順番を選ぶ対象（2タグ以上）{len(multi)}件")
    print("[s2] タグ数の分布: " + " ".join(f"{k}タグ={dist[k]}" for k in sorted(dist)))
    print(f"[s2] プルダウン: 2タグ=2通り+「{EVEN}」/ 3タグ=6通り+「{EVEN}」")
    print(f"[s2] 出力: {OUTDIR / 'ratio_review.csv'}")
    print(f"[s2] 出力: {OUTDIR / 'ratio_baseline.json'}")


if __name__ == "__main__":
    main()
