#!/usr/bin/env python3
"""フェーズ2 Stage 3 — 確認済みの割合シートを読み戻して検証し、確定版を作る。

使い方:
  1. スプレッドシートの「タグ割合」シートを
     ファイル > ダウンロード > カンマ区切り形式(.csv) で保存
  2. tag_ratio/out/reviewed_ratios.csv として置く
  3. python3 tag_ratio/s3_ingest_ratios.py

出力:
  tag_ratio/out/soft_tags_final.json … 後段（カード・2部グラフ）が読む確定版

## 正は「順番」列

人が触るのは順番のプルダウンだけなので、**割合はここで作り直す**
（シートの割合列は数式の結果＝表示用）。こうしておくと、
シートの数式が古いままでも壊れていても、成果物の数字は狂わない。

導出は s2_make_ratio_sheet.resolve_ratio を共用する。シート側（setup_ratio.gs）は
同じ規則を数式で書いてあるので（押し広げ後の値は s2 が「調整」列に書き込み、
数式はそれを参照するだけ）、読み戻したときに両者が食い違っていれば警告する。

検証に落ちた場合は何も出力せず終了する（不正な割合を成果物に流さないため）。
タグ集合はフェーズ1で確定済みなので、シート側で増減していたらエラーにする。

作成: 2026-08-20。
"""

import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s2_make_ratio_sheet import (  # noqa: E402
    EVEN, MAX_SLOTS, MIN_SHARE_PT, apply_min_share, default_order, resolve_ratio,
)

OUTDIR = HERE / "out"
REVIEWED = OUTDIR / "reviewed_ratios.csv"
TOTAL = 100


def parse_num(cell):
    """セルの数値を読む。

    Googleスプレッドシートは CSV に「見た目の値」を書き出すことがあるので、
    書式で付けた % や全角数字・カンマを落としてから解釈する。
    """
    s = unicodedata.normalize("NFKC", (cell or "").strip())
    if not s:
        return None
    s = s.replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def main():
    if not REVIEWED.exists():
        sys.exit(f"{REVIEWED} がありません。手順1〜2を先に実施してください。")
    cases_path = OUTDIR / "cases_reviewed.json"
    soft_path = OUTDIR / "soft_tags.json"
    for p in (cases_path, soft_path):
        if not p.exists():
            sys.exit(f"{p} がありません。s0_cases.py / s1_ratios.py を先に実行してください。")

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    expected = {c["id"]: c for c in cases["cases"]}
    soft = json.loads(soft_path.read_text(encoding="utf-8"))
    computed = soft["ratios"]
    baseline_path = OUTDIR / "ratio_baseline.json"
    baseline = (json.loads(baseline_path.read_text(encoding="utf-8"))["ratios"]
                if baseline_path.exists() else {})

    with REVIEWED.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("データ行がありません。")

    needed = (["case_id", "順番"]
              + [f"タグ{k}" for k in range(1, MAX_SLOTS + 1)]
              + [f"計算{k}" for k in range(1, MAX_SLOTS + 1)])
    missing_cols = [c for c in needed if c not in rows[0]]
    if missing_cols:
        sys.exit(f"必要な列がありません: {missing_cols}\n"
                 f"  読み込んだ列: {list(rows[0])}")

    errors, warnings, result, changes, evens, floored = [], [], {}, [], [], []
    seen = Counter()

    for i, r in enumerate(rows, start=2):
        cid = (r["case_id"] or "").strip()
        # チェックボックス等があるとGoogleは末尾の空行まで書き出す
        if not cid and not (r.get("順番") or "").strip():
            continue
        if cid not in expected:
            errors.append(f"行{i}: 未知のcase_id {cid!r}")
            continue
        seen[cid] += 1

        slot_tags, mags, bad = [], [], False
        for k in range(1, MAX_SLOTS + 1):
            tag = (r[f"タグ{k}"] or "").strip()
            if not tag:
                continue
            v = parse_num(r[f"計算{k}"])
            if v is None:
                errors.append(f"行{i} {cid}: 「計算{k}」が数値として読めません "
                              f"({r[f'計算{k}']!r})")
                bad = True
                continue
            slot_tags.append(tag)
            mags.append(int(round(v)))
        if bad:
            continue

        if sorted(slot_tags) != sorted(expected[cid]["tags"]):
            errors.append(f"行{i} {cid}: タグ集合がフェーズ1の確定と違います "
                          f"シート={slot_tags} 確定={expected[cid]['tags']}")
            continue
        if sum(mags) != TOTAL:
            errors.append(f"行{i} {cid}: 計算値の合計が{TOTAL}ではありません（{sum(mags)}）"
                          "／「計算」列が書き換えられている可能性があります")
            continue

        # 半角 > で書かれていても読めるようにしておく
        order = unicodedata.normalize("NFKC", (r["順番"] or "").strip()).replace(">", "＞")
        if len(slot_tags) == 1:
            if order and order not in (EVEN, slot_tags[0]):
                warnings.append(f"{cid}: 1タグなのに順番が入っています（{order}）。無視します")
            ratio_pct = {slot_tags[0]: TOTAL}
        elif not order:
            errors.append(f"行{i} {cid}: 順番が選ばれていません（{'・'.join(slot_tags)}）")
            continue
        else:
            try:
                ratio_pct = resolve_ratio(order, slot_tags, mags)
            except ValueError as e:
                errors.append(f"行{i} {cid}: {e}")
                continue
            if order == EVEN:
                evens.append(cid)

        if sum(ratio_pct.values()) != TOTAL:
            errors.append(f"行{i} {cid}: 割合の合計が{TOTAL}になりません {ratio_pct}")
            continue

        # シートの数式が出した値と突き合わせる（数式のバグに気づくため）。
        # ただし「最低割合の底上げ」はレビュー後に追加した規則なので、
        # 底上げだけで説明できる差は数式のバグではない。区別して報告する。
        shown_row = {}
        for k in range(1, MAX_SLOTS + 1):
            tag = (r[f"タグ{k}"] or "").strip()
            if not tag or f"割合{k}" not in r:
                continue
            v = parse_num(r[f"割合{k}"])
            if v is not None:
                shown_row[tag] = int(round(v))
        if shown_row and shown_row != ratio_pct:
            explained = (sum(shown_row.values()) == TOTAL
                         and min(shown_row.values()) < MIN_SHARE_PT
                         and apply_min_share(
                             [shown_row[t] for t in sorted(shown_row,
                                                           key=lambda x: -shown_row[x])])
                         == [ratio_pct[t] for t in sorted(ratio_pct,
                                                          key=lambda x: -ratio_pct[x])])
            if explained:
                floored.append((cid, shown_row, dict(ratio_pct)))
            else:
                for tag, v in shown_row.items():
                    if v != ratio_pct.get(tag):
                        warnings.append(
                            f"{cid}: シートの表示({tag}={v}%)と"
                            f"読み戻しの計算({ratio_pct[tag]}%)が違います"
                            "／setup_ratio.gs の数式を確認してください")

        vocab = {t: j for j, t in enumerate(cases["tags"])}
        result[cid] = {t: round(p / TOTAL, 4)
                       for t, p in sorted(ratio_pct.items(), key=lambda kv: vocab[kv[0]])}

        # 「人が変えた」の判定は**順番の比較**で行う。割合の値で比べると、
        # 最低割合の底上げのように後から入れた規則の差まで
        # 「人が変えた」に混ざってしまう。
        base = baseline.get(cid) or {t: round(v * 100) for t, v in computed[cid].items()}
        if len(slot_tags) > 1 and order != default_order(slot_tags, mags):
            changes.append({
                "case_id": cid,
                "title": expected[cid]["title"],
                "order": order,
                "before": {t: base.get(t) for t in ratio_pct},
                "after": dict(ratio_pct),
                "comment": (r.get("コメント・理由") or "").strip(),
                "reviewer": (r.get("確認者") or "").strip(),
                "decision": (r.get("CALL4最終判断") or "").strip(),
            })

    for cid, k in seen.items():
        if k > 1:
            errors.append(f"{cid}: {k}行に重複して出現")
    for cid in sorted(set(expected) - set(seen)):
        errors.append(f"{cid} の行がありません: {expected[cid]['title']}")

    if errors:
        print(f"検証エラー {len(errors)}件：出力せず終了します", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        sys.exit(1)

    out = dict(soft)
    out["ratios"] = result
    out["ratio_source"] = "human-reviewed order (tag_ratio/out/reviewed_ratios.csv)"
    out["ratios_computed"] = computed          # 計算値も残して差分を追えるようにする
    out["ratio_changes"] = changes
    out["even_cases"] = evens
    out["min_share_pt"] = MIN_SHARE_PT
    out["floored_cases"] = [{"case_id": c, "before": b, "after": a}
                            for c, b, a in floored]
    dest = OUTDIR / "soft_tags_final.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"検証OK: {len(result)}件 / 順番を変えたケース {len(changes)}件"
          + (f"／「{EVEN}」{len(evens)}件" if evens else ""))
    if floored:
        print(f"\n最低割合{MIN_SHARE_PT}%への底上げ {len(floored)}件"
              "（レビュー後に追加した規則。シートの表示より新しい値）:")
        for cid, b, a in floored:
            bs = " ".join(f"{t}{v}" for t, v in sorted(b.items(), key=lambda kv: -kv[1]))
            as_ = " ".join(f"{t}{v}" for t, v in sorted(a.items(), key=lambda kv: -kv[1]))
            print(f"  {cid}: {bs}  →  {as_}")
    if warnings:
        print(f"\n注意 {len(warnings)}件:")
        for w in warnings:
            print("  - " + w)
    if changes:
        print("\n順番が変わったケース:")
        for ch in changes:
            before = " ".join(f"{t}{v}" for t, v in
                              sorted(ch["before"].items(), key=lambda kv: -(kv[1] or 0)))
            after = " ".join(f"{t}{v}" for t, v in
                             sorted(ch["after"].items(), key=lambda kv: -kv[1]))
            print(f"  {ch['case_id']} {ch['title'][:24]}")
            print(f"      {before}  →  {after}")
            if ch["comment"]:
                print(f"      コメント: {ch['comment'][:70]}")

    # タグ別の重み合計（そのタグが全体の何ケース分に相当するか）
    print(f"\n{'タグ':<24}{'件数':>5}{'計算':>8}{'確定':>8}{'増減':>8}")
    for t in cases["tags"]:
        n = sum(1 for cid in result if t in result[cid])
        w_new = sum(result[cid].get(t, 0) for cid in result)
        w_old = sum(computed[cid].get(t, 0) for cid in computed)
        print(f"{t:<24}{n:>5}{w_old:>8.2f}{w_new:>8.2f}{w_new - w_old:>+8.2f}")
    print(f"\n出力: {dest}")


if __name__ == "__main__":
    main()
