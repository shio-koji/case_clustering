#!/usr/bin/env python3
"""フェーズ1の確認済みシートを読み戻して検証し、フェーズ2の入力を作る。

使い方:
  1. スプレッドシートの「タグ確認」シートを
     ファイル > ダウンロード > カンマ区切り形式(.csv) で保存
  2. tag_review/out/reviewed.csv として置く
  3. python3 tag_review/ingest_reviewed.py

出力:
  tag_review/out/reviewed_tags.json … {case_id: [タグ, ...]} と差分サマリ

検証に落ちた場合は何も出力せず終了する（不正なタグ集合を後段に流さないため）。
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tag_review import (  # noqa: E402
    ARCHIVE_TAG, CASES, MAX_TAGS, MAX_TAGS_HARD, OUTDIR, TAGS, UNREACHABLE,
)

REVIEWED = OUTDIR / "reviewed.csv"
NEEDED = ["case_id", "修正後タグ"]


def split_tags(cell):
    """マルチセレクトのセルを分解する。

    Googleスプレッドシートは「タグA, タグB」の形で連結するが、
    手入力が混ざって全角カンマや読点になっていても拾えるようにしておく。
    """
    return [t for t in re.split(r"[,、，]", cell or "") if t.strip()]


def main():
    if not REVIEWED.exists():
        sys.exit(f"{REVIEWED} がありません。手順1〜2を先に実施してください。")

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    expected = {c["id"]: c for c in cases if c["id"] not in UNREACHABLE}

    with REVIEWED.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit("データ行がありません。")
    missing_cols = [c for c in NEEDED if c not in rows[0]]
    if missing_cols:
        sys.exit(f"必要な列がありません: {missing_cols}")
    errors, warnings, result, changes = [], [], {}, []
    seen = Counter()

    for i, r in enumerate(rows, start=2):
        cid = (r["case_id"] or "").strip()
        if cid not in expected:
            errors.append(f"行{i}: 未知のcase_id {cid!r}")
            continue
        seen[cid] += 1

        revised = [t.strip() for t in split_tags(r["修正後タグ"])]

        unknown = [t for t in revised if t not in TAGS]
        if unknown:
            errors.append(f"行{i} {cid}: 既存タグに無い値 {unknown}")
        if len(set(revised)) != len(revised):
            errors.append(f"行{i} {cid}: タグ重複 {revised}")
        if not revised:
            errors.append(f"行{i} {cid}: タグが1つもありません")
        if len(revised) > MAX_TAGS_HARD:
            errors.append(f"行{i} {cid}: {MAX_TAGS_HARD}個を超えています {revised}")
        elif len(revised) > MAX_TAGS:
            # 4個は例外として許容する。止めずに一覧で報告してCALL4の確認に回す。
            warnings.append(f"{cid} {expected[cid]['title']}: "
                            f"{len(revised)}タグ（原則上限{MAX_TAGS}の例外） {revised}")

        canonical = [t for t in TAGS if t in set(revised)]
        result[cid] = canonical

        before = [t for t in TAGS if t in expected[cid]["tags"]]
        if before != canonical:
            changes.append({
                "case_id": cid,
                "title": expected[cid]["title"],
                "before": before,
                "after": canonical,
                "added": [t for t in canonical if t not in before],
                "removed": [t for t in before if t not in canonical],
                "comment": (r.get("コメント・理由") or "").strip(),
                "reviewer": (r.get("確認者") or "").strip(),
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

    counts = Counter(t for tags in result.values() for t in tags)
    before_counts = Counter(t for cid in result for t in expected[cid]["tags"] if t != ARCHIVE_TAG)

    payload = {
        "n_cases": len(result),
        "max_tags": MAX_TAGS,
        "max_tags_hard": MAX_TAGS_HARD,
        "over_max_tags": [cid for cid, tags in result.items() if len(tags) > MAX_TAGS],
        "tags": TAGS,
        "excluded_unreachable": sorted(UNREACHABLE),
        "archive": sorted(cid for cid in result if ARCHIVE_TAG in expected[cid]["tags"]),
        "tags_by_case": result,
        "changes": changes,
        "tag_counts_before": {t: before_counts[t] for t in TAGS},
        "tag_counts_after": {t: counts[t] for t in TAGS},
    }
    out = OUTDIR / "reviewed_tags.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"検証OK: {len(result)}件 / 変更 {len(changes)}件")
    if warnings:
        print(f"\n例外（原則上限{MAX_TAGS}超過）{len(warnings)}件：")
        for w in warnings:
            print("  - " + w)
        print()
    print(f"{'タグ':<24}{'現行':>5}{'修正後':>7}{'増減':>6}")
    for t in TAGS:
        b, a = before_counts[t], counts[t]
        flag = "  ← 3件未満" if a < 3 else ""
        print(f"{t:<24}{b:>5}{a:>7}{a - b:>+6}{flag}")
    print(f"\n出力: {out}")


if __name__ == "__main__":
    main()
