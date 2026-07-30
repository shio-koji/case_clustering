#!/usr/bin/env python3
"""フェーズ1（タグ目視確認）用のスプレッドシート取り込みCSVを生成する。

入力:
  cache/cases_list.json      … CALL4から取得した全95件
  tag_review/shiina_list.tsv … 椎名さん作成の88件リスト（件数/状況/メモ/ケース名）
出力:
  tag_review/out/tag_review.csv  … Googleスプレッドシートにインポートする1枚
  tag_review/out/id_map.tsv      … No↔case_id 対応表（再取り込み時の照合用）

設計上の要点:
  - 「アーカイブ」タグは上限3のカウント対象外なので現行タグに含めず、状況列で表す。
  - 現行タグ・修正後タグはそれぞれ1列にまとめ、Googleスプレッドシートの
    マルチセレクト・ドロップダウン（複数選択を許可したチップ表示）で扱う。
    そのため区切りは Sheets が使う "、" ではなく SEP = ", " でなければならない。
  - 両列を隣接させて同一の検証ルールを共有させると、手作業になる
    「複数選択を許可」＋チップ色11個の設定が1回で両方に効く。
  - 変更有無の判定は両側を「タグ一覧」の並び順で組み直してから比較するので、
    入力した順番には依存しない（setup.gs 側の数式）。

作成: 2026-07-30。
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cache" / "cases_list.json"
SHIINA = ROOT / "tag_review" / "shiina_list.tsv"
OUTDIR = ROOT / "tag_review" / "out"

# 音信不通のため今回の対象外（椎名さん確認済み）
UNREACHABLE = {
    "I0000050", "I0000072", "I0000088", "I0000089",
    "I0000097", "I0000100", "I0000102",
}

# 既存タグ。現行の付与件数が多い順。setup.gs の TAGS と順序まで一致させること。
TAGS = [
    "公正な手続",
    "政治参加・表現の自由",
    "外国にルーツを持つ人々",
    "刑事司法",
    "ジェンダー・セクシュアリティ",
    "環境・災害",
    "働き方",
    "医療・福祉・障がい",
    "情報公開",
    "沖縄",
    "個人情報・プライバシー",
]

ARCHIVE_TAG = "アーカイブ"
MAX_TAGS = 3        # 原則の上限
MAX_TAGS_HARD = 4   # 例外として入力できる上限（リンさんのケースの4タグ維持を許すため）
# Googleスプレッドシートのマルチセレクトはカンマ+半角スペースで値を連結する。
# ここを変えるとチップとして認識されないので固定。
SEP = ", "
CASE_URL = "https://www.call4.jp/info.php?type=items&id={}"

HEADER = [
    "No", "case_id", "状況", "メモ", "ケース名", "概要",
    "現行タグ", "修正後タグ",
    "変更あり?", "コメント・理由", "確認者", "CALL4最終判断",
    "URL",
]


def norm(s: str) -> str:
    """タイトル照合用の正規化。全角/半角・波ダッシュ・各種ダッシュ・引用符・空白を吸収。"""
    s = unicodedata.normalize("NFKC", s)
    for ch in "〜～~":
        s = s.replace(ch, "~")
    for ch in "—―－–ー-−":
        s = s.replace(ch, "-")
    for ch in "“”\"「」『』":
        s = s.replace(ch, "")
    return re.sub(r"\s+", "", s)


def load_shiina():
    rows = []
    with SHIINA.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                sys.exit(f"列数が足りない行: {line!r}")
            rows.append({
                "no": int(parts[0]),
                "status": parts[1].strip(),
                "memo": parts[2].strip(),
                "title": parts[3].strip(),
            })
    return rows


def main():
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    shiina = load_shiina()

    # マルチセレクトの1セルに複数タグが入るため、件数集計も変更判定も
    # 部分一致（COUNTIF のワイルドカード / SEARCH）で行う。
    # タグ名が互いの部分文字列だと誤カウントするので、ここで保証しておく。
    for a in TAGS:
        for b in TAGS:
            if a != b and a in b:
                sys.exit(f"タグ名が部分文字列関係にあります: {a!r} ⊂ {b!r}")

    by_exact = {c["title"]: c for c in cases}
    by_norm = {}
    for c in cases:
        by_norm.setdefault(norm(c["title"]), []).append(c)

    matched, unmatched, fuzzy = [], [], []
    for r in shiina:
        c = by_exact.get(r["title"])
        if c is None:
            cand = by_norm.get(norm(r["title"]), [])
            if len(cand) == 1:
                c = cand[0]
                fuzzy.append((r["no"], r["title"], c["title"]))
        if c is None:
            unmatched.append(r)
        else:
            matched.append((r, c))

    if unmatched:
        for r in unmatched:
            print(f"  !! 未マッチ no={r['no']} {r['title']}", file=sys.stderr)
        sys.exit("タイトルが照合できない行があります。処理を中止しました。")

    ids = {c["id"] for _, c in matched}
    if len(ids) != len(matched):
        sys.exit("同一 case_id に複数行が対応しています。")
    if ids & UNREACHABLE:
        sys.exit(f"音信不通ケースが混入: {sorted(ids & UNREACHABLE)}")

    unknown = {t for _, c in matched for t in c["tags"]} - set(TAGS) - {ARCHIVE_TAG}
    if unknown:
        sys.exit(f"TAGS に無いタグが出現: {sorted(unknown)}")

    for r, c in matched:
        if (ARCHIVE_TAG in c["tags"]) != (r["status"] == ARCHIVE_TAG):
            print(f"  ! 状況不一致 no={r['no']} {c['id']} "
                  f"シート={r['status']} タグ={c['tags']}", file=sys.stderr)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    over_limit = []

    # BOMは付けない。GoogleスプレッドシートはBOM無しUTF-8を正しく読むが、
    # BOM付きだとA1が "﻿No" になり setup.gs の見出し検出が落ちる。
    with (OUTDIR / "tag_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r, c in matched:
            tags = [t for t in TAGS if t in c["tags"]]   # TAGS の並び順に正規化
            memo = r["memo"]
            if len(tags) > MAX_TAGS:
                over_limit.append((c["id"], c["title"], tags))
                # 3つに減らすか4つのまま維持するかは判断事項。現行を初期値として提示し、
                # どちらも選べるようにする（空にすると「維持」が選びにくくなる）。
                memo = (f"{memo} ※現行{len(tags)}タグ（原則上限{MAX_TAGS}の例外）。"
                        f"{MAX_TAGS}つに減らすか{len(tags)}つのまま維持するか要判断").strip()
            # 概要は改行を畳む（セル内改行はレビューしづらく、再取り込みも壊れやすい）
            desc = re.sub(r"\s+", " ", (c.get("description") or "")).strip()

            joined = SEP.join(tags)
            w.writerow([
                r["no"], c["id"], r["status"], memo, c["title"], desc,
                joined,     # 現行タグ（読み取り専用）
                joined,     # 修正後タグ（初期値は現行と同じ）
                "", "", "", "",              # 変更あり?/コメント/確認者/最終判断
                CASE_URL.format(c["id"]),
            ])

    with (OUTDIR / "id_map.tsv").open("w", encoding="utf-8") as f:
        f.write("No\tcase_id\t状況\t現行タグ\tケース名\n")
        for r, c in matched:
            tags = [t for t in TAGS if t in c["tags"]]
            f.write(f"{r['no']}\t{c['id']}\t{r['status']}\t{SEP.join(tags)}\t{c['title']}\n")

    print(f"照合: {len(matched)}件（うち正規化一致 {len(fuzzy)}件）/ 未マッチ 0件")
    if over_limit:
        print(f"原則上限{MAX_TAGS}の超過: {len(over_limit)}件"
              f"（現行タグを初期値として提示し、削減／維持のどちらも選べるようにしました）")
        for cid, title, tags in over_limit:
            print(f"    {cid} {title} {tags}")
    counts = {t: sum(1 for _, c in matched if t in c["tags"]) for t in TAGS}
    print("現行タグ件数:")
    for t in TAGS:
        flag = "  ← 件数不足（フェーズ2の割合計算が不安定）" if counts[t] < 3 else ""
        print(f"    {counts[t]:3d}  {t}{flag}")
    print(f"出力: {OUTDIR / 'tag_review.csv'}")
    print(f"出力: {OUTDIR / 'id_map.tsv'}")


if __name__ == "__main__":
    main()
