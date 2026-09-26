#!/usr/bin/env python3
"""フェーズ2 Stage 0 — 生データ＋確定タグから「ケース情報」を1本に組み直す。

入力:
  cache/cases_list.json           … CALL4の一覧（95件・タグ・サムネイル）
  cache/case_<id>.json            … ケース本文（contents）
  cache/corpus_clean.json         … 整形済みの説明文（行順の基準にもなる）
  tag_review/out/reviewed_tags.json … フェーズ1の確定タグ（88件）
  tag_review/out/tag_review.csv    … No / 状況 の対応（椎名さんの通し番号）

出力:
  tag_ratio/out/cases_reviewed.json … 88件。以降の工程が参照する唯一の入力

なぜ作るか:
  タグ割合の再計算・カード生成・2部グラフが、それぞれ別のファイル
  （tag_review.csv / soft_tags.json / cases_list.json）を突き合わせていたので、
  タグが変わるたびに整合を取る場所が増えていた。確定タグを反映した
  「ケース情報」をここで1つに固め、後段はこれだけを読む。

タグは cases_list.json の生タグではなく reviewed_tags.json の確定タグを使う。
生タグは tags_before として残し、差分を追えるようにしておく。

作成: 2026-08-20。
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
REVIEW = ROOT / "tag_review" / "out"
OUTDIR = Path(__file__).resolve().parent / "out"

sys.path.insert(0, str(ROOT / "tag_review"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tag_review import ARCHIVE_TAG, TAGS, UNREACHABLE  # noqa: E402
from phase1_notes import load_notes  # noqa: E402

CASE_URL = "https://www.call4.jp/info.php?type=items&id={}"


def clean_html(text: str) -> str:
    """plan02/s0_tokenize.clean_html と同じ整形（実体参照・タグ・URL・空白）。

    埋め込みに渡すテキストを過去と1文字も変えないため、ここは同じ処理で通す。
    """
    if not text:
        return ""
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&rdquo;", '"')
    text = text.replace("&ldquo;", '"').replace("&nbsp;", " ")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    reviewed_path = REVIEW / "reviewed_tags.json"
    if not reviewed_path.exists():
        sys.exit(f"{reviewed_path} がありません。\n"
                 "  1. スプレッドシート「タグ確認」を CSV でダウンロード\n"
                 "  2. tag_review/out/reviewed.csv として配置\n"
                 "  3. python3 tag_review/ingest_reviewed.py")

    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    tags_by_case = reviewed["tags_by_case"]

    if reviewed["tags"] != TAGS:
        sys.exit("reviewed_tags.json のタグ語彙が make_tag_review.TAGS と一致しません。")

    listing = {c["id"]: c for c in
               json.loads((CACHE / "cases_list.json").read_text(encoding="utf-8"))["cases"]}
    corpus = {c["id"]: c for c in
              json.loads((CACHE / "corpus_clean.json").read_text(encoding="utf-8"))}

    # No と 状況 はフェーズ1のシート（椎名さんの通し番号）に由来する。
    meta = {}
    with (REVIEW / "tag_review.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            meta[r["case_id"]] = {"no": int(r["No"]), "status_label": r["状況"]}

    missing = sorted(set(tags_by_case) - set(meta))
    if missing:
        sys.exit(f"tag_review.csv に無い case_id: {missing}")

    # フェーズ1の申し送り（タグの大小関係の指摘・未決着フラグ）。
    # 「割合をどうしたいか」がすでに書かれているので、フェーズ2の入力として持ち回す。
    notes = load_notes(REVIEW / "reviewed.csv")

    cases, unresolved = [], []
    for cid in sorted(tags_by_case):
        if cid in UNREACHABLE:
            sys.exit(f"音信不通ケースが確定タグに混入: {cid}")
        src = listing[cid]
        detail = json.loads((CACHE / f"case_{cid}.json").read_text(encoding="utf-8"))
        contents = clean_html(detail.get("contents") or "")
        desc = re.sub(r"\s+", " ", (src.get("description") or "")).strip()
        # 埋め込みに渡すテキストは plan02 と同一構成（タイトル＋説明＋本文、更新は含めない）
        parts = [p for p in [src["title"], desc, contents] if p.strip()]
        text = unicodedata.normalize("NFKC", "\n".join(parts))

        raw_tags = [t for t in TAGS if t in src["tags"]]
        cases.append({
            "id": cid,
            "no": meta[cid]["no"],
            "title": src["title"],
            "status": "archived" if ARCHIVE_TAG in src["tags"] else "active",
            "status_label": meta[cid]["status_label"],
            "description": desc,
            "url": CASE_URL.format(cid),
            "thumbnail": src.get("thumbnail"),
            "tags": tags_by_case[cid],       # ← フェーズ1で確定したタグ
            "tags_before": raw_tags,         # ← CALL4の生タグ（アーカイブを除く）
            "n_tags": len(tags_by_case[cid]),
            "text": text,
            "text_length": len(text),
            "corpus_text_length": corpus[cid]["text_length"],
        })
        nt = notes.get(cid)
        if nt:
            # 指摘に出てくるタグが確定タグから消えている場合は照合できない
            refs = list(nt["chain"]) + ([nt["max_tag"]] if nt["max_tag"] else [])
            stale = [t for t in refs if t not in tags_by_case[cid]]
            cases[-1]["phase1"] = {
                "note": nt["text"],
                "status": nt["status"],
                "unsettled": nt["unsettled"],
                "chain": [] if stale else nt["chain"],
                "max_tag": None if stale else nt["max_tag"],
                "stale_refs": stale,
            }
            if nt["unsettled"]:
                unresolved.append(cid)

    # 状況ラベル（椎名さんのシート）と生タグの「アーカイブ」が食い違う場合は黙って進めない
    for c in cases:
        if (c["status"] == "archived") != (c["status_label"] == ARCHIVE_TAG):
            print(f"  ! 状況不一致 {c['id']} シート={c['status_label']} "
                  f"タグ={c['tags_before']}", file=sys.stderr)

    counts = {t: sum(1 for c in cases if t in c["tags"]) for t in TAGS}
    before = {t: sum(1 for c in cases if t in c["tags_before"]) for t in TAGS}
    changed = [c["id"] for c in cases if c["tags"] != c["tags_before"]]

    payload = {
        "generated_from": {
            "cases_list": "cache/cases_list.json",
            "reviewed_tags": "tag_review/out/reviewed_tags.json",
        },
        "n_cases": len(cases),
        "tags": TAGS,
        "excluded_unreachable": sorted(UNREACHABLE),
        "n_archived": sum(1 for c in cases if c["status"] == "archived"),
        "changed_cases": changed,
        "unsettled_cases": unresolved,
        "tag_counts_before": before,
        "tag_counts_after": counts,
        "cases": cases,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "cases_reviewed.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[s0] {len(cases)}件（アーカイブ {payload['n_archived']} / "
          f"進行中 {len(cases) - payload['n_archived']}）"
          f"／タグが変わったケース {len(changed)}件")
    n_note = sum(1 for c in cases if c.get("phase1", {}).get("note"))
    n_read = sum(1 for c in cases
                 if c.get("phase1", {}).get("chain") or c.get("phase1", {}).get("max_tag"))
    print(f"[s0] フェーズ1の指摘 {n_note}件（うち大小関係として読めたもの {n_read}件）"
          f"／未決着 {len(unresolved)}件")
    print(f"[s0] タグ数の分布: "
          + " ".join(f"{k}タグ={sum(1 for c in cases if c['n_tags'] == k)}"
                     for k in sorted({c["n_tags"] for c in cases})))
    print(f"{'タグ':<24}{'生タグ':>6}{'確定':>6}{'増減':>6}")
    for t in TAGS:
        flag = "  ← 3件未満（割合計算が不安定）" if counts[t] < 3 else ""
        print(f"{t:<24}{before[t]:>6}{counts[t]:>6}{counts[t] - before[t]:>+6}{flag}")
    print(f"[s0] 出力: {out}")


if __name__ == "__main__":
    main()
