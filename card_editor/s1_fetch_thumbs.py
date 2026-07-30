#!/usr/bin/env python3
"""card_editor/s1 — 各ケースのサムネイル画像を取得してローカルに置く。

対象は tag_review/out/tag_review.csv の88件（音信不通7件を除いたもの）。
cache/case_<id>.json の thumbnail は CALL4 のドキュメントルート相対パスなので、
BASE を頭に付ければそのまま取れる。

原本は最大 5760x3840 / 2.6MB のものがあるので、印刷に必要な分だけ残して縮小する。
カードの版下は最大 MASTER_MM（既定90mm）角なので、
その幅を 300dpi で満たす 1063px あれば足りる。実測ではサムネの最小幅が1200pxなので、
ほとんどのケースは「そのまま」か「わずかに縮小」で済む。

再実行時は既にあるファイルを飛ばす（冪等）。--force で取り直し。
"""
import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

from PIL import Image

BASE = "https://www.call4.jp/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "tag_review", "out", "tag_review.csv")
OUT = os.path.join(HERE, "cache", "thumbs")

MASTER_MM = 90.0      # 版下カードの一辺（mm）。s2 と一致させること
DPI = 300
MAX_W = int(round(MASTER_MM / 25.4 * DPI))   # 1063px
UA = "Mozilla/5.0 (case_clustering/card_editor; +research use)"


def case_rows():
    with open(CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.4, help="連続取得の間隔（秒）")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rows = case_rows()
    print(f"[s1] {len(rows)} 件（tag_review.csv）")

    manifest, errs, got, skip = {}, [], 0, 0
    for i, row in enumerate(rows, 1):
        cid = row["case_id"]
        dst = os.path.join(OUT, f"{cid}.jpg")
        meta_p = os.path.join(ROOT, "cache", f"case_{cid}.json")
        if not os.path.exists(meta_p):
            errs.append((cid, "cache/case_*.json が無い"))
            continue
        thumb = json.load(open(meta_p, encoding="utf-8")).get("thumbnail")
        if not thumb:
            errs.append((cid, "thumbnail フィールドが空"))
            continue

        if os.path.exists(dst) and not a.force:
            skip += 1
        else:
            try:
                raw = fetch(BASE + thumb)
                im = Image.open(io.BytesIO(raw))
                im = im.convert("RGB")
                if im.width > MAX_W:
                    h = round(im.height * MAX_W / im.width)
                    im = im.resize((MAX_W, h), Image.LANCZOS)
                im.save(dst, "JPEG", quality=92, optimize=True)
                got += 1
                print(f"  [{i:3d}/{len(rows)}] {cid} {im.width}x{im.height}")
                time.sleep(a.sleep)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                errs.append((cid, f"{type(e).__name__}: {e}"))
                continue

        with Image.open(dst) as im:
            manifest[cid] = {"file": os.path.relpath(dst, HERE),
                             "w": im.width, "h": im.height,
                             "src": BASE + thumb}

    json.dump(manifest, open(os.path.join(HERE, "cache", "thumbs.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[s1] 取得 {got} / 既存 {skip} / 失敗 {len(errs)}")
    for cid, msg in errs:
        print(f"  ! {cid}: {msg}", file=sys.stderr)

    # 印刷解像度の警告：版下 MASTER_MM に対して 300dpi を満たさないもの
    thin = [(c, m["w"]) for c, m in manifest.items() if m["w"] < MAX_W]
    if thin:
        print(f"[s1] 注意: {len(thin)} 件が {MASTER_MM}mm角で300dpi未満です。"
              f"版下を小さくするか、その分だけ拡大補間になります:")
        for c, w in sorted(thin, key=lambda x: x[1])[:15]:
            print(f"     {c}  幅{w}px → {MASTER_MM}mm で {w/(MASTER_MM/25.4):.0f}dpi")


if __name__ == "__main__":
    main()
