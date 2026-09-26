#!/usr/bin/env python3
"""Canvaのケースページの上下ライン色が、確定した主要タグの色と合っているか照合する。

Canvaの1ケース1ページのデザインは、ページ上端と下端に細いラインが入っていて、
その色がそのケースの主要タグ（＝割合がいちばん大きいタグ）を表している。
ページ数が多いと目視では追えないので、書き出した画像から色を測って突き合わせる。

使い方:
  1. Canva で 共有 > ダウンロード > ファイルの種類「PNG」>「すべてのページ」を選び、
     ZIPを canva_check/pages/ に展開する（PNGは無圧縮なので色が正確に取れる）
  2. python3 canva_check/check_colors.py

  ページとケースの対応は、ページ内のQRコード（CALL4のケースURL）から取る。
  QRを読むには opencv が必要:  .venv/bin/pip install opencv-python-headless
  opencv が無い場合はファイル名の順番をケースのNo順とみなす（--by-order）。

出力:
  canva_check/out/report.tsv   … 1ページ1行の照合結果
  標準出力に不一致の一覧

色の判定:
  ページの地色を除いたうえで上端・下端のラインの色を取り、
  タグ配色の中から OKLab 上で最も近い色を選ぶ。
  どのタグからも離れている場合は「該当なし」として報告する（配色ミスの検出）。
"""

import argparse
import colorsys
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAGES = HERE / "pages"
OUT = HERE / "out"

# 照合先の候補。Canvaがどれを使っているか分からないので全部試して、
# 全体として最も当てはまるものを採用する。
PALETTES = {
    "カード・GUIの濃色": {
        "個人情報・プライバシー": "#22504e", "医療・福祉・障がい": "#ff4709",
        "ジェンダー・セクシュアリティ": "#ff9423", "刑事司法": "#9f6e34",
        "環境・災害": "#2e9d7e", "働き方": "#99b73d", "公正な手続": "#fe7389",
        "沖縄": "#3970cb", "外国にルーツを持つ人々": "#6b5498",
        "政治参加・表現の自由": "#3daac8", "情報公開": "#033064",
    },
    "GUIの淡色": {
        "個人情報・プライバシー": "#bac9c8", "医療・福祉・障がい": "#ffcfbf",
        "ジェンダー・セクシュアリティ": "#ffdccb", "刑事司法": "#e2d3c2",
        "環境・災害": "#b6ddd2", "働き方": "#e0e9c5", "公正な手続": "#ffdbe0",
        "沖縄": "#bacded", "外国にルーツを持つ人々": "#d4cde1",
        "政治参加・表現の自由": "#c5e5ee", "情報公開": "#b3c1d0",
    },
    "確認シートのチップ色": {
        "公正な手続": "#daecfd", "政治参加・表現の自由": "#fce584",
        "外国にルーツを持つ人々": "#ffc4fe", "刑事司法": "#a4f6b5",
        "ジェンダー・セクシュアリティ": "#68f2ff", "環境・災害": "#fdae73",
        "働き方": "#c1b0fe", "医療・福祉・障がい": "#5bd7b8", "情報公開": "#b8c86a",
        "沖縄": "#66caff", "個人情報・プライバシー": "#fd9cba",
    },
}
CASE_ID_RE = re.compile(r"(I\d{7})")


# ---- 色の距離（OKLab）----------------------------------------------------
def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def srgb_to_oklab(rgb):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(x) for x in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (x ** (1 / 3) if x > 0 else -((-x) ** (1 / 3)) for x in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def dE(a, b):
    """OKLab距離×100。人の感覚に近い尺度で、2〜3以下ならほぼ同色。"""
    x, y = srgb_to_oklab(a), srgb_to_oklab(b)
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(x, y))) * 100


def nearest(rgb, palette):
    best = min(palette.items(), key=lambda kv: dE(rgb, hex2rgb(kv[1])))
    return best[0], dE(rgb, hex2rgb(best[1]))


# ---- ラインの色を測る ----------------------------------------------------
def row_colors(im):
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    # 各行の代表色は中央値（文字やロゴの影響を受けにくい）
    return np.median(a, axis=1)


def page_background(im):
    a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    mid = a[int(h * .4):int(h * .6), int(w * .4):int(w * .6)].reshape(-1, 3)
    c = Counter(map(tuple, mid[::7]))
    return tuple(int(x) for x in c.most_common(1)[0][0])


def find_band(rows, bg, top, min_de=6.0, max_scan=0.12,
              min_thick=4, edge_tol=8):
    """紙の端に接している帯（＝ライン）の色を返す。

    条件を2つ課している。どちらか片方だけでは誤検出した:
      - **端から edge_tol px 以内に始まる**こと。これが無いと、
        ページ下部にある白い吹き出し（引用ボックス）を拾ってしまう。
      - **太さが min_thick px 以上**あること。これが無いと、
        最下部に入る1〜2pxの白い縁を拾ってしまうページがあった。
    条件を満たす帯が複数あれば太い方を採る。

    返り値: (色, 太さ[px], 端からの位置[px]) / 見つからなければ None
    """
    n = len(rows)
    span = max(3, int(n * max_scan))
    idx = list(range(span)) if top else list(range(n - 1, n - span - 1, -1))
    runs, cur, start = [], [], None
    for i in idx:
        c = tuple(int(x) for x in rows[i])
        if dE(c, bg) >= min_de:
            if start is None:
                start = i
            cur.append(c)
        else:
            if cur:
                runs.append((cur, start))
            cur, start = [], None
    if cur:
        runs.append((cur, start))
    if not runs:
        return None

    def edge_dist(start):
        return start if top else n - 1 - start

    near = [r for r in runs if edge_dist(r[1]) <= edge_tol]
    thick = [r for r in near if len(r[0]) >= min_thick]
    pick = thick or near or runs
    run, start = max(pick, key=lambda r: len(r[0]))
    col = tuple(int(x) for x in np.median(np.array(run), axis=0))
    return col, len(run), (start if top else n - 1 - start)


# ---- ページとケースの対応 ------------------------------------------------
def read_qr(path):
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    det = cv2.QRCodeDetector()
    # 3倍まで拡大し、必要なら2値化して試す。等倍では読めず3倍で読めたページがあった
    # （Canvaの書き出しでQRが小さめに入るページがある）。
    for scale in (1, 2, 3):
        m = (cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
             if scale > 1 else img)
        variants = [m]
        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        variants.append(gray)
        variants.append(cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        for v in variants:
            try:
                ok, infos, _, _ = det.detectAndDecodeMulti(v)
                if ok:
                    for t in infos:
                        hit = CASE_ID_RE.search(t or "")
                        if hit:
                            return hit.group(1)
            except Exception:
                pass
            try:
                txt, _, _ = det.detectAndDecode(v)
            except Exception:
                txt = ""
            hit = CASE_ID_RE.search(txt or "")
            if hit:
                return hit.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=str(PAGES), help="書き出したページ画像のフォルダ")
    ap.add_argument("--by-order", action="store_true",
                    help="QRを使わず、ファイル名順をケースのNo順とみなす")
    ap.add_argument("--palette", choices=list(PALETTES),
                    help="照合する配色を固定する（既定は自動判定）")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.pages).glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not files:
        sys.exit(f"{args.pages} に画像がありません。CanvaからPNGで書き出してください。")

    final = json.loads(
        (ROOT / "tag_ratio" / "out" / "soft_tags_final.json").read_text(encoding="utf-8"))
    ratios = final["ratios"]
    cases = json.loads(
        (ROOT / "tag_ratio" / "out" / "cases_reviewed.json").read_text(encoding="utf-8"))
    meta = {c["id"]: c for c in cases["cases"]}
    by_no = {c["no"]: c["id"] for c in cases["cases"]}
    main_tag = {cid: max(v, key=v.get) for cid, v in ratios.items()}

    rows_out, observed = [], []
    for i, f in enumerate(files):
        im = Image.open(f)
        bg = page_background(im)
        rows = row_colors(im)
        top = find_band(rows, bg, True)
        bot = find_band(rows, bg, False)
        cid = None if args.by_order else read_qr(f)
        if cid is None and args.by_order:
            cid = by_no.get(i + 1)
        rows_out.append({"file": f.name, "bg": bg, "top": top, "bot": bot, "case_id": cid})
        if top:
            observed.append(top[0])

    # どの配色が全体として当てはまるかを決める
    if args.palette:
        pname = args.palette
    else:
        score = {k: np.median([nearest(c, p)[1] for c in observed]) if observed else 1e9
                 for k, p in PALETTES.items()}
        pname = min(score, key=score.get)
        print("配色の当てはまり（中央値ΔE・小さいほど一致）:")
        for k, v in sorted(score.items(), key=lambda kv: kv[1]):
            print(f"  {k:22} ΔE={v:.1f}{'  ← 採用' if k == pname else ''}")
    pal = PALETTES[pname]
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    bad, unknown, nocase = [], [], []
    with (OUT / "report.tsv").open("w", encoding="utf-8") as fh:
        fh.write("file\tcase_id\tNo\tケース名\t期待する主要タグ\t上端の色\t上端の判定\tΔE"
                 "\t下端の色\t下端の判定\t上下一致\t結果\n")
        for r in rows_out:
            cid = r["case_id"]
            want = main_tag.get(cid) if cid else None
            def fmt(b):
                return "#%02x%02x%02x" % b[0] if b else "—"
            t_name, t_de = (nearest(r["top"][0], pal) if r["top"] else ("—", 0))
            b_name, b_de = (nearest(r["bot"][0], pal) if r["bot"] else ("—", 0))
            same = "○" if (r["top"] and r["bot"]
                           and dE(r["top"][0], r["bot"][0]) < 3) else "×"
            if not cid:
                verdict = "ケース不明"
                nocase.append(r["file"])
            elif t_de > 10:
                verdict = "配色外"
                unknown.append((r["file"], cid, fmt(r["top"]), t_name, t_de))
            elif t_name == want and (b_name == want or not r["bot"]):
                verdict = "一致"
            else:
                verdict = "不一致"
                bad.append((r["file"], cid, want, t_name, b_name, fmt(r["top"]), t_de))
            fh.write("\t".join([
                r["file"], cid or "", str(meta[cid]["no"]) if cid else "",
                meta[cid]["title"] if cid else "", want or "",
                fmt(r["top"]), t_name, f"{t_de:.1f}",
                fmt(r["bot"]), b_name, same, verdict]) + "\n")

    print(f"照合: {len(rows_out)}ページ / 配色「{pname}」で判定")
    seen = Counter(r["case_id"] for r in rows_out if r["case_id"])
    dup = {k: v for k, v in seen.items() if v > 1}
    if dup:
        print(f"\n■ 同じケースが複数ページにあります {len(dup)}件")
        for cid, k in dup.items():
            files = [r["file"] for r in rows_out if r["case_id"] == cid]
            print(f"  {cid} {meta[cid]['title'][:26]} … {k}ページ（{', '.join(files)}）")
    absent = sorted(set(meta) - set(seen))
    if absent:
        print(f"\n■ Canvaに見つからないケース {len(absent)}件")
        for cid in absent:
            print(f"  No{meta[cid]['no']} {cid} {meta[cid]['title'][:30]}")
    if bad:
        print(f"\n■ 主要タグと色が合っていないページ {len(bad)}件")
        for f, cid, want, got, gotb, hexv, de in bad:
            print(f"  {f}  {cid} {meta[cid]['title'][:22]}")
            print(f"       期待: {want} ／ 実際の色 {hexv} は「{got}」に最も近い（ΔE {de:.1f}）")
            if gotb != got:
                print(f"       ※下端は「{gotb}」で上端と違う")
    if unknown:
        print(f"\n■ どのタグ色からも離れているページ {len(unknown)}件（配色ミスの疑い）")
        for f, cid, hexv, got, de in unknown:
            print(f"  {f}  {cid}  実際の色 {hexv}（最も近いのは {got} でもΔE {de:.1f}）")
    if nocase:
        print(f"\n■ ケースを特定できなかったページ {len(nocase)}件")
        print("   " + ", ".join(nocase[:12]))
        print("   QRが読めていません。opencv を入れるか --by-order を使ってください。")
    if not (bad or unknown or nocase):
        print("\nすべてのページで上下のラインが主要タグの色と一致しています。")
    print(f"\n詳細: {OUT / 'report.tsv'}")


if __name__ == "__main__":
    main()
