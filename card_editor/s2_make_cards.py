#!/usr/bin/env python3
"""card_editor/s2 — ケースごとの「カード」画像を作る（正方形・印刷解像度）。

構成は指定の見本どおり:
    上: CALL4のサムネイル（3:2。タイトルが焼き込まれている画像が多い）
    中: ケース名（テキスト）
    下: タグ割合の積み上げバー ＋ タグ名と割合の凡例

すべて同じ一辺（MASTER_MM）の正方形に揃える。ネットワーク図に88枚並べたときに
高さがバラバラだと配置が組めないため、タイトルは
「2行に収まるまでフォントを縮める → それでも入らなければ省略記号」で正方形に押し込む。

出力:
    out/cards/<id>.jpg          版下（MASTER_MM角 / DPI）。実寸の試し刷り用
    out/cards_preview/<id>.webp エディタ埋め込み用の軽量版
    out/cards.json              ケース情報＋各カードの実効解像度
    out/card_geometry.json      mm単位のレイアウト仕様と、確定した行分割・文字サイズ
    out/proof_A4.pdf            --proof 指定時。A4に実寸で並べた試し刷り

card_geometry.json を残しているのは、本番のPDFでは文字を
ラスタではなくベクタで置き直す必要があるため（写真だけを埋め込み画像にする）。
このファイルがあれば、同じ座標・同じ行分割でベクタ組版を再現できる。
"""
import argparse
import csv
import json
import os

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
THUMBS = os.path.join(HERE, "cache", "thumbs")

JP_FONT = ("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
           "54ef167d6c8e99a69a0d41ce252cc5995ba47580.asset/AssetData/YuGothic-Medium.otf")
JP_FONT_BOLD = JP_FONT   # YuGothic-Medium で兼用（Bold が無い環境でも動くように）

MASTER_MM = 90.0     # カードの一辺（mm）。s1 の MASTER_MM と一致させること
DPI = 300
PT = 25.4 / 72.0     # 1pt = 0.3528mm

# ---- カード内レイアウト（すべて mm。ここを直せば版下とベクタPDFの両方に効く） ----
G = {
    "side": MASTER_MM,
    "pad": 3.2,
    "img_ratio": 3 / 2,        # サムネイルの縦横比
    "gap_img_title": 2.6,
    "title_pt_max": 12.5,
    "title_pt_min": 9.0,
    "title_lines": 2,
    "title_leading": 1.42,     # 行送り（フォントサイズ倍）
    "gap_title_bar": 2.6,
    "bar_h": 5.2,
    "bar_pt": 8.0,             # バー内の「70%」
    "gap_bar_legend": 1.7,
    "legend_pt": 7.5,
    "border_pt": 0.75,         # 外周の罫線
}

# GUIのタグノードと同じ濃色。カード内の積み上げバーに使う。
# 配列順ではなくタグ名で対応させ、データ側の順序変更で色がずれないようにする。
TAG_COLORS = {
    "個人情報・プライバシー": "#22504e",
    "医療・福祉・障がい": "#fe7389",
    "ジェンダー・セクシュアリティ": "#ff9423",
    "刑事司法": "#9f6e34",
    "環境・災害": "#2e9d7e",
    "働き方": "#99b73d",
    "公正な手続": "#033064",
    "沖縄": "#3970cb",
    "外国にルーツを持つ人々": "#6b5498",
    "政治参加・表現の自由": "#3daac8",
    "情報公開": "#ff4709",
}

ARCHIVE_SUFFIX = "【アーカイブ】"
# 行頭に置かない文字（簡易禁則）
NO_LINE_START = "」』）］｝、。，．・？！ー〜:;：；%％"


def mm(v):
    """mm → px（DPI基準）"""
    return v / 25.4 * DPI


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def font(pt, path=JP_FONT):
    return ImageFont.truetype(path, int(round(mm(pt * PT))))


def wrap(text, f, width_px, max_lines):
    """日本語向けの折り返し。空白で切れないので1文字ずつ詰める。
    max_lines に収まらない場合は最終行を省略記号で締める。
    戻り値: (行のリスト, 全部入ったか)"""
    lines, cur = [], ""
    for ch in text:
        trial = cur + ch
        if f.getlength(trial) <= width_px or not cur:
            cur = trial
            continue
        # 行頭禁則: 次の行が禁則文字で始まるなら1文字分だけ持ち越す
        if ch in NO_LINE_START and len(cur) > 1:
            cur = trial
            continue
        lines.append(cur)
        cur = ch
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
        return lines, True
    # 溢れた分がある
    rest = text[sum(len(x) for x in lines):]
    if rest:
        last = lines[-1]
        while last and f.getlength(last + "…") > width_px:
            last = last[:-1]
        lines[-1] = last + "…"
        return lines, False
    return lines, True


def fit_title(text, width_px, g):
    """2行に収まる最大のフォントサイズを探す。収まらなければ最小サイズ＋省略。"""
    pt = g["title_pt_max"]
    while pt >= g["title_pt_min"]:
        f = font(pt)
        lines, ok = wrap(text, f, width_px, g["title_lines"])
        if ok:
            return pt, lines, True
        pt -= 0.5
    f = font(g["title_pt_min"])
    lines, ok = wrap(text, f, width_px, g["title_lines"])
    return g["title_pt_min"], lines, ok


def load_data():
    rows = list(csv.DictReader(
        open(os.path.join(ROOT, "tag_review", "out", "tag_review.csv"), encoding="utf-8")))
    soft = json.load(open(os.path.join(ROOT, "plan05", "results", "soft_tags.json"),
                          encoding="utf-8"))
    tags = soft["tags"]
    missing_colors = [t for t in tags if t not in TAG_COLORS]
    if missing_colors:
        raise ValueError(f"TAG_COLORS に未定義のタグがあります: {missing_colors}")
    tcol = {t: TAG_COLORS[t] for t in tags}
    thumbs = json.load(open(os.path.join(HERE, "cache", "thumbs.json"), encoding="utf-8"))
    cases = []
    for r in rows:
        cid = r["case_id"]
        parts = sorted(soft["ratios"][cid].items(), key=lambda x: -x[1])
        cases.append({
            "id": cid,
            "no": int(r["No"]),
            "title": r["ケース名"],
            "title_short": r["ケース名"].replace(ARCHIVE_SUFFIX, "").strip(),
            "status": "archived" if r["状況"] == "アーカイブ" else "active",
            "url": r["URL"],
            "tags": [{"t": k, "v": round(v, 3), "c": tcol[k]} for k, v in parts],
            "thumb_w": thumbs[cid]["w"],
        })
    return cases, tags, tcol


def draw_card(c, g, scale=1.0):
    """カード1枚を描いて (画像, 使った文字サイズ情報) を返す。
    scale は版下(1.0)とプレビューで共用するための倍率。"""
    S = int(round(mm(g["side"]) * scale))
    img = Image.new("RGB", (S, S), "white")
    d = ImageDraw.Draw(img)

    def P(v):        # mm → このカード内のpx
        return mm(v) * scale

    pad = P(g["pad"])
    inner_w = S - 2 * pad
    arch = c["status"] == "archived"

    # --- サムネイル ---
    ih = inner_w / g["img_ratio"]
    src = Image.open(os.path.join(THUMBS, f"{c['id']}.jpg")).convert("RGB")
    tw, th = int(round(inner_w)), int(round(ih))
    sr, dr = src.width / src.height, tw / th
    if sr > dr:      # 元が横長 → 左右を切る
        w2 = int(round(src.height * dr))
        src = src.crop(((src.width - w2) // 2, 0, (src.width + w2) // 2, src.height))
    else:            # 元が縦長 → 上下を切る
        h2 = int(round(src.width / dr))
        src = src.crop((0, (src.height - h2) // 2, src.width, (src.height + h2) // 2))
    src = src.resize((tw, th), Image.LANCZOS)
    if arch:         # アーカイブは彩度を落として進行中を目立たせる
        src = ImageEnhance.Color(src).enhance(0.45)
        src = ImageEnhance.Brightness(src).enhance(1.06)
    img.paste(src, (int(round(pad)), int(round(pad))))
    d.rectangle([pad, pad, pad + tw - 1, pad + th - 1], outline=(255, 255, 255), width=1)

    y = pad + th + P(g["gap_img_title"])

    # --- タイトル ---
    # 行分割と文字サイズの決定は必ず scale=1.0 の寸法で行う。
    # そうしないと版下とプレビューで改行位置がずれて、別物の絵になってしまう。
    full_inner = mm(g["side"] - 2 * g["pad"])
    pt, lines, fit = fit_title(c["title_short"], full_inner, g)
    f = ImageFont.truetype(JP_FONT, int(round(mm(pt * PT) * scale)))
    lead = mm(pt * PT * g["title_leading"]) * scale
    for ln in lines:
        d.text((pad, y), ln, font=f, fill=(26, 26, 26) if not arch else (90, 100, 110))
        y += lead
    y += (g["title_lines"] - len(lines)) * lead        # 1行のときも高さを揃える
    y += P(g["gap_title_bar"])

    # --- タグ割合バー ---
    bh = P(g["bar_h"])
    fb = ImageFont.truetype(JP_FONT, int(round(mm(g["bar_pt"] * PT) * scale)))
    x = pad
    total = sum(t["v"] for t in c["tags"]) or 1.0
    for i, t in enumerate(c["tags"]):
        w = inner_w * t["v"] / total
        if i == len(c["tags"]) - 1:
            w = pad + inner_w - x                       # 端数を最後で吸収
        col = hex2rgb(t["c"])
        d.rectangle([x, y, x + w, y + bh], fill=col)
        pctxt = f"{round(t['v'] * 100)}%"
        if w > fb.getlength(pctxt) * 1.7:
            fg = (255, 255, 255) if lum(col) < 150 else (30, 34, 40)
            bb = d.textbbox((0, 0), pctxt, font=fb)
            d.text((x + w / 2 - (bb[2] - bb[0]) / 2,
                    y + bh / 2 - (bb[3] + bb[1]) / 2), pctxt, font=fb, fill=fg)
        x += w
    y += bh + P(g["gap_bar_legend"])

    # --- 凡例 ---
    fl = ImageFont.truetype(JP_FONT, int(round(mm(g["legend_pt"] * PT) * scale)))
    leg = " ／ ".join(f"{t['t']} {round(t['v'] * 100)}%" for t in c["tags"])
    lg_lines, _ = wrap(leg, font(g["legend_pt"]), full_inner, 1)   # 分割は scale=1.0 基準
    d.text((pad, y), lg_lines[0] if lg_lines else "", font=fl, fill=(90, 98, 110))

    # --- 外枠。アーカイブはグレー、進行中は濃色 ---
    bw = max(1, int(round(mm(g["border_pt"] * PT) * scale)))
    d.rectangle([0, 0, S - 1, S - 1],
                outline=(176, 190, 197) if arch else (55, 71, 79), width=bw)

    return img, {"title_pt": pt, "title_lines": lines, "title_fit": fit}


def proof_sheet(cases, g, sizes_mm=(55, 75), out=None):
    """A4に実寸でカードを並べた試し刷り。手元で刷って読めるか確かめる用。"""
    W, H = int(round(mm(210))), int(round(mm(297)))
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(JP_FONT, int(round(mm(3.5))))
    margin = mm(12)
    y = margin
    d.text((margin, y), "CALL4 ケースカード 実寸テスト（A4を100%・等倍で印刷してください）",
           font=f, fill=(20, 20, 20))
    y += mm(8)
    picks = [cases[0], cases[len(cases) // 2], cases[-1]]
    for s in sizes_mm:
        d.text((margin, y), f"■ 一辺 {s}mm", font=f, fill=(20, 20, 20))
        y += mm(6)
        x = margin
        for c in picks:
            card, _ = draw_card(c, g, scale=s / g["side"])
            if x + card.width > W - margin:
                break
            sheet.paste(card, (int(round(x)), int(round(y))))
            x += card.width + mm(4)
        y += mm(s) + mm(9)
    if out:
        sheet.save(out, "PDF", resolution=DPI)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview-px", type=int, default=320,
                    help="エディタ埋め込み用プレビューの一辺（px）")
    ap.add_argument("--proof", action="store_true", help="A4実寸テストPDFも出す")
    ap.add_argument("--skip-master", action="store_true",
                    help="版下(高解像度)を作り直さない。エディタだけ更新したいとき")
    a = ap.parse_args()

    os.makedirs(os.path.join(OUT, "cards"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "cards_preview"), exist_ok=True)
    cases, tags, tcol = load_data()
    print(f"[s2] {len(cases)} 件 / 版下 {MASTER_MM}mm角 {int(mm(MASTER_MM))}px @ {DPI}dpi")

    geo_cases, shrunk, clipped = {}, [], []
    for i, c in enumerate(cases, 1):
        if not a.skip_master:
            master, info = draw_card(c, G, scale=1.0)
            master.save(os.path.join(OUT, "cards", f"{c['id']}.jpg"),
                        "JPEG", quality=95, optimize=True, dpi=(DPI, DPI))
        else:
            _, info = draw_card(c, G, scale=0.2)
        prev, _ = draw_card(c, G, scale=a.preview_px / mm(MASTER_MM))
        prev.save(os.path.join(OUT, "cards_preview", f"{c['id']}.webp"),
                  "WEBP", quality=82, method=5)
        geo_cases[c["id"]] = info
        if info["title_pt"] < G["title_pt_max"]:
            shrunk.append((c["id"], info["title_pt"]))
        if not info["title_fit"]:
            clipped.append(c["id"])
        if i % 20 == 0:
            print(f"  {i}/{len(cases)}")

    for c in cases:
        c["title_pt"] = geo_cases[c["id"]]["title_pt"]
        c["title_clipped"] = not geo_cases[c["id"]]["title_fit"]
        # 「このサムネイルが300dpiを満たす最大の配置サイズ(mm)」
        c["max_mm_300dpi"] = round(c["thumb_w"] / 300 * 25.4 /
                                  ((G["side"] - 2 * G["pad"]) / G["side"]), 1)

    json.dump({"cases": cases, "tags": tags, "tag_color": tcol,
               "master_mm": MASTER_MM, "dpi": DPI},
              open(os.path.join(OUT, "cards.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"layout_mm": G, "pt_mm": PT, "cases": geo_cases},
              open(os.path.join(OUT, "card_geometry.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    if a.proof:
        p = os.path.join(OUT, "proof_A4.pdf")
        proof_sheet(cases, G, out=p)
        print(f"[s2] 実寸テスト → {p}")

    print(f"[s2] 文字を縮めたカード {len(shrunk)} 件"
          f"（最小 {min([s[1] for s in shrunk], default=G['title_pt_max'])}pt）")
    if clipped:
        print(f"[s2] タイトルが2行に入りきらず省略したもの {len(clipped)} 件: "
              f"{', '.join(clipped[:8])}{' …' if len(clipped) > 8 else ''}")
    low = [c for c in cases if c["max_mm_300dpi"] < 75]
    if low:
        print(f"[s2] 注意: 75mm角に置くと300dpiを割るサムネイル {len(low)} 件。"
              f"最小は {min(c['max_mm_300dpi'] for c in low)}mm 相当")


if __name__ == "__main__":
    main()
