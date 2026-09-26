#!/usr/bin/env python3
"""フェーズ1シートの「マッピング(スクショ)との齟齬確認」欄を機械可読にする。

この欄には、確認者が旧マップのスクリーンショットを見ながら書いた
**タグの大小関係**が入っている（例: `外国>手続` ＝ 外国にルーツを持つ人々の割合が
公正な手続より大きいはず）。フェーズ2で決めたい内容そのものなので、
読める形にして「再計算した割合がこの指摘と合っているか」を照合する。

書式は自由記述なので、**確実に読める形だけを制約として扱う**。
迷いが書かれているもの（`?` や「わかんない」）や記号が混在しているもの
（`環境>><手続`）は、こちらで解釈せずに原文のまま人へ返す。

さらに「ポスターカラー変更」列（TRUE/FALSE）を未決着フラグとして読む。
FALSE の12件はすべて迷いが書かれた行と一致しており、
「色（＝主タグ）を決めきれていない」という意味に取れる。

作成: 2026-08-20。
"""

import re
import unicodedata

NOTE_COL = "マッピング(スクショ)との齟齬確認"
DECIDED_COL = "ポスターカラー変更"

# 略記 → 正式なタグ名。長いものから照合するので、順序に意味がある
# （「情報公開」より先に「個人情報」を試さないと "個人情報" の "情報" を拾ってしまう）。
ALIASES = [
    ("個人情報・プライバシー", "個人情報・プライバシー"),
    ("ジェンダー・セクシュアリティ", "ジェンダー・セクシュアリティ"),
    ("外国にルーツを持つ人々", "外国にルーツを持つ人々"),
    ("政治参加・表現の自由", "政治参加・表現の自由"),
    ("医療・福祉・障がい", "医療・福祉・障がい"),
    ("プライバシー", "個人情報・プライバシー"),
    ("個人情報", "個人情報・プライバシー"),
    ("セクシュアリティ", "ジェンダー・セクシュアリティ"),
    ("セクシャリティ", "ジェンダー・セクシュアリティ"),
    ("ジェンダー", "ジェンダー・セクシュアリティ"),
    ("政治参加", "政治参加・表現の自由"),
    ("政治表現", "政治参加・表現の自由"),
    ("情報公開", "情報公開"),
    ("公正な手続", "公正な手続"),
    ("環境・災害", "環境・災害"),
    ("刑事司法", "刑事司法"),
    ("働き方", "働き方"),
    ("外国", "外国にルーツを持つ人々"),
    ("表現", "政治参加・表現の自由"),
    ("手続", "公正な手続"),
    ("医療", "医療・福祉・障がい"),
    ("福祉", "医療・福祉・障がい"),
    ("障がい", "医療・福祉・障がい"),
    ("障害", "医療・福祉・障がい"),
    ("環境", "環境・災害"),
    ("災害", "環境・災害"),
    ("刑事", "刑事司法"),
    ("情報", "情報公開"),
    ("沖縄", "沖縄"),
]

# 「決めきれていない」と読める語。これがあれば制約にはせず、原文を人に返す
DOUBT = ["?", "？", "わかんな", "気がする", "微妙", "どこ", "では", "な気も", "そう"]
PROVISIONAL = ["一旦"]


def _find_tags(text):
    """テキスト中のタグを出現順に拾う。一度使った範囲は再利用しない。"""
    spans = []
    for alias, canon in ALIASES:
        start = 0
        while True:
            i = text.find(alias, start)
            if i < 0:
                break
            if not any(s <= i < e or s < i + len(alias) <= e for s, e in
                       [(s, e) for s, e, _ in spans]):
                spans.append((i, i + len(alias), canon))
            start = i + 1
    spans.sort()
    out = []
    for _, _, canon in spans:
        if canon not in out:          # 同じタグの2回目は無視
            out.append(canon)
    return out


def parse_note(text):
    """1つの指摘を解釈する。

    返り値:
      status … "決定" / "暫定" / "未解釈"
      chain  … 大きい順に並べたタグ（比較として読めたときだけ2つ以上入る）
      max_tag… 「A>>」のように片側だけ書かれている場合の最大タグ
      text   … 原文（そのまま人に返す）
    """
    raw = (text or "").strip()
    if not raw:
        return {"status": "", "chain": [], "max_tag": None, "text": ""}
    s = unicodedata.normalize("NFKC", raw)

    has_gt, has_lt = ">" in s, "<" in s
    doubt = any(d in s for d in DOUBT)
    provisional = any(p in s for p in PROVISIONAL)
    tags = _find_tags(s)

    # 記号が混在しているものは解釈しない（"環境>><手続" がどちらの意味か決められない）
    if has_gt and has_lt:
        return {"status": "未解釈", "chain": [], "max_tag": None, "text": raw}

    # 「一旦A>B」のように暫定でも向きが1つなら、暫定として読む。
    # 迷いだけが書かれていて向きが無いものは解釈しない。
    if not (has_gt or has_lt):
        return {"status": "未解釈", "chain": [], "max_tag": None, "text": raw}

    if len(tags) == 1:
        # "プライバシー>>" のように「これが一番大きい」とだけ書かれている形
        return {"status": "暫定" if (doubt or provisional) else "決定",
                "chain": [], "max_tag": tags[0], "text": raw}
    if len(tags) < 2:
        return {"status": "未解釈", "chain": [], "max_tag": None, "text": raw}

    chain = tags if has_gt else list(reversed(tags))
    status = "暫定" if (doubt or provisional) else "決定"
    return {"status": status, "chain": chain, "max_tag": None, "text": raw}


TIE_PT = 2      # これ未満の差は「ほぼ同じ」として扱う（%ポイント）


def check(note, ratios):
    """指摘と、再計算した割合 {タグ: 割合} を照合する。

    返り値:
      "◯" … 指摘どおりの順序（差もはっきりある）
      "≒" … 順序は合っているが差が {TIE_PT}pt 未満、または向きが微妙
      "×" … 指摘と逆
      ""  … 照合できない（解釈できない指摘、または指摘されたタグが
             そのケースの確定タグに無い＝フェーズ1でタグ自体が変わった）

    差が1pt程度しかないものを「一致」と表示すると、
    実質は同率なのに確認済みのように見えてしまうので分けている。
    """
    if note["max_tag"]:
        if note["max_tag"] not in ratios:
            return ""
        top = max(ratios, key=lambda t: ratios[t])
        gap = (ratios[note["max_tag"]] - max(
            (v for t, v in ratios.items() if t != note["max_tag"]), default=0)) * 100
        if top != note["max_tag"]:
            return "×" if -gap >= TIE_PT else "≒"
        return "◯" if gap >= TIE_PT else "≒"

    chain = list(note["chain"])
    if len(chain) < 2 or any(t not in ratios for t in chain):
        return ""
    gaps = [(ratios[a] - ratios[b]) * 100 for a, b in zip(chain, chain[1:])]
    if any(g <= -TIE_PT for g in gaps):
        return "×"
    if all(g >= TIE_PT for g in gaps):
        return "◯"
    return "≒"


def load_notes(csv_path):
    """フェーズ1の記入済みCSVから、ケースごとの申し送りを読む。"""
    import csv
    out = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            cid = (r.get("case_id") or "").strip()
            if not cid:
                continue
            decided = (r.get(DECIDED_COL) or "").strip().upper()
            note = parse_note(r.get(NOTE_COL))
            # FALSE は「主タグを決めきれていない」と読む（12件すべて迷いのある行と一致）
            note["unsettled"] = (decided == "FALSE")
            note["comment"] = re.sub(r"\s+", " ", (r.get("コメント・理由") or "")).strip()
            out[cid] = note
    return out


if __name__ == "__main__":
    # 解釈が妥当かを目で確かめるための一覧表示
    import json
    import sys
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    notes = load_notes(HERE.parent / "tag_review" / "out" / "reviewed.csv")
    soft = json.loads((HERE / "out" / "soft_tags.json").read_text(encoding="utf-8"))
    ratios = soft["ratios"]

    counts = {"◯": 0, "≒": 0, "×": 0, "": 0}
    print(f"{'case_id':10}{'判定':5}{'状態':6}{'未決着':7}指摘 → 読み取り / 再計算")
    for cid, nt in notes.items():
        if not nt["text"]:
            continue
        v = check(nt, ratios.get(cid, {}))
        counts[v] += 1
        read = (" > ".join(nt["chain"]) if nt["chain"]
                else (f"{nt['max_tag']} が最大" if nt["max_tag"] else "—"))
        calc = " > ".join(f"{t}{int(round(p*100))}"
                          for t, p in sorted(ratios.get(cid, {}).items(),
                                             key=lambda kv: -kv[1]))
        print(f"{cid:10}{v or '—':5}{nt['status']:6}{'未決着' if nt['unsettled'] else '':7}"
              f"{nt['text'][:34]:36} → {read}  /  {calc}")
    print(f"\n一致 {counts['◯']} / ほぼ同率 {counts['≒']} / "
          f"不一致 {counts['×']} / 照合不能 {counts['']}")
