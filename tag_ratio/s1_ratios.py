#!/usr/bin/env python3
"""フェーズ2 Stage 1 — 確定タグでケース内タグ割合を再計算する（plan05 と同じ方法）。

方法は plan05/s0_softtag.py と同一（判別的プロトタイプ＋LOO＋softmax）。
変えたのは入力だけ:

  - タグ: CALL4の生タグ → フェーズ1で確定した修正後タグ
  - 母集団: 95件 → 88件（音信不通7件を除く。plan05 は95件で計算していた）

    88件に絞る理由: 除外した7件はフェーズ1の目視確認を受けていないので、
    確定タグと未確認タグが混ざったプロトタイプになる。最終成果物（カードマップ）の
    対象も88件なので、母集団を一致させた方が解釈も一貫する。
    副作用として稀なタグの陽性件数が減るため、件数と信頼度は出力に必ず残す。

入力:
  tag_ratio/out/cases_reviewed.json  … s0_cases.py の出力（確定タグ）
  plan02/features/emb.npz            … bge-m3 埋め込み（95×1024, L2正規化）
  plan02/features/case_index.json    … emb.npz の行順

出力:
  tag_ratio/out/soft_tags.json  … plan05/results/soft_tags.json と同じ構造
                                   （後段のカード生成がそのまま読める）

作成: 2026-08-20。
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "out"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    cases_path = OUTDIR / "cases_reviewed.json"
    if not cases_path.exists():
        raise SystemExit(f"{cases_path} がありません。先に s0_cases.py を実行してください。")
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = data["cases"]
    tags = data["tags"]

    emb_path = ROOT / "plan02" / "features" / "emb.npz"
    if not emb_path.exists():
        raise SystemExit(
            f"{emb_path} がありません（派生バイナリなのでGitには入っていません）。\n"
            "  plan02/s1_features.py を実行して bge-m3 埋め込みを作り直してください。")

    E_all = np.load(emb_path)["matrix"].astype(np.float64)
    E_all = E_all / np.linalg.norm(E_all, axis=1, keepdims=True)
    all_ids = json.loads(
        (ROOT / "plan02" / "features" / "case_index.json").read_text(encoding="utf-8"))
    if E_all.shape[0] != len(all_ids):
        raise SystemExit("emb.npz の行数と case_index.json が一致しません。")
    row = {cid: i for i, cid in enumerate(all_ids)}

    case_ids = [c["id"] for c in cases]
    missing = [cid for cid in case_ids if cid not in row]
    if missing:
        raise SystemExit(f"埋め込みが無いケース: {missing}")

    E = E_all[[row[cid] for cid in case_ids]]
    titles = [c["title"] for c in cases]
    N, T = len(case_ids), len(tags)

    M = np.zeros((N, T), dtype=int)
    for i, c in enumerate(cases):
        for t in c["tags"]:
            M[i, tags.index(t)] = 1
    if (M.sum(1) == 0).any():
        raise SystemExit("タグが0個のケースがあります。")

    # --- 判別スコア（LOO付き陽性プロトタイプ） ------------------------------
    disc = np.zeros((N, T))
    pos_only = np.zeros((N, T))
    reliable = []
    for t in range(T):
        pos = np.where(M[:, t] == 1)[0]
        neg = np.where(M[:, t] == 0)[0]
        n_pos = len(pos)
        reliable.append(n_pos >= 2)
        neg_proto = unit(E[neg].mean(0)) if len(neg) else np.zeros(E.shape[1])
        pos_sum = E[pos].sum(0)
        for i in range(N):
            if M[i, t] == 1 and n_pos >= 2:
                proto = unit((pos_sum - E[i]) / (n_pos - 1))      # leave-one-out
            elif M[i, t] == 1 and n_pos == 1:
                proto = unit(pos_sum)                              # 自己参照（低信頼）
            else:
                proto = unit(pos_sum / n_pos) if n_pos else np.zeros(E.shape[1])
            # numpy 2.0 + macOS Accelerate は正常な内積でも
            # divide-by-zero / overflow / invalid の警告を出すことがある
            # （plan05/s0_softtag.py でも同じ警告が出る。値は有限で正しい）。
            # 本物の異常を見落とさないよう、ループを抜けたところで有限性を検査する。
            with np.errstate(all="ignore"):
                pc = float(E[i] @ proto)
                nc = float(E[i] @ neg_proto)
            pos_only[i, t] = pc
            disc[i, t] = pc - nc

    if not np.isfinite(disc).all():
        raise SystemExit("判別スコアに有限でない値があります（埋め込みを確認してください）。")

    # --- 付与タグ内での割合（softmax） --------------------------------------
    assigned_disc = np.array([disc[i, t] for i in range(N) for t in range(T) if M[i, t]])
    tau = float(assigned_disc.std())
    ratios = {}
    for i in range(N):
        ts = [t for t in range(T) if M[i, t] == 1]
        d = np.array([disc[i, t] for t in ts])
        w = np.exp((d - d.max()) / tau)
        w = w / w.sum()
        ratios[case_ids[i]] = {tags[t]: round(float(w[k]), 3) for k, t in enumerate(ts)}

    # --- 貼り忘れ候補 -------------------------------------------------------
    missing_cand = []
    for t in range(T):
        pos = np.where(M[:, t] == 1)[0]
        if not reliable[t]:
            continue
        pos_disc = np.sort([disc[i, t] for i in pos])
        p25 = np.percentile(pos_disc, 25)
        for i in range(N):
            if M[i, t] == 0 and disc[i, t] >= p25:
                missing_cand.append({
                    "case_id": case_ids[i], "title": titles[i], "tag": tags[t],
                    "disc": round(float(disc[i, t]), 4),
                    "pct_vs_positives": round(float((pos_disc <= disc[i, t]).mean()), 2),
                    "n_pos": int(len(pos)),
                    "low_conf": len(pos) < 3,
                })
    missing_cand.sort(key=lambda d: -d["disc"])

    # --- 過剰タグ候補 -------------------------------------------------------
    over = []
    for t in range(T):
        pos = np.where(M[:, t] == 1)[0]
        neg = np.where(M[:, t] == 0)[0]
        if not reliable[t]:
            continue
        p50_neg = np.percentile([disc[i, t] for i in neg], 50)
        for i in pos:
            if disc[i, t] <= p50_neg:
                over.append({
                    "case_id": case_ids[i], "title": titles[i], "tag": tags[t],
                    "disc": round(float(disc[i, t]), 4),
                    "assigned_ratio": ratios[case_ids[i]].get(tags[t]),
                })
    over.sort(key=lambda d: d["disc"])

    # --- 検証: タグ別ROC-AUC と top-k 自己再現 ------------------------------
    auc = {}
    for t in range(T):
        if not reliable[t] or M[:, t].sum() in (0, N):
            auc[tags[t]] = None
            continue
        try:
            auc[tags[t]] = round(float(roc_auc_score(M[:, t], disc[:, t])), 3)
        except ValueError:
            auc[tags[t]] = None
    recov = {1: 0, 2: 0, 3: 0}
    for i in range(N):
        assigned = set(np.where(M[i] == 1)[0])
        order = list(np.argsort(-disc[i]))
        for k in recov:
            if assigned & set(order[:k]):
                recov[k] += 1
    recovery = {f"top{k}": round(recov[k] / N, 3) for k in recov}
    aucs = [v for v in auc.values() if v is not None]

    out = {
        "method": "discriminative prototype (pos-neg centroid) + LOO, softmax ratio",
        "input": {
            "tags": "tag_review/out/reviewed_tags.json（フェーズ1確定）",
            "embeddings": "plan02/features/emb.npz (bge-m3, L2正規化)",
            "population": f"{N}件（音信不通7件を除く）",
        },
        "tau": round(tau, 4),
        "tags": tags,
        "tag_counts": {tags[t]: int(M[:, t].sum()) for t in range(T)},
        "reliable_tags": {tags[t]: bool(reliable[t]) for t in range(T)},
        "disc_matrix": disc.round(4).tolist(),
        "case_ids": case_ids,
        "titles": titles,
        "ratios": ratios,
        "missing_candidates": missing_cand,
        "over_candidates": over,
        "validation": {
            "per_tag_auc": auc,
            "self_recovery": recovery,
            "mean_auc_reliable": round(float(np.mean(aucs)), 3) if aucs else None,
        },
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dest = OUTDIR / "soft_tags.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 旧版（plan05・生タグ・95件）との比較 -------------------------------
    old_path = ROOT / "plan05" / "results" / "soft_tags.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))["ratios"]
        deltas = []
        for cid, new_r in ratios.items():
            o = old.get(cid, {})
            shared = set(new_r) & set(o)
            if not shared:
                continue
            d = max(abs(new_r[t] - o[t]) for t in shared)
            # タグ自体が変わったケースがあるので、片側にしか無いタグは None で並べる
            deltas.append((d, cid, {t: (o.get(t), new_r.get(t))
                                    for t in sorted(set(new_r) | set(o))}))
        deltas.sort(reverse=True)
        print(f"[s1] 旧版（95件・生タグ）との最大割合差 上位5件:")
        for d, cid, det in deltas[:5]:
            body = " ".join(f"{t}:{'—' if a is None else a}→{'—' if b is None else b}"
                            for t, (a, b) in det.items())
            print(f"     Δ{d:.3f} {cid} {body}")

    print(f"[s1] N={N} tau={tau:.4f} 平均AUC={out['validation']['mean_auc_reliable']}")
    print("[s1] タグ別 件数 / AUC:")
    for t in tags:
        n = out["tag_counts"][t]
        a = auc[t]
        flag = "  ← 低信頼（陽性3件未満）" if n < 3 else ""
        print(f"     {n:3d}  AUC={'—' if a is None else f'{a:.3f}'}  {t}{flag}")
    print(f"[s1] 自己再現: {recovery}")
    print(f"[s1] 貼り忘れ候補 {len(missing_cand)}件 / 過剰タグ候補 {len(over)}件")
    print(f"[s1] 出力: {dest}")


if __name__ == "__main__":
    main()
