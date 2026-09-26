# CALL4 ケースマップ

CALL4（[call4.jp](https://www.call4.jp/)）に掲載されている公共訴訟ケースを分類・可視化し、
最終的に**1600mm幅ロール紙1枚のケースマップ（紙の印刷物）**にするプロジェクト。

いま生きているのは **`tag_ratio/`（データの確定）→ `card_editor/`（配置と入稿データ）** の2つ。
`plan01`〜`plan05` は「どう分類するか」を決めるまでの調査で、**結論が出て役目を終えた検討記録**。
消さずに残してあるのは、いまの方針を選んだ理由がそこにしか書かれていないため。

**配置エディタ（公開中）**: https://shio-koji.github.io/case_clustering/
**確定データのレポート**: `tag_ratio/report/call4_tag_ratio_report.html`

---

## いまのデータの流れ

```
CALL4のAPI ──→ cache/                    生データ（95件）。全ての出発点
                  │
                  ├→ plan02/features/     形態素・語彙・埋め込み（他トラックが共用）
                  │
tag_review/ ──────┤  フェーズ1：タグを人が目視確認（10件変更）
                  ↓
tag_ratio/  ─────── フェーズ2：割合を再計算 → 順番を人が確認（10件変更）
                  ↓
              soft_tags_final.json        ★確定データ。これが唯一の正
                  ↓
card_editor/ ────── カード画像88枚 + タグとの2部グラフ → 配置エディタ → 入稿データ
                  ↓
            GitHub Pages（配置作業用のGUI）
```

**対象は88件**。CALL4掲載の95件から、音信不通で確認が取れなかった7件を除いてある。
アーカイブ38件・進行中50件。タグは11種類、1ケースにつき原則3個まで。

---

## ディレクトリの趣旨

### 現行（これを見ればよい）

| | 趣旨 | 主な成果物 | 状態 |
|---|---|---|---|
| `cache/` | CALL4から取得した**生データ**。ケース一覧・本文・更新情報・整形済みコーパス。全ての工程の出発点で、ここだけは取り直さない限り不変 | `cases_list.json`（95件）、`case_<id>.json`、`corpus_clean.json` | 2026-07 取得 |
| `tag_review/` | **フェーズ1：タグそのものを人が目視確認**する一式。Googleスプレッドシートを作る側のスクリプトと、記入結果を読み戻す検証 | `out/tag_review.csv`（配布したシート）、`out/reviewed_tags.json`（確定タグ） | 完了 |
| `tag_ratio/` | **フェーズ2：ケース内のタグ割合を確定**させる一式。確定タグで割合を再計算し、「どのタグが主役か」の順番を人に選んでもらって確定する | `out/soft_tags_final.json`（★確定データ）、`report/call4_tag_ratio_report.html` | 完了 |
| `card_editor/` | 紙のケースマップを作るための**カード画像生成と配置エディタ**。出力は「どのケースをどこに何mmで置くか」の座標と、Illustrator/Photoshop向けの入稿データ | `out/cards/*.jpg`（版下）、`out/editor.html`（GUI）、`out/proof_A4.pdf` | 進行中 |
| `canva_check/` | Canvaで作っている**1ケース1ページのパネル**（7周年展示用）の色照合。ページ上下のラインの色が確定した主要タグの色と合っているかを、書き出したPNGから自動で確かめる | `out/report.tsv` | 2026-08-21 照合済み |
| `doc/` | **人に説明するための文書**。手法を専門知識なしで読める形に書き直したもの。数値は確定データと突き合わせてある | `タグ割合の説明.md` | 現行 |

### 検討記録（結論が出て役目を終えたもの。数値は当時の入力に基づく）

| | 趣旨 | いまの位置づけ |
|---|---|---|
| `methodology_review.md` | 分類手法の**棚卸し**。「どんな方法があり、何が向いているか」を選択前に網羅的に整理した文書。以降の全計画の土台 | 方針決定の根拠として有効 |
| `plan01.md` | **プロジェクト方針の文書**。目的・進め方・成果物の要件（単一自己完結HTML・日英併記など）を定めたブリーフ。以降の計画はこれを前提にしている | 方針として現役 |
| `plan01/` | 最初の探索。文字n-gramのTF-IDF＋SVD、埋め込み、UMAP、k-means/階層/HDBSCAN で**ひとまず分類してみた**段階。plan02 の開始時に**凍結**された | 探索の記録。数値は使わない |
| `plan02.md` + `plan02/` | **本命トラック**。「特徴表現→次元圧縮→構造発見→解釈→評価」を段階ごとにレビューを挟んで実行。テキスト（語彙TF-IDFと意味bge-m3）の2表現×5手法を比較し、**6グループのLeidenクラスタリング**に到達。plan01の埋め込みは再利用せず作り直している（plan01のテキストは更新情報を含んでいて、表現間の入力条件が揃わないため） | 分類手法の結論。`features/` は現行工程が共用 |
| `plan03.md` + `plan03/` | **タグだけ**を特徴量にした対照トラック。テキストを使わず既存タグのみで分類し、plan02の結果と突き合わせた（ARI 0.44で同じ6テーマに到達） | 妥当性の裏付け |
| `plan04/` | plan02のクラスタ数**Kの決め方**を詰めた追試（NMFのKスイープ・指標比較）。計画文書は無く、plan02の続きとして走らせた | Kの選定根拠 |
| `plan05.md` + `plan05/` | **ソフトタグ・トラック**。「1ケースの中で各タグがどれだけ中心的か」の割合と、タグの貼り忘れ／過剰の候補を出した。**いまのタグ割合の手法そのもの** | 手法の出所。`results/soft_tags.json` は旧値 |
| `plan_final_report/` | plan01〜05を**1本にまとめた報告書**。分類の全体像を外部に説明するための単一HTML | 経緯の説明用。数値は旧タグ時点 |
| `mcp-bridge/` | CALL4のAPIをClaude Codeから叩くためのMCPサーバ（データ取得時に使用） | 再取得するとき用 |
| `01_fetch_data.py` `02_build_corpus.py` | `cache/` を作るスクリプト。取得と整形 | 再取得するとき用 |

---

## 旧データを読むときの注意

**`plan05/results/soft_tags.json` と `plan_final_report/` の割合は、いまの確定値とは違います。**
それぞれ何を入力にした数字なのかを整理しておく。

| データ | タグ | 母集団 | 位置づけ |
|---|---|---|---|
| `plan05/results/soft_tags.json` | CALL4の**生タグ**（人の確認前） | 95件 | plan05当時の値。人が見ていた「旧マップ」の元データで、`tag_ratio` が「色が変わったか」を判定する基準として今も参照する |
| `tag_ratio/out/soft_tags.json` | フェーズ1の**確定タグ** | 88件 | 確定タグで再計算した値。人の順番判断は入っていない（＝機械の推定そのまま） |
| `tag_ratio/out/soft_tags_final.json` | フェーズ1の**確定タグ** | 88件 | **★これが正。** 上の計算値に、フェーズ2で人が決めた順番を反映したもの |

3つとも構造は同じなので取り違えやすい。**現行の工程は必ず `soft_tags_final.json` を使う**
（`card_editor/s2_make_cards.py` は新しい順に自動で探し、どれを使ったかを標準出力に出す）。

`plan01`〜`plan05` のレポートHTMLは、**それぞれの当時の入力で内部的に一貫している**ので、
数字を今の値に書き換えることはしていない。「いつ・何を入力に出した結論か」を上の表で確認して読むこと。

**ファイルは動かしていない。** 旧データを `archive/` などへ移すと、
`card_editor` の参照先や `plan02/features/` の共用が壊れ、再現できなくなる。
整理は「場所を変える」ではなく「趣旨を書く」方針にしてある（このファイルがそれ）。

---

## 再現のしかた

```bash
# 0. 生データ（取り直す必要がなければ不要）
.venv/bin/python 01_fetch_data.py
.venv/bin/python 02_build_corpus.py

# 1. 共用の特徴量（埋め込み。bge-m3のDLに約2.3GB・CPUで約9分）
.venv/bin/python plan02/s0_tokenize.py     # 形態素（SudachiPyが必要）
.venv/bin/python plan02/s1_features.py     # tags/tfidf/count/emb を作る

# 2. フェーズ1：確定タグ（記入済みシートを out/reviewed.csv に置いてから）
.venv/bin/python tag_review/ingest_reviewed.py

# 3. フェーズ2：割合の確定
.venv/bin/python tag_ratio/s0_cases.py            # 生データ＋確定タグ → ケース情報
.venv/bin/python tag_ratio/s1_ratios.py           # 割合を再計算
.venv/bin/python tag_ratio/s2_make_ratio_sheet.py # 確認シート用CSV
.venv/bin/python tag_ratio/s3_ingest_ratios.py    # 記入結果を読み戻して確定
.venv/bin/python tag_ratio/s4_report.py           # レポート

# 4. カードと配置エディタ
.venv/bin/python card_editor/s1_fetch_thumbs.py   # サムネイル（初回のみ）
.venv/bin/python card_editor/s2_make_cards.py --proof
.venv/bin/python card_editor/s3_build_editor.py
./card_editor/deploy_pages.sh                     # GitHub Pages を更新
```

各ディレクトリの詳細な手順と設計判断は `tag_review/README.md`・`tag_ratio/README.md`・
`card_editor/README.md` にある。

### 派生バイナリはGitに入れていない

埋め込み（`plan02/features/emb.npz` 等）と pickle は大きく、再生成できるので追跡外。
消えていても `plan02/s1_features.py` で作り直せる（テキスト構成が同一なので**同じ値が再現する**。
実測で割合の差 0.000、AUC完全一致）。

### 記入済みシートもGitに入れていない

確認者名とコメントが入るため。このリポジトリは GitHub Pages のために公開してある
（`tag_review/out/reviewed.csv`、`tag_ratio/out/reviewed_ratios.csv`、
`cases_reviewed.json`、`ratio_review.csv`、`soft_tags_final.json`）。
**確定データそのものが追跡外なので、手元から消さないよう注意。**

---

## 公開している範囲

`gh-pages` ブランチに置いた**配置エディタと高解像度カード画像だけ**が外から見える。
無料プランではプライベートリポジトリで Pages が使えないため、このリポジトリ自体も公開にしてある。
検索インデックスは抑止してある（HTMLに `noindex`、Pagesに `robots.txt`）が、
**URLを知れば誰でも開ける**。CALL4掲載のサムネイルを含む図なので、
公開物として扱うにはCALL4の了解が前提。

---

## 残っている課題

紙にするまでの未解決点は `card_editor/README.md` に、
データ側の限界は `tag_ratio/report/call4_tag_ratio_report.html` の「限界」に書いてある。要点だけ:

- **CMYK色校正** — タグ11色は画面用RGB。印刷では色の差が縮む
- **実寸の試し刷り** — `card_editor/out/proof_A4.pdf` をA4等倍で刷ってカードサイズを決める
- **サムネイル解像度** — 75mm角に置くと7件が300dpiを割る
- **画像の権利** — CALL4の了解
- **確認者名・CALL4側の最終判断** — シートの該当列が空のまま
