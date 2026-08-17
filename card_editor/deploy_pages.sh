#!/bin/bash
# card_editor/deploy_pages.sh — 配置エディタを GitHub Pages に上げ直す。
#
# gh-pages ブランチには index.html（＝エディタ）と、書き出し時だけ読む
# 高解像度カードJPEGを置く。
# リポジトリはプライベートだが、Pages で配信されるファイルは公開になるため、
# ここに置いたものだけが外から見える状態になる（他のファイルは巻き込まれない）。
#
# 作業ツリーもHEADも触らない。git の低レベルコマンドでコミットを直接組んで、
# それを push するだけなので、いま編集中のファイルに影響しない。
#
#   ./card_editor/deploy_pages.sh
#
# タグ確認のあとカードを作り直したら、s3 を回してからこれを実行すれば
# 同じURLのまま中身が入れ替わる。
set -euo pipefail

cd "$(dirname "$0")/.."
SRC=card_editor/out/editor.html
CARDS=card_editor/out/cards
BRANCH=gh-pages

[ -f "$SRC" ] || { echo "$SRC がありません。先に s3_build_editor.py を実行してください。"; exit 1; }
[ -d "$CARDS" ] || { echo "$CARDS がありません。先に s2_make_cards.py を実行してください。"; exit 1; }

# 検索に載せないための措置（権利の確認が済むまで）。HTML側にも noindex を入れてある。
ROBOTS=$(printf 'User-agent: *\nDisallow: /\n' | git hash-object -w --stdin)
INDEX=$(git hash-object -w "$SRC")
NOJEKYLL=$(printf '' | git hash-object -w --stdin)   # Jekyll の処理を通さない

CARDS_TREE=$(
  for f in "$CARDS"/*.jpg; do
    BLOB=$(git hash-object -w "$f")
    printf '100644 blob %s\t%s\n' "$BLOB" "$(basename "$f")"
  done | git mktree
)

TREE=$(printf '100644 blob %s\t.nojekyll\n040000 tree %s\tcards\n100644 blob %s\tindex.html\n100644 blob %s\trobots.txt\n' \
  "$NOJEKYLL" "$CARDS_TREE" "$INDEX" "$ROBOTS" | git mktree)

# 前回のデプロイがあれば親にして履歴を繋ぐ
if PARENT=$(git rev-parse --verify --quiet "refs/remotes/origin/$BRANCH"); then
  COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "deploy card_editor to Pages")
else
  COMMIT=$(git commit-tree "$TREE" -m "deploy card_editor to Pages (initial)")
fi

git push --force origin "$COMMIT:refs/heads/$BRANCH"
git fetch --quiet origin "$BRANCH"
echo "pushed $COMMIT -> $BRANCH"

REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
if ! gh api "repos/$REPO/pages" >/dev/null 2>&1; then
  echo "Pages を有効化します"
  gh api -X POST "repos/$REPO/pages" -f "source[branch]=$BRANCH" -f "source[path]=/"
else
  gh api -X POST "repos/$REPO/pages/builds" >/dev/null
  echo "再ビルドを要求しました"
fi
gh api "repos/$REPO/pages" --jq '"URL: \(.html_url)  状態: \(.status)"'
