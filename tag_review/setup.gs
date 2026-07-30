/**
 * CALL4 タグ確認シート セットアップ（フェーズ1用）
 *
 * 使い方:
 *   1. 新規スプレッドシートを作成
 *   2. ファイル > インポート で tag_review/out/tag_review.csv を取り込む
 *      （インポート場所は「現在のシートを置換する」、区切り文字は「カンマ」）
 *   3. 拡張機能 > Apps Script を開き、このファイルの中身を全部貼り付けて保存
 *   4. 関数 setup を選んで実行（初回のみ権限の承認画面が出ます）
 *   5. 最後にドロップダウンの設定を手で仕上げる（下記）
 *
 * ■ 手作業が必要な2点（どちらも同じダイアログの中）
 * 「複数の選択肢を許可」とチップ色は、Apps Script / Sheets API v4 では設定できません
 * （DataValidationBuilder に該当メソッドが無く、API v4 の DataValidationRule も
 * condition / inputMessage / strict / showCustomUi の4フィールドだけ）。
 *
 *   「タグ確認」シートのG2を選択 → データ > データの入力規則 → 出てきたルールをクリック
 *     (a) 詳細オプション → 表示スタイル を「チップ」にする
 *         （Apps Script から作ると「矢印」になることがあり、
 *           「複数の選択肢を許可」はチップ形式でしか出ません）
 *     (b) 「複数の選択肢を許可」にチェック
 *     (c) 各タグ名の左の丸をクリック → Customize に「タグ一覧」シートB列の色を入れる
 *   → 完了
 *
 * タグ列(G/H)はセルを塗っていません。色はチップだけが持つようにしてあります。
 *
 * 現行タグ列(G)と修正後タグ列(H)を隣接させて同一ルールを共有させてあるので、
 * この設定1回で両方に効きます。
 *
 * ※ (a) を有効にするまでは、複数タグが入っているセルが「無効」扱いで
 *    フラグが付きます。チェックを入れると消えます。
 *
 * ★ setup を再実行して検証ルールを作り直すと上記の設定が消えるため、
 *   addValidation_ は既に同じルールがある場合は何もしません。
 *
 * 再実行しても安全（冪等）に作ってあります。
 */

// 既存タグ。make_tag_review.py の TAGS と順序まで一致させること
// （「タグ一覧」シートA列の並びがそのままドロップダウンの並びになる）。
var TAGS = [
  '公正な手続',
  '政治参加・表現の自由',
  '外国にルーツを持つ人々',
  '刑事司法',
  'ジェンダー・セクシュアリティ',
  '環境・災害',
  '働き方',
  '医療・福祉・障がい',
  '情報公開',
  '沖縄',
  '個人情報・プライバシー'
];

/** チップ色の推奨値。dataviz スキル同梱の ΔE / CVD 実装で探索・実測した11色。
 *  全ペア最小ΔE は通常視 10.9（色相等間隔に置くと 5.5 しか出ない）。
 *  ただし二色覚下では最小 0.6 まで落ちるので、色だけでは識別できない。
 *  セルには必ずタグ名が入るため、識別は文字が担保する前提の配色。
 *  付与件数の多いタグに淡い色、稀なタグに濃い色を割り当てて、
 *  88行に敷いても画面が騒がしくならないようにしている。 */
var TAG_COLOR = {
  '公正な手続': '#daecfd',
  '政治参加・表現の自由': '#fce584',
  '外国にルーツを持つ人々': '#ffc4fe',
  '刑事司法': '#a4f6b5',
  'ジェンダー・セクシュアリティ': '#68f2ff',
  '環境・災害': '#fdae73',
  '働き方': '#c1b0fe',
  '医療・福祉・障がい': '#5bd7b8',
  '情報公開': '#b8c86a',
  '沖縄': '#66caff',
  '個人情報・プライバシー': '#fd9cba'
};

var MAIN = 'タグ確認';
var TAGLIST = 'タグ一覧';
var RULES = 'ルール';
var MAX_TAGS = 3;   // 原則の上限。4個以上は例外として許容するが目立たせる

// 列番号（tag_review.csv の並びと一致させること）
// 現行タグ(G)と修正後タグ(H)を隣接させ、1つの検証ルールを共有させている
var C = {
  no: 1, caseId: 2, status: 3, memo: 4, title: 5, desc: 6,
  current: 7, revised: 8,
  changed: 9, comment: 10, reviewer: 11, decision: 12,
  url: 13   // リンク埋め込み後に削除する作業列
};

var COLOR = {
  header: '#37474f',
  readonly: '#f5f5f5',
  editable: '#fffde7',
  changed: '#fff3cd',
  error: '#f8d7da',
  warn: '#ffe0b2',
  band: '#eceff1'
};


function setup() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = findMainSheet_(ss);
  var n = sh.getLastRow() - 1;
  if (n < 1) throw new Error('データ行がありません。先にCSVをインポートしてください。');

  sh.setName(MAIN);
  buildTagListSheet_(ss, n);
  buildRulesSheet_(ss);

  linkTitles_(sh, n);      // ケース名をリンク化し、URL作業列を削除
  layout_(sh, n);
  var validationAdded = addValidation_(ss, sh, n);
  addChangedFormula_(sh, n);
  addConditionalFormats_(sh, n);
  protectReadOnly_(sh, n);

  ss.setActiveSheet(sh);
  sh.getRange(2, C.revised).activate();
  SpreadsheetApp.getUi().alert(
      'セットアップ完了：' + n + '件\n\n' +
      (validationAdded
        ? 'あと1回だけ手作業をお願いします。\n\n'
          + 'G2を選択 → データ > データの入力規則 → ルールをクリック\n'
          + '  (a) 詳細オプション → 表示スタイル を「チップ」に\n'
          + '  (b) 「複数の選択肢を許可」にチェック\n'
          + '  (c) 各タグ左の丸 → Customize に色を貼り付け\n'
          + '     （色は「タグ一覧」シートB列に並べてあります）\n\n'
          + '現行タグ列と修正後タグ列は同じルールなので、1回で両方に効きます。\n'
          + '(a) を入れるまで複数タグのセルに無効フラグが付きますが、'
          + 'チェックを入れれば消えます。'
        : '既存の入力規則を検出したので、そのまま残しました'
          + '（複数選択の設定とチップ色は保持されます）。'));
}


/** 見出しセルの比較用。CSVインポート時に先頭へ入り込むBOMや、
 *  ゼロ幅スペース・ノーブレークスペースを落としてから比較する。 */
function normHeader_(v) {
  return String(v === null || v === undefined ? '' : v)
      .replace(/[\uFEFF\u200B\u200C\u200D\u00A0]/g, '')
      .trim();
}


/** B1が「case_id」のシートを本体とみなす。「case_id」は他に出てこない見出しなので
 *  これ1つで十分に特定できる。見つからない場合は各シートの1行目を添えて投げる。 */
function findMainSheet_(ss) {
  var sheets = ss.getSheets();
  var seen = [];
  for (var i = 0; i < sheets.length; i++) {
    var sh = sheets[i];
    var raw1 = sh.getRange(1, 1).getValue();
    var a1 = normHeader_(raw1);
    var b1 = normHeader_(sh.getRange(1, 2).getValue());
    seen.push('「' + sh.getName() + '」A1=' + JSON.stringify(String(raw1)) +
              ' B1=' + JSON.stringify(String(sh.getRange(1, 2).getValue())));
    if (b1 === 'case_id') {
      if (a1 !== String(raw1)) sh.getRange(1, 1).setValue(a1);   // BOM等を除去して書き戻す
      return sh;
    }
  }
  throw new Error('B1が「case_id」のシートが見つかりません。' +
                  '取り込むファイルは tag_review/out/tag_review.csv です' +
                  '（tag_review/shiina_list.tsv は入力元なので違います）。' +
                  ' 各シートの1行目: ' + seen.join(' / '));
}


/** ケース名セルにCALL4へのリンクを埋め込み、末尾のURL作業列を削除する。
 *  HYPERLINK数式ではなくリッチテキストを使うのは、あとでCSVに書き出し直したときに
 *  ケース名がプレーンテキストのまま残り、再取り込みが壊れないようにするため。 */
function linkTitles_(sh, n) {
  if (sh.getLastColumn() < C.url) return;   // 既に実行済み
  var urls = sh.getRange(2, C.url, n, 1).getValues();
  var titles = sh.getRange(2, C.title, n, 1).getValues();
  var rich = [];
  for (var i = 0; i < n; i++) {
    var b = SpreadsheetApp.newRichTextValue().setText(String(titles[i][0]));
    if (urls[i][0]) b.setLinkUrl(String(urls[i][0]));
    rich.push([b.build()]);
  }
  sh.getRange(2, C.title, n, 1).setRichTextValues(rich);
  sh.deleteColumn(C.url);
}


function layout_(sh, n) {
  var lastCol = sh.getLastColumn();

  sh.getRange(1, 1, 1, lastCol)
    .setBackground(COLOR.header).setFontColor('#ffffff').setFontWeight('bold')
    .setVerticalAlignment('middle').setWrap(true);
  sh.setRowHeight(1, 38);

  // 「概要」「現行タグ」「修正後タグ」が同一画面に収まる幅配分
  var widths = {};
  widths[C.no] = 40;
  widths[C.caseId] = 80;
  widths[C.status] = 60;
  widths[C.memo] = 110;
  widths[C.title] = 240;
  widths[C.desc] = 380;
  widths[C.current] = 210;
  widths[C.revised] = 210;
  widths[C.changed] = 60;
  widths[C.comment] = 260;
  widths[C.reviewer] = 78;
  widths[C.decision] = 110;
  for (var col in widths) sh.setColumnWidth(Number(col), widths[col]);

  // 読み取り専用ブロックと記入ブロックを色で区別する。
  // ただしタグ列(G/H)は塗らない —— セルを塗るとチップの色が埋もれてしまうため、
  // ここは白地のままにして、色はチップだけが持つようにする。
  sh.getRange(2, 1, n, C.desc).setBackground(COLOR.readonly);
  sh.getRange(2, C.changed, n, 1).setBackground(COLOR.readonly);
  sh.getRange(2, C.current, n, 2).setBackground(null);
  sh.getRange(2, C.comment, n, C.decision - C.comment + 1).setBackground(COLOR.editable);

  // 記入欄であることは塗りではなく枠線で示す
  sh.getRange(1, C.revised, n + 1, 1)
    .setBorder(true, true, true, true, false, false, '#f9a825',
               SpreadsheetApp.BorderStyle.SOLID_MEDIUM);

  sh.getRange(2, 1, n, lastCol).setVerticalAlignment('top');
  [C.memo, C.title, C.desc, C.current, C.revised, C.comment].forEach(function (c) {
    sh.getRange(2, c, n, 1).setWrap(true);
  });
  [C.no, C.status, C.changed].forEach(function (c) {
    sh.getRange(2, c, n, 1).setHorizontalAlignment('center');
  });
  sh.getRange(2, C.desc, n, 1).setFontSize(9).setFontColor('#455a64');

  sh.setFrozenRows(1);
  sh.setFrozenColumns(C.title);   // No〜ケース名 を固定
  sh.autoResizeRows(2, n);        // 概要とチップの折り返しに合わせて行の高さを詰める

  var existing = sh.getFilter();  // 再実行時は貼り直す
  if (existing) existing.remove();
  sh.getRange(1, 1, n + 1, lastCol).createFilter();
}


/** 現行タグ列＋修正後タグ列にまとめて同一のドロップダウンを設定する。
 *  1つのルールを共有させることで、手作業（複数選択の許可とチップ色）が
 *  1度で両方に効く。既に同じ範囲参照のルールがある場合は、
 *  設定済みの内容を壊さないよう何もしない。
 *  戻り値: 新しくルールを設定したら true。 */
function addValidation_(ss, sh, n) {
  var tagRange = ss.getSheetByName(TAGLIST).getRange(2, 1, TAGS.length, 1);
  var target = sh.getRange(2, C.current, n, C.revised - C.current + 1);

  var current = sh.getRange(2, C.current).getDataValidation();
  if (current &&
      current.getCriteriaType() === SpreadsheetApp.DataValidationCriteria.VALUE_IN_RANGE) {
    var vals = current.getCriteriaValues();
    if (vals && vals[0] &&
        vals[0].getSheet().getName() === TAGLIST &&
        vals[0].getA1Notation() === tagRange.getA1Notation()) {
      return false;   // 既に同じルール → 複数選択設定とチップ色を保持するため触らない
    }
  }

  // setAllowInvalid(true) にしているのは、「複数の選択肢を許可」を手で有効にするまでの間、
  // 複数タグが入ったセルが弾かれないようにするため。有効化後は複数値が正当になる。
  var rule = SpreadsheetApp.newDataValidation()
      .requireValueInRange(tagRange, true)
      .setAllowInvalid(true)
      .setHelpText('既存の' + TAGS.length + 'タグから選んでください（原則' + MAX_TAGS + '個まで）')
      .build();
  target.clearDataValidations();
  target.setDataValidation(rule);
  return true;
}


/** 現行タグと修正後タグを、どちらも「タグ一覧」の並び順で組み直してから比較する。
 *  マルチセレクトのセルは「タグA, タグB」という1つの文字列なので、
 *  各タグが含まれるかを SEARCH で判定して正規化する。
 *  （どのタグも他タグの部分文字列でないことを生成側で保証済み）
 *  入力した順番に依存せず、集合として同じなら「変更なし」と判定できる。 */
function addChangedFormula_(sh, n) {
  var master = "'" + TAGLIST + "'!$A$2:$A$" + (TAGS.length + 1);
  function key(col, r) {
    return 'TEXTJOIN("|",1,ARRAYFORMULA(IF(ISNUMBER(SEARCH(' + master + ',$' + col + r +
           ')),' + master + ',"")))';
  }
  var f = [];
  for (var i = 0; i < n; i++) {
    var r = i + 2;
    f.push(['=IFERROR(IF(' + key('G', r) + '=' + key('H', r) + ',"","変更"),"要確認")']);
  }
  sh.getRange(2, C.changed, n, 1).setFormulas(f);
}


/** 1セルに入っているタグ数を数える数式（文字列として組み立てて使う）。 */
function tagCountExpr_(col, r) {
  var master = "'" + TAGLIST + "'!$A$2:$A$" + (TAGS.length + 1);
  return 'SUMPRODUCT(--ISNUMBER(SEARCH(' + master + ',$' + col + r + ')))';
}


function addConditionalFormats_(sh, n) {
  var revised = sh.getRange(2, C.revised, n, 1);
  var rules = [];

  // 1) タグが1つも入っていない（全ケース最低1タグ必要）
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$H2=""')
      .setBackground(COLOR.error)
      .setRanges([revised]).build());

  // 2) 原則上限を超えている（例外として許容するが目立たせる）。
  //    タグ列に背景色を敷くとチップの色が読めなくなるので、目印はNo列に出す。
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=' + tagCountExpr_('H', 2) + '>' + MAX_TAGS)
      .setBackground(COLOR.warn)
      .setRanges([sh.getRange(2, C.no, n, 1)]).build());

  // 3) 変更あり
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$I2="変更"')
      .setBackground(COLOR.changed).setBold(true)
      .setRanges([sh.getRange(2, C.changed, n, 1)]).build());

  // 4) アーカイブ行は色を落として進行中ケースを目立たせる
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$C2="アーカイブ"')
      .setFontColor('#90a4ae')
      .setRanges([sh.getRange(2, C.status, n, 1), sh.getRange(2, C.title, n, 1)]).build());

  sh.setConditionalFormatRules(rules);
}


/** 自動生成列は警告付き保護。完全ロックにしないのは、
 *  気づいた誤りをその場で直せる余地を残すため（編集時に警告は出る）。 */
function protectReadOnly_(sh, n) {
  sh.getProtections(SpreadsheetApp.ProtectionType.RANGE).forEach(function (p) {
    if (p.getDescription().indexOf('自動生成') === 0) p.remove();
  });
  [[1, 1, n + 1, C.current], [1, C.changed, n + 1, 1]].forEach(function (a) {
    sh.getRange(a[0], a[1], a[2], a[3]).protect()
      .setDescription('自動生成データ（原則編集不可）').setWarningOnly(true);
  });
}


function buildTagListSheet_(ss, n) {
  var sh = ss.getSheetByName(TAGLIST) || ss.insertSheet(TAGLIST);
  sh.clear();
  sh.setConditionalFormatRules([]);

  var last = n + 1;
  var rows = [['タグ名', 'チップ色', '現行件数', '修正後件数', '増減', '備考']];
  for (var i = 0; i < TAGS.length; i++) {
    var r = i + 2;
    // マルチセレクトの1セルに複数タグが入るのでワイルドカードで数える
    // （どのタグも他タグの部分文字列でないことを生成側で保証している）
    rows.push([
      TAGS[i],
      TAG_COLOR[TAGS[i]],
      "=COUNTIF('" + MAIN + "'!$G$2:$G$" + last + ',"*"&$A' + r + '&"*")',
      "=COUNTIF('" + MAIN + "'!$H$2:$H$" + last + ',"*"&$A' + r + '&"*")',
      '=$D' + r + '-$C' + r,
      ''
    ]);
  }
  var tr = TAGS.length + 2;
  rows.push(['合計（タグ付与数）', '',
             '=SUM($C$2:$C$' + (tr - 1) + ')',
             '=SUM($D$2:$D$' + (tr - 1) + ')',
             '=$D' + tr + '-$C' + tr, '']);

  sh.getRange(1, 1, rows.length, 6).setValues(rows);
  sh.getRange(1, 1, 1, 6)
    .setBackground(COLOR.header).setFontColor('#ffffff').setFontWeight('bold');
  sh.getRange(tr, 1, 1, 6).setFontWeight('bold').setBackground(COLOR.band);
  sh.getRange(2, 5, TAGS.length + 1, 1).setNumberFormat('"+"0;"△"0;0');

  // B列は色見本。手作業でチップ色を入れるときにここから色をコピーする
  for (var j = 0; j < TAGS.length; j++) {
    sh.getRange(j + 2, 2)
      .setBackground(TAG_COLOR[TAGS[j]])
      .setHorizontalAlignment('center')
      .setFontSize(9);
  }

  sh.setColumnWidth(1, 230);
  sh.setColumnWidth(2, 90);
  sh.setColumnWidths(3, 3, 95);
  sh.setColumnWidth(6, 320);
  sh.setFrozenRows(1);

  // 件数が少ないタグはフェーズ2の割合計算が成立しないので目立たせる。
  // 範囲からB列（色見本）を外して、見本の色が上書きされないようにする。
  sh.setConditionalFormatRules([
    SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=AND($A2<>"",$D2<3)')
      .setBackground(COLOR.error)
      .setRanges([sh.getRange(2, 3, TAGS.length, 4)]).build()
  ]);

  sh.getRange(tr + 2, 1).setValue(
      'B列はチップ色の推奨値です。「タグ確認」シートのG2を選択 → データ > データの入力規則 → '
      + 'ルールをクリック → (a) 詳細オプションで表示スタイルを「チップ」に → '
      + '(b)「複数の選択肢を許可」にチェック → '
      + '(c) 各タグ左の丸 → Customize に貼り付け。'
      + '現行タグ列と修正後タグ列は同じルールなので、1回で両方に効きます。');
  sh.getRange(tr + 3, 1).setValue(
      '修正後件数が3件未満のタグは赤くなります（次工程のタグ割合計算が成立しないため）。')
    .setFontColor('#b71c1c');
  sh.getRange(tr + 2, 1, 2, 1).setWrap(true);
}


function buildRulesSheet_(ss) {
  var sh = ss.getSheetByName(RULES) || ss.insertSheet(RULES);
  sh.clear();
  var lines = [
    ['CALL4 ケースタグ 目視確認のお願い（フェーズ1）'],
    [''],
    ['やっていただきたいこと'],
    ['H列「修正後タグ」を、そのケースに付けるべきタグの最終形にしてください。'],
    ['初期値には現行タグが入っています。過不足がなければ触らなくて大丈夫です。'],
    ['セルをクリックするとタグの一覧が出ます。チェックで足す・外すができます。'],
    ['迷った点や判断の理由は「コメント・理由」列に自由に書いてください。'],
    ['担当された方のお名前を「確認者」列に入れていただけると助かります。'],
    [''],
    ['ルール'],
    ['1ケースにつけるタグは原則3個までです（11タグから選択）。'],
    ['4個以上にすると、そのセルがオレンジになります。禁止ではなく目印です。'],
    ['「アーカイブ」はこの個数には数えません。C列「状況」で表しているので選択不要です。'],
    ['最低1個はタグが必要です。空になっているセルは赤くなります。'],
    ['最終的な付与判断はCALL4側で行う想定です（L列「CALL4最終判断」）。'],
    ['現行4タグの1件（リンさんのケース）は、現行の4つを初期値として入れてあります。'],
    ['3つに減らしても、4つのまま維持しても構いません。判断を「コメント・理由」に書いてください。'],
    [''],
    ['シートの見方'],
    ['グレーの列は自動生成データです。編集すると警告が出ます（直せないわけではありません）。'],
    ['クリーム色の列と、オレンジの枠で囲まれたH列がご記入いただく欄です。'],
    ['G列が現在ついているタグ、H列がご記入いただく修正後のタグです。'],
    ['タグ列は色付きのチップ（角丸）で表示されます。セルをクリックすると一覧が出ます。'],
    ['タグは色分けされています。ただし色が似ているタグもあるので、識別は文字でお願いします。'],
    ['「変更あり?」列は、現行タグと修正後タグに差があると自動で「変更」と表示されます。'],
    ['タグを選ぶ順番は問いません。順番だけが違う場合は「変更」になりません。'],
    ['E列のケース名はCALL4のページへのリンクになっています。'],
    ['F列の概要は、CALL4掲載の説明文（200字以内）です。'],
    ['セルにコメントを付けたい場合は、右クリック > コメント が使えます。'],
    [''],
    ['「タグ一覧」シート'],
    ['各タグの現行件数と、修正後の件数・増減がリアルタイムで見えます。'],
    ['全体のバランスを確認するのに使ってください。'],
    [''],
    ['対象'],
    ['音信不通の7件を除いた88件です（アーカイブ38件 / 進行中50件）。'],
    [''],
    ['このあとの流れ（フェーズ2）'],
    ['このシートの確定後、徳永が各ケースのタグ割合を再計算し、別途ご確認をお願いします。'],
    ['タグの追加・削除で全体の割合が動くため、先にこちらを確定させる必要があります。']
  ];
  sh.getRange(1, 1, lines.length, 1).setValues(lines);
  sh.getRange(1, 1).setFontSize(14).setFontWeight('bold');
  lines.forEach(function (l, i) {
    if (['やっていただきたいこと', 'ルール', 'シートの見方',
         '「タグ一覧」シート', '対象', 'このあとの流れ（フェーズ2）'].indexOf(l[0]) >= 0) {
      sh.getRange(i + 1, 1).setFontWeight('bold').setBackground(COLOR.band);
    }
  });
  sh.setColumnWidth(1, 720);
  sh.getRange(1, 1, lines.length, 1).setWrap(true);
}
