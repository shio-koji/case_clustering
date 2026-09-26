/**
 * CALL4 タグ割合 確認シート セットアップ（フェーズ2用）
 *
 * 使い方:
 *   1. 新規スプレッドシートを作成
 *   2. ファイル > インポート で tag_ratio/out/ratio_review.csv を取り込む
 *      （インポート場所は「現在のシートを置換する」、区切り文字は「カンマ」）
 *   3. 拡張機能 > Apps Script を開き、このファイルの中身を全部貼り付けて保存
 *   4. 関数 setup を選んで実行（初回のみ権限の承認画面が出ます）
 *   5. シートを再読み込みすると、メニューに「割合ツール」が出ます
 *
 * フェーズ1（タグ確認）と違い、手作業で仕上げる箇所はありません。
 * マルチセレクトのチップを使わないので、Apps Script だけで完結します。
 *
 * 再実行しても安全（冪等）に作ってあります。
 */

// タグの並び。make_tag_review.py の TAGS と一致させること
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

/** フェーズ1のシート・カード・2部グラフと同じ11色。 */
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

// このファイルの版。setup 完了時のダイアログとルールシートに出す。
// 「変更あり?」が全行に出るなどの症状は、CSVとGASの版が食い違っているときに起きる。
var VERSION = '2026-08-20 順番方式';

// ratio_review.csv の見出し（この順で並んでいることを setup の最初に確認する）。
// s2_make_ratio_sheet.py の HEADER と一致させること。
var EXPECTED_HEADER = [
  'No', 'case_id', '状況', 'ケース名', '概要', 'タグ数', 'タグ（検索用）',
  'フェーズ1の指摘', '順番',
  'タグ1', '計算1', '割合1', 'タグ2', '計算2', '割合2',
  'タグ3', '計算3', '割合3', 'タグ4', '計算4', '割合4',
  '合計', '指摘と一致?', '色の変化', '変更あり?', 'コメント・理由', '確認者',
  'CALL4最終判断', '指摘の読み取り', '旧マップの主タグ', '既定の順番',
  '調整1', '調整2', '調整3', '調整4', 'URL'
];

var MAIN = 'タグ割合';
var SUMMARY = 'タグ別サマリ';
var RULES = 'ルール';
var SLOTS = 4;        // タグのスロット数（s2_make_ratio_sheet.py の MAX_SLOTS と一致）
var TOTAL = 100;      // 1ケースの合計（%）
var FLOOR = 1;        // 付与タグの最小%（0%はカードのバーから消えるため）
// 押し広げ後の値そのものは s2_make_ratio_sheet.py が「調整」列に書き込んでいる。
// 同じ計算を2か所に持たないため、こちらは列を参照するだけにしてある。
var ORDER_SEP = ' ＞ ';        // 順番のプルダウンの区切り（s2 の ORDER_SEP と一致）
var EVEN = 'どれも同じくらい';  // 均等割りを選ぶ選択肢（s2 の EVEN と一致）

// 列番号（ratio_review.csv の並びと一致させること）
var C = {
  no: 1, caseId: 2, status: 3, title: 4, desc: 5, nTags: 6, tagList: 7, note: 8,
  order: 9,                     // ★ここだけが記入欄（順番のプルダウン）
  slot: [                       // [タグ, 計算, 割合] × 4
    {tag: 10, calc: 11, fix: 12},
    {tag: 13, calc: 14, fix: 15},
    {tag: 16, calc: 17, fix: 18},
    {tag: 19, calc: 20, fix: 21}
  ],
  sum: 22, verdict: 23, flip: 24, changed: 25, comment: 26, reviewer: 27, decision: 28,
  reading: 29,   // 照合の数式を組むための作業列（setup で非表示にする）
  oldTop: 30,    // 旧マップの主タグ（同じく作業列）
  defOrder: 31,  // 既定の選択肢（「計算どおりに戻す」で使う。同じく作業列）
  adj: [32, 33, 34, 35],  // 押し広げ後の割合（順位1..4に渡す値。同じく作業列）
  url: 36        // リンク埋め込み後に削除する作業列
};

var TIE_PT = 2;   // これ未満の差は「ほぼ同率」とする（phase1_notes.py と同じ値）

var COLOR = {
  header: '#37474f',
  readonly: '#f5f5f5',
  editable: '#fffde7',
  changed: '#fff3cd',
  error: '#f8d7da',
  band: '#eceff1',
  warn: '#ffe0b2',
  fixed: '#fafafa'      // 1タグ（100%固定）の行
};


function onOpen() {
  SpreadsheetApp.getUi().createMenu('順番ツール')
      .addItem('選択行をフェーズ1のメモどおりにする', 'applyNoteSelection')
      .addItem('選択行を計算どおりの順番に戻す', 'resetSelection')
      .addSeparator()
      .addItem('全行を点検する', 'checkAll')
      .addToUi();
}


function setup() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = findMainSheet_(ss);
  var n = sh.getLastRow() - 1;
  if (n < 1) throw new Error('データ行がありません。先にCSVをインポートしてください。');

  assertLayout_(sh);       // 版の食い違いを最初に弾く
  sh.setName(MAIN);
  buildSummarySheet_(ss, n);
  buildRulesSheet_(ss);

  linkTitles_(sh, n);      // ケース名をリンク化し、URL作業列を削除
  layout_(sh, n);
  addFormulas_(sh, n);
  addRatioFormulas_(sh, n);
  addOrderValidation_(sh, n);
  addVerdictFormulas_(sh, n);
  addFlipFormulas_(sh, n);
  addConditionalFormats_(sh, n);
  protectReadOnly_(sh, n);
  onOpen();

  ss.setActiveSheet(sh);
  sh.getRange(2, C.order).activate();
  SpreadsheetApp.getUi().alert(
      'セットアップ完了：' + n + '件\n\n'
      + '版: ' + VERSION + '\n\n'
      + '記入していただくのは「順番」列（オレンジ枠）だけです。\n'
      + 'セルをクリックして、主役の順に並んだものを選んでください。\n'
      + '割合の数字は自動で入ります（数字を打つ必要はありません）。\n\n'
      + 'タグが1つだけの35行は100%固定なので、プルダウンはありません。\n'
      + '計算がほぼ同率だった2行は既定が「' + EVEN + '」です。\n'
      + 'そこで順番を選ぶと、判断が見えるように差をつけます。\n\n'
      + '「指摘と一致?」が × の行は、計算がフェーズ1のメモと食い違っています。\n'
      + 'そこから見ていただくのが早いです。\n'
      + 'メニュー「順番ツール > 選択行をフェーズ1のメモどおりにする」で一括で直せます。\n'
      + '（メニューが出ていない場合はシートを再読み込みしてください）');
}


/** 見出しが想定どおり並んでいるかを確認する。
 *
 *  CSVとこのスクリプトの版が食い違っていると、列番号がずれたまま
 *  数式が入り、「変更あり?」が全行に出るなどの分かりにくい壊れ方をする。
 *  黙って進めず、どの列が違うのかを名指しして止める。
 *
 *  setup を2回目に走らせるときはURL列が既に削除されているので、
 *  最後のURLが無い形も正しいものとして受け入れる。 */
function assertLayout_(sh) {
  var got = sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0]
              .map(function (v) { return normHeader_(v); });
  var full = EXPECTED_HEADER;
  var noUrl = EXPECTED_HEADER.slice(0, EXPECTED_HEADER.length - 1);
  var want = (got.length === noUrl.length) ? noUrl : full;

  var diff = [];
  for (var i = 0; i < Math.max(got.length, want.length); i++) {
    if (got[i] !== want[i]) {
      diff.push((i + 1) + '列目: シート=' + JSON.stringify(got[i] || '')
                + ' / 想定=' + JSON.stringify(want[i] || ''));
    }
  }
  if (diff.length) {
    throw new Error(
        '列の並びが想定と違います（' + got.length + '列。想定は '
        + full.length + '列 または ' + noUrl.length + '列）。\n'
        + 'いまの tag_ratio/out/ratio_review.csv を取り込み直し、'
        + 'このスクリプト（' + VERSION + '）を貼り直してください。\n\n'
        + diff.slice(0, 8).join('\n')
        + (diff.length > 8 ? '\n…他 ' + (diff.length - 8) + '件' : ''));
  }
}


/** 見出しセルの比較用。BOM・ゼロ幅スペース・ノーブレークスペースを落とす。 */
function normHeader_(v) {
  return String(v === null || v === undefined ? '' : v)
      .replace(/[﻿​‌‍ ]/g, '')
      .trim();
}


/** B1が「case_id」でF1が「タグ数」のシートを本体とみなす。
 *  フェーズ1のシート（B1=case_id だが F1=概要）と取り違えないための二重条件。 */
function findMainSheet_(ss) {
  var sheets = ss.getSheets();
  var seen = [];
  for (var i = 0; i < sheets.length; i++) {
    var sh = sheets[i];
    var raw1 = sh.getRange(1, 1).getValue();
    var a1 = normHeader_(raw1);
    var b1 = normHeader_(sh.getRange(1, C.caseId).getValue());
    var f1 = normHeader_(sh.getRange(1, C.nTags).getValue());
    seen.push('「' + sh.getName() + '」B1=' + JSON.stringify(b1) + ' F1=' + JSON.stringify(f1));
    if (b1 === 'case_id' && f1 === 'タグ数') {
      if (a1 !== String(raw1)) sh.getRange(1, 1).setValue(a1);
      return sh;
    }
  }
  throw new Error('B1が「case_id」・F1が「タグ数」のシートが見つかりません。'
                  + '取り込むファイルは tag_ratio/out/ratio_review.csv です'
                  + '（フェーズ1の tag_review.csv とは別物です）。'
                  + ' 各シートの1行目: ' + seen.join(' / '));
}


/** ケース名にCALL4へのリンクを埋め込み、末尾のURL作業列を削除する。
 *  HYPERLINK数式ではなくリッチテキストにするのは、CSVへ書き出し直したときに
 *  ケース名がプレーンテキストで残り、読み戻しが壊れないようにするため。 */
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
  sh.setRowHeight(1, 40);

  var widths = {};
  widths[C.no] = 40;
  widths[C.caseId] = 78;
  widths[C.status] = 58;
  widths[C.title] = 230;
  widths[C.desc] = 300;
  widths[C.nTags] = 44;
  widths[C.tagList] = 150;
  widths[C.note] = 210;
  widths[C.order] = 260;
  widths[C.sum] = 52;
  widths[C.verdict] = 52;
  widths[C.flip] = 230;
  widths[C.changed] = 52;
  widths[C.comment] = 240;
  widths[C.reviewer] = 74;
  widths[C.decision] = 110;
  for (var s = 0; s < SLOTS; s++) {
    widths[C.slot[s].tag] = 150;
    widths[C.slot[s].calc] = 50;
    widths[C.slot[s].fix] = 56;
  }
  for (var col in widths) sh.setColumnWidth(Number(col), widths[col]);

  // 読み取り専用ブロックと記入ブロック。記入欄は「順番」とコメント類だけ
  sh.getRange(2, 1, n, C.note).setBackground(COLOR.readonly);
  sh.getRange(2, C.sum, n, 4).setBackground(COLOR.readonly);
  sh.getRange(2, C.comment, n, C.decision - C.comment + 1).setBackground(COLOR.editable);
  sh.getRange(2, C.order, n, 1).setBackground(COLOR.editable).setFontWeight('bold');
  // 記入欄であることは塗りではなく枠線で示す（フェーズ1と同じ扱い）
  sh.getRange(1, C.order, n + 1, 1)
    .setBorder(true, true, true, true, false, false, '#f9a825',
               SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
  for (var s2 = 0; s2 < SLOTS; s2++) {
    // タグ名セルは条件付き書式でタグ色に塗るので、ここでは触らない
    sh.getRange(2, C.slot[s2].tag, n, 1).setBackground(null);
    sh.getRange(2, C.slot[s2].calc, n, 1)
      .setBackground(COLOR.readonly).setFontColor('#b0bec5');
    sh.getRange(2, C.slot[s2].fix, n, 1)
      .setBackground(COLOR.readonly).setFontWeight('bold');
    sh.getRange(2, C.slot[s2].calc, n, 1).setNumberFormat('0"%"');
    sh.getRange(2, C.slot[s2].fix, n, 1).setNumberFormat('0"%"');
    sh.getRange(2, C.slot[s2].calc, n, 2).setHorizontalAlignment('center');
    // 「計算」列は隠す。人が見るべき数字は「割合」だけで、
    // 2種類の数字が並んでいるとどちらを判断すればよいのか分からなくなる。
    // 消さずに隠すのは、割合の数式がこの値を元にしているため
    // （読み戻しの s3_ingest_ratios.py も参照する。隠した列もCSVには出る）。
    sh.hideColumns(C.slot[s2].calc);
  }
  sh.getRange(2, C.sum, n, 1).setNumberFormat('0"%"');

  sh.getRange(2, 1, n, lastCol).setVerticalAlignment('top');
  [C.title, C.desc, C.tagList, C.note, C.order, C.flip, C.comment].forEach(function (c) {
    sh.getRange(2, c, n, 1).setWrap(true);
  });
  [C.no, C.status, C.nTags, C.sum, C.verdict, C.changed].forEach(function (c) {
    sh.getRange(2, c, n, 1).setHorizontalAlignment('center');
  });
  sh.getRange(2, C.desc, n, 1).setFontSize(9).setFontColor('#455a64');
  sh.getRange(2, C.tagList, n, 1).setFontSize(9);
  sh.getRange(2, C.note, n, 1).setFontSize(9).setFontColor('#5d4037');
  sh.getRange(2, C.verdict, n, 1).setFontSize(12).setFontWeight('bold');
  sh.getRange(2, C.flip, n, 1).setFontSize(9).setWrap(true);
  sh.getRange(2, C.order, n, 1).setFontSize(10);
  sh.hideColumns(C.reading, 7);   // 照合・色比較・既定の順番・調整値の作業列

  // 見出しは書き換えない。読み戻し（s3_ingest_ratios.py）が見出し名で列を引くうえ、
  // 書き換えると2回目の setup で assertLayout_ が通らなくなる。
  // 何の数字かは「ルール」シートと、計算列を隠すことで示す。
  sh.setFrozenRows(1);
  sh.setFrozenColumns(C.title);   // No〜ケース名 を固定
  sh.autoResizeRows(2, n);

  var existing = sh.getFilter();
  if (existing) existing.remove();
  sh.getRange(1, 1, n + 1, lastCol).createFilter();
}


/** 合計と「変更あり?」を数式で持つ。手で書き換えられないようにするため。 */
function addFormulas_(sh, n) {
  var sums = [], changed = [];
  var ord = '$' + colLetter_(C.order);
  for (var i = 0; i < n; i++) {
    var r = i + 2;
    var fixRefs = [];
    for (var s = 0; s < SLOTS; s++) {
      fixRefs.push('$' + colLetter_(C.slot[s].fix) + r);

    }
    sums.push(['=SUM(' + fixRefs.join(',') + ')']);
    // 「変更あり?」は、選んだ順番が既定の選択肢と違うかどうか。
    // 既定は「計算どおりの順番」だが、計算がほぼ同率の行だけは
    // 「どれも同じくらい」にしてあるので、タグ列の並びではなく既定の列と比べる。
    // 列がずれたまま静かに誤判定しないよう、setup の冒頭で assertLayout_ が
    // 見出しの並びを確認している。
    changed.push(['=IF(' + ord + r + '="","",'
                  + 'IF(TRIM(' + ord + r + ')=TRIM($' + colLetter_(C.defOrder) + r + '),'
                  + '"","変更"))']);
  }
  sh.getRange(2, C.sum, n, 1).setFormulas(sums);
  sh.getRange(2, C.changed, n, 1).setFormulas(changed);
}


/** 割合の列を「順番」から導く数式にする。
 *
 *  順位は「順番」の文字列中での出現位置を他のタグと比べて数える。
 *  タグ名は互いの部分文字列でないことを make_tag_review.py が保証しているので、
 *  SEARCH で位置を取って比較して差し支えない。
 *
 *  渡す値は4通りに分かれる:
 *    1. 順番が空（1タグ）           → 計算値そのまま（=100）
 *    2. 「どれも同じくらい」        → 均等割り（端数は先頭のスロットへ）
 *    3. 計算どおりの順番            → 計算値そのまま。人は何も動かしていないので
 *                                     数字を作らない
 *    4. 計算と違う順番を選んだ      → 「調整」列の値（隣り合う差を最低10ptまで
 *                                     押し広げた値）を順位に割り当てる。
 *       計算値が同値（50:50 や 34:33:33）のとき、順番を変えても数字が動かないと
 *       人が決めたことが紙に出ないため。既に十分な差はそのまま残る。
 *
 *  数値を人が打たないので、割合列は数式のままでよい（＝合計は常に100になる）。 */
function addRatioFormulas_(sh, n) {
  var ord = '$' + colLetter_(C.order);
  var nt  = '$' + colLetter_(C.nTags);

  function pos(slotIdx, r) {          // 順番の中でのそのタグの位置（無ければ大きい数）
    var tg = '$' + colLetter_(C.slot[slotIdx].tag) + r;
    return 'IF(' + tg + '="",999,IFERROR(SEARCH(' + tg + ',' + ord + r + '),999))';
  }

  var out = [];
  for (var i = 0; i < n; i++) {
    var r = i + 2;
    var row = [];
    for (var k = 0; k < SLOTS; k++) {
      var tg = '$' + colLetter_(C.slot[k].tag) + r;
      var calcCells = [], adjCells = [];
      for (var s = 0; s < SLOTS; s++) {
        calcCells.push('$' + colLetter_(C.slot[s].calc) + r);
        adjCells.push('$' + colLetter_(C.adj[s]) + r);
      }
      // 自分より前に出てくるタグの数 + 1 = 順位
      var terms = [];
      for (var j = 0; j < SLOTS; j++) {
        if (j === k) continue;
        terms.push('IF(' + pos(j, r) + '<' + pos(k, r) + ',1,0)');
      }
      var rank = '1+' + terms.join('+');
      // 均等割り: 先頭 (100 mod n) スロットだけ +1
      var even = 'INT(' + TOTAL + '/' + nt + r + ')'
               + '+IF(' + (k + 1) + '<=' + TOTAL + '-INT(' + TOTAL + '/' + nt + r + ')*'
               + nt + r + ',1,0)';
      // 押し広げるかどうかは「既定の選択肢」と比べて決める。タグ列の並びと
      // 比べてはいけない: ほぼ同率の行は既定が「どれも同じくらい」なので、
      // タグ順を選ぶこと自体が人の判断になる（resolve_ratio と同じ規則）。
      var isDefault = 'TRIM(' + ord + r + ')=TRIM($' + colLetter_(C.defOrder) + r + ')';
      row.push('=IF(' + tg + '="","",'
             + 'IF(' + ord + r + '="",' + calcCells[k] + ','
             + 'IF(' + ord + r + '="' + EVEN + '",' + even + ','
             + 'IF(' + isDefault + ',' + calcCells[k] + ','
             + 'CHOOSE(' + rank + ',' + adjCells.join(',') + ')))))');
    }
    out.push(row);
  }
  // 割合列は連続していない（タグ・計算と交互）ので1列ずつ入れる
  for (var k2 = 0; k2 < SLOTS; k2++) {
    var col = [];
    for (var m = 0; m < n; m++) col.push([out[m][k2]]);
    sh.getRange(2, C.slot[k2].fix, n, 1).setFormulas(col);
  }
}


/** 「順番」列に、行ごとのプルダウンを付ける。
 *
 *  選択肢はその行のタグの並べ替え（2タグなら2通り、3タグなら6通り）＋
 *  「どれも同じくらい」。行ごとに中身が違うので1行ずつ設定する。
 *  1タグの行にはプルダウンを付けない（順番を決める余地が無い）。 */
function addOrderValidation_(sh, n) {
  var tags = sh.getRange(2, C.slot[0].tag, n, C.slot[SLOTS - 1].tag - C.slot[0].tag + 1)
               .getValues();
  var step = C.slot[1].tag - C.slot[0].tag;
  var target = sh.getRange(2, C.order, n, 1);
  target.clearDataValidations();

  for (var i = 0; i < n; i++) {
    var ts = [];
    for (var s = 0; s < SLOTS; s++) {
      var v = String(tags[i][s * step] || '').trim();
      if (v) ts.push(v);
    }
    if (ts.length < 2) continue;
    var opts = permutations_(ts).map(function (p) { return p.join(ORDER_SEP); });
    opts.push(EVEN);
    sh.getRange(i + 2, C.order).setDataValidation(
        SpreadsheetApp.newDataValidation()
            .requireValueInList(opts, true)
            .setAllowInvalid(false)
            .setHelpText('主役の順に選んでください。割合は自動で入ります。')
            .build());
  }
}


/** 並べ替えを全部作る（最大4タグ＝24通り）。先頭は渡された順のまま。 */
function permutations_(arr) {
  if (arr.length <= 1) return [arr.slice()];
  var out = [];
  for (var i = 0; i < arr.length; i++) {
    var rest = arr.slice(0, i).concat(arr.slice(i + 1));
    permutations_(rest).forEach(function (p) {
      out.push([arr[i]].concat(p));
    });
  }
  return out;
}


/** その行の主タグ（＝いちばん割合が大きいタグ）を返す式。
 *
 *  「順番」の先頭に書かれているタグがそれ。割合の最大値を MATCH で探す形にすると、
 *  探索範囲にタグ・計算・割合が交互に入っているため、計算値の1位と2位が
 *  同値のケースで計算セルを先に拾ってしまう（タグ名ではなく数値が返る）。
 *  順番の文字列から取れば、その曖昧さが原理的に生じない。
 *
 *  「どれも同じくらい」と1タグの行は順位が無いので、
 *  計算値がいちばん大きいタグ（＝スロット1）を主タグとして扱う。 */
function topTagExpr_(r) {
  var tag1 = '$' + colLetter_(C.slot[0].tag) + r;
  var ord = '$' + colLetter_(C.order) + r;
  return 'IF(OR(' + ord + '="",' + ord + '="' + EVEN + '"),' + tag1 + ','
       + 'TRIM(INDEX(SPLIT(' + ord + ',"＞"),1,1)))';
}


/** 紙のポスターの色（＝主タグ）が、確認者が見ていた旧マップから変わるかどうか。
 *  フェーズ1で「ポスターカラー変更」列を作って追っていた関心事なので、
 *  割合を直した瞬間に色が変わる／戻ることが見えるように数式にしてある。 */
function addFlipFormulas_(sh, n) {
  var out = [];
  for (var i = 0; i < n; i++) {
    var r = i + 2;
    var oldTop = '$' + colLetter_(C.oldTop) + r;
    out.push(['=IFERROR(IF(OR(' + oldTop + '="",' + oldTop + '=' + topTagExpr_(r) + '),"",'
              + oldTop + '&" → "&' + topTagExpr_(r) + '),"")']);
  }
  sh.getRange(2, C.flip, n, 1).setFormulas(out);
}


/** フェーズ1の指摘（「指摘の読み取り」列）と、いま入っている修正値を照合する数式。
 *
 *  タグ列とその修正値は2列おきに並んでいるので、
 *  SUMIF(タグの範囲, タグ名, 2列右の範囲) でそのタグの修正値が取れる。
 *  （タグは1行に1回しか出てこないので合計＝その値になる）
 *
 *  静的な値ではなく数式にしているのは、割合を直した瞬間に
 *  「指摘どおりになったか」が分かるようにするため。 */
function addVerdictFormulas_(sh, n) {
  var tagFrom = colLetter_(C.slot[0].tag);
  var tagTo   = colLetter_(C.slot[SLOTS - 1].tag);
  var fixFrom = colLetter_(C.slot[0].fix);
  var fixTo   = colLetter_(C.slot[SLOTS - 1].fix);
  var readings = sh.getRange(2, C.reading, n, 1).getValues();
  var out = [];

  for (var i = 0; i < n; i++) {
    var r = i + 2;
    var reading = String(readings[i][0] || '').trim();
    if (!reading) { out.push(['']); continue; }

    function val(tag) {   // その行の tag の修正値
      return 'SUMIF($' + tagFrom + r + ':$' + tagTo + r + ',"' + tag + '",$'
             + fixFrom + r + ':$' + fixTo + r + ')';
    }

    var f;
    if (reading.indexOf('MAX:') === 0) {
      // 「Aが最大」。2番目に大きい値との差で、はっきり最大かどうかを見る
      var a = reading.slice(4);
      var cells = [];
      for (var s = 0; s < SLOTS; s++) cells.push('$' + colLetter_(C.slot[s].fix) + r);
      var arr = '{' + cells.join(',') + '}';
      var v = val(a);
      f = '=IFERROR(IF(' + v + '<LARGE(' + arr + ',1),'
        + 'IF(LARGE(' + arr + ',1)-' + v + '>=' + TIE_PT + ',"×","≒"),'
        + 'IF(' + v + '-IFERROR(LARGE(' + arr + ',2),0)>=' + TIE_PT + ',"◯","≒")),"!")';
    } else {
      // 「A>B」「A>B>C」。隣り合う差の最小値で判定する
      var chain = reading.split('>');
      var gaps = [];
      for (var k = 0; k + 1 < chain.length; k++) {
        gaps.push('(' + val(chain[k]) + '-' + val(chain[k + 1]) + ')');
      }
      var g = gaps.length > 1 ? 'MIN(' + gaps.join(',') + ')' : gaps[0];
      f = '=IFERROR(IF(' + g + '<=-' + TIE_PT + ',"×",'
        + 'IF(' + g + '>=' + TIE_PT + ',"◯","≒")),"!")';
    }
    out.push([f]);
  }
  sh.getRange(2, C.verdict, n, 1).setFormulas(out);
}


function colLetter_(col) {
  var s = '';
  while (col > 0) {
    var m = (col - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    col = (col - 1 - m) / 26;
  }
  return s;
}


function addConditionalFormats_(sh, n) {
  var rules = [];
  var tagRanges = [], fixRanges = [];
  for (var s = 0; s < SLOTS; s++) {
    tagRanges.push(sh.getRange(2, C.slot[s].tag, n, 1));
    fixRanges.push(sh.getRange(2, C.slot[s].fix, n, 1));
  }

  // 1) タグ名セルをタグ色で塗る（カード・2部グラフと同じ11色）
  TAGS.forEach(function (t) {
    rules.push(SpreadsheetApp.newConditionalFormatRule()
        .whenTextEqualTo(t)
        .setBackground(TAG_COLOR[t])
        .setRanges(tagRanges).build());
  });

  // 2) 合計が100でない（最優先で気づいてほしいので No 列も一緒に光らせる）
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$' + colLetter_(C.sum) + '2<>' + TOTAL)
      .setBackground(COLOR.error).setBold(true)
      .setRanges([sh.getRange(2, C.sum, n, 1), sh.getRange(2, C.no, n, 1)]).build());

  // 3) 順番が空（2タグ以上なのに選ばれていない）
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=AND($' + colLetter_(C.nTags) + '2>1,$'
                            + colLetter_(C.order) + '2="")')
      .setBackground(COLOR.error)
      .setRanges([sh.getRange(2, C.order, n, 1)]).build());

  // 3.1) 「どれも同じくらい」を選んだ行は分かるようにしておく
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo(EVEN)
      .setBackground('#e0f7fa')
      .setRanges([sh.getRange(2, C.order, n, 1)]).build());

  // 3.5) 指摘との照合。×は赤、ほぼ同率は橙、◯は薄い緑。
  //      「!」は数式が評価できなかった印（出たら知らせてほしい）
  var vr = [sh.getRange(2, C.verdict, n, 1)];
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('×').setBackground(COLOR.error).setRanges(vr).build());
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('≒').setBackground(COLOR.warn).setRanges(vr).build());
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('◯').setBackground('#e8f5e9').setRanges(vr).build());
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('!').setBackground('#ce93d8').setRanges(vr).build());

  // 3.6) フェーズ1で未決着だった行は、指摘欄を目立たせる
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=REGEXMATCH($' + colLetter_(C.note) + '2&"","未決着")')
      .setBackground(COLOR.warn)
      .setRanges([sh.getRange(2, C.note, n, 1)]).build());

  // 3.7) ポスターの色（主タグ）が旧マップから変わる行
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$' + colLetter_(C.flip) + '2<>""')
      .setBackground('#ede7f6').setFontColor('#4527a0')
      .setRanges([sh.getRange(2, C.flip, n, 1)]).build());

  // 4) 変更あり
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$' + colLetter_(C.changed) + '2="変更"')
      .setBackground(COLOR.changed).setBold(true)
      .setRanges([sh.getRange(2, C.changed, n, 1)]).build());

  // 5) タグ1つ＝100%固定。判断の余地がないので淡くして「触らなくてよい」と分かるように
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$' + colLetter_(C.nTags) + '2=1')
      .setBackground(COLOR.fixed).setFontColor('#b0bec5')
      .setRanges(fixRanges.concat([sh.getRange(2, C.nTags, n, 1),
                                   sh.getRange(2, C.order, n, 1)])).build());

  // 6) アーカイブ行は色を落とす（フェーズ1と同じ見え方に揃える）
  rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=$' + colLetter_(C.status) + '2="アーカイブ"')
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
  var blocks = [[1, 1, n + 1, C.note], [1, C.sum, n + 1, 4],
                [1, C.reading, n + 1, 7]];
  for (var s = 0; s < SLOTS; s++) {
    blocks.push([1, C.slot[s].tag, n + 1, 3]);   // タグ名 + 計算値 + 割合（全部自動）
  }
  blocks.forEach(function (a) {
    sh.getRange(a[0], a[1], a[2], a[3]).protect()
      .setDescription('自動生成データ（原則編集不可）').setWarningOnly(true);
  });
}


/** タグ別サマリ。「1ケースあたりの重み」ではなく、タグごとの重み合計＝
 *  そのタグが全体の何ケース分に相当するかを出す。おりえさんの
 *  「全体のバランスを担保したい」に対応する面で、カードマップでの
 *  タグの太さ（2部グラフのエッジ総量）に直結する。 */
function buildSummarySheet_(ss, n) {
  var sh = ss.getSheetByName(SUMMARY) || ss.insertSheet(SUMMARY);
  sh.clear();
  sh.setConditionalFormatRules([]);

  var last = n + 1;
  var q = "'" + MAIN + "'!";
  function sumifChain(kind) {          // kind: 'calc' | 'fix'
    var parts = [];
    for (var s = 0; s < SLOTS; s++) {
      var tagCol = colLetter_(C.slot[s].tag);
      var valCol = colLetter_(C.slot[s][kind]);
      parts.push('SUMIF(' + q + '$' + tagCol + '$2:$' + tagCol + '$' + last + ',$A2,'
                 + q + '$' + valCol + '$2:$' + valCol + '$' + last + ')');
    }
    return parts.join('+');
  }

  var rows = [['タグ名', '色', '付与件数', '計算の重み合計', '選択後の重み合計',
               '増減', '平均割合', '備考']];
  for (var i = 0; i < TAGS.length; i++) {
    var r = i + 2;
    rows.push([
      TAGS[i],
      TAG_COLOR[TAGS[i]],
      "=COUNTIF(" + q + "$" + colLetter_(C.tagList) + "$2:$" + colLetter_(C.tagList)
        + "$" + last + ',"*"&$A' + r + '&"*")',
      '=ROUND((' + sumifChain('calc').replace(/\$A2/g, '$A' + r) + ')/100,2)',
      '=ROUND((' + sumifChain('fix').replace(/\$A2/g, '$A' + r) + ')/100,2)',
      '=ROUND($E' + r + '-$D' + r + ',2)',
      '=IFERROR(ROUND($E' + r + '/$C' + r + '*100,0)&"%","—")',
      ''
    ]);
  }
  var tr = TAGS.length + 2;
  rows.push(['合計', '',
             '=SUM($C$2:$C$' + (tr - 1) + ')',
             '=ROUND(SUM($D$2:$D$' + (tr - 1) + '),2)',
             '=ROUND(SUM($E$2:$E$' + (tr - 1) + '),2)',
             '=ROUND($E' + tr + '-$D' + tr + ',2)', '', '']);

  sh.getRange(1, 1, rows.length, 8).setValues(rows);
  sh.getRange(1, 1, 1, 8)
    .setBackground(COLOR.header).setFontColor('#ffffff').setFontWeight('bold').setWrap(true);
  sh.getRange(tr, 1, 1, 8).setFontWeight('bold').setBackground(COLOR.band);
  sh.getRange(2, 6, TAGS.length + 1, 1).setNumberFormat('"+"0.00;"△"0.00;0.00');

  for (var j = 0; j < TAGS.length; j++) {
    sh.getRange(j + 2, 2)
      .setValue(TAG_COLOR[TAGS[j]])
      .setBackground(TAG_COLOR[TAGS[j]])
      .setHorizontalAlignment('center').setFontSize(9);
  }

  sh.setColumnWidth(1, 220);
  sh.setColumnWidth(2, 80);
  sh.setColumnWidths(3, 5, 105);
  sh.setColumnWidth(8, 280);
  sh.setFrozenRows(1);

  sh.getRange(tr + 2, 1).setValue(
      '「重み合計」は、そのタグが全体の何ケース分に相当するかです'
      + '（各ケースの割合を足して100で割った値）。'
      + '付与件数が同じでも、割合を小さくすると重みは減ります。'
      + '紙のマップでは、この重みがタグから伸びる線の太さの合計になります。');
  sh.getRange(tr + 3, 1).setValue(
      '「平均割合」は、そのタグが付いたケースでの平均的な割合です。'
      + '低いタグは「どのケースでも脇役」ということになります。');
  sh.getRange(tr + 2, 1, 2, 1).setWrap(true);
  sh.setRowHeights(tr + 2, 2, 44);
}


function buildRulesSheet_(ss) {
  var sh = ss.getSheetByName(RULES) || ss.insertSheet(RULES);
  sh.clear();
  var lines = [
    ['CALL4 タグ割合の確認のお願い（フェーズ2）'],
    [''],
    ['何を決める作業か'],
    ['1つのケースに複数のタグが付いているとき、そのケースの中で'],
    ['どのタグが主題で、どれが脇役かの順番を決めます。'],
    ['この配分は、紙のケースマップでカード下部の帯の幅と、'],
    ['タグとカードを結ぶ線の太さになります。'],
    [''],
    ['やっていただきたいこと'],
    ['オレンジ枠の「順番」列で、そのケースの主役の順に並んだものを選ぶだけです。'],
    ['数字を打つ必要はありません。割合は選んだ順番から自動で入ります。'],
    ['初期値は計算どおりの順番です。違和感がなければ触らなくて大丈夫です。'],
    ['どれが主役とも言えない場合は「どれも同じくらい」を選んでください。'],
    ['判断の理由や、「もっと差をつけたい」などの要望は'],
    ['「コメント・理由」列に自由に書いてください（数字はこちらで調整します）。'],
    ['担当された方のお名前を「確認者」列に入れていただけると助かります。'],
    [''],
    ['割合の数字はどう決まるか'],
    ['選んだ順番の1位に、計算で出た一番大きい割合を割り当てます。'],
    ['2位には2番目の割合、というように順に当てはめていきます。'],
    ['つまり「差の大きさ」は本文から出た値を使い、「どちらが上か」だけを'],
    ['決めていただく形です。順番を変えなければ数字は1つも動きません。'],
    ['「どれも同じくらい」を選ぶと均等割り（3タグなら34/33/33）になります。'],
    [''],
    ['計算がほぼ同じ数字だったケースについて'],
    ['計算が 50:50 や 34:33:33 のように差の無い結果になったケースは、'],
    ['機械の側に「どれが主役か」の情報がありません。'],
    ['そういう行は既定を「どれも同じくらい」にしてあります（2件）。'],
    ['ここで順番を選んでいただくと、選んだ判断が紙に出るように'],
    ['最低10ポイントの差をつけます（2タグなら55:45、3タグなら43:33:24）。'],
    ['計算の時点で十分な差があるケースは、その差をそのまま使います。'],
    [''],
    ['フェーズ1の指摘'],
    ['タグ確認のときに「マッピングとの齟齬確認」欄に書いていただいたメモです。'],
    ['例：「外国>手続」＝ 外国にルーツを持つ人々の方が公正な手続より大きいはず。'],
    ['「指摘と一致?」列が、いま選んでいる順番がその指摘どおりかを自動判定します。'],
    ['  ◯ … 指摘どおり（差もはっきりある）'],
    ['  ≒ … 順序は合っているが差が2%未満（実質は同率）'],
    ['  × … 指摘と逆（再計算がメモと食い違っている＝要判断）'],
    ['  空欄 … 指摘が無い、または大小関係として読み取れなかったメモ'],
    ['順番を選び直すとその場で判定が変わります。'],
    ['「色の変化」列は、いちばん割合が大きいタグ（＝ポスターの色）が'],
    ['以前のマップから変わる場合に「旧 → 新」と表示します。4件あります。'],
    ['これも自動なので、順番を選び直して色を戻せば表示は消えます。'],
    ['「【未決着】」が付いた12件は、フェーズ1で色を決めきれなかった行です。'],
    ['（ポスターカラー変更列がFALSEだった行をそう解釈しています）'],
    [''],
    ['触らなくてよい行'],
    ['タグが1つだけのケースは100%固定です（88件のうち35件）。'],
    ['プルダウンを置いていないので、飛ばして進んでください。'],
    ['F列「タグ数」でフィルタをかけると、2つ以上の53行だけを表示できます。'],
    [''],
    ['計算値は何か'],
    ['ケース本文を多言語AIモデルで数値化し、そのタグらしさを比べた結果です。'],
    ['「文章がどのタグに寄っているか」であって、法的な重要度ではありません。'],
    ['違うと感じたら、遠慮なく上書きしてください。人の判断を最終とします。'],
    [''],
    ['メニュー「順番ツール」'],
    ['選択行をフェーズ1のメモどおりにする … メモ（例「外国>手続」）の順に並べ替えます。'],
    ['  メモに出てこないタグは、いまの順のまま後ろに回します。'],
    ['選択行を計算どおりの順番に戻す … 初期状態に戻します。'],
    ['全行を点検する … 未選択の行・指摘と食い違う行を一覧で出します。'],
    ['（メニューが出ていない場合はシートを再読み込みしてください）'],
    [''],
    ['シートの見方'],
    ['グレーの列は自動生成データです。編集すると警告が出ます。'],
    ['タグ名のセルは11色に塗り分けてあります。紙のマップでも同じ色を使います。'],
    ['色が似ているタグもあるので、識別は文字でお願いします。'],
    ['「割合」列が、選んだ順番から決まった最終的な割合です（自動・編集不可）。'],
    ['紙のカードの帯とタグへの線の太さは、この数字になります。'],
    ['「変更あり?」列は、計算と違う順番を選ぶと「変更」と表示されます。'],
    ['「指摘と一致?」に「!」が出た場合は、判定が計算できていないので知らせてください。'],
    ['D列のケース名はCALL4のページへのリンクになっています。'],
    [''],
    ['「タグ別サマリ」シート'],
    ['タグごとの重み合計（全体の何ケース分に相当するか）と平均割合が見えます。'],
    ['個々のケースを直したあと、全体のバランスが崩れていないか確認できます。'],
    [''],
    ['対象'],
    ['フェーズ1でタグを確定した88件です（アーカイブ38件 / 進行中50件）。'],
    ['タグ自体の追加・削除はフェーズ1で確定済みなので、ここでは変えられません。'],
    ['タグを直したい場合は「コメント・理由」に書いてください。'],
    [''],
    ['このあとの流れ'],
    ['このシートの確定後、割合を反映したカードとタグの2部グラフを作り直し、'],
    ['配置エディタ（ブラウザで開くGUI）に反映します。'],
    [''],
    ['版: ' + VERSION]
  ];
  sh.getRange(1, 1, lines.length, 1).setValues(lines);
  sh.getRange(1, 1).setFontSize(14).setFontWeight('bold');
  var heads = ['何を決める作業か', 'やっていただきたいこと', '割合の数字はどう決まるか',
               '計算がほぼ同じ数字だったケースについて',
               'フェーズ1の指摘', '触らなくてよい行',
               '計算値は何か', 'メニュー「順番ツール」', 'シートの見方',
               '「タグ別サマリ」シート', '対象', 'このあとの流れ'];
  lines.forEach(function (l, i) {
    if (heads.indexOf(l[0]) >= 0) {
      sh.getRange(i + 1, 1).setFontWeight('bold').setBackground(COLOR.band);
    }
  });
  sh.setColumnWidth(1, 720);
  sh.getRange(1, 1, lines.length, 1).setWrap(true);
}


// ---------------------------------------------------------------- ツール群

function mainSheet_() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(MAIN);
  if (!sh) throw new Error('「' + MAIN + '」シートが見つかりません。setup を実行してください。');
  return sh;
}


/** 選択範囲が触れている行番号を集める（見出し行は除く）。 */
function selectedRows_(sh) {
  var sel = sh.getActiveRangeList();
  if (!sel) return null;
  var rows = [], seen = {}, ranges = sel.getRanges();
  for (var i = 0; i < ranges.length; i++) {
    var r0 = ranges[i].getRow(), rn = ranges[i].getNumRows();
    for (var r = Math.max(r0, 2); r < r0 + rn; r++) {
      if (!seen[r] && r <= sh.getLastRow()) { seen[r] = true; rows.push(r); }
    }
  }
  return rows;
}


/** 選択行の順番を、フェーズ1のメモ（「指摘の読み取り」列）どおりに書き換える。
 *
 *  動かすのはメモに出てくるタグだけ。いまそのタグたちが占めている順位に、
 *  メモの順で入れ直す（メモに出てこないタグは順位を動かさない）。
 *  例: いま「刑事 ＞ 外国 ＞ 手続」でメモが「外国>手続」なら、
 *      外国と手続は2位・3位のままなので何も変わらない。
 *      いま「刑事 ＞ 手続 ＞ 外国」なら 2位・3位を入れ替えて
 *      「刑事 ＞ 外国 ＞ 手続」にする（刑事は1位のまま）。
 *
 *  メモが言っていないことまで決めてしまわないための最小限の入れ替え。
 *  「Aが最大」形式（MAX:）のメモだけは、Aを先頭に出す（他は相対順を保つ）。 */
function applyNoteSelection() {
  var sh = mainSheet_();
  var rows = selectedRows_(sh);
  if (!rows) { SpreadsheetApp.getUi().alert('行を選択してから実行してください。'); return; }

  var applied = 0, skipped = [];
  for (var i = 0; i < rows.length; i++) {
    var r = rows[i];
    var reading = String(sh.getRange(r, C.reading).getValue() || '').trim();
    var current = String(sh.getRange(r, C.order).getValue() || '').trim();
    if (!reading || !current || current === EVEN) {
      skipped.push(sh.getRange(r, C.no).getValue());
      continue;
    }
    var cur = current.split(ORDER_SEP.trim()).map(function (x) { return x.trim(); });
    var wanted = reading.indexOf('MAX:') === 0
        ? [reading.slice(4)]
        : reading.split('>').map(function (x) { return x.trim(); });
    // メモに出てくるタグが、そのケースのタグに無ければ触らない
    var known = true;
    wanted.forEach(function (t) { if (cur.indexOf(t) < 0) known = false; });
    if (!known) { skipped.push(sh.getRange(r, C.no).getValue()); continue; }

    var next;
    if (reading.indexOf('MAX:') === 0) {
      // 最大のタグを先頭へ。残りは相対順のまま
      var head = wanted[0];
      next = [head].concat(cur.filter(function (t) { return t !== head; })).join(ORDER_SEP);
    } else {
      // メモに出てくるタグが占めている順位を集め、そこへメモの順で入れ直す
      var slotsOfWanted = [];
      for (var k = 0; k < cur.length; k++) {
        if (wanted.indexOf(cur[k]) >= 0) slotsOfWanted.push(k);
      }
      var arranged = cur.slice();
      for (var m = 0; m < slotsOfWanted.length; m++) {
        arranged[slotsOfWanted[m]] = wanted[m];
      }
      next = arranged.join(ORDER_SEP);
    }
    if (next !== current) {
      sh.getRange(r, C.order).setValue(next);
      applied += 1;
    }
  }
  var msg = applied + '行の順番をメモどおりにしました。';
  if (skipped.length) {
    msg += '\n\nメモが無い／読み取れない行は変えていません（No: '
         + skipped.join(', ') + '）。';
  }
  SpreadsheetApp.getUi().alert(msg);
}


/** 選択行の順番を、計算どおりの並び（既定）に戻す。 */
function resetSelection() {
  var sh = mainSheet_();
  var rows = selectedRows_(sh);
  if (!rows) { SpreadsheetApp.getUi().alert('行を選択してから実行してください。'); return; }
  var touched = 0;
  for (var i = 0; i < rows.length; i++) {
    var def = sh.getRange(rows[i], C.defOrder).getValue();
    if (def === '' || def === null) continue;
    sh.getRange(rows[i], C.order).setValue(def);
    touched += 1;
  }
  SpreadsheetApp.getUi().alert(touched + '行を計算どおりの順番に戻しました。');
}


/** 順番が未選択の行・フェーズ1の指摘と食い違う行を一覧で出す。 */
function checkAll() {
  var sh = mainSheet_();
  var n = sh.getLastRow() - 1;
  var vals = sh.getRange(2, 1, n, C.adj[SLOTS - 1]).getValues();
  var blank = [], mismatch = [], changed = 0, even = 0;
  for (var i = 0; i < n; i++) {
    var row = vals[i];
    var nTags = Number(row[C.nTags - 1]) || 0;
    var ord = String(row[C.order - 1] || '').trim();
    var def = String(row[C.defOrder - 1] || '').trim();
    var verdict = String(row[C.verdict - 1] || '').trim();
    if (nTags > 1 && !ord) blank.push(row[C.no - 1]);
    if (def && ord && ord !== def) changed += 1;
    if (ord === EVEN) even += 1;
    if (verdict === '×') mismatch.push(row[C.no - 1]);
  }
  var msg = '確認しました：' + n + '行\n\n'
          + '順番を変えた行: ' + changed + '行\n'
          + '「' + EVEN + '」にした行: ' + even + '行\n\n';
  msg += mismatch.length
      ? 'フェーズ1の指摘と食い違う行 — No: ' + mismatch.join(', ') + '\n\n'
      : 'フェーズ1の指摘との食い違いはありません。\n\n';
  msg += blank.length ? '順番が未選択の行 — No: ' + blank.join(', ')
                      : '順番はすべて選ばれています。';
  SpreadsheetApp.getUi().alert(msg);
}
