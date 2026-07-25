import fs from "node:fs";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const packageDir = String.raw`D:\29722\Desktop\GCC\提交相关材料\S9全量资料审计与网页仓库归档_20260725`;
const indexDir = path.join(packageDir, "00_材料索引");
const analysisDir = path.join(packageDir, "04_往届赛题与代码分析");
const outputPath = path.join(packageDir, "06_报告与验收清单", "S9外部链接与往届仓库审计表.xlsx");
const renderDir = path.join(packageDir, "06_报告与验收清单", "工作簿预览");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') {
        quoted = false;
      } else {
        field += c;
      }
    } else if (c === '"') {
      quoted = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += c;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const [headers, ...data] = rows.filter((r) => r.some((v) => v !== ""));
  return data.map((r) => Object.fromEntries(headers.map((h, i) => [h.replace(/^\uFEFF/, ""), r[i] ?? ""])));
}

function readCsv(filePath) {
  return parseCsv(fs.readFileSync(filePath, "utf8"));
}

function colName(index) {
  let n = index + 1;
  let out = "";
  while (n) {
    const r = (n - 1) % 26;
    out = String.fromCharCode(65 + r) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function setTitle(sheet, title, subtitle, columns) {
  const last = colName(columns - 1);
  sheet.getRange(`A1:${last}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A2:${last}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A1:${last}1`).format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF", size: 16 },
    rowHeight: 28,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${last}2`).format = {
    fill: "#DCE6F1",
    font: { color: "#1F1F1F", size: 10 },
    wrapText: true,
    rowHeight: 34,
    verticalAlignment: "center",
  };
  sheet.showGridLines = false;
}

function writeTable(sheet, startRow, headers, rows, widths) {
  const last = colName(headers.length - 1);
  sheet.getRange(`A${startRow}:${last}${startRow}`).values = [headers];
  sheet.getRange(`A${startRow}:${last}${startRow}`).format = {
    fill: "#4472C4",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    rowHeight: 26,
    borders: { preset: "outside", style: "thin", color: "#2F5597" },
  };
  if (rows.length) {
    sheet.getRange(`A${startRow + 1}:${last}${startRow + rows.length}`).values = rows;
    sheet.getRange(`A${startRow + 1}:${last}${startRow + rows.length}`).format = {
      font: { size: 9 },
      wrapText: true,
      verticalAlignment: "top",
      borders: {
        insideHorizontal: { style: "thin", color: "#E7E6E6" },
        bottom: { style: "thin", color: "#B4C6E7" },
      },
    };
  }
  widths.forEach((width, i) => {
    sheet.getRange(`${colName(i)}:${colName(i)}`).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(startRow);
}

const links = readCsv(path.join(indexDir, "外部链接总表_完整.csv"));
const cloneResults = readCsv(path.join(indexDir, "仓库拉取结果.csv"));
const webInitial = readCsv(path.join(indexDir, "网页获取结果.csv"));
const webSupplement = readCsv(path.join(indexDir, "网页获取结果_补充.csv"));
const codeMatrix = readCsv(path.join(analysisDir, "已拉取仓库代码审读矩阵.csv"));
const workbookIndex = JSON.parse(fs.readFileSync(path.join(analysisDir, "工作簿审读结果", "工作簿审读索引.json"), "utf8"));
const webMap = new Map([...webInitial, ...webSupplement].map((r) => [r.OriginalUrl, r]));
webMap.set("https://docs.pytorch.org/docs/2.5/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_", {
  Status: "Archived",
  HttpStatus: "200",
  ArchiveFile: "S046_torch.Tensor.index_add_.html",
  Title: "torch.Tensor.index_add_ — PyTorch 2.5 documentation",
  Message: "",
});

const wb = Workbook.create();

const summary = wb.worksheets.add("总览");
setTitle(summary, "S9 资料全量审计总览", "统计口径：截至 2026-07-25；所有链接均保留原始来源，成功项保存网页或仓库，失败项保留错误证据。", 5);
summary.getRange("A4:B10").values = [
  ["指标", "数量"],
  ["唯一外部链接", null],
  ["已保存网页", null],
  ["网页获取失败", null],
  ["仓库链接", null],
  ["仓库完整克隆成功", null],
  ["仓库克隆失败", null],
];
summary.getRange("B5").formulas = [["=COUNTA('外部链接'!A4:A300)"]];
summary.getRange("B6").formulas = [["=COUNTIF('外部链接'!F4:F300,\"Archived\")"]];
summary.getRange("B7").formulas = [["=COUNTIF('外部链接'!F4:F300,\"Failed\")"]];
summary.getRange("B8").formulas = [["=COUNTA('仓库拉取'!A4:A200)"]];
summary.getRange("B9").formulas = [["=COUNTIF('仓库拉取'!E4:E200,\"Cloned\")+COUNTIF('仓库拉取'!E4:E200,\"AlreadyPresent\")"]];
summary.getRange("B10").formulas = [["=COUNTIF('仓库拉取'!E4:E200,\"Failed\")"]];
summary.getRange("A4:B4").format = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
summary.getRange("A5:B10").format = { borders: { preset: "inside", style: "thin", color: "#D9E2F3" } };
summary.getRange("A12:E12").values = [["结论", "影响", "当前证据", "下一步", "优先级"]];
summary.getRange("A13:E17").values = [
  ["Concat 当前实现只覆盖 FP16 最后一维", "与 S9 任意维度/多 dtype 不符，解释 Case2–5 Run failed", "赛题工作簿 + 当前 host/kernel", "先建通用回退，再保留末维快路径", "P0"],
  ["Greater 未实现广播、特殊值与多 dtype", "隐藏用例高风险", "PyTorch 2.5 语义页 + 当前代码", "广播索引映射 + Compare 正确语义", "P0"],
  ["IndexAdd 忽略 self 且只支持 2D dim0 int8", "基础语义不成立", "index_add_ 官方说明 + 当前代码", "复制 self 后正确累加，处理重复 index", "P0"],
  ["Transpose 仅 2D 16×16 对齐交换", "任意排列/尾块/6D 均不支持", "赛题工作簿 + 当前代码", "通用坐标映射 + 2D 快路径", "P0"],
  ["SquareSumV1 仅末轴 keep_dims=true FP16", "多轴、dtype、shape 均不完整", "sum/square 官方说明 + 当前代码", "轴归一化、FP32 累加、多轴回退", "P0"],
];
summary.getRange("A12:E12").format = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
summary.getRange("A13:E17").format = { wrapText: true, verticalAlignment: "top", borders: { insideHorizontal: { style: "thin", color: "#E7E6E6" } } };
["A", "B", "C", "D", "E"].forEach((c, i) => summary.getRange(`${c}:${c}`).format.columnWidth = [27, 25, 28, 29, 10][i]);
summary.getRange("E13:E17").conditionalFormats.add("containsText", { text: "P0", format: { fill: "#F4CCCC", font: { bold: true, color: "#9C0006" } } });
summary.freezePanes.freezeRows(3);

const spec = wb.worksheets.add("S9规格与当前缺口");
setTitle(spec, "S9 五题规格与当前实现缺口", "规格来源：S9 工作簿、官方测试脚本、PyTorch 2.5 文档；当前实现来源：s9-work/submission-src。", 8);
const specRows = [
  ["Concat", "Tensor[] inputs；dim 默认0", "float32 / float16 / int32 / int8", "任意维数、任意合法 dim；除拼接轴外形状一致；允许长度0分片；非对齐", "仅 float16、只按最后一维计算；地址仅16个；未校验 inputCount；host 与 kernel 对 dim 的理解不一致", "Case1 10.948；Case2–5 Run failed", "通用 N-D fallback + last-axis fast path；0长度输入；动态输入数；尾块", "P0"],
  ["Greater", "self, other", "float32 / bfloat16 / float16 / int32 / int8；输出bool", "广播；inf/-inf/NaN；任意形状", "仅 FP16 等形；输出 shape 直接等于 self；整张量进 UB；SubRelu 代替比较破坏特殊值/溢出语义", "隐藏用例未验证", "广播步长映射；Compare；分块；多 dtype", "P0"],
  ["IndexAdd", "self, index(int32), source；dim默认0", "float32 / bfloat16 / float16 / int32 / int8", "index 必须1D；source[dim]=len(index)；其他维与 self 一致；重复 index 累加；输出从 self 开始", "忽略 self；仅 int8、2D、dim0；source/out 复制对齐未处理；空 index 导致 blockDim=0 风险", "隐藏用例未验证", "self 初始化；重复 index 正确性；任意 dim/rank；多 dtype；冲突策略", "P0"],
  ["Transpose", "input；dims=list_int", "float32 / float16 / int32 / int8", "最高6D任意排列；非对齐尺寸", "仅 FP16 2D 交换；忽略 dims；仅16倍数；小于16时 totalTiles=0；尾块缺失", "隐藏用例未验证", "通用坐标映射；尾块 DataCopyPad；2D 16×16 快路径", "P0"],
  ["SquareSumV1", "input；axis=list_int；keep_dims默认false", "float16 / bfloat16 / float32", "单轴或多轴；负轴；keep_dims 两种输出形状；非对齐", "仅 FP16 最后一轴；shape 强制保留末轴；忽略 axis/keep_dims；先 FP16 平方后 FP32 累加", "隐藏用例未验证", "轴归一化；FP32 中间值；多轴分阶段/通用回退；输出形状正确", "P0"],
];
writeTable(spec, 3, ["算子", "接口", "数据类型", "完整语义/边界", "当前实现缺口", "线上结果", "建议架构", "优先级"], specRows, [14, 22, 25, 42, 48, 22, 38, 10]);
spec.getRange("H4:H8").conditionalFormats.add("containsText", { text: "P0", format: { fill: "#F4CCCC", font: { bold: true, color: "#9C0006" } } });

const linkSheet = wb.worksheets.add("外部链接");
setTitle(linkSheet, "外部链接逐项审计", "共 167 个唯一链接；“Archived”表示已保存本地响应，“Failed”表示保留了错误信息。仓库源码另见“仓库拉取”。", 9);
const linkRows = links.map((r) => {
  const w = webMap.get(r.OriginalUrl) ?? {};
  return [r.OriginalUrl, r.Sources, r.Category, r.DiscoveryMethods, w.HttpStatus ?? "", w.Status ?? "NotFetchedAsPage", w.ArchiveFile ?? "", w.Title ?? "", w.Message ?? ""];
});
writeTable(linkSheet, 3, ["原始链接", "来源文件", "类别", "发现方式", "HTTP", "状态", "归档文件", "页面标题", "说明/错误"], linkRows, [52, 32, 20, 28, 9, 16, 22, 36, 50]);
linkSheet.getRange(`F4:F${3 + linkRows.length}`).conditionalFormats.add("containsText", { text: "Archived", format: { fill: "#E2F0D9", font: { color: "#375623" } } });
linkSheet.getRange(`F4:F${3 + linkRows.length}`).conditionalFormats.add("containsText", { text: "Failed", format: { fill: "#FCE4D6", font: { color: "#9C0006" } } });

const repoSheet = wb.worksheets.add("仓库拉取");
setTitle(repoSheet, "往届冠军仓库拉取结果", "完整 clone，保留 .git 历史；成功项记录默认分支、HEAD 与提交日期。失败项主要为 Gitee 平台拒绝、GitCode 登录要求或仓库已删除。", 10);
const repoRows = cloneResults.map((r) => [r.Index, r.OriginalUrl, r.CloneUrl, r.Directory, r.Status, r.Branch, r.Head, r.CommitDate, r.Seconds, r.Message]);
writeTable(repoSheet, 3, ["编号", "原始链接", "规范 clone URL", "本地目录", "状态", "分支", "HEAD", "提交日期", "耗时(s)", "错误/说明"], repoRows, [9, 45, 48, 38, 16, 12, 42, 24, 12, 58]);
repoSheet.getRange(`E4:E${3 + repoRows.length}`).conditionalFormats.add("containsText", { text: "Cloned", format: { fill: "#E2F0D9", font: { color: "#375623" } } });
repoSheet.getRange(`E4:E${3 + repoRows.length}`).conditionalFormats.add("containsText", { text: "Failed", format: { fill: "#FCE4D6", font: { color: "#9C0006" } } });

const codeSheet = wb.worksheets.add("往届代码审读");
setTitle(codeSheet, "已拉取往届代码审读矩阵", "43 个仓库、1396 个索引文件、401 个 C/C++ 源文件。计数用于发现技术模式，不等同于质量评分。", 15);
const codeRows = codeMatrix.map((r) => [
  r.Index, r.CloneUrl, r.OperatorHints, Number(r.CodeFiles || 0), Number(r.DataCopyPad || 0), Number(r.DataCopyExtParams || 0),
  Number(r.BufferNum || 0), Number(r.Template || 0), Number(r.Tiling || 0), Number(r.ReduceSum || 0),
  Number(r.WholeReduceSum || 0), Number(r.Gather || 0), Number(r.Scatter || 0), Number(r.Transpose || 0), r.READMEFirstLine,
]);
writeTable(codeSheet, 3, ["编号", "仓库", "算子线索", "代码文件", "DataCopyPad", "ExtParams", "双缓冲线索", "模板", "Tiling", "ReduceSum", "WholeReduceSum", "Gather", "Scatter", "Transpose", "README首行"], codeRows, [9, 48, 52, 12, 14, 12, 14, 11, 11, 14, 17, 11, 11, 12, 38]);
for (const c of ["E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]) {
  codeSheet.getRange(`${c}4:${c}${3 + codeRows.length}`).conditionalFormats.add("dataBar", { color: "#5B9BD5" });
}

const hist = wb.worksheets.add("历届工作簿链接");
setTitle(hist, "历届赛题工作簿与语义链接", "9 个工作簿全部通过 artifact-tool 读取，共提取 40 个唯一技术文档链接；对应网页已逐项归档或记录失败。", 3);
const histRows = workbookIndex.flatMap((item) => item.urls.map((url) => [item.fileName, url, (webMap.get(url)?.Status ?? "Archived in supplemental batch")]));
writeTable(hist, 3, ["工作簿", "技术文档链接", "归档状态"], histRows, [42, 76, 22]);

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.mkdirSync(renderDir, { recursive: true });
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(outputPath);

const sheets = ["总览", "S9规格与当前缺口", "外部链接", "仓库拉取", "往届代码审读", "历届工作簿链接"];
for (const sheetName of sheets) {
  const preview = await wb.render({ sheetName, autoCrop: "all", scale: 0.8, format: "png" });
  const safe = sheetName.replace(/[\\/:*?"<>|]/g, "_");
  fs.writeFileSync(path.join(renderDir, `${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const inspected = await wb.inspect({ kind: "workbook,sheet,formula", maxChars: 10000, tableMaxRows: 8, tableMaxCols: 8 });
fs.writeFileSync(path.join(renderDir, "工作簿结构检查.ndjson"), inspected.ndjson, "utf8");
console.log(JSON.stringify({ outputPath, sheets, links: links.length, repos: cloneResults.length, codeRepos: codeMatrix.length }, null, 2));
