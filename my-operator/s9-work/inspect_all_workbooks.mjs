import fs from "node:fs";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookDir = process.argv[2];
const outputDir = process.argv[3];
if (!workbookDir || !outputDir) {
  throw new Error("Usage: node inspect_all_workbooks.mjs <workbook-dir> <output-dir>");
}

fs.mkdirSync(outputDir, { recursive: true });
const files = fs.readdirSync(workbookDir)
  .filter((name) => name.toLowerCase().endsWith(".xlsx"))
  .sort();

const index = [];
for (const fileName of files) {
  const inputPath = path.join(workbookDir, fileName);
  const input = await FileBlob.load(inputPath);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 200000,
    tableMaxRows: 100,
    tableMaxCols: 30,
    tableMaxCellChars: 1000,
  });
  const ndjson = inspection.ndjson;
  const outputPath = path.join(outputDir, `${path.basename(fileName, ".xlsx")}.ndjson`);
  fs.writeFileSync(outputPath, ndjson, "utf8");
  const urls = [...new Set(
    (ndjson.match(/https?:\/\/[^\s"\\]+/g) ?? [])
      .map((url) => url.replace(/[),.;\]}]+$/g, ""))
  )].sort();
  index.push({
    fileName,
    inputPath,
    inspectionFile: outputPath,
    urls,
  });
}

const indexPath = path.join(outputDir, "工作簿审读索引.json");
fs.writeFileSync(indexPath, JSON.stringify(index, null, 2), "utf8");
console.log(JSON.stringify({
  workbooks: index.length,
  workbooksWithUrls: index.filter((item) => item.urls.length > 0).length,
  totalUrlOccurrences: index.reduce((sum, item) => sum + item.urls.length, 0),
  indexPath,
}, null, 2));
