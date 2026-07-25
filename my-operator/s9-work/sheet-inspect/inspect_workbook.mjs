import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const inputPath = process.argv[2];
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table,region",
  include: "values,formulas",
  maxChars: 30000,
  tableMaxRows: 100,
  tableMaxCols: 30,
  tableMaxCellChars: 300,
});

console.log(overview.ndjson);
