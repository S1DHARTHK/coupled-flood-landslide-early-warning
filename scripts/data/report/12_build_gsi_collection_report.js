// Build landslide_events/GSI_LANDSLIDE_COLLECTION_REPORT.docx   (needs: npm install docx)
// Usage: node 12_build_gsi_collection_report.js <output.docx>
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  HeadingLevel, BorderStyle, LevelFormat, Footer, PageNumber, TableLayoutType, VerticalAlign,
} = require("docx");

const FONT = "Calibri", NAVY = "1F3864", GREY = "595959", RULE = "BFBFBF";
const P_W = 11906, P_H = 16838, M = 1134, W = P_W - 2 * M;
const t = (s, o = {}) => new TextRun({ text: s, font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics, color: o.color });
const p = (runs, o = {}) => new Paragraph({ children: Array.isArray(runs) ? runs : [t(runs, o)], spacing: { after: o.after ?? 100, line: 264 } });
const h1 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 260, after: 110 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 2 } },
  children: [new TextRun({ text: s, font: FONT, size: 26, bold: true, color: NAVY })] });
const li = (runs, ref = "bul") => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 264 },
  children: Array.isArray(runs) ? runs : [t(runs)] });
const note = (s) => p([t(s, { size: 17, italics: true, color: GREY })], { after: 120 });
function table(head, rows, widths, o = {}) {
  const b = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const cell = (s, w, hd, fill, bold) => new TableCell({ width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.TOP,
    shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined, margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { after: 0, line: 240 }, children: [new TextRun({ text: String(s), font: FONT,
      size: o.size || 17, bold: hd || bold, color: hd ? "FFFFFF" : undefined })] })] });
  return new Table({ width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: widths,
    layout: TableLayoutType.FIXED, borders: { top: b, bottom: b, left: b, right: b, insideHorizontal: b, insideVertical: b },
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: head.map((h, i) => cell(h, widths[i], true, NAVY)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((v, i) =>
        cell(v, widths[i], false, i === 0 && o.firstBold ? "F2F2F2" : (o.fillCol === i ? o.fill : undefined), i === 0 && o.firstBold)) }))] });
}

const body = [
  p([t("GSI LANDSLIDE COLLECTION REPORT", { size: 34, bold: true, color: NAVY })], { after: 40 }),
  p([t("Official GSI landslide inventory for Kerala · updated_data/landslide_events/ · 12 September 2026", { size: 21, color: GREY })], { after: 180 }),
  table(["Item", "Result"], [
    ["Outcome", "NOT AVAILABLE — the official GSI Bhukosh landslide inventory could not be obtained from this environment. No new records were collected. The existing GSI data are unchanged."],
    ["Source (preferred)", "Geological Survey of India (GSI), Ministry of Mines — Bhukosh, GSI's official geoscience data portal"],
    ["Official URL", "https://bhukosh.gsi.gov.in/Bhukosh/Public (GSI states its landslide inventory is published on NGDR and Bhukosh for free download by registered users)"],
    ["Dataset obtained", "None"],
    ["Number of new records", "0"],
    ["Date / spatial coverage (new)", "Not applicable — nothing obtained"],
    ["Additional records vs existing", "0 — no records could be compared or added, so nothing was duplicated"],
    ["Existing data", "existing_collected_data/GSI_Kerala_Landslides_All.csv, _2013_2024.csv and _Extracted.xlsx — not modified (checksums unchanged)"],
  ], [2600, W - 2600], { firstBold: true, size: 18 }),

  h1("1. Access attempts (all official GSI channels)"),
  table(["Channel", "Result", "Meaning"], [
    ["GSI Bhukosh portal (bhukosh.gsi.gov.in)", "DNS resolves (144.24.99.164) but the connection never completes; the in-app browser cannot load it either", "Not reachable from this network — no page loads at all (not a login wall). Not bypassed."],
    ["GSI NGDR (ngdr.gsi.gov.in)", "Host name does not resolve", "Not reachable"],
    ["GSI Bhusanket inventory services (India_All_Landslided, GSI_Landslide_India, Landslide_Polygon)", "“Token Required” (error 499)", "Access-controlled. No token requested or bypassed."],
    ["GSI Bhusanket Landslidedata_1 (public)", "2 records nationally, 0 in Kerala", "User-submission layer, not the inventory; contains personal contact fields — not downloaded"],
    ["GSI Bhusanket state-wise landslide reports", "Index of individual GSI project reports (PDF)", "Narrative studies, not a structured inventory — not collected"],
    ["Open Government Data Platform, BHUKOSH catalogue (data.gov.in)", "Only 1 resource: Digital Seismotectonic Atlas of India", "GSI landslide inventory is not published there; the platform's search did not respond"],
  ], [3000, 3200, W - 6200], { firstBold: true, size: 16 }),
  note("Full log with URLs: landslide_events/GSI_access_attempts_2026-09-12.csv."),

  h1("2. Current GSI data in the project — limitations"),
  table(["Aspect", "Finding"], [
    ["Provenance", "Extracted from a 904-page GSI landslide report PDF (README sheet of the .xlsx), not an official structured download. The README itself warns that PDF extraction can contain formatting anomalies."],
    ["Event count", "3,378 records (All file); 2,396 in the 2013–2024 file (a strict subset). 3,269 distinct slide numbers; 109 repeated slide numbers (re-surveys / reactivations); 3,141 distinct coordinate pairs."],
    ["Dates", "Free-text History field. Exact date: 2,236 records · month and year only: 13 · year only: 621 · no date: 508. Only 64 distinct exact dates; 2,018 of the exact dates fall in 2018. One impossible date (2028); 11 dates in 2025 (outside the study window). Year column is derived and 760 are empty. 172 records carry a time of day inside the text."],
    ["Spatial", "Every record has latitude/longitude (6 decimals). 4 points fall outside Kerala and 50 lie in a different district from the District column (checked against simplified boundaries). Only 12 of 14 districts (none in Alappuzha or Kasaragod); Idukki holds 1,689 records (50 %)."],
    ["Attributes", "Slide name, road location, material (35 spelling variants) and movement type (78 variants). No size, trigger, damage, activity or separate date/time fields."],
    ["Effect on the landslide target", "97 exact-date landslide district-days in the 2012–2024 grid; only 7 fall in the test split."],
  ], [2300, W - 2300], { firstBold: true, size: 17 }),

  h1("3. What the official inventory would add"),
  p("GSI's inventory data model (visible in the schema of GSI's public Bhusanket layer) has separate fields for landslide date, landslide time, an exact-date flag, latitude/longitude, district, taluk and village, material, movement type, dimensions (length, width, depth, area, volume), geology, inducing factor, damage counts and initiation year. The actual contents of the Bhukosh Kerala download could not be seen, so the number of records, date coverage and attribute completeness remain unknown."),

  h1("4. Manual action required"),
  li("From a network where bhukosh.gsi.gov.in loads (it did not load from this environment — typically an Indian internet connection), open https://bhukosh.gsi.gov.in/Bhukosh/Public.", "num"),
  li("Register as an External User (free) and log in — GSI requires login for downloads.", "num"),
  li("In the map viewer, select the landslide inventory layer, restrict it to Kerala (state boundary or the 14 districts), add it to the download cart and download it (shapefile or equivalent).", "num"),
  li("Save the download unchanged in updated_data/landslide_events/gsi_bhukosh_source/ and record the download date.", "num"),
  li("Alternative official route: request the Kerala landslide inventory from GSI in writing (e.g. the GSI Chief Data Officer listed on data.gov.in, or the GSI Kerala State Unit, Thiruvananthapuram).", "num"),
  p([t("After the file is in place, it can be checked and compared with the existing records without changing them: ", { bold: true }),
    t("exact duplicates by slide number; otherwise matching by location (within about 100 m) and date; only records with a stated exact date would count toward day-level labels. No date or location would be inferred.")]),

  h1("5. Remaining limitations"),
  li("Until the Bhukosh inventory is obtained, the landslide target rests on the PDF-extracted GSI records (dominated by 2018) plus IMD exact-day records collected earlier."),
  li("The 621 year-only and 508 undated GSI records cannot be used for day-level labels, and no dates were assigned to them."),
  li("The test split remains very thin (7 GSI landslide district-days; 8 including IMD)."),
];

const doc = new Document({ creator: "Capstone data task", title: "GSI Landslide Collection Report",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] }] },
  sections: [{ properties: { page: { size: { width: P_W, height: P_H }, margin: { top: M, bottom: M, left: M, right: M } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "GSI Landslide Collection Report · 12 September 2026 · page ", font: FONT, size: 16, color: GREY }),
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY })] })] }) }, children: body }] });
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[2], b); console.log("written", process.argv[2], b.length); });
