// Build DATA_VERIFICATION_REPORT.docx from verification_report_data.json (10a_verification_report_data.py).
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, BorderStyle, LevelFormat, Footer, PageNumber, TableLayoutType, VerticalAlign,
} = require("docx");

const D = JSON.parse(fs.readFileSync(__dirname + "/verification_report_data.json", "utf8"));
const OUT = process.argv[2];
const FONT = "Calibri", NAVY = "1F3864", GREY = "595959", RULE = "BFBFBF";
const P_W = 11906, P_H = 16838, M = 1134, W = P_W - 2 * M;
const FILL = { "VERIFIED-CORRECT": "D9EAD3", "CORRECTED": "FFF2CC", "UNRESOLVED": "F4CCCC" };

const t = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics, color: o.color });
const p = (runs, o = {}) => new Paragraph({ children: Array.isArray(runs) ? runs : [t(runs, o)],
  spacing: { before: 0, after: o.after ?? 100, line: 264 } });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 280, after: 120 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 2 } },
  children: [new TextRun({ text, font: FONT, size: 26, bold: true, color: NAVY })] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 264 },
  children: Array.isArray(runs) ? runs : [t(runs)] });
const note = (text) => p([t(text, { size: 17, italics: true, color: GREY })], { after: 120 });

function table(header, rows, widths, o = {}) {
  const b = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const cell = (txt, w, head, fill, bold) => new TableCell({
    width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.TOP,
    shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { after: 0, line: 240 }, children: [new TextRun({ text: String(txt), font: FONT,
      size: o.size || 16, bold: head || bold, color: head ? "FFFFFF" : undefined })] })] });
  const rowsX = [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, widths[i], true, NAVY)) })];
  rows.forEach((r) => rowsX.push(new TableRow({ cantSplit: true, children: r.map((v, i) =>
    cell(v, widths[i], false, o.statusCol === i ? FILL[v] : (o.firstBold && i === 0 ? "F2F2F2" : undefined), o.firstBold && i === 0)) })));
  return new Table({ width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: widths,
    layout: TableLayoutType.FIXED, borders: { top: b, bottom: b, left: b, right: b, insideHorizontal: b, insideVertical: b }, rows: rowsX });
}
const c = D.counts, nV = c["VERIFIED-CORRECT"] || 0, nC = c["CORRECTED"] || 0, nU = c["UNRESOLVED"] || 0;

const body = [
  p([t("DATA VERIFICATION REPORT", { size: 36, bold: true, color: NAVY })], { after: 40 }),
  p([t("Final verification of the 18 flagged flood-event records · Kerala Coupled Flood–Landslide Early Warning System", { size: 22, color: GREY })], { after: 40 }),
  p([t("updated_data/  ·  12 September 2026", { size: 20, color: GREY })], { after: 200 }),
  table(["Field", "Detail"], [
    ["Records examined", "The 18 Kerala records of IFI-Impacts v3.0 (hazard_events/India_Flood_Inventory_v3.csv) whose start–end span disagrees with IFI's own Duration(Days) by more than one day — reproduced from the current file (608 Kerala rows), not taken from the earlier audit."],
    ["What the records contain", "UEI, start/end date, duration, district list, cause, casualties and damage text. All 18 give Event Source = IMD; none has latitude, longitude, location or an event-source ID, so districts are the only spatial information."],
    ["Authoritative source", "IFI-Impacts documentation (Zenodo 10.5281/zenodo.11275211; Saharia et al. 2021, Natural Hazards) states the events are sourced from the India Meteorological Department — digitised from IMD’s annual Disastrous Weather Events (DWE) reports. The DWE report of each year (official IMD Pune Library PDFs, unchanged, SHA-256 checked) is therefore the original record for every flagged event."],
    ["Method (independent)", "Each record's IMD entry was located by CONTENT, not by date: phrases and place names from IFI's own casualty/damage text were searched in the Kerala section of that year's report, counting only evidence found in exactly one IMD entry. The date IMD printed for that entry was then read. One record without descriptive text (2021-0078) was located by its position in IFI's IMD-derived sequence. No date was inferred or estimated."],
    ["Not done", "No Master Dataset built, no model trained, no ML code changed, no original file modified, no record removed."],
  ], [2300, W - 2300], { firstBold: true, size: 18 }),

  h1("Verification table"),
  table(["Record (UEI)", "Original date (IFI, DD-MM-YYYY)", "Verified / corrected date", "District / location", "Status", "Evidence source"],
    D.main, [1000, 1650, 1750, 1900, 1150, W - 7450], { statusCol: 4, firstBold: true, size: 15 }),
  note("Record = UEI-IMD-FL-<year>-<no>. Where IMD lists separate days (e.g. “11 & 17 Nov.”) the verified date lists those days; the range between them is not an event period. Status colours: green VERIFIED-CORRECT, yellow CORRECTED, red UNRESOLVED."),

  h1("Summary"),
  table(["Result", "Records"], [
    ["VERIFIED-CORRECT — existing date supported by IMD", String(nV)],
    ["CORRECTED — existing date contradicted by IMD; IMD date adopted", String(nC)],
    ["UNRESOLVED — insufficient authoritative evidence", String(nU)],
  ], [W - 1600, 1600], { firstBold: true, size: 18 }),
  p(""),
  p([t("Cause of every correction: ", { bold: true }), t("in all 14 corrected records IFI stored day and month the wrong way round (month-first), e.g. IFI “09-05-2017” is IMD’s 5 September 2017. No correction moved an event to a different event or merely to a “more plausible” date: each adopted date is the one printed in the IMD entry that contains the record’s own casualty and damage details.")]),

  h1("Evidence for each corrected record"),
  table(["Record", "Original", "Corrected", "Authoritative evidence (IMD date as printed)", "IFI text found only in that IMD entry"],
    D.corr, [950, 1650, 1700, 2600, W - 6900], { firstBold: true, size: 15 }),
  note("Every corrected record: source = India Meteorological Department, Disastrous Weather Events <year>; URLs https://www.imdpune.gov.in/library/public/DWE-2017_final.pdf, DWE-2018_final.pdf, DWE-2019.pdf, DWE-2020.pdf. Full source excerpts: flood_events/verification_18/excerpts/<UEI>.txt."),

  h1("Records verified correct"),
  table(["Record", "IMD date as printed", "How located", "Note"], D.ver, [1100, 1900, 1200, W - 4200], { firstBold: true, size: 16 }),
  note("These dates are correct. For 2019-0079, 2020-0045 and 2021-0078 the event consists of the listed days only; for 2018-0035 IMD reports a seasonal period (29 May – 31 Jul 2018) and IFI’s Duration field (34) is inconsistent with it."),

  h1("Sources used"),
  bullet([t("India Meteorological Department — Disastrous Weather Events 2017, 2018, 2019, 2020, 2021 ", { bold: true }), t("(original source; IMD Pune Library, https://www.imdpune.gov.in/library/publication.html).")]),
  bullet([t("IFI-Impacts v3.0 documentation ", { bold: true }), t("— Zenodo record 10.5281/zenodo.11275211 and Saharia et al. (2021), Natural Hazards, doi:10.1007/s11069-021-04698-6 — to establish that IMD DWE is IFI’s source.")]),
  bullet([t("KSDMA / Government of Kerala ", { bold: true }), t("— not needed: the original IMD source was available for every record, and KSDMA publishes narrative documents rather than a dated event archive.")]),

  h1("Limitations"),
  bullet("Evidence comes from the text layer of the official IMD PDFs; all 18 excerpts were read in full. The DWE 2017–2021 reports have a text layer (no OCR involved)."),
  bullet("2021-0078 has no casualty or damage text in IFI; it was located by sequence (IFI 2021-0076…0081 ↔ IMD entries 15…20, with matching attributes), which is strong but not a text match."),
  bullet("Verification covers the event dates. District lists were compared with IMD’s Area-affected column but not re-assigned; IFI’s district attribution is retained."),
  bullet("2018-0035 and 2018-0036 are IMD multi-week summary periods (64 and 30 days, all 14 districts). Their dates are now verified, but whether every day of such a period is a flood day is a labelling decision for Master Dataset construction."),

  h1("Audit trail and dataset to use"),
  bullet([t("Original (unchanged): ", { bold: true }), t("hazard_events/India_Flood_Inventory_v3.csv — SHA-256 identical to the 8 September manifest.")]),
  bullet([t("18-record working copy: ", { bold: true }), t("flood_events/verification_18/IFI_Kerala_flagged18_verified.csv — every original column unchanged + final_status, verified_start, verified_end, event_days, evidence and method.")]),
  bullet([t("Change log: ", { bold: true }), t(`flood_events/verification_18/CHANGE_LOG.csv — ${D.log_rows} entries (field, original value, corrected value, reason, source, URL, page, entry, IMD text).`)]),
  bullet([t("Consistency: ", { bold: true }), t(`this independent verification agrees with the full corrected copy for ${D.agree} of 18 records (dates) and ${D.days_equal} of 18 (event-day lists).`)]),
  bullet([t("Use for Master Dataset construction: ", { bold: true }), t("flood_events/processed/IFI_Kerala_2012_2023_corrected.csv (all 288 Kerala records 2012–2023, including these 18 and 27 further IMD-verified corrections) — use verified_start / verified_end / event_days, not the original Start/End columns — together with flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv for events after 2023-07-23.")]),
  bullet([t("Scripts: ", { bold: true }), t("updated_data/scripts/10_verify_flagged_18.py and 11_finalize_flagged_18.py reproduce this verification.")]),
];

const doc = new Document({
  creator: "Capstone data task", title: "Data Verification Report",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [{ properties: { page: { size: { width: P_W, height: P_H }, margin: { top: M, bottom: M, left: M, right: M } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "Data Verification Report · 12 September 2026 · page ", font: FONT, size: 16, color: GREY }),
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY })] })] }) },
    children: body }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("written", OUT, b.length); });
