const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, BorderStyle, PageOrientation, LevelFormat, Footer, PageNumber,
  TableLayoutType, VerticalAlign,
} = require("docx");

const D = JSON.parse(fs.readFileSync(__dirname + "/report_data.json", "utf8"));
const OUT = process.argv[2];

const FONT = "Calibri";
const NAVY = "1F3864", GREY = "595959", RULE = "BFBFBF";
const P_W = 11906, P_H = 16838, M = 1134;
const W_PORT = P_W - 2 * M, W_LAND = P_H - 2 * M;
const STATUS_FILL = {
  "COLLECTED": "D9EAD3", "PARTIALLY COLLECTED": "FFF2CC", "NOT AVAILABLE": "EDEDED", "UNRESOLVED": "F4CCCC",
};

const t = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics, color: o.color });
const p = (runs, o = {}) => new Paragraph({
  children: Array.isArray(runs) ? runs : [t(runs, o)],
  spacing: { before: o.before ?? 0, after: o.after ?? 100, line: 264 },
  alignment: o.align, keepNext: o.keepNext,
});
const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, keepNext: true,
  spacing: { before: 300, after: 120 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 2 } },
  children: [new TextRun({ text, font: FONT, size: 26, bold: true, color: NAVY })],
});
const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 200, after: 80 },
  children: [new TextRun({ text, font: FONT, size: 22, bold: true, color: NAVY })],
});
const bullet = (runs) => new Paragraph({
  numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 264 },
  children: Array.isArray(runs) ? runs : [t(runs)],
});
const note = (text) => p([t(text, { size: 17, italics: true, color: GREY })], { after: 120 });

function table(header, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const fs_ = o.size || 17;
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border };
  const cell = (txt, w, isHead, fill, bold) => new TableCell({
    width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.TOP,
    shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: String(txt).split("\n").map((line) => new Paragraph({
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: line, font: FONT, size: fs_, bold: isHead || bold, color: isHead ? "FFFFFF" : undefined })],
    })),
  });
  const trs = [new TableRow({ tableHeader: true, cantSplit: true,
    children: header.map((h, i) => cell(h, widths[i], true, NAVY)) })];
  rows.forEach((r) => trs.push(new TableRow({ cantSplit: true, children: r.map((v, i) => {
    let fill; if (o.statusCol === i) { const k = Object.keys(STATUS_FILL).find((s) => String(v).startsWith(s)); fill = k && STATUS_FILL[k]; }
    if (o.firstBold && i === 0) return cell(v, widths[i], false, o.firstFill || "F2F2F2", true);
    return cell(v, widths[i], false, fill);
  }) })));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, borders, rows: trs });
}
const kv = (rows, w) => table(["Field", "Detail"], rows, [Math.round(w * 0.24), w - Math.round(w * 0.24)], { firstBold: true, size: 18 });
const gap = () => p("", { after: 60 });

// ------------------------------------------------------------------ numbers
const fsc = D.flood_status_counts;
const nCorr = (fsc.CORRECTED_DAY_MONTH_SWAP || 0) + (fsc.CORRECTED_WRONG_MONTH || 0) + (fsc.CORRECTED_INCOMPLETE || 0);
const byYear = Object.entries(D.flood_corr_by_year).map(([y, n]) => `${y}: ${n}`).join(" · ");
const cov = Object.fromEntries(D.river_split_cov);
const flCov = D.flood_cov, lsCov = D.ls_cov;
const ext = D.flood_ext.filter((r) => r[0].startsWith("Extension"));

// ------------------------------------------------------------------ portrait part 1
const part1 = [
  p([t("DATA SOURCE AND CORRECTION REPORT", { size: 36, bold: true, color: NAVY })], { after: 40 }),
  p([t("Kerala Coupled Flood–Landslide Early Warning System", { size: 24, color: GREY })], { after: 40 }),
  p([t("updated_data/  ·  11 September 2026", { size: 20, color: GREY })], { after: 200 }),
  kv([
    ["Purpose", "Obtain missing real data and verify/correct existing real data from authoritative sources so that the project data are scientifically usable: river_level_m, flood-event dates, flood-label coverage, landslide events, lithology."],
    ["Training period (from code)", "ml/flood_xgboost.py chronological_split(): 70 / 15 / 15 on unique dates. On the only real daily grid that can be built (2012-01-30 → 2024-12-31, after the 29-day rainfall_30d_mm warm-up): train 2012-01-30 → 2021-02-14 · validation 2021-02-15 → 2023-01-23 · test 2023-01-24 → 2024-12-31."],
    ["Not done", "No model trained. No Master Dataset built. No synthetic, estimated, interpolated or randomly generated value. No ML code or feature definition changed. No existing file modified or deleted — every correction is a separate, labelled file."],
    ["Reproducibility", "Every processed file is regenerated by updated_data/scripts/01–08. Downloads, sizes and SHA-256 checksums: documentation/download_log.tsv and documentation/SOURCE_MANIFEST_2026-09-11.json."],
    ["Final status", "All 22 Master Dataset columns now have a verified real source covering the training period. The real Master Dataset can be constructed, subject to four documented construction decisions (Section 9)."],
  ], W_PORT),

  h1("Status at a glance"),
  table(["Dataset / item", "Status", "Authoritative source", "Result"], [
    ["river_level_m — observed river stage", "PARTIALLY COLLECTED", "Central Water Commission (CWC), via National Water Data Portal",
      `${D.river_n_gauges} CWC gauges; 11 of 14 districts have training-period data (4 at ≥ 97 %, 7 from June 2015). Observed on ${cov.train.split(" district")[0].replace(" of", " of")} training district-days.`],
    ["river_level_m — Alappuzha, Wayanad, Kottayam (training)", "NOT AVAILABLE", "—", "No CWC gauge in Alappuzha or Wayanad; the only Kottayam gauge starts in 2019."],
    ["Flood-event date verification (IFI-Impacts v3.0)", "COLLECTED", "India Meteorological Department — Disastrous Weather Events 2012–2023",
      `${D.flood_n_records} of ${D.flood_n_records} Kerala records verified; ${nCorr} corrected with evidence; 0 unresolved.`],
    ["Flood-event dates — unresolved records", "NONE (0 UNRESOLVED)", "—", "Every record was matched to its IMD source entry."],
    ["Flood-label coverage after 2023-07-23", "COLLECTED", "IMD Disastrous Weather Events 2023, 2024", `${ext.length} additional Kerala events (2023-10-14 → 2024-10-25); label source now reaches 2024-12-31.`],
    ["Additional landslide events", "PARTIALLY COLLECTED", "IMD Disastrous Weather Events 2012–2024",
      `${D.imd_ls_n} IMD entries report a landslide; ${D.imd_ls_attr_exact} are tied to one district and one day; 23 district-days not in GSI.`],
    ["GSI national landslide inventory (Bhukosh)", "NOT AVAILABLE", "Geological Survey of India", "Host unreachable; GSI web services require a token (not bypassed)."],
    ["lithology — CGWB principal-aquifer rock type", "COLLECTED", "Central Ground Water Board, via National Water Data Portal", `Kept. Independently reproduced: identical in ${D.lith_same}/14 districts.`],
    ["GSI 1:50,000 geology (better lithology)", "NOT AVAILABLE", "Geological Survey of India (Bhukosh)", "Host unreachable from this environment; no public GSI geology service."],
  ], [2600, 1500, 2300, W_PORT - 6400], { statusCol: 1, firstBold: true }),
  note("Status colours: green = COLLECTED, yellow = PARTIALLY COLLECTED, grey = NOT AVAILABLE, red = UNRESOLVED."),

  h1("1. River-level dataset collected"),
  p(`The CWC series of observed river water level for Kerala's basin ("West flowing rivers from Tadri to Kanyakumari") was found on the National Water Data Portal and downloaded in full: 731,816 Kerala readings for 1991–2020 and 792,543 for 2021–2025, from ${D.river_n_gauges} CWC gauges in Kerala. The earlier collection report stated that the CWC national dataset did not cover Kerala's basin; it does.`),
  p("Processing (scripts/01_river_level_cwc_process.py) uses observed values only — no interpolation, gap filling, discharge-to-stage or datum conversion:"),
  bullet("Readings are aggregated to daily mean, max, min and the 08:00 reading per gauge (128,571 gauge-days)."),
  bullet("Datum QC. CWC publishes some readings as the gauge reading and others as gauge reading + RL of gauge zero (verified: conflicting pairs at Vandiperiyar differ by exactly its RL, 789.0 m). Readings on a station's minority reference are removed, never converted."),
  bullet("Reference breaks (month-to-month jumps > 5 m) split a series into datum-consistent segments; short bracketed blocks on the other reference are removed. In total 19,373 minority-reference readings, 187 readings in RL-offset runs, 587 readings in bracketed blocks and 2,198 conflicting duplicate readings were removed (of 1.52 million Kerala readings)."),
  bullet("District attribution uses CWC's own LGD district code; reservoir-level stations are excluded."),
  p("Outputs: river_level/processed/CWC_Kerala_RiverLevel_Daily_Station.csv, CWC_Kerala_station_inventory.csv, CWC_Kerala_datum_breaks.csv, District_Gauge_Assignment.csv. Originals: river_level/cwc_nwdp_source/ (unchanged)."),

  h1("2. Official source of river-level data"),
  kv([
    ["Dataset", "River water level, CWC hydrological observation stations, Kerala"],
    ["Source Organization", "Central Water Commission (CWC), Ministry of Jal Shakti, Government of India; published by the National Water Informatics Centre (NWIC)"],
    ["Official Dataset Name", "River Water Level (Manual - Hourly), Central Water Commission (CWC) — basin “West flowing rivers from Tadri to Kanyakumari” (NWDP dataset code WQT00004)"],
    ["Official Source", "National Water Data Portal (NWDP), Government of India"],
    ["URL", "https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-cwc-subernarekha\nFiles: rwl_manual_hr_cwc_031_1991_2020.csv (259.7 MB), rwl_manual_hr_cwc_031_2021_2025.csv (234.0 MB)"],
    ["Access Date", "11 September 2026"],
    ["Coverage", "31 gauges in Kerala; Kerala readings 1991-01-01 → 2025-12-31. 5 gauges continuous 2012–2024; 10 more from mid-2015"],
    ["Resolution", "Point gauges; manual readings (typically 08:00, 13:00, 18:00; hourly during floods); aggregated to daily per gauge"],
    ["Variables Used", "Station, District LGD Code, River, Latitude, Longitude, RL_of_zeroGauge, Data Acquisition Time, River Water Level Manual Hourly (meter)"],
    ["Units", "Metres (portal metadata “Unit = m”); coordinates in decimal degrees (WGS84)"],
    ["Why It Is Suitable", "Observed river stage in metres from the national authority named first in the brief, with station identity, coordinates and timestamps, covering the whole training window. Portal: Data Access Control = Public; licence “Other (Open)”. No registration or access control involved."],
    ["Limitations", "Point gauges represent one river reach, not a whole district. Gauges are on different datums, so absolute stage is not comparable across districts. 7 river gauges keep unexplained reference breaks (series split, not used across the break). No gauge in Alappuzha or Wayanad."],
  ], W_PORT),
  note("India-WRIS (the second preferred source) did not respond from this environment on any attempt; it was not needed once the CWC series was obtained. A Kerala Surface Water Department manual series (2022–2025, 6 stations) was also downloaded but has no training-period data and is not used."),

  h1("3. River stations and coverage"),
  p("Gauge-to-district rule: a gauge may represent a district only if CWC attributes it to that district (LGD code), it measures river stage (not a reservoir), and it has a datum-consistent record covering at least 40 % of training days. Where several qualify, the one with the longest training record is the primary gauge; the others are listed. A point gauge is not assumed to represent the whole district — this is recorded as a limitation, not hidden."),
  table(["District", "Status", "Primary CWC gauge (river)", "Lat, Lon", "Train data from", "Train", "Val", "Test", "Other usable gauges"],
    D.river_districts, [1250, 1350, 1700, 1100, 950, 650, 650, 650, W_PORT - 8300], { size: 15, firstBold: true }),
  note("Train / Val / Test = % of days in each model split with an observed reading at the primary gauge."),
  p([t("Observed river_level_m via primary gauges: ", { bold: true }),
    t(`train ${cov.train} · validation ${cov.validation} · test ${cov.test}. Before this task the training split had 0 observed values.`)]),
  p([t("Physical check: ", { bold: true }), t("10 of the 11 primary gauges reach their 2018 maximum on 15–17 August 2018 — the documented peak of the 2018 Kerala flood; Perumannu (Kannur) peaks on 8 August 2018, the first flood spell in the north.")]),
  p([t("River gauges with remaining reference breaks: ", { bold: true }), t(D.river_breaks.map(([s, b]) => `${s[0] + s.slice(1).toLowerCase()} (${b.replace(/;/g, ", ")})`).join("; ") + ". Reservoir-level stations (Idamalayar, Idukki Arch) are excluded as they are not river stage.")]),
];

// ------------------------------------------------------------------ landscape: corrections + extension
const part2 = [
  h1("4. Flood-event date corrections"),
  p(`IFI-Impacts v3.0 cites the India Meteorological Department as the source of all 608 of its Kerala records. Every Kerala record starting 2012–2023 (${D.flood_n_records}) was therefore checked against the IMD Disastrous Weather Events (DWE) report of the same year, reading each date four ways (as given, start swapped, end swapped, both swapped) and accepting only the reading that matches an IMD entry with the same district(s). The 2012 report is a scanned image and was read visually; 2014 and 2016 entries that could not be machine-parsed were read manually.`),
  table(["Result", "Records"], [
    ["Verified — dates unchanged", String(fsc.VERIFIED_NO_CHANGE)],
    ["Corrected — day and month swapped (IFI stored the date month-first)", String(fsc.CORRECTED_DAY_MONTH_SWAP)],
    ["Corrected — wrong month (2012-0054: IFI November, IMD August)", String(fsc.CORRECTED_WRONG_MONTH)],
    ["Corrected — day missing (2021-0073: IMD “18 and 19 May”)", String(fsc.CORRECTED_INCOMPLETE)],
    ["Verified to week precision only (2014-0050: IMD “1st week of Aug.”)", String(fsc.VERIFIED_APPROXIMATE)],
    ["Unresolved", "0"],
  ], [W_LAND - 1600, 1600], { size: 18 }),
  gap(),
  p([t("Key finding. ", { bold: true }), t(`The error is systematic, not limited to the 18 records flagged in the audit. In 2017–2020 IFI stores many dates month-first when the day is 12 or less, including one-day events whose duration looks correct (e.g. IFI 06-12-2019 is 12 June per IMD). Of the 18 audit-flagged records, 14 were corrected and 4 are correct as given (3 IMD non-contiguous day lists, 1 IMD 64-day seasonal period); ${nCorr - 14} further records were corrected. Corrections by year: ${byYear}.`)]),
  p("The table lists every record whose dates changed or whose IMD event days are non-contiguous (expansion must use the listed days, not the start–end range). * = one of the 18 records flagged by the audit. Original dates are shown as written in IFI (DD-MM-YYYY); corrected dates as YYYY-MM-DD."),
  table(["Original record (UEI)", "IFI districts", "Original date", "Corrected date / event days", "Evidence / source", "Status"],
    D.flood_corr_rows, [1350, 3000, 1900, 2350, 3700, W_LAND - 12300], { size: 15, firstBold: true }),
  note("Corrected working copy: flood_events/processed/IFI_Kerala_2012_2023_corrected.csv (all original IFI columns unchanged + verified_start, verified_end, event_days, verification_status, evidence, flags). Correction table: IFI_Kerala_date_corrections.csv. The original IFI file is unchanged."),

  h1("5. Additional flood-event data collected"),
  kv([
    ["Dataset", "Disastrous Weather Events — annual official record of weather disasters, by state"],
    ["Source Organization", "India Meteorological Department (IMD), Climate Research & Services, Pune — Ministry of Earth Sciences, Government of India"],
    ["Official Dataset Name", "“Disastrous Weather Events 2012” … “Disastrous Weather Events 2024” (13 annual reports)"],
    ["Official Source / URL", "IMD Pune Library publications — https://www.imdpune.gov.in/library/publication.html (files under /library/public/)"],
    ["Access Date", "11 September 2026"],
    ["Coverage", "All India 2012–2024; Kerala section of the Floods and Heavy Rains tables"],
    ["Resolution", "Event level — day, list of days or period; district names"],
    ["Variables Used", "Date/Period, Area affected (districts), Intensity, Casualties, Extent of damage"],
    ["Units", "Dates; district names"],
    ["Why It Is Suitable", "It is the original source of every Kerala IFI record, so extending with it keeps the same authority and inclusion rule. DWE 2024 covers the whole calendar year, so labels can be defined to 2024-12-31. IFI-Impacts v3.0 (Zenodo 10.5281/zenodo.11275211) is the latest release and ends in 2023."],
    ["Limitations", "Records damaging events reported to IMD; absence of an entry does not prove absence of flooding. Some entries give periods rather than days. KSDMA publishes only narrative documents, not a dated event archive."],
  ], W_LAND),
  gap(),
  p(`Events found in IMD DWE but not in IFI (${D.flood_ext.length}): ${ext.length} after IFI's last Kerala record (2023-07-23) and 1 inside IFI's period that IFI omitted. Districts are read from the Area-affected column only; no event already in IFI is duplicated.`),
  table(["Category", "Evidence", "IMD date", "Districts", "IMD intensity"],
    D.flood_ext, [2200, 2600, 1500, W_LAND - 8800, 2500], { size: 15 }),
  gap(),
  h2("Flood-label coverage (positive district-days; counts only — no labels were written)"),
  table(["Label basis", "Positive district-days", "Rate on 66,080 grid", "Train", "Validation", "Test", "Label source ends"],
    flCov, [W_LAND - 8400, 1500, 1500, 1000, 1100, 1000, 2300], { size: 16, firstBold: true }),
  note("The published count (5,764) was inflated by the date errors. The test split previously had flood labels only to 2023-07-23; with the IMD extension it is labelled to 2024-12-31 (28 → 119 positive district-days). The sensitivity row shows the effect of the 4 multi-week IMD summary periods (e.g. 29 May – 31 Jul 2018)."),
];

// ------------------------------------------------------------------ portrait part 3
const part3 = [
  h1("6. Additional landslide data collected"),
  h2("GSI inventory — date precision (existing file, re-classified, no dates invented)"),
  table(["Precision", "GSI 2013–2024 file", "GSI all-records file"], [
    ["EXACT_DATE (day, month, year)", String(D.gsi_prec_2013.EXACT_DATE || 0), String(D.gsi_prec_all.EXACT_DATE || 0)],
    ["MONTH_YEAR (month and year only)", String(D.gsi_prec_2013.MONTH_YEAR || 0), String(D.gsi_prec_all.MONTH_YEAR || 0)],
    ["YEAR_ONLY", String(D.gsi_prec_2013.YEAR_ONLY || 0), String(D.gsi_prec_all.YEAR_ONLY || 0)],
    ["NO_DATE", String(D.gsi_prec_2013.NO_DATE || 0), String(D.gsi_prec_all.NO_DATE || 0)],
  ], [W_PORT - 3600, 1800, 1800], { size: 18, firstBold: true }),
  note(`${D.gsi_time} records carry a time of day; ${D.gsi_ambig} numeric dates are day/month-ambiguous (read day-first and flagged). Output: landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv.`),
  h2("IMD Disastrous Weather Events — dated landslide occurrences (new)"),
  p(`${D.imd_ls_n} Kerala entries in IMD DWE 2012–2024 report a landslide. ${D.imd_ls_attr} can be tied to one district; ${D.imd_ls_attr_exact} of those to an exact day. Multi-district entries do not state where the landslide occurred and are kept unattributed; multi-day entries are recorded as PERIOD, not as exact dates. A text-based district attribution for multi-district entries was tested and rejected as unreliable.`),
  table(["Year", "Evidence", "IMD date", "Landslide district"], D.imd_ls_exact, [700, 2900, 2000, W_PORT - 5600], { size: 15 }),
  gap(),
  h2("Landslide-label coverage (exact-day district-days; counts only)"),
  table(["Basis", "District-days", "Dates", "Train", "Val", "Test"], lsCov, [W_PORT - 4400, 1000, 800, 900, 800, 900], { size: 16, firstBold: true }),
  note("GSI's national landslide inventory on Bhukosh could not be obtained: the host did not respond, and GSI's web services for the national inventory return “Token Required”. Access control was not bypassed. Manual step: register on GSI Bhukosh and download the Kerala landslide inventory. KSDMA publishes narrative reports (memoranda, PDNA) without a structured dated inventory."),

  h1("7. Lithology source and current status"),
  p([t("Status: COLLECTED (kept) — CGWB. ", { bold: true }), t("The existing lithology table was derived from the Central Ground Water Board Major Principal Aquifer map (NWDP). Its derivation script had not been kept, so it was independently reproduced from the original shapefile (scripts/07_lithology_cgwb_reproduce.py):" )]),
  bullet(`Dominant class identical in ${D.lith_same} of 14 districts; class fractions within ${D.lith_maxdiff.toFixed(3)}; district coverage identical (99.3–100 %).`),
  bullet("All values are inside the model's fixed CATEGORY_SCHEMA; the mapping of CGWB rock names to the six model levels is the one documented in lithology/STATUS_COLLECTED.md."),
  p([t("GSI geology: NOT AVAILABLE. ", { bold: true }), t("GSI Bhukosh and NGDR (1:50,000 geology) did not respond on any attempt; GSI's public GIS server exposes no geology layer. Nothing was replaced and no geological class was forced.")]),
  p([t("Limitation (unchanged): ", { bold: true }), t("at district scale lithology takes only 2 values (granite_gneiss in 13 districts, alluvium in Alappuzha); shale and sandstone do not occur in Kerala. A finer spatial unit, not a different source, is what would make this feature informative.")]),

  h1("8. Remaining unresolved issues"),
  bullet([t("River stage, 3 districts: ", { bold: true }), t("no observed training-period stage for Alappuzha, Wayanad (no CWC gauge) or Kottayam (gauge from 2019). These rows stay missing — the code passes NaN to XGBoost; nothing may be imputed.")]),
  bullet([t("River stage, partial record: ", { bold: true }), t("7 districts have gauges only from June 2015 (monsoon-season-only until 2017).")]),
  bullet([t("Stage comparability: ", { bold: true }), t("gauges are on different datums (e.g. Vandiperiyar ~791 m MSL, Ayilam ~1.7 m gauge reading). One river_level_m column pooled across districts mixes these references — a Master Dataset construction decision.")]),
  bullet([t("Flood-label policy: ", { bold: true }), t(`records flagged LONG_PERIOD_SUMMARY (${D.flood_flag_counts.LONG_PERIOD_SUMMARY}), LANDSLIDE_ONLY_CAUSE (${D.flood_flag_counts.LANDSLIDE_ONLY_CAUSE}) and NO_DISTRICT (${D.flood_flag_counts.NO_DISTRICT}) need an explicit include/exclude rule.`)]),
  bullet([t("Landslide target remains sparse: ", { bold: true }), t(`${lsCov[2][1]} exact-day district-days in total, only ${lsCov[2][5]} in the test split. This limits evaluation; it cannot be fixed by more processing.`)]),
  bullet([t("GSI data not obtainable here: ", { bold: true }), t("national landslide inventory and 1:50,000 geology require manual Bhukosh registration and download from a network that can reach the host.")]),
  bullet([t("Stale earlier documents (not edited): ", { bold: true }), t("SOURCE_MANIFEST.json (2026-09-08), river_level/STATUS_PARTIAL.md and the three 8 September reports predate this task; the new status files and SOURCE_MANIFEST_2026-09-11.json supersede them.")]),

  h1("9. Final data-readiness status"),
  table(["Component", "Status", "Basis"], [
    ["Rainfall, weather, soil moisture, terrain, land cover, spatial keys", "COLLECTED", "Unchanged from the previous collection; all 14 districts, daily 2012–2024"],
    ["river_level_m", "PARTIALLY COLLECTED", `CWC observed stage; ${cov.train.split(" district")[0]} training district-days; 3 districts without training data`],
    ["distance_to_river_km, drainage_density", "COLLECTED", "HydroRIVERS (derivable / derived)"],
    ["lithology", "COLLECTED", "CGWB, independently reproduced; near-constant at district scale"],
    ["flood (target)", "COLLECTED", `IFI v3.0 verified and corrected against IMD (0 unresolved) + IMD 2023–2024 extension; labelled to 2024-12-31`],
    ["landslide (target)", "PARTIALLY COLLECTED", "GSI exact-date records + IMD exact-day records; sparse (test split: 8 district-days)"],
  ], [3000, 1700, W_PORT - 4700], { statusCol: 1, firstBold: true, size: 17 }),
  gap(),
  p([t("Verdict: READY TO CONSTRUCT THE REAL MASTER DATASET — with four construction decisions to document first. ", { bold: true, color: NAVY }),
    t("No required column is missing any more, and the flood target is now verified against its source and labelled across the whole test period. Decisions for the construction step: (1) how river_level_m represents each district (primary gauge as published, or a per-gauge relative stage — the feature name stays unchanged); (2) the rule for the flagged flood records; (3) the landslide label rule (exact-date records only; GSI + IMD); (4) keep missing river stage as missing. The landslide target will remain statistically weak for evaluation regardless of construction choices.")]),

  h1("Appendix A. Data quality checks"),
  p(`documentation/QUALITY_CHECKS.csv (scripts/08_quality_checks.py): ${D.qc_counts.PASS} checks PASS, ${D.qc_counts.WARN || 0} WARN, 0 FAIL. All ${D.qc_integrity} downloaded originals are byte-identical to the download (size and SHA-256). Checked for every new dataset: file integrity, required columns, date range, spatial coverage, temporal resolution, units, missing values, duplicates, coordinate system, compatibility with the model and overlap with the train / validation / test periods.`),
  table(["Dataset", "Check", "Result (WARN)"], D.qc_warn, [2800, 2800, W_PORT - 5600], { size: 16 }),
  h1("Appendix B. Sources checked but not obtained"),
  table(["Source", "Outcome", "Manual step"], [
    ["India-WRIS (indiawris.gov.in)", "Host did not respond (timeout)", "Not needed — CWC data obtained via NWDP"],
    ["GSI Bhukosh / NGDR", "Hosts did not respond (timeout)", "Register at bhukosh.gsi.gov.in; download Kerala 1:50,000 lithology and landslide inventory"],
    ["GSI Bhusanket national landslide services", "“Token Required” — access-controlled; not bypassed", "Request access from GSI, or use the Bhukosh download"],
    ["GSI Bhusanket Landslidedata_1 (public)", "Only 2 records nationally, none in Kerala; contains personal contact fields", "None — not relevant, not downloaded"],
    ["KSDMA (sdma.kerala.gov.in)", "Narrative PDFs only; no structured dated event archive", "None — IMD DWE covers the same events"],
    ["IFI-Impacts newer version", "v3.0 is the latest release (ends 2023)", "None"],
  ], [2600, 3400, W_PORT - 6000], { size: 16, firstBold: true }),
];

const numbering = { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 360, hanging: 240 } }, run: { font: FONT } } }] }] };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
  new TextRun({ text: "Data Source and Correction Report · 11 September 2026 · page ", font: FONT, size: 16, color: GREY }),
  new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY })] })] });
const port = { page: { size: { width: P_W, height: P_H }, margin: { top: M, bottom: M, left: M, right: M } } };
const land = { page: { size: { width: P_W, height: P_H, orientation: PageOrientation.LANDSCAPE }, margin: { top: M, bottom: M, left: M, right: M } } };

const doc = new Document({
  creator: "Capstone data task", title: "Data Source and Correction Report",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering,
  sections: [
    { properties: port, footers: { default: footer }, children: part1 },
    { properties: land, footers: { default: footer }, children: part2 },
    { properties: port, footers: { default: footer }, children: part3 },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("written", OUT, b.length); });
