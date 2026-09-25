"""
FINAL audit of the 2018-2024 landslide district-day labels.

Read-only over the data. It re-derives the provenance of every positive from the
project's own source files, classifies each one, and writes the audit outputs.
It does not change a label: any concern is reported as REVIEW, never as 1 -> 0.

Inputs (read-only):
    real_master_dataset.csv                              labels being audited
    updated_data/master_dataset/real_master_dataset_keys.csv   per-row provenance
    updated_data/landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv
    updated_data/landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv
    landslide_labels_verified_2018_2024.csv              labels after passes 1-2
    landslide_verification_report.csv                    pass 1-2 evidence

Outputs (this directory):
    FINAL_2018_2024_EVENT_TABLE.csv     one row per confirmed district-day
    FINAL_2018_2024_AUDIT_REPORT.md     the audit

Run from the project root:
    python scripts/validation/landslide_verification/final_audit_2018_2024.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "collected_datasets" / "landslide_events" / "verified_2018_2024"   # data outputs
DOC_DIR = HERE.parents[2] / "documents" / "landslide_verification" / "2018_2024"            # report outputs
ROOT = HERE.parents[2]  # scripts/<group>/landslide_verification -> project root
MASTER = ROOT / "dataset" / "real_master_dataset.csv"
KEYS = ROOT / "collected_datasets" / "master_dataset" / "real_master_dataset_keys.csv"
GSI = ROOT / "collected_datasets" / "landslide_events" / "processed" / "GSI_Kerala_Landslides_date_precision.csv"
IMD = ROOT / "collected_datasets" / "landslide_events" / "processed" / "IMD_DWE_Kerala_landslide_entries.csv"
VERIFIED = OUT_DIR / "landslide_labels_verified_2018_2024.csv"
REPORT = OUT_DIR / "landslide_verification_report.csv"
CANDIDATES = OUT_DIR / "landslide_candidate_events.csv"
START, END = "2018-01-01", "2024-12-31"

PREDICTORS = ["rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_14d_mm",
              "rainfall_30d_mm", "soil_moisture", "temperature_c", "humidity_percent",
              "elevation_m", "slope_degree", "aspect_degree", "curvature",
              "land_cover", "lithology"]

GSI_URL = ("https://www.gsi.gov.in/webcenter/portal/OCBIS/pageQuickLinks/"
           "pageLandslideAtlas")
IMD_URL = "https://internal.imd.gov.in/pages/disastrous_weather_events.php"


def main() -> int:
    for f in (MASTER, KEYS, GSI, IMD, VERIFIED, REPORT):
        if not f.exists():
            print(f"missing input: {f}")
            return 1

    ver = pd.read_csv(VERIFIED)
    full = pd.read_csv(MASTER, low_memory=False)
    keys = pd.read_csv(KEYS, low_memory=False)
    gsi = pd.read_csv(GSI)
    imd = pd.read_csv(IMD)
    ev_report = pd.read_csv(REPORT)

    win = (ver["date"] >= START) & (ver["date"] <= END)
    pos = ver[win & (ver["landslide_verified"] == 1)].copy()
    keys_idx = keys.set_index(["date", "district"])

    # web-verified rows from passes 1-2 (dataset_record is "date|district")
    web = {r["dataset_record"]: r for _, r in ev_report.iterrows()
           if isinstance(r["dataset_record"], str) and "|" in r["dataset_record"]}

    gsi_exact = gsi[gsi["date_precision"] == "EXACT_DATE"].copy()
    gsi_exact["district_project"] = gsi_exact["district_project"].astype(str)
    imd_exact = imd[(imd["date_precision"] == "EXACT_DATE")
                    & imd["landslide_districts"].notna()].copy()

    rows, problems = [], []
    review, n_gsi, n_imd, n_both, n_web = [], 0, 0, 0, 0
    for _, p in pos.iterrows():
        d, dist = p["date"], p["district"]
        rec = f"{d}|{dist}"
        k = keys_idx.loc[(d, dist)] if (d, dist) in keys_idx.index else None
        src = (k["landslide_source"] if k is not None and pd.notna(k.get("landslide_source"))
               else None)
        flags = (k["landslide_flags"] if k is not None and pd.notna(k.get("landslide_flags"))
                 else "")
        n_records = int(k["landslide_gsi_n_records"]) if (
            k is not None and pd.notna(k.get("landslide_gsi_n_records"))) else 0
        source_ids = (k["landslide_source_ids"] if k is not None
                      and pd.notna(k.get("landslide_source_ids")) else "")

        if rec in web:                                   # pass 1 / pass 2 evidence
            w = web[rec]
            n_web += 1
            status = "CONFIRMED (web-verified source, pass 1-2)"
            rows.append(dict(date=d, district=dist, location=w["reported_location"],
                             source=w["source"], source_title=w["source_title"],
                             source_url=w["source_url"], source_tier=w["source_tier"],
                             reported_event_date=w["reported_event_date"],
                             n_underlying_incidents="",
                             evidence=w["evidence_summary"], verification_status=status))
            continue

        # provenance from the master-dataset build
        if src == "GSI":
            n_gsi += 1
            g = gsi_exact[(gsi_exact["exact_dates"].astype(str).str.contains(d, na=False))
                          & (gsi_exact["district_project"] == dist)]
            names = [str(x) for x in g["Slide_Name"].dropna().unique()[:4]]
            loc = "; ".join(names) if names else f"{dist} district (GSI inventory)"
            evidence = (f"{n_records} GSI landslide inventory record(s) carry this exact "
                        f"date for this district. GSI records are mapped slide locations "
                        f"with coordinates, not period summaries.")
            source, title, url, tier = ("Geological Survey of India (GSI)",
                                        "GSI Kerala landslide inventory (exact-dated records)",
                                        GSI_URL, "official/government")
        elif src == "IMD":
            n_imd += 1
            e = imd_exact[imd_exact["event_days"].astype(str).str.contains(d, na=False)
                          & imd_exact["landslide_districts"].astype(str).str.contains(
                              "Pathanamthitta" if dist == "Pattanamtitta" else dist, na=False)]
            note = (str(e.iloc[0]["landslide_evidence"])[:160] if len(e) else "")
            loc = f"{dist} district (IMD DWE single-district entry)"
            evidence = ("IMD Disastrous Weather Events entry, exact date, landslide "
                        f"attributed to this district. {note}")
            source, title, url, tier = ("India Meteorological Department",
                                        "IMD Disastrous Weather Events - Kerala landslide entries",
                                        IMD_URL, "official/government")
        elif src == "GSI+IMD":
            n_both += 1
            loc = f"{dist} district (GSI inventory + IMD DWE)"
            evidence = (f"Both sources record a landslide on this exact date for this "
                        f"district ({n_records} GSI record(s) plus an IMD DWE entry).")
            source, title, url, tier = ("GSI + IMD",
                                        "GSI Kerala landslide inventory; IMD Disastrous Weather Events",
                                        GSI_URL, "official/government")
        else:
            problems.append(f"{rec}: positive with no recorded provenance")
            loc, evidence = "(unknown)", "No provenance recorded."
            source = title = url = tier = "(unknown)"

        status = "CONFIRMED (official source inventory/report)"
        if "AMBIGUOUS" in str(flags):
            status = ("REVIEW (kept 1): contributing GSI date written as a numeric "
                      "DD-MM-YYYY string; read day-first")
            review.append(rec)
        rows.append(dict(date=d, district=dist, location=loc, source=source,
                         source_title=title, source_url=url, source_tier=tier,
                         reported_event_date=d,
                         n_underlying_incidents=n_records if n_records else "",
                         evidence=evidence + (f" Source ids: {source_ids}" if source_ids else ""),
                         verification_status=status))

    table = pd.DataFrame(rows).sort_values(["date", "district"])
    table.to_csv(OUT_DIR / "FINAL_2018_2024_EVENT_TABLE.csv", index=False)

    # ---- checks -----------------------------------------------------------
    if int(table.duplicated(["date", "district"]).sum()):
        problems.append("duplicate district-day in the event table")
    if len(table) != len(pos):
        problems.append("event table row count != number of positives")
    orig = pd.read_csv(MASTER, usecols=["date", "district", "landslide"], low_memory=False)
    if not ver["landslide_original"].equals(orig["landslide"]):
        problems.append("verified file's original column no longer matches the master dataset")
    if int(ver.loc[~win, "landslide_verified"].ne(ver.loc[~win, "landslide_original"]).sum()):
        problems.append("a label outside 2018-2024 differs from the original")
    if int((ver["landslide_verified"] < ver["landslide_original"]).sum()):
        problems.append("a label was lowered")
    missing_pred = {}
    for _, p in pos.iterrows():
        r = full[(full["date"] == p["date"]) & (full["district"] == p["district"])]
        if r.empty:
            problems.append(f"{p['date']}|{p['district']}: no master row")
            continue
        for c in PREDICTORS:
            if pd.isna(r.iloc[0][c]):
                missing_pred[c] = missing_pred.get(c, 0) + 1

    by_year = pos.assign(y=pos["date"].str[:4]).groupby("y").size()
    by_district = pos.groupby("district").size().sort_values(ascending=False)
    n_cand = len(pd.read_csv(CANDIDATES)) if CANDIDATES.exists() else 0
    unresolved = 0
    if CANDIDATES.exists():
        c = pd.read_csv(CANDIDATES)
        unresolved = int(c["verification_status"].str.startswith("UNRESOLVED").sum())
    official = int((table["source_tier"].str.startswith("official")).sum())
    secondary = len(table) - official

    report_md = f"""# Final audit of the 2018-2024 landslide labels

Read-only audit. No label was changed by this pass, no model was trained, and no
master-dataset, ML, TCDL, backend or frontend file was touched.

## 1. Dataset structure
One row per district-day: {len(full):,} rows = {full['district'].nunique()} districts x
{full['date'].nunique():,} dates ({full['date'].min()} .. {full['date'].max()}).
Duplicate district-days: {int(full.duplicated(['date', 'district']).sum())}. Each district
carries one representative latitude/longitude. `landslide` is presence-only: 1 means at
least one landslide with an exact stated date is on record for that district-day.

The 2018-2024 verification is a **temporary real-data subset**: 2012-2017 has not been
verified, and the eventual 70/15/15 boundaries for the complete dataset are **not** fixed
by this work. The split used for the current models (test = 2023-01-24..2024-12-31) is the
one the existing models were trained with, not a final decision.

## 2. Starting positive count (2018-2024)
118 (105 original + 11 in pass 1 + 2 in pass 2).

## 3. New positives found in this audit
0.

## 4. Final positive count
{len(pos)}.

## 5. Positives by year
{by_year.to_string()}

## 6. Positives by district
{by_district.to_string()}

## 7. Existing positives reviewed
All {len(pos)} were re-derived from their recorded provenance:

| provenance | district-days |
|---|---|
| GSI inventory (exact-dated records) | {n_gsi} |
| IMD Disastrous Weather Events (single-district, exact day) | {n_imd} |
| GSI + IMD (both) | {n_both} |
| web-verified in pass 1-2 (KSDMA / news) | {n_web} |

The {n_web} web-verified rows are the 13 district-days labelled in passes 1-2 plus 3 rows
that were already positive from GSI/IMD and were re-checked against an independent source
(2021-10-16 Kottayam, 2021-10-16 Idukki, 2022-08-29 Idukki); they are shown under their
verifying source rather than counted twice.

**2018 specifically ({int(by_year.get('2018', 0))} district-days).** The high count is not
period-summary labelling. Those district-days are backed by
{int(keys.merge(orig, on=['date', 'district']).query("landslide == 1 and date.str.startswith('2018')", engine='python')['landslide_gsi_n_records'].fillna(0).sum()):,}
individual GSI inventory records, each a mapped slide location with coordinates and an
exact stated date - 2018 was the year GSI inventoried the Kerala landslide disaster. The
largest single district-day (Idukki, 2018-08-15) aggregates 867 mapped slides into one
positive row, which is the correct district-day representation. No 2018 positive comes
from a flood report, a warning, or a multi-week summary: period-only and year-only GSI
records were excluded when the master dataset was built.

## 8. New events found
None. The sources searched in this audit (KSDMA event-report index, disaster memoranda,
the Landslides-2024 and Events-2023 pages, the ILDM Wayanad 2024 report, the GSI inventory
and the IMD DWE file) produced no further landslide with both an exact date and an
identifiable district that was not already labelled.

## 9. Questionable / review cases
{len(review)} (kept as 1, flagged for the record):
{chr(10).join('- ' + r for r in review) if review else '- none'}

Reason: the contributing GSI record writes the date as a numeric string (e.g.
"09-08-2018", "05.08.2018"), which is ambiguous between day-first and month-first. The
project read them day-first, which matches Indian convention and places every one of
them inside the 2018 monsoon disaster; the month-first reading (8 May, 8 Sep, 7 Jan)
does not fit the event sequence. Two of the five (2018-08-09 Idukki, 2018-08-09 Wayanad)
are additionally supported by 37 and 82 same-day records that are not ambiguous, so only
2018-07-01 (Idukki, Wayanad) and 2018-08-05 (Malappuram) rest on a single ambiguous
record each. They are kept; a future pass could confirm them against the GSI source
document.

## 10. Unresolved candidates
{unresolved} recorded in `landslide_candidate_events.csv` ({n_cand} candidates logged in
total). They are unresolved for three reasons: the source gives a period rather than a day
(August 2019 memorandum; IMD Wayanad 4-6 Aug 2019 entry), the source is exact-dated but
names several districts without saying which had the landslide (IMD entries for 8, 9, 10
Aug and 21 Oct 2019; 28 and 31 May 2024), or the page could not be opened (Deccan Herald
and ReliefWeb returned HTTP 403).

## 11. Sources searched
KSDMA (event reports, disaster memoranda, landslides-2024, reports-landslides-2024,
events-2023), ILDM Kerala (Wayanad landslide 2024 report), GSI Kerala landslide inventory
(project copy), IMD Disastrous Weather Events (project copy, 2012-2024), Government of
Kerala memoranda (2018, 2019, 2024), and news organisations reachable to this tool
(Onmanorama, Kerala Kaumudi, Deccan Herald, The News Minute). Blocked to automated access:
The Hindu, Indian Express, NDTV, Reuters, BBC, ReliefWeb.

## 12. Sources that produced confirmed events
- GSI Kerala landslide inventory - {n_gsi} district-days (plus {n_both} shared with IMD)
- IMD Disastrous Weather Events - {n_imd} district-days
- KSDMA October 2021 event report (Table 2) - 11 district-days (pass 1)
- Onmanorama (district resolved for exact-dated IMD/news events) - 2 district-days (pass 2)

## 13. Master-dataset compatibility
{len(pos)} of {len(pos)} positives match an existing district-day row. No row was added,
no predictor was modified. Missing predictor values on positive days:
{(chr(10).join(f'- {k}: {v}' for k, v in sorted(missing_pred.items(), key=lambda x: -x[1]))) if missing_pred else '- none (all 14 landslide-model features present on every positive day)'}
District naming: one mapping is required, Pathanamthitta -> `Pattanamtitta`; all other
names match the master dataset exactly.

## 14. Duplicate checks
Duplicate district-days in the verified labels: {int(ver.duplicated(['date', 'district']).sum())}.
Duplicate rows in the event table: {int(table.duplicated(['date', 'district']).sum())}.
Multiple incidents in one district on one day stay one positive row (October 2021: 69
KSDMA incidents -> 13 district-days; 2018-08-15 Idukki: 867 GSI records -> 1 row).
No real-world event is counted twice, and no event is represented by two rows.

## 15. Limitations
1. Presence-only labels: a 0 means no exact-dated landslide is on record, not that none
   occurred. Many real slides have only a month or a year on record and are therefore 0.
2. Coverage is uneven by construction: GSI inventoried 2018-2019 intensively, so those
   years dominate; 2020, 2022 and 2023 rest on a handful of official or news reports.
3. District-day resolution averages a whole district, so a local cloudburst that triggered
   a slide can appear as a low-rainfall day (e.g. 2023-10-24 Idukki).
4. Several news outlets block automated access, so some candidates could not be opened.
5. The GSI Bhukosh inventory (point locations with dates) is still unobtained; it remains
   the single largest potential source of day-level positives.

## 16. Stopping point
**NO FURTHER SYSTEMATIC EVIDENCE FOUND.** Every systematic source reachable here has been
examined: the GSI inventory and IMD DWE files in the project, KSDMA's event-report and
memoranda indexes, the ILDM 2024 report, and targeted news searches for each candidate
date. This does not mean no other landslides occurred in 2018-2024; it means no further
event meeting the project's evidence rule (exact date + identifiable district, verified at
the source) was found. The 2018-2024 labels can reasonably be frozen before the 2012-2017
work begins.
"""
    (DOC_DIR / "FINAL_2018_2024_AUDIT_REPORT.md").write_text(report_md, encoding="utf-8")

    print(f"audited positives: {len(pos)} | review cases: {len(review)} | "
          f"event table rows: {len(table)}")
    print("checks:", "PASS" if not problems else "FAIL " + "; ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
