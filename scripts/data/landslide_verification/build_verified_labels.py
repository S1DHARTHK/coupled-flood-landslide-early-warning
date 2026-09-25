"""
Build a VERIFIED copy of the landslide district-day labels for 2018-2024.

Nothing is inferred or generated here. Every label change below comes from a
source that was opened and read (see EVIDENCE); the script only writes those
verified decisions out beside the originals.

Inputs (read-only, never modified):
    real_master_dataset.csv          -- existing district-day landslide labels

Outputs (this directory):
    landslide_labels_verified_2018_2024.csv   original + verified label per row
    landslide_verification_report.csv         one row per investigated record
    landslide_candidate_events.csv            every candidate examined + outcome
    VERIFICATION_SUMMARY.md                   counts, sources, open items
    MASTER_DATASET_COMPATIBILITY.md           do the labels fit the master dataset

Pass 1 (2026-09-25): 11 labels from the KSDMA October 2021 event report.
Pass 2 (2026-09-25): 2 labels for 2023 (the first verified 2023 positives),
    found by taking the IMD DWE exact-dated entries that the master-dataset
    build had to drop for naming several districts without saying which had
    the landslide, and resolving the district from news reporting.

Run from the project root:
    python scripts/data/landslide_verification/build_verified_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "collected_datasets" / "landslide_events" / "verified_2018_2024"   # data outputs
DOC_DIR = HERE.parents[2] / "documents" / "landslide_verification" / "2018_2024"            # report outputs
ROOT = HERE.parents[2]          # scripts/data/landslide_verification -> project root
MASTER = ROOT / "dataset" / "real_master_dataset.csv"
START, END = "2018-01-01", "2024-12-31"

# ---------------------------------------------------------------------------
# Verified evidence.
#
# Each entry was read at the URL given (not from a search snippet) and states
# an actual landslide, with the district and the exact date printed in the
# source. `district` uses the CHIRPS spelling of the dataset
# (Pathanamthitta -> Pattanamtitta).
#
# KSDMA Oct-2021: Table 2 "REPORTED LANDSLIDE EVENTS IN KERALA BETWEEN
# 11-10-2021 to 25-10-2021" of the official Kerala State Disaster Management
# Authority event report (source stated in the table: District Emergency
# Operations Centres). The table lists 69 incidents with taluk, village,
# incident location and date; they collapse to the district-days below.
# ---------------------------------------------------------------------------
KSDMA_2021 = {
    "source": "Kerala State Disaster Management Authority (KSDMA)",
    "source_title": ('Event Report on "Extreme Rainfall over Kerala" October 2021, '
                     "Table 2: List of reported landslide events occurred between "
                     "11/10/2021 and 26/10/2021"),
    "source_url": ("https://sdma.kerala.gov.in/wp-content/uploads/2022/05/"
                   "Event-report_October-2021.pdf"),
    "source_tier": "official/government",
}
ONMANORAMA_2022 = {
    "source": "Onmanorama (Malayala Manorama)",
    "source_title": "5 of a family die as landslide buries Thodupuzha house",
    "source_url": ("https://www.onmanorama.com/news/kerala/2022/08/29/"
                   "thodupuzha-landslide-family-trapped-under-debris-one-death.html"),
    "source_tier": "reputable secondary",
}

ONMANORAMA_2023_10 = {
    "source": "Onmanorama (Malayala Manorama)",
    "source_title": ("Heavy rain wreaks havoc in Kerala; landslide in Pachadi, "
                     "Kallar dam shutter raised"),
    "source_url": ("https://www.onmanorama.com/news/kerala/2023/10/25/"
                   "rain-thunderstorm-landslide-flood-idukki-chavara-kollam.html"),
    "source_tier": "reputable secondary",
}
ONMANORAMA_2023_11 = {
    "source": "Onmanorama (Malayala Manorama)",
    "source_title": ("Idukki on high alert after landslides wreak havoc; "
                     "2 shutters of Kallar Dam opened"),
    "source_url": ("https://www.onmanorama.com/news/kerala/2023/11/06/"
                   "idukki-high-alert-landslides-shutters-kallar-dam-opened.html"),
    "source_tier": "reputable secondary",
}
# Supporting (not counted as a separate event): the IMD Disastrous Weather
# Events entry for 5 Nov 2023 is exact-dated and reports a landslide, but names
# three districts without saying which one, so the master-dataset build could
# not use it. The news report above resolves the district as Idukki.
IMD_DWE_2023 = {
    "source": "India Meteorological Department, Disastrous Weather Events 2023",
    "source_title": "DWE 2023 p.134 entry KERALA6 (5 Nov. 2023)",
    "source_url": ("https://internal.imd.gov.in/press_release/"
                   "20240318_pr_3159.pdf"),
    "source_tier": "official/government (supporting)",
}

EVIDENCE: list[dict] = [
    # --- KSDMA October 2021 event report -----------------------------------
    dict(date="2021-10-12", district="Palakkad", reported_event_date="12-10-2021",
         reported_location="Attapadi road, Mannarkadu taluk, Palakkad",
         evidence="KSDMA table lists a reported landslide event at Attapadi road "
                  "(Mannarkadu taluk) on 12-10-2021.", **KSDMA_2021),
    dict(date="2021-10-16", district="Kollam", reported_event_date="16-10-2021",
         reported_location="Karavoor, Piravanthoor village, Pathanapuram taluk, Kollam",
         evidence="KSDMA table lists a reported landslide event at Karavoor "
                  "(Piravanthoor, Pathanapuram taluk) on 16-10-2021.", **KSDMA_2021),
    dict(date="2021-10-17", district="Palakkad", reported_event_date="17-10-2021",
         reported_location="Vellinezhi, Ottapalam taluk, Palakkad",
         evidence="KSDMA table lists a reported landslide event at Vellinezhi "
                  "(Ottapalam taluk) on 17-10-2021.", **KSDMA_2021),
    dict(date="2021-10-18", district="Palakkad", reported_event_date="18-10-2021",
         reported_location="Between Parambikulam and Thoonkadavu, Chittoor taluk, Palakkad",
         evidence="KSDMA table lists a reported landslide event between Parambikulam "
                  "and Thoonkadavu (Chittoor taluk) on 18-10-2021.", **KSDMA_2021),
    dict(date="2021-10-20", district="Palakkad", reported_event_date="20-10-2021",
         reported_location="Odathodu-Padagithodu road, Kizhakancheri, Alathur taluk, Palakkad",
         evidence="KSDMA table lists a reported landslide event at the odathodu-"
                  "padagithodu road (Kizhakancheri, Alathur taluk) on 20-10-2021.",
         **KSDMA_2021),
    dict(date="2021-10-20", district="Kottayam", reported_event_date="20-10-2021",
         reported_location="Mangalagiri 36 acre, Teekoy village, Meenachil taluk, Kottayam",
         evidence="KSDMA table lists a reported landslide event at Teekoy "
                  "(Meenachil taluk) on 20-10-2021.", **KSDMA_2021),
    dict(date="2021-10-20", district="Malappuram", reported_event_date="20-10-2021",
         reported_location="Mattarakal, Araku prambu, Perinthalmanna taluk, Malappuram",
         evidence="KSDMA table lists a reported landslide event at Araku prambu / "
                  "mattarakal (Perinthalmanna taluk) on 20-10-2021.", **KSDMA_2021),
    dict(date="2021-10-20", district="Idukki", reported_event_date="20-10-2021",
         reported_location="Ancham mile, Manakandam, Devikulam taluk, Idukki",
         evidence="KSDMA table lists a reported landslide event at Ancham mile "
                  "(Manakandam, Devikulam taluk) on 20-10-2021.", **KSDMA_2021),
    dict(date="2021-10-20", district="Wayanad", reported_event_date="20-10-2021",
         reported_location="Chena Mala, Kalpetta, Vythiri taluk, Wayanad",
         evidence="KSDMA table lists a reported landslide event at Chena Mala "
                  "(Kalpetta, Vythiri taluk) on 20-10-2021.", **KSDMA_2021),
    dict(date="2021-10-23", district="Pattanamtitta", reported_event_date="23-10-2021",
         reported_location=("Kottaman para (Seethathode, Konni taluk) and Kurumban muzhi "
                            "(Naranipuram, Ranni taluk), Pathanamthitta"),
         evidence="KSDMA table lists two reported landslide events in Pathanamthitta "
                  "district on 23-10-2021 (one district-day).", **KSDMA_2021),
    dict(date="2021-10-25", district="Pattanamtitta", reported_event_date="25-10-2021",
         reported_location="Kurumban muzhi, Naranipuram, Ranni taluk, Pathanamthitta",
         evidence="KSDMA table lists a reported landslide event at Kurumban muzhi "
                  "(Naranipuram, Ranni taluk) on 25-10-2021.", **KSDMA_2021),

    # --- pass 2: first verified 2023 positives (both in the test period) ---
    dict(date="2023-10-24", district="Idukki",
         reported_event_date="Tuesday 24 October 2023",
         reported_location="Pachadi (Third camp), Nedumkandam, Idukki",
         evidence=('Article of 25 Oct 2023: "A landslide was reported in Idukki\'s '
                   'Nedumkandam after Third camp in Pachadi received heavy rain for '
                   'more than 3 hours on Tuesday." An acre of farmland was destroyed '
                   "and 25 families were evacuated. The day is the Tuesday the article "
                   "names, i.e. 24 Oct 2023 (verified as a Tuesday); it is not inferred "
                   "from the publication date alone."),
         **ONMANORAMA_2023_10),
    dict(date="2023-11-05", district="Idukki",
         reported_event_date="Sunday night, 5 November 2023",
         reported_location=("Cheriyar-Dalam, Santhanpara panchayat; also Udumbanchola "
                            "and Pambadumpara, Idukki"),
         evidence=("Article of 6 Nov 2023: three landslides in the Cheriyar-Dalam "
                   "region of Santhanpara 'on Sunday night', with further slides at "
                   "Santhanpara, Udumbanchola and Pambadumpara; one death at Cheriyar "
                   "and 25 hectares of farmland destroyed. The Sunday before publication "
                   "is 5 Nov 2023 (verified as a Sunday). Independently supported by the "
                   "IMD DWE 2023 entry dated 5 Nov. which reports a landslide for a "
                   "district group including Idukki."),
         **ONMANORAMA_2023_11),
]

# Records already labelled 1 that this pass re-checked against an independent
# source. No label changes; recorded for traceability.
CROSS_CHECKED: list[dict] = [
    dict(date="2021-10-16", district="Kottayam", reported_event_date="16-10-2021",
         reported_location=("Koottickal, Edakunnam, Mundakkayam, Erumely North, Koruthodu "
                            "(Kanjirappally taluk) and Meenachil taluk, Kottayam"),
         evidence="KSDMA reports about 23 landslide events in Kanjirappally taluk on "
                  "16 October 2021, Koottickal worst affected, 13 bodies recovered.",
         **KSDMA_2021),
    dict(date="2021-10-16", district="Idukki", reported_event_date="16-10-2021",
         reported_location="Kokkayar (Mackochi) and Peruvandhanam, Peerumadu taluk, Idukki",
         evidence="KSDMA reports about 23 landslides in Peerumedu taluk on 16 October "
                  "2021; the major slide at Mackochi, Kokkayar killed 7 people.",
         **KSDMA_2021),
    dict(date="2022-08-29", district="Idukki", reported_event_date="29 August 2022",
         reported_location="Kudayathoor, near Thodupuzha, Idukki",
         evidence="Landslide at about 4.15 am on 29 August 2022 buried a house at "
                  "Kudayathoor; five members of one family died.", **ONMANORAMA_2022),
]

# Investigated but NOT labelled: evidence exists for the period, not for a day.
UNRESOLVED: list[dict] = [
    dict(date="2024-05-28", district="Kottayam (claimed)",
         reported_event_date="28 May 2024",
         reported_location="Kottayam district",
         evidence="A news headline reports a landslide in Kottayam during the 28 May "
                  "2024 rain spell, and the IMD DWE 2024 entry for 28 May is exact-dated "
                  "but names five districts without saying which had the landslide. The "
                  "article could not be opened (HTTP 403), so the district could not be "
                  "verified at the source. Not labelled.",
         source="Deccan Herald",
         source_title=("Incessant rains disrupt life in Kerala, submerge Kochi roads "
                       "(inaccessible: HTTP 403)"),
         source_url=("https://www.deccanherald.com/india/kerala/"
                     "incessant-rains-disrupt-life-in-kerala-submerge-kochi-roads-3041372"),
         source_tier="reputable secondary (not accessible)"),
    dict(date="2024-08-05", district="(not determined)",
         reported_event_date="situation report dated 5 August 2024",
         reported_location="Kerala",
         evidence="A situation report titled 'Landslide in Kerala (August 05, 2024)' "
                  "exists, but the page returned HTTP 403 and could not be read. It is "
                  "probably a status update on the 30 July 2024 Wayanad disaster rather "
                  "than a separate 5 August landslide; unverified either way, so no "
                  "label was set.",
         source="ReliefWeb",
         source_title="LOCAL Situation Report 011-2024 - Landslide in Kerala (August 05, 2024)",
         source_url=("https://reliefweb.int/report/india/"
                     "local-situation-report-011-2024-landslide-kerala-august-05-2024"),
         source_tier="secondary aggregator (not accessible)"),
    dict(date="2019-08-08; 2019-08-09; 2019-08-10; 2019-10-21",
         district="(multi-district IMD entries)",
         reported_event_date="exact dates, districts not attributable",
         reported_location="8-14 districts per entry",
         evidence="IMD DWE entries for these days are exact-dated and report landslides, "
                  "but list many districts without saying which had the landslide, so "
                  "they cannot be turned into district-days. Districts already labelled 1 "
                  "on 8-10 Aug 2019 come from the GSI inventory, not from these entries.",
         source="India Meteorological Department, Disastrous Weather Events",
         source_title="IMD DWE Kerala landslide entries (processed in this project)",
         source_url=("https://internal.imd.gov.in/pages/"
                     "disastrous_weather_events.php"),
         source_tier="official/government"),
    dict(date="2019-08-04..2019-08-06", district="Wayanad",
         reported_event_date="4 to 6 Aug. 2019 (period)",
         reported_location="Wayanad",
         evidence="IMD DWE single-district landslide entry for Wayanad covering a "
                  "three-day period; the source does not state which day the landslide "
                  "occurred, so no day can be labelled without inventing one.",
         source="India Meteorological Department, Disastrous Weather Events",
         source_title="IMD DWE Kerala landslide entries (processed in this project)",
         source_url=("https://internal.imd.gov.in/pages/"
                     "disastrous_weather_events.php"),
         source_tier="official/government"),
    dict(date="2019-08-01..2019-08-31", district="(13 districts, notified)",
         reported_event_date="1-31 August 2019",
         reported_location="Kerala, 1038 notified flood/landslide affected villages",
         evidence="Official memorandum covers floods AND landslides for August 2019 as "
                  "one period and lists affected villages, but gives no per-incident "
                  "dates. Cannot be matched to a specific district-day without "
                  "inventing a date.",
         source="Kerala State Disaster Management Authority (KSDMA)",
         source_title="Memorandum - Kerala Floods & Landslides 2019 (1-31 August 2019)",
         source_url=("https://sdma.kerala.gov.in/wp-content/uploads/2020/03/"
                     "Memorandum-pages-deleted-Copy-compressed.pdf"),
         source_tier="official/government"),
    dict(date="2020-08-06..2020-08-10", district="Kozhikode; Malappuram; Kottayam",
         reported_event_date="August 2020 (no day given per place)",
         reported_location="Kakkayam (Kozhikode), Nilambur (Malappuram), Kottayam",
         evidence="Secondary source states landslides also occurred at these places "
                  "during the August 2020 event but gives no dates per place. Pettimudi "
                  "(Idukki, 6 Aug 2020) is already labelled 1.",
         source="Wikipedia (tertiary; used only to locate candidates)",
         source_title="2020 Kerala floods",
         source_url="https://en.wikipedia.org/wiki/2020_Kerala_floods",
         source_tier="tertiary"),
    dict(date="2023 (rest of year)", district="(all)",
         reported_event_date="n/a",
         reported_location="Kerala",
         evidence="Beyond the two Idukki days verified in pass 2 (24 Oct and 5 Nov "
                  "2023), no further 2023 landslide with an exact date and district was "
                  "found. KSDMA published no extreme-rainfall event report for 2023 (the "
                  "2023 events index lists training activities only) and the GSI "
                  "inventory holds no exact-dated 2023 record, so day-level 2023 "
                  "coverage rests on news reporting. Absence of evidence is NOT evidence "
                  "that no other landslide occurred.",
         source="Kerala State Disaster Management Authority (KSDMA)",
         source_title="Events 2023 index (no 2023 rainfall/landslide event report)",
         source_url="https://sdma.kerala.gov.in/events-2023/",
         source_tier="official/government"),
]

# Candidates examined and rejected outright (recorded so the pass is auditable).
REJECTED: list[dict] = [
    dict(date="2025-05-28", district="Wayanad",
         reported_event_date="28 May 2025",
         reported_location="Karimattom Hill, near Mundakkai-Chooralmala, Wayanad",
         evidence="A real landslide, but it occurred in 2025 - outside the 2018-2024 "
                  "window and outside the master dataset, which ends 2024-12-31. Not used.",
         source="Kerala Kaumudi",
         source_title=("Landslide at Karimattom Hill near Mundakkai-Chooralmala region; "
                       "government found out only when debris reached river"),
         source_url=("https://keralakaumudi.com/en/news/news.php?id=1548933&u="
                     "landslide-at-karimattom-hill-near-mundakkai--chooralmala-region-"
                     "government-found-out-only-when-debris-reached-river-1548933"),
         source_tier="reputable secondary",
         status="REJECTED (outside 2018-2024)"),
]

REPORT_COLUMNS = ["dataset_record", "date", "district", "latitude", "longitude",
                  "original_label", "verified_label", "source", "source_title",
                  "source_url", "source_tier", "reported_event_date",
                  "reported_location", "evidence_summary", "status"]


def main() -> int:
    if not MASTER.exists():
        print(f"missing input: {MASTER}")
        return 1
    master = pd.read_csv(MASTER, usecols=["date", "district", "latitude", "longitude",
                                          "landslide"])

    # ---- verified label column (original preserved beside it) -------------
    out = master.rename(columns={"landslide": "landslide_original"}).copy()
    out["landslide_verified"] = out["landslide_original"]
    out["label_changed"] = False
    out["verification_source"] = ""
    out["verification_source_url"] = ""

    key = out["date"] + "|" + out["district"]
    changes, problems = [], []
    for ev in EVIDENCE:
        mask = key == f"{ev['date']}|{ev['district']}"
        if not mask.any():
            problems.append(f"no dataset row for {ev['date']} {ev['district']}")
            continue
        if not (START <= ev["date"] <= END):
            problems.append(f"{ev['date']} is outside {START}..{END}")
            continue
        row = out.loc[mask].iloc[0]
        if int(row["landslide_original"]) == 1:
            continue                      # already positive: cross-check only
        out.loc[mask, ["landslide_verified", "label_changed",
                       "verification_source", "verification_source_url"]] = [
            1, True, ev["source"], ev["source_url"]]
        changes.append(ev)

    # ---- verification report ---------------------------------------------
    rows = []
    for ev, status in ([(e, "CONFIRMED (0 -> 1)") for e in changes]
                       + [(e, "CONFIRMED (already 1; cross-checked)") for e in CROSS_CHECKED]):
        m = master[(master["date"] == ev["date"]) & (master["district"] == ev["district"])]
        r = m.iloc[0]
        rows.append({
            "dataset_record": f"{ev['date']}|{ev['district']}",
            "date": ev["date"], "district": ev["district"],
            "latitude": r["latitude"], "longitude": r["longitude"],
            "original_label": int(r["landslide"]),
            "verified_label": 1,
            "source": ev["source"], "source_title": ev["source_title"],
            "source_url": ev["source_url"], "source_tier": ev["source_tier"],
            "reported_event_date": ev["reported_event_date"],
            "reported_location": ev["reported_location"],
            "evidence_summary": ev["evidence"], "status": status,
        })
    for ev in UNRESOLVED:
        rows.append({
            "dataset_record": "(period / no single district-day)",
            "date": ev["date"], "district": ev["district"],
            "latitude": "", "longitude": "",
            "original_label": 0, "verified_label": 0,
            "source": ev["source"], "source_title": ev["source_title"],
            "source_url": ev["source_url"], "source_tier": ev["source_tier"],
            "reported_event_date": ev["reported_event_date"],
            "reported_location": ev["reported_location"],
            "evidence_summary": ev["evidence"], "status": "UNRESOLVED (label kept 0)",
        })
    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)

    # ---- consistency checks ----------------------------------------------
    win = (out["date"] >= START) & (out["date"] <= END)
    outside = out[~win]
    if (outside["landslide_verified"] != outside["landslide_original"]).any():
        problems.append("a label outside 2018-2024 changed")
    if (out["landslide_verified"] < out["landslide_original"]).any():
        problems.append("a label was lowered (1 -> 0)")
    changed = out[out["label_changed"]]
    if len(changed) != len(changes):
        problems.append("changed-row count does not match the evidence list")
    if (changed["verification_source_url"] == "").any():
        problems.append("a changed row has no source URL")
    if len(out) != len(master) or not out["date"].equals(master["date"]):
        problems.append("row count or ordering changed")
    for col in ("latitude", "longitude"):
        if not out[col].equals(master[col]):
            problems.append(f"non-label column '{col}' changed")

    labels_path = OUT_DIR / "landslide_labels_verified_2018_2024.csv"
    out.to_csv(labels_path, index=False)
    report.to_csv(OUT_DIR / "landslide_verification_report.csv", index=False)

    # ---- candidate event table (every candidate examined, with outcome) ----
    cand_rows = []
    for ev, st in ([(e, "CONFIRMED -> labelled 1") for e in changes]
                   + [(e, "CONFIRMED -> already 1 (cross-check)") for e in CROSS_CHECKED]
                   + [(e, "UNRESOLVED -> label kept 0") for e in UNRESOLVED]
                   + [(e, e.get("status", "REJECTED")) for e in REJECTED]):
        cand_rows.append({
            "event_date": ev["date"], "district": ev["district"],
            "reported_location": ev["reported_location"],
            "reported_event_date": ev["reported_event_date"],
            "evidence_summary": ev["evidence"],
            "source_name": ev["source"], "source_title": ev["source_title"],
            "source_url": ev["source_url"], "source_tier": ev["source_tier"],
            "verification_status": st,
        })
    pd.DataFrame(cand_rows).to_csv(OUT_DIR / "landslide_candidate_events.csv", index=False)

    # ---- master-dataset compatibility -------------------------------------
    full = pd.read_csv(MASTER, low_memory=False)
    splits = {"train": ("2012-01-30", "2021-02-14"),
              "validation": ("2021-02-15", "2023-01-23"),
              "test": ("2023-01-24", "2024-12-31")}

    def split_of(day: str) -> str:
        for name, (lo, hi) in splits.items():
            if lo <= day <= hi:
                return name
        return "outside"

    confirmed = changes + CROSS_CHECKED
    predictors = ["rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_14d_mm",
                  "rainfall_30d_mm", "soil_moisture", "temperature_c", "humidity_percent",
                  "elevation_m", "slope_degree", "aspect_degree", "curvature",
                  "land_cover", "lithology"]
    matched, unmatched, missing_counts, per_split = 0, [], {}, {}
    rows_missing_any = 0
    for ev in confirmed:
        r = full[(full["date"] == ev["date"]) & (full["district"] == ev["district"])]
        if r.empty:
            unmatched.append(f"{ev['date']} {ev['district']}")
            continue
        matched += 1
        per_split[split_of(ev["date"])] = per_split.get(split_of(ev["date"]), 0) + 1
        miss = [c for c in predictors if pd.isna(r.iloc[0][c])]
        if miss:
            rows_missing_any += 1
        for c in miss:
            missing_counts[c] = missing_counts.get(c, 0) + 1

    new_by_split = {}
    for ev in changes:
        new_by_split[split_of(ev["date"])] = new_by_split.get(split_of(ev["date"]), 0) + 1
    ver = out[win]
    by_year = (ver.assign(y=ver["date"].str[:4]).groupby("y")["landslide_verified"]
               .sum().astype(int))
    by_district = (ver.groupby("district")["landslide_verified"].sum().astype(int)
                   .sort_values(ascending=False))
    dup = int(out.duplicated(["date", "district"]).sum())
    river_missing = sum(
        int(full[(full["date"] == e["date"]) & (full["district"] == e["district"])]
            ["river_level_m"].isna().iloc[0])
        for e in changes)

    compat = f"""# Master dataset compatibility of the verified landslide labels

Checked against `real_master_dataset.csv` (read-only; this task did not modify it,
did not train any model, and did not touch backend/frontend/TCDL code).

**1. Unit of observation.** One row per district-day: {len(full):,} rows =
{full['district'].nunique()} districts x {full['date'].nunique():,} dates
({full['date'].min()} .. {full['date'].max()}). Duplicate district-days: {dup}.
Each district carries one representative latitude/longitude, so a row describes
the whole district, not a slide location.

**2. How events map to district-days.** A landslide event becomes
`landslide = 1` for the (date, district) it happened in. Several slides in one
district on one day stay a single positive row: the October 2021 KSDMA table
lists 69 incidents, which collapse to 13 district-days (11 new + 2 already
labelled). The label is presence-only: 0 means no exact-dated landslide is on
record, not that none occurred.

**3. Confirmed events matching a master row:** {matched} of {len(confirmed)}.

**4. Confirmed events with no master row:** {len(unmatched)}
{('- ' + '; '.join(unmatched)) if unmatched else '(none)'}

**5. District naming.** Sources use official spellings; the dataset uses CHIRPS
spellings. One mapping was needed: Pathanamthitta -> `Pattanamtitta`. All other
district names matched exactly. No location was assigned to a district unless
the source named the district or a taluk/village clearly within it.

**6. Duplicate district-days created:** 0. Labels are set on existing rows only;
no row was added.

**7-9. Split placement of the newly labelled days** (chronological boundaries
unchanged: train {splits['train'][0]}..{splits['train'][1]},
validation {splits['validation'][0]}..{splits['validation'][1]},
test {splits['test'][0]}..{splits['test'][1]}):

| split | newly labelled | all confirmed events checked |
|---|---|---|
| train | {new_by_split.get('train', 0)} | {per_split.get('train', 0)} |
| validation | {new_by_split.get('validation', 0)} | {per_split.get('validation', 0)} |
| test | {new_by_split.get('test', 0)} | {per_split.get('test', 0)} |

**10. Confirmed events with missing predictor values:** {rows_missing_any} of
{matched} (model features only). `river_level_m` is not a landslide-model
feature; it is missing on {river_missing} of the newly labelled rows.

**11. Predictors missing most often on event days:**
{(chr(10).join(f'- {k}: {v}' for k, v in sorted(missing_counts.items(), key=lambda x: -x[1]))) if missing_counts else '- none; every landslide-model feature is present on every confirmed event day'}

**12. Where the representation is inadequate.** Every confirmed event fits the
schema, but two limits matter for modelling and should be stated in the report:
- **Spatial.** A district-day row averages a whole district. The 24 Oct 2023
  Idukki slide at Pachadi followed "heavy rain for more than 3 hours" locally,
  yet the district row shows only {full[(full['date'] == '2023-10-24') & (full['district'] == 'Idukki')]['rainfall_1d_mm'].iloc[0]:.1f} mm of 1-day rainfall. A district
  average cannot represent a local cloudburst, so such a positive looks like a
  dry-day event to the model.
- **Static terrain.** slope, aspect, curvature, land cover and lithology take one
  value per district, so they cannot distinguish a slide-prone slope from flat
  land inside the same district.

**13. Useful data-collection changes for future versions.**
- Obtain the GSI Bhukosh landslide inventory (point locations with dates); it is
  the only systematic source that would give many day-level positives.
- Ask KSDMA/District Emergency Operations Centres for incident tables of the kind
  the October 2021 event report reproduces, for 2018-2020 and 2022-2024.
- Consider a finer spatial unit (taluk or grid cell) so the rainfall that
  triggered a slide is not averaged away across a district.
- Consider rainfall relative to each district's own climatology, so a local
  extreme is visible as an anomaly.
"""
    (DOC_DIR / "MASTER_DATASET_COMPATIBILITY.md").write_text(compat, encoding="utf-8")

    # ---- summary ----------------------------------------------------------
    n_examined = int(win.sum())
    orig_pos = int(out.loc[win, "landslide_original"].sum())
    new_pos = int(out.loc[win, "landslide_verified"].sum())
    ev_dates = sorted({e["date"] for e in changes})
    ev_districts = sorted({e["district"] for e in changes})
    official = sum(1 for e in changes if e["source_tier"] == "official/government")
    summary = f"""# Landslide label verification 2018-2024

Scope: the landslide label only, for district-days from {START} to {END}.
Method: documented-event verification. The window holds {n_examined:,} district-day
records, so each row could not be searched individually; instead landslide events
were researched from official reports and news, each candidate source was opened
and read, and only events whose **district and exact date** are stated in the
source were labelled.

- **Pass 1** worked from official event reports: the KSDMA October 2021 event
  report yielded 11 district-days.
- **Pass 2** started from the project's own IMD Disastrous Weather Events file.
  Its exact-dated entries that name several districts without saying which had
  the landslide had to be dropped when the master dataset was built; news
  reporting was used to resolve the district for those days. This produced the
  first verified 2023 positives (2 district-days, both Idukki, both in the test
  period). No ML model was trained or retrained in either pass.

**2023:** no longer zero. Two 2023 district-days are now evidence-backed
(24 Oct and 5 Nov, both Idukki). Beyond those, no further 2023 landslide with an
exact date and district was found: KSDMA published no 2023 extreme-rainfall event
report and the GSI inventory holds no exact-dated 2023 record. That is absence of
evidence, not proof that no other landslide occurred.

## Counts
| item | value |
|---|---|
| records in the dataset (all years) | {len(out):,} |
| records examined ({START}..{END}) | {n_examined:,} |
| positives before verification (in window) | {orig_pos} |
| records changed 0 -> 1 | {len(changes)} |
| positives after verification (in window) | {new_pos} |
| records remaining 0 (in window) | {n_examined - new_pos:,} |
| unresolved items recorded | {len(UNRESOLVED)} |
| unique event dates among the changes | {len(ev_dates)} |
| unique districts among the changes | {len(ev_districts)} |
| changes supported by official/government sources | {official} |
| changes supported only by reputable secondary sources | {len(changes) - official} |
| already-positive records cross-checked against a source | {len(CROSS_CHECKED)} |

New positive dates: {', '.join(ev_dates) or 'none'}
Districts: {', '.join(ev_districts) or 'none'}

## Verified positives by year ({START}..{END})
{by_year.to_string()}

## Verified positives by district ({START}..{END})
{by_district[by_district > 0].to_string()}

## Newly labelled days by split (boundaries unchanged)
{chr(10).join(f'- {k}: {v}' for k, v in sorted(new_by_split.items()))}

## Sources used
- Kerala State Disaster Management Authority (KSDMA), Event Report on "Extreme
  Rainfall over Kerala" October 2021 -- Table 2 lists 69 reported landslide
  incidents with taluk, village and exact date (source stated in the table:
  District Emergency Operations Centres). This is the evidence for every
  0 -> 1 change in this pass.
  https://sdma.kerala.gov.in/wp-content/uploads/2022/05/Event-report_October-2021.pdf
- KSDMA, Memorandum - Kerala Floods & Landslides 2019 (1-31 August 2019):
  period-level only, no per-incident dates -> unresolved, no labels set.
- KSDMA, Memorandum - Meppadi Landslide (30-7-2024): confirms the 2024-07-30
  Wayanad event, which is already labelled 1.
- Onmanorama, 29 August 2022 Kudayathoor (Thodupuzha) landslide, Idukki:
  cross-check of an existing positive.
- Wikipedia was used only to locate candidate events, never as evidence for a
  label.

## What is NOT covered (open items)
1. **2023 has no positive label and none was added.** No individual landslide
   with an exact date and district was found in this pass; KSDMA published no
   extreme-rainfall event report for 2023. This is absence of evidence, not
   evidence of absence.
2. **August 2019, August 2020 and the 2024 monsoon beyond the labelled days**
   are documented at period or state level in the sources reached here. Day-level
   incident lists (District Emergency Operations Centre tables of the kind the
   October 2021 report reproduces) would be needed to label them.
3. Several Indian news sites (The Hindu, Indian Express, NDTV, Reuters, BBC)
   block automated access, so candidate events there could not be opened and
   verified.
4. The original GSI/IMD-derived labels were not re-examined row by row; this
   pass only adds verified events and cross-checks a sample.

## Guarantees
- The raw dataset was not modified: labels are written to a separate file with
  the original label kept beside the verified one.
- No label outside {START}..{END} changed; no label was lowered; no non-label
  column changed.
- Every 0 -> 1 change has a source name, title and URL in
  landslide_verification_report.csv.
- No dates were invented: an event whose source gave only a month or a period
  was left unresolved.
"""
    (DOC_DIR / "VERIFICATION_SUMMARY.md").write_text(summary, encoding="utf-8")

    print(f"records examined {n_examined:,} | changed 0->1 {len(changes)} | "
          f"positives {orig_pos} -> {new_pos}")
    print("checks:", "PASS" if not problems else "FAIL " + "; ".join(problems))
    print(f"written -> {labels_path.parent}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
