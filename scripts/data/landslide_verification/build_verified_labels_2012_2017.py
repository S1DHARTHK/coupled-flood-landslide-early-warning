"""
2012-2017 landslide label verification (same methodology as verified_2018_2024/).

Read-only over the project data: the master dataset is NOT modified and no label
is applied to it. This script records what the evidence supports, so the labels
can be applied later in a separate, deliberate step.

Every entry below was read at the source given (an opened PDF page or web page),
and is kept only when the source states BOTH an exact day and a district.

Inputs (read-only):
    real_master_dataset.csv                                  current labels
    updated_data/master_dataset/real_master_dataset_keys.csv  provenance of existing labels
    updated_data/landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv
    updated_data/landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv

Outputs (this directory):
    landslide_labels_verified_2012_2017.csv    original + verified label per row (window only)
    landslide_verification_report_2012_2017.csv  one row per candidate, with evidence
    VERIFICATION_SUMMARY_2012_2017.md          counts, coverage bias, limitations

Run from the project root:
    python scripts/data/landslide_verification/build_verified_labels_2012_2017.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "collected_datasets" / "landslide_events" / "verified_2012_2017"   # data outputs
DOC_DIR = HERE.parents[2] / "documents" / "landslide_verification" / "2012_2017"            # report outputs
ROOT = HERE.parents[2]  # scripts/<group>/landslide_verification -> project root
MASTER = ROOT / "dataset" / "real_master_dataset.csv"
KEYS = ROOT / "collected_datasets" / "master_dataset" / "real_master_dataset_keys.csv"
IMD = ROOT / "collected_datasets" / "landslide_events" / "processed" / "IMD_DWE_Kerala_landslide_entries.csv"
GSI = ROOT / "collected_datasets" / "landslide_events" / "processed" / "GSI_Kerala_Landslides_date_precision.csv"
START, END = "2012-01-01", "2017-12-31"
GRID_START = "2012-01-30"          # the master dataset starts here

# --- sources ---------------------------------------------------------------
KSDMA_2012 = dict(
    source="Government of Kerala / KSDMA",
    source_type="official government memorandum",
    source_title=("Memorandum: Landslides and Flood Losses - 2012, State Relief Commissioner & "
                  "Principal Secretary, Dept. of Revenue and Disaster Management, 17-09-2012, "
                  "section 1.2 'Landslides and floods' (pp. 3-5)"),
    source_url="https://sdma.kerala.gov.in/wp-content/uploads/2018/11/9.Memorandum-Landslides-2012.pdf",
    source_tier="official/government")
KSDMA_2014 = dict(
    source="Government of Kerala / KSDMA",
    source_type="official government memorandum",
    source_title=("Memorandum: Monsoon Calamity Losses 2014 (1 June - 15 Sept 2014), dated "
                  "photo-documentation captions (pp. 30-34)"),
    source_url=("https://sdma.kerala.gov.in/wp-content/uploads/2018/11/"
                "(2.1)2014-Memorandum-Monsoon-Rainfall.pdf"),
    source_tier="official/government")
ONMANORAMA_2017 = dict(
    source="Onmanorama (Malayala Manorama)",
    source_type="reputable news organisation",
    source_title="Rain fury brings Kerala to its knees: one more person killed",
    source_url=("https://www.onmanorama.com/news/kerala/2017/09/18/"
                "rain-havoc-kerala-idukki-dams-water-level.html"),
    source_tier="reputable secondary")
IMD_DWE = dict(
    source="India Meteorological Department",
    source_type="official annual report (Disastrous Weather Events)",
    source_title="IMD Disastrous Weather Events - Kerala landslide entries (project copy)",
    source_url="https://internal.imd.gov.in/pages/disastrous_weather_events.php",
    source_tier="official/government")
GSI_SRC = dict(
    source="Geological Survey of India (GSI)",
    source_type="official landslide inventory",
    source_title="GSI Kerala landslide inventory (exact-dated records)",
    source_url="https://www.gsi.gov.in/webcenter/portal/OCBIS/pageQuickLinks/pageLandslideAtlas",
    source_tier="official/government")

# --- newly verified events (not in the master) -----------------------------
NEW_VERIFIED: list[dict] = [
    dict(date="2012-08-06", district="Kannur",
         location="Karikottakiri and Murikkan Kara, Ayyankunnu village, Thalassery taluk",
         reported_event_date="6 and 7 August 2012",
         evidence=("Memorandum: 'Two landslides were reported from Karikottakiri and Murikkan "
                   "Kara in Ayyankunnu Village, Thalasherry Taluk on 6 and 7 August.' One "
                   "fatality. Both named days are labelled; neither is inferred."), **KSDMA_2012),
    dict(date="2012-08-07", district="Kannur",
         location="Karikottakiri and Murikkan Kara, Ayyankunnu village, Thalassery taluk",
         reported_event_date="6 and 7 August 2012",
         evidence=("Same memorandum sentence as 2012-08-06: landslides reported 'on 6 and 7 "
                   "August'."), **KSDMA_2012),
    dict(date="2012-08-26", district="Kannur",
         location="Thirumeni village, Thalassery taluk",
         reported_event_date="26 August 2012",
         evidence=("Memorandum: 'On 26 August a landslide was reported from Thirumeni village, "
                   "Thalasherry Taluk which damaged the arterial road of the village.'"),
         **KSDMA_2012),
    dict(date="2012-08-06", district="Kozhikode",
         location="Pulloorampara-Anakkampoyil region",
         reported_event_date="6 August 2012",
         evidence=("Memorandum: 'Over 35 minor and major debris flows were reported from "
                   "Pulloorampara-Anakkampoyil region on 6 August 2012.' Eight fatalities. "
                   "The 35+ incidents collapse to one district-day."), **KSDMA_2012),
    dict(date="2012-08-14", district="Idukki",
         location="Kaduvappara, Peerumedu taluk",
         reported_event_date="14 August 2012, 4 pm",
         evidence=("Memorandum: 'A major landslide occurred in Kaduvappara, Peerumedu Taluk on "
                   "14 August 2012, 4pm. Two people were injured.'"), **KSDMA_2012),
    dict(date="2012-08-17", district="Ernakulam",
         location="Kadavoor village, Kothamangalam taluk",
         reported_event_date="17 August 2012",
         evidence=("Memorandum: 'A major debris flow occurred in Kadavoor village of "
                   "Kothamangalam Taluk on 17 August 2012. Six fatalities were reported.' "
                   "This also resolves the district for the IMD DWE 2012 entry of 17 & 18 Aug., "
                   "which reports a landslide for five districts without naming one."),
         **KSDMA_2012),
    dict(date="2014-07-18", district="Idukki",
         location="Munnar",
         reported_event_date="18.07.2014",
         evidence=("Memorandum photo-documentation caption: 'Shop collapsed due to landslide at "
                   "Munnar, Idukki 18.07.2014'."), **KSDMA_2014),
    dict(date="2014-07-25", district="Wayanad",
         location="Pithrukkad",
         reported_event_date="25.07.2014",
         evidence=("Memorandum photo-documentation caption: 'Landslide at Pithrukkad, Wayanad "
                   "at 25.07.2014'."), **KSDMA_2014),
    dict(date="2014-08-24", district="Kollam",
         location="Chadamangalam",
         reported_event_date="24.08.2014",
         evidence=("Memorandum photo-documentation caption: 'Trees uprooted out of debris flow "
                   "damaged house, Chadamangalam, Kollam 24.08.2014'. A debris flow is a "
                   "landslide type in this project (GSI movement type 'Flow')."), **KSDMA_2014),
    dict(date="2017-09-17", district="Palakkad",
         location="Attappadi (Anakkal, Puthur, Jellyppara); also Kanjirapuzha, Mannarkkad, Poonjola",
         reported_event_date="Sunday 17 September 2017",
         evidence=("Article of 18 Sep 2017: 'Incessant rain triggered landslip in the tribal "
                   "hamlet of Attappadi and adjacent areas in Palakkad district', with further "
                   "landslips at Kanjirapuzha, Mannarkkad and Poonjola. The Sunday before "
                   "publication is 17 Sep 2017 (verified as a Sunday)."), **ONMANORAMA_2017),
]

# --- existing master positives, re-derived from their own source -----------
EXISTING: list[dict] = [
    dict(date="2012-04-23", district="Pattanamtitta", source_id="IMD:DWE_2012:p52:s3",
         location="Pathanamthitta district", reported_event_date="23 Apr. 2012",
         evidence="IMD DWE single-district entry, exact date; extent of damage: 'Landslide reported.'",
         **IMD_DWE),
    dict(date="2013-06-17", district="Kasaragod", source_id="IMD:DWE_2013:p56:s12",
         location="Kasaragod district", reported_event_date="17 Jun. 2013",
         evidence="IMD DWE single-district entry, exact date; landslide in the extent-of-damage / casualty text.",
         **IMD_DWE),
    dict(date="2013-07-23", district="Ernakulam", source_id="IMD:DWE_2013:p60:s21",
         location="Ernakulam district", reported_event_date="23 Jul. 2013",
         evidence="IMD DWE single-district entry, exact date; landslide in the extent-of-damage / casualty text.",
         **IMD_DWE),
    dict(date="2013-09-12", district="Wayanad", source_id="IMD:DWE_2013:p63:s28",
         location="Wayanad district", reported_event_date="12 Sep. 2013",
         evidence="IMD DWE single-district entry, exact date; landslide in the extent-of-damage / casualty text.",
         **IMD_DWE),
    dict(date="2014-06-21", district="Idukki", source_id="IMD:DWE_2014:p59:s8",
         location="Idukki district", reported_event_date="21 Jun. 2014",
         evidence="IMD DWE single-district entry, exact date; landslide in the extent-of-damage / casualty text.",
         **IMD_DWE),
    dict(date="2015-06-26", district="Ernakulam", source_id="GSI:14261",
         location="Vellurkunnam slide, Vellurkunnam (9.9927 N, 76.5748 E)",
         reported_event_date="26.06.2015 (stated in the record's History field)",
         evidence=("GSI inventory record 14261, a mapped slide location whose History field "
                   "states the exact date 26.06.2015."), **GSI_SRC),
    dict(date="2016-06-07", district="Idukki", source_id="IMD:DWE_2016:p61:s3",
         location="Idukki district", reported_event_date="7 June 2016",
         evidence=("IMD DWE casualty text names both district and day: 'One person died & other "
                   "injured in Idukki Dist. due to landslide on 7th June.' The entry covers "
                   "7-9 Jun but the landslide day is stated."), **IMD_DWE),
    dict(date="2016-09-18", district="Kozhikode", source_id="IMD:DWE_2016:p65:s9",
         location="Kozhikode district", reported_event_date="18 Sep. 2016",
         evidence="IMD DWE single-district entry, exact date; intensity column: 'Landslide'.",
         **IMD_DWE),
    dict(date="2017-06-05", district="Thiruvananthapuram", source_id="IMD:DWE-2017_final:p70:s3",
         location="Thiruvananthapuram district", reported_event_date="5 Jun. 2017",
         evidence="IMD DWE single-district entry, exact date; landslide in the intensity column.",
         **IMD_DWE),
]

# --- investigated, not labelled --------------------------------------------
REVIEW: list[dict] = [
    dict(date="2012-08 (day not stated)", district="Palakkad",
         location="100 acre, Ambalappara, Kottappadam panchayat (Silent Valley buffer zone)",
         reported_event_date="reported on 22 August 2012; occurrence date unknown",
         evidence=("Memorandum states plainly: 'The event was reported on 22 August as the region "
                   "is only accessible by foot... The actual date of occurrence is unknown.' A "
                   "report date is not an event date, so no day is labelled."), **KSDMA_2012),
    dict(date="2012 (no date given)", district="Kottayam",
         location="Thalappalam and Tikoy villages, Meenachil taluk",
         reported_event_date="not stated",
         evidence=("Memorandum: 'Two minor debris flows were reported from Thalappalam and Tikoy "
                   "villages of Meenachil Taluk' with no date. Real events, but undatable."),
         **KSDMA_2012),
    dict(date="2012-08-17; 2012-08-18", district="Idukki; Kottayam; Pattanamtitta; Thiruvananthapuram",
         location="five districts named in one entry",
         reported_event_date="17 & 18 Aug. 2012",
         evidence=("IMD DWE exact-dated entry reporting a landslide for five districts without "
                   "naming which. Only Ernakulam (17 Aug) could be resolved, from the 2012 "
                   "memorandum; the other four districts stay unresolved."), **IMD_DWE),
    dict(date="2013-08-04..2013-08-08", district="Idukki",
         location="Idukki district (46 landslides)",
         reported_event_date="between 4 and 8 August 2013",
         evidence=("Special package memorandum: 'in Idukki alone between 4 - 8 August, fourty six "
                   "(46) landslides have occurred'. District is certain, the individual days are "
                   "not stated, so no day is labelled. Strongest candidate for a future pass."),
         source="Government of Kerala / KSDMA",
         source_type="official government memorandum",
         source_title="Special package request - Monsoon Calamity 2013 (CM to PM)",
         source_url=("https://sdma.kerala.gov.in/wp-content/uploads/2018/11/"
                     "4.Special-Package-CM-to-PM-Monsoon-Calamity-2013.pdf"),
         source_tier="official/government"),
    dict(date="2013-06-01..2013-06-30", district="Kozhikode; Palakkad; Idukki; Pattanamtitta",
         location="four districts",
         reported_event_date="1-30 June 2013 (period)",
         evidence=("Monsoon calamity memorandum: 'Landslides were reported from four districts "
                   "namely Kozhikode, Palakkad, Idukki and Pathanamthitta' for the June period, "
                   "with no per-incident dates."),
         source="Government of Kerala / KSDMA",
         source_type="official government memorandum",
         source_title="Memorandum - Monsoon Calamity Losses 2013 (1-30 June 2013)",
         source_url=("https://sdma.kerala.gov.in/wp-content/uploads/2018/11/"
                     "6.Memorandum-Monsoon-Calamity-2013.pdf"),
         source_tier="official/government"),
    dict(date="2013-07-01..2013-07-04", district="Wayanad",
         location="Wayanad district", reported_event_date="1-4 Jul. 2013 (period)",
         evidence="IMD DWE single-district landslide entry; the landslide day is not stated.",
         **IMD_DWE),
    dict(date="2013-08-10; 2013-08-16", district="Wayanad",
         location="Wayanad district", reported_event_date="10 & 16 Aug. 2013",
         evidence=("IMD DWE single-district entry covering two separate days; the source does not "
                   "say which day the landslide occurred."), **IMD_DWE),
    dict(date="2014-08 (1st week)", district="Kannur",
         location="Kelakattum and Periyothu areas",
         reported_event_date="1st week of August 2014",
         evidence=("IMD DWE damage text names the district and places ('Landslide reported in "
                   "Kelakattum & Periyothu areas') but gives only a week, not a day."), **IMD_DWE),
    dict(date="2014-09-01..2014-09-03", district="Thiruvananthapuram",
         location="Thiruvananthapuram district", reported_event_date="1-3 Sep. 2014 (period)",
         evidence="IMD DWE single-district landslide entry; the landslide day is not stated.",
         **IMD_DWE),
    dict(date="2017-09-17", district="Idukki; Kottayam; Thiruvananthapuram",
         location="three districts named in one entry", reported_event_date="17 Sep. 2017",
         evidence=("IMD DWE exact-dated entry reporting a landslide for three districts without "
                   "naming which. Palakkad on the same day is labelled from independent news "
                   "reporting; these three districts stay unresolved."), **IMD_DWE),
    dict(date="multiple periods 2013-2017", district="various (multi-district IMD entries)",
         location="2013-06-25..29, 2013-07-05..12, 2014-07-12..15, 2014-08-22..24, "
                  "2015-06-19..29, 2017-09-18..19",
         reported_event_date="periods, districts not attributed",
         evidence=("Six further IMD DWE landslide entries covering multi-day periods across "
                   "several districts, with neither the day nor the district of the landslide "
                   "stated. None can produce a district-day."), **IMD_DWE),
    dict(date="2012-2017 (year or month only)", district="various",
         location="127 GSI records dated to a year only, 2 to a month only",
         reported_event_date="year or month only",
         evidence=("GSI inventory records whose History gives only a year (e.g. '2014') or a "
                   "month ('September, 2017'). Real mapped slides, but undatable to a day, so "
                   "they cannot become district-day labels."), **GSI_SRC),
]

REPORT_COLUMNS = ["date", "district", "status", "existing_or_new", "location", "source",
                  "source_type", "source_title", "source_url", "source_tier",
                  "reported_event_date", "evidence_summary", "in_master_already", "notes"]


def main() -> int:
    for f in (MASTER, KEYS, IMD, GSI):
        if not f.exists():
            print(f"missing input: {f}")
            return 1
    master = pd.read_csv(MASTER, usecols=["date", "district", "latitude", "longitude",
                                          "landslide"], low_memory=False)
    mkey = set(zip(master["date"], master["district"]))
    pos = set(zip(master.loc[master["landslide"] == 1, "date"],
                  master.loc[master["landslide"] == 1, "district"]))

    problems, rows = [], []

    def add(ev, status, existing_or_new, notes=""):
        in_master = (ev["date"], ev["district"]) in pos
        rows.append({"date": ev["date"], "district": ev["district"], "status": status,
                     "existing_or_new": existing_or_new, "location": ev["location"],
                     "source": ev["source"], "source_type": ev["source_type"],
                     "source_title": ev["source_title"], "source_url": ev["source_url"],
                     "source_tier": ev["source_tier"],
                     "reported_event_date": ev["reported_event_date"],
                     "evidence_summary": ev["evidence"],
                     "in_master_already": in_master, "notes": notes})

    for ev in EXISTING:
        if (ev["date"], ev["district"]) not in pos:
            problems.append(f"existing positive not found in master: {ev['date']} {ev['district']}")
        add(ev, "VERIFIED", "EXISTING_POSITIVE", f"source id in master keys: {ev['source_id']}")

    for ev in NEW_VERIFIED:
        if not (START <= ev["date"] <= END):
            problems.append(f"{ev['date']} outside {START}..{END}")
        if (ev["date"], ev["district"]) not in mkey:
            problems.append(f"no master row for {ev['date']} {ev['district']}")
            add(ev, "VERIFIED", "NO_MASTER_ROW", "date/district not in the master grid")
            continue
        already = (ev["date"], ev["district"]) in pos
        add(ev, "VERIFIED", "EXISTING_POSITIVE" if already else "NEW_POSITIVE",
            "already positive - no duplicate created" if already
            else "not currently in the master; applying it is a separate task")

    for ev in REVIEW:
        add(ev, "REVIEW", "REVIEW", "left unresolved: evidence does not fix an exact district-day")

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    report.to_csv(OUT_DIR / "landslide_verification_report_2012_2017.csv", index=False)

    # proposed label table for the window (master untouched)
    win = (master["date"] >= START) & (master["date"] <= END)
    lab = master[win].copy().rename(columns={"landslide": "landslide_current_master"})
    lab["landslide_verified"] = lab["landslide_current_master"]
    lab["verified_source"] = ""
    lab["verified_source_url"] = ""
    lab["would_change"] = False
    key = lab["date"] + "|" + lab["district"]
    new_applied = 0
    for ev in NEW_VERIFIED:
        m = key == f"{ev['date']}|{ev['district']}"
        if not m.any():
            continue
        if int(lab.loc[m, "landslide_current_master"].iloc[0]) == 1:
            continue
        lab.loc[m, ["landslide_verified", "verified_source", "verified_source_url",
                    "would_change"]] = [1, ev["source"], ev["source_url"], True]
        new_applied += 1
    lab.to_csv(OUT_DIR / "landslide_labels_verified_2012_2017.csv", index=False)

    # ---- checks ----------------------------------------------------------
    if new_applied != len(NEW_VERIFIED):
        problems.append(f"{len(NEW_VERIFIED)} new events but {new_applied} rows marked")
    if int(lab.duplicated(["date", "district"]).sum()):
        problems.append("duplicate district-day in the proposed label table")
    if int((lab["landslide_verified"] < lab["landslide_current_master"]).sum()):
        problems.append("a label would be lowered")
    dup_new = pd.DataFrame(NEW_VERIFIED).duplicated(["date", "district"]).sum()
    if dup_new:
        problems.append("duplicate event in NEW_VERIFIED")

    ver = report[report["status"] == "VERIFIED"]
    new_pos = report[report["existing_or_new"] == "NEW_POSITIVE"]
    by_year = lab[lab["landslide_verified"] == 1].groupby(lab["date"].str[:4]).size()
    by_dist = (lab[lab["landslide_verified"] == 1].groupby("district").size()
               .sort_values(ascending=False))
    official = int((new_pos["source_tier"] == "official/government").sum())

    summary = f"""# Landslide label verification 2012-2017

Same methodology as `verified_2018_2024/`: an event is labelled only when a source
that was opened and read states **both an exact day and a district**. Periods, weeks,
months and years are not converted into days, and a multi-district report is not spread
across its districts.

**The master dataset was not modified.** This directory records what the evidence
supports; applying it is a separate task.

## Counts
| item | value |
|---|---|
| existing 2012-2017 positives reviewed | {len(EXISTING)} |
| of those VERIFIED | {int((report['existing_or_new'] == 'EXISTING_POSITIVE').sum())} |
| of those REVIEW | 0 |
| of those REJECTED | 0 |
| additional events VERIFIED | {len(NEW_VERIFIED)} |
| additional events REVIEW | {len(REVIEW)} |
| additional events REJECTED | 0 |
| **verified 2012-2017 positives in total** | **{int(lab['landslide_verified'].sum())}** |
| of which not currently in the master | {len(new_pos)} |
| new positives from official government sources | {official} of {len(new_pos)} |
| unresolved / review candidates | {len(REVIEW)} |

## Verified positives by year (current master + newly verified)
{by_year.to_string()}

## Verified positives by district
{by_dist.to_string()}

## New district-days not currently in the master
{new_pos[['date', 'district', 'location', 'source']].to_string(index=False)}

## Why the review cases stay unresolved
{chr(10).join(f"- **{r['date']} / {r['district']}** - {r['evidence_summary'][:190]}" for _, r in report[report['status'] == 'REVIEW'].iterrows())}

## Source coverage bias (important)
The number of verified events per year reflects **what was documented and reachable**,
not how many landslides happened.

- **2012** is the best-covered year, because the state published a dedicated memorandum
  on the 2012 landslides with a district-by-district account and dates.
- **2014** yields events only because that year's memorandum carries dated photo captions.
- **2013** is badly under-represented: the memoranda state that 70 landslides occurred by
  late July, 46 of them in Idukki between 4 and 8 August, but give no per-incident dates.
  Three IMD entries are the only day-level 2013 evidence found.
- **2015-2017** have no equivalent state memorandum, so coverage rests on IMD entries plus
  one news report.
- GSI's inventory, which dominates 2018-2019, contains only **one** exact-dated record for
  the whole of 2012-2017; 127 of its records for these years carry a year only.

A zero or a low count for a year is therefore a statement about sources, not about Kerala.

## Limitations
1. Presence-only labels: a 0 means no exact-dated landslide is on record for that
   district-day.
2. Three of the ten new events come from photo captions or a news report rather than an
   incident table; each names the district and the date, and each is quoted in the report.
3. The 2013 Idukki cluster (46 landslides, 4-8 Aug) is certainly real but undatable from
   the sources reached; district-level daily reports would resolve it.
4. Several Indian news sites block automated access, so some candidate reports could not
   be opened.
"""
    (DOC_DIR / "VERIFICATION_SUMMARY_2012_2017.md").write_text(summary, encoding="utf-8")

    print(f"existing reviewed {len(EXISTING)} | new verified {len(NEW_VERIFIED)} | "
          f"review {len(REVIEW)} | verified total in window {int(lab['landslide_verified'].sum())}")
    print("checks:", "PASS" if not problems else "FAIL " + "; ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
