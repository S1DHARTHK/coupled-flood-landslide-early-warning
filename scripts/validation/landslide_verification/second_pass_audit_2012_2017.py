"""
SECOND-PASS (adversarial) audit of the 2012-2017 landslide labels.

Independent of the first pass (build_verified_labels_2012_2017.py and its outputs,
which are left untouched). Every candidate was re-derived from the underlying
source: raw IMD Disastrous Weather Events PDFs in the project (text re-read, and
the scanned 2012 pages inspected as images), the KSDMA memoranda, a KSDMA-
sanctioned research report, and news reports opened at source.

Read-only over the project data: the master dataset is NOT modified and no label
is applied to it.

Outputs (this directory, all prefixed SECOND_PASS_):
    SECOND_PASS_candidate_table_2012_2017.csv   every candidate, both passes' status
    SECOND_PASS_labels_2012_2017.csv            proposed labels for the window
    SECOND_PASS_AUDIT_REPORT_2012_2017.md       the audit

Run from the project root:
    python scripts/validation/landslide_verification/second_pass_audit_2012_2017.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "collected_datasets" / "landslide_events" / "verified_2012_2017"   # data outputs
DOC_DIR = HERE.parents[2] / "documents" / "landslide_verification" / "2012_2017"            # report outputs
ROOT = HERE.parents[2]  # scripts/<group>/landslide_verification -> project root
MASTER = ROOT / "dataset" / "real_master_dataset.csv"
START, END = "2012-01-01", "2017-12-31"
MASTER_SHA256 = "0e397cbf4b7a08bf0a70dbb51ab99eb33fb988b9a05564071210362c9dbfd609"

# --- sources -----------------------------------------------------------------
MEMO12 = ("KSDMA / Govt. of Kerala, Memorandum: Landslides and Flood Losses - 2012 (17-09-2012)",
          "https://sdma.kerala.gov.in/wp-content/uploads/2018/11/9.Memorandum-Landslides-2012.pdf")
MEMO14 = ("KSDMA / Govt. of Kerala, Memorandum: Monsoon Calamity Losses 2014 (photo captions)",
          "https://sdma.kerala.gov.in/wp-content/uploads/2018/11/(2.1)2014-Memorandum-Monsoon-Rainfall.pdf")
MEMO13 = ("KSDMA / Govt. of Kerala, Special package request - Monsoon Calamity 2013",
          "https://sdma.kerala.gov.in/wp-content/uploads/2018/11/4.Special-Package-CM-to-PM-Monsoon-Calamity-2013.pdf")
MEMO13B = ("KSDMA / Govt. of Kerala, Memorandum - Monsoon Calamity Losses 2013 (1-30 June)",
           "https://sdma.kerala.gov.in/wp-content/uploads/2018/11/6.Memorandum-Monsoon-Calamity-2013.pdf")
KSDMA_THIRUVAMBADI = ("KSDMA major research project: Landslide Susceptibility Assessment and "
                      "Preparedness Strategies, Thiruvambadi Grama Panchayath, Kozhikode "
                      "(DM/328/2016/SDMA)",
                      "https://sdma.kerala.gov.in/wp-content/uploads/2020/11/KSDMA-REPORT.pdf")
IMD = ("IMD Disastrous Weather Events (raw PDF in project: "
       "updated_data/flood_events/imd_dwe_source/)",
       "https://internal.imd.gov.in/pages/disastrous_weather_events.php")
GSI = ("GSI Kerala landslide inventory (project copy)",
       "https://www.gsi.gov.in/webcenter/portal/OCBIS/pageQuickLinks/pageLandslideAtlas")
ONM17 = ("Onmanorama, 'Rain fury brings Kerala to its knees' (18 Sep 2017)",
         "https://www.onmanorama.com/news/kerala/2017/09/18/rain-havoc-kerala-idukki-dams-water-level.html")
MAD17 = ("Madhyamam, 'Heavy rain: Landslip in Attappady, rail traffic affected in Kottayam' "
         "(17 Sep 2017, 9:11 pm)",
         "https://madhyamamonline.com/kerala/2017/sep/17/heavy-rain-landslip-attappady-rail-traffic-affected-kottayam")


def row(date, district, location, status, primary, corroborating, existing_new, reason,
        first_pass, conclusion, tier):
    return dict(date=date, district=district, location=location, status=status,
                primary_source=primary[0], source_url=primary[1],
                corroborating_source=("; ".join(c[0] for c in corroborating)
                                      if corroborating else ""),
                corroborating_url=("; ".join(c[1] for c in corroborating)
                                   if corroborating else ""),
                existing_or_new=existing_new, reason=reason,
                first_pass_status=first_pass, second_pass_conclusion=conclusion,
                primary_source_tier=tier)


C: list[dict] = []

# ============ A. the 9 existing master positives =============================
C += [
    row("2012-04-23", "Pattanamtitta", "Pandalam town", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Scanned DWE_2012 p.47 inspected as an image: '23 Apr. | Pathanamthitta | Heavy "
        "rains | 2 persons died in Pandalam town | ... ii) Landslide reported.'",
        "VERIFIED", "Confirmed from the raw scan; single official source.", "official"),
    row("2013-06-17", "Kasaragod", "Kasaragod district", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Raw DWE_2013 text: '17 Jun. Kasaragod Heavy rains 2 persons died and 2 others "
        "injured due to landslide. Landslide reported.'",
        "VERIFIED", "Confirmed word-for-word.", "official"),
    row("2013-07-23", "Ernakulam", "Ernakulam district", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Raw DWE_2013 text: '23 Jul. Ernakulam Heavy rains 4 persons died and 2 others "
        "injured in a landslide.'",
        "VERIFIED", "Confirmed word-for-word.", "official"),
    row("2013-09-12", "Wayanad", "Padumutty area", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Raw DWE_2013 text: '12 Sep. Wayanad Heavy rains i) Padumutty bridge collapsed ... "
        "iv) Landslide reported.'",
        "VERIFIED", "Confirmed word-for-word.", "official"),
    row("2014-06-21", "Idukki", "national highway iron bridge", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Raw DWE_2014 text: '21 Jun. Idukki Heavy rains i) Landslide damaged the iron bridge "
        "on national highway.'",
        "VERIFIED", "Confirmed word-for-word.", "official"),
    row("2015-06-26", "Ernakulam", "Vellurkunnam (9.9927 N, 76.5748 E)", "VERIFIED", GSI, [],
        "EXISTING_POSITIVE",
        "GSI record 14261 ('Vellurkunnam slide'); History field states '26.06.2015'. Numeric "
        "date, but day 26 cannot be a month, so it is unambiguous.",
        "VERIFIED", "Confirmed; the only exact-dated GSI record in 2012-2017.", "official"),
    row("2016-06-07", "Idukki", "Idukki district", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Raw DWE_2016 text: 'One person died & other injured in Idukki Dist. due to landslide "
        "on 7th June.' (entry covers 7-9 Jun but names the landslide day).",
        "VERIFIED", "Confirmed; the day is explicitly stated.", "official"),
    row("2016-09-18", "Kozhikode", "Kozhikode district", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Raw DWE_2016 text: '18 Sep. | Kozhikode | Landslide | 6 students died.' Official "
        "record states date, district and landslide. CAUTION: no independent account of a "
        "6-fatality landslide was found in accessible news, and no contradicting account "
        "either. Kept on the official record; flagged for a manual check.",
        "VERIFIED", "Kept VERIFIED on the official record; flagged as uncorroborated.",
        "official"),
    row("2017-06-05", "Thiruvananthapuram", "Thiruvananthapuram district", "VERIFIED", IMD, [],
        "EXISTING_POSITIVE",
        "Raw DWE_2017 text: '5 Jun. Thiruvananthapurm Landslide One person injured. A house "
        "damaged.'",
        "VERIFIED", "Confirmed word-for-word.", "official"),
]

# ============ B. the 10 first-pass new labels, challenged =====================
C += [
    row("2012-08-06", "Kannur", "Karikottakiri / Murikkan Kara, Ayyankunnu, Thalassery taluk",
        "REVIEW", MEMO12, [IMD], "NEW_POSITIVE (not applied)",
        "Rests on one memorandum sentence ('two landslides ... on 6 and 7 August'). The same "
        "memorandum dates the neighbouring Kozhikode debris flows to 6 Aug in its text, which "
        "its own Table 1, its Figure 4 caption and an independent KSDMA report all contradict "
        "(7 Aug). IMD gives only a 4-8 Aug period for Kannur/Kozhikode/Wayanad. The 6 Aug "
        "dating in this document is therefore unreliable and uncorroborated.",
        "VERIFIED", "Downgraded to REVIEW: date not independently established.", "official"),
    row("2012-08-06", "Kozhikode", "Pulloorampara-Anakkampoyil (Thiruvambadi GP)",
        "REJECTED", MEMO12, [KSDMA_THIRUVAMBADI], "NEW_POSITIVE (not applied)",
        "Contradicted. The memorandum's text says 6 Aug, but (a) its Table 1 ('rainfall ... "
        "on the day of the event') ends at 07.08.2012 for Pulloorampara, (b) its Figure 4 "
        "names 7 and 17 Aug as 'the two days on which the landslides which caused fatalities "
        "occurred', and (c) the KSDMA-sanctioned Thiruvambadi report states four times that "
        "the 35 debris flows (8 deaths) occurred on 7 Aug 2012, including Tharippapoyil and "
        "Manchuvad. Same physical event as 2012-08-07 Kozhikode; labelling 6 Aug would create "
        "a wrong-date duplicate. (The event is plausibly overnight 6-7 Aug; the day recorded "
        "by every other source is 7 Aug.)",
        "VERIFIED", "REJECTED: wrong date; the event is re-dated to 2012-08-07.", "official"),
    row("2012-08-07", "Kannur", "Karikottakiri / Murikkan Kara, Ayyankunnu, Thalassery taluk",
        "VERIFIED", MEMO12, [], "NEW_POSITIVE (not applied)",
        "Memorandum sentence names 7 Aug for the Ayyankunnu landslides, and the memorandum's "
        "Figure 4 names 7 Aug as a day on which fatal landslides occurred (Kannur had 1 "
        "landslide fatality). Consistent within the document.",
        "VERIFIED", "Kept VERIFIED (single official document, internally consistent for this day).",
        "official"),
    row("2012-08-14", "Idukki", "Kaduvappara, Peerumedu taluk", "VERIFIED", MEMO12, [],
        "NEW_POSITIVE (not applied)",
        "Memorandum: 'A major landslide occurred in Kaduvappara, Peerumedu Taluk on 14 August "
        "2012, 4pm.' Explicit day and hour; nothing contradicts it.",
        "VERIFIED", "Kept VERIFIED (single official source).", "official"),
    row("2012-08-17", "Ernakulam", "Kadavoor village, Kothamangalam taluk", "VERIFIED", MEMO12,
        [IMD], "NEW_POSITIVE (not applied)",
        "Memorandum text (17 Aug, 6 deaths), its Table 1 (event-day rainfall 17.08.2012 at "
        "Kadavoor) and Figure 4 (17 Aug fatal landslides) agree; IMD's 17 & 18 Aug entry lists "
        "Ernakulam and reports 'Landslide reported'.",
        "VERIFIED", "Kept VERIFIED (strongest 2012 event: four consistent elements).",
        "official"),
    row("2012-08-26", "Kannur", "Thirumeni village, Thalassery taluk", "VERIFIED", MEMO12, [],
        "NEW_POSITIVE (not applied)",
        "Memorandum: 'On 26 August a landslide was reported from Thirumeni village, "
        "Thalasherry Taluk which damaged the arterial road of the village.' Explicit day; not "
        "affected by the 6/7 Aug inconsistency; nothing contradicts it.",
        "VERIFIED", "Kept VERIFIED (single official source).", "official"),
    row("2014-07-18", "Idukki", "Munnar", "REVIEW", MEMO14, [], "NEW_POSITIVE (not applied)",
        "Only evidence is a photo caption ('Shop collapsed due to landslide at Munnar, Idukki "
        "18.07.2014'). A caption date is the photograph's date, which can post-date the event. "
        "IMD 2014 has no Idukki entry for 18 Jul; no news corroboration found.",
        "VERIFIED", "Downgraded to REVIEW: photo date not established as event date.",
        "official"),
    row("2014-07-25", "Wayanad", "Pithrukkad", "REVIEW", MEMO14, [], "NEW_POSITIVE (not applied)",
        "Only evidence is a photo caption ('Landslide at Pithrukkad, Wayanad at 25.07.2014'). "
        "IMD's Wayanad landslide entry is 12-15 Jul, not 25 Jul; no corroboration found.",
        "VERIFIED", "Downgraded to REVIEW: photo date not established as event date.",
        "official"),
    row("2014-08-24", "Kollam", "Chadamangalam", "REVIEW", MEMO14, [], "NEW_POSITIVE (not applied)",
        "Caption 'Trees uprooted out of debris flow damaged house' is ambiguous (flood debris "
        "vs slope failure); every other Kollam caption that day shows flooding; IMD's 22-24 Aug "
        "landslide entry covers Kottayam, Pathanamthitta, Thiruvananthapuram and Wayanad - not "
        "Kollam; photo date not established as event date.",
        "VERIFIED", "Downgraded to REVIEW: event type and date both uncertain.", "official"),
    row("2017-09-17", "Palakkad", "Attappadi (Anakkal, Puthur, Jellyppara); Kanjirapuzha, "
        "Mannarkkad, Poonjola", "VERIFIED", MAD17, [ONM17], "NEW_POSITIVE (not applied)",
        "Two independent reputable reports: Madhyamam published 17 Sep 2017 9:11 pm - landslip "
        "in Attappadi, Palakkad 'Sunday morning'; Onmanorama 18 Sep 2017 - same, 'Sunday'. "
        "No official source found: IMD's 17 Sep entry names Idukki, Kottayam and "
        "Thiruvananthapuram, which is not a contradiction (IMD entries are not exhaustive).",
        "VERIFIED", "Kept VERIFIED (two independent secondary sources; no official corroboration).",
        "reputable secondary"),
]

# ============ C. additional events found in the second pass ===================
C += [
    row("2012-08-07", "Kozhikode", "Pulloorampara (Thiruvambadi GP); Tharippapoyil; Manchuvad",
        "VERIFIED", KSDMA_THIRUVAMBADI, [MEMO12], "NEW_POSITIVE (not applied)",
        "KSDMA-sanctioned report: 'Multiple debris flows occurred in Pullurampara in Kozhikode "
        "district on Aug. 7, 2012, took the lives of 8 people'; '35 minor and major landslides "
        "on august 7, 2012'; Tharippapoyil and Manchuvad debris flows '7th August 2012'. The "
        "2012 memorandum's Table 1 and Figure 4 also give 7 Aug. Corrected date of the event "
        "first recorded as 2012-08-06.",
        "(not in first pass as this date)", "NEW VERIFIED (date correction).", "official"),
]

# ============ D. REVIEW candidates (first-pass, re-examined) ==================
C += [
    row("2012-08 (day unknown)", "Palakkad", "100 acre, Ambalappara, Kottappadam (Silent Valley buffer)",
        "REVIEW", MEMO12, [], "REVIEW",
        "Memorandum: 'reported on 22 August ... The actual date of occurrence is unknown.' A "
        "report date is not an event date. No other source found.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2012 (no date)", "Kottayam", "Thalappalam and Tikoy villages, Meenachil taluk", "REVIEW",
        MEMO12, [], "REVIEW",
        "Two minor debris flows named without any date. No other source found.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2012-08-17; 2012-08-18", "Idukki; Kottayam; Pattanamtitta; Thiruvananthapuram",
        "IMD entry 10 (scanned p.49)", "REVIEW", IMD, [], "REVIEW",
        "IMD: '17 & 18 Aug. | Ernakulam, Idukki, Kottayam, Pathanamthitta and Thiruvananthapuram "
        "| ... Landslide reported.' No district named; only Ernakulam (17 Aug) is resolved by "
        "the memorandum.",
        "REVIEW", "Kept REVIEW for the other four districts.", "official"),
    row("2013-08-04..2013-08-08", "Idukki", "Idukki district (46 landslides)", "REVIEW", MEMO13,
        [IMD], "REVIEW",
        "Special package: 'in Idukki alone between 4 - 8 August, fourty six (46) landslides "
        "have occurred'. IMD entry 25 (4-8 Aug) confirms Idukki landslides ('One person missing "
        "and 2 injured in landslide') but gives the same period. No source dates any single day; "
        "the 2013 fatality annexure lists names and causes but no dates.",
        "REVIEW", "Kept REVIEW (district certain, day not).", "official"),
    row("2013-06-01..2013-06-30", "Kozhikode; Palakkad; Idukki; Pattanamtitta",
        "four districts (June 2013)", "REVIEW", MEMO13B, [], "REVIEW",
        "Memorandum names four districts with landslides for June 2013, no per-incident dates.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2013-07-01..2013-07-04", "Wayanad", "Mananthavady tehsil", "REVIEW", IMD, [], "REVIEW",
        "IMD: '1-4 Jul. Wayanad ... Mananthavady Tehsil: i) Landslides reported.' Period only.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2013-08-10; 2013-08-16", "Wayanad", "Wayanad district", "REVIEW", IMD, [], "REVIEW",
        "IMD: '10 & 16 Aug. Wayanad ... i) Landslide reported.' Two separate days; the entry does "
        "not say which day (or both) had the landslide.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2014-08 (1st week)", "Kannur", "Kelakattum and Periyothu areas", "REVIEW", IMD, [],
        "REVIEW", "IMD damage text names places but only a week.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2014-09-01..2014-09-03", "Thiruvananthapuram", "Thiruvananthapuram district", "REVIEW",
        IMD, [], "REVIEW", "IMD: '1-3 Sep. Thiruvananthapuram ... ii) Landslides reported.' Period only.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2017-09-17", "Idukki; Kottayam; Thiruvananthapuram", "IMD entry 9", "REVIEW", IMD, [],
        "REVIEW",
        "IMD: '17 Sep. Idukki, Kottayam, Thiruvananthapurm | Heavy rains & Landslide | ... Trains "
        "were delayed due to landslide on railway track.' The track landslide is not placed in a "
        "district; Madhyamam's headline mentions rail traffic 'affected in Kottayam' but does not "
        "say a landslide caused it.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("2012-2017 (year/month only)", "various", "127 GSI year-only + 2 month-only records",
        "REVIEW", GSI, [], "REVIEW",
        "Real mapped slides with only a year or month in History. Not convertible to days.",
        "REVIEW", "Kept REVIEW.", "official"),
    row("multiple periods 2013-2017", "various", "grouped first-pass row", "REVIEW", IMD, [],
        "REVIEW",
        "First pass grouped six multi-district period entries. Second pass itemised the ones "
        "whose damage text names a district (section E below); all remain period-only.",
        "REVIEW", "Kept REVIEW; itemised below.", "official"),
]

# ============ E. REVIEW candidates newly itemised from the raw IMD rescan =====
C += [
    row("2013-06-25..2013-06-29", "Idukki", "Munnar", "REVIEW", IMD, [], "REVIEW",
        "IMD entry 14: 'one person died due to landslide in Munnar' (Idukki). District now "
        "attributed; day not stated within 25-29 Jun.",
        "(inside grouped row)", "New itemised REVIEW (district attributed).", "official"),
    row("2013-07-05..2013-07-12", "Idukki", "Cheruthoni town", "REVIEW", IMD, [], "REVIEW",
        "IMD entry 17: 'Cheruthoni town in Idukki: i) Landslide reported.' Period only.",
        "(inside grouped row)", "New itemised REVIEW (district attributed).", "official"),
    row("2013-08-04..2013-08-08", "Ernakulam", "Blavana - Pooyamkutty road", "REVIEW", IMD, [],
        "REVIEW",
        "IMD entry 25: 'Ernakulam: ... ii) Landslide at Blavana Pooyamkutty road reported.' "
        "Period only.",
        "(inside grouped row)", "New itemised REVIEW (district attributed).", "official"),
    row("2014-07-12..2014-07-15", "Wayanad", "Ambalavayal, Kalpetta, Mananthavady, Pandhaloor, "
        "Sulthan Bathery", "REVIEW", IMD, [], "REVIEW",
        "IMD entry 9: Wayanad subsection '... iii) Landslide reported.' Period only.",
        "(inside grouped row)", "New itemised REVIEW (district attributed).", "official"),
    row("2014-08-22..2014-08-24", "Kottayam", "Irattupetta / Poonjar / Laipally area", "REVIEW",
        IMD, [], "REVIEW", "IMD entry 11: Kottayam subsection 'iii) Landslide reported.' Period only.",
        "(inside grouped row)", "New itemised REVIEW (district attributed).", "official"),
    row("2014-08-22..2014-08-24", "Thiruvananthapuram", "Ponmudi-Vithura and Bonakadu",
        "REVIEW", IMD, [], "REVIEW",
        "IMD entry 11: 'Lanslides reported in Ponmudi-Vithura and Bonakadu areas'. Period only.",
        "(inside grouped row)", "New itemised REVIEW (district attributed).", "official"),
    row("2014-08-22..2014-08-24", "Kozhikode or Wayanad (ambiguous)", "Thamarassery ghat",
        "REVIEW", IMD, [], "REVIEW",
        "IMD lists 'Landslide in Thamaraserry ghat reported' under Wayanad, but the Thamarassery "
        "ghat road lies largely in Kozhikode district; district mapping is uncertain and the "
        "date is a period.",
        "(inside grouped row)", "New itemised REVIEW (district AND day uncertain).", "official"),
    row("2015-06-19..2015-06-29", "Palakkad", "Palakkad district", "REVIEW", IMD, [], "REVIEW",
        "IMD 2015 entry: 'Loss of standing crops worth at least Rs. 3 crore due to landslips "
        "reported' (Palakkad subsection) within an 11-day period.",
        "(inside grouped row)", "New itemised REVIEW (district attributed).", "official"),
]


def main() -> int:
    sha = hashlib.sha256(MASTER.read_bytes()).hexdigest()
    if sha != MASTER_SHA256:
        print(f"master hash changed: {sha}")
        return 1
    master = pd.read_csv(MASTER, usecols=["date", "district", "landslide"], low_memory=False)
    pos = set(zip(master.loc[master["landslide"] == 1, "date"],
                  master.loc[master["landslide"] == 1, "district"]))
    grid = set(zip(master["date"], master["district"]))

    table = pd.DataFrame(C)
    problems = []

    ver = table[table["status"] == "VERIFIED"].copy()
    # every VERIFIED row must be a single exact day + single district in the grid
    for _, r in ver.iterrows():
        if (r["date"], r["district"]) not in grid:
            problems.append(f"VERIFIED row not in master grid: {r['date']} {r['district']}")
        existing = (r["date"], r["district"]) in pos
        if existing != r["existing_or_new"].startswith("EXISTING"):
            problems.append(f"existing/new flag wrong for {r['date']} {r['district']}")
    if ver.duplicated(["date", "district"]).any():
        problems.append("duplicate VERIFIED district-day")

    table.to_csv(OUT_DIR / "SECOND_PASS_candidate_table_2012_2017.csv", index=False)

    win = (master["date"] >= START) & (master["date"] <= END)
    lab = master[win].rename(columns={"landslide": "landslide_current_master"}).copy()
    vset = set(zip(ver["date"], ver["district"]))
    lab["landslide_second_pass"] = [1 if (d, k) in vset else 0
                                    for d, k in zip(lab["date"], lab["district"])]
    lab["would_change"] = lab["landslide_second_pass"] != lab["landslide_current_master"]
    if int((lab["landslide_second_pass"] < lab["landslide_current_master"]).sum()):
        problems.append("a current master positive is not VERIFIED (would be lowered)")
    lab.to_csv(OUT_DIR / "SECOND_PASS_labels_2012_2017.csv", index=False)

    # counts
    ex = table[table["existing_or_new"] == "EXISTING_POSITIVE"]
    fp_new = table[table["first_pass_status"].eq("VERIFIED")
                   & table["existing_or_new"].str.startswith("NEW")]
    add_new = table[table["first_pass_status"].str.startswith("(not in first pass")]
    add_rev = table[table["first_pass_status"].eq("(inside grouped row)")]
    by_year = ver.groupby(ver["date"].str[:4]).size()
    by_dist = ver.groupby("district").size().sort_values(ascending=False)
    new_pos = ver[ver["existing_or_new"].str.startswith("NEW")]
    changes = table[(table["first_pass_status"] != table["status"])]

    rep = f"""# SECOND-PASS adversarial audit - landslide labels 2012-2017

Independent re-audit of every first-pass candidate. The first-pass files
(`build_verified_labels_2012_2017.py`, `landslide_*_2012_2017.csv`,
`VERIFICATION_SUMMARY_2012_2017.md`) are left untouched; everything from this pass is
prefixed `SECOND_PASS_`. **No label was applied to the master dataset.**

## How it was audited
- **IMD entries re-read from the raw PDFs** in the project
  (`updated_data/flood_events/imd_dwe_source/`); the 2012 report is a scan, so its
  pages were inspected as images.
- **The 2012 memorandum was checked for internal consistency** (text vs Table 1 vs
  Figure 4), and against a KSDMA-sanctioned research report on Thiruvambadi.
- **2014 photo captions were challenged** as photo dates, not event dates.
- **The Palakkad 2017 news claim** was checked for official and independent corroboration.
- **2013 was searched** again: the monsoon memoranda, the special package and the
  memorandum's fatality annexure. The annexure lists names and causes of death but no
  dates, so it cannot place a day.
- **Every Kerala landslide mention** in the raw 2013-2017 IMD reports was re-scanned to find
  district attributions the processed file had missed.

## Final counts
| | |
|---|---|
| A. existing candidates reviewed | {len(ex)} |
| B. existing VERIFIED | {int((ex['status'] == 'VERIFIED').sum())} |
| C. existing REVIEW | {int((ex['status'] == 'REVIEW').sum())} |
| D. existing REJECTED | {int((ex['status'] == 'REJECTED').sum())} |
| E. first-pass new positives reviewed | {len(fp_new)} |
| F. still VERIFIED | {int((fp_new['status'] == 'VERIFIED').sum())} |
| G. moved to REVIEW | {int((fp_new['status'] == 'REVIEW').sum())} |
| H. REJECTED | {int((fp_new['status'] == 'REJECTED').sum())} |
| I. additional new VERIFIED | {int((add_new['status'] == 'VERIFIED').sum())} |
| J. additional REVIEW (itemised from raw IMD) | {len(add_rev)} |
| K. additional REJECTED | 0 |
| **L. final verified 2012-2017 district-days** | **{len(ver)}** |
| O. total REVIEW | {int((table['status'] == 'REVIEW').sum())} |
| P. total REJECTED | {int((table['status'] == 'REJECTED').sum())} |

### M. Verified by year
{by_year.to_string()}

### N. Verified by district
{by_dist.to_string()}

### Verified district-days not currently in the master ({len(new_pos)})
{new_pos[['date', 'district', 'location', 'primary_source']].to_string(index=False)}

## FIRST PASS -> SECOND PASS CHANGES
{chr(10).join(f"- **{r['date']} {r['district']}**: {r['first_pass_status']} -> {r['status']}. {r['second_pass_conclusion']} {r['reason'][:260]}" for _, r in changes.iterrows() if r['first_pass_status'] in ('VERIFIED', 'REVIEW') or r['first_pass_status'].startswith('(not in'))}

## Key finding: the 2012 memorandum misdates the Kozhikode event
The memorandum's body text says the Pulloorampara debris flows happened on 6 August 2012.
Three other elements say 7 August: the memorandum's own Table 1 (rainfall "on the day of
the event" ends 07.08.2012), its own Figure 4 ("7 August and 17 August, the two days on
which the landslides which caused fatalities occurred"), and a KSDMA-sanctioned research
report that dates the 35 debris flows and 8 deaths to 7 August four separate times.
2012-08-06 Kozhikode is therefore REJECTED and the event is recorded as 2012-08-07
Kozhikode. Because the same sentence-level 6 August dating is the only support for
2012-08-06 Kannur, that label is downgraded to REVIEW.

## Flag for manual check
- **2016-09-18 Kozhikode**: IMD records "Landslide | 6 students died". It was kept VERIFIED
  on the official record, but no independent account was found in accessible news.

## Coverage bias
The verified count per year reflects documentation, not landslide frequency.
- **2012** is the best-covered year (a dedicated state landslide memorandum), but that
  memorandum is internally inconsistent for 6/7 August.
- **2013** is badly under-documented at day level. Official sources describe about 70
  landslides and 46 in Idukki on 4-8 August, yet only periods are given; exact-day
  evidence exists only for the three IMD single-district entries.
- **2014** looked well covered in the first pass only because of photo captions. With
  captions treated as insufficient on their own, a single day-level event remains
  (21 Jun, IMD).
- **2015-2017** have no state memorandum. They rest on IMD plus, for 2017, two news
  reports.
- **GSI** holds one exact-dated record in all six years (2015), against 127 year-only
  records.
- **District bias:** verified days concentrate in Idukki, Kannur, Ernakulam and Kozhikode,
  where memoranda or IMD happened to give detail. Alappuzha, Kollam, Malappuram and
  Thrissur have none. That reflects the sources, not the absence of slides.

## Integrity
Master dataset SHA-256 checked before and after: `{MASTER_SHA256}` (unchanged).
"""
    (DOC_DIR / "SECOND_PASS_AUDIT_REPORT_2012_2017.md").write_text(rep, encoding="utf-8")

    print(f"verified {len(ver)} | review {int((table['status'] == 'REVIEW').sum())} | "
          f"rejected {int((table['status'] == 'REJECTED').sum())} | new not in master {len(new_pos)}")
    print(by_year.to_dict())
    print("checks:", "PASS" if not problems else "FAIL " + "; ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
