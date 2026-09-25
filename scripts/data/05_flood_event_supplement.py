"""
Find IMD DWE Kerala flood / heavy-rain events that are NOT in IFI-Impacts v3.0 and summarise
flood-label coverage per model split.

IFI-Impacts builds every Kerala record from the IMD Disastrous Weather Events (DWE) reports,
so a DWE Kerala entry is the same kind of record, from the same authority, with the same
inclusion rule. Two groups are separated:
  EXTENSION_AFTER_IFI_END  - event days after IFI's last Kerala record (2023-07-23)
  OMITTED_FROM_IFI         - inside IFI's period but no IFI record shares a day and a district

Inputs : flood_events/processed/IMD_DWE_Kerala_entries.csv
         flood_events/processed/IFI_Kerala_2012_2023_corrected.csv
Outputs: flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv
         flood_events/processed/Flood_label_coverage_summary.csv   (counts only - no labels written)
"""
from pathlib import Path
import re
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"
P = U / "flood_events" / "processed"
IFI_END = pd.Timestamp("2023-07-23")
GRID = pd.date_range("2012-01-30", "2024-12-31", freq="D")
n = len(GRID)
TRAIN_END, VAL_END = GRID[int(round(n * 0.70)) - 1], GRID[int(round(n * 0.85)) - 1]
KER = {"alappuzha": "Alappuzha", "ernakulam": "Ernakulam", "idukki": "Idukki", "kannur": "Kannur",
       "kasaragod": "Kasaragod", "kollam": "Kollam", "kottayam": "Kottayam", "kozhikode": "Kozhikode",
       "malappuram": "Malappuram", "palakkad": "Palakkad", "pathanamthitta": "Pathanamthitta",
       "thiruvananthapuram": "Thiruvananthapuram", "thrissur": "Thrissur", "wayanad": "Wayanad"}
HAZARD_WORDS = [("Massive Landslide", r"massive\s+landslide"), ("Heavy Rains", r"heavy\s+rain"),
                ("Flash flood", r"flash\s+flood"), ("Floods", r"(?<!flash )\bfloods?\b"),
                ("Landslide", r"(?<!massive )\bland\s?(?:slides?|slips?)\b"),
                ("Mudslide", r"\bmud\s?slides?\b")]


def hazard_keywords(intensity_text: str) -> str:
    """IMD hazard words found in the Intensity column (column text can carry casualty words)."""
    t = str(intensity_text).lower()
    return "; ".join(k for k, rx in HAZARD_WORDS if re.search(rx, t))


def ifi_district_days(df, use_verified=True):
    rows = []
    for _, r in df.iterrows():
        ds = {KER[t.strip().lower()] for t in re.split(r"[,;]", str(r["Districts"])) if t.strip().lower() in KER}
        if use_verified:
            if isinstance(r["event_days"], str) and r["event_days"]:
                days = [pd.Timestamp(x) for x in r["event_days"].split(";")]
            else:
                days = list(pd.date_range(r["verified_start"], r["verified_end"]))
        else:
            s = pd.to_datetime(r["Start Date"], format="%d-%m-%Y %H:%M")
            e = pd.to_datetime(r["End Date"], format="%d-%m-%Y %H:%M")
            days = list(pd.date_range(s, e))          # empty when end < start, as a naive expansion
        rows += [(d, day) for d in ds for day in days]
    return pd.DataFrame(rows, columns=["district", "date"]).drop_duplicates()


def split_counts(dd, label, source_end):
    w = dd[(dd["date"] >= GRID[0]) & (dd["date"] <= GRID[-1])]
    return dict(label_basis=label, positive_district_days=len(w),
                positive_rate_on_66080_grid=round(len(w) / (len(GRID) * 14), 4),
                train=int((w["date"] <= TRAIN_END).sum()),
                validation=int(((w["date"] > TRAIN_END) & (w["date"] <= VAL_END)).sum()),
                test=int((w["date"] > VAL_END).sum()),
                label_source_ends=str(pd.Timestamp(source_end).date()))


def main():
    dwe = pd.read_csv(P / "IMD_DWE_Kerala_entries.csv")
    ifi = pd.read_csv(P / "IFI_Kerala_2012_2023_corrected.csv")
    ifi_dd = ifi_district_days(ifi)
    ifi_set = set(zip(ifi_dd["district"], ifi_dd["date"]))

    out = []
    for _, e in dwe.iterrows():
        days = [pd.Timestamp(x) for x in str(e["event_days"]).split(";") if x and x != "nan"]
        ds = [KER.get(x.lower(), x) for x in str(e["districts"]).split(";") if x and x != "nan"]
        if not days:
            continue
        covered = any((d, day) in ifi_set for d in ds for day in days)
        if covered:
            continue
        cat = "EXTENSION_AFTER_IFI_END" if min(days) > IFI_END else "OMITTED_FROM_IFI"
        out.append(dict(category=cat, source="IMD Disastrous Weather Events", report=e["report"],
                        pdf_page=e["page"], entry=f"KERALA {e['sno']}", imd_date_text=e["date_text"],
                        event_days=e["event_days"], n_event_days=e["n_event_days"],
                        districts=";".join(ds), district_source=e["district_source"],
                        imd_intensity=hazard_keywords(e["intensity_text"]),
                        flood_mentioned=e["mentions_flood"], landslide_mentioned=e["mentions_landslide"],
                        entry_text=e["entry_text"][:600]))
    sup = pd.DataFrame(out)
    sup.to_csv(P / "IMD_DWE_Kerala_events_not_in_IFI.csv", index=False)
    print(sup.groupby(["category", sup["report"]]).size().to_string())

    # --- coverage summary (counts only) ------------------------------------------------
    naive = ifi_district_days(ifi, use_verified=False)
    sup_dd = []
    for _, r in sup.iterrows():
        sup_dd += [(d, pd.Timestamp(x)) for d in r["districts"].split(";") if d in KER.values()
                   for x in r["event_days"].split(";")]
    sup_dd = pd.DataFrame(sup_dd, columns=["district", "date"])
    ext = sup_dd[sup_dd["date"] > IFI_END]
    short = ifi[~ifi["flags"].fillna("").str.contains("LONG_PERIOD_SUMMARY")]
    dwe_end = pd.Timestamp("2024-12-31")          # DWE 2024 covers the full calendar year
    rows = [split_counts(naive, "IFI v3.0 as published (dates read DD-MM, start-end expansion)", IFI_END),
            split_counts(ifi_dd, "IFI v3.0 corrected against IMD DWE (this task)", IFI_END),
            split_counts(ifi_district_days(short),
                         "IFI corrected, excluding LONG_PERIOD_SUMMARY records (sensitivity)", IFI_END),
            split_counts(pd.concat([ifi_dd, ext]).drop_duplicates(),
                         "IFI corrected + IMD DWE events after 2023-07-23", dwe_end),
            split_counts(pd.concat([ifi_dd, sup_dd]).drop_duplicates(),
                         "IFI corrected + all IMD DWE Kerala entries missing from IFI", dwe_end)]
    cov = pd.DataFrame(rows)
    cov.to_csv(P / "Flood_label_coverage_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
