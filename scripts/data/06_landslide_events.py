"""
Landslide event sources for the landslide target: date-precision audit of the GSI inventory and
extraction of dated landslide occurrences from the IMD Disastrous Weather Events (DWE) reports.

Inputs (unchanged):
  existing_collected_data/GSI_Kerala_Landslides_All.csv, GSI_Kerala_Landslides_2013_2024.csv
  flood_events/processed/IMD_DWE_Kerala_entries.csv            (02_parse_imd_dwe_kerala.py)
  landslide_events/processed/IMD_DWE_landslide_manual_entries.csv (hand-read entries, with citations)

Outputs (landslide_events/processed/):
  GSI_Kerala_Landslides_date_precision.csv  every GSI record + parsed dates + precision class
  IMD_DWE_Kerala_landslide_entries.csv      IMD entries reporting a landslide, with attribution
  Landslide_label_coverage_summary.csv      exact-date district-day counts per split (counts only)

Precision classes: EXACT_DATE (day, month, year), MONTH_YEAR, YEAR_ONLY, WEEK_ONLY, NO_DATE.
No day is ever assigned to a record that does not state one.
"""
from pathlib import Path
import re
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"
OUT = U / "landslide_events" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                    "oct", "nov", "dec"], 1)}
MONTH = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
TEXT_DATE = re.compile(rf"((?:\d{{1,2}}(?:st|nd|rd|th)?\s*(?:&|and|,|-|to)?\s*)+)(?:of\s+)?{MONTH},?\s*(\d{{4}})", re.I)
NUM_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b")
MONTH_YEAR = re.compile(rf"\b{MONTH},?\s*(\d{{4}})", re.I)
YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
TIME = re.compile(r"\b(\d{1,2})\s*[:.]\s*(\d{2})\s*(hrs?|am|pm|a\.m|p\.m)", re.I)
GSI_DISTRICT = {"IDUKKI": "Idukki", "Idukki (Devikulam Taluk)": "Idukki", "Plakkad": "Palakkad",
                "Pathanamthitta": "Pattanamtitta"}
IMD_TO_PROJECT = {"Pathanamthitta": "Pattanamtitta"}
GRID = pd.date_range("2012-01-30", "2024-12-31", freq="D")
n = len(GRID)
TRAIN_END, VAL_END = GRID[int(round(n * 0.70)) - 1], GRID[int(round(n * 0.85)) - 1]
KER = ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam", "Kozhikode",
       "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"]


def parse_history(h: str):
    h = "" if pd.isna(h) else str(h)
    exact, ambiguous, spans = [], False, []
    for m in TEXT_DATE.finditer(h):
        month, year = MON[m.group(2)[:3].lower()], int(m.group(3))
        nums = [int(x) for x in re.findall(r"\d{1,2}", m.group(1))]
        is_range = bool(re.search(r"\d\s*(?:st|nd|rd|th)?\s*(?:-|to)\s*\d", m.group(1)))
        if is_range and len(nums) == 2 and nums[0] <= nums[1]:
            nums = list(range(nums[0], nums[1] + 1))
        for d in nums:
            try: exact.append(pd.Timestamp(year, month, d))
            except ValueError: pass
        spans.append(m.span())
    for m in NUM_DATE.finditer(h):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        try:
            exact.append(pd.Timestamp(y, mo, d))
            ambiguous |= (d <= 12 and mo <= 12 and d != mo)
        except ValueError:
            pass
        spans.append(m.span())
    rest = "".join(ch if not any(a <= i < b for a, b in spans) else " " for i, ch in enumerate(h))
    months = [f"{int(m.group(2))}-{MON[m.group(1)[:3].lower()]:02d}" for m in MONTH_YEAR.finditer(rest)]
    rest2 = MONTH_YEAR.sub(" ", rest)
    years = sorted({int(y) for y in YEAR.findall(rest2)})
    times = [f"{int(a):02d}:{b} {c}" for a, b, c in TIME.findall(h)]
    exact = sorted(set(exact))
    if exact:
        prec = "EXACT_DATE"
    elif months:
        prec = "MONTH_YEAR"
    elif years:
        prec = "YEAR_ONLY"
    else:
        prec = "NO_DATE"
    other_events = bool(exact) and bool(months or years)
    return dict(date_precision=prec,
                exact_dates=";".join(d.strftime("%Y-%m-%d") for d in exact),
                month_year=";".join(months), year_only=";".join(map(str, years)),
                time_of_day=";".join(times), numeric_date_ambiguous_dm=ambiguous,
                also_mentions_other_undated_events=other_events)


def gsi_part():
    allf = pd.read_csv(U / "existing_collected_data" / "GSI_Kerala_Landslides_All.csv")
    sub = pd.read_csv(U / "existing_collected_data" / "GSI_Kerala_Landslides_2013_2024.csv")
    allf["in_2013_2024_file"] = allf["Sl_No"].isin(sub["Sl_No"])
    allf["district_project"] = allf["District"].map(lambda x: GSI_DISTRICT.get(x, x))
    parsed = allf["History"].apply(parse_history).apply(pd.Series)
    g = pd.concat([allf, parsed], axis=1)
    flags = []
    for _, r in g.iterrows():
        f = []
        if r["Year"] == 2028: f.append("YEAR_2028_IMPOSSIBLE")
        if r["numeric_date_ambiguous_dm"]: f.append("NUMERIC_DATE_DAY_MONTH_AMBIGUOUS (read day-first)")
        if r["exact_dates"] and any(pd.Timestamp(d) > pd.Timestamp("2025-12-31") for d in r["exact_dates"].split(";")):
            f.append("DATE_AFTER_2025")
        flags.append(" | ".join(f))
    g["flags"] = flags
    g.to_csv(OUT / "GSI_Kerala_Landslides_date_precision.csv", index=False)
    return g


def imd_part():
    e = pd.read_csv(U / "flood_events" / "processed" / "IMD_DWE_Kerala_entries.csv")
    man = pd.read_csv(OUT / "IMD_DWE_landslide_manual_entries.csv")
    rows = []
    for _, r in e.iterrows():
        text, inten = str(r["entry_text"]), str(r["intensity_text"])
        in_int = bool(re.search(r"land\s?slides?|land\s?slips?|mud\s?slides?", inten, re.I))
        mentions = [m.start() for m in re.finditer(r"land\s?slides?|land\s?slips?|lanslides?|mud\s?slides?", text, re.I)]
        if not (in_int or mentions):
            continue
        ds = [x for x in str(r["districts"]).split(";") if x and x != "nan"]
        # Only a single-district entry ties the landslide to a district. In multi-district
        # entries the casualty and damage columns interleave in the PDF text layer, so an
        # automatic district attribution is not reliable; those entries are kept, unattributed.
        if len(ds) == 1:
            att, lds = "single-district entry", ds
        else:
            att, lds = "not attributable (multi-district entry)", []
        nd = int(r["n_event_days"]) if pd.notna(r["n_event_days"]) else 0
        prec = ("EXACT_DATE" if nd == 1 else
                f"PERIOD ({nd} days; landslide day not stated)" if nd > 1 else "UNPARSED")
        rows.append(dict(report=r["report"], year=r["year"], page=r["page"], sno=r["sno"],
                         date_text=r["date_text"], event_days=r["event_days"], districts=";".join(ds),
                         landslide_districts=";".join(lds), landslide_attribution=att,
                         date_precision=prec,
                         landslide_evidence=("Intensity column" if in_int else "Extent-of-damage / casualty text"),
                         note=""))
    imd = pd.concat([pd.DataFrame(rows), man], ignore_index=True).sort_values(["year", "sno"])
    imd.to_csv(OUT / "IMD_DWE_Kerala_landslide_entries.csv", index=False)
    return imd


def counts(dd, label, source):
    w = dd[(dd["date"] >= GRID[0]) & (dd["date"] <= GRID[-1])].drop_duplicates()
    return dict(label_basis=label, source=source, positive_district_days=len(w),
                distinct_dates=w["date"].nunique(), districts=w["district"].nunique(),
                train=int((w["date"] <= TRAIN_END).sum()),
                validation=int(((w["date"] > TRAIN_END) & (w["date"] <= VAL_END)).sum()),
                test=int((w["date"] > VAL_END).sum()),
                years=";".join(f"{y}:{c}" for y, c in w.groupby(w["date"].dt.year).size().items()))


def main():
    g = gsi_part()
    print("GSI (all 3,378):", g["date_precision"].value_counts().to_dict())
    s = g[g["in_2013_2024_file"]]
    print("GSI 2013-2024 file (2,396):", s["date_precision"].value_counts().to_dict())
    print("numeric day/month ambiguous:", int(s["numeric_date_ambiguous_dm"].sum()),
          "| with time of day:", int((s["time_of_day"] != "").sum()),
          "| exact + other undated events:", int(s["also_mentions_other_undated_events"].sum()))
    imd = imd_part()
    print("IMD landslide entries:", len(imd), imd["landslide_attribution"].value_counts().to_dict())

    gsi_dd = pd.DataFrame([(r["district_project"], pd.Timestamp(d)) for _, r in s.iterrows()
                           if r["exact_dates"] for d in r["exact_dates"].split(";")],
                          columns=["district", "date"])
    def imd_days(df):
        return pd.DataFrame([(IMD_TO_PROJECT.get(d, d), pd.Timestamp(x)) for _, r in df.iterrows()
                             if isinstance(r["event_days"], str) and r["event_days"]
                             for d in r["landslide_districts"].split(";") for x in r["event_days"].split(";")],
                            columns=["district", "date"])
    att = imd[imd["landslide_districts"].fillna("") != ""]
    imd_exact = imd_days(att[att["date_precision"] == "EXACT_DATE"])
    imd_period = imd_days(att[att["date_precision"].str.startswith("PERIOD")])
    rows = [counts(gsi_dd, "GSI exact-date records (2013-2024 file)", "GSI"),
            counts(imd_exact, "IMD DWE single-district landslide entries, exact day", "IMD"),
            counts(pd.concat([gsi_dd, imd_exact]), "GSI + IMD exact day (union)", "GSI+IMD")]
    new = imd_exact.merge(gsi_dd.drop_duplicates(), how="left", indicator=True)
    rows.append(counts(new[new["_merge"] == "left_only"][["district", "date"]],
                       "IMD exact-day district-days not already in GSI", "IMD only"))
    rows.append(counts(imd_period, "IMD single-district PERIOD entries - every day of the period "
                       "(upper bound; landslide day not stated)", "IMD period"))
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "Landslide_label_coverage_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
