"""
Cross-check every Kerala record of IFI-Impacts v3.0 against its original source, the
IMD Disastrous Weather Events (DWE) annual report of the same year.

Inputs (unchanged):
  hazard_events/India_Flood_Inventory_v3.csv
  flood_events/processed/IMD_DWE_Kerala_entries.csv   (from 02_parse_imd_dwe_kerala.py)

Output:
  flood_events/processed/IFI_Kerala_date_check_auto.csv

For each IFI row the start/end dates are read four ways: as given (DD-MM), start swapped,
end swapped, both swapped. An interpretation is supported only if a DWE Kerala entry of
that year has exactly that first and last event day. District overlap is reported so the
match can be judged. This script only proposes; final decisions (with evidence) are
recorded by hand in IFI_Kerala_date_corrections.csv.
"""
from pathlib import Path
import re
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"
IFI = U / "hazard_events" / "India_Flood_Inventory_v3.csv"
DWE = U / "flood_events" / "processed" / "IMD_DWE_Kerala_entries.csv"
OUT = U / "flood_events" / "processed" / "IFI_Kerala_date_check_auto.csv"
KER = ["alappuzha", "ernakulam", "idukki", "kannur", "kasaragod", "kollam", "kottayam",
       "kozhikode", "malappuram", "palakkad", "pathanamthitta", "thiruvananthapuram",
       "thrissur", "wayanad"]


def swap(s):
    d, m, rest = s[:2], s[3:5], s[6:]
    try:
        return pd.to_datetime(f"{m}-{d}-{rest}", format="%d-%m-%Y %H:%M")
    except ValueError:
        return pd.NaT


def main():
    f = pd.read_csv(IFI, low_memory=False)
    k = f[f["State"].astype(str).str.contains("Kerala", case=False, na=False)].copy()
    k["start_given"] = pd.to_datetime(k["Start Date"], format="%d-%m-%Y %H:%M")
    k["end_given"] = pd.to_datetime(k["End Date"], format="%d-%m-%Y %H:%M")
    k = k[k["start_given"].dt.year >= 2012]
    dwe = pd.read_csv(DWE)
    dwe = dwe[dwe["date_parse_flag"] == "ok"].copy()
    dwe["days"] = dwe["event_days"].str.split(";").apply(lambda x: [pd.Timestamp(v) for v in x])
    dwe["first"] = dwe["days"].apply(min)
    dwe["last"] = dwe["days"].apply(max)
    years = set(dwe["year"])

    out = []
    for _, r in k.iterrows():
        ifi_d = {t.strip().lower() for t in re.split(r"[,;]", str(r["Districts"]))} & set(KER)
        interp = {"as_given": (r["start_given"], r["end_given"]),
                  "start_swapped": (swap(r["Start Date"]), r["end_given"]),
                  "end_swapped": (r["start_given"], swap(r["End Date"])),
                  "both_swapped": (swap(r["Start Date"]), swap(r["End Date"]))}
        y = r["start_given"].year
        res = dict(UEI=r["UEI"], start_given=r["Start Date"][:10], end_given=r["End Date"][:10],
                   duration_days=r["Duration(Days)"], districts=r["Districts"],
                   span_given=(r["end_given"] - r["start_given"]).days + 1,
                   ambiguous_format=(int(r["Start Date"][:2]) <= 12 and r["Start Date"][:2] != r["Start Date"][3:5])
                                    or (int(r["End Date"][:2]) <= 12 and r["End Date"][:2] != r["End Date"][3:5]))
        if y not in years:
            res.update(check="NO_DWE_REPORT_DOWNLOADED")
            out.append(res); continue
        hits = []
        seen = set()
        for name, (s, e) in interp.items():
            if pd.isna(s) or pd.isna(e) or e < s or (s, e) in seen:
                continue
            seen.add((s, e))
            for _, d in dwe[(dwe["first"] == s) & (dwe["last"] == e)].iterrows():
                dd = {x.lower() for x in str(d["districts"]).split(";")} if pd.notna(d["districts"]) else set()
                overlap = len(ifi_d & dd) > 0 or "all_districts" in dd
                hits.append((name, d["sno"], d["page"], d["date_text"], d["event_days"],
                             d["districts"], overlap))
        given = [h for h in hits if h[0] == "as_given"]
        other = [h for h in hits if h[0] != "as_given"]
        if given and any(h[6] for h in given):
            chk = "MATCH_AS_GIVEN"
        elif other and any(h[6] for h in other) and not given:
            chk = "MATCH_ONLY_WHEN_SWAPPED"
        elif given or other:
            chk = "DATE_MATCH_DISTRICT_UNCONFIRMED"
        else:
            chk = "NO_DWE_MATCH"
        best = next((h for h in hits if h[6]), hits[0] if hits else None)
        res.update(check=chk,
                   interpretation=best[0] if best else "",
                   dwe_report_page=best[2] if best else "",
                   dwe_sno=best[1] if best else "",
                   dwe_date_text=best[3] if best else "",
                   dwe_event_days=best[4] if best else "",
                   dwe_districts=best[5] if best else "",
                   n_candidate_matches=len(hits))
        out.append(res)
    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False)
    print(df.groupby(df["start_given"].str[-4:])["check"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
