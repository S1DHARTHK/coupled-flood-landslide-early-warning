"""
Build the verified / corrected working copy of the Kerala records of IFI-Impacts v3.0.

Inputs (unchanged):
  hazard_events/India_Flood_Inventory_v3.csv                 original IFI-Impacts v3.0
  flood_events/processed/IFI_Kerala_date_check_auto.csv      03_verify_ifi_kerala_dates.py
  flood_events/processed/IMD_DWE_Kerala_entries.csv          02_parse_imd_dwe_kerala.py
  flood_events/processed/IFI_Kerala_manual_verification.csv  hand-verified records, with citations

Outputs (flood_events/processed/):
  IFI_Kerala_2012_2023_corrected.csv   every Kerala IFI record starting 2012-2023: all original
                                       columns unchanged + verified_start/verified_end/event_days,
                                       verification_status, evidence, flags
  IFI_Kerala_date_corrections.csv      only the records whose dates were changed or left unresolved

The original IFI file is never modified. A date is changed only when the IMD Disastrous
Weather Events report of that year (the source IFI cites for every Kerala record) shows it.
"""
from pathlib import Path
import re
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"
P = U / "flood_events" / "processed"
IFI = U / "hazard_events" / "India_Flood_Inventory_v3.csv"


def main():
    f = pd.read_csv(IFI, low_memory=False)
    k = f[f["State"].astype(str).str.contains("Kerala", case=False, na=False)].copy()
    k["_start"] = pd.to_datetime(k["Start Date"], format="%d-%m-%Y %H:%M")
    k = k[(k["_start"].dt.year >= 2012)].drop(columns="_start")
    auto = pd.read_csv(P / "IFI_Kerala_date_check_auto.csv").set_index("UEI")
    dwe = pd.read_csv(P / "IMD_DWE_Kerala_entries.csv")
    man = pd.read_csv(P / "IFI_Kerala_manual_verification.csv").set_index("UEI")

    out = []
    for _, r in k.iterrows():
        uei = r["UEI"]
        a = auto.loc[uei]
        rec = dict(verified_start="", verified_end="", event_days="", verification_status="",
                   evidence_report="", evidence_pdf_page="", evidence_entry="", imd_date_text="",
                   verification_note="")
        if uei in man.index:
            m = man.loc[uei]
            rec.update(verified_start=m["verified_start"], verified_end=m["verified_end"],
                       event_days=m["event_days"] if pd.notna(m["event_days"]) else "",
                       verification_status=m["decision"], evidence_report=m["evidence_report"],
                       evidence_pdf_page=int(m["evidence_pdf_page"]), evidence_entry=m["evidence_entry"],
                       imd_date_text=m["imd_date_text"], verification_note=m["note"])
        elif a["check"] in ("MATCH_AS_GIVEN", "MATCH_ONLY_WHEN_SWAPPED", "DATE_MATCH_DISTRICT_UNCONFIRMED"):
            days = str(a["dwe_event_days"]).split(";")
            e = dwe[(dwe["year"] == int(days[0][:4])) & (dwe["sno"] == int(a["dwe_sno"]))].iloc[0]
            swapped = a["interpretation"] != "as_given"
            span = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days + 1
            status = "CORRECTED_DAY_MONTH_SWAP" if swapped else "VERIFIED_NO_CHANGE"
            notes = []
            if swapped:
                notes.append(f"IFI date(s) stored month-first ({a['interpretation'].replace('_', ' ')}).")
            if len(days) < span:
                notes.append(f"IMD lists {len(days)} separate day(s) within a {span}-day span: expand by "
                             f"event_days, not by the start-end range.")
            if a["check"] == "DATE_MATCH_DISTRICT_UNCONFIRMED":
                notes.append("Date confirmed; district column of the IMD table is misaligned in the "
                             "PDF text layer, so IFI's district assignment is retained.")
            rec.update(verified_start=days[0], verified_end=days[-1], event_days=";".join(days),
                       verification_status=status, evidence_report=e["report"],
                       evidence_pdf_page=int(e["page"]), evidence_entry=f"KERALA {int(e['sno'])}",
                       imd_date_text=e["date_text"], verification_note=" ".join(notes))
        else:
            rec.update(verification_status="UNRESOLVED",
                       verification_note=f"No matching IMD entry found ({a['check']}).")

        flags = []
        vs, ve = rec["verified_start"], rec["verified_end"]
        if vs and ve and (pd.Timestamp(ve) - pd.Timestamp(vs)).days + 1 > 14:
            flags.append("LONG_PERIOD_SUMMARY: IMD reports a multi-week period, not day-by-day flooding")
        if pd.notna(r["Duration(Days)"]) and vs and ve:
            n_days = (len(rec["event_days"].split(";")) if rec["event_days"]
                      else (pd.Timestamp(ve) - pd.Timestamp(vs)).days + 1)
            if abs(n_days - r["Duration(Days)"]) > 1:
                flags.append("IFI_DURATION_INCONSISTENT_WITH_SOURCE_PERIOD")
        cause = str(r["Main Cause"]).lower()
        if re.search(r"land\s?slide|land\s?slip|mud\s?sl", cause) and not re.search(r"rain|flood", cause):
            flags.append("LANDSLIDE_ONLY_CAUSE: listed in IFI as a flood event but cause is landslide only")
        if pd.isna(r["Districts"]) or not str(r["Districts"]).strip():
            flags.append("NO_DISTRICT: cannot be placed on the district grid")
        rec["flags"] = " | ".join(flags)
        out.append({**r.to_dict(), **rec})

    df = pd.DataFrame(out)
    df.to_csv(P / "IFI_Kerala_2012_2023_corrected.csv", index=False)
    corr = df[df["verification_status"].str.startswith("CORRECTED") | (df["verification_status"] == "UNRESOLVED")
              | df["verification_note"].str.contains("separate day", na=False)]
    cols = ["UEI", "Start Date", "End Date", "Duration(Days)", "Districts", "verified_start", "verified_end",
            "event_days", "verification_status", "evidence_report", "evidence_pdf_page", "evidence_entry",
            "imd_date_text", "verification_note"]
    corr[cols].rename(columns={"Start Date": "original_start", "End Date": "original_end",
                               "Duration(Days)": "original_duration_days"}).to_csv(
        P / "IFI_Kerala_date_corrections.csv", index=False)
    print(df["verification_status"].value_counts().to_string())
    print("flags:", df["flags"].str.split(" \\| ").explode().str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
