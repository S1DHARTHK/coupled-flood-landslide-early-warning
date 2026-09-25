"""
Final classification of the 18 flagged Kerala IFI records (after 10_verify_flagged_18.py).

For each record the IMD source entry was located by content (unique phrases of IFI's own
casualty/damage text) or, for the one record without text (2021-0078), by its position in
IFI's sequence of IMD-derived records. The date IMD printed for that entry decides:
  VERIFIED-CORRECT  IFI start and end dates equal IMD's first and last event day
  CORRECTED         IMD's printed date differs from IFI's; IMD's date is adopted
  UNRESOLVED        no IMD entry could be identified with confidence

Outputs (flood_events/verification_18/):
  IFI_Kerala_flagged18_verified.csv   18 records: every original IFI column unchanged + verification fields
  CHANGE_LOG.csv                      one row per changed value, with evidence
  consistency_with_full_corrected_copy.csv  agreement with processed/IFI_Kerala_2012_2023_corrected.csv
"""
from pathlib import Path
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"
V = U / "flood_events" / "verification_18"
URL = {2017: "https://www.imdpune.gov.in/library/public/DWE-2017_final.pdf",
       2018: "https://www.imdpune.gov.in/library/public/DWE-2018_final.pdf",
       2019: "https://www.imdpune.gov.in/library/public/DWE-2019.pdf",
       2020: "https://www.imdpune.gov.in/library/public/DWE-2020.pdf",
       2021: "https://www.imdpune.gov.in/library/public/DWE_2021.pdf"}
# the one record located structurally (no descriptive text in IFI)
STRUCTURAL = {"UEI-IMD-FL-2021-0078": 17}


def main():
    ifi = pd.read_csv(U / "hazard_events" / "India_Flood_Inventory_v3.csv", low_memory=False)
    cand = pd.read_csv(V / "content_match_candidates.csv")
    dwe = pd.read_csv(U / "flood_events" / "processed" / "IMD_DWE_Kerala_entries.csv")
    rows, log = [], []
    for c in cand.itertuples():
        r = ifi[ifi["UEI"] == c.UEI].iloc[0]
        y = int(c.UEI[11:15])
        sno = STRUCTURAL.get(c.UEI, c.matched_entry)
        method = ("structural: IFI records 2021-0076..0081 map one-to-one, in order, onto IMD Kerala entries "
                  "15..20 (0079's 1 death = IMD 'One woman died')" if c.UEI in STRUCTURAL else
                  f"content: {int(c.n_phrases_matched)} phrase(s)/place names from IFI's own text found only in this IMD entry")
        if pd.isna(sno):
            rows.append({**r.to_dict(), "final_status": "UNRESOLVED", "verification_method": "no unique IMD entry found"})
            continue
        e = dwe[(dwe["year"] == y) & (dwe["sno"] == int(sno))].iloc[0]
        days = e["event_days"].split(";")
        s0 = pd.to_datetime(r["Start Date"], format="%d-%m-%Y %H:%M").strftime("%Y-%m-%d")
        e0 = pd.to_datetime(r["End Date"], format="%d-%m-%Y %H:%M").strftime("%Y-%m-%d")
        status = "VERIFIED-CORRECT" if (s0, e0) == (days[0], days[-1]) else "CORRECTED"
        span = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days + 1
        note = []
        if len(days) < span:
            note.append(f"IMD lists {len(days)} separate day(s) inside the {span}-day range; use event_days.")
        if abs(len(days) - r["Duration(Days)"]) > 1:
            note.append(f"IFI Duration(Days)={r['Duration(Days)']:.0f} does not equal the {len(days)} source days "
                        f"(IMD reports a {span}-day period).")
        if status == "CORRECTED":
            note.insert(0, "IFI stored the date(s) month-first (DD and MM swapped); IMD prints "
                           f"“{' '.join(str(e['date_text']).split())}”.")
        ev = dict(final_status=status, verified_start=days[0], verified_end=days[-1], event_days=";".join(days),
                  n_event_days=len(days), imd_date_text=" ".join(str(e["date_text"]).split()),
                  imd_districts_area_column=e["districts"], evidence_source="India Meteorological Department, "
                  f"Disastrous Weather Events {y}", evidence_url=URL[y], evidence_pdf_page=int(e["page"]),
                  evidence_entry=f"KERALA {int(sno)}", verification_method=method,
                  matched_evidence=("" if c.UEI in STRUCTURAL else str(c.matched_phrases)[:300]),
                  verification_note=" ".join(note), excerpt_file=f"excerpts/{c.UEI}.txt")
        rows.append({**r.to_dict(), **ev})
        if status == "CORRECTED":
            for fld, old, new in [("Start Date", r["Start Date"], days[0]), ("End Date", r["End Date"], days[-1])]:
                if pd.to_datetime(old, format="%d-%m-%Y %H:%M").strftime("%Y-%m-%d") != new:
                    log.append(dict(UEI=c.UEI, field=fld, original_value=old, corrected_value=new,
                                    reason="Day and month swapped in IFI; date printed in the IMD source entry adopted",
                                    source=ev["evidence_source"], url=URL[y], pdf_page=ev["evidence_pdf_page"],
                                    entry=ev["evidence_entry"], imd_date_text=ev["imd_date_text"],
                                    evidence=ev["matched_evidence"][:200], changed_on="2026-09-12",
                                    note="Original file hazard_events/India_Flood_Inventory_v3.csv not modified"))
            log.append(dict(UEI=c.UEI, field="event_days (new column)", original_value="(start–end range)",
                            corrected_value=";".join(days), reason="Exact event days as printed by IMD",
                            source=ev["evidence_source"], url=URL[y], pdf_page=ev["evidence_pdf_page"],
                            entry=ev["evidence_entry"], imd_date_text=ev["imd_date_text"], evidence="",
                            changed_on="2026-09-12", note=""))
    out = pd.DataFrame(rows)
    out.to_csv(V / "IFI_Kerala_flagged18_verified.csv", index=False)
    pd.DataFrame(log).to_csv(V / "CHANGE_LOG.csv", index=False)

    full = pd.read_csv(U / "flood_events" / "processed" / "IFI_Kerala_2012_2023_corrected.csv").set_index("UEI")
    chk = []
    for r in out.itertuples():
        f = full.loc[r.UEI]
        chk.append(dict(UEI=r.UEI, final_status=r.final_status, verified_start=r.verified_start,
                        full_copy_start=f["verified_start"], verified_end=r.verified_end, full_copy_end=f["verified_end"],
                        event_days_equal=(r.event_days == (f["event_days"] if isinstance(f["event_days"], str) else "")),
                        agree=(r.verified_start == f["verified_start"] and r.verified_end == f["verified_end"])))
    chk = pd.DataFrame(chk)
    chk.to_csv(V / "consistency_with_full_corrected_copy.csv", index=False)
    print(out["final_status"].value_counts().to_string())
    print("agree with full corrected copy:", int(chk["agree"].sum()), "/", len(chk),
          "| event_days equal:", int(chk["event_days_equal"].sum()))
    pd.set_option("display.width", 220)
    print(out[["UEI", "Start Date", "End Date", "verified_start", "verified_end", "final_status", "imd_date_text",
               "evidence_pdf_page", "evidence_entry"]].to_string(index=False))


if __name__ == "__main__":
    main()
