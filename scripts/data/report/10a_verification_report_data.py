# Run from this folder: python 10a_verification_report_data.py  (writes verification_report_data.json), then
# node 10b_build_verification_report.js ../../DATA_VERIFICATION_REPORT.docx   (needs: npm install docx)
"""Collect the verification table and counts for DATA_VERIFICATION_REPORT.docx from the outputs."""
import json
import re
from pathlib import Path
import pandas as pd

U = Path(__file__).resolve().parents[3] / "collected_datasets"
V = U / "flood_events" / "verification_18"
v = pd.read_csv(V / "IFI_Kerala_flagged18_verified.csv")
log = pd.read_csv(V / "CHANGE_LOG.csv")
chk = pd.read_csv(V / "consistency_with_full_corrected_copy.csv")


def fmt_orig(s, e):
    s, e = s[:10], e[:10]
    return s if s == e else f"{s} → {e}"


def fmt_new(r):
    d = r.event_days.split(";")
    if len(d) == 1:
        return d[0]
    span = (pd.Timestamp(d[-1]) - pd.Timestamp(d[0])).days + 1
    return f"{d[0]} → {d[-1]}" if len(d) == span else ", ".join(d)


main = []
for r in v.itertuples():
    y = r.UEI[11:15]
    main.append([r.UEI.replace("UEI-IMD-FL-", ""), fmt_orig(r._3, r._4), fmt_new(r),
                 re.sub(r"\s*,\s*", ", ", str(r.Districts)).strip(), r.final_status,
                 f"IMD Disastrous Weather Events {y}, PDF p.{r.evidence_pdf_page}, {r.evidence_entry.replace("KERALA ", "Kerala entry ")}: “{r.imd_date_text}”"])
corr = []
for r in v[v.final_status == "CORRECTED"].itertuples():
    ph = [p.strip() for p in str(r.matched_evidence).split("|") if len(p.strip().split()) >= 3]
    # most distinctive first: phrases with numbers, then the longest
    ph = sorted(ph, key=lambda p: (not any(ch.isdigit() for ch in p), -len(p)))
    src = f"{v.at[r.Index, 'Description of Casualties/injured']} {v.at[r.Index, 'Extent of damage ']}"
    words = [p.strip() for p in str(r.matched_evidence).split("|") if len(p.strip().split()) == 1
             and re.search(rf"\b{p.strip().capitalize()}\b", src)][:2]      # place names only
    corr.append([r.UEI.replace("UEI-IMD-FL-", ""), fmt_orig(r._3, r._4), fmt_new(r),
                 f"IMD DWE {r.UEI[11:15]} p.{r.evidence_pdf_page}, {r.evidence_entry.replace('KERALA ', 'Kerala entry ')} prints “{r.imd_date_text}”",
                 "; ".join([f"“{p}”" for p in ph[:2]] + [f"“{w}”" for w in words])])
ver = []
for r in v[v.final_status == "VERIFIED-CORRECT"].itertuples():
    ver.append([r.UEI.replace("UEI-IMD-FL-", ""), f"“{r.imd_date_text}”",
                (r.verification_method.split(":")[0]).title(), str(r.verification_note)])
out = dict(main=main, corr=corr, ver=ver,
           counts=v.final_status.value_counts().to_dict(),
           log_rows=int(len(log)), agree=int(chk.agree.sum()), days_equal=int(chk.event_days_equal.sum()))
json.dump(out, open(Path(__file__).with_name("verification_report_data.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(out["counts"], len(main), len(corr), len(ver))
