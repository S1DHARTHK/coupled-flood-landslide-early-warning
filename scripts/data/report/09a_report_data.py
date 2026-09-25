# Run from this folder: python 09a_report_data.py  (writes report_data.json), then
# node 09b_build_report.js ../../DATA_SOURCE_AND_CORRECTION_REPORT.docx   (needs: npm install docx)
"""Collect every number/table for the report directly from the processed outputs."""
import json
import pandas as pd

from pathlib import Path
U = str(Path(__file__).resolve().parents[3] / "collected_datasets") + "/"
out = {}
import re
rep_year = lambda f: re.search(r"(20\d\d)", str(f)).group(1)

# river level
asg = pd.read_csv(U + "river_level/processed/District_Gauge_Assignment.csv")
inv = pd.read_csv(U + "river_level/processed/CWC_Kerala_station_inventory.csv").set_index("station")
rows = []
for r in asg.itertuples():
    if isinstance(r.primary_gauge, str) and r.primary_gauge:
        i = inv.loc[r.primary_gauge]
        rows.append([r.district, r.status.replace("_", " "), f"{r.primary_gauge.strip().title()} ({str(i.river).strip()})",
                     f"{i.latitude:.3f}, {i.longitude:.3f}", str(pd.Timestamp(i.train_first_obs).date()),
                     f"{i.train_coverage_pct:.1f}%", f"{i.validation_coverage_pct:.1f}%", f"{i.test_coverage_pct:.1f}%",
                     (r.other_usable_gauges or "").replace(";", ", ").title() if isinstance(r.other_usable_gauges, str) else "—"])
    else:
        rows.append([r.district, r.status.replace("_", " "), "—", "—", "—", "0%", "—", "—", "—"])
out["river_districts"] = rows
out["river_n_gauges"] = int(len(inv))
out["river_n_usable"] = int(inv["usable_for_training"].sum())
brk = inv[inv["n_datum_breaks"] > 0]
out["river_breaks"] = [[s, str(b)] for s, b in brk["datum_break_months"].items() if "Reservoir" not in s]

q = pd.read_csv(U + "documentation/QUALITY_CHECKS.csv")
cov = q[q["check"].str.startswith("observed river_level_m")]
out["river_split_cov"] = [[c.check.split(" in ")[1].split(" window")[0], c.result] for c in cov.itertuples()]

# flood corrections
c = pd.read_csv(U + "flood_events/processed/IFI_Kerala_date_corrections.csv")
fl = pd.read_csv(U + "flood_events/processed/IFI_Kerala_2012_2023_corrected.csv")
fl18 = fl[((pd.to_datetime(fl["End Date"], format="%d-%m-%Y %H:%M") - pd.to_datetime(fl["Start Date"], format="%d-%m-%Y %H:%M")).dt.days + 1 - fl["Duration(Days)"]).abs() > 1]
flagged = set(fl18["UEI"])
status_map = {"CORRECTED_DAY_MONTH_SWAP": "CORRECTED (day/month swap)", "CORRECTED_WRONG_MONTH": "CORRECTED (wrong month)",
              "CORRECTED_INCOMPLETE": "CORRECTED (missing day)", "VERIFIED_NO_CHANGE": "VERIFIED – dates unchanged; non-contiguous days",
              "UNRESOLVED": "UNRESOLVED"}
def days_str(ev, s, e):
    if isinstance(ev, str) and ev and ev.count(";") >= 1:
        ds = ev.split(";")
        if len(ds) > 4:
            return f"{ds[0]} → {ds[-1]}"
        return ", ".join(ds)
    return s if s == e else f"{s} → {e}"
rows = []
for r in c.itertuples():
    orig = r.original_start[:10] if r.original_start[:10] == r.original_end[:10] else f"{r.original_start[:10]} → {r.original_end[:10]}"
    st = status_map.get(r.verification_status, r.verification_status)
    if r.verification_status == "VERIFIED_NO_CHANGE" and r.UEI == "UEI-IMD-FL-2018-0035":
        st = "VERIFIED – dates unchanged"
    rows.append([r.UEI.replace("UEI-IMD-FL-", "") + (" *" if r.UEI in flagged else ""),
                 str(r.Districts).strip() if isinstance(r.Districts, str) else "—",
                 orig, days_str(r.event_days, r.verified_start, r.verified_end),
                 f"IMD DWE {rep_year(r.evidence_report)}, PDF p.{int(r.evidence_pdf_page)}, Kerala entry {r.evidence_entry.split()[-1]}: “{' '.join(str(r.imd_date_text).split())}”",
                 st])
# add the flagged 2018-0035 (seasonal period, verified) which is not in the corrections file
x = fl[fl["UEI"] == "UEI-IMD-FL-2018-0035"].iloc[0]
rows.append(["2018-0035 *", "All 14 districts", "2018-05-29 → 2018-07-31", "2018-05-29 → 2018-07-31",
             f"IMD DWE {rep_year(x.evidence_report)}, PDF p.{int(x.evidence_pdf_page)}, Kerala entry {x.evidence_entry.split()[-1]}: “{x.imd_date_text}”",
             "VERIFIED – dates as in source (64-day seasonal period; IFI duration 34 inconsistent)"])
rows.sort(key=lambda r: r[0])
out["flood_corr_rows"] = rows
out["flood_status_counts"] = fl["verification_status"].value_counts().to_dict()
out["flood_flag_counts"] = fl["flags"].fillna("").str.split(" \\| ").explode().str.split(":").str[0].replace("", pd.NA).dropna().value_counts().to_dict()
out["flood_n_records"] = int(len(fl))
out["flood_corr_by_year"] = fl[fl["verification_status"].str.startswith("CORRECTED")]["UEI"].str[11:15].value_counts().sort_index().to_dict()
fcov = pd.read_csv(U + "flood_events/processed/Flood_label_coverage_summary.csv")
out["flood_cov"] = [[r.label_basis, f"{r.positive_district_days:,}", f"{100*r.positive_rate_on_66080_grid:.2f}%",
                     f"{r.train:,}", f"{r.validation:,}", f"{r.test:,}", r.label_source_ends] for r in fcov.itertuples()]
sup = pd.read_csv(U + "flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv")
out["flood_ext"] = [[r.category.replace("_", " ").title().replace("Ifi", "IFI"), f"IMD DWE {rep_year(r.report)}, p.{r.pdf_page}, #{r.entry.split()[-1]}",
                     " ".join(str(r.imd_date_text).split()), r.districts.replace(";", ", "), r.imd_intensity if isinstance(r.imd_intensity, str) else "—"]
                    for r in sup.itertuples()]

# landslides
g = pd.read_csv(U + "landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv")
s = g[g["in_2013_2024_file"]]
out["gsi_prec_2013"] = s["date_precision"].value_counts().to_dict()
out["gsi_prec_all"] = g["date_precision"].value_counts().to_dict()
out["gsi_time"] = int((s["time_of_day"].fillna("") != "").sum())
out["gsi_ambig"] = int(s["numeric_date_ambiguous_dm"].sum())
m = pd.read_csv(U + "landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv")
out["imd_ls_n"] = int(len(m))
out["imd_ls_attr"] = int((m["landslide_districts"].fillna("") != "").sum())
out["imd_ls_attr_exact"] = int(((m["landslide_districts"].fillna("") != "") & (m["date_precision"] == "EXACT_DATE")).sum())
lc = pd.read_csv(U + "landslide_events/processed/Landslide_label_coverage_summary.csv")
out["ls_cov"] = [[r.label_basis, str(r.positive_district_days), str(r.distinct_dates), str(r.train), str(r.validation), str(r.test)] for r in lc.itertuples()]
ex = m[(m["landslide_districts"].fillna("") != "") & (m["date_precision"] == "EXACT_DATE")]
out["imd_ls_exact"] = [[str(r.year), f"IMD DWE {rep_year(r.report)}, p.{r.page}, #{r.sno}", " ".join(str(r.date_text).split()), r.landslide_districts] for r in ex.itertuples()]

# lithology + QC
chk = pd.read_csv(U + "lithology/processed/Lithology_reproduction_check.csv")
out["lith_same"] = int(chk["same_dominant_class"].sum()); out["lith_maxdiff"] = float(chk["max_abs_fraction_diff"].max())
out["qc_counts"] = q["status"].value_counts().to_dict()
out["qc_warn"] = [[r.dataset.split("/")[-1], r.check, str(r.result)[:240]] for r in q[q["status"] != "PASS"].itertuples()]
out["qc_integrity"] = int((q["check"].str.startswith("integrity") & (q["status"] == "PASS")).sum())
json.dump(out, open("report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()})
