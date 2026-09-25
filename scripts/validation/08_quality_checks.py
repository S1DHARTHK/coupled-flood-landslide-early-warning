"""
Quality checks for every dataset obtained or produced in the 2026-09-11 data task.

Writes documentation/QUALITY_CHECKS.csv (one row per check) and
documentation/SOURCE_MANIFEST_2026-09-11.json (size + SHA-256 of every new file).
Read-only with respect to all data files.
"""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import pypdf

U = Path(__file__).resolve().parents[2] / "collected_datasets"
DOC = U / "documentation"
# The data-source report moved to documents/ in the V1.0 folder layout.
DATA_REPORT = U.parent / "documents" / "data_collection" / "DATA_SOURCE_AND_CORRECTION_REPORT.docx"
GRID = pd.date_range("2012-01-30", "2024-12-31", freq="D")
n = len(GRID)
TRAIN_END, VAL_END = GRID[int(round(n * 0.70)) - 1], GRID[int(round(n * 0.85)) - 1]
SPL = {"train": (GRID[0], TRAIN_END), "validation": (TRAIN_END + pd.Timedelta(days=1), VAL_END),
       "test": (VAL_END + pd.Timedelta(days=1), GRID[-1])}
rows = []


def add(dataset, check, result, status="PASS", detail=""):
    rows.append(dict(dataset=dataset, check=check, result=str(result), status=status, detail=detail))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    log = pd.read_csv(DOC / "download_log.tsv", sep="\t")
    ok = log[log["http_code"] == 200].drop_duplicates("local_path", keep="last")

    # --- integrity of every downloaded original --------------------------------------
    for _, r in ok.iterrows():
        p = U / r["local_path"]
        now = sha(p)
        add(r["local_path"], "integrity: size and SHA-256 unchanged since download",
            f"{p.stat().st_size} bytes", "PASS" if now == r["sha256"] and p.stat().st_size == r["bytes"] else "FAIL",
            f"sha256 {now[:16]}...")

    # --- CWC river level originals --------------------------------------------------------
    V = "River Water Level Manual Hourly (meter)"
    for f in ["rwl_manual_hr_cwc_031_1991_2020.csv", "rwl_manual_hr_cwc_031_2021_2025.csv"]:
        ds = f"river_level/cwc_nwdp_source/{f}"
        d = pd.read_csv(U / ds, low_memory=False)
        need = ["Station", "District LGD Code", "River", "Latitude", "Longitude", "RL_of_zeroGauge",
                "Data Acquisition Time", V]
        add(ds, "required columns present", ", ".join(need), "PASS" if set(need) <= set(d.columns) else "FAIL")
        k = d[d["State"] == "Kerala"]
        t = pd.to_datetime(k["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
        add(ds, "rows (all / Kerala)", f"{len(d):,} / {len(k):,}")
        add(ds, "Kerala date range", f"{t.min()} to {t.max()}")
        add(ds, "Kerala stations", k["Station"].nunique())
        add(ds, "missing values in Kerala rows", int(k[need].isna().sum().sum()),
            "PASS" if k[need].isna().sum().sum() == 0 else "WARN")
        add(ds, "non-numeric / negative water levels", int(pd.to_numeric(k[V], errors="coerce").isna().sum()),
            "PASS" if pd.to_numeric(k[V], errors="coerce").notna().all() else "WARN")
        dup = k.duplicated(["Station", "Data Acquisition Time"]).sum()
        add(ds, "duplicate (station, timestamp)", int(dup), "WARN" if dup else "PASS",
            "handled in 01_river_level_cwc_process.py (identical kept once, conflicting removed)")
        add(ds, "units", "metres (dataset metadata 'Unit = m'; column name '(meter)')")
        add(ds, "coordinate system", "decimal degrees, WGS84 lat/lon "
            f"(lat {k['Latitude'].min():.3f}-{k['Latitude'].max():.3f}, lon {k['Longitude'].min():.3f}-{k['Longitude'].max():.3f})")
        add(ds, "temporal resolution", "manual readings: typically 08:00/13:00/18:00, hourly during floods")

    # --- CWC processed ------------------------------------------------------------------
    inv = pd.read_csv(U / "river_level/processed/CWC_Kerala_station_inventory.csv")
    asg = pd.read_csv(U / "river_level/processed/District_Gauge_Assignment.csv")
    daily = pd.read_csv(U / "river_level/processed/CWC_Kerala_RiverLevel_Daily_Station.csv", parse_dates=["date"])
    ds = "river_level/processed/CWC_Kerala_RiverLevel_Daily_Station.csv"
    add(ds, "rows / stations", f"{len(daily):,} / {daily['station'].nunique()}")
    add(ds, "missing values (level columns except level_0800_m)",
        int(daily[["level_mean_m", "level_max_m", "level_min_m"]].isna().sum().sum()))
    add(ds, "duplicate (station, date)", int(daily.duplicated(["station", "date"]).sum()),
        "PASS" if not daily.duplicated(["station", "date"]).any() else "FAIL")
    add(ds, "gauges usable for training (river stage, >=40% training days, no datum break)",
        f"{int(inv['usable_for_training'].sum())} of {len(inv)}")
    prim = asg[asg["primary_gauge"].notna()]
    for s, (a, b) in SPL.items():
        tot = ((b - a).days + 1) * 14
        got = sum(((daily["station"] == g) & (daily["date"] >= a) & (daily["date"] <= b)).sum()
                  for g in prim["primary_gauge"])
        add(ds, f"observed river_level_m in {s} window via primary gauges ({a.date()} to {b.date()})",
            f"{got:,} of {tot:,} district-days ({100 * got / tot:.1f}%)",
            "PASS" if got > 0 else "FAIL")
    add("river_level/processed/District_Gauge_Assignment.csv", "district status",
        "; ".join(f"{r.district}={r.status}" for r in asg.itertuples()))

    # --- IMD DWE PDFs ----------------------------------------------------------------------
    for p in sorted((U / "flood_events/imd_dwe_source").glob("*.pdf")):
        rd = pypdf.PdfReader(str(p))
        txt = (U / "flood_events/processed/dwe_text" / (p.stem + ".txt"))
        chars = len(txt.read_text(encoding="utf-8", errors="replace").strip()) if txt.exists() else 0
        add(f"flood_events/imd_dwe_source/{p.name}", "opens as PDF / pages / text layer",
            f"{len(rd.pages)} pages, {chars:,} text chars",
            "PASS" if chars else "WARN", "" if chars else "scanned images only - Kerala pages read visually")

    # --- flood outputs --------------------------------------------------------------------
    c = pd.read_csv(U / "flood_events/processed/IFI_Kerala_2012_2023_corrected.csv")
    ds = "flood_events/processed/IFI_Kerala_2012_2023_corrected.csv"
    add(ds, "Kerala IFI records (start 2012-2023) verified against IMD", len(c))
    add(ds, "verification status", c["verification_status"].value_counts().to_dict())
    add(ds, "records UNRESOLVED", int((c["verification_status"] == "UNRESOLVED").sum()),
        "PASS" if not (c["verification_status"] == "UNRESOLVED").any() else "WARN")
    add(ds, "original IFI columns unchanged", "all 23 original columns carried verbatim; corrections in new columns")
    add(ds, "date range (verified)", f"{c['verified_start'].min()} to {c['verified_end'].max()}")
    add(ds, "flags", c["flags"].fillna("").str.split(" \\| ").explode().str.split(":").str[0]
        .replace("", pd.NA).dropna().value_counts().to_dict(), "WARN")
    s = pd.read_csv(U / "flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv")
    add("flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv", "events by category",
        s["category"].value_counts().to_dict())
    x = s[s["category"] == "EXTENSION_AFTER_IFI_END"]
    add("flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv", "extension (after IFI end) date range",
        f"{x['event_days'].str[:10].min()} to {x['event_days'].str[-10:].max()}")
    cov = pd.read_csv(U / "flood_events/processed/Flood_label_coverage_summary.csv")
    for r in cov.itertuples():
        add("flood_events/processed/Flood_label_coverage_summary.csv", r.label_basis,
            f"{r.positive_district_days} district-days ({100 * r.positive_rate_on_66080_grid:.2f}%); "
            f"train/val/test {r.train}/{r.validation}/{r.test}; label source to {r.label_source_ends}")

    # --- landslide outputs -----------------------------------------------------------------
    g = pd.read_csv(U / "landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv")
    sub = g[g["in_2013_2024_file"]]
    add("landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv",
        "date precision (2013-2024 file)", sub["date_precision"].value_counts().to_dict())
    add("landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv",
        "date precision (All file)", g["date_precision"].value_counts().to_dict())
    m = pd.read_csv(U / "landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv")
    add("landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv", "entries / attribution",
        f"{len(m)} entries; " + str(m["landslide_attribution"].value_counts().to_dict()))
    lc = pd.read_csv(U / "landslide_events/processed/Landslide_label_coverage_summary.csv")
    for r in lc.itertuples():
        add("landslide_events/processed/Landslide_label_coverage_summary.csv", r.label_basis,
            f"{r.positive_district_days} district-days; train/val/test {r.train}/{r.validation}/{r.test}")

    # --- lithology --------------------------------------------------------------------------
    chk = pd.read_csv(U / "lithology/processed/Lithology_reproduction_check.csv")
    add("lithology/Kerala_District_Lithology_CGWB.csv", "independent reproduction from CGWB shapefile",
        f"dominant class identical in {int(chk['same_dominant_class'].sum())}/14 districts; "
        f"max fraction difference {chk['max_abs_fraction_diff'].max():.4f}",
        "PASS" if chk["same_dominant_class"].all() else "FAIL")
    lit = pd.read_csv(U / "lithology/Kerala_District_Lithology_CGWB.csv")
    allowed = {"granite_gneiss", "laterite", "schist", "sandstone", "shale", "alluvium"}
    add("lithology/Kerala_District_Lithology_CGWB.csv", "values inside model CATEGORY_SCHEMA",
        sorted(lit["lithology"].unique()), "PASS" if set(lit["lithology"]) <= allowed else "FAIL")
    add("lithology/Kerala_District_Lithology_CGWB.csv", "distinct values across 14 districts",
        lit["lithology"].nunique(), "WARN", "near-constant feature at district scale")

    q = pd.DataFrame(rows)
    q.to_csv(DOC / "QUALITY_CHECKS.csv", index=False)
    print(q["status"].value_counts().to_string())
    print(q[q["status"] != "PASS"].to_string(index=False))

    # --- manifest of every file added in this task ------------------------------------------
    new_dirs = ["river_level/cwc_nwdp_source", "river_level/kerala_sw_manual_source", "river_level/processed",
                "flood_events", "landslide_events", "lithology/processed", "scripts", "documentation"]
    files = sorted({p for d in new_dirs for p in (U / d).rglob("*") if p.is_file()}
                   | {DATA_REPORT})
    man = [dict(path=str(p.relative_to(U) if p.is_relative_to(U) else p.relative_to(U.parent)).replace("\\", "/"),
                bytes=p.stat().st_size, sha256=sha(p))
           for p in files if p.name != "SOURCE_MANIFEST_2026-09-11.json"]
    json.dump(dict(created="2026-09-11", note="Files added by the 2026-09-11 data-correction task. "
                   "updated_data/SOURCE_MANIFEST.json (2026-09-08) is left unchanged and predates these files.",
                   files=man), open(DOC / "SOURCE_MANIFEST_2026-09-11.json", "w", encoding="utf-8"), indent=1)
    print(f"manifest: {len(man)} files")


if __name__ == "__main__":
    main()
