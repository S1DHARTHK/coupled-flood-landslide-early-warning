"""
Process CWC observed river water level (NWDP) for Kerala.

Inputs (originals, never modified):
  river_level/cwc_nwdp_source/rwl_manual_hr_cwc_031_1991_2020.csv
  river_level/cwc_nwdp_source/rwl_manual_hr_cwc_031_2021_2025.csv
  reference/geoBoundaries-IND-ADM2_simplified.geojson, reference/Kerala_district_reference.csv

Outputs (river_level/processed/):
  CWC_Kerala_station_inventory.csv       one row per Kerala CWC gauge: location, datum QC,
                                         coverage per model split, training usability
  CWC_Kerala_station_yearly_median.csv   yearly median stage per gauge (after datum QC)
  CWC_Kerala_datum_breaks.csv            detected reference-level breaks per gauge
  CWC_Kerala_RiverLevel_Daily_Station.csv daily stage per gauge (observed readings only)
  District_Gauge_Assignment.csv          which gauge may represent which district, and why

Rules
  * Observed values only: no interpolation, gap filling, datum conversion, or discharge-to-
    stage conversion. Every value in the daily table is an aggregate of that day's readings.
  * Duplicate readings (same station and timestamp) that agree within 5 mm are kept once.
  * Datum QC. CWC publishes some readings as the gauge reading and others as gauge reading +
    RL of gauge zero (the pairs differ by exactly RL_of_zeroGauge). Where |RL| >= 10 m the two
    references separate cleanly, so readings on the station's minority reference are REMOVED
    (not converted). Where |RL| < 10 m the two cannot be told apart from values alone; this
    is reported as a limitation.
    For 0.5 <= |RL| < 10 m, short runs (<= 30 readings) bracketed by jumps of exactly +RL and
    -RL are readings on the other reference and are removed.
  * Reference breaks are detected as month-to-month median jumps > 5 m. A block of <= 6 months
    bracketed by two opposite breaks, after which the series returns to within 3 m of its earlier
    level, is a temporary change of reference and is removed. Any other break splits the series
    into datum-consistent segments; values from different segments must
    not be compared.
  * District attribution uses CWC's own District / LGD code (official). The point-in-polygon
    test against the simplified geoBoundaries layer is reported as a cross-check only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point

U = Path(__file__).resolve().parents[2] / "collected_datasets"
SRC = U / "river_level" / "cwc_nwdp_source"
OUT = U / "river_level" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

LEVEL = "River Water Level Manual Hourly (meter)"
USECOLS = ["Station", "Agency", "State", "District LGD Code", "District", "Tehsil", "River",
           "Basin", "Latitude", "Longitude", "Is_DischargeDataAvailable", "RL_of_zeroGauge",
           "MeanSeaLevel", "Data Acquisition Time", LEVEL]
DUP_TOL_M = 0.005
RL_SEPARABLE_M = 10.0
BREAK_JUMP_M = 5.0
STEP_REVIEW_M = 0.7
RL_RUN_MAX_READINGS = 30        # max length of an RL-offset run that is removed
TRAIN_MIN_COVERAGE = 40.0          # % of training days with an observation

# Kerala LGD district codes -> project (CHIRPS) district names
LGD = {554: "Alappuzha", 555: "Ernakulam", 556: "Idukki", 557: "Kannur", 558: "Kasaragod",
       559: "Kollam", 560: "Kottayam", 561: "Kozhikode", 562: "Malappuram", 563: "Palakkad",
       564: "Pattanamtitta", 565: "Thiruvananthapuram", 566: "Thrissur", 567: "Wayanad"}

# --- Split windows, reproduced from ml/flood_xgboost.py chronological_split() ---------
# 70/15/15 on unique-date boundaries, applied to the only real daily grid that can
# currently be built: the CHIRPS window after the 29-day rainfall_30d_mm warm-up.
GRID = pd.date_range("2012-01-30", "2024-12-31", freq="D")
n = len(GRID)
TRAIN_END = GRID[int(round(n * 0.70)) - 1]
VAL_END = GRID[int(round(n * 0.85)) - 1]
SPLITS = {"train": (GRID[0], TRAIN_END),
          "validation": (TRAIN_END + pd.Timedelta(days=1), VAL_END),
          "test": (VAL_END + pd.Timedelta(days=1), GRID[-1])}


def load():
    frames = []
    for f in ["rwl_manual_hr_cwc_031_1991_2020.csv", "rwl_manual_hr_cwc_031_2021_2025.csv"]:
        d = pd.read_csv(SRC / f, usecols=USECOLS, low_memory=False)
        d["source_file"] = f
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[d["State"].str.strip().str.lower() == "kerala"].copy()
    d["timestamp"] = pd.to_datetime(d["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
    d["date"] = d["timestamp"].dt.normalize()
    d[LEVEL] = pd.to_numeric(d[LEVEL], errors="coerce")
    d["district"] = d["District LGD Code"].map(LGD)
    return d.dropna(subset=[LEVEL])


def district_polygons():
    ref = pd.read_csv(U / "reference" / "Kerala_district_reference.csv")
    gb = json.load(open(U / "reference" / "geoBoundaries-IND-ADM2_simplified.geojson", encoding="utf-8"))
    ids = dict(zip(ref["shape_id"], ref["district_chirps"]))
    return {ids[f["properties"]["shapeID"]]: shape(f["geometry"])
            for f in gb["features"] if f["properties"]["shapeID"] in ids}


def datum_qc(g: pd.DataFrame):
    """Return (kept readings, qc dict) for one station."""
    rl = g["RL_of_zeroGauge"].abs().max()
    qc = dict(rl_of_zero_gauge_m=",".join(sorted(g["RL_of_zeroGauge"].astype(str).unique())),
              rl_separable=bool(rl >= RL_SEPARABLE_M), dominant_reference="",
              minority_reference_readings_removed=0)
    if rl >= RL_SEPARABLE_M:
        msl_like = g[LEVEL] >= g["RL_of_zeroGauge"].abs() / 2
        dom_msl = msl_like.mean() >= 0.5
        keep = msl_like if dom_msl else ~msl_like
        qc["dominant_reference"] = "gauge reading + RL of gauge zero" if dom_msl else "gauge reading"
        qc["minority_reference_readings_removed"] = int((~keep).sum())
        g = g[keep]
    else:
        qc["dominant_reference"] = "not separable (|RL| < 10 m)"
    # duplicates: agree within tolerance -> keep one; disagree -> drop all copies
    g = g.sort_values("timestamp")
    spread = g.groupby("timestamp")[LEVEL].transform(lambda s: s.max() - s.min())
    qc["conflicting_duplicate_readings_removed"] = int((spread > DUP_TOL_M).sum())
    g = g[spread <= DUP_TOL_M].drop_duplicates("timestamp")
    # |RL| in [0.5, 10): short runs bracketed by a jump of exactly +RL and a jump of exactly
    # -RL (or the reverse) are readings on the other reference -> removed, not converted
    qc["rl_offset_run_readings_removed"] = 0
    if 0.5 <= rl < RL_SEPARABLE_M and len(g) > 2:
        x = g[LEVEL].to_numpy()
        dx = np.diff(x)
        hit = np.where(np.isclose(np.abs(dx), rl, atol=0.011))[0]
        drop = np.zeros(len(x), dtype=bool)
        for k, i in enumerate(hit):
            for j in hit[k + 1:]:
                if j - i > RL_RUN_MAX_READINGS:
                    break
                if np.sign(dx[j]) != np.sign(dx[i]):
                    drop[i + 1:j + 1] = True
                    break
        qc["rl_offset_run_readings_removed"] = int(drop.sum())
        g = g[~drop]
    return g, qc


def detect_breaks(g: pd.DataFrame):
    m = g.set_index("timestamp")[LEVEL].resample("MS").median().dropna()
    jumps = m.diff().abs()
    br = jumps[jumps > BREAK_JUMP_M]
    return [(ts, m.shift(1)[ts], m[ts]) for ts in br.index]


def main():
    d = load()
    polys = district_polygons()
    kept, inv_rows, brk_rows = [], [], []

    for st, g in d.groupby("Station"):
        g, qc = datum_qc(g)
        breaks = detect_breaks(g)
        # a short block (<= 6 months) bracketed by two opposite breaks, after which the series
        # returns to within 3 m of its earlier level, is a temporary change of reference:
        # its readings are removed and the two breaks disappear
        qc["bracketed_block_readings_removed"] = 0
        changed = True
        while changed:
            changed = False
            for k in range(len(breaks) - 1):
                (t0, before0, _), (t1, _, after1) = breaks[k], breaks[k + 1]
                if (t1 - t0).days <= 184 and abs(before0 - after1) <= 3.0:
                    blk = (g["timestamp"] >= t0) & (g["timestamp"] < t1)
                    qc["bracketed_block_readings_removed"] += int(blk.sum())
                    g = g[~blk]
                    breaks = detect_breaks(g)
                    changed = True
                    break
        seg = np.zeros(len(g), dtype=int)
        for k, (ts, _, _) in enumerate(breaks, 1):
            seg[(g["timestamp"] >= ts).to_numpy()] = k
        g = g.assign(datum_segment=seg)
        kept.append(g)
        for ts, before, after in breaks:
            brk_rows.append(dict(station=st, break_month=ts.strftime("%Y-%m"),
                                 median_before_m=round(before, 3), median_after_m=round(after, 3)))

        lat, lon = g["Latitude"].iloc[0], g["Longitude"].iloc[0]
        poly_d = next((k for k, p in polys.items() if p.contains(Point(lon, lat))), None)
        ym = g.groupby(g["timestamp"].dt.year)[LEVEL].median()
        steps = ym.diff().abs()
        small_steps = [f"{y}:{steps[y]:+.2f}" for y in steps.index
                       if STEP_REVIEW_M < steps[y] <= BREAK_JUMP_M]
        r = dict(station=st, agency=g["Agency"].iloc[0], district=g["district"].iloc[0],
                 cwc_district_text=g["District"].iloc[0], district_lgd_code=g["District LGD Code"].iloc[0],
                 polygon_district_check=poly_d, polygon_agrees=(poly_d == g["district"].iloc[0]),
                 river=str(g["River"].iloc[0]).strip(), basin=g["Basin"].iloc[0],
                 latitude=lat, longitude=lon,
                 is_reservoir_level="reservoir" in st.lower(),
                 discharge_also_observed=g["Is_DischargeDataAvailable"].iloc[0],
                 first_obs=g["timestamp"].min(), last_obs=g["timestamp"].max(),
                 n_readings_kept=len(g), **qc,
                 n_datum_breaks=len(breaks),
                 datum_break_months=";".join(b[0].strftime("%Y-%m") for b in breaks),
                 yearly_median_steps_0p7_to_5m=";".join(small_steps),
                 level_min_m=g[LEVEL].min(), level_median_m=g[LEVEL].median(), level_max_m=g[LEVEL].max())
        days = set(g["date"])
        for s, (a, b) in SPLITS.items():
            got = sum(1 for x in days if a <= x <= b)
            r[f"{s}_days_observed"] = got
            r[f"{s}_coverage_pct"] = round(100 * got / ((b - a).days + 1), 2)
            r[f"{s}_datum_breaks"] = sum(1 for ts, _, _ in breaks if a < ts <= b)
        a, b = SPLITS["train"]
        tr = g[(g["date"] >= a) & (g["date"] <= b)]
        r["train_first_obs"] = tr["date"].min()
        r["train_segments"] = tr["datum_segment"].nunique()
        reasons = []
        if r["is_reservoir_level"]: reasons.append("reservoir level, not river stage")
        if r["train_coverage_pct"] < TRAIN_MIN_COVERAGE: reasons.append(f"training coverage < {TRAIN_MIN_COVERAGE:.0f}%")
        if r["train_segments"] > 1: reasons.append("datum break inside training window")
        r["usable_for_training"] = not reasons
        r["not_usable_reason"] = "; ".join(reasons)
        inv_rows.append(r)

    inv = pd.DataFrame(inv_rows).sort_values(["district", "station"])
    inv.to_csv(OUT / "CWC_Kerala_station_inventory.csv", index=False)
    pd.DataFrame(brk_rows).to_csv(OUT / "CWC_Kerala_datum_breaks.csv", index=False)
    d = pd.concat(kept)

    ym = (d.assign(year=d["timestamp"].dt.year)
            .groupby(["Station", "year"])[LEVEL].agg(["median", "min", "max", "size"])
            .reset_index().rename(columns={"size": "n_readings"}))
    ym.to_csv(OUT / "CWC_Kerala_station_yearly_median.csv", index=False)

    grp = d.groupby(["Station", "date"])
    daily = grp[LEVEL].agg(level_mean_m="mean", level_max_m="max", level_min_m="min",
                           n_readings="size").reset_index()
    daily["datum_segment"] = grp["datum_segment"].max().to_numpy()
    at08 = d[d["timestamp"].dt.hour == 8].groupby(["Station", "date"])[LEVEL].first().rename("level_0800_m")
    daily = daily.merge(at08, on=["Station", "date"], how="left")
    daily = daily.merge(inv[["station", "district", "river"]], left_on="Station",
                        right_on="station", how="left").drop(columns="station")
    daily = daily.rename(columns={"Station": "station"})
    daily = daily[["district", "station", "river", "date", "level_mean_m", "level_max_m",
                   "level_min_m", "level_0800_m", "n_readings", "datum_segment"]]
    daily.sort_values(["station", "date"]).to_csv(OUT / "CWC_Kerala_RiverLevel_Daily_Station.csv", index=False)

    # --- district assignment ------------------------------------------------------------
    rows = []
    for dist in sorted(LGD.values()):
        c = inv[(inv["district"] == dist) & (~inv["is_reservoir_level"])]
        usable = c[c["usable_for_training"]].sort_values(["train_coverage_pct", "first_obs"],
                                                         ascending=[False, True])
        if c.empty:
            status, prim, why = "NO_CWC_GAUGE", "", "No CWC river-stage gauge is attributed to this district."
        elif usable.empty:
            status, prim = "NO_USABLE_TRAINING_DATA", ""
            why = "CWC gauge(s) exist but none has a datum-consistent record covering >= 40% of the training window."
        else:
            p = usable.iloc[0]
            prim = p["station"]
            status = "REPRESENTED_FULL" if p["train_coverage_pct"] >= 90 else "REPRESENTED_PARTIAL"
            why = (f"{prim} is a CWC river-stage gauge on the {p['river']} located in {dist} "
                   f"(CWC LGD district {p['district_lgd_code']}); it has the longest datum-consistent "
                   f"record in the training window ({p['train_coverage_pct']}% of days, first obs "
                   f"{pd.Timestamp(p['train_first_obs']).date()}). A point gauge measures one river "
                   f"reach, not the whole district.")
        rows.append(dict(district=dist, status=status, primary_gauge=prim,
                         primary_river=(usable.iloc[0]["river"] if prim else ""),
                         primary_train_coverage_pct=(usable.iloc[0]["train_coverage_pct"] if prim else 0),
                         primary_train_first_obs=(usable.iloc[0]["train_first_obs"] if prim else ""),
                         other_usable_gauges=";".join(usable["station"].iloc[1:]),
                         gauges_not_usable_for_training=";".join(
                             f"{s} ({why_})" for s, why_ in zip(c.loc[~c["usable_for_training"], "station"],
                                                               c.loc[~c["usable_for_training"], "not_usable_reason"])),
                         justification=why))
    asg = pd.DataFrame(rows)
    asg.to_csv(OUT / "District_Gauge_Assignment.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("Split windows:", {k: (str(a.date()), str(b.date())) for k, (a, b) in SPLITS.items()})
    print(inv[["station", "district", "polygon_agrees", "river", "rl_of_zero_gauge_m", "dominant_reference",
               "minority_reference_readings_removed", "conflicting_duplicate_readings_removed",
               "datum_break_months", "train_coverage_pct", "usable_for_training"]].to_string(index=False))
    print(asg[["district", "status", "primary_gauge", "primary_train_coverage_pct", "other_usable_gauges"]].to_string(index=False))


if __name__ == "__main__":
    main()
