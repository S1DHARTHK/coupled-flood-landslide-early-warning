"""
13 - Build the REAL Master Dataset (Kerala Coupled Flood-Landslide Early Warning System).

Specification: documents/master_dataset/MASTER_DATASET_BUILD_PROMPT.md (schema, sources, the four
agreed decisions, validation). Re-run this script whenever an input changes, e.g. after a GSI Bhukosh
landslide inventory has been processed into landslide_events/processed/:

    python scripts/data/13_build_master_dataset.py

Inputs (read-only, under collected_datasets/):
  reference/Kerala_district_reference.csv                       district vocabulary + representative point
  existing_collected_data/Kerala_CHIRPS_District_Rainfall_2012.csv, _2013_2024.csv
  weather_soil/NASA_POWER_Kerala_District_Daily_2012_2024.csv   GWETROOT, T2M, RH2M
  terrain/Kerala_District_Terrain_Summary.csv, terrain/Kerala_District_LandCover_Summary.csv
  hydrology/Kerala_District_Drainage_Density.csv, hydrology/Kerala_HydroRIVERS_v10.geojson
  lithology/Kerala_District_Lithology_CGWB.csv
  river_level/processed/District_Gauge_Assignment.csv, CWC_Kerala_RiverLevel_Daily_Station.csv
  flood_events/processed/IFI_Kerala_2012_2023_corrected.csv, IMD_DWE_Kerala_events_not_in_IFI.csv
  landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv, IMD_DWE_Kerala_landslide_entries.csv,
  landslide_events/processed/VERIFIED_web_landslide_entries.csv (web-verified district-days from the
  2018-2024 verification passes; written by scripts/06b_verified_landslide_entries.py),
  landslide_events/processed/FINAL_2012_2017_landslide_label_updates.csv (frozen 2012-2017
  verification: ADD rows are verified district-days, REMOVE rows are rejected ones whose
  source records are dropped)
  ../synthetic_master_dataset.csv (header only - column order), ../ml/*.py (split + validate_columns)

Outputs (collected_datasets/master_dataset/; REAL_DATA_DICTIONARY.md + MASTER_DATASET_REPORT.docx go to
documents/master_dataset/; all overwritten on re-run - they are products of this script. The frozen V1.0
master used by the models is dataset/real_master_dataset.csv and is never written here):
  real_master_dataset.csv        23 columns = the 22 synthetic columns + `district` after `longitude`
  real_master_dataset_keys.csv   same row order: gauge, published stage, label sources and flags
  VALIDATION_RESULTS.csv         every check with expected / observed / status
  REAL_DATA_DICTIONARY.md        column-by-column documentation with the numbers of this build
  MASTER_DATASET_REPORT.docx     short report

Rules: no model is trained; no input file is modified; nothing is synthetic, estimated, interpolated,
forward-filled or imputed (missing stays NaN); no date is assigned to a record that does not state one;
no random negative sampling; every value computed from the data (gauge medians) uses TRAINING dates only.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"          # collected_datasets/
ROOT = U.parent                                   # project root
OUT = U / "master_dataset"
DOCS = ROOT / "documents" / "master_dataset"      # generated documentation

# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
WARMUP_START = pd.Timestamp("2012-01-01")         # 29-day warm-up for rainfall_30d_mm (dropped)
GRID_START = pd.Timestamp("2012-01-30")
GRID_END = pd.Timestamp("2024-12-31")
WINDOWS = {3: "rainfall_3d_mm", 7: "rainfall_7d_mm", 14: "rainfall_14d_mm", 30: "rainfall_30d_mm"}
TRAIN_FRACTION, VALIDATION_FRACTION = 0.70, 0.15  # identical to ml/flood_xgboost.py and ml/landslide_xgboost.py
METRIC_CRS = "EPSG:32643"                         # WGS 84 / UTM zone 43N (Kerala lies in 74.8-77.4 E)
POWER_FILL = -990.0                               # NASA POWER missing-data sentinel is -999

# A GSI exact date is used only if the record's own History text states that day. The upstream parser
# (scripts/06) can split a 4-digit year into two "days" ("2014, June 2018" -> 14 and 20 June 2018);
# such dates were never stated by the record, so using them would break the hard rule
# "never assign a date to a record that does not state one". Set False only to reproduce the raw column.
GSI_REQUIRE_DATE_STATED = True

# IMD_DWE_Kerala_events_not_in_IFI.csv holds 24 EXTENSION_AFTER_IFI_END events and 1 OMITTED_FROM_IFI event
# (2023-06-28, Ernakulam/Kannur/Kasaragod/Kozhikode) that IMD reports but IFI missed. Decision 2 includes the file;
# keeping the omitted event is what makes "0 = no event reported by IMD" true for those 4 district-days.
# Documented pre-exclusion totals (flood_events/processed/Flood_label_coverage_summary.csv):
#   with the omitted event    3,136 (2,764 / 249 / 123)  "IFI corrected + all IMD DWE Kerala entries missing from IFI"
#   extension events only     3,132 (2,764 / 249 / 119)  "IFI corrected + IMD DWE events after 2023-07-23"
INCLUDE_OMITTED_FROM_IFI = True

LAND_COVER_CATEGORIES = ["forest", "agriculture", "shrubland", "built_up", "barren"]
LITHOLOGY_CATEGORIES = ["granite_gneiss", "laterite", "schist", "sandstone", "shale", "alluvium"]

SYNTHETIC_COLUMNS = ["date", "latitude", "longitude", "rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm",
                     "rainfall_14d_mm", "rainfall_30d_mm", "soil_moisture", "temperature_c", "humidity_percent",
                     "elevation_m", "river_level_m", "distance_to_river_km", "drainage_density", "slope_degree",
                     "aspect_degree", "curvature", "land_cover", "lithology", "flood", "landslide"]
MASTER_COLUMNS = SYNTHETIC_COLUMNS[:3] + ["district"] + SYNTHETIC_COLUMNS[3:]
FEATURE_COLUMNS = [c for c in SYNTHETIC_COLUMNS if c not in ("date", "latitude", "longitude", "flood", "landslide")]

# Numbers stated in the build prompt / earlier reports, reproduced (not assumed) below.
EXPECTED = dict(rows=66080, train_end="2021-02-14", val_end="2023-01-23",
                flood_before_exclusions=(3136, 2764, 249, 123) if INCLUDE_OMITTED_FROM_IFI else (3132, 2764, 249, 119),
                flood_before_prompt=(3132, 2764, 249, 119), landslide=(120, 101, 11, 8),
                landslide_before_verification=114, landslide_verified_added=13,
                landslide_2012_2017_added=6, landslide_2012_2017_removed=1,
                river_coverage_pct=(54.4, 75.0, 78.3))

GSI_ALIASES = {"IDUKKI": "Idukki", "Idukki (Devikulam Taluk)": "Idukki", "Plakkad": "Palakkad",
               "Pathanamthitta": "Pattanamtitta"}
MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                    "oct", "nov", "dec"], 1)}
MONTH = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
# Same grammar as scripts/06 TEXT_DATE, but a day number may not be glued to other digits.
TEXT_DATE_STRICT = re.compile(
    rf"((?:(?<!\d)\d{{1,2}}(?!\d)(?:st|nd|rd|th)?\s*(?:&|and|,|-|to)?\s*)+)(?:of\s+)?{MONTH},?\s*(\d{{4}})", re.I)
NUM_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------
# District vocabulary
# ---------------------------------------------------------------------------------------------
def load_reference() -> tuple[pd.DataFrame, dict]:
    ref = pd.read_csv(U / "reference" / "Kerala_district_reference.csv")
    ref = ref.rename(columns={"district_chirps": "district"})
    assert len(ref) == 14 and ref["district"].is_unique, "district reference must hold 14 unique districts"
    names = {}
    for chirps, gb in zip(ref["district"], ref["district_geoboundaries"]):
        names[chirps] = chirps
        names[gb] = chirps
    names.update(GSI_ALIASES)
    return ref[["district", "district_geoboundaries", "latitude", "longitude"]], names


def to_chirps(name, names: dict) -> str:
    key = str(name).strip()
    if key not in names:
        raise KeyError(f"District name {name!r} is not in the reference vocabulary")
    return names[key]


def split_names(cell, names: dict, seps=",;") -> list[str]:
    if pd.isna(cell):
        return []
    parts = re.split(f"[{re.escape(seps)}]", str(cell))
    return sorted({to_chirps(p, names) for p in parts if p.strip()})


# ---------------------------------------------------------------------------------------------
# Split (replica of chronological_split in ml/flood_xgboost.py; the real function is also run)
# ---------------------------------------------------------------------------------------------
def split_bounds(dates) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = np.array(sorted(pd.unique(pd.Series(dates))))
    n_dates = len(unique_dates)
    train_end_i = int(round(n_dates * TRAIN_FRACTION)) - 1
    val_end_i = int(round(n_dates * (TRAIN_FRACTION + VALIDATION_FRACTION))) - 1
    return pd.Timestamp(unique_dates[train_end_i]), pd.Timestamp(unique_dates[val_end_i])


def split_of(d: pd.Series, train_end, val_end) -> pd.Series:
    return pd.Series(np.where(d <= train_end, "train", np.where(d <= val_end, "validation", "test")), index=d.index)


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
def build_rainfall(districts: list[str]) -> pd.DataFrame:
    src = U / "existing_collected_data"
    c = pd.concat([pd.read_csv(src / "Kerala_CHIRPS_District_Rainfall_2012.csv"),
                   pd.read_csv(src / "Kerala_CHIRPS_District_Rainfall_2013_2024.csv")], ignore_index=True)
    c["date"] = pd.to_datetime(c["date"])
    c = c.rename(columns={"rainfall": "rainfall_1d_mm"})
    assert set(c["district"]) == set(districts), "CHIRPS district vocabulary differs from the reference"
    assert not c.duplicated(["date", "district"]).any(), "duplicate CHIRPS (date, district)"
    full = pd.MultiIndex.from_product([districts, pd.date_range(WARMUP_START, GRID_END, freq="D")],
                                      names=["district", "date"]).to_frame(index=False)
    c = full.merge(c, on=["district", "date"], how="left")   # a missing source day would surface as NaN
    c = c.sort_values(["district", "date"], kind="mergesort").reset_index(drop=True)
    out = []
    for _, g in c.groupby("district", sort=False):
        g = g.copy()
        x = g["rainfall_1d_mm"].to_numpy(float)
        for w, col in WINDOWS.items():
            # trailing window ending on day t (inclusive); min_periods = window -> first w-1 days NaN.
            # Each window is summed directly (no running accumulator), so there is no drift and a sum
            # of non-negative values can never come out negative.
            s = np.full(len(x), np.nan)
            s[w - 1:] = np.lib.stride_tricks.sliding_window_view(x, w).sum(axis=1)
            g[col] = s
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_power() -> pd.DataFrame:
    p = pd.read_csv(U / "weather_soil" / "NASA_POWER_Kerala_District_Daily_2012_2024.csv")
    p["date"] = pd.to_datetime(p["date"])
    assert not p.duplicated(["date", "district"]).any(), "duplicate NASA POWER (date, district)"
    p = p.rename(columns={"GWETROOT": "soil_moisture", "T2M": "temperature_c", "RH2M": "humidity_percent"})
    cols = ["soil_moisture", "temperature_c", "humidity_percent"]
    for c in cols:   # decode the published missing-data sentinel; this is not imputation
        p.loc[p[c] <= POWER_FILL, c] = np.nan
    return p[["date", "district"] + cols]


def build_distance_to_river(ref: pd.DataFrame) -> pd.DataFrame:
    import shapely
    from pyproj import Transformer
    from shapely.geometry import LineString, MultiLineString, Point

    gj = json.loads((U / "hydrology" / "Kerala_HydroRIVERS_v10.geojson").read_text(encoding="utf-8"))
    tr = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    geoms, ids, orders = [], [], []
    for f in gj["features"]:
        g = f["geometry"]
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        lines = []
        for part in parts:
            xs, ys = tr.transform([c[0] for c in part], [c[1] for c in part])
            lines.append(LineString(list(zip(xs, ys))))
        geoms.append(lines[0] if len(lines) == 1 else MultiLineString(lines))
        ids.append(f["properties"]["HYRIV_ID"])
        orders.append(f["properties"]["ORD_STRA"])
    geoms = np.array(geoms, dtype=object)
    rows = []
    for _, r in ref.iterrows():
        px, py = tr.transform(r["longitude"], r["latitude"])
        d = shapely.distance(geoms, Point(px, py))        # exhaustive: every reach, any Strahler order
        i = int(np.argmin(d))
        rows.append(dict(district=r["district"], distance_to_river_km=round(float(d[i]) / 1000.0, 3),
                         nearest_reach_hyriv_id=int(ids[i]), nearest_reach_strahler=int(orders[i])))
    return pd.DataFrame(rows), len(geoms)


def build_static(ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    t = pd.read_csv(U / "terrain" / "Kerala_District_Terrain_Summary.csv")
    t = t.rename(columns={"elevation_m_mean": "elevation_m", "slope_degree_mean": "slope_degree",
                          "aspect_degree_circmean": "aspect_degree", "curvature_mean": "curvature"})
    dd = pd.read_csv(U / "hydrology" / "Kerala_District_Drainage_Density.csv")
    dd = dd.rename(columns={"drainage_density_km_per_km2": "drainage_density"})
    lc = pd.read_csv(U / "terrain" / "Kerala_District_LandCover_Summary.csv")
    lc = lc.rename(columns={"dominant_project_land_cover": "land_cover"})
    li = pd.read_csv(U / "lithology" / "Kerala_District_Lithology_CGWB.csv")
    dist, n_reaches = build_distance_to_river(ref)
    s = ref[["district", "latitude", "longitude"]]
    for part, cols in [(t, ["elevation_m", "slope_degree", "aspect_degree", "curvature"]),
                       (dd, ["drainage_density"]), (lc, ["land_cover"]), (li, ["lithology"])]:
        assert part["district"].is_unique and set(part["district"]) == set(ref["district"]), \
            f"static table does not cover the 14 districts exactly once: {cols}"
        s = s.merge(part[["district"] + cols], on="district", how="left")
    s = s.merge(dist[["district", "distance_to_river_km"]], on="district", how="left")
    return s, dist, n_reaches


def build_river_level(districts: list[str], grid_dates: pd.DatetimeIndex, train_end) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(U / "river_level" / "processed" / "District_Gauge_Assignment.csv")
    st = pd.read_csv(U / "river_level" / "processed" / "CWC_Kerala_RiverLevel_Daily_Station.csv",
                     parse_dates=["date"])
    st["station_key"] = st["station"].str.strip()
    period = (st["date"] >= pd.Timestamp("2012-01-01")) & (st["date"] <= GRID_END)
    rows, gauges = [], []
    for d in districts:
        ar = a[a["district"] == d]
        assert len(ar) == 1, f"gauge assignment missing for {d}"
        ar = ar.iloc[0]
        base = pd.DataFrame({"date": grid_dates, "district": d})
        gauge = str(ar["primary_gauge"]).strip() if pd.notna(ar["primary_gauge"]) else ""
        info = dict(district=d, river_gauge=gauge, river_gauge_river=ar["primary_river"] if gauge else "",
                    river_gauge_status=ar["status"], n_datum_segments_2012_2024=0, datum_break_date="",
                    train_median_m=np.nan, n_train_obs_for_median=0, last_date_used_for_median="")
        if not gauge:
            base["river_stage_published_m"] = np.nan
            base["river_datum_segment"] = np.nan
            base["river_level_m"] = np.nan
            base["river_level_flag"] = "NO_PRIMARY_GAUGE"
        else:
            s = st[(st["station_key"] == gauge) & period]
            assert (s["district"] == d).all(), f"gauge {gauge} is not in district {d}"
            assert not s["date"].duplicated().any(), f"duplicate daily rows for gauge {gauge}"
            seg_first = s.groupby("datum_segment")["date"].min().sort_values()
            info["n_datum_segments_2012_2024"] = int(len(seg_first))
            ref_seg = seg_first.index[0]                      # reference = segment in force first
            break_date = seg_first.iloc[1] if len(seg_first) > 1 else None
            info["datum_break_date"] = "" if break_date is None else str(break_date.date())
            base = base.merge(s[["date", "level_mean_m", "datum_segment"]], on="date", how="left")
            base = base.rename(columns={"level_mean_m": "river_stage_published_m",
                                        "datum_segment": "river_datum_segment"})
            usable = base["river_stage_published_m"].notna()
            if break_date is not None:                        # never mix references: NaN after a break
                usable &= base["date"] < break_date
            train = usable & (base["date"] >= GRID_START) & (base["date"] <= train_end)
            med = float(base.loc[train, "river_stage_published_m"].median()) if train.any() else np.nan
            info.update(train_median_m=med, n_train_obs_for_median=int(train.sum()),
                        last_date_used_for_median=str(base.loc[train, "date"].max().date()) if train.any() else "")
            base["river_level_m"] = np.where(usable, (base["river_stage_published_m"] - med).round(4), np.nan)
            base["river_level_flag"] = np.where(
                usable & np.isfinite(med), "OBSERVED",
                np.where(base["river_stage_published_m"].isna(), "NO_READING_THAT_DAY", "AFTER_DATUM_BREAK"))
        for k in ("river_gauge", "river_gauge_river", "river_gauge_status"):
            base[k] = info[k]
        base["river_gauge_train_median_m"] = info["train_median_m"]
        rows.append(base)
        gauges.append(info)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(gauges)


# ---------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------
def _days(event_days, start, end) -> list[pd.Timestamp]:
    if isinstance(event_days, str) and event_days.strip():
        return [pd.Timestamp(x) for x in event_days.split(";") if x.strip()]
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"))


def build_flood_labels(names: dict) -> tuple[pd.DataFrame, dict]:
    ifi = pd.read_csv(U / "flood_events" / "processed" / "IFI_Kerala_2012_2023_corrected.csv")
    ext = pd.read_csv(U / "flood_events" / "processed" / "IMD_DWE_Kerala_events_not_in_IFI.csv")
    recs, meta = [], dict(ifi_records=len(ifi), ext_records=len(ext), excluded_landslide_only=[],
                          excluded_no_district=[], long_period=[], empty_event_days=[])
    for _, r in ifi.iterrows():
        flags = "" if pd.isna(r["flags"]) else str(r["flags"])
        ds = split_names(r["Districts"], names)
        excl = ("LANDSLIDE_ONLY_CAUSE" if "LANDSLIDE_ONLY_CAUSE" in flags else
                "NO_DISTRICT" if ("NO_DISTRICT" in flags or not ds) else "")
        if excl == "LANDSLIDE_ONLY_CAUSE":
            meta["excluded_landslide_only"].append(r["UEI"])
        elif excl == "NO_DISTRICT":
            meta["excluded_no_district"].append(r["UEI"])
        lp = "LONG_PERIOD_SUMMARY" in flags
        if lp:
            meta["long_period"].append(r["UEI"])
        if not (isinstance(r["event_days"], str) and r["event_days"].strip()):
            meta["empty_event_days"].append(r["UEI"])
        for day in _days(r["event_days"], r["verified_start"], r["verified_end"]):   # verified_* only
            for d in ds:
                recs.append(dict(district=d, date=day, source_id=r["UEI"], source="IFI",
                                 excluded=excl, long_period=lp))
    meta["ext_landslide_not_flood"], meta["omitted_from_ifi"] = [], []
    for _, r in ext.iterrows():
        sid = f"IMD:{Path(r['report']).stem}:p{int(r['pdf_page'])}:{str(r['entry']).replace(' ', '')}"
        if r["category"] == "OMITTED_FROM_IFI":
            meta["omitted_from_ifi"].append(f"{sid} ({r['event_days']}; {r['districts']})")
            if not INCLUDE_OMITTED_FROM_IFI:
                continue
        if bool(r["landslide_mentioned"]) and not bool(r["flood_mentioned"]):
            meta["ext_landslide_not_flood"].append(f"{sid} ({r['event_days']}; {r['districts']})")
        for day in _days(r["event_days"], None, None):
            for d in split_names(r["districts"], names):
                recs.append(dict(district=d, date=day, source_id=sid, source="IMD_EXT", excluded="", long_period=False))
    fr = pd.DataFrame(recs)
    return fr, meta


def strictly_stated_dates(history) -> set[pd.Timestamp]:
    h = "" if pd.isna(history) else str(history)
    out = set()
    for m in TEXT_DATE_STRICT.finditer(h):
        month, year = MON[m.group(2)[:3].lower()], int(m.group(3))
        nums = [int(x) for x in re.findall(r"\d{1,2}", m.group(1))]
        if re.search(r"\d\s*(?:st|nd|rd|th)?\s*(?:-|to)\s*\d", m.group(1)) and len(nums) == 2 and nums[0] <= nums[1]:
            nums = list(range(nums[0], nums[1] + 1))
        for d in nums:
            try:
                out.add(pd.Timestamp(year, month, d))
            except ValueError:
                pass
    for m in NUM_DATE.finditer(h):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            out.add(pd.Timestamp(y + 2000 if y < 100 else y, mo, d))
        except ValueError:
            pass
    return out


def build_landslide_labels(names: dict) -> tuple[pd.DataFrame, dict]:
    g = pd.read_csv(U / "landslide_events" / "processed" / "GSI_Kerala_Landslides_date_precision.csv")
    imd = pd.read_csv(U / "landslide_events" / "processed" / "IMD_DWE_Kerala_landslide_entries.csv")
    recs, rejected = [], []
    ge = g[g["exact_dates"].fillna("").str.strip() != ""]
    for _, r in ge.iterrows():
        stated = strictly_stated_dates(r["History"])
        amb = bool(r["numeric_date_ambiguous_dm"])
        for x in str(r["exact_dates"]).split(";"):
            day = pd.Timestamp(x)
            if GSI_REQUIRE_DATE_STATED and day not in stated:
                rejected.append(dict(sl_no=int(r["Sl_No"]), district=to_chirps(r["district_project"], names),
                                     date=day, history=r["History"]))
                continue
            recs.append(dict(district=to_chirps(r["district_project"], names), date=day,
                             source_id=f"GSI:{int(r['Sl_No'])}", source="GSI",
                             flag="GSI_NUMERIC_DATE_DM_AMBIGUOUS" if amb else ""))
    im = imd[(imd["date_precision"] == "EXACT_DATE") & (imd["landslide_districts"].fillna("").str.strip() != "")]
    for _, r in im.iterrows():
        sid = f"IMD:{Path(r['report']).stem}:p{int(r['page'])}:s{int(r['sno'])}"
        for day in _days(r["event_days"], None, None):
            for d in split_names(r["landslide_districts"], names, seps=";"):
                recs.append(dict(district=d, date=day, source_id=sid, source="IMD", flag=""))

    # Web-verified district-days (2018-2024 verification passes). Same rule as the
    # other two sources: exact stated date + named district, one record per
    # district-day. Each row carries its source URL in the file itself.
    vpath = U / "landslide_events" / "processed" / "VERIFIED_web_landslide_entries.csv"
    ver = pd.read_csv(vpath) if vpath.exists() else pd.DataFrame(
        columns=["district", "date", "date_precision", "source_id"])
    for _, r in ver[ver["date_precision"] == "EXACT_DATE"].iterrows():
        recs.append(dict(district=to_chirps(r["district"], names), date=pd.Timestamp(r["date"]),
                         source_id=str(r["source_id"]), source="VERIFIED", flag=""))

    # Frozen 2012-2017 verification. ADD = verified district-day (one record, like the
    # sources above). REMOVE = rejected district-day: every record for it, from any
    # source, is dropped so the label cannot come back through GSI or IMD.
    upath = U / "landslide_events" / "processed" / "FINAL_2012_2017_landslide_label_updates.csv"
    upd = pd.read_csv(upath) if upath.exists() else pd.DataFrame(
        columns=["date", "district", "action", "source_id"])
    for _, r in upd[upd["action"] == "ADD"].iterrows():
        recs.append(dict(district=to_chirps(r["district"], names), date=pd.Timestamp(r["date"]),
                         source_id=str(r["source_id"]), source="VERIFIED_2012_2017", flag=""))
    lr = pd.DataFrame(recs)
    removed = []
    for _, r in upd[upd["action"] == "REMOVE"].iterrows():
        d, day = to_chirps(r["district"], names), pd.Timestamp(r["date"])
        hit = (lr["district"] == d) & (lr["date"] == day)
        removed.append(dict(district=d, date=day, n_records_dropped=int(hit.sum()),
                            source_ids=";".join(sorted(lr.loc[hit, "source_id"]))))
        lr = lr[~hit]
    meta = dict(gsi_rows=len(g), gsi_exact_rows=len(ge), imd_rows=len(imd), imd_used_rows=len(im),
                verified_rows=len(ver), verified_path=str(vpath.relative_to(U.parent)),
                update_adds=int((upd["action"] == "ADD").sum()),
                update_removes=int((upd["action"] == "REMOVE").sum()),
                update_removed=removed, update_path=str(upath.relative_to(U.parent)),
                imd_excluded_precision=imd[imd["date_precision"] != "EXACT_DATE"]["date_precision"]
                .str.split(" ").str[0].value_counts().to_dict(),
                imd_excluded_unattributed=int(((imd["date_precision"] == "EXACT_DATE") &
                                               (imd["landslide_districts"].fillna("").str.strip() == "")).sum()),
                gsi_excluded_precision=g[g["exact_dates"].fillna("").str.strip() == ""]["date_precision"]
                .value_counts().to_dict(),
                rejected=pd.DataFrame(rejected))
    return lr, meta


def aggregate_labels(grid: pd.DataFrame, recs: pd.DataFrame, hazard: str) -> tuple[pd.Series, pd.DataFrame]:
    """1 on district-days covered by at least one included record, 0 elsewhere; plus key columns."""
    inwin = recs[(recs["date"] >= GRID_START) & (recs["date"] <= GRID_END)]
    key = grid[["date", "district"]].copy()
    if hazard == "flood":
        inc = inwin[inwin["excluded"] == ""]
        agg = inc.groupby(["district", "date"]).agg(
            flood_source_ids=("source_id", lambda s: ";".join(sorted(set(s)))),
            flood_n_sources=("source_id", "nunique"),
            flood_long_period_only=("long_period", "all")).reset_index()
        exc = (inwin[inwin["excluded"] != ""].groupby(["district", "date"])["source_id"]
               .agg(lambda s: ";".join(sorted(set(s)))).rename("flood_excluded_source_ids").reset_index())
        key = key.merge(agg, on=["district", "date"], how="left").merge(exc, on=["district", "date"], how="left")
        y = key["flood_source_ids"].notna().astype(int)
        key["flood_n_sources"] = key["flood_n_sources"].fillna(0).astype(int)
        key["flood_long_period_only"] = key["flood_long_period_only"].eq(True).astype(int)
        cols = ["flood_source_ids", "flood_n_sources", "flood_long_period_only", "flood_excluded_source_ids"]
    else:
        agg = inwin.groupby(["district", "date"]).agg(
            landslide_source=("source", lambda s: "+".join(sorted(set(s)))),
            landslide_gsi_n_records=("source", lambda s: int((s == "GSI").sum())),
            landslide_source_ids=("source_id", lambda s: ";".join(sorted(set(s)))),
            landslide_flags=("flag", lambda s: ";".join(sorted({f for f in s if f})))).reset_index()
        key = key.merge(agg, on=["district", "date"], how="left")
        y = key["landslide_source"].notna().astype(int)
        key["landslide_gsi_n_records"] = key["landslide_gsi_n_records"].fillna(0).astype(int)
        cols = ["landslide_source", "landslide_gsi_n_records", "landslide_source_ids", "landslide_flags"]
    return y, key[cols]


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, section, check, expected, observed, status, detail="", split="all"):
        self.rows.append(dict(check_id=f"V{len(self.rows) + 1:03d}", section=section, check=check, split=split,
                              expected=str(expected), observed=str(observed), status=status, detail=str(detail)))

    def ok(self, section, check, cond, expected, observed, detail="", split="all"):
        self.add(section, check, expected, observed, "PASS" if cond else "FAIL", detail, split)

    def frame(self):
        return pd.DataFrame(self.rows)


def load_ml_module(name: str):
    """Import ml/<name>.py read-only (no bytecode cache is written into ml/)."""
    spec = importlib.util.spec_from_file_location(f"_ml_{name}", ROOT / "scripts" / "models" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = previous
    return mod


def validate(master, keys, rain_all, gauges, train_end, val_end, flood_meta, flood_recs, land_meta, land_recs,
             flood_before) -> Checks:
    V = Checks()
    m = master.copy()
    m["date"] = pd.to_datetime(m["date"])
    sp = split_of(m["date"], train_end, val_end)
    n_dates = m["date"].nunique()

    # ---- structure ---------------------------------------------------------------------------
    S = "structure"
    V.ok(S, "row count", len(m) == EXPECTED["rows"], EXPECTED["rows"], len(m))
    V.ok(S, "column count", m.shape[1] == 23, 23, m.shape[1])
    synth_cols = list(pd.read_csv(ROOT / "synthetic" / "dataset" / "synthetic_master_dataset.csv", nrows=0).columns)
    expected_cols = synth_cols[:3] + ["district"] + synth_cols[3:]
    V.ok(S, "column order = synthetic_master_dataset.csv with `district` inserted after `longitude`",
         list(master.columns) == expected_cols, "|".join(expected_cols), "|".join(master.columns))
    per_date = m.groupby("date").size()
    V.ok(S, "14 locations on every date", (per_date == 14).all(), "14 on all dates",
         f"min {per_date.min()} / max {per_date.max()} over {n_dates} dates")
    V.ok(S, "no duplicate (date, latitude, longitude)", not m.duplicated(["date", "latitude", "longitude"]).any(),
         0, int(m.duplicated(["date", "latitude", "longitude"]).sum()))
    V.ok(S, "no duplicate (date, district)", not m.duplicated(["date", "district"]).any(), 0,
         int(m.duplicated(["date", "district"]).sum()))
    gapfree = all(len(g) == len(pd.date_range(GRID_START, GRID_END)) and
                  (g["date"].sort_values().diff().dropna() == pd.Timedelta(days=1)).all()
                  for _, g in m.groupby("district"))
    V.ok(S, "every district has a gap-free daily series 2012-01-30..2024-12-31", gapfree,
         f"{len(pd.date_range(GRID_START, GRID_END))} consecutive days x 14", "gap-free" if gapfree else "gaps")
    pairs = m.groupby("district")[["latitude", "longitude"]].nunique()
    one_pair = bool((pairs == 1).all().all()) and m[["latitude", "longitude"]].drop_duplicates().shape[0] == 14
    V.ok(S, "each district maps to exactly one (latitude, longitude) and vice versa", one_pair, "14 <-> 14",
         f"{m[['latitude', 'longitude']].drop_duplicates().shape[0]} distinct points")
    V.ok(S, "date range", (m["date"].min() == GRID_START) and (m["date"].max() == GRID_END),
         f"{GRID_START.date()}..{GRID_END.date()}", f"{m['date'].min().date()}..{m['date'].max().date()}")
    V.ok(S, "rows sorted by (date, latitude, longitude)",
         m[["date", "latitude", "longitude"]].equals(
             m.sort_values(["date", "latitude", "longitude"], kind="mergesort")[["date", "latitude", "longitude"]]
             .reset_index(drop=True)), "sorted", "sorted")
    V.ok(S, "keys file has the same row order (date, district)",
         (keys["date"].to_numpy() == master["date"].to_numpy()).all() and
         (keys["district"].to_numpy() == master["district"].to_numpy()).all(), "identical", "identical")
    rs = rain_all[(rain_all["date"] >= WARMUP_START)]
    V.ok(S, "CHIRPS source covers 2012-01-01..2024-12-31 x 14 with no missing day",
         rs["rainfall_1d_mm"].notna().all() and len(rs) == 14 * len(pd.date_range(WARMUP_START, GRID_END)),
         14 * len(pd.date_range(WARMUP_START, GRID_END)), f"{len(rs)} rows, {int(rs['rainfall_1d_mm'].isna().sum())} NaN")

    # ---- split -----------------------------------------------------------------------------------
    S = "split"
    V.ok(S, "train end (replica of chronological_split)", str(train_end.date()) == EXPECTED["train_end"],
         EXPECTED["train_end"], train_end.date())
    V.ok(S, "validation end (replica of chronological_split)", str(val_end.date()) == EXPECTED["val_end"],
         EXPECTED["val_end"], val_end.date())
    mods = {}
    for name in ("flood_xgboost", "landslide_xgboost"):
        try:
            mods[name] = load_ml_module(name)
        except Exception as e:  # noqa: BLE001 - reported, not hidden
            V.add(S, f"import ml/{name}.py", "importable", f"import failed: {e!r}", "FAIL")
    for name, mod in mods.items():
        with contextlib.redirect_stdout(io.StringIO()):
            tr, va, te = mod.chronological_split(m)
        obs = f"train {tr['date'].min().date()}..{tr['date'].max().date()} ({len(tr)}) | " \
              f"val {va['date'].min().date()}..{va['date'].max().date()} ({len(va)}) | " \
              f"test {te['date'].min().date()}..{te['date'].max().date()} ({len(te)})"
        V.ok(S, f"ml/{name}.py chronological_split() on the real file gives the same boundaries",
             tr["date"].max() == train_end and va["date"].max() == val_end and
             tr["date"].min() == GRID_START and te["date"].max() == GRID_END,
             f"train ..{train_end.date()} | val ..{val_end.date()} | test ..{GRID_END.date()}", obs)
    for s_name in ("train", "validation", "test"):
        V.add(S, "rows / dates in split", "", f"{int((sp == s_name).sum())} rows / {m.loc[sp == s_name, 'date'].nunique()} dates",
              "INFO", f"{m.loc[sp == s_name, 'date'].min().date()}..{m.loc[sp == s_name, 'date'].max().date()}", s_name)

    # ---- schema (the training scripts' own validate_columns) --------------------------------------
    S = "schema"
    for name, mod in mods.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.validate_columns(m)
            V.ok(S, f"ml/{name}.py validate_columns() passes on the real file", True, "no exception", "passed")
        except Exception as e:  # noqa: BLE001
            V.ok(S, f"ml/{name}.py validate_columns() passes on the real file", False, "no exception", repr(e))
    if "flood_xgboost" in mods:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mods["flood_xgboost"].report_missing_values(m)
            V.ok(S, "ml/flood_xgboost.py report_missing_values() accepts the file (target has no NaN)", True,
                 "no exception", "passed")
        except Exception as e:  # noqa: BLE001
            V.ok(S, "ml/flood_xgboost.py report_missing_values() accepts the file", False, "no exception", repr(e))
    # replica of both validate_columns (in case the ML modules cannot be imported)
    flood_req = FEATURE_COLUMNS[:9] + ["river_level_m", "distance_to_river_km", "drainage_density", "flood",
                                       "date", "latitude", "longitude"]
    land_req = FEATURE_COLUMNS[:9] + ["slope_degree", "aspect_degree", "curvature", "land_cover", "lithology",
                                      "landslide", "date", "latitude", "longitude"]
    V.ok(S, "replica: all flood-model required columns present", set(flood_req) <= set(m.columns), "all", "all"
         if set(flood_req) <= set(m.columns) else sorted(set(flood_req) - set(m.columns)))
    V.ok(S, "replica: all landslide-model required columns present", set(land_req) <= set(m.columns), "all", "all"
         if set(land_req) <= set(m.columns) else sorted(set(land_req) - set(m.columns)))
    for t in ("flood", "landslide"):
        vals = {int(v) for v in pd.unique(m[t].dropna())}
        V.ok(S, f"target `{t}` is integer 0/1 with no NaN",
             vals <= {0, 1} and m[t].notna().all() and pd.api.types.is_integer_dtype(m[t]),
             "{0,1}, int, 0 NaN", f"{sorted(vals)}, {m[t].dtype}, {int(m[t].isna().sum())} NaN")
    for col, cats in (("land_cover", LAND_COVER_CATEGORIES), ("lithology", LITHOLOGY_CATEGORIES)):
        vals = sorted(m[col].dropna().unique())
        V.ok(S, f"`{col}` values inside the fixed CATEGORY_SCHEMA", set(vals) <= set(cats) and m[col].notna().all(),
             "|".join(cats), f"{'|'.join(vals)} (NaN {int(m[col].isna().sum())})")

    # ---- physical ranges (as validate_synthetic_dataset.py, where applicable) ----------------------
    S = "ranges"
    rcols = [c for c in m if c.startswith("rainfall")]
    V.ok(S, "rainfall >= 0", (m[rcols] >= 0).all().all(), ">= 0", f"min {m[rcols].min().min():.4f}")
    for col, lo, hi in (("soil_moisture", 0, 1), ("temperature_c", -10, 55), ("humidity_percent", 0, 100),
                        ("slope_degree", 0, 90), ("aspect_degree", 0, 360)):
        x = m[col].dropna()
        V.ok(S, f"{col} in [{lo},{hi}]", x.between(lo, hi).all(), f"[{lo},{hi}]", f"[{x.min()}, {x.max()}]")
    for col in ("elevation_m", "distance_to_river_km", "drainage_density"):
        V.ok(S, f"{col} >= 0", (m[col] >= 0).all(), ">= 0", f"min {m[col].min()}")
    V.add(S, "river_level_m >= 0 (synthetic check)", "not applicable", f"[{m['river_level_m'].min()}, "
          f"{m['river_level_m'].max()}]", "INFO", "real river_level_m is stage relative to the gauge's training "
          "median, so negative values (below normal) are expected")
    V.add(S, "curvature has both signs", "", f"{int((m.groupby('district')['curvature'].first() > 0).sum())} "
          f"districts > 0, {int((m.groupby('district')['curvature'].first() < 0).sum())} < 0", "INFO")

    # ---- rolling sums ----------------------------------------------------------------------------
    S = "rolling_sums"
    for w, col in WINDOWS.items():
        err_in = 0.0
        for _, g in m.groupby("district"):
            g = g.sort_values("date")
            e = (g[col] - g["rainfall_1d_mm"].rolling(w).sum()).abs().max()
            err_in = max(err_in, float(np.nan_to_num(e)))
        V.ok(S, f"{col} = sum of the last {w} rainfall_1d_mm within the file (validate_synthetic_dataset.py)",
             err_in < 1e-6, "< 1e-6 mm", f"{err_in:.2e} mm", "first w-1 rows per district use the warm-up and "
             "are checked in the next test")
        ref = []
        src = rain_all.sort_values(["district", "date"])
        for _, g in src.groupby("district"):
            ref.append(pd.DataFrame({"district": g["district"], "date": g["date"],
                                     "ref": g["rainfall_1d_mm"].rolling(w, min_periods=w).sum()}))
        ref = pd.concat(ref)
        j = m[["district", "date", col]].merge(ref, on=["district", "date"], how="left")
        err_all = float((j[col] - j["ref"]).abs().max())
        V.ok(S, f"{col} = trailing {w}-day sum incl. warm-up, all 66,080 rows (independent pandas recompute)",
             err_all < 1e-6 and j[col].notna().all(), "< 1e-6 mm, 0 NaN",
             f"{err_all:.2e} mm, {int(j[col].isna().sum())} NaN")

    # ---- missing values ----------------------------------------------------------------------------
    S = "missing_values"
    no_missing = [c for c in master.columns if m[c].notna().all()]
    V.add(S, "columns with no missing value in any split", "", "|".join(no_missing), "INFO")
    for c in master.columns:
        if c in no_missing:
            continue
        for s_name in ("train", "validation", "test"):
            n_s = int((sp == s_name).sum())
            miss = int(m.loc[sp == s_name, c].isna().sum())
            V.add(S, f"missing {c}", "", f"{miss} of {n_s} ({100 * miss / n_s:.1f} %)", "INFO", "left as NaN", s_name)
        V.add(S, f"missing {c}", "", f"{int(m[c].isna().sum())} of {len(m)} ({100 * m[c].isna().mean():.1f} %)",
              "INFO", "left as NaN", "all")
    exp_cov = dict(zip(("train", "validation", "test"), EXPECTED["river_coverage_pct"]))
    for s_name in ("train", "validation", "test"):
        cov = 100 * m.loc[sp == s_name, "river_level_m"].notna().mean()
        V.ok(S, "river_level_m coverage of district-days", abs(cov - exp_cov[s_name]) < 0.5,
             f"~{exp_cov[s_name]} %", f"{cov:.1f} %", split=s_name)
    no_gauge = sorted(keys.loc[keys["river_gauge"] == "", "district"].unique())
    V.ok(S, "districts without a primary gauge are entirely NaN",
         m.loc[m["district"].isin(no_gauge), "river_level_m"].isna().all(), "Alappuzha|Kottayam|Wayanad",
         "|".join(no_gauge))

    # ---- labels ------------------------------------------------------------------------------------
    S = "labels"
    for t, key_col in (("flood", "flood_source_ids"), ("landslide", "landslide_source")):
        for s_name in ("train", "validation", "test"):
            V.add(S, f"{t} positives", "", int(m.loc[sp == s_name, t].sum()), "INFO",
                  f"{100 * m.loc[sp == s_name, t].mean():.3f} % of {int((sp == s_name).sum())} district-days", s_name)
        V.add(S, f"{t} positives", "", int(m[t].sum()), "INFO", f"{100 * m[t].mean():.3f} %", "all")
        V.ok(S, f"{t} = 1 exactly where the keys file lists an included source",
             ((m[t] == 1) == keys[key_col].notna()).all(), "identical", "identical")
    fb = flood_before
    V.ok(S, "flood positives BEFORE exclusions reproduce the documented total/train/val/test "
         "(Flood_label_coverage_summary.csv, row matching INCLUDE_OMITTED_FROM_IFI)",
         tuple(fb) == EXPECTED["flood_before_exclusions"], "/".join(map(str, EXPECTED["flood_before_exclusions"])),
         "/".join(map(str, fb)), f"INCLUDE_OMITTED_FROM_IFI = {INCLUDE_OMITTED_FROM_IFI}")
    V.add(S, "difference from the build prompt's 3,132 (extension events only)",
          "/".join(map(str, EXPECTED["flood_before_prompt"])), "/".join(map(str, fb)), "INFO",
          "the prompt's figure omits the OMITTED_FROM_IFI event of the same file: "
          + "; ".join(flood_meta["omitted_from_ifi"]) + (" (included)" if INCLUDE_OMITTED_FROM_IFI else " (excluded)"))
    n_ex_only = int(((m["flood"] == 0) & keys["flood_excluded_source_ids"].notna()).sum())
    V.add(S, "flood district-days removed by the exclusions (covered only by excluded records)", "",
          n_ex_only, "INFO", f"LANDSLIDE_ONLY_CAUSE {len(flood_meta['excluded_landslide_only'])} records, "
          f"NO_DISTRICT {len(flood_meta['excluded_no_district'])} records (no district -> 0 district-days)")
    lp = keys["flood_long_period_only"] == 1
    for s_name in ("train", "validation", "test"):
        V.add(S, "flood positives resting only on LONG_PERIOD_SUMMARY records", "", int((lp & (sp == s_name)).sum()),
              "INFO", "kept per decision 2; use flood_long_period_only in the keys file for sensitivity runs", s_name)
    src = keys.loc[m["flood"] == 1, "flood_source_ids"]
    V.add(S, "flood positives by source", "", f"IFI only {int((~src.str.contains('IMD:')).sum())} | "
          f"IMD extension only {int((~src.str.contains('UEI-')).sum())} | both "
          f"{int((src.str.contains('IMD:') & src.str.contains('UEI-')).sum())}", "INFO")
    exp_l = EXPECTED["landslide"]
    obs_l = (int(m["landslide"].sum()),) + tuple(int(m.loc[sp == s, "landslide"].sum())
                                               for s in ("train", "validation", "test"))
    V.add(S, "landslide positives vs documented ~120 (101/11/8)", "/".join(map(str, exp_l)),
          "/".join(map(str, obs_l)), "PASS" if obs_l == exp_l else "INFO",
          "difference explained by the GSI stated-date guard" if obs_l != exp_l else "")
    ls = keys.loc[m["landslide"] == 1, "landslide_source"]
    V.add(S, "landslide positives by source", "", " | ".join(f"{k} {v}" for k, v in ls.value_counts().items()), "INFO")

    # ---- web-verified labels (2018-2024 verification passes) ---------------------------------
    vsrc = keys["landslide_source"].fillna("")
    # landslide_source joins source names with "+"; match whole tokens, not substrings
    vtok = vsrc.str.split("+")
    n_ver_rows = int(vtok.apply(lambda t: "VERIFIED" in t).sum())
    exp_add = EXPECTED["landslide_verified_added"]
    V.ok(S, "verified label file rows loaded", land_meta["verified_rows"] == exp_add,
         exp_add, land_meta["verified_rows"], land_meta["verified_path"])
    V.ok(S, "verified district-days present in the master", n_ver_rows == exp_add,
         exp_add, n_ver_rows, "district-days whose label comes from the verification passes")
    add12, rem12 = EXPECTED["landslide_2012_2017_added"], EXPECTED["landslide_2012_2017_removed"]
    exp_total = EXPECTED["landslide_before_verification"] + exp_add + add12 - rem12
    V.ok(S, "landslide positives = documented baseline + verified labels",
         int(m["landslide"].sum()) == exp_total, exp_total, int(m["landslide"].sum()),
         f"{EXPECTED['landslide_before_verification']} before the verification passes + {exp_add} "
         f"verified 2018-2024 + {add12} verified 2012-2017 - {rem12} rejected 2012-2017")
    V.ok(S, "2012-2017 update file: ADD / REMOVE rows loaded",
         (land_meta["update_adds"], land_meta["update_removes"]) == (add12, rem12),
         f"{add12} / {rem12}", f"{land_meta['update_adds']} / {land_meta['update_removes']}",
         land_meta["update_path"])
    n_v12 = int(vtok.apply(lambda t: "VERIFIED_2012_2017" in t).sum())
    V.ok(S, "2012-2017 verified district-days present in the master", n_v12 == add12, add12, n_v12)
    rem_ok = all(int(m.loc[(m["district"] == r["district"]) & (m["date"] == r["date"]),
                          "landslide"].iloc[0]) == 0 and r["n_records_dropped"] > 0
                 for r in land_meta["update_removed"])
    V.ok(S, "2012-2017 rejected district-days are 0 in the master (source records dropped)",
         rem_ok and len(land_meta["update_removed"]) == rem12, "all 0", "all 0" if rem_ok else "NOT 0",
         "; ".join(f"{r['date'].date()} {r['district']} <- dropped {r['source_ids']}"
                   for r in land_meta["update_removed"]))
    vy = m.loc[(m["landslide"] == 1) & vtok.apply(lambda t: "VERIFIED" in t), "date"].astype(str).str[:4]
    V.add(S, "verified positives by year", "", vy.value_counts().sort_index().to_dict(), "INFO")
    V.ok(S, "verified 2023 positives present (first day-level 2023 labels)",
         int((vy == "2023").sum()) == 2, 2, int((vy == "2023").sum()),
         "2023-10-24 Idukki; 2023-11-05 Idukki")
    rej = land_meta["rejected"]
    if len(rej):
        rw = rej[(rej["date"] >= GRID_START) & (rej["date"] <= GRID_END)]
        V.add(S, "GSI exact dates rejected: day not stated in the record's History text", "0 used",
              f"{len(rej)} record-dates ({len(rw)} in grid window)", "INFO",
              "; ".join(f"GSI:{r.sl_no} {r.district} {r.date.date()} <- '{r.history}'" for r in rw.itertuples()))
        lw_all = pd.concat([land_recs[["district", "date"]], rw[["district", "date"]]])
        lw_all = lw_all[(lw_all["date"] >= GRID_START) & (lw_all["date"] <= GRID_END)].drop_duplicates()
        gsp = split_of(lw_all["date"], train_end, val_end)
        new_days = len(lw_all) - int(m["landslide"].sum())
        V.add(S, "landslide positives WITHOUT the stated-date guard (raw exact_dates column)", "not used",
              f"{len(lw_all)}/{int((gsp == 'train').sum())}/{int((gsp == 'validation').sum())}/"
              f"{int((gsp == 'test').sum())}", "INFO",
              f"the guard removes {new_days} district-days that only the parser artifacts would have created")
    lw = land_recs[(land_recs["date"] >= GRID_START) & (land_recs["date"] <= GRID_END)]
    V.add(S, "landslide record-dates outside the grid window (not used)", "", int(len(land_recs) - len(lw)), "INFO",
          "GSI dates before 2012-01-30 or after 2024-12-31")
    V.ok(S, "landslide target has no NaN", m["landslide"].notna().all(), 0, int(m["landslide"].isna().sum()))

    # ---- leakage -----------------------------------------------------------------------------------
    S = "leakage"
    k = keys.copy()
    k["date"] = pd.to_datetime(k["date"])
    worst, last_used, ok_last = 0.0, [], True
    for _, gi in gauges[gauges["river_gauge"] != ""].iterrows():
        kk = k[(k["district"] == gi["district"])]
        tr_mask = (kk["date"] <= train_end) & (kk["river_level_flag"] == "OBSERVED")
        med = kk.loc[tr_mask, "river_stage_published_m"].median()
        worst = max(worst, abs(med - gi["train_median_m"]))
        ok_last &= pd.Timestamp(gi["last_date_used_for_median"]) <= train_end
        last_used.append(f"{gi['river_gauge']} {gi['last_date_used_for_median']}")
    V.ok(S, "gauge medians recomputed from TRAINING rows only equal the medians used", worst < 1e-9,
         "difference 0", f"max |diff| {worst:.2e} m")
    V.ok(S, "last date entering any gauge median <= train end", ok_last, f"<= {train_end.date()}",
         max(pd.Timestamp(x.split(' ')[-1]) for x in last_used).date(), "; ".join(last_used))
    j = k[k["river_level_flag"] == "OBSERVED"]
    rec = (j["river_stage_published_m"] - j["river_gauge_train_median_m"]).round(4)
    V.ok(S, "river_level_m = published stage - training median (observed days)",
         np.allclose(rec.to_numpy(), m.loc[j.index, "river_level_m"].to_numpy(), atol=1e-9), "identical", "identical")
    V.ok(S, "rolling windows are trailing (end on day t), not centred or forward",
         all(r["status"] == "PASS" for r in V.rows if r["section"] == "rolling_sums"), "trailing",
         "matches trailing sums on all rows")
    same = [f"{f}~{t}" for f in FEATURE_COLUMNS for t in ("flood", "landslide")
            if pd.api.types.is_numeric_dtype(m[f]) and m[f].fillna(-1e9).equals(m[t].astype(float))]
    V.ok(S, "no feature column equals a target column", not same, "none", "none" if not same else same)
    V.add(S, "features are built before, and independently of, the label files", "by construction",
          "rainfall/POWER/terrain/hydrology/lithology/CWC inputs contain no hazard label", "PASS")
    # district must not be a model feature anywhere
    hits = []
    for name, mod in mods.items():
        for lst in ("FLOOD_FEATURES", "LANDSLIDE_FEATURES", "COMMON_FEATURES"):
            if "district" in getattr(mod, lst, []):
                hits.append(f"ml/{name}.py:{lst}")
    for jf in ("flood_features.json", "landslide_features.json"):
        p = ROOT / "synthetic" / "models" / jf
        if p.exists() and "district" in json.loads(p.read_text(encoding="utf-8")).get("feature_order", []):
            hits.append(f"ml/{jf}:feature_order")
    code_hits = []
    for p in list((ROOT / "scripts" / "models").glob("*.py")) + list((ROOT / "scripts" / "analysis").glob("*.py")) + list((ROOT / "backend").rglob("*.py")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"""["']district["']""", txt):
            code_hits.append(str(p.relative_to(ROOT)))
    V.ok(S, "`district` is in no feature list (FLOOD_FEATURES, LANDSLIDE_FEATURES, feature_order JSON)",
         not hits, "none", "none" if not hits else hits)
    V.add(S, "string 'district' referenced in ml/ or backend/ Python code", "0 files", f"{len(code_hits)} files",
          "PASS" if not code_hits else "INFO", ";".join(code_hits))
    return V


# ---------------------------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------------------------
def fmt_int(x) -> str:
    return f"{int(x):,}"


def column_stats(m: pd.DataFrame, c: str) -> str:
    x = m[c]
    if pd.api.types.is_numeric_dtype(x) and c not in ("flood", "landslide"):
        return f"{x.min():.4g} – {x.max():.4g}"
    return "|".join(f"{k} ({v:,})" for k, v in x.value_counts().items())


def write_dictionary(path, m, keys, static, dist, gauges, n_reaches, train_end, val_end, V, flood_meta, land_meta,
                     flood_before):
    sp = split_of(m["date"], train_end, val_end)
    counts = {s: int((sp == s).sum()) for s in ("train", "validation", "test")}
    miss = {c: {s: int(m.loc[sp == s, c].isna().sum()) for s in counts} for c in m.columns}

    def mcell(c):
        t = sum(miss[c].values())
        return "0" if t == 0 else f"{t:,} ({100 * t / len(m):.1f} %)"

    def pos(t):
        return {s: int(m.loc[sp == s, t].sum()) for s in counts}
    fp, lp = pos("flood"), pos("landslide")
    vf = V.frame()
    n_fail = int((vf["status"] == "FAIL").sum())
    rej = land_meta["rejected"]
    rej_w = rej[(rej["date"] >= GRID_START) & (rej["date"] <= GRID_END)] if len(rej) else rej
    st = static.merge(dist[["district", "nearest_reach_hyriv_id", "nearest_reach_strahler"]], on="district").merge(
        gauges[["district", "river_gauge", "river_gauge_river", "train_median_m", "n_train_obs_for_median"]],
        on="district")
    st = st.sort_values("latitude", ascending=False)
    lpo = keys["flood_long_period_only"] == 1

    L = []
    a = L.append
    a("# REAL DATA — Kerala district-day Master Dataset")
    a("")
    a("**`updated_data/master_dataset/real_master_dataset.csv` holds observed / published data only.** "
      "Every value comes from a named public source in `updated_data/`; nothing is synthetic, estimated, "
      "interpolated, forward-filled or imputed. Missing values are left empty (NaN) and are passed to XGBoost's "
      "native missing-value handling by the training scripts.")
    a("")
    a(f"Built by `updated_data/scripts/13_build_master_dataset.py` on {_date.today().isoformat()} "
      "(re-run it to rebuild everything, e.g. after adding a GSI Bhukosh landslide inventory). Specification: "
      "`updated_data/documentation/MASTER_DATASET_BUILD_PROMPT.md`. Validation: "
      f"`VALIDATION_RESULTS.csv` ({len(vf)} checks, {n_fail} FAIL).")
    a("")
    a("> Unlike `synthetic_master_dataset.csv`, this file describes the real world — but one row is a **whole "
      "district on one day**, with static features taken at district level (14 values each). Read §10 before "
      "interpreting any model result.")
    a("")
    a("---")
    a("")
    a("## 1. Size")
    a("")
    a("| Property | Value |")
    a("|---|---|")
    a(f"| Rows | **{len(m):,}** |")
    a(f"| Columns | **{m.shape[1]}** (the 22 synthetic columns + `district` after `longitude`) |")
    a("| Locations | 14 Kerala districts (one representative point each) |")
    a(f"| Time steps | {m['date'].nunique():,} consecutive days per district |")
    a(f"| Date range | {GRID_START.date()} → {GRID_END.date()} (2012-01-01..01-29 used only as the 29-day "
      "warm-up of `rainfall_30d_mm`, then dropped) |")
    a("| Row ordering | `date`, then `latitude`, `longitude` ascending (14 rows per date) — the order "
      "`chronological_split()` sorts to |")
    a(f"| Missing values | {', '.join(f'`{c}` {mcell(c)}' for c in m.columns if sum(miss[c].values()))}; "
      "all other columns complete |")
    a("")
    a("Row identity is `(date, latitude, longitude)` — equivalently `(date, district)`. "
      "Companion file `real_master_dataset_keys.csv` has the same row order (§8).")
    a("")
    a("---")
    a("")
    a("## 2. Columns")
    a("")
    a("Observed range and missing count are for this build (all 66,080 rows).")
    a("")
    a("### Time and location")
    a("| Column | Type | Source | Rule / caveat |")
    a("|---|---|---|---|")
    a("| `date` | string `YYYY-MM-DD` | CHIRPS daily series | Daily and gap-free for every district. Split key only. |")
    a("| `latitude` | float | `reference/Kerala_district_reference.csv` → `latitude` | District representative point "
      "(geoBoundaries ADM2 polygon centroid, inside the polygon); static per district. Metadata only. |")
    a("| `longitude` | float | same → `longitude` | as above. |")
    a("| `district` | string | same → `district_chirps` | CHIRPS vocabulary (`Pattanamtitta`, not `Pathanamthitta`). "
      "**Identifier only — never a model feature**: every ML script, TCDL, SHAP and the backend select features by "
      "explicit lists, and `district` is in none of them (checked). |")
    a("")
    a("### Common environmental features (inputs to **both** models)")
    a("| Column | Unit | Source (under `updated_data/`) | Rule / caveat | Observed range | Missing |")
    a("|---|---|---|---|---|---|")
    a(f"| `rainfall_1d_mm` | mm/day | `existing_collected_data/Kerala_CHIRPS_District_Rainfall_2012.csv` + "
      f"`_2013_2024.csv` → `rainfall` | CHIRPS v2.0 district-mean daily rainfall; the two files concatenated. | "
      f"{column_stats(m, 'rainfall_1d_mm')} | {mcell('rainfall_1d_mm')} |")
    for w, c in WINDOWS.items():
        a(f"| `{c}` | mm | same | Trailing {w}-day sum per district, **inclusive of day t**, `min_periods = {w}`, "
          f"sorted by date within district; warm-up from 2012-01-01. Reconstructs exactly from `rainfall_1d_mm`. | "
          f"{column_stats(m, c)} | {mcell(c)} |")
    a(f"| `soil_moisture` | fraction 0–1 | `weather_soil/NASA_POWER_Kerala_District_Daily_2012_2024.csv` → `GWETROOT` | "
      f"NASA POWER (MERRA-2) **root-zone soil wetness = saturation fraction** (0 = dry, 1 = saturated). "
      f"**Not** volumetric water content as in the synthetic data — same column name, different quantity. POWER "
      f"grid cell (~0.5° × 0.625°) at the district point. | {column_stats(m, 'soil_moisture')} | {mcell('soil_moisture')} |")
    a(f"| `temperature_c` | °C | same → `T2M` | Daily mean 2-m air temperature (POWER cell). | "
      f"{column_stats(m, 'temperature_c')} | {mcell('temperature_c')} |")
    a(f"| `humidity_percent` | % | same → `RH2M` | Daily mean 2-m relative humidity (POWER cell). | "
      f"{column_stats(m, 'humidity_percent')} | {mcell('humidity_percent')} |")
    a(f"| `elevation_m` | m a.s.l. | `terrain/Kerala_District_Terrain_Summary.csv` → `elevation_m_mean` | District "
      f"mean of the DEM (AWS Terrarium z11, ~74 m). **Static per district** (14 values). | "
      f"{column_stats(m, 'elevation_m')} | {mcell('elevation_m')} |")
    a("")
    a("### Flood-specific features (Model 1 only)")
    a("| Column | Unit | Source | Rule / caveat | Observed range | Missing |")
    a("|---|---|---|---|---|---|")
    a(f"| `river_level_m` | m, relative | `river_level/processed/District_Gauge_Assignment.csv` (district → "
      f"`primary_gauge`) + `river_level/processed/CWC_Kerala_RiverLevel_Daily_Station.csv` (`level_mean_m`) | "
      f"**Stage of the district's primary CWC gauge relative to that gauge's own median over the TRAINING period "
      f"only** ({GRID_START.date()}..{train_end.date()}): metres above (+) / below (−) the gauge's normal level. "
      f"Needed because gauges sit on different datums (e.g. Vandiperiyar ~791 m MSL vs Ayilam ~1.7 m). Every primary "
      f"gauge has a single `datum_segment` over 2012–2024 (a later segment would be set to NaN, never converted). "
      f"Alappuzha, Wayanad, Kottayam have no primary gauge → NaN. 7 gauges start 2015-06-01 → NaN before. Days "
      f"without a reading → NaN. Rounded to 0.1 mm. A point gauge measures one river reach, not the whole district. | "
      f"{column_stats(m, 'river_level_m')} | {mcell('river_level_m')} |")
    a(f"| `distance_to_river_km` | km | `hydrology/Kerala_HydroRIVERS_v10.geojson` + district point | **Computed "
      f"here**: shortest distance from the district representative point to the nearest HydroRIVERS v1.0 reach "
      f"(any Strahler order; all {n_reaches:,} reaches of the clipped file searched exhaustively), measured in "
      f"{METRIC_CRS} (UTM 43N). HydroRIVERS contains rivers with ≥ 10 km² catchment or ≥ 0.1 m³/s mean flow, so "
      f"smaller streams are not counted. Static per district. | {column_stats(m, 'distance_to_river_km')} | "
      f"{mcell('distance_to_river_km')} |")
    a(f"| `drainage_density` | km/km² | `hydrology/Kerala_District_Drainage_Density.csv` → "
      f"`drainage_density_km_per_km2` | HydroRIVERS reach length inside the district / district area. Static. | "
      f"{column_stats(m, 'drainage_density')} | {mcell('drainage_density')} |")
    a("")
    a("### Landslide-specific features (Model 2 only)")
    a("| Column | Unit | Source | Rule / caveat | Observed range | Missing |")
    a("|---|---|---|---|---|---|")
    a(f"| `slope_degree` | ° | `terrain/Kerala_District_Terrain_Summary.csv` → `slope_degree_mean` | District mean "
      f"slope from the DEM. Static. | {column_stats(m, 'slope_degree')} | {mcell('slope_degree')} |")
    a(f"| `aspect_degree` | °, 0 = N, clockwise | same → `aspect_degree_circmean` | Circular mean aspect. Static. | "
      f"{column_stats(m, 'aspect_degree')} | {mcell('aspect_degree')} |")
    a(f"| `curvature` | dimensionless | same → `curvature_mean` | District mean curvature (negative = concave). "
      f"District means are close to 0. Static. | {column_stats(m, 'curvature')} | {mcell('curvature')} |")
    a(f"| `land_cover` | category | `terrain/Kerala_District_LandCover_Summary.csv` → `dominant_project_land_cover` | "
      f"ESA WorldCover 2021 classes mapped to the 5 project classes; dominant class per district. **`forest` in all "
      f"14 districts — constant, carries no information.** | {column_stats(m, 'land_cover')} | {mcell('land_cover')} |")
    a(f"| `lithology` | category | `lithology/Kerala_District_Lithology_CGWB.csv` → `lithology` | CGWB Major Principal "
      f"Aquifer map, dominant class per district (independently reproduced 14/14). **Only 2 values**; an aquifer "
      f"(hydrogeological) map used as a lithology source. | {column_stats(m, 'lithology')} | {mcell('lithology')} |")
    a("")
    a("### Targets")
    a("| Column | Values | Source | Definition |")
    a("|---|---|---|---|")
    a("| `flood` | 0 / 1 (int, no NaN) | `flood_events/processed/IFI_Kerala_2012_2023_corrected.csv` + "
      "`flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv` | **1** = the district-day lies in a damaging "
      "flood / heavy-rain event reported by IMD (IFI-Impacts v3.0 records with IMD-verified dates, extended to "
      "2024 from IMD DWE). **0 = no damaging flood/heavy-rain event reported by IMD for that district-day** — not a "
      "proof that no flooding occurred. See §5. |")
    a("| `landslide` | 0 / 1 (int, no NaN) | `landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv` + "
      "`landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv` | **1** = at least one landslide with an "
      "exact, stated date is recorded in that district on that day (GSI inventory or IMD DWE). **0 = no exact-dated "
      "landslide record** (presence-only data). See §5. |")
    a("")
    a("---")
    a("")
    a("## 3. Feature groups for the two models")
    a("")
    a("Unchanged from `DATA_DICTIONARY.md` and the code — the real file plugs into the same lists:")
    a("")
    a("```")
    a('COMMON = ["rainfall_1d_mm","rainfall_3d_mm","rainfall_7d_mm",')
    a('          "rainfall_14d_mm","rainfall_30d_mm","soil_moisture",')
    a('          "temperature_c","humidity_percent","elevation_m"]')
    a("")
    a('FLOOD_SPECIFIC     = ["river_level_m","distance_to_river_km","drainage_density"]')
    a('LANDSLIDE_SPECIFIC = ["slope_degree","aspect_degree","curvature",')
    a('                      "land_cover","lithology"]')
    a("")
    a("Model 1 (flood):     X = COMMON + FLOOD_SPECIFIC          y = flood       (12 features)")
    a("Model 2 (landslide): X = COMMON + LANDSLIDE_SPECIFIC      y = landslide   (14 features)")
    a("```")
    a("")
    a("`date`, `latitude`, `longitude` and `district` are indexing/splitting columns, not model inputs.")
    a("")
    a("---")
    a("")
    a("## 4. Categorical encoding")
    a("")
    a("Stored as strings, inside the fixed `CATEGORY_SCHEMA` of `ml/landslide_xgboost.py` (codes as in "
      "`DATA_DICTIONARY.md` §4).")
    a("")
    a("**`land_cover`**")
    a("| Label | Code | Rows |")
    a("|---|---|---|")
    for i, c in enumerate(LAND_COVER_CATEGORIES):
        a(f"| `{c}` | {i} | {int((m['land_cover'] == c).sum()):,} |")
    a("")
    a("**`lithology`**")
    a("| Label | Code | Rows |")
    a("|---|---|---|")
    for i, c in enumerate(LITHOLOGY_CATEGORIES):
        a(f"| `{c}` | {i} | {int((m['lithology'] == c).sum()):,} |")
    a("")
    a("---")
    a("")
    a("## 5. Target behaviour and label assumptions")
    a("")
    a("| Split | Dates | District-days | `flood` = 1 | `landslide` = 1 |")
    a("|---|---|---|---|---|")
    for s in counts:
        d = m.loc[sp == s, "date"]
        a(f"| {s} | {d.min().date()} → {d.max().date()} | {counts[s]:,} | {fp[s]:,} ({100 * fp[s] / counts[s]:.2f} %) | "
          f"{lp[s]:,} ({100 * lp[s] / counts[s]:.3f} %) |")
    a(f"| **all** | | **{len(m):,}** | **{int(m['flood'].sum()):,}** ({100 * m['flood'].mean():.2f} %) | "
      f"**{int(m['landslide'].sum()):,}** ({100 * m['landslide'].mean():.3f} %) |")
    a("")
    both = int(((m["flood"] == 1) & (m["landslide"] == 1)).sum())
    a(f"`flood=1 & landslide=1`: {both:,} district-days. φ (Pearson) = "
      f"{np.corrcoef(m['flood'], m['landslide'])[0, 1]:.3f}.")
    a("")
    a("**Flood label (decision 2).**")
    a("")
    a("* 1 on every (district, day) listed in an event's `event_days`; where `event_days` is empty "
      f"({', '.join(flood_meta['empty_event_days']) or 'none'}), every day `verified_start`..`verified_end`. "
      "The `verified_*` dates are used — never IFI's original Start/End Date columns.")
    a(f"* IMD events from `IMD_DWE_Kerala_events_not_in_IFI.csv` (`event_days` × `districts`): the 24 events after "
      f"IFI's last Kerala record and the OMITTED_FROM_IFI event "
      f"({'; '.join(flood_meta['omitted_from_ifi'])}) — "
      f"{'included' if INCLUDE_OMITTED_FROM_IFI else 'excluded'} (`INCLUDE_OMITTED_FROM_IFI` in the build script).")
    a(f"* **Kept:** LONG_PERIOD_SUMMARY records ({', '.join(flood_meta['long_period'])}) — the only coverage of the 2018 "
      f"flood. {int(lpo.sum()):,} positive district-days rest only on such multi-week summary periods "
      f"(train {int((lpo & (sp == 'train')).sum()):,} / val {int((lpo & (sp == 'validation')).sum()):,} / "
      f"test {int((lpo & (sp == 'test')).sum()):,}); they are flagged `flood_long_period_only = 1` in the keys file "
      "for sensitivity runs.")
    a(f"* **Excluded:** {len(flood_meta['excluded_landslide_only'])} LANDSLIDE_ONLY_CAUSE records "
      f"({', '.join(flood_meta['excluded_landslide_only'])}) and {len(flood_meta['excluded_no_district'])} NO_DISTRICT "
      f"records ({', '.join(flood_meta['excluded_no_district'])}).")
    a(f"* Before exclusions: {flood_before[0]:,} positive district-days (train {flood_before[1]:,} / val "
      f"{flood_before[2]:,} / test {flood_before[3]:,}) — reproduces the documented row of "
      f"`Flood_label_coverage_summary.csv` "
      f"({'“IFI corrected + all IMD DWE Kerala entries missing from IFI”' if INCLUDE_OMITTED_FROM_IFI else '“IFI corrected + IMD DWE events after 2023-07-23”'}). "
      f"The build prompt's 3,132 (2,764 / 249 / 119) is the extension-only row; the 4 test district-days in between "
      f"are the omitted 2023-06-28 event. After exclusions: **{int(m['flood'].sum()):,}**.")
    if flood_meta["ext_landslide_not_flood"]:
        a(f"* Caveat: the extension file has no cause flag. {len(flood_meta['ext_landslide_not_flood'])} extension "
          f"entries mention a landslide but not a flood and are labelled flood = 1 as decided: "
          f"{'; '.join(flood_meta['ext_landslide_not_flood'])}.")
    a("")
    a("**Landslide label (decision 3).** Exact-date records only; presence-only.")
    a("")
    a("* GSI: rows of the all-records date-precision file with non-empty `exact_dates`, district from "
      "`district_project`, filtered to the grid window. Excluded precision classes: "
      f"{', '.join(f'{k} {v:,}' for k, v in land_meta['gsi_excluded_precision'].items())} records.")
    a(f"* IMD: rows with `date_precision == EXACT_DATE` and non-empty `landslide_districts` "
      f"({land_meta['imd_used_rows']} of {land_meta['imd_rows']}). Excluded: "
      f"{', '.join(f'{k} {v}' for k, v in land_meta['imd_excluded_precision'].items())} entries and "
      f"{land_meta['imd_excluded_unattributed']} exact-day multi-district entries that do not say where the landslide was.")
    a("* Union, de-duplicated at district-day level.")
    if len(rej):
        a(f"* **Stated-date guard:** {len(rej)} GSI record-dates in `exact_dates` are not stated in the record's own "
          f"`History` text (the upstream parser split a 4-digit year into two day numbers). They are not used — a "
          f"date may never be assigned to a record that does not state one. In the grid window: "
          + "; ".join(f"GSI Sl_No {r.sl_no}, {r.district}, {r.date.date()} (History: “{r.history}”)"
                      for r in rej_w.itertuples()) + ".")
    a(f"* Result: **{int(m['landslide'].sum())}** district-days (train {lp['train']} / val {lp['validation']} / "
      f"test {lp['test']}); documented expectation ≈ 120 (101 / 11 / 8). Without the stated-date guard the build "
      f"gives exactly those 120, so the guard accounts for the whole difference (see `VALIDATION_RESULTS.csv`).")
    a("* Presence-only assumption: a 0 means no exact-dated landslide is on record for that district-day. GSI's "
      "inventory is dominated by the 2018–2019 monsoons, and many real landslides have only a month or year, so "
      "true positives are certainly missed. No random negative sampling is done.")
    a("")
    a("---")
    a("")
    a("## 6. Temporal structure and split")
    a("")
    a(f"`chronological_split()` (70/15/15 on unique dates, from `ml/flood_xgboost.py`) on {m['date'].nunique():,} "
      f"dates gives train {GRID_START.date()} → {train_end.date()}, validation → {val_end.date()}, test → "
      f"{GRID_END.date()} — recomputed here and confirmed by running the scripts' own function on this file.")
    a("")
    a("Rows are not i.i.d.: consecutive days of a district share rainfall windows and a slowly varying soil-moisture "
      "and river state, and neighbouring districts share storms. Always split chronologically.")
    a("")
    a("---")
    a("")
    a("## 7. Data leakage")
    a("")
    a("* No column encodes a label; no feature file contains hazard information. No feature equals a target.")
    a("* Rolling sums are trailing windows ending on day t (verified on every row); nothing is centred or forward.")
    a(f"* The only statistic estimated from the data — each gauge's median — uses TRAINING dates only (last date "
      f"used: {max(gauges['last_date_used_for_median'].replace('', np.nan).dropna())}; train ends {train_end.date()}); "
      "recomputed and matched in validation.")
    a("* `district`, `latitude`, `longitude` are identifiers, not features.")
    a("* Same-day features: rainfall, T2M, RH2M and soil wetness of day t are daily aggregates of day t, as in the "
      "synthetic design; a strict nowcast/lead-time experiment must lag them.")
    a("")
    a("---")
    a("")
    a("## 8. Companion keys file (`real_master_dataset_keys.csv`)")
    a("")
    a("Same row order as the master file. Not a model input.")
    a("")
    a("| Column | Meaning |")
    a("|---|---|")
    for c, d in KEY_COLUMN_DOC.items():
        a(f"| `{c}` | {d} |")
    a("")
    a("---")
    a("")
    a("## 9. Static values per district")
    a("")
    a("| District | Lat | Lon | Elev. m | Slope ° | Aspect ° | Curv. | Drain. dens. | Dist. river km (HYRIV_ID, Strahler) | "
      "Land cover | Lithology | Primary gauge (river) | Gauge train median m (n obs) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in st.itertuples():
        g = f"{r.river_gauge} ({r.river_gauge_river})" if r.river_gauge else "— (none)"
        med = f"{r.train_median_m:.3f} ({r.n_train_obs_for_median:,})" if r.river_gauge else "—"
        a(f"| {r.district} | {r.latitude} | {r.longitude} | {r.elevation_m} | {r.slope_degree} | {r.aspect_degree} | "
          f"{r.curvature} | {r.drainage_density} | {r.distance_to_river_km} ({r.nearest_reach_hyriv_id}, "
          f"{r.nearest_reach_strahler}) | {r.land_cover} | {r.lithology} | {g} | {med} |")
    a("")
    a("---")
    a("")
    a("## 10. Known limitations")
    a("")
    for x in LIMITATIONS:
        a(f"1. {x}")
    a("")
    a("---")
    a("")
    a("## 11. Regenerating")
    a("")
    a("```bash")
    a("python updated_data/scripts/13_build_master_dataset.py")
    a("```")
    a("")
    a("Deterministic (no randomness). Adding new landslide records: process them into "
      "`landslide_events/processed/` with the same columns (`district_project`, `exact_dates`, `History`), re-run, "
      "and compare `VALIDATION_RESULTS.csv`.")
    a("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


KEY_COLUMN_DOC = {
    "date, district": "row identity, identical order to the master file",
    "split": "train / validation / test from `chronological_split()`",
    "river_gauge": "primary CWC gauge of the district (empty = none)",
    "river_gauge_river": "river of that gauge",
    "river_gauge_status": "assignment status from `District_Gauge_Assignment.csv`",
    "river_stage_published_m": "raw published daily mean stage of the gauge (`level_mean_m`, gauge datum), for transparency",
    "river_datum_segment": "datum segment of that reading (all primary gauges: single segment 0)",
    "river_gauge_train_median_m": "the gauge's training-period median that is subtracted",
    "river_level_flag": "OBSERVED / NO_READING_THAT_DAY / NO_PRIMARY_GAUGE / AFTER_DATUM_BREAK",
    "flood_source_ids": "included flood records covering the day (IFI UEI or IMD:<report>:p<page>:<entry>)",
    "flood_n_sources": "number of included flood records covering the day",
    "flood_long_period_only": "1 if the flood label rests only on LONG_PERIOD_SUMMARY records",
    "flood_excluded_source_ids": "excluded records (LANDSLIDE_ONLY_CAUSE) that cover the day — not used",
    "landslide_source": "GSI, IMD or GSI+IMD",
    "landslide_gsi_n_records": "number of GSI records dated to that district-day",
    "landslide_source_ids": "GSI:<Sl_No> / IMD:<report>:p<page>:s<sno>",
    "landslide_flags": "GSI_NUMERIC_DATE_DM_AMBIGUOUS if a contributing GSI date was numeric and read day-first",
}

LIMITATIONS = [
    "Spatial unit = district (14 locations). Static terrain, drainage, distance-to-river, land-cover and lithology "
    "features take one value per district, so a tree model can memorise district identity through them; "
    "`land_cover` is constant (forest ×14) and `lithology` has 2 values (granite_gneiss ×13, alluvium ×1). "
    "Terrain feature importances from this file are not evidence about processes.",
    "Landslide target is sparse (≈ 8 test positives) and presence-only; test metrics will have very wide "
    "uncertainty.",
    "River stage is missing for Alappuzha, Wayanad and Kottayam (no usable CWC gauge) and before 2015-06-01 for 7 "
    "districts; one point gauge represents a whole district.",
    "Flood labels in 2013, 2015 and 2018 include IMD multi-week summary periods (LONG_PERIOD_SUMMARY), which mark "
    "whole periods rather than flood days — including the 2018 flood season in all 14 districts.",
    "Flood 0 = not reported by IMD, not proven absence; IMD records damaging events.",
    "`soil_moisture` is a 0–1 saturation fraction (GWETROOT), not volumetric water content; POWER cells are ~50 km.",
    "The ML scripts read the hard-coded `synthetic_master_dataset.csv`. Training on this file later needs a path "
    "change in `ml/flood_xgboost.py`, `ml/landslide_xgboost.py`, `ml/tcdl_v1.py`, `ml/shap_explainability.py` and "
    "`backend/config.py` (not changed here).",
]


def build_report(path, m, keys, gauges, train_end, val_end, V, flood_meta, land_meta, flood_before, n_reaches):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor, Cm

    NAVY = RGBColor(0x1F, 0x38, 0x64)
    GREY = RGBColor(0x59, 0x59, 0x59)
    sp = split_of(m["date"], train_end, val_end)
    counts = {s: int((sp == s).sum()) for s in ("train", "validation", "test")}
    vf = V.frame()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.0))
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(4)
    for lvl in (1, 2):
        hs = doc.styles[f"Heading {lvl}"]
        hs.font.name, hs.font.color.rgb = "Calibri", NAVY
        hs.font.size = Pt(14 if lvl == 1 else 11.5)
        hs.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    def shade(cell, fill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"), shd.set(qn("w:color"), "auto"), shd.set(qn("w:fill"), fill)
        tcPr.append(shd)

    def table(header, rows, widths_cm, size=8.5, status_col=None):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(str(h))
            r.bold, r.font.size, r.font.color.rgb = True, Pt(size), RGBColor(0xFF, 0xFF, 0xFF)
            shade(c, "1F3864")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(size)
                if status_col == i:
                    shade(cells[i], {"PASS": "D9EAD3", "FAIL": "F4CCCC", "INFO": "FFF2CC"}.get(str(v), "FFFFFF"))
        for row in t.rows:
            for i, w in enumerate(widths_cm):
                row.cells[i].width = Cm(w)
                row.cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
        doc.add_paragraph()
        return t

    def para(text, bold_lead=None, size=None, color=None, italic=False):
        p = doc.add_paragraph()
        if bold_lead:
            p.add_run(bold_lead).bold = True
        r = p.add_run(text)
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
        r.italic = italic
        return p

    def bullet(text, bold_lead=None):
        p = doc.add_paragraph(style="List Bullet")
        if bold_lead:
            p.add_run(bold_lead).bold = True
        p.add_run(text)
        p.paragraph_format.space_after = Pt(2)
        return p

    t = doc.add_paragraph()
    r = t.add_run("MASTER DATASET REPORT")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(20), NAVY
    para("Real district-day Master Dataset · Kerala Coupled Flood–Landslide Early Warning System", size=11, color=GREY)
    para(f"updated_data/master_dataset/  ·  built {_date.today().strftime('%d %B %Y')} by "
         "updated_data/scripts/13_build_master_dataset.py", size=9, color=GREY)

    n_fail = int((vf["status"] == "FAIL").sum())
    n_pass = int((vf["status"] == "PASS").sum())
    table(["Item", "Result"], [
        ["File", "real_master_dataset.csv — 23 columns: the 22 columns of synthetic_master_dataset.csv, same order, "
                 "plus `district` (identifier only) after `longitude`"],
        ["Size", f"{len(m):,} rows = 14 districts × {m['date'].nunique():,} days ({GRID_START.date()} → {GRID_END.date()})"],
        ["Split (code's chronological_split)", f"train → {train_end.date()} ({counts['train']:,} rows) · validation → "
                                               f"{val_end.date()} ({counts['validation']:,}) · test → {GRID_END.date()} "
                                               f"({counts['test']:,}) — confirmed by running the scripts' own function"],
        ["Labels", f"flood = 1 on {int(m['flood'].sum()):,} district-days; landslide = 1 on {int(m['landslide'].sum())}"],
        ["Validation", f"{len(vf)} checks: {n_pass} PASS, {n_fail} FAIL, {len(vf) - n_pass - n_fail} INFO "
                       "(VALIDATION_RESULTS.csv)"],
        ["Not done", "No model trained; no ML, backend or frontend code changed; no input file modified; no value "
                     "imputed, interpolated or estimated."],
    ], [4.0, 13.0], size=9)

    doc.add_heading("1. Inputs", level=1)
    table(["Column(s)", "Source (updated_data/…)", "Rule"], [
        ["date, district, rainfall_1d…30d_mm", "existing_collected_data/Kerala_CHIRPS_District_Rainfall_2012 + _2013_2024",
         "CHIRPS district daily; trailing N-day sums incl. day t, min_periods = N, warm-up from 2012-01-01"],
        ["latitude, longitude", "reference/Kerala_district_reference.csv", "district representative point (static)"],
        ["soil_moisture, temperature_c, humidity_percent", "weather_soil/NASA_POWER_…_2012_2024.csv",
         "GWETROOT (saturation 0–1), T2M, RH2M"],
        ["elevation_m, slope, aspect, curvature", "terrain/Kerala_District_Terrain_Summary.csv", "district means (static)"],
        ["drainage_density", "hydrology/Kerala_District_Drainage_Density.csv", "km/km² (static)"],
        ["distance_to_river_km", "hydrology/Kerala_HydroRIVERS_v10.geojson",
         f"computed: district point → nearest of {n_reaches:,} reaches, any order, {METRIC_CRS}"],
        ["land_cover", "terrain/Kerala_District_LandCover_Summary.csv", "dominant project class (forest ×14)"],
        ["lithology", "lithology/Kerala_District_Lithology_CGWB.csv", "CGWB dominant class (2 values)"],
        ["river_level_m", "river_level/processed/District_Gauge_Assignment + CWC_Kerala_RiverLevel_Daily_Station",
         "primary gauge stage − its training median"],
        ["flood", "flood_events/processed/IFI_Kerala_2012_2023_corrected + IMD_DWE_Kerala_events_not_in_IFI",
         "decision 2"],
        ["landslide", "landslide_events/processed/GSI_…_date_precision + IMD_DWE_Kerala_landslide_entries", "decision 3"],
    ], [4.2, 6.6, 6.2], size=8)

    doc.add_heading("2. Decisions applied", level=1)
    med_list = ", ".join(f"{r.district} {r.river_gauge.strip()} {r.train_median_m:.2f} m"
                         for r in gauges[gauges["river_gauge"] != ""].itertuples())
    bullet(f" primary CWC gauge's daily mean stage minus that gauge's median over training dates only "
           f"({GRID_START.date()}..{train_end.date()}). All 11 primary gauges have a single datum segment over 2012–2024. "
           f"No gauge: Alappuzha, Wayanad, Kottayam → NaN. Medians: {med_list}.", "River level —")
    bullet(f" 1 on every district-day of an event's event_days (verified dates; one record with empty event_days "
           f"expanded verified_start..verified_end), IFI corrected + {flood_meta['ext_records']} IMD events. "
           f"LONG_PERIOD_SUMMARY kept ({len(flood_meta['long_period'])} records); "
           f"{len(flood_meta['excluded_landslide_only'])} LANDSLIDE_ONLY_CAUSE and "
           f"{len(flood_meta['excluded_no_district'])} NO_DISTRICT records excluded. The IMD event IFI omitted "
           f"(2023-06-28, 4 districts) is {'included' if INCLUDE_OMITTED_FROM_IFI else 'excluded'}. Before exclusions "
           f"{flood_before[0]:,} (train {flood_before[1]:,} / val {flood_before[2]} / test {flood_before[3]}) = the "
           f"documented total for this choice (the prompt's 3,132 / test 119 leaves the omitted event out); after "
           f"exclusions: {int(m['flood'].sum()):,}.", "Flood —")
    rej = land_meta["rejected"]
    rej_w = rej[(rej["date"] >= GRID_START) & (rej["date"] <= GRID_END)] if len(rej) else rej
    bullet(f" exact-date GSI records (all-records file, grid window) ∪ IMD EXACT_DATE single-district entries, "
           f"de-duplicated per district-day; MONTH_YEAR, YEAR_ONLY, NO_DATE, PERIOD, WEEK_ONLY and unattributed "
           f"multi-district entries excluded. {len(rej)} GSI record-dates were not stated in the record's text "
           f"(parser split a year into day numbers; {len(rej_w)} in the grid window) and were not used. Result: "
           f"{int(m['landslide'].sum())} positive district-days; with those artifact dates it would be the documented "
           f"120 (101 / 11 / 8) — the whole difference lies in train (2018–2019).", "Landslide —")
    bullet(" missing stays NaN; nothing imputed. 0 labels mean “not reported”, not proven absence.", "Missing values —")

    doc.add_heading("3. Missing values (by split)", level=1)
    rows = []
    for c in m.columns:
        tot = int(m[c].isna().sum())
        if tot == 0:
            continue
        rows.append([c] + [f"{int(m.loc[sp == s, c].isna().sum()):,} ({100 * m.loc[sp == s, c].isna().mean():.1f} %)"
                           for s in counts] + [f"{tot:,} ({100 * tot / len(m):.1f} %)"])
    table(["Column", "Train", "Validation", "Test", "All"], rows, [3.6, 3.35, 3.35, 3.35, 3.35], size=9)
    para("All other 22 columns are complete. River stage observed on "
         + " / ".join(f"{100 * m.loc[sp == s, 'river_level_m'].notna().mean():.1f} %" for s in counts)
         + " of train / validation / test district-days (expected ≈ 54 / 75 / 78 %).", size=9)

    doc.add_heading("4. Label counts per split", level=1)
    lpo = keys["flood_long_period_only"] == 1
    table(["Split", "Dates", "District-days", "flood = 1", "of which long-period only", "landslide = 1"],
          [[s, f"{m.loc[sp == s, 'date'].min().date()} → {m.loc[sp == s, 'date'].max().date()}", f"{counts[s]:,}",
            f"{int(m.loc[sp == s, 'flood'].sum()):,} ({100 * m.loc[sp == s, 'flood'].mean():.2f} %)",
            f"{int((lpo & (sp == s)).sum()):,}", f"{int(m.loc[sp == s, 'landslide'].sum())}"] for s in counts]
          + [["all", "", f"{len(m):,}", f"{int(m['flood'].sum()):,}", f"{int(lpo.sum()):,}", f"{int(m['landslide'].sum())}"]],
          [2.0, 4.4, 2.4, 3.0, 2.8, 2.4], size=9)

    doc.add_heading("5. Validation", level=1)
    keep = vf[(vf["section"] != "missing_values") & ~((vf["section"] == "labels") & (vf["status"] == "INFO"))]
    table(["ID", "Section", "Check", "Observed", "Status"],
          [[r.check_id, r.section, r.check + ("" if r.split == "all" else f" [{r.split}]"),
            (r.observed[:110] + "…") if len(r.observed) > 110 else r.observed, r.status] for r in keep.itertuples()],
          [1.1, 2.0, 7.6, 5.0, 1.3], size=7.5, status_col=4)

    doc.add_heading("6. Limitations", level=1)
    for x in LIMITATIONS:
        bullet(" " + x.replace("`", ""))

    doc.add_heading("7. Before training", level=1)
    for x in [
        "Point the five hard-coded paths (ml/flood_xgboost.py, ml/landslide_xgboost.py, ml/tcdl_v1.py, "
        "ml/shap_explainability.py, backend/config.py) at updated_data/master_dataset/real_master_dataset.csv — "
        "or make the path configurable. Not changed in this task.",
        "Replace the SYNTHETIC-DATA warnings/plot titles only when the real file is actually used.",
        "Decide whether to keep LONG_PERIOD_SUMMARY flood days (use flood_long_period_only in the keys file for a "
        "sensitivity run) and whether the 8 test landslide positives are enough to report a landslide metric.",
        "Consider dropping the constant land_cover and near-constant lithology/static terrain features or moving to a "
        "finer spatial unit; interpret feature importance accordingly.",
        "Obtain the GSI Bhukosh landslide inventory (manual download) and re-run this script.",
    ]:
        bullet(" " + x)
    doc.save(path)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ref, names = load_reference()
    districts = sorted(ref["district"])
    grid_dates = pd.date_range(GRID_START, GRID_END, freq="D")
    train_end, val_end = split_bounds(grid_dates)
    log(f"Grid: 14 districts x {len(grid_dates)} days; split train ..{train_end.date()} | val ..{val_end.date()}")

    # ---- features (no label file is touched here) ------------------------------------------------
    rain_all = build_rainfall(districts)
    power = build_power()
    static, dist, n_reaches = build_static(ref)
    river, gauges = build_river_level(districts, grid_dates, train_end)
    log("Features built: rainfall, NASA POWER, static terrain/hydrology/land cover/lithology, CWC river level")

    grid = pd.MultiIndex.from_product([grid_dates, districts], names=["date", "district"]).to_frame(index=False)
    grid = grid.merge(static, on="district", how="left")
    grid = grid.sort_values(["date", "latitude", "longitude"], kind="mergesort").reset_index(drop=True)
    df = (grid.merge(rain_all, on=["district", "date"], how="left")
              .merge(power, on=["district", "date"], how="left")
              .merge(river, on=["district", "date"], how="left"))
    assert len(df) == len(grid), "a merge changed the row count"

    # ---- labels -------------------------------------------------------------------------------
    flood_recs, flood_meta = build_flood_labels(names)
    land_recs, land_meta = build_landslide_labels(names)
    df["flood"], flood_keys = aggregate_labels(df, flood_recs, "flood")
    df["landslide"], land_keys = aggregate_labels(df, land_recs, "landslide")
    # the documented pre-exclusion count, reproduced from the same records
    fb_days = flood_recs[(flood_recs["date"] >= GRID_START) & (flood_recs["date"] <= GRID_END)][["district", "date"]]
    fb_days = fb_days.drop_duplicates()
    fsp = split_of(fb_days["date"], train_end, val_end)
    flood_before = (len(fb_days), int((fsp == "train").sum()), int((fsp == "validation").sum()),
                    int((fsp == "test").sum()))
    log(f"Labels: flood {int(df['flood'].sum())} (before exclusions {flood_before[0]}), "
        f"landslide {int(df['landslide'].sum())}")

    # ---- outputs --------------------------------------------------------------------------------
    master = df[MASTER_COLUMNS].copy()
    master["date"] = master["date"].dt.strftime("%Y-%m-%d")
    keys = pd.concat([df[["date", "district"]].assign(date=master["date"],
                                                      split=split_of(df["date"], train_end, val_end).to_numpy()),
                      df[["river_gauge", "river_gauge_river", "river_gauge_status", "river_stage_published_m",
                          "river_datum_segment", "river_gauge_train_median_m", "river_level_flag"]],
                      flood_keys, land_keys], axis=1)
    keys["river_datum_segment"] = keys["river_datum_segment"].astype("Int64")

    V = validate(master, keys, rain_all, gauges, train_end, val_end, flood_meta, flood_recs, land_meta, land_recs,
                 flood_before)
    vf = V.frame()
    master.to_csv(OUT / "real_master_dataset.csv", index=False)
    keys.to_csv(OUT / "real_master_dataset_keys.csv", index=False)
    vf.to_csv(OUT / "VALIDATION_RESULTS.csv", index=False)
    m = master.assign(date=pd.to_datetime(master["date"]))
    write_dictionary(DOCS / "REAL_DATA_DICTIONARY.md", m, keys, static, dist, gauges, n_reaches, train_end, val_end, V,
                     flood_meta, land_meta, flood_before)
    build_report(DOCS / "MASTER_DATASET_REPORT.docx", m, keys, gauges, train_end, val_end, V, flood_meta, land_meta,
                 flood_before, n_reaches)

    n_fail = int((vf["status"] == "FAIL").sum())
    log(f"\nWritten to {OUT}")
    log(f"  real_master_dataset.csv  {master.shape[0]} rows x {master.shape[1]} columns")
    log(f"  VALIDATION_RESULTS.csv   {len(vf)} checks: {int((vf['status'] == 'PASS').sum())} PASS, {n_fail} FAIL, "
        f"{int((vf['status'] == 'INFO').sum())} INFO")
    if n_fail:
        pd.set_option("display.width", 250), pd.set_option("display.max_colwidth", 200)
        log(vf[vf["status"] == "FAIL"].to_string(index=False))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
