"""
Temporal Coupled Decision Layer (TCDL) -- Version 1.0
=====================================================================
Rule-based coupling layer for the flood-landslide early warning
research project.

    Master Dataset
        |-- Flood XGBoost V1.0      -> flood probability      (frozen)
        |-- Landslide XGBoost V1.0  -> landslide probability  (frozen)
                          |
                          v
              Temporal Coupled Decision Layer  <-- THIS FILE
                          |
                          v
                 Early Warning Decision
                          |
                          v
                 Lead-Time Evaluation

What this layer is
------------------
A transparent, rule-based decision layer. It trains NOTHING. It loads
the two frozen boosters, generates their probabilities, derives temporal
signals (smoothing, trend, rate of increase) from those probabilities
and from rainfall / soil moisture, and applies an explicit rule set.
Every warning records exactly which rules fired.

What this layer is NOT
----------------------
It is NOT demonstrated to maximise lead time. TCDL V1.0 is a PROPOSED
coupling mechanism. Whether it helps is an experimental question, and
this file exists to measure it -- not to assert it.

Read this before interpreting any lead-time number
--------------------------------------------------
1. STRUCTURAL DOMINANCE. Rules R1 and R2 replicate the two single-model
   baselines exactly (probability >= 0.50). The coupled warning set is
   therefore a strict SUPERSET of the union of both baselines, so it
   cannot detect fewer events and cannot warn later. Any lead-time
   "improvement" is partly a construction artefact, not evidence. The
   honest question this file answers is: how much ADDITIONAL lead time
   do the coupling rules (R3-R7) buy, and at what false-alarm cost?
   Both numbers are reported side by side.
2. DAILY QUANTISATION. The master dataset is daily and carries no
   sub-daily timestamps. Lead time is therefore a multiple of 24 h and
   is an UPPER BOUND (see lead_time_note in the results file).
3. UNCALIBRATED PROBABILITIES. Both models were trained with
   scale_pos_weight, so their probabilities run well above the observed
   event rate. Thresholds here are expressed relative to the existing
   0.50 baseline, but they are NOT calibrated probabilities. Calibration
   is deliberately out of scope for this version.

*** SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS ***
Every number produced here describes a simulated dataset. None of it is
evidence about real-world flood or landslide warning performance.

Usage
-----
    python ml/tcdl_v1.py

Reads  : synthetic_master_dataset.csv          (read-only)
         ml/flood_xgboost_model.json           (frozen, read-only)
         ml/flood_features.json                (read-only)
         ml/landslide_xgboost_model.json       (frozen, read-only)
         ml/landslide_features.json            (read-only)
         ml/event_timestamps.csv               (OPTIONAL -- see below)
Writes : ml/tcdl_timeseries.csv
         ml/tcdl_warnings.csv
         ml/tcdl_lead_time.csv
         ml/tcdl_results.json
         ml/tcdl_example_timeline.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import xgboost as xgb

# =====================================================================
# Paths
# =====================================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_PATH = PROJECT_ROOT / "synthetic_master_dataset.csv"

FLOOD_MODEL_PATH = HERE / "flood_xgboost_model.json"
FLOOD_FEATURES_PATH = HERE / "flood_features.json"
FLOOD_RESULTS_PATH = HERE / "flood_model_results.json"
LANDSLIDE_MODEL_PATH = HERE / "landslide_xgboost_model.json"
LANDSLIDE_FEATURES_PATH = HERE / "landslide_features.json"

# OPTIONAL external event table. When real event onset timestamps become
# available, drop a CSV here with columns:
#     date,latitude,longitude,hazard_type,event_timestamp
# where `event_timestamp` is an ISO datetime with real hour resolution
# and `hazard_type` is one of flood | landslide.
# If the file is absent, onsets are DERIVED from the dataset's own label
# columns at daily resolution (documented, not invented).
EVENT_TIMESTAMPS_PATH = HERE / "event_timestamps.csv"

TIMESERIES_PATH = HERE / "tcdl_timeseries.csv"
WARNINGS_PATH = HERE / "tcdl_warnings.csv"
LEAD_TIME_PATH = HERE / "tcdl_lead_time.csv"
RESULTS_PATH = HERE / "tcdl_results.json"
TIMELINE_PLOT_PATH = HERE / "tcdl_example_timeline.png"

# =====================================================================
# TCDL parameters -- every threshold is explicit and reported
# =====================================================================
# Baseline decision threshold inherited from both V1.0 models.
BASELINE_THRESHOLD = 0.50

# Smoothing / trend windows, in days.
SMOOTH_WINDOW = 3          # moving average length
RATE_WINDOW = 3            # rate of increase measured over this many days
SUSTAIN_DAYS = 2           # consecutive days required by the "sustained" rule

# --- How the rule thresholds are set ---------------------------------
# NOT hand-picked absolute probabilities. Both models were trained with
# scale_pos_weight, so their outputs are inflated relative to the true
# event rate -- a "0.45 probability" here does not mean 45%. Fixing
# absolute cut-offs against that scale produced a rule that fired on 52%
# of all days.
#
# Instead each threshold is a QUANTILE of the corresponding signal over
# the TRAINING period only. That makes the rule set
#   * principled     -- "fire in the top decile of joint risk", not a guess
#   * leakage-free   -- the test period is never consulted
#   * self-adapting  -- if the models are calibrated later, the absolute
#                       values move with them and the rules still mean
#                       the same thing
# The resolved absolute values are printed and written to the results
# file, so every decision remains fully traceable.
THRESHOLD_QUANTILES = {
    "alert_level":  ("coupled_prob_ma", 0.90),        # R3 sustained joint risk
    "watch_level":  ("coupled_prob_ma", 0.75),        # R6 environmental precursor floor
    "rising_flood_level":     ("flood_prob_ma", 0.75),        # R4
    "rising_landslide_level": ("landslide_prob_ma", 0.75),    # R5
    "joint_flood_level":      ("flood_prob_ma", 0.80),        # R7
    "joint_landslide_level":  ("landslide_prob_ma", 0.80),    # R7
    "flood_rate":     ("flood_prob_rate", 0.90),      # R4
    "landslide_rate": ("landslide_prob_rate", 0.90),  # R5
    "rain_rate":      ("rainfall_rate", 0.95),        # R6
    "soil_rate":      ("soil_moisture_rate", 0.90),   # R6
}

# Lead-time evaluation window: how far back before an onset a warning may
# be credited. A warning earlier than this is treated as unrelated.
MAX_LOOKBACK_DAYS = 14

# Separate, much tighter horizon for judging whether a warning DAY was
# operationally useful. This is deliberately NOT MAX_LOOKBACK_DAYS:
# with hazards occurring on ~14% of days, a 14-day window makes almost
# every day "shortly before some event", so the resulting false-alarm
# rate cannot distinguish a real system from one that warns every single
# day. The always-warn control below exists to keep that honest.
USEFUL_HORIZON_DAYS = 3

HOURS_PER_DAY = 24

# Evaluation is restricted to the frozen models' TEST period. Earlier
# rows are scored only to warm up the rolling windows -- never evaluated.
EVAL_SPLIT = "test"


def banner(text: str, char: str = "=") -> None:
    print("\n" + char * 70)
    print(text)
    print(char * 70)


# =====================================================================
# 1. Load frozen models and score the master dataset
# =====================================================================
def load_master_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["date", "latitude", "longitude"],
                        kind="mergesort").reset_index(drop=True)
    # Stable per-cell identifier for all groupby-time operations.
    df["location_id"] = (df["latitude"].astype(str) + "_" + df["longitude"].astype(str))
    print("Master dataset loaded")
    print(f"  rows={len(df)}  columns={df.shape[1]}  "
          f"locations={df['location_id'].nunique()}  "
          f"dates {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def _load_meta(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_with_frozen_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate both hazard probabilities. The models are LOADED, never
    retrained and never modified. Each model sees only its own feature
    list, so the two remain independent -- coupling happens strictly
    downstream of this function.
    """
    flood_meta = _load_meta(FLOOD_FEATURES_PATH)
    land_meta = _load_meta(LANDSLIDE_FEATURES_PATH)

    flood_features = flood_meta["feature_order"]
    land_features = land_meta["feature_order"]
    cat_schema = land_meta["categorical_handling"]["category_schema"]

    # Independence guard: neither model may see the other's target or output.
    for name, feats in [("flood", flood_features), ("landslide", land_features)]:
        forbidden = {"flood", "landslide", "flood_probability",
                     "landslide_probability"} & set(feats)
        if forbidden:
            raise RuntimeError(
                f"{name} model feature list contains {sorted(forbidden)} -- "
                "the independent baselines have been compromised.")

    flood_booster = xgb.Booster()
    flood_booster.load_model(str(FLOOD_MODEL_PATH))
    land_booster = xgb.Booster()
    land_booster.load_model(str(LANDSLIDE_MODEL_PATH))

    # Rebuild the landslide categorical schema exactly as saved.
    X_land = df[land_features].copy()
    for col, cats in cat_schema.items():
        X_land[col] = X_land[col].astype(CategoricalDtype(categories=cats, ordered=False))

    out = df.copy()
    out["flood_probability"] = flood_booster.predict(xgb.DMatrix(df[flood_features]))
    out["landslide_probability"] = land_booster.predict(
        xgb.DMatrix(X_land, enable_categorical=True))

    print("\nFrozen models loaded and scored (no retraining)")
    print(f"  flood     : {FLOOD_MODEL_PATH.name}  "
          f"({len(flood_features)} features, {flood_meta.get('n_trees')} trees)")
    print(f"  landslide : {LANDSLIDE_MODEL_PATH.name}  "
          f"({len(land_features)} features, {land_meta.get('n_trees')} trees)")
    print(f"  flood_probability     range "
          f"{out['flood_probability'].min():.4f} - {out['flood_probability'].max():.4f}")
    print(f"  landslide_probability range "
          f"{out['landslide_probability'].min():.4f} - {out['landslide_probability'].max():.4f}")
    return out


def tag_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row train / validation / test using the frozen models'
    own split boundaries, so evaluation can be restricted to the period
    neither model ever saw.
    """
    split_meta = _load_meta(FLOOD_RESULTS_PATH)["split"]
    train_end = pd.Timestamp(split_meta["train_date_range"][1])
    val_end = pd.Timestamp(split_meta["validation_date_range"][1])

    df = df.copy()
    df["split"] = np.where(df["date"] <= train_end, "train",
                    np.where(df["date"] <= val_end, "validation", "test"))
    print("\nSplit tagging (inherited from the frozen models)")
    for s in ("train", "validation", "test"):
        part = df[df["split"] == s]
        print(f"  {s:<11} {len(part):5d} rows  "
              f"{part['date'].min().date()} -> {part['date'].max().date()}")
    print(f"  TCDL is EVALUATED on '{EVAL_SPLIT}' only; earlier rows serve")
    print("  solely to warm up the rolling windows.")
    return df


# =====================================================================
# 2. Temporal signals
# =====================================================================
def build_temporal_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive smoothed levels and rates of increase.

    EVERY rolling operation is grouped by location and ordered by date.
    Mixing locations, or letting a window straddle a location boundary,
    would silently blend unrelated catchments -- so the grouping is not
    optional.

    All windows are TRAILING (they use day t and earlier), so no signal
    on day t depends on the future.
    """
    df = df.sort_values(["location_id", "date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("location_id", sort=False)

    def roll_mean(col: str) -> pd.Series:
        return g[col].transform(
            lambda s: s.rolling(SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean())

    # --- coupled probability: transparent noisy-OR, no fitting --------
    # P(at least one hazard) under an independence assumption. Stated as
    # an assumption, not a claim: the two hazards share rainfall drivers,
    # so this is an approximation used for ranking, not a calibrated
    # joint probability.
    df["coupled_probability"] = (
        1.0 - (1.0 - df["flood_probability"]) * (1.0 - df["landslide_probability"]))

    # --- smoothed signals (moving average) ---------------------------
    df["flood_prob_ma"] = roll_mean("flood_probability")
    df["landslide_prob_ma"] = roll_mean("landslide_probability")
    df["coupled_prob_ma"] = roll_mean("coupled_probability")
    df["rainfall_ma"] = roll_mean("rainfall_1d_mm")
    df["soil_moisture_ma"] = roll_mean("soil_moisture")

    # --- rate of increase over RATE_WINDOW days ----------------------
    # Change in the SMOOTHED signal, so day-to-day noise does not
    # masquerade as a trend.
    def rate(col: str) -> pd.Series:
        return g[col].transform(lambda s: s.diff(RATE_WINDOW))

    df["flood_prob_rate"] = rate("flood_prob_ma")
    df["landslide_prob_rate"] = rate("landslide_prob_ma")
    df["coupled_prob_rate"] = rate("coupled_prob_ma")
    df["rainfall_rate"] = rate("rainfall_ma")
    df["soil_moisture_rate"] = rate("soil_moisture_ma")

    # --- boolean trend flags (used by the rules, kept for traceability)
    df["flood_trend_up"] = df["flood_prob_rate"] > 0
    df["landslide_trend_up"] = df["landslide_prob_rate"] > 0
    df["rainfall_trend_up"] = df["rainfall_rate"] > 0
    df["soil_trend_up"] = df["soil_moisture_rate"] > 0

    print("\nTemporal signals built (per location, trailing windows only)")
    print(f"  moving average window : {SMOOTH_WINDOW} days")
    print(f"  rate-of-increase span : {RATE_WINDOW} days (diff of the smoothed series)")
    print("  signals: flood_prob, landslide_prob, coupled_prob (noisy-OR), "
          "rainfall, soil_moisture")
    return df


# =====================================================================
# 3. The rule set
# =====================================================================
# Each rule is a named, independently-inspectable predicate. A warning is
# raised when ANY rule fires, and the firing rule IDs are recorded on the
# row so every decision is traceable.
COUPLING_RULES = ["R3_SUSTAINED_JOINT", "R4_RISING_FLOOD", "R5_RISING_LANDSLIDE",
                  "R6_ENV_PRECURSOR", "R7_JOINT_MODERATE"]
BASELINE_RULES = ["R1_FLOOD_LEVEL", "R2_LANDSLIDE_LEVEL"]
RULE_IDS = BASELINE_RULES + COUPLING_RULES


def resolve_thresholds(df: pd.DataFrame) -> dict:
    """Resolve every quantile spec into an absolute value, TRAIN split only."""
    train = df[df["split"] == "train"]
    th = {}
    for name, (signal, q) in THRESHOLD_QUANTILES.items():
        th[name] = float(train[signal].quantile(q))
    print("\nRule thresholds resolved from TRAINING-period quantiles "
          "(test never consulted)")
    for name, (signal, q) in THRESHOLD_QUANTILES.items():
        print(f"  {name:<24} = p{int(q*100):<3d} of {signal:<20} = {th[name]:.4f}")
    return th


def describe_rules(th: dict) -> dict:
    """Human-readable rule text with the resolved numbers substituted in."""
    return {
        "R1_FLOOD_LEVEL": (
            f"flood_probability >= {BASELINE_THRESHOLD} "
            "(replicates the flood-only baseline)"),
        "R2_LANDSLIDE_LEVEL": (
            f"landslide_probability >= {BASELINE_THRESHOLD} "
            "(replicates the landslide-only baseline)"),
        "R3_SUSTAINED_JOINT": (
            f"smoothed coupled probability >= {th['alert_level']:.4f} "
            f"(train p90) for {SUSTAIN_DAYS} consecutive days"),
        "R4_RISING_FLOOD": (
            f"smoothed flood probability >= {th['rising_flood_level']:.4f} (train p75) "
            f"AND rising by >= {th['flood_rate']:.4f} (train p90) over {RATE_WINDOW} days"),
        "R5_RISING_LANDSLIDE": (
            f"smoothed landslide probability >= {th['rising_landslide_level']:.4f} "
            f"(train p75) AND rising by >= {th['landslide_rate']:.4f} (train p90) "
            f"over {RATE_WINDOW} days"),
        "R6_ENV_PRECURSOR": (
            f"smoothed coupled probability >= {th['watch_level']:.4f} (train p75) AND "
            f"rainfall rising by >= {th['rain_rate']:.2f} mm/day (train p95) AND "
            f"soil moisture rising by >= {th['soil_rate']:.4f} (train p90), "
            f"rates over {RATE_WINDOW} days"),
        "R7_JOINT_MODERATE": (
            f"smoothed flood probability >= {th['joint_flood_level']:.4f} AND smoothed "
            f"landslide probability >= {th['joint_landslide_level']:.4f} simultaneously "
            f"(both train p80) -- joint moderate risk where neither hazard alone "
            f"reaches {BASELINE_THRESHOLD}"),
    }


def apply_tcdl_rules(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    """Evaluate every rule, then combine. NaN (warm-up) never fires."""
    df = df.sort_values(["location_id", "date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("location_id", sort=False)

    # R1 / R2 -- single-model level rules (identical to the baselines).
    df["R1_FLOOD_LEVEL"] = df["flood_probability"] >= BASELINE_THRESHOLD
    df["R2_LANDSLIDE_LEVEL"] = df["landslide_probability"] >= BASELINE_THRESHOLD

    # R3 -- sustained joint risk: elevated smoothed coupled probability on
    # SUSTAIN_DAYS consecutive days.
    above = (df["coupled_prob_ma"] >= th["alert_level"]).fillna(False)
    df["_above_alert"] = above
    df["R3_SUSTAINED_JOINT"] = g["_above_alert"].transform(
        lambda s: s.rolling(SUSTAIN_DAYS, min_periods=SUSTAIN_DAYS).sum() == SUSTAIN_DAYS
    ).fillna(False).astype(bool)

    # R4 / R5 -- rising single-hazard risk below the baseline level.
    df["R4_RISING_FLOOD"] = (
        (df["flood_prob_ma"] >= th["rising_flood_level"]) &
        (df["flood_prob_rate"] >= th["flood_rate"])).fillna(False)
    df["R5_RISING_LANDSLIDE"] = (
        (df["landslide_prob_ma"] >= th["rising_landslide_level"]) &
        (df["landslide_prob_rate"] >= th["landslide_rate"])).fillna(False)

    # R6 -- environmental precursor: moderate joint risk confirmed by a
    # simultaneous rainfall AND soil-moisture rise.
    df["R6_ENV_PRECURSOR"] = (
        (df["coupled_prob_ma"] >= th["watch_level"]) &
        (df["rainfall_rate"] >= th["rain_rate"]) &
        (df["soil_moisture_rate"] >= th["soil_rate"])).fillna(False)

    # R7 -- the distinctly coupled rule: both hazards moderately elevated
    # at once, where neither on its own would trigger a baseline warning.
    df["R7_JOINT_MODERATE"] = (
        (df["flood_prob_ma"] >= th["joint_flood_level"]) &
        (df["landslide_prob_ma"] >= th["joint_landslide_level"])).fillna(False)

    rule_cols = RULE_IDS
    for c in rule_cols:
        df[c] = df[c].astype(bool)

    df["tcdl_warning"] = df[rule_cols].any(axis=1).astype(int)
    df["tcdl_triggered_rules"] = df[rule_cols].apply(
        lambda r: "|".join([c for c in rule_cols if r[c]]), axis=1)
    df["tcdl_n_rules"] = df[rule_cols].sum(axis=1).astype(int)

    # Coupling-only view: what the trend rules would do WITHOUT the two
    # baseline-replicating rules. This is what isolates the coupling
    # contribution from the structural dominance of R1/R2.
    df["tcdl_coupling_only_warning"] = df[COUPLING_RULES].any(axis=1).astype(int)

    # --- baselines ----------------------------------------------------
    df["flood_only_warning"] = (df["flood_probability"] >= BASELINE_THRESHOLD).astype(int)
    df["landslide_only_warning"] = (
        df["landslide_probability"] >= BASELINE_THRESHOLD).astype(int)

    # Warning timestamp. The decision for day t uses data through the END
    # of day t, so the timestamp is the day itself at 00:00 ONLY as a
    # daily-resolution convention -- the dataset carries no time of day.
    df["warning_timestamp"] = np.where(
        df["tcdl_warning"] == 1, df["date"].dt.strftime("%Y-%m-%d"), "")

    df = df.drop(columns=["_above_alert"])
    return df


# =====================================================================
# 4. Event onsets
# =====================================================================
def derive_event_onsets(df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Build the event table used for lead-time evaluation.

    Preferred source: a real external timestamp table at
    EVENT_TIMESTAMPS_PATH (hour resolution). Nothing here fabricates one.

    Fallback (current situation): onsets are DERIVED from the dataset's
    own binary label columns -- the first day of each run of consecutive
    hazard days at a location. That is a documented derivation from data
    that exists, not an invented timestamp, but it is limited to DAILY
    resolution.
    """
    if EVENT_TIMESTAMPS_PATH.exists():
        ev = pd.read_csv(EVENT_TIMESTAMPS_PATH, parse_dates=["event_timestamp"])
        ev["location_id"] = ev["latitude"].astype(str) + "_" + ev["longitude"].astype(str)
        source = {
            "source": "external_event_timestamps",
            "path": EVENT_TIMESTAMPS_PATH.name,
            "resolution": "as supplied (hour resolution expected)",
            "n_events": int(len(ev)),
        }
        print(f"\nEvent onsets: loaded {len(ev)} real timestamps from "
              f"{EVENT_TIMESTAMPS_PATH.name}")
        return ev, source

    # --- fallback: derive daily onsets from the label columns ---------
    d = df.sort_values(["location_id", "date"], kind="mergesort").copy()
    d["any_hazard"] = ((d["flood"] == 1) | (d["landslide"] == 1)).astype(int)

    rows = []
    for hazard in ("flood", "landslide", "any_hazard"):
        prev = d.groupby("location_id", sort=False)[hazard].shift(1)
        onset = (d[hazard] == 1) & (prev != 1)
        sub = d.loc[onset, ["date", "latitude", "longitude", "location_id", "split"]].copy()
        sub["hazard_type"] = hazard
        sub["event_timestamp"] = sub["date"]      # 00:00, daily resolution
        rows.append(sub)
    ev = pd.concat(rows, ignore_index=True)
    ev = ev[ev["split"] == EVAL_SPLIT].reset_index(drop=True)

    source = {
        "source": "derived_from_label_columns",
        "method": ("first day of each run of consecutive hazard days per location "
                   "(onset = hazard==1 and previous day != 1)"),
        "resolution": "daily (00:00); the master dataset has no time-of-day component",
        "n_events": int(len(ev)),
        "external_hook": (f"place a CSV at ml/{EVENT_TIMESTAMPS_PATH.name} with columns "
                          "date,latitude,longitude,hazard_type,event_timestamp to use "
                          "real hour-resolution timestamps instead"),
    }
    print(f"\nEvent onsets: DERIVED from label columns (no external timestamp file)")
    for hz in ("flood", "landslide", "any_hazard"):
        print(f"  {hz:<12} {int((ev['hazard_type'] == hz).sum()):4d} onsets "
              f"in the {EVAL_SPLIT} period")
    print(f"  MISSING DATA: real event onset timestamps with hour resolution.")
    print(f"  Lead time is therefore quantised to {HOURS_PER_DAY} h multiples.")
    return ev, source


# =====================================================================
# 5. Lead-time evaluation
# =====================================================================
def evaluate_lead_time(eval_df: pd.DataFrame, events: pd.DataFrame,
                       warning_col: str, system_name: str,
                       hazard_types: tuple[str, ...]) -> pd.DataFrame:
    """
    Lead Time = Event Onset Time - Warning Time.

    For each event onset, credit the EARLIEST warning day inside
    [onset - MAX_LOOKBACK_DAYS, onset] at the same location. Two variants
    are reported because they answer different questions:

      * lead_time_hours_earliest   -- earliest warning in the window,
        even if the warning lapsed and returned. Optimistic.
      * lead_time_hours_contiguous -- length of the UNBROKEN warning run
        ending at the onset. Conservative; this is the one an operator
        would actually experience as continuous alert time.

    A warning on the onset day itself scores 0 h: detected, but with no
    lead, and it is strictly a nowcast (it uses that day's own rainfall).
    """
    warn = (eval_df[eval_df[warning_col] == 1]
            .groupby("location_id")["date"].apply(set).to_dict())

    ev = events[events["hazard_type"].isin(hazard_types)].copy()
    out = []
    for _, e in ev.iterrows():
        loc, onset = e["location_id"], pd.Timestamp(e["event_timestamp"]).normalize()
        days = warn.get(loc, set())
        window = [d for d in days
                  if 0 <= (onset - d).days <= MAX_LOOKBACK_DAYS]

        if window:
            earliest = min(window)
            lead_earliest = (onset - earliest).days
            # contiguous run ending at onset (walking backwards day by day)
            k = 0
            while (onset - pd.Timedelta(days=k)) in days:
                k += 1
            lead_contig = max(k - 1, 0)
            detected = 1
        else:
            earliest, lead_earliest, lead_contig, detected = pd.NaT, np.nan, np.nan, 0

        out.append(dict(
            system=system_name,
            hazard_type=e["hazard_type"],
            location_id=loc,
            latitude=e["latitude"], longitude=e["longitude"],
            event_timestamp=onset.strftime("%Y-%m-%d"),
            detected=detected,
            first_warning_timestamp=("" if pd.isna(earliest)
                                     else earliest.strftime("%Y-%m-%d")),
            lead_time_days_earliest=lead_earliest,
            lead_time_hours_earliest=(np.nan if np.isnan(lead_earliest)
                                      else lead_earliest * HOURS_PER_DAY),
            lead_time_days_contiguous=lead_contig,
            lead_time_hours_contiguous=(np.nan if np.isnan(lead_contig)
                                        else lead_contig * HOURS_PER_DAY),
        ))
    return pd.DataFrame(out)


def summarise_lead_time(tbl: pd.DataFrame) -> dict:
    det = tbl[tbl["detected"] == 1]
    n = len(tbl)
    return {
        "n_events": int(n),
        "n_detected": int(len(det)),
        "detection_rate": round(len(det) / n, 4) if n else None,
        "mean_lead_time_hours_earliest": (round(float(det["lead_time_hours_earliest"].mean()), 2)
                                          if len(det) else None),
        "median_lead_time_hours_earliest": (round(float(det["lead_time_hours_earliest"].median()), 2)
                                            if len(det) else None),
        "max_lead_time_hours_earliest": (float(det["lead_time_hours_earliest"].max())
                                         if len(det) else None),
        "mean_lead_time_hours_contiguous": (round(float(det["lead_time_hours_contiguous"].mean()), 2)
                                            if len(det) else None),
        "median_lead_time_hours_contiguous": (round(float(det["lead_time_hours_contiguous"].median()), 2)
                                              if len(det) else None),
        "n_zero_lead_nowcast": int((det["lead_time_hours_earliest"] == 0).sum()) if len(det) else 0,
    }


def warning_load(eval_df: pd.DataFrame, events: pd.DataFrame,
                 warning_col: str) -> dict:
    """
    The cost side of the ledger. A warning day is 'useful' if some onset
    at that location occurs within MAX_LOOKBACK_DAYS after it; otherwise
    it is a false-alarm day. Lead time without this number is meaningless
    -- warning every single day would maximise lead time.
    """
    onsets = (events.groupby("location_id")["event_timestamp"]
              .apply(lambda s: {pd.Timestamp(x).normalize() for x in s}).to_dict())
    w = eval_df[eval_df[warning_col] == 1]
    useful_short, useful_long = 0, 0
    for _, r in w.iterrows():
        days = onsets.get(r["location_id"], set())
        deltas = [(o - r["date"]).days for o in days]
        if any(0 <= d <= USEFUL_HORIZON_DAYS for d in deltas):
            useful_short += 1
        if any(0 <= d <= MAX_LOOKBACK_DAYS for d in deltas):
            useful_long += 1
    n_warn = len(w)
    return {
        "n_warning_days": int(n_warn),
        # Time spent under warning: the plainest cost measure, and the one
        # the always-warn control saturates at 100%.
        "time_in_warning_rate": round(n_warn / len(eval_df), 4) if len(eval_df) else None,
        # PRIMARY false-alarm metric: no onset within USEFUL_HORIZON_DAYS.
        "n_false_alarm_days": int(n_warn - useful_short),
        "false_alarm_day_rate": round((n_warn - useful_short) / n_warn, 4) if n_warn else None,
        "useful_horizon_days": USEFUL_HORIZON_DAYS,
        # Secondary, lenient variant kept for transparency.
        "false_alarm_day_rate_lookback_window": (
            round((n_warn - useful_long) / n_warn, 4) if n_warn else None),
        "lookback_window_days": MAX_LOOKBACK_DAYS,
    }


# =====================================================================
# 6. Plot
# =====================================================================
def plot_example_timeline(eval_df: pd.DataFrame, events: pd.DataFrame,
                          th: dict, path: Path) -> str:
    """Explainability aid: one location's signals, warnings and onsets."""
    counts = (events[events["hazard_type"] == "any_hazard"]
              .groupby("location_id").size().sort_values(ascending=False))
    loc = counts.index[0]
    d = eval_df[eval_df["location_id"] == loc].sort_values("date")
    ev = events[(events["location_id"] == loc) &
                (events["hazard_type"] == "any_hazard")]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.2]})

    ax = axes[0]
    ax.plot(d["date"], d["flood_probability"], lw=0.9, alpha=0.45, color="#2c7fb8")
    ax.plot(d["date"], d["flood_prob_ma"], lw=1.8, color="#2c7fb8", label="flood (smoothed)")
    ax.plot(d["date"], d["landslide_probability"], lw=0.9, alpha=0.45, color="#d95f0e")
    ax.plot(d["date"], d["landslide_prob_ma"], lw=1.8, color="#d95f0e", label="landslide (smoothed)")
    ax.plot(d["date"], d["coupled_prob_ma"], lw=1.6, color="#31a354", ls="--", label="coupled (smoothed)")
    ax.axhline(BASELINE_THRESHOLD, color="k", lw=0.8, ls=":", label=f"baseline {BASELINE_THRESHOLD}")
    ax.axhline(th["alert_level"], color="#31a354", lw=0.8, ls=":",
               label=f"alert {th['alert_level']:.2f}")
    for t in ev["event_timestamp"]:
        ax.axvline(pd.Timestamp(t), color="crimson", lw=1.0, alpha=0.65)
    ax.set_ylabel("probability")
    ax.set_title(f"TCDL V1.0 -- location {loc}  (red lines = hazard onsets)\n"
                 "SYNTHETIC DATA -- NOT FOR RESEARCH RESULTS", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper left")

    ax = axes[1]
    ax.plot(d["date"], d["rainfall_ma"], color="#3182bd", lw=1.4, label="rainfall MA (mm/day)")
    ax.set_ylabel("rainfall MA")
    ax2 = ax.twinx()
    ax2.plot(d["date"], d["soil_moisture_ma"], color="#756bb1", lw=1.4, label="soil moisture MA")
    ax2.set_ylabel("soil moisture MA")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    ax = axes[2]
    for i, (col, lab, c) in enumerate([
            ("flood_only_warning", "flood-only", "#2c7fb8"),
            ("landslide_only_warning", "landslide-only", "#d95f0e"),
            ("tcdl_warning", "TCDL coupled", "#31a354")]):
        days = d.loc[d[col] == 1, "date"]
        ax.scatter(days, np.full(len(days), i), s=9, color=c, marker="s", label=lab)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["flood-only", "landslide-only", "TCDL"], fontsize=8)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("date")
    for t in ev["event_timestamp"]:
        ax.axvline(pd.Timestamp(t), color="crimson", lw=1.0, alpha=0.65)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Example timeline plot -> {path}")
    return loc


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    banner("TEMPORAL COUPLED DECISION LAYER (TCDL) -- VERSION 1.0")
    print("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS")
    print("Rule-based coupling layer. Trains nothing. Both XGBoost models frozen.")
    print("TCDL V1.0 is a PROPOSED mechanism -- its value is measured here,")
    print("not asserted.")

    # ---- 1. Inputs ---------------------------------------------------
    banner("1. LOADING DATA AND FROZEN MODELS", "-")
    df = load_master_dataset()
    df = score_with_frozen_models(df)
    df = tag_splits(df)

    # ---- 2. Temporal signals ----------------------------------------
    banner("2. TEMPORAL SIGNALS", "-")
    df = build_temporal_signals(df)

    # ---- 3. Rules ----------------------------------------------------
    banner("3. TCDL RULE SET", "-")
    thresholds = resolve_thresholds(df)
    rule_descriptions = describe_rules(thresholds)
    print()
    for rid, desc in rule_descriptions.items():
        tag = "baseline-replicating" if rid in BASELINE_RULES else "coupling"
        print(f"  {rid:<22} [{tag}]")
        print(f"      {desc}")
    df = apply_tcdl_rules(df, thresholds)

    eval_df = df[df["split"] == EVAL_SPLIT].copy()
    print(f"\n  Rules applied. Evaluation set: {len(eval_df)} rows "
          f"({eval_df['date'].min().date()} -> {eval_df['date'].max().date()})")

    print("\n  Rule firing counts on the evaluation period:")
    for rid in RULE_IDS:
        n = int(eval_df[rid].sum())
        print(f"    {rid:<22} {n:5d} days  ({100*n/len(eval_df):5.2f}%)")

    # ---- 4. Events ---------------------------------------------------
    banner("4. EVENT ONSETS", "-")
    events, event_source = derive_event_onsets(df, eval_df)

    # ---- 5. Lead time ------------------------------------------------
    banner("5. LEAD-TIME EVALUATION", "-")
    print(f"  Lead Time = Event Onset - Warning Time, credited within "
          f"{MAX_LOOKBACK_DAYS} days.")
    print(f"  Resolution: {HOURS_PER_DAY} h (daily dataset). Reported hours are an")
    print("  UPPER BOUND: the true value lies within 24 h below each figure,")
    print("  because neither warnings nor onsets carry a time of day.")

    # Trivial control: warn every day, everywhere. It achieves 100%
    # detection and the maximum possible lead time, and is useless. Any
    # lead-time figure must be read against it -- without this row, a
    # system that simply warns more often looks like a better system.
    eval_df["always_warn"] = 1

    systems = [
        ("flood_only", "flood_only_warning"),
        ("landslide_only", "landslide_only_warning"),
        ("tcdl_coupled", "tcdl_warning"),
        ("tcdl_coupling_rules_only", "tcdl_coupling_only_warning"),
        ("always_warn_control", "always_warn"),
    ]

    # Primary comparison: all three systems against the SAME event set
    # (any hazard). This is the fair basis -- a coupled warning system is
    # judged on warning before ANY hazard, which is exactly where the
    # single-hazard baselines are structurally blind.
    lead_tables, summary = [], {"any_hazard": {}, "flood": {}, "landslide": {}}
    for name, col in systems:
        t = evaluate_lead_time(eval_df, events, col, name, ("any_hazard",))
        lead_tables.append(t)
        summary["any_hazard"][name] = {**summarise_lead_time(t),
                                       **warning_load(eval_df, events[events.hazard_type == "any_hazard"], col)}

    # Secondary: each system against its own hazard.
    for hz in ("flood", "landslide"):
        for name, col in systems:
            t = evaluate_lead_time(eval_df, events, col, f"{name}__vs_{hz}", (hz,))
            lead_tables.append(t)
            summary[hz][name] = {**summarise_lead_time(t),
                                 **warning_load(eval_df, events[events.hazard_type == hz], col)}

    lead_df = pd.concat(lead_tables, ignore_index=True)

    def show(block: str, title: str) -> None:
        print(f"\n  --- {title} ---")
        print(f"    {'system':<26}{'events':>7}{'det%':>7}"
              f"{'mean h':>9}{'med h':>8}{'contig h':>10}"
              f"{'in-warn%':>10}{'FA%(3d)':>9}")
        for name, _ in systems:
            s = summary[block][name]
            print(f"    {name:<26}{s['n_events']:>7}"
                  f"{100*s['detection_rate']:>6.1f}%"
                  f"{(s['mean_lead_time_hours_earliest'] or 0):>9.1f}"
                  f"{(s['median_lead_time_hours_earliest'] or 0):>8.1f}"
                  f"{(s['mean_lead_time_hours_contiguous'] or 0):>10.1f}"
                  f"{100*(s['time_in_warning_rate'] or 0):>9.1f}%"
                  f"{100*(s['false_alarm_day_rate'] or 0):>8.1f}%")

    show("any_hazard", "PRIMARY: all systems vs ANY-HAZARD onsets")
    show("flood", "SECONDARY: vs FLOOD onsets")
    show("landslide", "SECONDARY: vs LANDSLIDE onsets")

    print("\n  Reading these numbers:")
    print("   * 'mean h'/'med h' use the EARLIEST-warning variant and are close to")
    print(f"     saturation -- the {MAX_LOOKBACK_DAYS}-day window caps them at "
          f"{MAX_LOOKBACK_DAYS*HOURS_PER_DAY} h and the always-warn control already")
    print("     reaches most of that. Do NOT headline this column.")
    print("   * 'contig h' (unbroken alert run ending at onset) is the discriminating")
    print("     lead-time measure and the one an operator would actually experience.")
    print("   * 'in-warn%' is the cost: fraction of location-days spent under warning.")
    print("\n  Interpretation guard: rules R1/R2 reproduce the two baselines, so")
    print("  'tcdl_coupled' cannot detect fewer events or warn later than either")
    print("  baseline -- that dominance is BY CONSTRUCTION, not evidence. The")
    print("  'tcdl_coupling_rules_only' row shows what rules R3-R7 achieve alone,")
    print("  and the false-alarm column is the cost that must be weighed against")
    print("  any lead-time gain.")

    # ---- 6. Outputs --------------------------------------------------
    banner("6. SAVING OUTPUTS", "-")
    ts_cols = (["date", "latitude", "longitude", "location_id", "split",
                "rainfall_1d_mm", "soil_moisture",
                "flood_probability", "landslide_probability", "coupled_probability",
                "flood_prob_ma", "landslide_prob_ma", "coupled_prob_ma",
                "rainfall_ma", "soil_moisture_ma",
                "flood_prob_rate", "landslide_prob_rate", "coupled_prob_rate",
                "rainfall_rate", "soil_moisture_rate",
                "flood_trend_up", "landslide_trend_up", "rainfall_trend_up",
                "soil_trend_up"]
               + RULE_IDS
               + ["tcdl_warning", "tcdl_coupling_only_warning", "tcdl_triggered_rules",
                  "tcdl_n_rules", "warning_timestamp",
                  "flood_only_warning", "landslide_only_warning",
                  "flood", "landslide"])
    ts = df[ts_cols].sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
    ts["date"] = ts["date"].dt.strftime("%Y-%m-%d")
    ts.to_csv(TIMESERIES_PATH, index=False)
    print(f"  Combined time series      -> {TIMESERIES_PATH}  ({len(ts)} rows)")

    warn_out = eval_df.loc[eval_df["tcdl_warning"] == 1,
                           ["date", "latitude", "longitude", "location_id",
                            "warning_timestamp", "flood_probability",
                            "landslide_probability", "coupled_prob_ma",
                            "tcdl_triggered_rules", "tcdl_n_rules",
                            "flood_only_warning", "landslide_only_warning",
                            "flood", "landslide"]].copy()
    warn_out["date"] = warn_out["date"].dt.strftime("%Y-%m-%d")
    warn_out.to_csv(WARNINGS_PATH, index=False)
    print(f"  TCDL warnings (test)      -> {WARNINGS_PATH}  ({len(warn_out)} rows)")

    lead_df.to_csv(LEAD_TIME_PATH, index=False)
    print(f"  Lead-time table           -> {LEAD_TIME_PATH}  ({len(lead_df)} rows)")

    example_loc = plot_example_timeline(eval_df, events, thresholds, TIMELINE_PLOT_PATH)

    rule_counts = {rid: int(eval_df[rid].sum()) for rid in RULE_IDS}
    combo_counts = (eval_df.loc[eval_df["tcdl_warning"] == 1, "tcdl_triggered_rules"]
                    .value_counts().head(15).to_dict())

    payload = {
        "_warning": ("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS. "
                     "These numbers describe a simulated dataset."),
        "_scope_note": ("TCDL V1.0 is a PROPOSED rule-based coupling mechanism. "
                        "This file measures whether it helps; it does not claim "
                        "that it maximises lead time."),
        "_structural_dominance_note": (
            "Rules R1/R2 replicate the two single-model baselines exactly, so the "
            "coupled warning set is a superset of their union and cannot warn later "
            "or detect fewer events. That is a construction artefact. The "
            "'tcdl_coupling_rules_only' system isolates the contribution of rules "
            "R3-R7, and false-alarm rates are reported alongside every lead time."),
        "version": "1.0",
        "layer_type": "rule-based, no training, no fitted parameters",
        "inputs": {
            "master_dataset": DATA_PATH.name,
            "flood_model": FLOOD_MODEL_PATH.name,
            "landslide_model": LANDSLIDE_MODEL_PATH.name,
            "models_modified": False,
        },
        "evaluation_period": {
            "split": EVAL_SPLIT,
            "date_range": [str(eval_df["date"].min().date()),
                           str(eval_df["date"].max().date())],
            "n_rows": int(len(eval_df)),
            "n_locations": int(eval_df["location_id"].nunique()),
            "note": ("Earlier rows are scored only to warm up rolling windows; "
                     "they are never evaluated."),
        },
        "parameters": {
            "baseline_threshold": BASELINE_THRESHOLD,
            "smooth_window_days": SMOOTH_WINDOW,
            "rate_window_days": RATE_WINDOW,
            "sustain_days": SUSTAIN_DAYS,
            "max_lookback_days": MAX_LOOKBACK_DAYS,
            "threshold_policy": ("quantiles of each signal over the TRAINING period "
                                 "only; absolute values resolved at runtime"),
            "threshold_quantiles": {k: {"signal": v[0], "quantile": v[1]}
                                    for k, v in THRESHOLD_QUANTILES.items()},
            "resolved_thresholds": {k: round(v, 6) for k, v in thresholds.items()},
        },
        "rules": rule_descriptions,
        "rule_classification": {"baseline_replicating": BASELINE_RULES,
                                "coupling": COUPLING_RULES},
        "coupled_probability_definition": (
            "noisy-OR: 1 - (1 - p_flood)(1 - p_landslide). Assumes conditional "
            "independence, which is only an approximation since both hazards share "
            "rainfall drivers. Used for ranking, not as a calibrated joint probability."),
        "rule_firing_counts": rule_counts,
        "top_rule_combinations": combo_counts,
        "warning_counts": {
            "tcdl_coupled": int(eval_df["tcdl_warning"].sum()),
            "tcdl_coupling_rules_only": int(eval_df["tcdl_coupling_only_warning"].sum()),
            "flood_only_baseline": int(eval_df["flood_only_warning"].sum()),
            "landslide_only_baseline": int(eval_df["landslide_only_warning"].sum()),
            "evaluation_rows": int(len(eval_df)),
        },
        "event_source": event_source,
        "lead_time_definition": {
            "formula": "lead_time = event_onset_time - warning_time",
            "unit": "hours",
            "earliest_variant": "earliest warning within the lookback window",
            "contiguous_variant": "length of the unbroken warning run ending at onset",
            "zero_lead_meaning": ("a warning on the onset day itself is a nowcast, "
                                  "not a forecast: it uses that day's own rainfall"),
        },
        "lead_time_note": (
            f"MISSING DATA: the project has no real event onset timestamps and the "
            f"master dataset has no time-of-day component. Lead time is quantised to "
            f"{HOURS_PER_DAY} h and is an UPPER BOUND -- the true value lies within "
            f"24 h below each reported figure. Supply "
            f"ml/{EVENT_TIMESTAMPS_PATH.name} to obtain hour-resolution results."),
        "results": summary,
        "example_plot_location": example_loc,
        "outputs": {
            "timeseries": TIMESERIES_PATH.name,
            "warnings": WARNINGS_PATH.name,
            "lead_time": LEAD_TIME_PATH.name,
            "timeline_plot": TIMELINE_PLOT_PATH.name,
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results summary           -> {RESULTS_PATH}")

    banner("TCDL V1.0 COMPLETE")
    a = summary["any_hazard"]
    print("  Any-hazard onsets, evaluation period:")
    for name, _ in systems:
        s = a[name]
        print(f"    {name:<26} detection {100*s['detection_rate']:5.1f}%  "
              f"contiguous lead {(s['mean_lead_time_hours_contiguous'] or 0):7.1f} h  "
              f"time-in-warning {100*(s['time_in_warning_rate'] or 0):5.1f}%  "
              f"false-alarm days {s['n_false_alarm_days']:4d}")
    print("\n  MISSING: real event timestamps (hour resolution). Lead times are")
    print(f"  quantised to {HOURS_PER_DAY} h upper bounds until those are supplied.")
    print("  SYNTHETIC DATA -- not evidence of real-world warning performance.")


if __name__ == "__main__":
    main()
