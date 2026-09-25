"""
SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS

Sanity checks for synthetic_master_dataset.csv:
  1. schema / constraints / nulls
  2. internal consistency of the rolling rainfall accumulations
  3. temporal (monsoon) structure
  4. a real chronological-split XGBoost run for both hazards
The AUCs printed here describe a SIMULATED dataset. They say nothing
about real-world flood or landslide predictability and must never be
reported as project results.
"""

import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "synthetic" / "dataset" / "synthetic_master_dataset.csv",
                 parse_dates=["date"])

COMMON = ["rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm",
          "rainfall_14d_mm", "rainfall_30d_mm", "soil_moisture",
          "temperature_c", "humidity_percent", "elevation_m"]
FLOOD_SPECIFIC = ["river_level_m", "distance_to_river_km", "drainage_density"]
LANDSLIDE_SPECIFIC = ["slope_degree", "aspect_degree", "curvature",
                      "land_cover", "lithology"]

print("=" * 62)
print("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS")
print("=" * 62)

# ---- 1. constraints -------------------------------------------------
checks = {
    "rainfall >= 0": (df[[c for c in df if c.startswith("rainfall")]] >= 0).all().all(),
    "soil_moisture in [0,1]": df.soil_moisture.between(0, 1).all(),
    "temperature in [-10,55]": df.temperature_c.between(-10, 55).all(),
    "humidity in [0,100]": df.humidity_percent.between(0, 100).all(),
    "elevation >= 0": (df.elevation_m >= 0).all(),
    "river_level >= 0": (df.river_level_m >= 0).all(),
    "distance_to_river >= 0": (df.distance_to_river_km >= 0).all(),
    "drainage_density >= 0": (df.drainage_density >= 0).all(),
    "slope in [0,90]": df.slope_degree.between(0, 90).all(),
    "aspect in [0,360]": df.aspect_degree.between(0, 360).all(),
    "curvature has both signs": (df.curvature > 0).any() and (df.curvature < 0).any(),
    "no nulls": not df.isna().any().any(),
    "dates non-decreasing": df.date.is_monotonic_increasing,
    "targets binary": set(df.flood.unique()) <= {0, 1} and set(df.landslide.unique()) <= {0, 1},
}
for k, v in checks.items():
    print(f"  [{'OK ' if v else 'FAIL'}] {k}")

# ---- 2. rolling-window consistency ---------------------------------
print("\nRainfall accumulation consistency (per location):")
worst = {}
for (la, lo), g in df.groupby(["latitude", "longitude"]):
    g = g.sort_values("date")
    for w, col in [(3, "rainfall_3d_mm"), (7, "rainfall_7d_mm"),
                   (14, "rainfall_14d_mm"), (30, "rainfall_30d_mm")]:
        ref = g.rainfall_1d_mm.rolling(w).sum()
        err = (g[col] - ref).abs().max()          # NaN over the first w-1 rows only
        worst[col] = max(worst.get(col, 0.0), np.nan_to_num(err))
for col, e in worst.items():
    print(f"  {col:<18} max |col - rolling_sum(1d)| within record = {e:.3f} mm")
print("  (first w-1 rows of each location use the 30-day pre-record burn-in,")
print("   so they are correct but not reproducible from this file alone)")

# ---- 3. temporal structure -----------------------------------------
print("\nMonthly mean rainfall_1d_mm and hazard rates:")
m = df.assign(month=df.date.dt.month).groupby("month").agg(
    rain=("rainfall_1d_mm", "mean"), flood=("flood", "mean"),
    landslide=("landslide", "mean"))
print(m.round(4).to_string())

print("\nRainfall autocorrelation (location 1, lags 1-5):")
g0 = df[df.latitude == df.latitude.unique()[0]].sort_values("date").rainfall_1d_mm
print("  " + "  ".join(f"lag{k}={g0.autocorr(k):.3f}" for k in range(1, 6)))

print("\nDaily rainfall regime mix (all rows):")
bins = [-0.01, 0.1, 10, 35, 75, 1e9]
labs = ["dry (0mm)", "light (<10)", "moderate (10-35)", "heavy (35-75)", "extreme (>75)"]
print(pd.cut(df.rainfall_1d_mm, bins, labels=labs).value_counts()
        .reindex(labs).to_frame("days").assign(
            pct=lambda x: (100 * x.days / len(df)).round(2)).to_string())

# ---- 4. XGBoost, chronological split --------------------------------
try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    d = df.copy()
    for c in ["land_cover", "lithology"]:
        d[c] = d[c].astype("category")

    cut1, cut2 = d.date.quantile(0.70), d.date.quantile(0.85)
    tr, va, te = d[d.date <= cut1], d[(d.date > cut1) & (d.date <= cut2)], d[d.date > cut2]
    print(f"\nChronological split  train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"  train {tr.date.min().date()}..{tr.date.max().date()} | "
          f"val {va.date.min().date()}..{va.date.max().date()} | "
          f"test {te.date.min().date()}..{te.date.max().date()}")

    for name, feats, target in [("FLOOD", COMMON + FLOOD_SPECIFIC, "flood"),
                                ("LANDSLIDE", COMMON + LANDSLIDE_SPECIFIC, "landslide")]:
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, eval_metric="auc",
            enable_categorical=True, tree_method="hist",
            early_stopping_rounds=40, random_state=0)
        model.fit(tr[feats], tr[target], eval_set=[(va[feats], va[target])], verbose=False)
        p = model.predict_proba(te[feats])[:, 1]
        print(f"\n{name} model  (n_features={len(feats)}, best_iter={model.best_iteration})")
        print(f"  test ROC-AUC = {roc_auc_score(te[target], p):.4f}"
              f"   PR-AUC = {average_precision_score(te[target], p):.4f}"
              f"   positives = {int(te[target].sum())}/{len(te)}")
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        print("  top features: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.head(6).items()))
except ImportError:
    print("\n[xgboost / scikit-learn not installed -- model check skipped]")

print("\n" + "=" * 62)
print("Reminder: simulated data. Not evidence of real-world skill.")
print("=" * 62)
