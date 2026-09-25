"""
Flood XGBoost Model -- Version 1.0
=================================================================
Independent flood baseline for the coupled flood-landslide early
warning research project.

    Master Dataset -> Flood Features -> Chronological Split
                   -> XGBoost -> Flood Probability -> Flood / No-Flood

Scope of V1.0 (deliberately limited):
  * ONLY the independent flood model. No landslide model, no Temporal
    Coupled Decision Layer, no warning rules, no lead-time analysis,
    no API/frontend, no SHAP, no threshold optimisation.
  * The `landslide` column is never read as a feature, so this baseline
    stays genuinely independent of the future landslide model. That
    independence is what makes the later
    (independent flood | independent landslide | coupled system)
    comparison meaningful.

Datasets (the model, features, split and metrics are identical for both):
  real      (default) REAL Kerala district-day Master Dataset built by
            scripts/data/13_build_master_dataset.py. 14 districts x
            daily 2012-01-30..2024-12-31. `river_level_m` has genuine gaps
            (no gauge / no reading) that stay NaN and go to XGBoost's native
            missing-value handling. `district` is an identifier only.
  synthetic the original V1.0 run on SYNTHETIC / DUMMY DATA. Its metrics
            describe a simulator and must not be reported as real-world
            flood-prediction performance. Reproduces the committed synthetic/
            artifacts byte for byte.

Usage
-----
    python scripts/models/flood_xgboost.py                      # real data -> artifacts/
    python scripts/models/flood_xgboost.py --dataset synthetic  # V1.0 synthetic -> synthetic/
    (--output-dir DIR writes the artifacts somewhere else)

Reads  : real_master_dataset.csv                     or
         synthetic_master_dataset.csv                (read-only; never modified)
Writes : <output dir>/flood_xgboost_model.json
         <output dir>/flood_features.json
         <output dir>/flood_model_results.json
         <output dir>/flood_test_predictions.csv
         <output dir>/flood_feature_importance.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

# =====================================================================
# Configuration
# =====================================================================
RANDOM_STATE = 42

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]  # scripts/<group>/ -> project root

# --- Dataset selection -------------------------------------------------
# Only the input file, the artifact directory and the wording of the
# data-status notices depend on the dataset; the pipeline does not.
# The synthetic artifacts in synthetic/ are what the backend, TCDL and SHAP read,
# so real-data artifacts go to artifacts/ and never overwrite them.
DATASETS = {
    "real": dict(
        data_path=PROJECT_ROOT / "dataset" / "real_master_dataset.csv",
        output_dir=PROJECT_ROOT / "artifacts" / "models",
        results_dir=PROJECT_ROOT / "artifacts" / "results",
        banner="REAL DATA -- Kerala district-day Master Dataset (see REAL_DATA_DICTIONARY.md)",
        warning=("REAL DATA -- Kerala district-day Master Dataset (14 districts, "
                 "2012-01-30..2024-12-31). Labels are IMD-reported damaging flood / "
                 "heavy-rain district-days (0 = not reported, not proven absence); "
                 "river_level_m is missing for 3 districts and before mid-2015 for 7; "
                 "static features take one value per district. Read the limitations in "
                 "documents/master_dataset/REAL_DATA_DICTIONARY.md before quoting metrics."),
        plot_note="REAL DATA -- Kerala district-day Master Dataset",
        location_note=("metadata only; excluded to prevent the model memorising "
                       "the 14 district points"),
        closing_note=("REAL DATA -- district-level labels and static features; see "
                      "REAL_DATA_DICTIONARY.md before interpreting these numbers."),
    ),
    "synthetic": dict(
        data_path=PROJECT_ROOT / "synthetic" / "dataset" / "synthetic_master_dataset.csv",
        output_dir=PROJECT_ROOT / "synthetic" / "models",
        results_dir=PROJECT_ROOT / "synthetic" / "results",
        banner="SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS",
        warning=("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS. "
                 "These metrics describe a simulated dataset and must not be "
                 "reported as real-world flood prediction performance."),
        plot_note="SYNTHETIC DATA -- NOT FOR RESEARCH RESULTS",
        location_note=("metadata only; excluded to prevent the model "
                       "memorising the 9 synthetic locations"),
        closing_note=("SYNTHETIC DATA -- these numbers characterise the pipeline,\n"
                      "  not real-world flood predictability."),
    ),
}
DEFAULT_DATASET = "real"


def configure(dataset: str = DEFAULT_DATASET, output_dir: Path | None = None) -> None:
    """Bind the module-level paths and notices to one dataset."""
    global DATASET, DATA_PATH, MODEL_PATH, FEATURES_PATH, RESULTS_PATH
    global PREDICTIONS_PATH, IMPORTANCE_PLOT_PATH
    DATASET = dict(DATASETS[dataset], name=dataset)
    # --output-dir keeps every artifact in one folder; by default the model files
    # and the evaluation outputs go to separate folders.
    out = Path(output_dir) if output_dir else DATASET["output_dir"]
    res = Path(output_dir) if output_dir else DATASET["results_dir"]
    DATA_PATH = DATASET["data_path"]
    MODEL_PATH = out / "flood_xgboost_model.json"
    FEATURES_PATH = out / "flood_features.json"
    RESULTS_PATH = res / "flood_model_results.json"
    PREDICTIONS_PATH = res / "flood_test_predictions.csv"
    IMPORTANCE_PLOT_PATH = res / "flood_feature_importance.png"


configure()

# --- Feature definition ------------------------------------------------
# Exactly the 9 common environmental + 3 flood-specific features.
COMMON_FEATURES = [
    "rainfall_1d_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_14d_mm",
    "rainfall_30d_mm",
    "soil_moisture",
    "temperature_c",
    "humidity_percent",
    "elevation_m",
]
FLOOD_SPECIFIC_FEATURES = [
    "river_level_m",
    "distance_to_river_km",
    "drainage_density",
]
FLOOD_FEATURES = COMMON_FEATURES + FLOOD_SPECIFIC_FEATURES
TARGET = "flood"

# Columns kept alongside the model matrix for splitting / reporting /
# future work, but NEVER fed to the model.
#   date          -> chronological split key only
#   latitude/longitude -> carried through to the prediction table so
#                    temporal-spatial trends can be studied later. They
#                    are deliberately NOT features: with only 9 synthetic
#                    locations they would let the model memorise cell
#                    identity instead of learning environmental drivers.
META_COLUMNS = ["date", "latitude", "longitude"]

# Identifier columns of the real dataset: carried into the prediction
# table for mapping, never a feature (same rationale as latitude/longitude).
# Absent from the synthetic file, in which case nothing changes.
IDENTIFIER_COLUMNS = ["district"]

# Columns that must never reach the flood model.
# `landslide` is the other hazard's target: using it would leak the
# coupled signal into the "independent" baseline.
FORBIDDEN_FEATURES = ["landslide"] + IDENTIFIER_COLUMNS

# --- Split configuration ----------------------------------------------
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
# test fraction is the remainder (~0.15)

# --- Decision threshold ------------------------------------------------
# Fixed at 0.50 for V1.0. Threshold optimisation is explicitly deferred.
DECISION_THRESHOLD = 0.50

# --- Class imbalance ---------------------------------------------------
# scale_pos_weight is computed from the TRAINING SET ONLY (see
# compute_scale_pos_weight). Set to False to train unweighted.
USE_SCALE_POS_WEIGHT = True

# --- Model hyperparameters --------------------------------------------
# Sensible defaults for a baseline. NOT tuned -- V1.0 is about getting
# the pipeline right, not about squeezing out metrics.
XGB_PARAMS = dict(
    objective="binary:logistic",   # binary classification w/ probability output
    eval_metric="aucpr",           # PR-AUC: appropriate for the minority class
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=4,
)
EARLY_STOPPING_ROUNDS = 50


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed every RNG this pipeline can touch."""
    random.seed(seed)
    np.random.seed(seed)


def banner(text: str, char: str = "=") -> None:
    print("\n" + char * 70)
    print(text)
    print(char * 70)


# =====================================================================
# 1. Data loading and validation
# =====================================================================
def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the master dataset read-only and parse dates."""
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {path}. "
            "This pipeline never generates data -- it expects the existing CSV."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    print("Dataset loaded successfully")
    print(f"  Path            : {path}")
    print(f"  Number of rows  : {len(df)}")
    print(f"  Number of columns: {df.shape[1]}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Fail loudly if the schema does not match what the model expects."""
    required = set(FLOOD_FEATURES + [TARGET] + META_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master dataset is missing required columns: {missing}")

    leaked = [c for c in FORBIDDEN_FEATURES if c in FLOOD_FEATURES]
    if leaked:
        raise ValueError(f"Forbidden column(s) present in the feature list: {leaked}")

    bad_target = set(pd.unique(df[TARGET])) - {0, 1}
    if bad_target:
        raise ValueError(f"Target '{TARGET}' must be binary; found {bad_target}")

    print("\nColumn validation passed")
    print(f"  Features ({len(FLOOD_FEATURES)}): {FLOOD_FEATURES}")
    print(f"  Target: '{TARGET}'")
    print(f"  Excluded from features: {FORBIDDEN_FEATURES} "
          f"(independence of the flood baseline; identifiers)")
    print(f"  Carried as metadata only: "
          f"{META_COLUMNS + [c for c in IDENTIFIER_COLUMNS if c in df.columns]}")


def report_missing_values(df: pd.DataFrame) -> dict:
    """
    Report missing values in the model columns.

    Handling policy (documented, not silent):
      * Rows are NEVER dropped -- doing so would tear holes in a
        chronological series and corrupt the temporal structure.
      * XGBoost handles NaN natively by learning a default split
        direction, so genuine gaps are passed through to the model
        rather than imputed with a fitted statistic. This also avoids
        leakage: no parameter is estimated from the data.
    """
    cols = FLOOD_FEATURES + [TARGET] + META_COLUMNS
    counts = df[cols].isna().sum()
    missing = counts[counts > 0]

    print("\nMissing value check")
    if missing.empty:
        print("  No missing values in any model column.")
    else:
        print("  Missing values found (rows retained, NaN passed to XGBoost):")
        for col, n in missing.items():
            print(f"    {col:<24} {n:6d}  ({100 * n / len(df):.2f}%)")
        if df[TARGET].isna().any():
            raise ValueError(
                f"Target '{TARGET}' contains missing values -- cannot train. "
                "Resolve this in the dataset before running the pipeline."
            )
    return {k: int(v) for k, v in counts.items()}


def report_missing_by_split(splits: dict, cols: list[str]) -> dict:
    """
    Missing values per feature and split, for columns that have any.

    Makes the NaN policy explicit on real data: the gaps in e.g.
    `river_level_m` (district without a gauge, gauge not yet installed,
    no reading that day) are passed unchanged to XGBoost, which learns a
    default branch for missing values from the TRAINING split only.
    Nothing is imputed. Returns {} when no model column has a gap.
    """
    out = {}
    for col in cols:
        per = {name: int(part[col].isna().sum()) for name, part in splits.items()}
        if any(per.values()):
            out[col] = {name: {"n_missing": n, "n_rows": int(len(splits[name])),
                               "pct_missing": round(100 * n / len(splits[name]), 2)}
                        for name, n in per.items()}
    if out:
        print("\nMissing values per split (kept as NaN -> XGBoost native handling)")
        for col, per in out.items():
            print(f"  {col:<24} " + "  ".join(
                f"{name}: {v['n_missing']} ({v['pct_missing']:.1f}%)" for name, v in per.items()))
    return out


def report_class_distribution(y: pd.Series, label: str) -> dict:
    """Print and return the flood / no-flood balance."""
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    ratio = (neg / pos) if pos else float("inf")
    print(f"  {label:<12} n={n:5d} | flood=0: {neg:5d} ({100*neg/n:5.2f}%) "
          f"| flood=1: {pos:4d} ({100*pos/n:5.2f}%) | neg/pos = {ratio:.2f}")
    return dict(n_rows=n, n_negative=neg, n_positive=pos,
                positive_rate=round(pos / n, 6), neg_pos_ratio=round(ratio, 4))


# =====================================================================
# 2. Chronological split
# =====================================================================
def chronological_split(df: pd.DataFrame):
    """
    Split by DATE BOUNDARY, not by row index.

    The master dataset holds several locations per date (9 in the current
    synthetic file). Slicing on row position would place some locations
    of a given day in train and the rest in validation/test -- same-day
    leakage across an almost-identical weather state. Cutting on unique
    dates keeps every observation of a day in exactly one split, so no
    future information can reach the training set.
    """
    df = df.sort_values(["date", "latitude", "longitude"],
                        kind="mergesort").reset_index(drop=True)

    unique_dates = np.array(sorted(df["date"].unique()))
    n_dates = len(unique_dates)
    train_end_i = int(round(n_dates * TRAIN_FRACTION)) - 1
    val_end_i = int(round(n_dates * (TRAIN_FRACTION + VALIDATION_FRACTION))) - 1

    train_end = unique_dates[train_end_i]
    val_end = unique_dates[val_end_i]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]

    # Hard guarantees -- cheap to check, catastrophic to get wrong.
    assert len(train) + len(val) + len(test) == len(df), "split lost rows"
    assert train["date"].max() < val["date"].min(), "train/val overlap in time"
    assert val["date"].max() < test["date"].min(), "val/test overlap in time"
    assert not (set(train["date"]) & set(val["date"])), "date spans train and val"
    assert not (set(val["date"]) & set(test["date"])), "date spans val and test"

    print("\nChronological split (cut on date boundaries, no shuffling)")
    print(f"  Unique dates: {n_dates}")
    for name, part in [("Training", train), ("Validation", val), ("Testing", test)]:
        print(f"  {name+' period':<19}: {part['date'].min().date()} -> "
              f"{part['date'].max().date()}  "
              f"({part['date'].nunique()} dates, {len(part)} rows, "
              f"{100*len(part)/len(df):.1f}%)")
    return train, val, test


# =====================================================================
# 3. Model
# =====================================================================
def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    negative / positive, computed on the TRAINING SET ONLY.

    Using the full dataset here would leak validation and test class
    balance into a training hyperparameter.
    """
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    return float(neg / pos) if pos else 1.0


def build_model(scale_pos_weight: float | None) -> xgb.XGBClassifier:
    params = dict(XGB_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(
        **params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)


# =====================================================================
# 4. Evaluation
# =====================================================================
def evaluate(model, X, y, split_name: str, threshold: float = DECISION_THRESHOLD) -> dict:
    """Full metric set for one split. Probability is P(flood = 1)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) > 1

    metrics = dict(
        split=split_name,
        n_rows=int(len(y)),
        n_actual_positives=int(y.sum()),
        n_predicted_positives=int(pred.sum()),
        threshold=threshold,
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, proba)) if both_classes else None,
        pr_auc=float(average_precision_score(y, proba)) if both_classes else None,
        brier_score=float(brier_score_loss(y, proba)),
        mean_predicted_probability=float(proba.mean()),
        observed_positive_rate=float(y.mean()),
        confusion_matrix=dict(
            true_negatives=int(tn), false_positives=int(fp),
            false_negatives=int(fn), true_positives=int(tp)),
    )

    print(f"\n  --- {split_name} ---")
    print(f"    Accuracy : {metrics['accuracy']:.4f}     "
          f"Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}     "
          f"F1-score : {metrics['f1']:.4f}")
    roc = metrics["roc_auc"]
    pr = metrics["pr_auc"]
    print(f"    ROC-AUC  : {roc:.4f}     PR-AUC   : {pr:.4f}"
          if roc is not None else "    ROC-AUC  : n/a (single class)")
    print(f"    Confusion matrix (threshold = {threshold:.2f}):")
    print(f"                    predicted 0   predicted 1")
    print(f"      actual 0      {tn:11d}   {fp:11d}")
    print(f"      actual 1      {fn:11d}   {tp:11d}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"    Missed floods (FN): {fn} of {int(y.sum())} actual flood rows")
    print(f"    Calibration: mean predicted p = "
          f"{metrics['mean_predicted_probability']:.4f} vs observed rate "
          f"{metrics['observed_positive_rate']:.4f} "
          f"(Brier {metrics['brier_score']:.4f})")
    return metrics


# =====================================================================
# 5. Feature importance
# =====================================================================
def feature_importance_table(model, features: list[str]) -> pd.DataFrame:
    """Gain-based importance, highest first, one row per model feature."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    tbl = pd.DataFrame({
        "feature": features,
        "importance_gain": [gain.get(f, 0.0) for f in features],
        "split_count": [weight.get(f, 0) for f in features],
    })
    total = tbl["importance_gain"].sum()
    tbl["importance_normalised"] = tbl["importance_gain"] / total if total else 0.0
    return tbl.sort_values("importance_gain", ascending=False).reset_index(drop=True)


def plot_feature_importance(tbl: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    t = tbl.sort_values("importance_gain")          # ascending for barh
    ax.barh(t["feature"], t["importance_gain"], color="#2c7fb8")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Flood XGBoost V1.0 -- Feature Importance\n"
                 + DATASET["plot_note"], fontsize=11)
    for y_i, v in enumerate(t["importance_gain"]):
        ax.text(v, y_i, f" {v:.1f}", va="center", fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Feature importance plot -> {path}")


# =====================================================================
# Main
# =====================================================================
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Flood XGBoost V1.0 (independent baseline)")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET,
                    help=f"master dataset to train on (default: {DEFAULT_DATASET})")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="artifact directory (default: artifacts/ subfolders for real, synthetic/ subfolders for synthetic)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    configure(args.dataset, args.output_dir)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_seeds()

    banner("FLOOD XGBOOST MODEL -- VERSION 1.0")
    print(DATASET["banner"])
    print("Independent flood baseline. No landslide model, no TCDL.")
    print(f"Dataset: {DATASET['name']} -> artifacts in {MODEL_PATH.parent}")

    # ---- Load / validate / inspect ----------------------------------
    banner("1. DATA LOADING AND VALIDATION", "-")
    df = load_dataset(DATA_PATH)
    validate_columns(df)
    missing_report = report_missing_values(df)

    print("\nClass distribution (full dataset)")
    overall_dist = report_class_distribution(df[TARGET], "overall")

    # ---- Split -------------------------------------------------------
    banner("2. CHRONOLOGICAL SPLIT", "-")
    train_df, val_df, test_df = chronological_split(df)

    print("\nClass distribution per split")
    dist = {
        "overall": overall_dist,
        "train": report_class_distribution(train_df[TARGET], "train"),
        "validation": report_class_distribution(val_df[TARGET], "validation"),
        "test": report_class_distribution(test_df[TARGET], "test"),
    }
    missing_by_split = report_missing_by_split(
        {"train": train_df, "validation": val_df, "test": test_df}, FLOOD_FEATURES)

    X_train, y_train = train_df[FLOOD_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FLOOD_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[FLOOD_FEATURES], test_df[TARGET]

    # ---- Preprocessing note -----------------------------------------
    banner("3. PREPROCESSING", "-")
    print("  No fitted transformation is applied.")
    print("  Rationale: XGBoost is a tree ensemble -- it is invariant to")
    print("  monotonic feature scaling, so standardisation would add a")
    print("  fitted parameter set (a leakage surface) for zero benefit.")
    print("  Missing values, if any, are passed to XGBoost's native NaN")
    print("  handling rather than imputed. No rows are dropped.")
    print("  => Every learned quantity comes from the training split only.")

    # ---- Class imbalance --------------------------------------------
    banner("4. CLASS IMBALANCE HANDLING", "-")
    spw = compute_scale_pos_weight(y_train)
    print(f"  Training neg/pos ratio: {spw:.2f}")
    if USE_SCALE_POS_WEIGHT:
        print(f"  Applying scale_pos_weight = {spw:.4f} (computed on TRAIN only).")
        print(f"  Justified: ~{spw:.0f}:1 imbalance, and in an early-warning context a")
        print("  false negative (missed flood) costs more than a false alarm.")
        print("  NOTE: this re-weighting deliberately inflates predicted")
        print("  probabilities relative to the true event rate. Recall improves;")
        print("  calibration degrades. Brier score and mean-predicted-vs-observed")
        print("  rate are reported below so the effect is visible. Probability")
        print("  calibration should be revisited before TCDL integration.")
    else:
        spw = None
        print("  scale_pos_weight disabled; training unweighted.")

    # ---- Train -------------------------------------------------------
    banner("5. MODEL TRAINING", "-")
    model = build_model(spw)
    print("  Model configuration:")
    for k, v in sorted(model.get_params().items()):
        if v is not None and k not in ("missing",):
            print(f"    {k:<22} {v}")

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              verbose=False)
    best_iter = int(model.best_iteration)
    print(f"\n  Trained. Early stopping selected iteration {best_iter} "
          f"(of max {XGB_PARAMS['n_estimators']}), "
          f"chosen on the VALIDATION split by {XGB_PARAMS['eval_metric']}.")
    # The booster must have seen exactly the declared features -- no
    # identifier (district, lat/lon, date) and no target slipped in.
    assert list(model.get_booster().feature_names) == FLOOD_FEATURES, \
        "trained feature set differs from FLOOD_FEATURES"

    # ---- Evaluate ----------------------------------------------------
    banner("6. EVALUATION", "-")
    print(f"Decision threshold fixed at {DECISION_THRESHOLD:.2f} "
          f"(threshold optimisation deferred to a later version).")
    results = {
        "train": evaluate(model, X_train, y_train, "Training"),
        "validation": evaluate(model, X_val, y_val, "Validation"),
        "test": evaluate(model, X_test, y_test, "Testing (final unseen)"),
    }

    # ---- Feature importance -----------------------------------------
    banner("7. FEATURE IMPORTANCE", "-")
    imp = feature_importance_table(model, FLOOD_FEATURES)
    print(imp[["feature", "importance_gain", "importance_normalised",
               "split_count"]].to_string(index=False,
                                         float_format=lambda v: f"{v:.4f}"))
    plot_feature_importance(imp, IMPORTANCE_PLOT_PATH)

    # ---- Test prediction table --------------------------------------
    banner("8. TEST-SET PROBABILITY TABLE", "-")
    test_proba = model.predict_proba(X_test)[:, 1]
    id_cols = [c for c in IDENTIFIER_COLUMNS if c in test_df.columns]
    pred_table = pd.DataFrame({
        "date": test_df["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "latitude": test_df["latitude"].to_numpy(),
        "longitude": test_df["longitude"].to_numpy(),
        **{c: test_df[c].to_numpy() for c in id_cols},       # mapping only
        "actual_flood": y_test.to_numpy().astype(int),
        "flood_probability": np.round(test_proba, 6),
        "flood_prediction": (test_proba >= DECISION_THRESHOLD).astype(int),
    }).sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
    pred_table.to_csv(PREDICTIONS_PATH, index=False)
    print(f"  {len(pred_table)} test rows -> {PREDICTIONS_PATH}")
    print("\n  Sample (highest predicted probability):")
    print(pred_table.sort_values("flood_probability", ascending=False)
          .head(5).to_string(index=False))

    # ---- Artifacts ---------------------------------------------------
    banner("9. SAVING ARTIFACTS", "-")
    # Save ONLY the trees up to the early-stopping best iteration.
    #
    # Why this matters: after early stopping the booster still holds every
    # tree that was built (best_iteration + EARLY_STOPPING_ROUNDS). The
    # sklearn wrapper quietly applies best_iteration when predicting, but a
    # plain `Booster.load_model()` does not -- it would use all trees and
    # produce different probabilities from the ones evaluated above.
    # Slicing makes the artifact self-contained, so any loader reproduces
    # the evaluated model exactly.
    full_booster = model.get_booster()
    best_booster = full_booster[: best_iter + 1]
    best_booster.save_model(str(MODEL_PATH))
    n_trees_saved = len(best_booster.get_dump())
    print(f"  Saved model path: {MODEL_PATH}")
    print(f"  Trees saved: {n_trees_saved} (sliced to best_iteration={best_iter}; "
          f"{len(full_booster.get_dump())} were trained before early stopping)")

    # Verify the artifact on disk reproduces the evaluated model exactly.
    reloaded = xgb.Booster()
    reloaded.load_model(str(MODEL_PATH))
    reload_proba = reloaded.predict(xgb.DMatrix(X_test))
    max_drift = float(np.abs(reload_proba - test_proba).max())
    if max_drift > 1e-6:
        raise RuntimeError(
            f"Saved model does not reproduce evaluated predictions "
            f"(max drift {max_drift:.6g}). Refusing to ship an inconsistent artifact.")
    print(f"  Reload check: saved model reproduces test probabilities "
          f"(max drift {max_drift:.2e})")

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "flood_xgboost",
            "version": "1.0",
            "target": TARGET,
            "feature_order": FLOOD_FEATURES,
            "n_trees": n_trees_saved,
            "best_iteration": best_iter,
            "common_features": COMMON_FEATURES,
            "flood_specific_features": FLOOD_SPECIFIC_FEATURES,
            "excluded_columns": {
                "landslide": "other hazard target -- keeps this baseline independent",
                "date": "chronological split key only",
                "latitude/longitude": DATASET["location_note"],
                **{c: "identifier / mapping only, never a feature" for c in id_cols},
                "land_cover/lithology": "categorical, reserved for the landslide model",
                "slope_degree/aspect_degree/curvature": "landslide-specific terrain",
            },
            "decision_threshold": DECISION_THRESHOLD,
            "note": "Feature order must be preserved for inference.",
        }, f, indent=2)
    print(f"  Saved feature list: {FEATURES_PATH}")

    payload = {
        "_warning": DATASET["warning"],
        "model": "flood_xgboost",
        "version": "1.0",
        "random_state": RANDOM_STATE,
        "dataset": {
            "path": str(DATA_PATH.name),
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "n_unique_dates": int(df["date"].nunique()),
            "n_locations": int(df[["latitude", "longitude"]].drop_duplicates().shape[0]),
            "missing_values_in_model_columns": missing_report,
        },
        "features": {
            "feature_names": FLOOD_FEATURES,
            "n_features": len(FLOOD_FEATURES),
            "common_features": COMMON_FEATURES,
            "flood_specific_features": FLOOD_SPECIFIC_FEATURES,
            "target": TARGET,
        },
        "split": {
            "method": "chronological, cut on unique-date boundaries (no shuffling)",
            "fractions_requested": {"train": TRAIN_FRACTION,
                                    "validation": VALIDATION_FRACTION,
                                    "test": round(1 - TRAIN_FRACTION - VALIDATION_FRACTION, 4)},
            "train_size": int(len(train_df)),
            "validation_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "train_date_range": [str(train_df["date"].min().date()),
                                 str(train_df["date"].max().date())],
            "validation_date_range": [str(val_df["date"].min().date()),
                                      str(val_df["date"].max().date())],
            "test_date_range": [str(test_df["date"].min().date()),
                                str(test_df["date"].max().date())],
        },
        "class_distribution": dist,
        "class_imbalance_handling": {
            "method": "scale_pos_weight" if USE_SCALE_POS_WEIGHT else "none",
            "scale_pos_weight": spw,
            "computed_on": "training split only",
        },
        "preprocessing": {
            "fitted_transformations": "none",
            "scaling": "not applied (tree model is scale-invariant)",
            "missing_value_policy": "no rows dropped; NaN passed to XGBoost native handling",
            # only written when a model column has gaps (real data)
            **({"missing_values_by_split": missing_by_split} if missing_by_split else {}),
        },
        "model_parameters": {k: v for k, v in model.get_params().items()
                             if isinstance(v, (int, float, str, bool, type(None)))},
        "best_iteration": best_iter,
        "decision_threshold": DECISION_THRESHOLD,
        "metrics": results,
        "feature_importance": imp.to_dict(orient="records"),
        "outputs": {
            "model": MODEL_PATH.name,
            "features": FEATURES_PATH.name,
            "test_predictions": PREDICTIONS_PATH.name,
            "feature_importance_plot": IMPORTANCE_PLOT_PATH.name,
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved results   : {RESULTS_PATH}")

    banner("FLOOD XGBOOST V1.0 COMPLETE")
    t = results["test"]
    print(f"  Final unseen test: accuracy {t['accuracy']:.4f} | "
          f"precision {t['precision']:.4f} | recall {t['recall']:.4f} | "
          f"F1 {t['f1']:.4f} | ROC-AUC {t['roc_auc']:.4f}")
    print(f"  Missed floods on test set: "
          f"{t['confusion_matrix']['false_negatives']} of {t['n_actual_positives']}")
    print("\n  " + DATASET["closing_note"])


if __name__ == "__main__":
    main()
