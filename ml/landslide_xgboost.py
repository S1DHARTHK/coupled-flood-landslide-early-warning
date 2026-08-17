"""
Landslide XGBoost Model -- Version 1.0
=================================================================
Independent landslide baseline for the coupled flood-landslide early
warning research project. Methodologically parallel to
`ml/flood_xgboost.py` (same seed, same split boundaries, same metric
set, same artifact conventions) so the two baselines can be compared
fairly.

    Master Dataset -> Landslide Features -> Chronological Split
                   -> XGBoost -> Landslide Probability
                   -> Landslide / No-Landslide

Scope of V1.0 (deliberately limited):
  * ONLY the independent landslide model. No Temporal Coupled Decision
    Layer, no coupled prediction, no warning rules, no lead-time
    calculation, no threshold optimisation, no probability calibration,
    no SHAP, no API/frontend.
  * The `flood` column is never read as a feature, and nothing produced
    by `flood_xgboost.py` is used as an input. That independence is what
    makes the later
    (independent flood | independent landslide | coupled system)
    comparison meaningful.

Difference from the flood model (not a blind copy):
  * The landslide feature set contains two CATEGORICAL columns
    (`land_cover`, `lithology`); the flood feature set was fully
    numeric. See `prepare_categoricals` for the handling and its
    leakage argument.
  * `scale_pos_weight` is recomputed from this target's own training
    distribution -- the landslide imbalance is NOT the flood imbalance.
  * Flood-specific logic (river level, river proximity, drainage
    density) is intentionally absent.

*** IMPORTANT ***
The input dataset is SYNTHETIC / DUMMY DATA. Every metric produced by
this script describes a simulator, not the physical world, and must not
be reported as real-world landslide-prediction performance. Feature
importances below are properties of the generator, not scientific
evidence about landslide processes.

Usage
-----
    python ml/landslide_xgboost.py

Reads  : synthetic_master_dataset.csv   (read-only; never modified)
Writes : ml/landslide_xgboost_model.json
         ml/landslide_features.json
         ml/landslide_model_results.json
         ml/landslide_test_predictions.csv
         ml/landslide_feature_importance.png
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

# =====================================================================
# Configuration
# =====================================================================
RANDOM_STATE = 42                   # identical to flood V1.0

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_PATH = PROJECT_ROOT / "synthetic_master_dataset.csv"

MODEL_PATH = HERE / "landslide_xgboost_model.json"
FEATURES_PATH = HERE / "landslide_features.json"
RESULTS_PATH = HERE / "landslide_model_results.json"
PREDICTIONS_PATH = HERE / "landslide_test_predictions.csv"
IMPORTANCE_PLOT_PATH = HERE / "landslide_feature_importance.png"

# Read only to CHECK that both baselines share split boundaries.
# Nothing from this file enters training, features, or labels.
FLOOD_RESULTS_PATH = HERE / "flood_model_results.json"

# --- Feature definition ------------------------------------------------
# The 9 common environmental features (identical to the flood model)
# plus the 5 landslide-specific terrain/surface features.
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
LANDSLIDE_SPECIFIC_FEATURES = [
    "slope_degree",
    "aspect_degree",
    "curvature",
    "land_cover",       # categorical
    "lithology",        # categorical
]
LANDSLIDE_FEATURES = COMMON_FEATURES + LANDSLIDE_SPECIFIC_FEATURES
CATEGORICAL_FEATURES = ["land_cover", "lithology"]
TARGET = "landslide"

# Columns kept alongside the model matrix but NEVER fed to the model.
#   date          -> chronological split key only
#   latitude/longitude -> carried into the prediction table so temporal
#                    and spatial trends can be studied later, and so the
#                    landslide table can be joined to the flood table
#                    inside the future TCDL. Deliberately NOT features:
#                    with only 9 synthetic locations they would let the
#                    model memorise cell identity instead of learning
#                    environmental drivers (same rationale as flood V1.0).
META_COLUMNS = ["date", "latitude", "longitude"]

# Columns that must never reach the landslide model.
# `flood` is the other hazard's target; using it -- or any flood model
# output -- would collapse the independence this baseline exists to
# establish. Flood-specific predictors are excluded too: they belong to
# the flood branch of the architecture.
FORBIDDEN_FEATURES = [
    "flood", "flood_probability", "flood_prediction",
    "river_level_m", "distance_to_river_km", "drainage_density",
]

# --- Categorical schema ------------------------------------------------
# FIXED category lists, in the code order documented in
# DATA_DICTIONARY.md section 4. Declaring them explicitly (rather than
# letting pandas infer them per split) matters for two reasons:
#   1. Leakage / correctness: an inferred category list is data
#      dependent. If a class were absent from one split, that split's
#      integer codes would silently shift and the model would read a
#      different feature than it was trained on.
#   2. Reproducibility: the same declaration is persisted to
#      landslide_features.json, so future inference encodes identically.
# These are NOMINAL identifiers, not an ordering -- XGBoost's native
# categorical support performs partition-based splits and does not treat
# the codes as ordered.
LAND_COVER_CATEGORIES = ["forest", "agriculture", "shrubland", "built_up", "barren"]
LITHOLOGY_CATEGORIES = ["granite_gneiss", "laterite", "schist",
                        "sandstone", "shale", "alluvium"]
CATEGORY_SCHEMA = {
    "land_cover": LAND_COVER_CATEGORIES,
    "lithology": LITHOLOGY_CATEGORIES,
}

# --- Split configuration ----------------------------------------------
# Same fractions and same boundary logic as flood V1.0. Because both
# models read the same master dataset, this reproduces the flood split
# exactly -- verified at runtime against flood_model_results.json.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# --- Decision threshold ------------------------------------------------
DECISION_THRESHOLD = 0.50           # fixed for V1.0; optimisation deferred

# --- Class imbalance ---------------------------------------------------
# Computed from THIS target's training split. The landslide imbalance is
# not assumed equal to the flood imbalance.
USE_SCALE_POS_WEIGHT = True

# --- Model hyperparameters --------------------------------------------
# Same baseline philosophy and same values as flood V1.0, so any
# performance difference between the two hazards reflects the data and
# not divergent tuning. NOT optimised.
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",           # PR-AUC: appropriate for the minority class
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,       # native categorical support (XGBoost >= 1.6)
    max_cat_to_onehot=1,           # prefer partition-based splits for categoricals
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
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the master dataset read-only and parse dates."""
    if not path.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {path}. "
            "This pipeline never generates data -- it expects the existing CSV."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    print("Dataset loaded successfully")
    print(f"  Path             : {path}")
    print(f"  Number of rows   : {len(df)}")
    print(f"  Number of columns: {df.shape[1]}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Fail loudly if the schema or the independence constraint is violated."""
    required = set(LANDSLIDE_FEATURES + [TARGET] + META_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master dataset is missing required columns: {missing}")

    leaked = [c for c in FORBIDDEN_FEATURES if c in LANDSLIDE_FEATURES]
    if leaked:
        raise ValueError(
            f"Forbidden column(s) present in the landslide feature list: {leaked}. "
            "The independent landslide baseline must not see flood information.")

    bad_target = set(pd.unique(df[TARGET])) - {0, 1}
    if bad_target:
        raise ValueError(f"Target '{TARGET}' must be binary; found {bad_target}")

    # The declared categorical schema must cover what is actually present.
    for col, cats in CATEGORY_SCHEMA.items():
        unexpected = sorted(set(df[col].dropna().unique()) - set(cats))
        if unexpected:
            raise ValueError(
                f"Column '{col}' contains values absent from the declared "
                f"schema: {unexpected}. Update CATEGORY_SCHEMA and "
                f"DATA_DICTIONARY.md together.")

    print("\nColumn validation passed")
    print(f"  Features ({len(LANDSLIDE_FEATURES)}): {LANDSLIDE_FEATURES}")
    print(f"  Target: '{TARGET}'")
    print(f"  Categorical features: {CATEGORICAL_FEATURES}")
    print(f"  Excluded from features: {FORBIDDEN_FEATURES}")
    print("    (independence of the landslide baseline -- no flood target,")
    print("     no flood probability/prediction, no flood-specific predictors)")
    print(f"  Carried as metadata only: {META_COLUMNS}")


def report_missing_values(df: pd.DataFrame) -> dict:
    """
    Report missing values in the model columns.

    Handling policy (documented, not silent), identical to flood V1.0:
      * Rows are NEVER dropped -- doing so would tear holes in a
        chronological series and corrupt the temporal structure.
      * XGBoost handles NaN natively by learning a default split
        direction, so genuine gaps pass through to the model rather than
        being imputed with a fitted statistic. No parameter is estimated
        from the data, so no leakage surface is created.
      * A missing CATEGORICAL value stays NaN; it is not folded into a
        synthetic "unknown" level, which would invent a category the
        data dictionary does not define.
    """
    cols = LANDSLIDE_FEATURES + [TARGET] + META_COLUMNS
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
                "Resolve this in the dataset before running the pipeline.")
    return {k: int(v) for k, v in counts.items()}


def report_class_distribution(y: pd.Series, label: str) -> dict:
    """Print and return the landslide / no-landslide balance."""
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    ratio = (neg / pos) if pos else float("inf")
    print(f"  {label:<12} n={n:5d} | landslide=0: {neg:5d} ({100*neg/n:5.2f}%) "
          f"| landslide=1: {pos:4d} ({100*pos/n:5.2f}%) | neg/pos = {ratio:.2f}")
    return dict(n_rows=n, n_negative=neg, n_positive=pos,
                positive_rate=round(pos / n, 6), neg_pos_ratio=round(ratio, 4))


# =====================================================================
# 2. Categorical preparation
# =====================================================================
def prepare_categoricals(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Bind `land_cover` and `lithology` to the FIXED category schema.

    Applied to the whole frame BEFORE splitting, which is safe precisely
    because the schema is a constant declared in this file (and in
    DATA_DICTIONARY.md) rather than something inferred from the data.
    No statistic crosses the split boundary, so there is no leakage:
    the mapping "forest -> 0" carries no information about any label.

    The alternative -- calling .astype("category") separately per split --
    would be the dangerous option: pandas would infer the levels from
    each split's contents, and a class missing from one split would
    silently renumber every code in it.

    One-hot encoding is deliberately NOT used. XGBoost >= 1.6 splits
    categorical features by partitioning their levels, which is a better
    fit for nominal geology/land-cover classes than 11 extra binary
    columns, and it keeps the saved feature list aligned with the
    dataset schema.
    """
    out = df.copy()
    for col, cats in CATEGORY_SCHEMA.items():
        out[col] = out[col].astype(CategoricalDtype(categories=cats, ordered=False))
    if not verbose:                      # describe the scheme once, not per split
        return out

    print("\nCategorical feature handling")
    print("  Method: XGBoost native categorical support "
          "(enable_categorical=True, partition splits)")
    print("  Encoding: fixed pandas CategoricalDtype, category order taken from")
    print("            DATA_DICTIONARY.md section 4 (nominal, not ordinal)")
    for col, cats in CATEGORY_SCHEMA.items():
        codes = ", ".join(f"{c}={i}" for i, c in enumerate(cats))
        print(f"    {col:<12} ({len(cats)} levels): {codes}")
    print("  Declared once for the whole frame, so train/validation/test/")
    print("  inference all share identical codes. No fitted parameters, no leakage.")
    print("  One-hot encoding not applied (see prepare_categoricals docstring).")
    return out


# =====================================================================
# 3. Chronological split
# =====================================================================
def chronological_split(df: pd.DataFrame):
    """
    Split by DATE BOUNDARY, not by row index -- identical logic to flood V1.0.

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


def verify_split_matches_flood(train, val, test) -> dict:
    """
    Confirm this baseline uses the SAME time boundaries as flood V1.0.

    Methodological cross-check only: it reads three date strings from the
    flood results file and touches no feature, label, probability, or
    prediction. Training is unaffected. Skipped silently if the flood
    model has not been run.
    """
    ranges = {
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "validation_date_range": [str(val["date"].min().date()), str(val["date"].max().date())],
        "test_date_range": [str(test["date"].min().date()), str(test["date"].max().date())],
    }
    if not FLOOD_RESULTS_PATH.exists():
        print("\n  [flood results not found -- split alignment check skipped]")
        return {"checked": False, "aligned": None, "landslide_ranges": ranges}

    flood_split = json.load(open(FLOOD_RESULTS_PATH, encoding="utf-8"))["split"]
    aligned = all(flood_split.get(k) == v for k, v in ranges.items())
    print("\n  Split alignment with Flood V1.0 (methodology check only, "
          "no flood data used):")
    for k, v in ranges.items():
        mark = "OK " if flood_split.get(k) == v else "DIFF"
        print(f"    [{mark}] {k:<24} landslide {v}  flood {flood_split.get(k)}")
    if not aligned:
        raise RuntimeError(
            "Landslide split boundaries differ from the flood baseline. "
            "The two models would no longer be time-aligned, making the "
            "later comparison invalid.")
    print("    Both baselines are evaluated over identical time periods.")
    return {"checked": True, "aligned": aligned, "landslide_ranges": ranges}


# =====================================================================
# 4. Model
# =====================================================================
def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    negative / positive, computed on the TRAINING SET ONLY.

    Recomputed for the landslide target -- deliberately not copied from
    the flood model, whose class balance is different.
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
# 5. Evaluation
# =====================================================================
def evaluate(model, X, y, split_name: str, threshold: float = DECISION_THRESHOLD) -> dict:
    """Full metric set for one split. Probability is P(landslide = 1)."""
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
    print(f"    Missed landslides (FN): {fn} of {int(y.sum())} actual landslide rows")
    print(f"    Calibration: mean predicted p = "
          f"{metrics['mean_predicted_probability']:.4f} vs observed rate "
          f"{metrics['observed_positive_rate']:.4f} "
          f"(Brier {metrics['brier_score']:.4f})")
    return metrics


# =====================================================================
# 6. Feature importance
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
    tbl["is_categorical"] = tbl["feature"].isin(CATEGORICAL_FEATURES)
    return tbl.sort_values("importance_gain", ascending=False).reset_index(drop=True)


def plot_feature_importance(tbl: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    t = tbl.sort_values("importance_gain")          # ascending for barh
    colors = ["#d95f0e" if c else "#2c7fb8" for c in t["is_categorical"]]
    ax.barh(t["feature"], t["importance_gain"], color=colors)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Landslide XGBoost V1.0 -- Feature Importance\n"
                 "SYNTHETIC DATA -- NOT FOR RESEARCH RESULTS", fontsize=11)
    for y_i, v in enumerate(t["importance_gain"]):
        ax.text(v, y_i, f" {v:.1f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2c7fb8"),
               plt.Rectangle((0, 0), 1, 1, color="#d95f0e")]
    ax.legend(handles, ["numeric", "categorical"], loc="lower right", frameon=False)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Feature importance plot -> {path}")


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    set_seeds()

    banner("LANDSLIDE XGBOOST MODEL -- VERSION 1.0")
    print("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS")
    print("Independent landslide baseline. No flood model, no TCDL.")

    # ---- Load / validate / inspect ----------------------------------
    banner("1. DATA LOADING AND VALIDATION", "-")
    df = load_dataset()
    validate_columns(df)
    missing_report = report_missing_values(df)

    print("\nClass distribution (full dataset)")
    overall_dist = report_class_distribution(df[TARGET], "overall")

    # ---- Split -------------------------------------------------------
    banner("2. CHRONOLOGICAL SPLIT", "-")
    train_df, val_df, test_df = chronological_split(df)
    split_check = verify_split_matches_flood(train_df, val_df, test_df)

    print("\nClass distribution per split")
    dist = {
        "overall": overall_dist,
        "train": report_class_distribution(train_df[TARGET], "train"),
        "validation": report_class_distribution(val_df[TARGET], "validation"),
        "test": report_class_distribution(test_df[TARGET], "test"),
    }

    # ---- Preprocessing ----------------------------------------------
    banner("3. PREPROCESSING", "-")
    print("  No fitted transformation is applied.")
    print("  Rationale: XGBoost is a tree ensemble -- it is invariant to")
    print("  monotonic feature scaling, so standardisation would add a")
    print("  fitted parameter set (a leakage surface) for zero benefit.")
    print("  Missing values, if any, are passed to XGBoost's native NaN")
    print("  handling rather than imputed. No rows are dropped.")
    print("  => Every learned quantity comes from the training split only.")

    # Categoricals are bound after the split has been reported, but the
    # schema is a constant, so applying it frame-wide is leakage-free.
    train_df = prepare_categoricals(train_df, verbose=True)
    val_df = prepare_categoricals(val_df)
    test_df = prepare_categoricals(test_df)

    X_train, y_train = train_df[LANDSLIDE_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[LANDSLIDE_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[LANDSLIDE_FEATURES], test_df[TARGET]

    # Codes must be identical across splits or the model reads garbage.
    for col in CATEGORICAL_FEATURES:
        assert (list(X_train[col].cat.categories)
                == list(X_val[col].cat.categories)
                == list(X_test[col].cat.categories)
                == CATEGORY_SCHEMA[col]), f"category codes diverged for '{col}'"
    print("  Category code alignment across train/validation/test: verified.")

    # ---- Class imbalance --------------------------------------------
    banner("4. CLASS IMBALANCE HANDLING", "-")
    spw = compute_scale_pos_weight(y_train)
    print(f"  Training neg/pos ratio (landslide): {spw:.2f}")
    print("  Computed from this target's own training split -- NOT copied")
    print("  from the flood baseline, whose class balance differs.")
    if USE_SCALE_POS_WEIGHT:
        print(f"  Applying scale_pos_weight = {spw:.4f}.")
        print("  Justified: substantial imbalance, and in an early-warning")
        print("  context a false negative (missed landslide) costs more than")
        print("  a false alarm.")
        print("  NOTE: this re-weighting deliberately inflates predicted")
        print("  probabilities relative to the true event rate. Recall improves;")
        print("  calibration degrades. Brier score and mean-predicted-vs-observed")
        print("  rate are reported below so the effect stays visible. Calibration")
        print("  is out of scope for V1.0 and must be revisited before the TCDL")
        print("  consumes these probabilities (same open item as flood V1.0).")
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
    print("  The test set was not used for early stopping, hyperparameter")
    print("  selection, or thresholding.")

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
    imp = feature_importance_table(model, LANDSLIDE_FEATURES)
    print(imp[["feature", "importance_gain", "importance_normalised",
               "split_count", "is_categorical"]].to_string(
                   index=False, float_format=lambda v: f"{v:.4f}"))
    plot_feature_importance(imp, IMPORTANCE_PLOT_PATH)
    print("  (Synthetic data: these rankings describe the generator, not")
    print("   real landslide physics.)")

    # ---- Test prediction table --------------------------------------
    banner("8. TEST-SET PROBABILITY TABLE", "-")
    test_proba = model.predict_proba(X_test)[:, 1]
    pred_table = pd.DataFrame({
        "date": test_df["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "latitude": test_df["latitude"].to_numpy(),
        "longitude": test_df["longitude"].to_numpy(),
        "actual_landslide": y_test.to_numpy().astype(int),
        "landslide_probability": np.round(test_proba, 6),
        "landslide_prediction": (test_proba >= DECISION_THRESHOLD).astype(int),
    }).sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
    pred_table.to_csv(PREDICTIONS_PATH, index=False)
    print(f"  {len(pred_table)} test rows -> {PREDICTIONS_PATH}")
    print(f"  Distinct probabilities: {pred_table['landslide_probability'].nunique()} "
          f"(range {pred_table['landslide_probability'].min():.4f} - "
          f"{pred_table['landslide_probability'].max():.4f})")
    print("\n  Sample (highest predicted probability):")
    print(pred_table.sort_values("landslide_probability", ascending=False)
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
    # the evaluated model exactly. (Same fix as flood V1.0.)
    full_booster = model.get_booster()
    best_booster = full_booster[: best_iter + 1]
    best_booster.save_model(str(MODEL_PATH))
    n_trees_saved = len(best_booster.get_dump())
    print(f"  Saved model path: {MODEL_PATH}")
    print(f"  Trees saved: {n_trees_saved} (sliced to best_iteration={best_iter}; "
          f"{len(full_booster.get_dump())} were trained before early stopping)")

    # Reload check: the artifact on disk must reproduce the evaluated model.
    # enable_categorical=True is required when rebuilding the DMatrix,
    # otherwise the category columns cannot be read back.
    reloaded = xgb.Booster()
    reloaded.load_model(str(MODEL_PATH))
    reload_proba = reloaded.predict(xgb.DMatrix(X_test, enable_categorical=True))
    max_drift = float(np.abs(reload_proba - test_proba).max())
    if max_drift > 1e-6:
        raise RuntimeError(
            f"Saved model does not reproduce evaluated predictions "
            f"(max drift {max_drift:.6g}). Refusing to ship an inconsistent artifact.")
    print(f"  Reload check: saved model reproduces test probabilities "
          f"(max drift {max_drift:.2e})")

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "landslide_xgboost",
            "version": "1.0",
            "target": TARGET,
            "feature_order": LANDSLIDE_FEATURES,
            "n_features": len(LANDSLIDE_FEATURES),
            "n_trees": n_trees_saved,
            "best_iteration": best_iter,
            "common_features": COMMON_FEATURES,
            "landslide_specific_features": LANDSLIDE_SPECIFIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "categorical_handling": {
                "method": "xgboost native categorical (enable_categorical=True)",
                "pandas_dtype": "CategoricalDtype(categories=<below>, ordered=False)",
                "one_hot": False,
                "category_schema": CATEGORY_SCHEMA,
                "note": ("Category ORDER defines the integer codes and must be "
                         "reused verbatim at inference. Codes are nominal "
                         "identifiers, not an ordering. Source: "
                         "DATA_DICTIONARY.md section 4."),
            },
            "excluded_columns": {
                "flood": "other hazard target -- keeps this baseline independent",
                "flood_probability/flood_prediction": "flood model outputs; "
                                                      "reserved for the TCDL, not for this model",
                "river_level_m/distance_to_river_km/drainage_density":
                    "flood-specific predictors",
                "date": "chronological split key only",
                "latitude/longitude": "metadata only; excluded to prevent the model "
                                      "memorising the 9 synthetic locations",
            },
            "decision_threshold": DECISION_THRESHOLD,
            "note": "Feature order must be preserved for inference.",
        }, f, indent=2)
    print(f"  Saved feature list: {FEATURES_PATH}")

    payload = {
        "_warning": ("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS. "
                     "These metrics describe a simulated dataset and must not be "
                     "reported as real-world landslide prediction performance."),
        "model": "landslide_xgboost",
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
            "feature_names": LANDSLIDE_FEATURES,
            "n_features": len(LANDSLIDE_FEATURES),
            "common_features": COMMON_FEATURES,
            "landslide_specific_features": LANDSLIDE_SPECIFIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
        },
        "categorical_handling": {
            "method": "xgboost native categorical (enable_categorical=True)",
            "one_hot": False,
            "category_schema": CATEGORY_SCHEMA,
            "fitted_from_data": False,
            "leakage_note": ("Category schema is a constant declared in code and "
                             "in DATA_DICTIONARY.md, not inferred per split, so "
                             "codes are identical across train/validation/test."),
        },
        "split": {
            "method": "chronological, cut on unique-date boundaries (no shuffling)",
            "fractions_requested": {"train": TRAIN_FRACTION,
                                    "validation": VALIDATION_FRACTION,
                                    "test": round(1 - TRAIN_FRACTION - VALIDATION_FRACTION, 4)},
            "train_size": int(len(train_df)),
            "validation_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "train_date_range": split_check["landslide_ranges"]["train_date_range"],
            "validation_date_range": split_check["landslide_ranges"]["validation_date_range"],
            "test_date_range": split_check["landslide_ranges"]["test_date_range"],
            "matches_flood_baseline": split_check["aligned"],
        },
        "class_distribution": dist,
        "class_imbalance_handling": {
            "method": "scale_pos_weight" if USE_SCALE_POS_WEIGHT else "none",
            "scale_pos_weight": spw,
            "computed_on": "training split only",
            "note": "Recomputed for the landslide target; not copied from flood.",
        },
        "preprocessing": {
            "fitted_transformations": "none",
            "scaling": "not applied (tree model is scale-invariant)",
            "missing_value_policy": "no rows dropped; NaN passed to XGBoost native handling",
        },
        "model_parameters": {k: v for k, v in model.get_params().items()
                             if isinstance(v, (int, float, str, bool, type(None)))},
        "early_stopping": {
            "best_iteration": best_iter,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "metric": XGB_PARAMS["eval_metric"],
            "selected_on": "validation split",
            "trees_saved": n_trees_saved,
        },
        "decision_threshold": DECISION_THRESHOLD,
        "metrics": results,
        "calibration_summary": {
            split: {
                "mean_predicted_probability": results[split]["mean_predicted_probability"],
                "observed_positive_rate": results[split]["observed_positive_rate"],
                "brier_score": results[split]["brier_score"],
            } for split in ("train", "validation", "test")
        },
        "feature_importance": imp.drop(columns=["is_categorical"]).to_dict(orient="records"),
        "artifact_reload_check": {"max_probability_drift": max_drift, "passed": True},
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

    banner("LANDSLIDE XGBOOST V1.0 COMPLETE")
    t = results["test"]
    print(f"  Final unseen test: accuracy {t['accuracy']:.4f} | "
          f"precision {t['precision']:.4f} | recall {t['recall']:.4f} | "
          f"F1 {t['f1']:.4f} | ROC-AUC {t['roc_auc']:.4f} | "
          f"PR-AUC {t['pr_auc']:.4f}")
    print(f"  Missed landslides on test set: "
          f"{t['confusion_matrix']['false_negatives']} of {t['n_actual_positives']}")
    print("\n  SYNTHETIC DATA -- these numbers characterise the pipeline,")
    print("  not real-world landslide predictability.")


if __name__ == "__main__":
    main()
