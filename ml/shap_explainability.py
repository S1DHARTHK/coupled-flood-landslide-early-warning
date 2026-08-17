"""
SHAP Explainability for the Flood and Landslide XGBoost models
=====================================================================
EXPLANATION LAYER ONLY.

This module explains why the two frozen XGBoost baselines produce the
probabilities they do. It sits beside the pipeline, never inside it:

    Input -> Flood XGBoost     -> flood probability     -> SHAP explanation
    Input -> Landslide XGBoost -> landslide probability -> SHAP explanation

It trains nothing, changes nothing, and is not consulted by TCDL. The
models are loaded from their saved artifacts, the feature order and
categorical schema are read from the saved feature contracts, and the
evaluation split is read from the saved training results -- so nothing
here can drift from, or silently redefine, the pipeline it describes.

TCDL is deliberately out of scope: it is a transparent rule-based layer,
and its explanations remain the R1-R7 rule traces it already emits. SHAP
is not applied to it and never will be by this module.

How the SHAP values are computed
--------------------------------
Via XGBoost's native `pred_contribs=True`, which is the same exact
TreeSHAP routine `shap.TreeExplainer` invokes for XGBoost models. Going
through the booster directly guarantees the landslide model's native
categorical features are decoded with `enable_categorical=True`; letting
a generic explainer rebuild the matrix risks silently mis-encoding them.
Additivity is asserted at runtime as proof the decomposition is exact.

INTERPRETATION
--------------
SHAP describes MODEL BEHAVIOUR, not physical causation. A positive SHAP
value means a feature pushed *the model's output* higher, not that it
caused a hazard. Values are in log-odds (margin) space, the space the
model is additive in; the probability is sigmoid(sum of contributions +
base value).

*** SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS ***
The underlying dataset is simulated, so these explanations describe the
data generator, not real flood or landslide processes.

Usage
-----
    python ml/shap_explainability.py          # full run + all outputs

    from ml.shap_explainability import explain_prediction, global_importance
    explain_prediction("flood", record)        # per-prediction, JSON-ready
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import shap
import xgboost as xgb

ModelKey = Literal["flood", "landslide"]

# ---------------------------------------------------------------------
# Paths -- all resolved relative to this file, never hard-coded
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_PATH = PROJECT_ROOT / "synthetic_master_dataset.csv"
OUTPUT_DIR = HERE / "outputs" / "shap"

ARTIFACTS: dict[str, dict[str, Path]] = {
    "flood": {
        "model": HERE / "flood_xgboost_model.json",
        "features": HERE / "flood_features.json",
        "results": HERE / "flood_model_results.json",
    },
    "landslide": {
        "model": HERE / "landslide_xgboost_model.json",
        "features": HERE / "landslide_features.json",
        "results": HERE / "landslide_model_results.json",
    },
}

MODEL_LABELS = {
    "flood": "Flood XGBoost V1.0",
    "landslide": "Landslide XGBoost V1.0",
}

# Additivity tolerance: sum(shap) + base must reconstruct the raw margin.
ADDITIVITY_TOL = 1e-4
# Prediction-consistency tolerance for the before/after check.
PREDICTION_TOL = 1e-9


def banner(text: str, char: str = "=") -> None:
    print("\n" + char * 70)
    print(text)
    print(char * 70)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: {path}. Run the training scripts first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# =====================================================================
# Model handle
# =====================================================================
class ExplainableModel:
    """
    A frozen XGBoost model plus everything needed to explain it.

    Loading is strictly read-only: `Booster.load_model` does not modify
    the artifact, and no training API is ever called.
    """

    def __init__(self, key: ModelKey) -> None:
        self.key = key
        self.label = MODEL_LABELS[key]
        paths = ARTIFACTS[key]

        self.meta = _read_json(paths["features"])
        self.results = _read_json(paths["results"])

        # Feature order comes from the saved contract -- NOT redefined here.
        self.features: list[str] = list(self.meta["feature_order"])
        self.threshold: float = float(self.meta.get("decision_threshold", 0.5))

        cat = self.meta.get("categorical_handling", {})
        self.category_schema: dict[str, list[str]] = dict(cat.get("category_schema", {}))
        self.categorical_features: list[str] = list(
            self.meta.get("categorical_features", [])
        )

        self.booster = xgb.Booster()
        self.booster.load_model(str(paths["model"]))

    # -- matrix construction -------------------------------------------
    def frame(self, records: pd.DataFrame | Iterable[dict]) -> pd.DataFrame:
        """Select and type the model's features exactly as it expects them."""
        df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"{self.label}: input is missing features {missing}")
        X = df[self.features].copy()
        for col, cats in self.category_schema.items():
            if col in X.columns:
                X[col] = X[col].astype(CategoricalDtype(categories=cats, ordered=False))
        return X

    def dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X, enable_categorical=True)

    # -- prediction (unchanged behaviour, used for the integrity check) --
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.dmatrix(X))

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.dmatrix(X), output_margin=True)

    # -- SHAP ----------------------------------------------------------
    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        TreeSHAP contributions in log-odds (margin) space.

        Returns (shap_values [n, n_features], base_values [n]).
        The final column of `pred_contribs` is the bias / expected value.
        """
        contribs = self.booster.predict(self.dmatrix(X), pred_contribs=True)
        contribs = np.asarray(contribs)
        values, base = contribs[:, :-1], contribs[:, -1]

        # Proof the decomposition is exact for THIS model on THIS input.
        reconstructed = values.sum(axis=1) + base
        actual = self.predict_margin(X)
        drift = float(np.abs(reconstructed - actual).max())
        if drift > ADDITIVITY_TOL:
            raise RuntimeError(
                f"{self.label}: SHAP additivity check failed (max drift {drift:.3g}). "
                "The explanation does not reconstruct the model output; refusing "
                "to emit misleading attributions.")
        self.last_additivity_drift = drift
        return values, base

    def numeric_view(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Plot-only copy with categoricals as their integer codes.

        SHAP's beeswarm colours points by feature value and cannot colour a
        pandas Categorical. The codes are the documented ones from the model's
        own schema, so the colour scale stays faithful to what the model saw.
        This affects PLOTTING ONLY -- the SHAP values were computed from the
        properly-typed categorical matrix above.
        """
        out = X.copy()
        for col in self.category_schema:
            if col in out.columns:
                out[col] = out[col].cat.codes.astype(float)
        return out


_MODEL_CACHE: dict[str, ExplainableModel] = {}


def get_model(key: ModelKey) -> ExplainableModel:
    """Load (and cache) a frozen model. Safe to call from a web process."""
    if key not in _MODEL_CACHE:
        if key not in ARTIFACTS:
            raise KeyError(f"Unknown model '{key}'. Expected one of {list(ARTIFACTS)}.")
        _MODEL_CACHE[key] = ExplainableModel(key)
    return _MODEL_CACHE[key]


# =====================================================================
# Dataset -- reuse the recorded evaluation split, do not redefine it
# =====================================================================
def load_test_split() -> pd.DataFrame:
    """
    The unseen test rows, using the date boundaries recorded by training.

    The split is read from flood_model_results.json rather than recomputed,
    so the explanations describe exactly the rows the models were evaluated
    on. Both models share these boundaries by construction.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    split = _read_json(ARTIFACTS["flood"]["results"])["split"]
    start, end = split["test_date_range"]
    test = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return test.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)


# =====================================================================
# Global explanation
# =====================================================================
def global_importance(key: ModelKey, X: pd.DataFrame | None = None) -> list[dict]:
    """
    Mean absolute SHAP value per feature, sorted descending.

    Machine-readable; the same structure is written to
    <model>_shap_importance.json.
    """
    model = get_model(key)
    if X is None:
        X = model.frame(load_test_split())
    values, _ = model.shap_values(X)
    mean_abs = np.abs(values).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    rows = [
        {
            "feature": f,
            "mean_abs_shap": round(float(m), 6),
            "share_of_total": round(float(m) / total, 6),
            "mean_signed_shap": round(float(values[:, i].mean()), 6),
            "is_categorical": f in model.categorical_features,
        }
        for i, (f, m) in enumerate(zip(model.features, mean_abs))
    ]
    return sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)


# =====================================================================
# Per-prediction explanation (machine-readable; FastAPI-ready)
# =====================================================================
def explain_prediction(
    key: ModelKey,
    record: dict | pd.Series | pd.DataFrame,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Explain one prediction from an existing model.

    Accepts a single observation (dict / Series / one-row DataFrame) whose
    keys cover the model's feature contract. Returns a JSON-serialisable
    structure ready for an API layer.

    Wording note: every contribution is described as contributing to the
    MODEL'S PREDICTION. Nothing here asserts physical causation.
    """
    model = get_model(key)

    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError("explain_prediction expects exactly one observation.")
        row_df = record
    elif isinstance(record, pd.Series):
        row_df = record.to_frame().T
    else:
        row_df = pd.DataFrame([record])

    X = model.frame(row_df)
    values, base = model.shap_values(X)
    v, b = values[0], float(base[0])

    proba = float(model.predict_proba(X)[0])
    margin = float(model.predict_margin(X)[0])

    contributions = []
    for i, feat in enumerate(model.features):
        raw = X.iloc[0, i]
        # Categorical values are reported as their label, not their code.
        value: Any = raw if isinstance(raw, str) else (
            None if pd.isna(raw) else (str(raw) if feat in model.category_schema else float(raw))
        )
        s = float(v[i])
        contributions.append({
            "feature": feat,
            "value": value,
            "shap_value": round(s, 6),
            "abs_shap_value": round(abs(s), 6),
            "direction": (
                "increases the model's predicted risk" if s > 0
                else "decreases the model's predicted risk" if s < 0
                else "no contribution"
            ),
            "effect": "increases risk" if s > 0 else "decreases risk" if s < 0 else "neutral",
            "magnitude": round(abs(s), 6),
            "is_categorical": feat in model.categorical_features,
        })

    contributions.sort(key=lambda c: c["abs_shap_value"], reverse=True)
    if top_k:
        contributions = contributions[:top_k]

    return {
        "_warning": ("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS. "
                     "Explanations describe the model's behaviour on simulated data."),
        "model": model.key,
        "model_label": model.label,
        "model_version": model.meta.get("version", "1.0"),
        "target": model.meta.get("target"),
        "probability": round(proba, 6),
        "prediction": int(proba >= model.threshold),
        "decision_threshold": model.threshold,
        "raw_margin": round(margin, 6),
        "base_value": round(b, 6),
        "base_value_probability": round(float(_sigmoid(np.array([b]))[0]), 6),
        "sum_shap_values": round(float(v.sum()), 6),
        "shap_space": "log-odds (margin); probability = sigmoid(base_value + sum of SHAP values)",
        "additivity_check": {
            "reconstructed_margin": round(b + float(v.sum()), 6),
            "model_margin": round(margin, 6),
            "max_abs_difference": round(abs(b + float(v.sum()) - margin), 9),
            "passed": abs(b + float(v.sum()) - margin) <= ADDITIVITY_TOL,
        },
        "n_features": len(model.features),
        "contributions": contributions,
        "interpretation_note": (
            "SHAP values quantify how much each feature contributed to THIS "
            "model's output relative to its base value. They describe model "
            "behaviour and must not be read as evidence of physical causation."
        ),
    }


def explain_test_sample(key: ModelKey, index: int = 0, top_k: int | None = None) -> dict:
    """Convenience wrapper: explain row `index` of the unseen test split."""
    test = load_test_split()
    if not 0 <= index < len(test):
        raise IndexError(f"index {index} out of range for test split of {len(test)} rows")
    row = test.iloc[[index]]
    out = explain_prediction(key, row, top_k=top_k)
    out["sample"] = {
        "test_row_index": int(index),
        "date": pd.Timestamp(row["date"].iloc[0]).strftime("%Y-%m-%d"),
        "latitude": float(row["latitude"].iloc[0]),
        "longitude": float(row["longitude"].iloc[0]),
        "actual_label": int(row[key].iloc[0]) if key in row.columns else None,
    }
    return out


# =====================================================================
# Plots
# =====================================================================
def _save_bar_plot(key: ModelKey, values: np.ndarray, X_plot: pd.DataFrame,
                   path: Path) -> None:
    model = get_model(key)
    plt.figure()
    shap.summary_plot(values, X_plot, feature_names=model.features,
                      plot_type="bar", show=False, max_display=len(model.features))
    plt.title(f"{model.label} -- global SHAP importance (mean |SHAP|)\n"
              "SYNTHETIC DATA -- NOT FOR RESEARCH RESULTS", fontsize=10)
    plt.xlabel("mean |SHAP value|  (log-odds impact on model output)", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    bar plot     -> {path.name}")


def _save_beeswarm_plot(key: ModelKey, values: np.ndarray, X_plot: pd.DataFrame,
                        path: Path) -> None:
    model = get_model(key)
    plt.figure()
    shap.summary_plot(values, X_plot, feature_names=model.features,
                      show=False, max_display=len(model.features))
    plt.title(f"{model.label} -- SHAP summary (beeswarm)\n"
              "SYNTHETIC DATA -- NOT FOR RESEARCH RESULTS", fontsize=10)
    plt.xlabel("SHAP value (log-odds impact on model output)", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    beeswarm     -> {path.name}")


# =====================================================================
# Integrity: predictions must be identical before and after SHAP
# =====================================================================
def prediction_consistency_check(key: ModelKey, X: pd.DataFrame) -> dict:
    """
    Predict -> build explainer + compute SHAP -> predict again -> compare.

    This is the guarantee the task requires: adding the explanation layer
    must not perturb a single probability.
    """
    model = get_model(key)
    before = model.predict_proba(X).copy()

    explainer = shap.TreeExplainer(model.booster)   # constructed, as required
    values, base = model.shap_values(X)

    after = model.predict_proba(X)
    max_diff = float(np.abs(before - after).max())

    # Independent reload from disk: the artifact itself must be untouched.
    fresh = xgb.Booster()
    fresh.load_model(str(ARTIFACTS[key]["model"]))
    reloaded = fresh.predict(model.dmatrix(X))
    reload_diff = float(np.abs(before - reloaded).max())

    return {
        "n_rows": int(len(X)),
        "max_abs_probability_difference": max_diff,
        "identical_within_tolerance": bool(max_diff <= PREDICTION_TOL),
        "max_abs_difference_after_reload_from_disk": reload_diff,
        "reload_identical": bool(reload_diff <= PREDICTION_TOL),
        "shap_additivity_max_drift": float(getattr(model, "last_additivity_drift", 0.0)),
        "explainer_type": type(explainer).__name__,
        "model_retrained": False,
    }


# =====================================================================
# Full run
# =====================================================================
def run(key: ModelKey, test: pd.DataFrame) -> dict:
    model = get_model(key)
    cat_note = (f"{', '.join(model.categorical_features)} categorical"
                if model.categorical_features else "all numeric")
    print(f"\n  {model.label}")
    print(f"    features     : {len(model.features)} ({cat_note})")
    print(f"    trees        : {model.meta.get('n_trees')}  "
          f"threshold {model.threshold}")

    X = model.frame(test)

    consistency = prediction_consistency_check(key, X)
    print(f"    prediction consistency: max |delta p| = "
          f"{consistency['max_abs_probability_difference']:.3e} "
          f"({'IDENTICAL' if consistency['identical_within_tolerance'] else 'CHANGED'})")
    print(f"    SHAP additivity       : max drift "
          f"{consistency['shap_additivity_max_drift']:.3e}")

    values, base = model.shap_values(X)
    importance = global_importance(key, X)

    X_plot = model.numeric_view(X)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_bar_plot(key, values, X_plot, OUTPUT_DIR / f"{key}_shap_bar.png")
    _save_beeswarm_plot(key, values, X_plot, OUTPUT_DIR / f"{key}_shap_summary.png")

    payload = {
        "_warning": ("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS. "
                     "These attributions describe a model trained on simulated data."),
        "_scope_note": ("Explanation layer only. The model was not retrained, "
                        "modified, or re-thresholded, and SHAP is not used by "
                        "TCDL or by any prediction path."),
        "model": key,
        "model_label": model.label,
        "model_version": model.meta.get("version", "1.0"),
        "target": model.meta.get("target"),
        "n_features": len(model.features),
        "feature_order": model.features,
        "categorical_features": model.categorical_features,
        "category_schema": model.category_schema,
        "explainer": "TreeSHAP via XGBoost pred_contribs (exact)",
        "shap_space": "log-odds (margin)",
        "base_value_mean": round(float(np.mean(base)), 6),
        "dataset": {
            "split": "test (unseen)",
            "date_range": _read_json(ARTIFACTS["flood"]["results"])["split"]["test_date_range"],
            "n_rows": int(len(X)),
            "source": DATA_PATH.name,
        },
        "prediction_integrity": consistency,
        "global_importance": importance,
        "outputs": {
            "bar_plot": f"{key}_shap_bar.png",
            "summary_plot": f"{key}_shap_summary.png",
        },
        "interpretation_note": (
            "Mean |SHAP| ranks how much each feature moved this model's output "
            "on average. It describes model behaviour, not physical causation."
        ),
    }

    out_path = OUTPUT_DIR / f"{key}_shap_importance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"    importance   -> {out_path.name}")

    print(f"    top features by mean |SHAP|:")
    for r in importance[:5]:
        tag = " (categorical)" if r["is_categorical"] else ""
        print(f"      {r['feature']:<22} {r['mean_abs_shap']:.4f}  "
              f"({100*r['share_of_total']:.1f}%){tag}")
    return payload


def main() -> None:
    banner("SHAP EXPLAINABILITY -- FLOOD & LANDSLIDE XGBOOST")
    print("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS")
    print("Explanation layer only: no retraining, no model or TCDL changes.")
    print(f"shap {shap.__version__} | xgboost {xgb.__version__}")

    banner("1. LOADING FROZEN MODELS AND TEST SPLIT", "-")
    test = load_test_split()
    print(f"  test split: {len(test)} rows "
          f"({test['date'].min().date()} -> {test['date'].max().date()})")
    for key in ("flood", "landslide"):
        m = get_model(key)
        print(f"  loaded {m.label}: {len(m.features)} features, "
              f"{m.meta.get('n_trees')} trees (read-only)")

    banner("2. GLOBAL SHAP EXPLANATIONS", "-")
    payloads = {key: run(key, test) for key in ("flood", "landslide")}

    banner("3. EXAMPLE PER-PREDICTION EXPLANATIONS", "-")
    for key in ("flood", "landslide"):
        model = get_model(key)
        X = model.frame(test)
        idx = int(np.argmax(model.predict_proba(X)))    # highest-risk test row
        ex = explain_test_sample(key, idx, top_k=6)
        print(f"\n  {ex['model_label']} -- highest-risk test row "
              f"({ex['sample']['date']}, {ex['sample']['latitude']:.3f}N "
              f"{ex['sample']['longitude']:.3f}E)")
        print(f"    probability {ex['probability']:.4f} "
              f"(prediction {ex['prediction']}, actual {ex['sample']['actual_label']})")
        print(f"    base value {ex['base_value']:.4f} log-odds "
              f"(= {ex['base_value_probability']:.4f} probability)")
        print(f"    {'feature':<22}{'value':>14}{'SHAP':>11}   effect")
        for c in ex["contributions"]:
            val = c["value"]
            vs = val if isinstance(val, str) else ("--" if val is None else f"{val:.3f}")
            shap_str = f"{c['shap_value']:+.4f}"
            print(f"    {c['feature']:<22}{vs:>14}{shap_str:>11}   {c['effect']}")
        ex_path = OUTPUT_DIR / f"{key}_shap_example_explanation.json"
        with open(ex_path, "w", encoding="utf-8") as f:
            json.dump(ex, f, indent=2)
        print(f"    saved -> {ex_path.name}")

    banner("4. MODEL INTEGRITY", "-")
    ok = True
    for key, p in payloads.items():
        c = p["prediction_integrity"]
        ok &= c["identical_within_tolerance"] and c["reload_identical"]
        print(f"  {key:<10} probabilities unchanged: "
              f"{c['identical_within_tolerance']} (max |delta p| "
              f"{c['max_abs_probability_difference']:.3e}) | "
              f"reload identical: {c['reload_identical']} | retrained: "
              f"{c['model_retrained']}")
    print(f"\n  TCDL: untouched -- SHAP is not applied to the rule-based layer.")
    print(f"  Overall integrity: {'PASS' if ok else 'FAIL'}")

    banner("SHAP EXPLAINABILITY COMPLETE")
    print(f"  outputs -> {OUTPUT_DIR}")
    print("  SYNTHETIC DATA -- attributions describe the model, not real hazards.")


if __name__ == "__main__":
    main()
