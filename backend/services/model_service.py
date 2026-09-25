"""
Model service -- loads one set of frozen XGBoost models ONCE at startup.

Responsibilities:
  * load flood + landslide boosters and their feature metadata
  * validate incoming observations against the saved feature lists
  * produce probabilities and predictions using the SAVED thresholds

It does not train, tune, or modify anything. Feature order, categorical
schema and decision threshold all come from the artifacts on disk, so
the API can never drift from the models it serves.

The backend holds two instances: the REAL model set (config.REAL_MODEL_SET,
served by /predict/flood and /predict/landslide) and the SYNTHETIC set
(config.SYNTHETIC_MODEL_SET, still used by TCDL, SHAP and the dashboard).
"""

from __future__ import annotations

import json
import math
from datetime import date as date_type
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.api.types import CategoricalDtype

from .. import config

# Physically impossible values are rejected before they reach a model.
# These are hard physical limits, not the training ranges: a value outside the
# training range is legitimate input and is scored as given.
PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "rainfall_1d_mm": (0.0, None), "rainfall_3d_mm": (0.0, None),
    "rainfall_7d_mm": (0.0, None), "rainfall_14d_mm": (0.0, None),
    "rainfall_30d_mm": (0.0, None),
    "soil_moisture": (0.0, 1.0),
    "humidity_percent": (0.0, 100.0),
    "slope_degree": (0.0, 90.0),
    "aspect_degree": (0.0, 360.0),
    "distance_to_river_km": (0.0, None),
    "drainage_density": (0.0, None),
}


class ArtifactError(RuntimeError):
    """Raised when a required model artifact is missing or unreadable."""


def _read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise ArtifactError(
            f"Required artifact '{label}' not found at {path}. "
            "Run the ML pipeline (scripts/models/flood_xgboost.py, scripts/models/landslide_xgboost.py, "
            "scripts/models/tcdl_v1.py) before starting the API.")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Artifact '{label}' at {path} is malformed JSON: {exc}") from exc


class ModelService:
    """Holds both frozen boosters and their metadata for the process lifetime."""

    def __init__(self, model_set: dict[str, Any] | None = None) -> None:
        # Default = the synthetic set, i.e. exactly the behaviour before the
        # real models existed.
        self.paths: dict[str, Any] = dict(model_set or config.SYNTHETIC_MODEL_SET)
        self.name: str = str(self.paths.get("name", "synthetic"))
        self.errors: list[str] = []
        self.flood_ready = False
        self.landslide_ready = False
        self.dataset_ready = False

        self.flood_meta: dict[str, Any] = {}
        self.landslide_meta: dict[str, Any] = {}
        self.flood_results: dict[str, Any] = {}
        self.landslide_results: dict[str, Any] = {}
        self.flood_features: list[str] = []
        self.landslide_features: list[str] = []
        self.category_schema: dict[str, list[str]] = {}
        self.flood_threshold = 0.5
        self.landslide_threshold = 0.5
        # Features that were genuinely missing (NaN) in the model's training
        # data. The trees learned a default branch for them, so a null value
        # is scored natively by XGBoost -- never imputed. Read from the
        # training results, not hard-coded (real flood model: river_level_m).
        self.nullable_features: dict[str, set[str]] = {"flood": set(), "landslide": set()}

        self._flood_booster: xgb.Booster | None = None
        self._landslide_booster: xgb.Booster | None = None
        self.master: pd.DataFrame | None = None

        self._load_flood()
        self._load_landslide()
        self._load_master_dataset()

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------
    @staticmethod
    def _nullable_from_results(results: dict, features: list[str]) -> set[str]:
        counts = results.get("dataset", {}).get("missing_values_in_model_columns", {})
        return {f for f in features if int(counts.get(f, 0) or 0) > 0}

    def _check_booster(self, booster: xgb.Booster, features: list[str], label: str) -> None:
        """The saved booster must expect exactly the saved feature order."""
        names = list(booster.feature_names or [])
        if names and names != features:
            raise ArtifactError(
                f"{label} booster feature names {names} differ from the saved "
                f"feature_order {features}; refusing to serve an inconsistent model.")

    def _load_flood(self) -> None:
        try:
            self.flood_meta = _read_json(self.paths["flood_features"], "flood_features")
            self.flood_results = _read_json(self.paths["flood_results"], "flood_model_results")
            self.flood_features = list(self.flood_meta["feature_order"])
            self.flood_threshold = float(self.flood_meta.get("decision_threshold", 0.5))
            model_path = Path(self.paths["flood_model"])
            if not model_path.exists():
                raise ArtifactError(f"Flood model not found at {model_path}")
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            self._check_booster(booster, self.flood_features, "flood")
            self._flood_booster = booster
            self.nullable_features["flood"] = self._nullable_from_results(
                self.flood_results, self.flood_features)
            self.flood_ready = True
        except Exception as exc:                       # surfaced via /health
            self.errors.append(f"{self.name} flood model: {exc}")

    def _load_landslide(self) -> None:
        try:
            self.landslide_meta = _read_json(self.paths["landslide_features"],
                                             "landslide_features")
            self.landslide_results = _read_json(self.paths["landslide_results"],
                                                "landslide_model_results")
            self.landslide_features = list(self.landslide_meta["feature_order"])
            self.landslide_threshold = float(
                self.landslide_meta.get("decision_threshold", 0.5))
            self.category_schema = dict(
                self.landslide_meta["categorical_handling"]["category_schema"])
            model_path = Path(self.paths["landslide_model"])
            if not model_path.exists():
                raise ArtifactError(f"Landslide model not found at {model_path}")
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            self._check_booster(booster, self.landslide_features, "landslide")
            self._landslide_booster = booster
            self.nullable_features["landslide"] = self._nullable_from_results(
                self.landslide_results, self.landslide_features)
            self.landslide_ready = True
        except Exception as exc:
            self.errors.append(f"{self.name} landslide model: {exc}")

    def _load_master_dataset(self) -> None:
        try:
            master_path = Path(self.paths["master_dataset"])
            if not master_path.exists():
                raise ArtifactError(f"Master dataset not found at {master_path}")
            df = pd.read_csv(master_path, parse_dates=["date"])
            df["location_id"] = (df["latitude"].astype(str) + "_"
                                 + df["longitude"].astype(str))
            self.master = df.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
            self.dataset_ready = True
        except Exception as exc:
            self.errors.append(f"{self.name} master dataset: {exc}")

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    def validate_features(self, records: list[dict], features: list[str],
                          model_name: str) -> list[str]:
        """Return the list of missing/invalid feature problems (empty = OK)."""
        problems: list[str] = []
        for i, rec in enumerate(records):
            missing = [f for f in features if f not in rec or rec[f] is None]
            if missing:
                problems.append(
                    f"record {i}: missing required {model_name} features {missing}")
            for col, cats in self.category_schema.items():
                if col in features and col in rec and rec[col] is not None:
                    if rec[col] not in cats:
                        problems.append(
                            f"record {i}: '{col}' value {rec[col]!r} is not in the "
                            f"documented schema {cats}")
        return problems

    def validate_observations(self, records: list, hazard: str) -> list[str]:
        """
        Strict validation used by the single-hazard prediction endpoints.

        Every feature of the model must be present. A null value is accepted
        only for features that were missing in the model's own training data
        (``nullable_features``); it is then passed to XGBoost as NaN and scored
        through the default branch the trees learned -- nothing is imputed.
        Numbers must be finite JSON numbers inside PHYSICAL_BOUNDS; categorical
        values must be in the saved schema. Returns the problems (empty = OK).
        """
        features = self.flood_features if hazard == "flood" else self.landslide_features
        nullable = self.nullable_features.get(hazard, set())
        problems: list[str] = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                problems.append(f"record {i}: must be a JSON object, got {type(rec).__name__}")
                continue
            missing = [f for f in features if f not in rec]
            if missing:
                problems.append(f"record {i}: missing required {hazard} features {missing}")
            for f in features:
                if f not in rec:
                    continue
                v = rec[f]
                if v is None:
                    if f not in nullable:
                        problems.append(
                            f"record {i}: '{f}' may not be null for the {self.name} {hazard} "
                            f"model (nullable features: {sorted(nullable) or 'none'})")
                    continue
                if f in self.category_schema:
                    if v not in self.category_schema[f]:
                        problems.append(
                            f"record {i}: '{f}' value {v!r} is not in the documented "
                            f"schema {self.category_schema[f]}")
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    problems.append(f"record {i}: '{f}' must be a number, got {v!r}")
                    continue
                if not math.isfinite(float(v)):
                    problems.append(f"record {i}: '{f}' must be finite, got {v!r} "
                                    "(send null for a genuinely missing value)")
                    continue
                lo, hi = PHYSICAL_BOUNDS.get(f, (None, None))
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    rng = f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"
                    problems.append(f"record {i}: '{f}' = {v} is outside the physically "
                                    f"possible range {rng}")
            # identifiers (never features)
            d = rec.get("date")
            if d is None:
                problems.append(f"record {i}: missing 'date'")
            else:
                try:
                    date_type.fromisoformat(str(d))
                except ValueError:
                    problems.append(f"record {i}: 'date' must be YYYY-MM-DD, got {d!r}")
            for key, lo, hi in (("latitude", -90, 90), ("longitude", -180, 180)):
                v = rec.get(key)
                if v is None:
                    problems.append(f"record {i}: missing '{key}' (or supply 'district')")
                elif isinstance(v, bool) or not isinstance(v, (int, float)) \
                        or not math.isfinite(float(v)) or not lo <= v <= hi:
                    problems.append(f"record {i}: '{key}' must be a number in [{lo}, {hi}], "
                                    f"got {v!r}")
        return problems

    @staticmethod
    def missing_features(record: dict, features: list[str]) -> list[str]:
        """Features sent as null (scored through XGBoost's default branch)."""
        return [f for f in features if record.get(f) is None]

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def _frame(self, records: list[dict], features: list[str],
               categorical: bool) -> pd.DataFrame:
        X = pd.DataFrame(records)[features].copy()
        cats = self.category_schema if categorical else {}
        for col in X.columns:
            if col in cats:
                X[col] = X[col].astype(CategoricalDtype(categories=cats[col], ordered=False))
            else:
                # null -> NaN (XGBoost's missing value); a column of nulls would
                # otherwise stay 'object' and could not be read by DMatrix.
                X[col] = pd.to_numeric(X[col], errors="raise").astype("float64")
        return X

    def predict_flood(self, records: list[dict]) -> np.ndarray:
        if not self.flood_ready or self._flood_booster is None:
            raise ArtifactError("Flood model is not loaded; see /health for details.")
        X = self._frame(records, self.flood_features, categorical=False)
        return self._flood_booster.predict(xgb.DMatrix(X))

    def predict_landslide(self, records: list[dict]) -> np.ndarray:
        if not self.landslide_ready or self._landslide_booster is None:
            raise ArtifactError("Landslide model is not loaded; see /health for details.")
        X = self._frame(records, self.landslide_features, categorical=True)
        return self._landslide_booster.predict(xgb.DMatrix(X, enable_categorical=True))

    # -----------------------------------------------------------------
    # Metadata exposure (no recomputation)
    # -----------------------------------------------------------------
    def performance(self) -> dict:
        """Metrics exactly as computed by the training scripts."""
        out: dict[str, Any] = {}
        for name, res in (("flood", self.flood_results),
                          ("landslide", self.landslide_results)):
            if not res:
                out[name] = None
                continue
            metrics = res.get("metrics", {})
            out[name] = {
                "model_version": res.get("version"),
                "target": res.get("features", {}).get("target"),
                "decision_threshold": res.get("decision_threshold"),
                "best_iteration": (res.get("best_iteration")
                                   or res.get("early_stopping", {}).get("best_iteration")),
                "class_imbalance_handling": res.get("class_imbalance_handling"),
                "splits": {k: metrics.get(k) for k in ("train", "validation", "test")},
                "calibration_summary": res.get("calibration_summary"),
                "split_periods": {
                    "train": res.get("split", {}).get("train_date_range"),
                    "validation": res.get("split", {}).get("validation_date_range"),
                    "test": res.get("split", {}).get("test_date_range"),
                },
            }
        return out

    def feature_importance(self) -> dict:
        """Feature importance exactly as computed during training."""
        return {
            "flood": {
                "features": self.flood_features,
                "importance": self.flood_results.get("feature_importance", []),
                "categorical_features": [],
            },
            "landslide": {
                "features": self.landslide_features,
                "importance": self.landslide_results.get("feature_importance", []),
                "categorical_features": list(self.category_schema),
                "category_schema": self.category_schema,
            },
        }

    def feature_contract(self) -> dict:
        """The exact input contract each model expects."""
        return {
            "flood": {
                "feature_order": self.flood_features,
                "n_features": len(self.flood_features),
                "common_features": self.flood_meta.get("common_features", []),
                "model_specific_features": self.flood_meta.get("flood_specific_features", []),
                "categorical_features": [],
                "decision_threshold": self.flood_threshold,
            },
            "landslide": {
                "feature_order": self.landslide_features,
                "n_features": len(self.landslide_features),
                "common_features": self.landslide_meta.get("common_features", []),
                "model_specific_features": self.landslide_meta.get(
                    "landslide_specific_features", []),
                "categorical_features": list(self.category_schema),
                "category_schema": self.category_schema,
                "decision_threshold": self.landslide_threshold,
            },
        }

    @property
    def ready(self) -> bool:
        return self.flood_ready and self.landslide_ready and self.dataset_ready
