"""
Model service -- loads the two frozen XGBoost models ONCE at startup.

Responsibilities:
  * load flood + landslide boosters and their feature metadata
  * validate incoming observations against the saved feature lists
  * produce probabilities and predictions using the SAVED thresholds

It does not train, tune, or modify anything. Feature order, categorical
schema and decision threshold all come from the artifacts on disk, so
the API can never drift from the models it serves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.api.types import CategoricalDtype

from .. import config


class ArtifactError(RuntimeError):
    """Raised when a required model artifact is missing or unreadable."""


def _read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise ArtifactError(
            f"Required artifact '{label}' not found at {path}. "
            "Run the ML pipeline (ml/flood_xgboost.py, ml/landslide_xgboost.py, "
            "ml/tcdl_v1.py) before starting the API.")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Artifact '{label}' at {path} is malformed JSON: {exc}") from exc


class ModelService:
    """Holds both frozen boosters and their metadata for the process lifetime."""

    def __init__(self) -> None:
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

        self._flood_booster: xgb.Booster | None = None
        self._landslide_booster: xgb.Booster | None = None
        self.master: pd.DataFrame | None = None

        self._load_flood()
        self._load_landslide()
        self._load_master_dataset()

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------
    def _load_flood(self) -> None:
        try:
            self.flood_meta = _read_json(config.FLOOD_FEATURES, "flood_features")
            self.flood_results = _read_json(config.FLOOD_RESULTS, "flood_model_results")
            self.flood_features = list(self.flood_meta["feature_order"])
            self.flood_threshold = float(self.flood_meta.get("decision_threshold", 0.5))
            if not config.FLOOD_MODEL.exists():
                raise ArtifactError(f"Flood model not found at {config.FLOOD_MODEL}")
            booster = xgb.Booster()
            booster.load_model(str(config.FLOOD_MODEL))
            self._flood_booster = booster
            self.flood_ready = True
        except Exception as exc:                       # surfaced via /health
            self.errors.append(f"flood model: {exc}")

    def _load_landslide(self) -> None:
        try:
            self.landslide_meta = _read_json(config.LANDSLIDE_FEATURES, "landslide_features")
            self.landslide_results = _read_json(config.LANDSLIDE_RESULTS,
                                                "landslide_model_results")
            self.landslide_features = list(self.landslide_meta["feature_order"])
            self.landslide_threshold = float(
                self.landslide_meta.get("decision_threshold", 0.5))
            self.category_schema = dict(
                self.landslide_meta["categorical_handling"]["category_schema"])
            if not config.LANDSLIDE_MODEL.exists():
                raise ArtifactError(f"Landslide model not found at {config.LANDSLIDE_MODEL}")
            booster = xgb.Booster()
            booster.load_model(str(config.LANDSLIDE_MODEL))
            self._landslide_booster = booster
            self.landslide_ready = True
        except Exception as exc:
            self.errors.append(f"landslide model: {exc}")

    def _load_master_dataset(self) -> None:
        try:
            if not config.MASTER_DATASET.exists():
                raise ArtifactError(f"Master dataset not found at {config.MASTER_DATASET}")
            df = pd.read_csv(config.MASTER_DATASET, parse_dates=["date"])
            df["location_id"] = (df["latitude"].astype(str) + "_"
                                 + df["longitude"].astype(str))
            self.master = df.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
            self.dataset_ready = True
        except Exception as exc:
            self.errors.append(f"master dataset: {exc}")

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

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def _frame(self, records: list[dict], features: list[str],
               categorical: bool) -> pd.DataFrame:
        X = pd.DataFrame(records)[features].copy()
        if categorical:
            for col, cats in self.category_schema.items():
                if col in X.columns:
                    X[col] = X[col].astype(CategoricalDtype(categories=cats, ordered=False))
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
