"""
SHAP service -- serving layer for the existing SHAP explainability module.

Design rule (identical to tcdl_service): this file does NOT compute SHAP.
It imports `scripts/analysis/shap_explainability.py` and calls that module's own
`global_importance` and `explain_prediction`, so the SHAP mathematics
live in exactly one place. If the ML layer changes, this service follows
automatically.

Nothing here retrains a model, alters a prediction, or feeds SHAP into
TCDL. SHAP is an explanation layer only; this service merely exposes it.

The module is bound to one dataset with its own `configure()` ("real" ->
artifacts/*, "synthetic" -> synthetic/*), chosen by the model set the service is
built for, so the explained models are the ones the dashboard shows.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .. import config

VALID_MODELS = ("flood", "landslide")


class SHAPUnavailable(RuntimeError):
    """Raised when the SHAP module or its inputs cannot be served."""


def _load_shap_module(path: Path):
    """Import scripts/analysis/shap_explainability.py by path, without modifying it."""
    if not path.exists():
        raise SHAPUnavailable(f"SHAP module not found at {path}")
    spec = importlib.util.spec_from_file_location("shap_explainability", path)
    if spec is None or spec.loader is None:
        raise SHAPUnavailable(f"Could not load SHAP module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Importing shap is heavy but side-effect free (main() is __main__-guarded).
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


class SHAPService:
    def __init__(self, dataset: str = "synthetic",
                 resolve_location: Callable[[str], str] | None = None) -> None:
        self.errors: list[str] = []
        self.ready = False
        self.module = None
        self.dataset = dataset
        # Maps a district name to its location_id (identifier lookup only).
        self._resolve = resolve_location or (lambda loc: loc)
        self._full: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None
        self._global_cache: dict[str, list[dict]] = {}

        try:
            self.module = _load_shap_module(config.SHAP_MODULE)
            self.module.configure(dataset)
            # Warm the frozen models once; also proves they load.
            for m in VALID_MODELS:
                self.module.get_model(m)
            self.ready = True
        except Exception as exc:  # surfaced via /health-style status
            self.errors.append(f"shap: {exc}")

    def _require_ready(self) -> None:
        if not self.ready:
            raise SHAPUnavailable("SHAP layer unavailable: " + "; ".join(self.errors))

    @staticmethod
    def _check_model(model: str) -> None:
        if model not in VALID_MODELS:
            raise KeyError(f"Unknown model '{model}'. Expected one of {list(VALID_MODELS)}.")

    def _test_split(self) -> pd.DataFrame:
        if self._test is None:
            df = self.module.load_test_split().copy()
            df["location_id"] = (
                df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
            )
            self._test = df
        return self._test

    def _full_dataset(self) -> pd.DataFrame:
        """Every row of the module's dataset (for explaining a chosen date)."""
        if self._full is None:
            df = pd.read_csv(self.module.DATA_PATH, parse_dates=["date"])
            df["location_id"] = (
                df["latitude"].astype(str) + "_" + df["longitude"].astype(str)
            )
            self._full = df.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
        return self._full

    @property
    def source(self) -> str:
        try:
            return str(Path(self.module.DATA_PATH).relative_to(config.PROJECT_ROOT)).replace("\\", "/")
        except Exception:
            return "scripts/analysis/shap_explainability.py"

    # -----------------------------------------------------------------
    # Global mean|SHAP| importance
    # -----------------------------------------------------------------
    def global_importance(self, model: str) -> dict[str, Any]:
        self._require_ready()
        self._check_model(model)
        if model not in self._global_cache:
            self._global_cache[model] = self.module.global_importance(model)
        rows = self._global_cache[model]
        em = self.module.get_model(model)
        return {
            "model": model,
            "model_label": em.label,
            "n_features": len(em.features),
            "categorical_features": em.categorical_features,
            "shap_space": "log-odds (margin)",
            "importance": rows,
        }

    # -----------------------------------------------------------------
    # Per-prediction explanation for a location's current (latest) row
    # -----------------------------------------------------------------
    def explain_current(self, model: str, location_id: str,
                        mode: str = "current", date: str | None = None) -> dict[str, Any]:
        """
        Explain one real row for a location, scored by the same frozen model
        the pipeline uses -- so the explained probability matches what the
        pipeline would display for that row.

        mode="current": the location's latest available row (what the
            dashboard shows as "current"). In this dataset that is a
            dry-season day, so contributions are mostly downward.
        mode="peak": the location's highest model-probability row in the
            evaluation split. A real observed row -- nothing is fabricated --
            chosen so the explanation shows a meaningful high-risk case with
            features pushing the prediction higher.
        date (optional, mode="current" only): explain the location's row for
            that day instead of the latest one -- the day the dashboard is
            showing when it is replaying history. Any split may be chosen;
            `sample.split` says whether the day was seen in training.
        """
        self._require_ready()
        self._check_model(model)
        if mode not in ("current", "peak"):
            raise ValueError(f"mode must be 'current' or 'peak', got '{mode}'")
        location_id = self._resolve(location_id)
        test = self._test_split()
        loc = test[test["location_id"] == location_id]
        if loc.empty:
            raise KeyError(f"Unknown location_id '{location_id}'")

        loc = loc.sort_values("date")
        if date is not None and mode == "current":
            full = self._full_dataset()
            day = pd.Timestamp(date)
            row = full[(full["location_id"] == location_id) & (full["date"] == day)]
            if row.empty:
                raise KeyError(f"No observation for '{location_id}' on {date}")
        elif mode == "peak":
            em = self.module.get_model(model)
            proba = em.predict_proba(em.frame(loc))
            row = loc.iloc[[int(proba.argmax())]]
        else:
            row = loc.iloc[[-1]]

        out = self.module.explain_prediction(model, row)
        ts = row.iloc[0]
        out["sample"] = {
            "location_id": location_id,
            "mode": mode,
            "date": pd.Timestamp(ts["date"]).strftime("%Y-%m-%d"),
            "latitude": float(ts["latitude"]),
            "longitude": float(ts["longitude"]),
            "district": (str(ts["district"]) if "district" in row.columns
                         and pd.notna(ts["district"]) else None),
            "split": self._split_of(pd.Timestamp(ts["date"])),
            "actual_label": int(ts[model]) if model in row.columns else None,
        }
        return out

    def _split_of(self, day: pd.Timestamp) -> str | None:
        split = self.module.get_model("flood").results.get("split", {})
        for name in ("train", "validation", "test"):
            rng = split.get(f"{name}_date_range")
            if rng and pd.Timestamp(rng[0]) <= day <= pd.Timestamp(rng[1]):
                return name
        return None

    def available_location_ids(self) -> list[str]:
        self._require_ready()
        return sorted(self._test_split()["location_id"].unique().tolist())
