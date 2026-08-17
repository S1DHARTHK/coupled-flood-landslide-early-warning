"""
TCDL service -- serving layer for the existing Temporal Coupled
Decision Layer.

Design rule for this file: it does NOT reimplement TCDL. It imports
`ml/tcdl_v1.py` and calls that module's own `build_temporal_signals`
and `apply_tcdl_rules`, so trend maths and rule conditions exist in
exactly one place. If the ML layer changes, this service follows
automatically.

Thresholds are read from `tcdl_results.json` (`resolved_thresholds`) --
the values the ML run actually used. They are NOT recomputed here:
recomputing on request data would silently change TCDL behaviour.

Two data paths are served:
  * PRECOMPUTED -- the pipeline's own outputs (tcdl_timeseries.csv,
    tcdl_warnings.csv, tcdl_lead_time.csv). Used by the read endpoints.
  * ON-DEMAND -- a caller-supplied observation series scored through the
    frozen models and the imported TCDL functions.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config
from .model_service import ArtifactError, ModelService


class TCDLUnavailable(RuntimeError):
    """Raised when the TCDL layer or its outputs cannot be served."""


class InsufficientHistory(ValueError):
    """Raised when a caller supplies too short a series for trend signals."""


def _load_tcdl_module(path: Path):
    """Import ml/tcdl_v1.py by file path, without modifying the ml/ package."""
    if not path.exists():
        raise TCDLUnavailable(f"TCDL implementation not found at {path}")
    spec = importlib.util.spec_from_file_location("tcdl_v1", path)
    if spec is None or spec.loader is None:
        raise TCDLUnavailable(f"Could not load TCDL module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # main() is __main__-guarded
    return module


# Presentation-only mapping from the rules that TCDL already fired to the
# four warning categories the architecture defines. This classifies an
# existing decision -- it never changes whether a warning is raised, and
# it introduces no threshold of its own.
FLOOD_RULES = {"R1_FLOOD_LEVEL", "R4_RISING_FLOOD"}
LANDSLIDE_RULES = {"R2_LANDSLIDE_LEVEL", "R5_RISING_LANDSLIDE"}
JOINT_RULES = {"R3_SUSTAINED_JOINT", "R6_ENV_PRECURSOR", "R7_JOINT_MODERATE"}

WARNING_TYPES = ["No Warning", "Flood Warning", "Landslide Warning",
                 "Coupled Hazard Warning"]


def classify_warning(triggered: list[str], is_warning: bool) -> str:
    if not is_warning or not triggered:
        return "No Warning"
    fired = set(triggered)
    if fired & JOINT_RULES:
        return "Coupled Hazard Warning"
    if (fired & FLOOD_RULES) and (fired & LANDSLIDE_RULES):
        return "Coupled Hazard Warning"
    if fired & FLOOD_RULES:
        return "Flood Warning"
    if fired & LANDSLIDE_RULES:
        return "Landslide Warning"
    return "No Warning"


class TCDLService:
    def __init__(self, models: ModelService) -> None:
        self.errors: list[str] = []
        self.ready = False
        self.module = None
        self.results: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}
        self.rules: dict[str, str] = {}
        self.timeseries: pd.DataFrame | None = None
        self.warnings: pd.DataFrame | None = None
        self.lead_time: pd.DataFrame | None = None
        self.models = models

        try:
            self.module = _load_tcdl_module(config.TCDL_MODULE)
            self._load_results()
            self._load_outputs()
            self.ready = True
        except Exception as exc:
            self.errors.append(f"tcdl: {exc}")

    # -----------------------------------------------------------------
    def _load_results(self) -> None:
        if not config.TCDL_RESULTS.exists():
            raise TCDLUnavailable(
                f"TCDL results not found at {config.TCDL_RESULTS}. "
                "Run ml/tcdl_v1.py before starting the API.")
        with open(config.TCDL_RESULTS, encoding="utf-8") as f:
            self.results = json.load(f)
        params = self.results.get("parameters", {})
        self.thresholds = dict(params.get("resolved_thresholds", {}))
        if not self.thresholds:
            raise TCDLUnavailable(
                "tcdl_results.json contains no resolved_thresholds; the API refuses "
                "to invent thresholds. Re-run ml/tcdl_v1.py.")
        self.rules = dict(self.results.get("rules", {}))

    def _load_outputs(self) -> None:
        def read(path: Path, label: str) -> pd.DataFrame:
            if not path.exists():
                raise TCDLUnavailable(f"TCDL output '{label}' missing at {path}")
            try:
                return pd.read_csv(path)
            except Exception as exc:
                raise TCDLUnavailable(
                    f"TCDL output '{label}' at {path} is malformed: {exc}") from exc

        ts = read(config.TCDL_TIMESERIES, "tcdl_timeseries")
        ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
        if ts["date"].isna().any():
            raise TCDLUnavailable("tcdl_timeseries.csv contains unparseable dates")
        ts["tcdl_triggered_rules"] = ts["tcdl_triggered_rules"].fillna("")
        self.timeseries = ts.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)

        w = read(config.TCDL_WARNINGS, "tcdl_warnings")
        w["date"] = pd.to_datetime(w["date"], errors="coerce")
        w["tcdl_triggered_rules"] = w["tcdl_triggered_rules"].fillna("")
        self.warnings = w.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)

        self.lead_time = read(config.TCDL_LEAD_TIME, "tcdl_lead_time")

    def _require_ready(self) -> None:
        if not self.ready:
            raise TCDLUnavailable(
                "TCDL layer is unavailable: " + "; ".join(self.errors))

    # -----------------------------------------------------------------
    # Constants exposed from the ML module (never redefined here)
    # -----------------------------------------------------------------
    @property
    def min_history_rows(self) -> int:
        """Rows of daily history a series needs before trends are defined."""
        return int(self.module.SMOOTH_WINDOW + self.module.RATE_WINDOW)

    @property
    def rule_ids(self) -> list[str]:
        return list(self.module.RULE_IDS)

    # -----------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------
    def explain(self, row: pd.Series) -> dict:
        """
        Machine-readable reason for the current decision: which rules
        fired, the values they looked at, and the thresholds they used.
        """
        triggered = [r for r in self.rule_ids if bool(row.get(r, False))]
        th = self.thresholds
        baseline = float(self.module.BASELINE_THRESHOLD)

        rule_detail = []
        evidence = {
            "R1_FLOOD_LEVEL": ({"flood_probability": row.get("flood_probability")},
                               {"baseline_threshold": baseline}),
            "R2_LANDSLIDE_LEVEL": ({"landslide_probability": row.get("landslide_probability")},
                                   {"baseline_threshold": baseline}),
            "R3_SUSTAINED_JOINT": ({"coupled_prob_ma": row.get("coupled_prob_ma")},
                                   {"alert_level": th.get("alert_level"),
                                    "sustain_days": int(self.module.SUSTAIN_DAYS)}),
            "R4_RISING_FLOOD": ({"flood_prob_ma": row.get("flood_prob_ma"),
                                 "flood_prob_rate": row.get("flood_prob_rate")},
                                {"rising_flood_level": th.get("rising_flood_level"),
                                 "flood_rate": th.get("flood_rate")}),
            "R5_RISING_LANDSLIDE": ({"landslide_prob_ma": row.get("landslide_prob_ma"),
                                     "landslide_prob_rate": row.get("landslide_prob_rate")},
                                    {"rising_landslide_level": th.get("rising_landslide_level"),
                                     "landslide_rate": th.get("landslide_rate")}),
            "R6_ENV_PRECURSOR": ({"coupled_prob_ma": row.get("coupled_prob_ma"),
                                  "rainfall_rate": row.get("rainfall_rate"),
                                  "soil_moisture_rate": row.get("soil_moisture_rate")},
                                 {"watch_level": th.get("watch_level"),
                                  "rain_rate": th.get("rain_rate"),
                                  "soil_rate": th.get("soil_rate")}),
            "R7_JOINT_MODERATE": ({"flood_prob_ma": row.get("flood_prob_ma"),
                                   "landslide_prob_ma": row.get("landslide_prob_ma")},
                                  {"joint_flood_level": th.get("joint_flood_level"),
                                   "joint_landslide_level": th.get("joint_landslide_level")}),
        }
        for rid in triggered:
            values, thresholds = evidence.get(rid, ({}, {}))
            rule_detail.append({
                "rule_id": rid,
                "description": self.rules.get(rid, ""),
                "rule_class": ("baseline_replicating"
                               if rid in self.results.get("rule_classification", {})
                               .get("baseline_replicating", []) else "coupling"),
                "values": {k: _num(v) for k, v in values.items()},
                "thresholds": {k: _num(v) for k, v in thresholds.items()},
            })

        return {
            "triggered_rules": triggered,
            "n_rules_triggered": len(triggered),
            "rule_details": rule_detail,
            "summary": (
                "No TCDL rule condition was met at this timestep."
                if not triggered else
                f"{len(triggered)} rule(s) met their conditions: " + ", ".join(triggered)),
        }

    # -----------------------------------------------------------------
    # Precomputed reads
    # -----------------------------------------------------------------
    def locations(self) -> list[dict]:
        self._require_ready()
        assert self.timeseries is not None
        g = (self.timeseries.groupby(["location_id", "latitude", "longitude"])
             .agg(n_records=("date", "size"),
                  first_date=("date", "min"), last_date=("date", "max"))
             .reset_index())
        return [{"location_id": r.location_id, "latitude": float(r.latitude),
                 "longitude": float(r.longitude), "n_records": int(r.n_records),
                 "first_date": r.first_date.strftime("%Y-%m-%d"),
                 "last_date": r.last_date.strftime("%Y-%m-%d")} for r in g.itertuples()]

    def timeseries_slice(self, location_id: str | None = None,
                         split: str | None = None,
                         start: str | None = None, end: str | None = None,
                         limit: int | None = None) -> pd.DataFrame:
        self._require_ready()
        assert self.timeseries is not None
        df = self.timeseries
        if location_id:
            df = df[df["location_id"] == location_id]
            if df.empty:
                raise KeyError(f"Unknown location_id '{location_id}'")
        if split:
            df = df[df["split"] == split]
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        df = df.sort_values(["date", "latitude", "longitude"])
        if limit:
            df = df.tail(limit)
        return df

    # -----------------------------------------------------------------
    # On-demand scoring of a caller-supplied series
    # -----------------------------------------------------------------
    def evaluate_series(self, records: list[dict]) -> pd.DataFrame:
        """
        Score an ordered observation series through the frozen models and
        the imported TCDL functions.

        A series is required, not a single row: the trend signals need
        SMOOTH_WINDOW days of moving average plus RATE_WINDOW days of
        rate-of-change. Fabricating that history would be inventing data,
        so a short series is rejected instead.
        """
        self._require_ready()
        need = self.min_history_rows
        if len(records) < need:
            raise InsufficientHistory(
                f"TCDL needs at least {need} consecutive daily observations for one "
                f"location to derive trend signals ({self.module.SMOOTH_WINDOW}-day "
                f"moving average + {self.module.RATE_WINDOW}-day rate of change); "
                f"{len(records)} supplied. The API will not fabricate history.")

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].duplicated().any():
            raise InsufficientHistory("Duplicate dates in the supplied series.")
        df = df.sort_values("date").reset_index(drop=True)
        if df[["latitude", "longitude"]].drop_duplicates().shape[0] != 1:
            raise InsufficientHistory(
                "All observations in one request must belong to a single location; "
                "trend signals are computed per location.")
        df["location_id"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

        # Frozen models -> probabilities (independent, no cross-feeding).
        df["flood_probability"] = self.models.predict_flood(records)
        df["landslide_probability"] = self.models.predict_landslide(records)

        # Trend signals and rules come from the ML module itself. Its
        # functions print a progress banner intended for the CLI pipeline;
        # that is swallowed here so it does not pollute request logs. The
        # computation is untouched.
        with contextlib.redirect_stdout(io.StringIO()):
            df = self.module.build_temporal_signals(df)
            df = self.module.apply_tcdl_rules(df, self.thresholds)
        return df


def _num(v: Any) -> Any:
    """JSON-safe float conversion; NaN becomes None rather than a fake 0."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if pd.isna(f) else round(f, 6)
