"""
FastAPI backend for the Flood-Landslide Early Warning system.

This is the SERVING layer only. It loads the frozen artifacts produced by
the ML pipeline once at startup and exposes them. It does not train,
tune, re-threshold, or re-implement any ML logic:

    REAL pipeline (default)   -- artifacts/ (+ dataset/)
        Flood / Landslide XGBoost V1.0 serve POST /predict/flood and
        /predict/landslide; TCDL V1.0 outputs (tcdl_*.csv / .json) and the
        SHAP layer serve the dashboard: /current, /trends, /locations,
        /warnings, /leadtime, /evaluation, /predict/coupled, /explain/*.
        Real district observations: GET /districts/{district}/observation.
    SYNTHETIC pipeline        -- synthetic/
        kept for reference: /models/performance?model_set=synthetic, or the
        whole dashboard with CAPSTONE_DASHBOARD_DATA=synthetic.
    TCDL V1.0 logic is imported from scripts/models/tcdl_v1.py; thresholds are read from
    the served set's tcdl_results.json (never recomputed here).

    Retraining = re-run scripts/models/ and scripts/analysis/ (default --dataset real); the
    backend picks the new files up on restart with no code change.

Run:
    uvicorn backend.main:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date as date_type
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .schemas import (
    CoupledWarningResponse, HazardPrediction, HazardPredictionResponse,
    HealthResponse, LeadTimeResponse, ModelStatusResponse, TrendSignals,
    WarningExplanation, build_observation_model,
)
from .services.district_service import DistrictService
from .services.model_service import ArtifactError, ModelService
from .services.shap_service import SHAPService, SHAPUnavailable
from .services.tcdl_service import (
    InsufficientHistory, TCDLService, TCDLUnavailable, WARNING_TYPES, classify_warning,
)

LEAD_TIME_UNIT_NOTE = (
    "Lead-time fields named *_hours are DATE-QUANTISED: the data has daily resolution, so "
    "each value is a whole number of days x 24, not an hour-level timing. Divide by 24 for "
    "days. Names are kept for API compatibility.")

CALIBRATION_NOTE = (
    "Both models were trained with scale_pos_weight, so probabilities are "
    "deliberately inflated relative to the observed event rate and are NOT "
    "calibrated. Probability calibration is a known open item and is out of "
    "scope for V1.0. Rank and threshold comparisons are valid; reading a "
    "probability as a literal likelihood is not."
)

# ---------------------------------------------------------------------
# Service singletons, built once at startup
# ---------------------------------------------------------------------
STATE: dict[str, Any] = {"models": None, "real_models": None, "dashboard": None,
                         "districts": None, "tcdl": None, "shap": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Synthetic set: reference only (?model_set=synthetic), unless the
    # dashboard is explicitly switched back to it.
    models = ModelService(config.SYNTHETIC_MODEL_SET)
    # Real set: the single-hazard prediction endpoints (and, by default,
    # the dashboard).
    real_models = ModelService(config.REAL_MODEL_SET)
    dashboard = real_models if config.DASHBOARD_MODEL_SET == "real" else models
    districts = DistrictService(real_models)
    tcdl = TCDLService(dashboard,
                       district_alias=(districts.resolve if districts.ready else None))
    shap_service = SHAPService(
        dashboard.name, resolve_location=(tcdl.resolve_location if tcdl.ready else None))
    STATE["models"] = models
    STATE["real_models"] = real_models
    STATE["dashboard"] = dashboard
    STATE["districts"] = districts
    STATE["tcdl"] = tcdl
    STATE["shap"] = shap_service

    # Request schemas are built from the loaded feature contracts, so the
    # OpenAPI documentation always matches the served models.
    if real_models.flood_ready:
        app.state.FloodObservation = build_observation_model(
            "FloodObservation", real_models.flood_features, {},
            "One observation for the real Flood XGBoost V1.0 model.",
            nullable=real_models.nullable_features["flood"], with_district=True)
    if real_models.landslide_ready:
        app.state.LandslideObservation = build_observation_model(
            "LandslideObservation", real_models.landslide_features,
            real_models.category_schema,
            "One observation for the real Landslide XGBoost V1.0 model.",
            nullable=real_models.nullable_features["landslide"], with_district=True)
    yield
    STATE.clear()


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    lifespan=lifespan,
    description=(
        "Serving layer for the coupled flood-landslide early warning ML pipeline.\n\n"
        "**Prediction endpoints** (`/predict/flood`, `/predict/landslide`) use the "
        "REAL-data XGBoost models in `artifacts/models/`; real district observations come "
        "from `/districts`. "
        f"{config.REAL_MODEL_NOTE}\n\n"
        f"**Dashboard, TCDL and SHAP endpoints** serve the `{config.DASHBOARD_MODEL_SET}` "
        f"pipeline: {config.data_notice(config.DATA_MODE)}\n\n"
        "Every location-bearing endpoint accepts a district name (e.g. `Idukki`) "
        "wherever it accepts a `location_id`.\n\n"
        "Architecture: two independent XGBoost classifiers feed a rule-based "
        "Temporal Coupled Decision Layer (TCDL). This API exposes their existing "
        "outputs; it contains no ML logic of its own."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_models() -> ModelService:
    """Synthetic (reference) model set."""
    m = STATE.get("models")
    if m is None:
        raise HTTPException(503, "Model service not initialised.")
    return m


def get_dashboard_models() -> ModelService:
    """Model set behind TCDL, SHAP and the dashboard endpoints."""
    m = STATE.get("dashboard")
    if m is None:
        raise HTTPException(503, "Dashboard model service not initialised.")
    return m


def get_real_models() -> ModelService:
    """Real model set (single-hazard prediction endpoints)."""
    m = STATE.get("real_models")
    if m is None:
        raise HTTPException(503, "Real model service not initialised.")
    return m


def get_model_set(name: str) -> ModelService:
    if name == "real":
        return get_real_models()
    if name == "synthetic":
        return get_models()
    raise HTTPException(422, {"error": f"Unknown model_set '{name}'",
                              "available": ["real", "synthetic"]})


def get_districts() -> DistrictService:
    d = STATE.get("districts")
    if d is None:
        raise HTTPException(503, "District service not initialised.")
    if not d.ready:
        raise HTTPException(503, {"error": "District data unavailable", "details": d.errors})
    return d


def _rel(path) -> str:
    try:
        return str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _warning_for(m: ModelService) -> str | None:
    return config.SYNTHETIC_WARNING if m.name == "synthetic" else None


def _data_fields() -> dict[str, Any]:
    """data_mode + notices every dashboard response carries."""
    return {"data_mode": config.DATA_MODE,
            "synthetic_data_warning": (config.SYNTHETIC_WARNING
                                       if config.DATA_MODE == "synthetic" else None),
            "data_notice": config.data_notice(config.DATA_MODE)}


def _date(value: str | None, name: str) -> str | None:
    """Validate an optional YYYY-MM-DD query value (422 on anything else)."""
    if value is None or value == "":
        return None
    try:
        return date_type.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise HTTPException(422, {"error": f"Invalid {name}",
                                  "detail": f"{name} must be YYYY-MM-DD, got {value!r}"}) from exc


def _district(row: pd.Series) -> str | None:
    v = row.get("district")
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def get_tcdl() -> TCDLService:
    t = STATE.get("tcdl")
    if t is None:
        raise HTTPException(503, "TCDL service not initialised.")
    if not t.ready:
        raise HTTPException(503, {"error": "TCDL unavailable", "details": t.errors})
    return t


def get_shap() -> SHAPService:
    sv = STATE.get("shap")
    if sv is None:
        raise HTTPException(503, "SHAP service not initialised.")
    if not sv.ready:
        # 503 is deliberate: the dashboard treats this as "explanation
        # unavailable" and keeps working; it never fabricates SHAP values.
        raise HTTPException(503, {"error": "SHAP unavailable", "details": sv.errors})
    return sv


def num(v: Any) -> Any:
    """JSON-safe numeric: NaN/NaT become null rather than a fabricated value."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if pd.isna(f) else round(f, 6)


def trend_signals(row: pd.Series) -> TrendSignals:
    fields = TrendSignals.model_fields
    return TrendSignals(**{k: num(row.get(k)) for k in fields})


def _env_block(models: ModelService, row: pd.Series) -> dict:
    """
    Current environmental conditions from the master dataset.

    Any column absent from the dataset is reported as null with an
    explicit note -- values are never invented.
    """
    common = ["rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_14d_mm",
              "rainfall_30d_mm", "soil_moisture", "temperature_c", "humidity_percent",
              "elevation_m"]
    flood_specific = ["river_level_m", "distance_to_river_km", "drainage_density"]
    landslide_specific = ["slope_degree", "aspect_degree", "curvature",
                          "land_cover", "lithology"]

    def block(cols: list[str]) -> dict:
        out = {}
        for c in cols:
            if c in row.index and pd.notna(row[c]):
                out[c] = row[c] if isinstance(row[c], str) else num(row[c])
            else:
                out[c] = None
        return out

    unavailable = [c for c in common + flood_specific + landslide_specific
                   if c not in row.index or pd.isna(row.get(c))]
    return {
        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        "latitude": num(row["latitude"]),
        "longitude": num(row["longitude"]),
        "district": _district(row),
        "common_environmental": block(common),
        "flood_specific": block(flood_specific),
        "landslide_specific": block(landslide_specific),
        "unavailable_fields": unavailable,
        "note": ("Fields listed in unavailable_fields have no value in the dataset for "
                 "this day (e.g. river_level_m where no CWC gauge reading exists) and "
                 "are returned as null. No value is ever substituted."),
    }


# ---------------------------------------------------------------------
# 1. System status
# ---------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"],
         summary="Liveness and readiness of the serving layer")
def health():
    m = get_models()
    r = STATE.get("real_models")
    dash = STATE.get("dashboard")
    d = STATE.get("districts")
    t = STATE.get("tcdl")
    sv = STATE.get("shap")
    errors = (list(m.errors)
              + (list(r.errors) if r else ["real model service not initialised"])
              + (list(d.errors) if d else ["district service not initialised"])
              + (list(t.errors) if t else ["tcdl service not initialised"])
              + (list(sv.errors) if sv else ["shap service not initialised"]))
    real_ready = bool(r and r.flood_ready and r.landslide_ready)
    all_ready = (real_ready and m.ready and bool(dash and dash.ready)
                 and bool(t and t.ready) and bool(d and d.ready) and bool(sv and sv.ready))
    return HealthResponse(
        status="ok" if all_ready else "degraded",
        api_version=config.API_VERSION,
        data_mode=config.DATA_MODE,
        synthetic_data_warning=(config.SYNTHETIC_WARNING
                                if config.DATA_MODE == "synthetic" else None),
        ready_for_prediction=real_ready,
        flood_model_loaded=bool(r and r.flood_ready),
        landslide_model_loaded=bool(r and r.landslide_ready),
        tcdl_available=bool(t and t.ready),
        master_dataset_loaded=bool(dash and dash.dataset_ready),
        errors=errors,
        prediction_model_set=config.PREDICTION_MODEL_SET,
        synthetic_models_loaded=m.flood_ready and m.landslide_ready,
        districts_loaded=bool(d and d.ready),
        dashboard_model_set=config.DASHBOARD_MODEL_SET,
        shap_available=bool(sv and sv.ready),
        data_notice=config.data_notice(config.DATA_MODE),
    )


@app.get("/models/status", response_model=ModelStatusResponse, tags=["system"],
         summary="Detailed status of both models and the TCDL layer")
def models_status():
    m = get_models()
    r = get_real_models()
    dash = get_dashboard_models()
    d = STATE.get("districts")
    t = STATE.get("tcdl")
    sv = STATE.get("shap")
    real_ready = r.flood_ready and r.landslide_ready
    ready = (real_ready and m.ready and dash.ready and bool(t and t.ready)
             and bool(d and d.ready) and bool(sv and sv.ready))
    artifacts = {k: {"path": str(p.relative_to(config.PROJECT_ROOT))
                     if p.is_relative_to(config.PROJECT_ROOT) else str(p),
                     "exists": p.exists()}
                 for k, p in config.REQUIRED_ARTIFACTS.items()}
    return ModelStatusResponse(
        status="ok" if ready else "degraded",
        data_mode=config.DATA_MODE,
        ready_for_prediction=real_ready,
        versions=config.PIPELINE_VERSIONS,
        # flood_model / landslide_model describe the models behind the
        # prediction endpoints, i.e. the REAL set.
        flood_model={
            "loaded": r.flood_ready, "version": r.flood_meta.get("version"),
            "n_features": len(r.flood_features), "n_trees": r.flood_meta.get("n_trees"),
            "decision_threshold": r.flood_threshold, "target": r.flood_meta.get("target"),
            "retrained_by_api": False, "model_set": r.name,
            "artifact": _rel(config.REAL_FLOOD_MODEL),
            "feature_order": r.flood_features,
            "nullable_features": sorted(r.nullable_features["flood"]),
        },
        landslide_model={
            "loaded": r.landslide_ready, "version": r.landslide_meta.get("version"),
            "n_features": len(r.landslide_features),
            "n_trees": r.landslide_meta.get("n_trees"),
            "decision_threshold": r.landslide_threshold,
            "target": r.landslide_meta.get("target"),
            "categorical_features": list(r.category_schema),
            "retrained_by_api": False, "model_set": r.name,
            "artifact": _rel(config.REAL_LANDSLIDE_MODEL),
            "feature_order": r.landslide_features,
            "nullable_features": sorted(r.nullable_features["landslide"]),
        },
        prediction_model_set=config.PREDICTION_MODEL_SET,
        synthetic_models={
            "flood_loaded": m.flood_ready, "landslide_loaded": m.landslide_ready,
            "flood_artifact": _rel(config.FLOOD_MODEL),
            "landslide_artifact": _rel(config.LANDSLIDE_MODEL),
            "used_by": (["/models/performance?model_set=synthetic",
                         "/models/features?model_set=synthetic"]
                        + (["/predict/coupled (TCDL)", "/current", "/trends", "/warnings",
                            "/leadtime", "/evaluation", "/explain/*"]
                           if dash.name == "synthetic" else [])),
            "reason": ("Reference only; switch the dashboard back with "
                       "CAPSTONE_DASHBOARD_DATA=synthetic."),
        },
        dashboard_model_set=dash.name,
        shap={
            "available": bool(sv and sv.ready),
            "dataset": (sv.dataset if sv else None),
            "model_set": dash.name,
            "precomputed_outputs": _rel(dash.paths["shap_output_dir"]),
            "errors": (list(sv.errors) if sv else ["shap service not initialised"]),
        },
        districts={
            "loaded": bool(d and d.ready),
            "n_districts": len(d.points) if d and d.ready else 0,
            "source": _rel(config.REAL_MASTER_DATASET),
            "district_is_model_feature": False,
        },
        tcdl={
            "available": bool(t and t.ready),
            "version": (t.results.get("version") if t and t.ready else None),
            "layer_type": (t.results.get("layer_type") if t and t.ready else None),
            "rule_ids": (t.rule_ids if t and t.ready else []),
            "thresholds_source": f"{_rel(dash.paths['tcdl_results'])} (resolved_thresholds)",
            "thresholds_recomputed_by_api": False,
            "min_history_rows_for_trends": (t.min_history_rows if t and t.ready else None),
            "model_set": dash.name,
            "outputs": {k: _rel(dash.paths[k]) for k in
                        ("tcdl_timeseries", "tcdl_warnings", "tcdl_lead_time")},
            "date_range": (list(t.date_range()) if t and t.ready else None),
        },
        artifacts=artifacts,
        errors=(list(m.errors) + list(r.errors) + (list(d.errors) if d else [])
                + (list(t.errors) if t else []) + (list(sv.errors) if sv else [])),
    )


# ---------------------------------------------------------------------
# 2-3. Single-hazard prediction
# ---------------------------------------------------------------------
class _ObsList(BaseModel):
    observations: list[dict] = Field(..., min_length=1,
                                     description="One or more observations.")


def _resolve_districts(records: list) -> tuple[list, list[str]]:
    """
    Canonicalise an optional `district` identifier on each observation.

    `district` never reaches a model. When latitude/longitude are omitted,
    the district's representative point (the coordinates the real Master
    Dataset uses for that district) is filled in -- a lookup, not an estimate.
    """
    d = STATE.get("districts")
    out, problems = [], []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            out.append(rec)                       # reported by validate_observations
            continue
        rec = dict(rec)                           # never mutate the caller's payload
        if rec.get("district") is not None:
            if d is None or not d.ready:
                problems.append(f"record {i}: district lookup unavailable "
                                f"({'; '.join(d.errors) if d else 'not initialised'})")
            else:
                try:
                    rec["district"] = d.resolve(rec["district"])
                except KeyError as exc:
                    problems.append(f"record {i}: {exc.args[0]}")
                else:
                    if rec.get("latitude") is None and rec.get("longitude") is None:
                        rec.update(d.point(rec["district"]))
        out.append(rec)
    return out, problems


def _predict(hazard: str, payload: dict) -> HazardPredictionResponse:
    """Score observations with the REAL model set (config.PREDICTION_MODEL_SET)."""
    m = get_model_set(config.PREDICTION_MODEL_SET)
    records = payload.get("observations")
    if not isinstance(records, list) or not records:
        raise HTTPException(422, "Body must contain a non-empty 'observations' list.")

    if hazard == "flood":
        if not m.flood_ready:
            raise HTTPException(503, {"error": "Flood model not loaded", "details": m.errors})
        features, threshold, version, artifact = (
            m.flood_features, m.flood_threshold, m.flood_meta.get("version", "1.0"),
            m.paths["flood_model"])
    else:
        if not m.landslide_ready:
            raise HTTPException(503, {"error": "Landslide model not loaded",
                                      "details": m.errors})
        features, threshold, version, artifact = (
            m.landslide_features, m.landslide_threshold,
            m.landslide_meta.get("version", "1.0"), m.paths["landslide_model"])

    records, problems = _resolve_districts(records)
    problems += m.validate_observations(records, hazard)
    if problems:
        raise HTTPException(422, {"error": "Invalid input", "problems": problems,
                                  "required_features": features,
                                  "nullable_features": sorted(m.nullable_features[hazard])})

    try:
        proba = (m.predict_flood(records) if hazard == "flood"
                 else m.predict_landslide(records))
    except ArtifactError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Prediction failed for the supplied input: {exc}") from exc

    preds = []
    for rec, p in zip(records, proba):
        pred = int(float(p) >= threshold)
        preds.append(HazardPrediction(
            date=str(rec["date"]), latitude=float(rec["latitude"]),
            longitude=float(rec["longitude"]),
            probability=round(float(p), 6), prediction=pred,
            warning_status="Warning" if pred else "No Warning",
            district=rec.get("district"),
            missing_features=m.missing_features(rec, features)))

    return HazardPredictionResponse(
        hazard=hazard, model_version=str(version), decision_threshold=threshold,
        data_mode=m.name, synthetic_data_warning=_warning_for(m),
        n_observations=len(preds), predictions=preds,
        calibration_note=CALIBRATION_NOTE,
        model_set=m.name, model_artifact=_rel(artifact),
        model_note=config.REAL_MODEL_NOTE if m.name == "real" else None)


# Examples are real rows of the real Master Dataset (Ernakulam, 16 Aug 2018,
# the peak of the 2018 Kerala flood; Wayanad, 30 Jul 2024, the Mundakkai
# landslide day with no CWC river gauge -> river_level_m null).
@app.post("/predict/flood", response_model=HazardPredictionResponse, tags=["prediction"],
          summary="Flood probability and prediction from the REAL Flood XGBoost V1.0")
def predict_flood(payload: dict = Body(..., examples=[{"observations": [{
        "date": "2018-08-16", "district": "Ernakulam",
        "latitude": 10.08408, "longitude": 76.5463,
        "rainfall_1d_mm": 82.914224, "rainfall_3d_mm": 308.872165,
        "rainfall_7d_mm": 440.876561, "rainfall_14d_mm": 575.499355,
        "rainfall_30d_mm": 976.200734, "soil_moisture": 1.0, "temperature_c": 25.66,
        "humidity_percent": 90.43, "elevation_m": 152.0, "river_level_m": 11.4046,
        "distance_to_river_km": 1.116, "drainage_density": 0.4117}]}])):
    """
    Real model, exactly as trained: same feature order and 0.50 threshold.
    `district` is an identifier only. `river_level_m` may be null (no gauge /
    no reading); it is then scored by XGBoost's learned missing-value branch.
    """
    return _predict("flood", payload)


@app.post("/predict/landslide", response_model=HazardPredictionResponse,
          tags=["prediction"],
          summary="Landslide probability and prediction from the REAL Landslide XGBoost V1.0")
def predict_landslide(payload: dict = Body(..., examples=[{"observations": [{
        "date": "2018-08-16", "district": "Ernakulam",
        "latitude": 10.08408, "longitude": 76.5463,
        "rainfall_1d_mm": 82.914224, "rainfall_3d_mm": 308.872165,
        "rainfall_7d_mm": 440.876561, "rainfall_14d_mm": 575.499355,
        "rainfall_30d_mm": 976.200734, "soil_moisture": 1.0, "temperature_c": 25.66,
        "humidity_percent": 90.43, "elevation_m": 152.0,
        "slope_degree": 6.73, "aspect_degree": 214.2, "curvature": 0.0001,
        "land_cover": "forest", "lithology": "granite_gneiss"}]}])):
    """Real model, exactly as trained: same feature order, category schema and threshold."""
    return _predict("landslide", payload)


# ---------------------------------------------------------------------
# 6. Coupled TCDL warning (on-demand series)
# ---------------------------------------------------------------------
@app.post("/predict/coupled", response_model=CoupledWarningResponse, tags=["prediction"],
          summary="TCDL coupled warning for a supplied observation series")
def predict_coupled(payload: dict = Body(..., description=(
        "{'observations': [...]} -- an ordered daily series for ONE location. "
        "TCDL derives moving averages and rates of change, so a minimum history "
        "is required; see /models/status -> tcdl.min_history_rows_for_trends."))):
    """
    Runs the caller's series through the frozen models and the existing TCDL
    functions imported from `scripts/models/tcdl_v1.py`.

    A series is required rather than a single row because the trend signals are
    only defined over a window of consecutive days. Too short a series is
    rejected -- the API will not fabricate the missing history.

    The decision returned is for the LAST timestep in the series.
    """
    m, t = get_dashboard_models(), get_tcdl()
    records = payload.get("observations")
    if not isinstance(records, list) or not records:
        raise HTTPException(422, "Body must contain a non-empty 'observations' list.")

    needed = sorted(set(m.flood_features) | set(m.landslide_features))
    # Same strict validation as /predict/*: nulls only where the model was
    # trained with missing values (real flood model: river_level_m).
    records, problems = _resolve_districts(records)
    for hazard in ("flood", "landslide"):
        for problem in m.validate_observations(records, hazard):
            if problem not in problems:
                problems.append(problem)
    if problems:
        raise HTTPException(422, {"error": "Invalid input", "problems": problems,
                                  "required_features": needed,
                                  "nullable_features": sorted(
                                      m.nullable_features["flood"]
                                      | m.nullable_features["landslide"]),
                                  "min_observations": t.min_history_rows})

    try:
        scored = t.evaluate_series(records)
    except InsufficientHistory as exc:
        raise HTTPException(422, {"error": "Insufficient history", "detail": str(exc),
                                  "min_observations": t.min_history_rows}) from exc
    except ArtifactError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"TCDL evaluation failed: {exc}") from exc

    row = scored.iloc[-1]
    return _coupled_response(t, row, extra_notes=[
        "Decision corresponds to the final timestep of the supplied series.",
        f"Trend signals require {t.min_history_rows} days of history; earlier "
        "timesteps in the series may have null trend values.",
    ])


def _coupled_response(t: TCDLService, row: pd.Series,
                      extra_notes: list[str] | None = None) -> CoupledWarningResponse:
    is_warning = bool(int(row.get("tcdl_warning", 0)))
    explanation = t.explain(row)
    triggered = explanation["triggered_rules"]
    wtype = classify_warning(triggered, is_warning)
    ts = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")

    notes = [
        ("Warning type is a presentation-level classification of the rules TCDL "
         "already fired; it introduces no new threshold or decision logic."),
        ("Timestamps are daily. The dataset carries no time of day, so no "
         "hour-of-day is reported or inferred."),
        CALIBRATION_NOTE,
    ] + (extra_notes or [])

    return CoupledWarningResponse(
        **_data_fields(),
        warning_status="Warning" if is_warning else "No Warning",
        warning_type=wtype,
        warning_timestamp=ts if is_warning else None,
        timestamp_resolution="daily (24-hour quantisation)",
        location={"latitude": num(row["latitude"]), "longitude": num(row["longitude"]),
                  "location_id": row.get("location_id"), "district": _district(row)},
        flood_probability=num(row.get("flood_probability")),
        landslide_probability=num(row.get("landslide_probability")),
        coupled_probability=num(row.get("coupled_probability")),
        trends=trend_signals(row),
        explanation=WarningExplanation(**explanation),
        baselines={
            "flood_only_warning": int(row.get("flood_only_warning", 0) or 0),
            "landslide_only_warning": int(row.get("landslide_only_warning", 0) or 0),
            "tcdl_coupling_rules_only": int(row.get("tcdl_coupling_only_warning", 0) or 0),
        },
        tcdl_version=str(t.results.get("version", "1.0")),
        notes=notes,
    )


# ---------------------------------------------------------------------
# 4-5. Current conditions and trends
# ---------------------------------------------------------------------
def _location_error(t: TCDLService, exc: KeyError) -> HTTPException:
    return HTTPException(404, {"error": "Unknown location",
                               "detail": exc.args[0] if exc.args else str(exc),
                               "valid_locations": [
                                   (loc["district"] or loc["location_id"])
                                   for loc in t.locations()]})


@app.get("/current", tags=["monitoring"],
         summary="Environmental conditions, predictions, trends and warning for one day")
def current(location_id: str | None = Query(None, description=(
                "location_id or district name; omit for all locations")),
            as_of: str | None = Query(None, description=(
                "YYYY-MM-DD; defaults to the latest day in the pipeline outputs. Any "
                "earlier day replays what the system showed on that day."))):
    """
    One timestep per location from the pipeline's own outputs (the latest
    day on or before `as_of`). Nothing is recomputed here.
    """
    m, t = get_dashboard_models(), get_tcdl()
    as_of = _date(as_of, "as_of")
    try:
        df = t.timeseries_slice(location_id=location_id, end=as_of)
    except KeyError as exc:
        raise _location_error(t, exc) from exc
    if df.empty:
        first, last = t.date_range()
        raise HTTPException(404, {"error": "No records for the requested location/date",
                                  "available_date_range": [first, last]})

    assert m.master is not None
    master = m.master
    out = []
    for loc, g in df.groupby("location_id"):
        row = g.sort_values("date").iloc[-1]
        mrow = master[(master["location_id"] == loc) & (master["date"] == row["date"])]
        env = (_env_block(m, mrow.iloc[0]) if not mrow.empty
               else {"date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                     "latitude": num(row["latitude"]), "longitude": num(row["longitude"]),
                     "district": _district(row),
                     "common_environmental": {}, "flood_specific": {},
                     "landslide_specific": {},
                     "unavailable_fields": ["all"],
                     "note": "No master-dataset row matched this timestep."})
        triggered = [r for r in t.rule_ids if bool(row.get(r, False))]
        is_warning = bool(int(row.get("tcdl_warning", 0)))
        out.append({
            "location_id": loc,
            "district": _district(row),
            "split": row.get("split"),
            "environment": env,
            "flood": {"probability": num(row["flood_probability"]),
                      "prediction": int(row.get("flood_only_warning", 0)),
                      "warning_status": ("Warning" if row.get("flood_only_warning", 0)
                                         else "No Warning")},
            "landslide": {"probability": num(row["landslide_probability"]),
                          "prediction": int(row.get("landslide_only_warning", 0)),
                          "warning_status": ("Warning" if row.get("landslide_only_warning", 0)
                                             else "No Warning")},
            "trends": trend_signals(row).model_dump(),
            "tcdl": {"warning_status": "Warning" if is_warning else "No Warning",
                     "warning_type": classify_warning(triggered, is_warning),
                     "triggered_rules": triggered,
                     "warning_timestamp": (pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
                                           if is_warning else None)},
            "recorded_labels": {h: int(row[h]) for h in ("flood", "landslide")
                                if h in row.index and pd.notna(row[h])},
        })
    first, last = t.date_range()
    return {**_data_fields(),
            "as_of": pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d"),
            "available_date_range": [first, last],
            "n_locations": len(out), "locations": out}


@app.get("/trends", tags=["monitoring"],
         summary="Temporal signals consumed by TCDL (smoothed values and rates)")
def trends(location_id: str = Query(..., description=(
               "location_id or district name; trends are per location")),
           limit: int = Query(90, ge=1, le=5000, description="Most recent N days returned"),
           split: str | None = Query(None, description="train | validation | test"),
           start: str | None = Query(None, description="YYYY-MM-DD (inclusive)"),
           end: str | None = Query(None, description=(
               "YYYY-MM-DD (inclusive); the dashboard passes its as_of day"))):
    """
    Values come from the TCDL pipeline output, not recomputed here.
    Nulls mark timesteps where a rolling window is not yet full.
    """
    t = get_tcdl()
    start, end = _date(start, "start"), _date(end, "end")
    try:
        df = t.timeseries_slice(location_id=location_id, split=split,
                                start=start, end=end, limit=limit)
    except KeyError as exc:
        raise _location_error(t, exc) from exc
    if df.empty:
        raise HTTPException(404, "No records for the requested filters.")

    series = [{"date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
               **trend_signals(r).model_dump(),
               "tcdl_warning": int(r.get("tcdl_warning", 0) or 0),
               "flood": int(r.get("flood", 0) or 0),
               "landslide": int(r.get("landslide", 0) or 0)}
              for _, r in df.iterrows()]
    first = df.iloc[0]
    return {"data_mode": config.DATA_MODE,
            "location_id": first["location_id"],
            "district": _district(first),
            "n_records": len(series),
            "parameters": {k: t.results["parameters"].get(k) for k in
                           ("smooth_window_days", "rate_window_days", "sustain_days")},
            "source": f"{_rel(t.paths['tcdl_timeseries'])} (produced by scripts/models/tcdl_v1.py)",
            "series": series}


@app.get("/locations", tags=["monitoring"], summary="Locations available in the pipeline")
def locations():
    t = get_tcdl()
    first, last = t.date_range()
    return {"data_mode": config.DATA_MODE, "available_date_range": [first, last],
            "locations": t.locations()}


# ---------------------------------------------------------------------
# 9. Historical warnings
# ---------------------------------------------------------------------
@app.get("/warnings", tags=["monitoring"],
         summary="Historical warning timeline from existing TCDL outputs")
def warnings(location_id: str | None = Query(None, description="location_id or district"),
             start: str | None = Query(None, description="YYYY-MM-DD"),
             end: str | None = Query(None, description="YYYY-MM-DD"),
             split: str | None = Query(None, description="train | validation | test"),
             warning_only: bool = Query(True, description="False returns every timestep"),
             limit: int = Query(500, ge=1, le=20000, description="Most recent N records")):
    t = get_tcdl()
    start, end = _date(start, "start"), _date(end, "end")
    try:
        df = t.timeseries_slice(location_id=location_id, split=split, start=start, end=end)
    except KeyError as exc:
        raise _location_error(t, exc) from exc
    total_days = len(df)
    if warning_only:
        df = df[df["tcdl_warning"] == 1]
    n_matching = len(df)
    df = df.tail(limit)

    items = []
    for _, r in df.iterrows():
        triggered = [x for x in t.rule_ids if bool(r.get(x, False))]
        is_warning = bool(int(r.get("tcdl_warning", 0)))
        items.append({
            "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
            "latitude": num(r["latitude"]), "longitude": num(r["longitude"]),
            "location_id": r["location_id"], "district": _district(r),
            "split": r.get("split"),
            "flood_probability": num(r["flood_probability"]),
            "landslide_probability": num(r["landslide_probability"]),
            "coupled_probability": num(r["coupled_probability"]),
            "coupled_prob_ma": num(r["coupled_prob_ma"]),
            "warning_status": "Warning" if is_warning else "No Warning",
            "warning_type": classify_warning(triggered, is_warning),
            "triggered_rules": triggered,
            "warning_timestamp": (pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
                                  if is_warning else None),
            "baselines": {"flood_only": int(r.get("flood_only_warning", 0)),
                          "landslide_only": int(r.get("landslide_only_warning", 0))},
            "actual": {"flood": int(r.get("flood", 0)), "landslide": int(r.get("landslide", 0))},
        })
    return {**_data_fields(),
            "warning_types": WARNING_TYPES,
            "timestamp_resolution": "daily (24-hour quantisation)",
            "n_location_days_in_range": total_days,
            "n_matching": n_matching,
            "truncated": n_matching > len(items),
            "n_records": len(items), "records": items}


@app.get("/warnings/{location_id}/{warning_date}", response_model=CoupledWarningResponse,
         tags=["monitoring"], summary="Full coupled warning detail for one timestep")
def warning_detail(location_id: str = PathParam(..., description="location_id or district"),
                   warning_date: str = PathParam(..., description="YYYY-MM-DD")):
    t = get_tcdl()
    warning_date = _date(warning_date, "warning_date")
    try:
        df = t.timeseries_slice(location_id=location_id, start=warning_date, end=warning_date)
    except KeyError as exc:
        raise _location_error(t, exc) from exc
    if df.empty:
        raise HTTPException(404, f"No record for {location_id} on {warning_date}.")
    return _coupled_response(t, df.iloc[0])


# ---------------------------------------------------------------------
# 8. Lead time
# ---------------------------------------------------------------------
@app.get("/leadtime", response_model=LeadTimeResponse, tags=["evaluation"],
         summary="Lead-time records produced by TCDL V1.0")
def leadtime(system: str | None = Query(None, description="e.g. tcdl_coupled"),
             hazard_type: str | None = Query(None, description="flood | landslide | any_hazard"),
             location_id: str | None = Query(None, description="location_id or district"),
             detected_only: bool = Query(False),
             limit: int = Query(500, ge=1, le=5000)):
    """
    Served directly from the set's tcdl_lead_time.csv. Nothing is recomputed,
    and no hour-of-day is inferred: the underlying data is daily.
    """
    t = get_tcdl()
    assert t.lead_time is not None
    df = t.lead_time
    if system:
        df = df[df["system"] == system]
    if hazard_type:
        df = df[df["hazard_type"] == hazard_type]
    if location_id:
        try:
            df = df[df["location_id"] == t.resolve_location(location_id)]
        except KeyError as exc:
            raise _location_error(t, exc) from exc
    if detected_only:
        df = df[df["detected"] == 1]
    if df.empty:
        raise HTTPException(404, {"error": "No lead-time records match the filters",
                                  "available_systems": sorted(t.lead_time["system"].unique()),
                                  "available_hazard_types": sorted(
                                      t.lead_time["hazard_type"].unique())})
    df = df.head(limit)

    def text(v: Any) -> str | None:
        return v if isinstance(v, str) and v else None

    records = [{
        "system": r["system"], "hazard_type": r["hazard_type"],
        "location_id": r["location_id"], "district": _district(r),
        "latitude": num(r["latitude"]), "longitude": num(r["longitude"]),
        "event_time": text(r["event_timestamp"]),
        "warning_time": text(r["first_warning_timestamp"]),
        "detected": int(r["detected"]),
        # *_hours are date-quantised (whole days x 24); *_days carry the same value in
        # the data's own resolution. Both read unchanged from the TCDL output.
        "lead_time_hours": num(r["lead_time_hours_earliest"]),
        "lead_time_hours_contiguous": num(r["lead_time_hours_contiguous"]),
        "lead_time_days": num(r["lead_time_days_earliest"]),
        "lead_time_days_contiguous": num(r["lead_time_days_contiguous"]),
    } for _, r in df.iterrows()]

    return LeadTimeResponse(
        data_mode=config.DATA_MODE,
        data_notice=config.data_notice(config.DATA_MODE),
        lead_time_resolution=(t.results.get("event_source", {}).get("resolution")
                              or "daily (24-hour quantisation)"),
        lead_time_note=t.results.get("lead_time_note", ""),
        lead_time_definition=t.results.get("lead_time_definition", {}),
        event_source=t.results.get("event_source", {}),
        n_records=len(records), records=records)


# ---------------------------------------------------------------------
# 10. Evaluation / system comparison
# ---------------------------------------------------------------------
@app.get("/evaluation", tags=["evaluation"],
         summary="System comparison exactly as computed by TCDL V1.0")
def evaluation(hazard_type: str = Query("any_hazard",
                                        description="any_hazard | flood | landslide")):
    """Returns the set's tcdl_results.json values verbatim. No metric is recomputed here."""
    t = get_tcdl()
    results = t.results.get("results", {})
    if hazard_type not in results:
        raise HTTPException(404, {"error": f"Unknown hazard_type '{hazard_type}'",
                                  "available": sorted(results)})
    return {
        **_data_fields(),
        "hazard_type": hazard_type,
        "evaluation_period": t.results.get("evaluation_period"),
        "systems": results[hazard_type],
        "warning_counts": t.results.get("warning_counts"),
        "rule_firing_counts": t.results.get("rule_firing_counts"),
        "parameters": t.results.get("parameters"),
        "rules": t.results.get("rules"),
        "rule_classification": t.results.get("rule_classification"),
        "event_source": t.results.get("event_source"),
        "lead_time_unit_note": LEAD_TIME_UNIT_NOTE,
        "interpretation_notes": {
            "data": t.results.get("_warning"),
            "scope": t.results.get("_scope_note"),
            "structural_dominance": t.results.get("_structural_dominance_note"),
            "lead_time": t.results.get("lead_time_note"),
        },
        "source": _rel(t.paths["tcdl_results"]),
    }


# ---------------------------------------------------------------------
# 11-12. Model performance and feature importance
# ---------------------------------------------------------------------
_MODEL_SET_QUERY = Query(config.PREDICTION_MODEL_SET, description=(
    "real (default: the models behind /predict/* and the dashboard) | synthetic "
    "(the simulated reference models)"))


@app.get("/models/performance", tags=["evaluation"],
         summary="Stored evaluation metrics for both XGBoost models")
def models_performance(model_set: str = _MODEL_SET_QUERY):
    """Read from the training result files. Nothing is retrained or recomputed."""
    m = get_model_set(model_set)
    return {"data_mode": m.name,
            "synthetic_data_warning": _warning_for(m),
            "model_set": m.name,
            "models": m.performance(),
            "source": [_rel(m.paths["flood_results"]), _rel(m.paths["landslide_results"])],
            "note": ("Metrics are those computed during training on the frozen "
                     "chronological split. The API does not recompute them."),
            **({"model_note": config.REAL_MODEL_NOTE} if m.name == "real" else {})}


@app.get("/models/features", tags=["evaluation"],
         summary="Feature contract and stored feature importance for both models")
def models_features(model_set: str = _MODEL_SET_QUERY):
    m = get_model_set(model_set)
    contract = m.feature_contract()
    for hazard in ("flood", "landslide"):
        contract[hazard]["nullable_features"] = sorted(m.nullable_features[hazard])
    return {"data_mode": m.name,
            "model_set": m.name,
            "feature_contract": contract,
            "feature_importance": m.feature_importance(),
            "source": [_rel(m.paths["flood_features"]), _rel(m.paths["landslide_features"]),
                       _rel(m.paths["flood_results"]), _rel(m.paths["landslide_results"])],
            "note": ("Importance values were computed during training; the API does "
                     "not calculate feature importance.")}


# ---------------------------------------------------------------------
# Real district data (identifiers + observed inputs for the real models)
# ---------------------------------------------------------------------
@app.get("/districts", tags=["real data"],
         summary="The 14 Kerala districts of the real Master Dataset")
def districts_list():
    """District names (CHIRPS spelling), representative points and data coverage."""
    d = get_districts()
    return {"data_mode": "real",
            "n_districts": len(d.points),
            "districts": d.listing(),
            "source": [_rel(config.DISTRICT_REFERENCE), _rel(config.REAL_MASTER_DATASET)],
            "note": ("district is an identifier only; it is never passed to a model. "
                     "river_level_days = days with an observed CWC river level.")}


@app.get("/districts/{district}/observation", tags=["real data"],
         summary="Real observed model inputs for one district and day")
def district_observation(district: str = PathParam(..., description="e.g. Idukki"),
                         date: str | None = Query(None, description=(
                             "YYYY-MM-DD; defaults to the latest available day"))):
    """
    Returns the row of the real Master Dataset exactly as stored, in the
    format POST /predict/flood and /predict/landslide accept. Missing values
    (e.g. river_level_m for a district without a gauge) are null, not filled.
    """
    d = get_districts()
    try:
        out = d.observation(district, date)
    except KeyError as exc:
        raise HTTPException(404, {"error": "Unknown district", "detail": exc.args[0],
                                  "valid_districts": d.names()}) from exc
    except ValueError as exc:
        raise HTTPException(422, {"error": "Invalid date", "detail": str(exc)}) from exc
    except LookupError as exc:
        raise HTTPException(404, {"error": "No observation", "detail": str(exc)}) from exc
    obs = out["observation"]
    return {"data_mode": "real",
            "district": obs["district"], "date": obs["date"],
            **out,
            "source": _rel(config.REAL_MASTER_DATASET),
            "note": ("Values are read unchanged from the real Master Dataset. "
                     "recorded_labels are the dataset's event labels for this day "
                     "(0 = not reported, not proven absence). split tells whether the "
                     "day was used to train the models.")}


# ---------------------------------------------------------------------
# SHAP explainability (explanation layer only -- never modifies a
# prediction, never feeds TCDL)
# ---------------------------------------------------------------------
@app.get("/explain/global", tags=["explainability"],
         summary="Global mean|SHAP| feature importance for one XGBoost model")
def explain_global(model: str = Query("flood", description="flood | landslide")):
    """
    Served from scripts/analysis/shap_explainability.py. SHAP is computed in the ML layer,
    not here and not in the browser. Values are in log-odds space.
    """
    sv = get_shap()
    try:
        payload = sv.global_importance(model)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except SHAPUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        **_data_fields(),
        **payload,
        "interpretation_note": (
            "Mean |SHAP| ranks how much each feature moves this model's output on "
            f"average over the {sv.dataset} test period. It describes model behaviour, "
            "not physical causation."),
        "source": f"scripts/analysis/shap_explainability.py ({sv.source})",
    }


@app.get("/explain/current", tags=["explainability"],
         summary="SHAP explanation of a location's current model prediction")
def explain_current(model: str = Query("flood", description="flood | landslide"),
                    location_id: str = Query(..., description=(
                        "location_id or district name, e.g. Idukki")),
                    mode: str = Query("current", description="current | peak"),
                    date: str | None = Query(None, description=(
                        "YYYY-MM-DD, mode=current only: explain that day instead of "
                        "the latest one (the dashboard passes its as_of day)"))):
    """
    Explains the same timestep the dashboard shows for this location. The row
    is scored by the same frozen model the pipeline uses, so the explained
    probability equals the displayed probability. No SHAP value is computed in
    the frontend; this endpoint returns the ML layer's own explanation.
    """
    sv = get_shap()
    date = _date(date, "date")
    try:
        payload = sv.explain_current(model, location_id, mode, date)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except SHAPUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"SHAP explanation failed: {exc}") from exc
    payload.update(_data_fields())
    payload["source"] = f"scripts/analysis/shap_explainability.py ({sv.source})"
    return payload


@app.get("/", tags=["system"], include_in_schema=False)
def root():
    return {"name": config.API_TITLE, "version": config.API_VERSION,
            "data_mode": config.DATA_MODE,
            "prediction_model_set": config.PREDICTION_MODEL_SET,
            "dashboard_model_set": config.DASHBOARD_MODEL_SET,
            "docs": "/docs", "openapi": "/openapi.json"}
