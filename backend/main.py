"""
FastAPI backend for the Flood-Landslide Early Warning system.

This is the SERVING layer only. It loads the frozen artifacts produced by
the ML pipeline once at startup and exposes them. It does not train,
tune, re-threshold, or re-implement any ML logic:

    Flood XGBoost V1.0      -- loaded from ml/flood_xgboost_model.json
    Landslide XGBoost V1.0  -- loaded from ml/landslide_xgboost_model.json
    TCDL V1.0               -- imported from ml/tcdl_v1.py, thresholds read
                               from ml/tcdl_results.json

Run:
    uvicorn backend.main:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
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
from .services.model_service import ArtifactError, ModelService
from .services.tcdl_service import (
    InsufficientHistory, TCDLService, TCDLUnavailable, WARNING_TYPES, classify_warning,
)

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
STATE: dict[str, Any] = {"models": None, "tcdl": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = ModelService()
    tcdl = TCDLService(models)
    STATE["models"] = models
    STATE["tcdl"] = tcdl

    # Request schemas are built from the loaded feature contracts, so the
    # OpenAPI documentation always matches the served models.
    if models.flood_ready:
        app.state.FloodObservation = build_observation_model(
            "FloodObservation", models.flood_features, {},
            "One observation for the Flood XGBoost V1.0 model.")
    if models.landslide_ready:
        app.state.LandslideObservation = build_observation_model(
            "LandslideObservation", models.landslide_features, models.category_schema,
            "One observation for the Landslide XGBoost V1.0 model.")
    yield
    STATE.clear()


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    lifespan=lifespan,
    description=(
        "Serving layer for the coupled flood-landslide early warning ML pipeline.\n\n"
        f"**{config.SYNTHETIC_WARNING}**\n\n"
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
    m = STATE.get("models")
    if m is None:
        raise HTTPException(503, "Model service not initialised.")
    return m


def get_tcdl() -> TCDLService:
    t = STATE.get("tcdl")
    if t is None:
        raise HTTPException(503, "TCDL service not initialised.")
    if not t.ready:
        raise HTTPException(503, {"error": "TCDL unavailable", "details": t.errors})
    return t


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
        "common_environmental": block(common),
        "flood_specific": block(flood_specific),
        "landslide_specific": block(landslide_specific),
        "unavailable_fields": unavailable,
        "note": ("Fields listed in unavailable_fields are absent from the current "
                 "dataset and are returned as null. No value is ever substituted."),
    }


# ---------------------------------------------------------------------
# 1. System status
# ---------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"],
         summary="Liveness and readiness of the serving layer")
def health():
    m = get_models()
    t = STATE.get("tcdl")
    errors = list(m.errors) + (list(t.errors) if t else ["tcdl service not initialised"])
    ready = m.ready and bool(t and t.ready)
    return HealthResponse(
        status="ok" if ready else "degraded",
        api_version=config.API_VERSION,
        data_mode=config.DATA_MODE,
        synthetic_data_warning=(config.SYNTHETIC_WARNING
                                if config.DATA_MODE == "synthetic" else None),
        ready_for_prediction=ready,
        flood_model_loaded=m.flood_ready,
        landslide_model_loaded=m.landslide_ready,
        tcdl_available=bool(t and t.ready),
        master_dataset_loaded=m.dataset_ready,
        errors=errors,
    )


@app.get("/models/status", response_model=ModelStatusResponse, tags=["system"],
         summary="Detailed status of both models and the TCDL layer")
def models_status():
    m = get_models()
    t = STATE.get("tcdl")
    ready = m.ready and bool(t and t.ready)
    artifacts = {k: {"path": str(p.relative_to(config.PROJECT_ROOT))
                     if p.is_relative_to(config.PROJECT_ROOT) else str(p),
                     "exists": p.exists()}
                 for k, p in config.REQUIRED_ARTIFACTS.items()}
    return ModelStatusResponse(
        status="ok" if ready else "degraded",
        data_mode=config.DATA_MODE,
        ready_for_prediction=ready,
        versions=config.PIPELINE_VERSIONS,
        flood_model={
            "loaded": m.flood_ready, "version": m.flood_meta.get("version"),
            "n_features": len(m.flood_features), "n_trees": m.flood_meta.get("n_trees"),
            "decision_threshold": m.flood_threshold, "target": m.flood_meta.get("target"),
            "retrained_by_api": False,
        },
        landslide_model={
            "loaded": m.landslide_ready, "version": m.landslide_meta.get("version"),
            "n_features": len(m.landslide_features),
            "n_trees": m.landslide_meta.get("n_trees"),
            "decision_threshold": m.landslide_threshold,
            "target": m.landslide_meta.get("target"),
            "categorical_features": list(m.category_schema),
            "retrained_by_api": False,
        },
        tcdl={
            "available": bool(t and t.ready),
            "version": (t.results.get("version") if t and t.ready else None),
            "layer_type": (t.results.get("layer_type") if t and t.ready else None),
            "rule_ids": (t.rule_ids if t and t.ready else []),
            "thresholds_source": "ml/tcdl_results.json (resolved_thresholds)",
            "thresholds_recomputed_by_api": False,
            "min_history_rows_for_trends": (t.min_history_rows if t and t.ready else None),
        },
        artifacts=artifacts,
        errors=list(m.errors) + (list(t.errors) if t else []),
    )


# ---------------------------------------------------------------------
# 2-3. Single-hazard prediction
# ---------------------------------------------------------------------
class _ObsList(BaseModel):
    observations: list[dict] = Field(..., min_length=1,
                                     description="One or more observations.")


def _predict(hazard: str, payload: dict) -> HazardPredictionResponse:
    m = get_models()
    records = payload.get("observations")
    if not isinstance(records, list) or not records:
        raise HTTPException(422, "Body must contain a non-empty 'observations' list.")

    if hazard == "flood":
        if not m.flood_ready:
            raise HTTPException(503, {"error": "Flood model not loaded", "details": m.errors})
        features, threshold, version = (m.flood_features, m.flood_threshold,
                                        m.flood_meta.get("version", "1.0"))
    else:
        if not m.landslide_ready:
            raise HTTPException(503, {"error": "Landslide model not loaded",
                                      "details": m.errors})
        features, threshold, version = (m.landslide_features, m.landslide_threshold,
                                        m.landslide_meta.get("version", "1.0"))

    problems = m.validate_features(records, features, hazard)
    for i, rec in enumerate(records):
        for key in ("date", "latitude", "longitude"):
            if key not in rec or rec[key] is None:
                problems.append(f"record {i}: missing '{key}'")
    if problems:
        raise HTTPException(422, {"error": "Invalid input", "problems": problems,
                                  "required_features": features})

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
            warning_status="Warning" if pred else "No Warning"))

    return HazardPredictionResponse(
        hazard=hazard, model_version=str(version), decision_threshold=threshold,
        data_mode=config.DATA_MODE,
        synthetic_data_warning=(config.SYNTHETIC_WARNING
                                if config.DATA_MODE == "synthetic" else None),
        n_observations=len(preds), predictions=preds,
        calibration_note=CALIBRATION_NOTE)


@app.post("/predict/flood", response_model=HazardPredictionResponse, tags=["prediction"],
          summary="Flood probability and prediction from Flood XGBoost V1.0")
def predict_flood(payload: dict = Body(..., examples=[{"observations": [{
        "date": "2020-08-01", "latitude": 9.283, "longitude": 76.774,
        "rainfall_1d_mm": 42.5, "rainfall_3d_mm": 96.0, "rainfall_7d_mm": 180.4,
        "rainfall_14d_mm": 260.1, "rainfall_30d_mm": 430.7, "soil_moisture": 0.41,
        "temperature_c": 25.4, "humidity_percent": 88.2, "elevation_m": 19.0,
        "river_level_m": 3.1, "distance_to_river_km": 0.16, "drainage_density": 3.38}]}])):
    """Uses the frozen model exactly as trained: same feature order, same 0.50 threshold."""
    return _predict("flood", payload)


@app.post("/predict/landslide", response_model=HazardPredictionResponse,
          tags=["prediction"],
          summary="Landslide probability and prediction from Landslide XGBoost V1.0")
def predict_landslide(payload: dict = Body(..., examples=[{"observations": [{
        "date": "2020-08-01", "latitude": 10.978, "longitude": 76.955,
        "rainfall_1d_mm": 55.2, "rainfall_3d_mm": 130.5, "rainfall_7d_mm": 240.8,
        "rainfall_14d_mm": 350.2, "rainfall_30d_mm": 610.3, "soil_moisture": 0.47,
        "temperature_c": 19.2, "humidity_percent": 91.5, "elevation_m": 1268.0,
        "slope_degree": 37.9, "aspect_degree": 249.0, "curvature": -0.77,
        "land_cover": "barren", "lithology": "shale"}]}])):
    """Uses the frozen model exactly as trained: same feature order, schema and threshold."""
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
    functions imported from `ml/tcdl_v1.py`.

    A series is required rather than a single row because the trend signals are
    only defined over a window of consecutive days. Too short a series is
    rejected -- the API will not fabricate the missing history.

    The decision returned is for the LAST timestep in the series.
    """
    m, t = get_models(), get_tcdl()
    records = payload.get("observations")
    if not isinstance(records, list) or not records:
        raise HTTPException(422, "Body must contain a non-empty 'observations' list.")

    needed = sorted(set(m.flood_features) | set(m.landslide_features))
    problems = m.validate_features(records, needed, "coupled")
    for i, rec in enumerate(records):
        for key in ("date", "latitude", "longitude"):
            if key not in rec or rec[key] is None:
                problems.append(f"record {i}: missing '{key}'")
    if problems:
        raise HTTPException(422, {"error": "Invalid input", "problems": problems,
                                  "required_features": needed,
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
        data_mode=config.DATA_MODE,
        synthetic_data_warning=(config.SYNTHETIC_WARNING
                                if config.DATA_MODE == "synthetic" else None),
        warning_status="Warning" if is_warning else "No Warning",
        warning_type=wtype,
        warning_timestamp=ts if is_warning else None,
        timestamp_resolution="daily (24-hour quantisation)",
        location={"latitude": num(row["latitude"]), "longitude": num(row["longitude"]),
                  "location_id": row.get("location_id")},
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
@app.get("/current", tags=["monitoring"],
         summary="Current environmental conditions, predictions, trends and warning")
def current(location_id: str | None = Query(None, description="Omit for all locations"),
            as_of: str | None = Query(None, description="ISO date; defaults to latest")):
    """Latest available timestep from the existing pipeline outputs."""
    m, t = get_models(), get_tcdl()
    try:
        df = t.timeseries_slice(location_id=location_id, end=as_of)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if df.empty:
        raise HTTPException(404, "No records for the requested location/date.")

    assert m.master is not None
    master = m.master
    out = []
    for loc, g in df.groupby("location_id"):
        row = g.sort_values("date").iloc[-1]
        mrow = master[(master["location_id"] == loc) & (master["date"] == row["date"])]
        env = (_env_block(m, mrow.iloc[0]) if not mrow.empty
               else {"date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                     "latitude": num(row["latitude"]), "longitude": num(row["longitude"]),
                     "common_environmental": {}, "flood_specific": {},
                     "landslide_specific": {},
                     "unavailable_fields": ["all"],
                     "note": "No master-dataset row matched this timestep."})
        triggered = [r for r in t.rule_ids if bool(row.get(r, False))]
        is_warning = bool(int(row.get("tcdl_warning", 0)))
        out.append({
            "location_id": loc,
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
        })
    return {"data_mode": config.DATA_MODE,
            "synthetic_data_warning": (config.SYNTHETIC_WARNING
                                       if config.DATA_MODE == "synthetic" else None),
            "as_of": pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d"),
            "n_locations": len(out), "locations": out}


@app.get("/trends", tags=["monitoring"],
         summary="Temporal signals consumed by TCDL (smoothed values and rates)")
def trends(location_id: str = Query(..., description="Required; trends are per location"),
           limit: int = Query(90, ge=1, le=2000),
           split: str | None = Query(None, description="train | validation | test")):
    """
    Values come from the TCDL pipeline output, not recomputed here.
    Nulls mark timesteps where a rolling window is not yet full.
    """
    t = get_tcdl()
    try:
        df = t.timeseries_slice(location_id=location_id, split=split, limit=limit)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if df.empty:
        raise HTTPException(404, "No records for the requested filters.")

    series = [{"date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
               **trend_signals(r).model_dump()} for _, r in df.iterrows()]
    return {"data_mode": config.DATA_MODE, "location_id": location_id,
            "n_records": len(series),
            "parameters": {k: t.results["parameters"].get(k) for k in
                           ("smooth_window_days", "rate_window_days", "sustain_days")},
            "source": "ml/tcdl_timeseries.csv (produced by ml/tcdl_v1.py)",
            "series": series}


@app.get("/locations", tags=["monitoring"], summary="Locations available in the pipeline")
def locations():
    return {"data_mode": config.DATA_MODE, "locations": get_tcdl().locations()}


# ---------------------------------------------------------------------
# 9. Historical warnings
# ---------------------------------------------------------------------
@app.get("/warnings", tags=["monitoring"],
         summary="Historical warning timeline from existing TCDL outputs")
def warnings(location_id: str | None = Query(None),
             start: str | None = Query(None), end: str | None = Query(None),
             warning_only: bool = Query(True, description="False returns every timestep"),
             limit: int = Query(500, ge=1, le=5000)):
    t = get_tcdl()
    try:
        df = t.timeseries_slice(location_id=location_id, start=start, end=end)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, f"Invalid date filter: {exc}") from exc
    if warning_only:
        df = df[df["tcdl_warning"] == 1]
    df = df.tail(limit)

    items = []
    for _, r in df.iterrows():
        triggered = [x for x in t.rule_ids if bool(r.get(x, False))]
        is_warning = bool(int(r.get("tcdl_warning", 0)))
        items.append({
            "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
            "latitude": num(r["latitude"]), "longitude": num(r["longitude"]),
            "location_id": r["location_id"], "split": r.get("split"),
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
    return {"data_mode": config.DATA_MODE,
            "synthetic_data_warning": (config.SYNTHETIC_WARNING
                                       if config.DATA_MODE == "synthetic" else None),
            "warning_types": WARNING_TYPES,
            "timestamp_resolution": "daily (24-hour quantisation)",
            "n_records": len(items), "records": items}


@app.get("/warnings/{location_id}/{warning_date}", response_model=CoupledWarningResponse,
         tags=["monitoring"], summary="Full coupled warning detail for one timestep")
def warning_detail(location_id: str = PathParam(...),
                   warning_date: str = PathParam(..., description="YYYY-MM-DD")):
    t = get_tcdl()
    try:
        df = t.timeseries_slice(location_id=location_id, start=warning_date, end=warning_date)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, f"Invalid date: {exc}") from exc
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
             detected_only: bool = Query(False),
             limit: int = Query(500, ge=1, le=5000)):
    """
    Served directly from ml/tcdl_lead_time.csv. Nothing is recomputed, and no
    hour-of-day is inferred: the underlying data is daily.
    """
    t = get_tcdl()
    assert t.lead_time is not None
    df = t.lead_time
    if system:
        df = df[df["system"] == system]
    if hazard_type:
        df = df[df["hazard_type"] == hazard_type]
    if detected_only:
        df = df[df["detected"] == 1]
    if df.empty:
        raise HTTPException(404, {"error": "No lead-time records match the filters",
                                  "available_systems": sorted(t.lead_time["system"].unique()),
                                  "available_hazard_types": sorted(
                                      t.lead_time["hazard_type"].unique())})
    df = df.head(limit)

    records = [{
        "system": r["system"], "hazard_type": r["hazard_type"],
        "location_id": r["location_id"],
        "latitude": num(r["latitude"]), "longitude": num(r["longitude"]),
        "event_time": r["event_timestamp"] or None,
        "warning_time": (r["first_warning_timestamp"]
                         if isinstance(r["first_warning_timestamp"], str)
                         and r["first_warning_timestamp"] else None),
        "detected": int(r["detected"]),
        "lead_time_hours": num(r["lead_time_hours_earliest"]),
        "lead_time_hours_contiguous": num(r["lead_time_hours_contiguous"]),
    } for _, r in df.iterrows()]

    return LeadTimeResponse(
        data_mode=config.DATA_MODE,
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
    """Returns tcdl_results.json values verbatim. No metric is recomputed here."""
    t = get_tcdl()
    results = t.results.get("results", {})
    if hazard_type not in results:
        raise HTTPException(404, {"error": f"Unknown hazard_type '{hazard_type}'",
                                  "available": sorted(results)})
    return {
        "data_mode": config.DATA_MODE,
        "synthetic_data_warning": (config.SYNTHETIC_WARNING
                                   if config.DATA_MODE == "synthetic" else None),
        "hazard_type": hazard_type,
        "evaluation_period": t.results.get("evaluation_period"),
        "systems": results[hazard_type],
        "warning_counts": t.results.get("warning_counts"),
        "rule_firing_counts": t.results.get("rule_firing_counts"),
        "parameters": t.results.get("parameters"),
        "rules": t.results.get("rules"),
        "rule_classification": t.results.get("rule_classification"),
        "interpretation_notes": {
            "scope": t.results.get("_scope_note"),
            "structural_dominance": t.results.get("_structural_dominance_note"),
            "lead_time": t.results.get("lead_time_note"),
        },
        "source": "ml/tcdl_results.json",
    }


# ---------------------------------------------------------------------
# 11-12. Model performance and feature importance
# ---------------------------------------------------------------------
@app.get("/models/performance", tags=["evaluation"],
         summary="Stored evaluation metrics for both XGBoost models")
def models_performance():
    """Read from the training result files. Nothing is retrained or recomputed."""
    return {"data_mode": config.DATA_MODE,
            "synthetic_data_warning": (config.SYNTHETIC_WARNING
                                       if config.DATA_MODE == "synthetic" else None),
            "models": get_models().performance(),
            "source": ["ml/flood_model_results.json", "ml/landslide_model_results.json"],
            "note": ("Metrics are those computed during training on the frozen "
                     "chronological split. The API does not recompute them.")}


@app.get("/models/features", tags=["evaluation"],
         summary="Feature contract and stored feature importance for both models")
def models_features():
    m = get_models()
    return {"data_mode": config.DATA_MODE,
            "feature_contract": m.feature_contract(),
            "feature_importance": m.feature_importance(),
            "source": ["ml/flood_features.json", "ml/landslide_features.json",
                       "ml/flood_model_results.json", "ml/landslide_model_results.json"],
            "note": ("Importance values were computed during training; the API does "
                     "not calculate feature importance.")}


@app.get("/", tags=["system"], include_in_schema=False)
def root():
    return {"name": config.API_TITLE, "version": config.API_VERSION,
            "data_mode": config.DATA_MODE, "docs": "/docs", "openapi": "/openapi.json"}
