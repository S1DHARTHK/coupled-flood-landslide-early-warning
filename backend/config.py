"""
Backend configuration and path resolution.

Every path is derived from this file's own location, so the project can
be cloned to any machine or directory without edits. No absolute,
machine-specific paths appear anywhere in the backend.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/config.py -> backend/ -> <project root>
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Allow deployment overrides without touching code.
if os.environ.get("CAPSTONE_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["CAPSTONE_PROJECT_ROOT"]).resolve()

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SYNTHETIC_DIR = PROJECT_ROOT / "synthetic"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# --- SYNTHETIC reference pipeline (synthetic/) -------------------------
# Simulated dataset and the models / TCDL / SHAP outputs built from it.
# Served only by ?model_set=synthetic or CAPSTONE_DASHBOARD_DATA=synthetic.
MASTER_DATASET = SYNTHETIC_DIR / "dataset" / "synthetic_master_dataset.csv"
DATA_DICTIONARY = SYNTHETIC_DIR / "dataset" / "DATA_DICTIONARY.md"

FLOOD_MODEL = SYNTHETIC_DIR / "models" / "flood_xgboost_model.json"
FLOOD_FEATURES = SYNTHETIC_DIR / "models" / "flood_features.json"
FLOOD_RESULTS = SYNTHETIC_DIR / "results" / "flood_model_results.json"

LANDSLIDE_MODEL = SYNTHETIC_DIR / "models" / "landslide_xgboost_model.json"
LANDSLIDE_FEATURES = SYNTHETIC_DIR / "models" / "landslide_features.json"
LANDSLIDE_RESULTS = SYNTHETIC_DIR / "results" / "landslide_model_results.json"

# --- TCDL implementation (shared by both sets) + synthetic outputs ----
TCDL_MODULE = SCRIPTS_DIR / "models" / "tcdl_v1.py"
TCDL_RESULTS = SYNTHETIC_DIR / "tcdl" / "tcdl_results.json"
TCDL_TIMESERIES = SYNTHETIC_DIR / "tcdl" / "tcdl_timeseries.csv"
TCDL_WARNINGS = SYNTHETIC_DIR / "tcdl" / "tcdl_warnings.csv"
TCDL_LEAD_TIME = SYNTHETIC_DIR / "tcdl" / "tcdl_lead_time.csv"

# --- SHAP explainability (scripts/analysis/shap_explainability.py, shared)
# Imported and bound to the served dataset with its own configure(); the
# SHAP maths is never reimplemented in the backend.
SHAP_MODULE = SCRIPTS_DIR / "analysis" / "shap_explainability.py"
SHAP_OUTPUT_DIR = SYNTHETIC_DIR / "shap"

# --- REAL-data pipeline (V1.0) -----------------------------------------
# Trained by scripts/models/flood_xgboost.py, scripts/models/landslide_xgboost.py,
# scripts/models/tcdl_v1.py and scripts/analysis/shap_explainability.py (all
# default to --dataset real) on the frozen Master Dataset V1.0 in dataset/.
# Outputs live in artifacts/{models,results,tcdl,shap}/, so a retrain on new
# data only needs those scripts re-run -- no backend edits.
REAL_MODELS_DIR = ARTIFACTS_DIR / "models"
REAL_RESULTS_DIR = ARTIFACTS_DIR / "results"
REAL_TCDL_DIR = ARTIFACTS_DIR / "tcdl"
REAL_FLOOD_MODEL = REAL_MODELS_DIR / "flood_xgboost_model.json"
REAL_FLOOD_FEATURES = REAL_MODELS_DIR / "flood_features.json"
REAL_FLOOD_RESULTS = REAL_RESULTS_DIR / "flood_model_results.json"
REAL_LANDSLIDE_MODEL = REAL_MODELS_DIR / "landslide_xgboost_model.json"
REAL_LANDSLIDE_FEATURES = REAL_MODELS_DIR / "landslide_features.json"
REAL_LANDSLIDE_RESULTS = REAL_RESULTS_DIR / "landslide_model_results.json"
REAL_TCDL_RESULTS = REAL_TCDL_DIR / "tcdl_results.json"
REAL_TCDL_TIMESERIES = REAL_TCDL_DIR / "tcdl_timeseries.csv"
REAL_TCDL_WARNINGS = REAL_TCDL_DIR / "tcdl_warnings.csv"
REAL_TCDL_LEAD_TIME = REAL_TCDL_DIR / "tcdl_lead_time.csv"
REAL_SHAP_OUTPUT_DIR = ARTIFACTS_DIR / "shap"

REAL_MASTER_DATASET = PROJECT_ROOT / "dataset" / "real_master_dataset.csv"
DISTRICT_REFERENCE = BACKEND_DIR / "data" / "Kerala_district_reference.csv"

# The two complete pipelines the backend knows. Each is served by its own
# ModelService (and, for the dashboard set, its own TCDL and SHAP services);
# nothing is shared between them.
SYNTHETIC_MODEL_SET = {
    "name": "synthetic",
    "flood_model": FLOOD_MODEL, "flood_features": FLOOD_FEATURES,
    "flood_results": FLOOD_RESULTS,
    "landslide_model": LANDSLIDE_MODEL, "landslide_features": LANDSLIDE_FEATURES,
    "landslide_results": LANDSLIDE_RESULTS,
    "master_dataset": MASTER_DATASET,
    "tcdl_results": TCDL_RESULTS, "tcdl_timeseries": TCDL_TIMESERIES,
    "tcdl_warnings": TCDL_WARNINGS, "tcdl_lead_time": TCDL_LEAD_TIME,
    "shap_dataset": "synthetic", "shap_output_dir": SHAP_OUTPUT_DIR,
}
REAL_MODEL_SET = {
    "name": "real",
    "flood_model": REAL_FLOOD_MODEL, "flood_features": REAL_FLOOD_FEATURES,
    "flood_results": REAL_FLOOD_RESULTS,
    "landslide_model": REAL_LANDSLIDE_MODEL, "landslide_features": REAL_LANDSLIDE_FEATURES,
    "landslide_results": REAL_LANDSLIDE_RESULTS,
    "master_dataset": REAL_MASTER_DATASET,
    "tcdl_results": REAL_TCDL_RESULTS, "tcdl_timeseries": REAL_TCDL_TIMESERIES,
    "tcdl_warnings": REAL_TCDL_WARNINGS, "tcdl_lead_time": REAL_TCDL_LEAD_TIME,
    "shap_dataset": "real", "shap_output_dir": REAL_SHAP_OUTPUT_DIR,
}
MODEL_SETS = {"real": REAL_MODEL_SET, "synthetic": SYNTHETIC_MODEL_SET}

# Model set used by the single-hazard prediction endpoints.
PREDICTION_MODEL_SET = "real"

# Model set behind the dashboard: /current, /trends, /locations, /warnings,
# /leadtime, /evaluation, /predict/coupled (TCDL) and /explain/* (SHAP).
# Set CAPSTONE_DASHBOARD_DATA=synthetic to serve the old simulated pipeline.
DASHBOARD_MODEL_SET = os.environ.get("CAPSTONE_DASHBOARD_DATA", "real").strip().lower()
if DASHBOARD_MODEL_SET not in MODEL_SETS:
    raise ValueError(f"CAPSTONE_DASHBOARD_DATA must be one of {sorted(MODEL_SETS)}, "
                     f"got {DASHBOARD_MODEL_SET!r}")
DASHBOARD_SET = MODEL_SETS[DASHBOARD_MODEL_SET]

REQUIRED_ARTIFACTS = {
    "real_master_dataset": REAL_MASTER_DATASET,
    "real_flood_model": REAL_FLOOD_MODEL,
    "real_flood_features": REAL_FLOOD_FEATURES,
    "real_flood_results": REAL_FLOOD_RESULTS,
    "real_landslide_model": REAL_LANDSLIDE_MODEL,
    "real_landslide_features": REAL_LANDSLIDE_FEATURES,
    "real_landslide_results": REAL_LANDSLIDE_RESULTS,
    "real_tcdl_results": REAL_TCDL_RESULTS,
    "real_tcdl_timeseries": REAL_TCDL_TIMESERIES,
    "real_tcdl_warnings": REAL_TCDL_WARNINGS,
    "real_tcdl_lead_time": REAL_TCDL_LEAD_TIME,
    "district_reference": DISTRICT_REFERENCE,
    "tcdl_module": TCDL_MODULE,
    "shap_module": SHAP_MODULE,
    "master_dataset": MASTER_DATASET,
    "flood_model": FLOOD_MODEL,
    "flood_features": FLOOD_FEATURES,
    "flood_results": FLOOD_RESULTS,
    "landslide_model": LANDSLIDE_MODEL,
    "landslide_features": LANDSLIDE_FEATURES,
    "landslide_results": LANDSLIDE_RESULTS,
    "tcdl_results": TCDL_RESULTS,
    "tcdl_timeseries": TCDL_TIMESERIES,
    "tcdl_warnings": TCDL_WARNINGS,
    "tcdl_lead_time": TCDL_LEAD_TIME,
}

# --- API metadata -----------------------------------------------------
API_TITLE = "Flood-Landslide Early Warning API"
API_VERSION = "1.0.0"
PIPELINE_VERSIONS = {
    "flood_model": "1.0",
    "landslide_model": "1.0",
    "tcdl": "1.0",
    "api": API_VERSION,
    "prediction_model_set": PREDICTION_MODEL_SET,
    "dashboard_model_set": DASHBOARD_MODEL_SET,
}

# CORS for local frontend development.
#
# A fixed port list breaks as soon as the dev server picks a different port
# (Vite auto-increments when 5173 is taken), so any loopback origin is allowed
# instead. This is a development convenience: tighten it to the deployed
# frontend origin before exposing the API beyond localhost.
CORS_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

# Explicit origins kept for deployments that prefer an allowlist.
CORS_ORIGINS = [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8080", "http://127.0.0.1:8080",
]


def detect_data_mode() -> str:
    """
    Report whether the dashboard pipeline runs on synthetic or real data.

    Derived from the dashboard set's dataset filename rather than hard-coded,
    so the flag follows the data actually served.
    """
    name = Path(DASHBOARD_SET["master_dataset"]).name.lower()
    return "synthetic" if "synthetic" in name or "dummy" in name else "real"


DATA_MODE = detect_data_mode()
SYNTHETIC_WARNING = (
    "SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS. Every probability, "
    "warning and metric served by this API is derived from a simulated dataset. "
    "These are not real-world hazard warnings and must not be acted upon or "
    "reported as research performance."
)
REAL_MODEL_NOTE = (
    "REAL-DATA MODELS -- Flood / Landslide XGBoost V1.0 trained on the real Kerala "
    "district-day Master Dataset (2012-2024). Research outputs, not operational "
    "warnings: labels are reported events at district level (0 = not reported), "
    "probabilities are not calibrated, and the landslide model rests on very few "
    "positive cases."
)
REAL_TCDL_NOTE = (
    "REAL DATA -- TCDL V1.0 applied unchanged to the real models. Its thresholds are "
    "training-period quantiles of the real-model signals and have not been "
    "recalibrated: on the 2023-2024 test period the flood probability sits above "
    "0.50 for much of each monsoon, so warnings are frequent and most are false "
    "alarms. Read every warning as a research output, not an operational alert."
)


def data_notice(mode: str) -> str:
    """The notice every response of a given data mode carries."""
    return SYNTHETIC_WARNING if mode == "synthetic" else REAL_TCDL_NOTE
