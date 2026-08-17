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
ML_DIR = PROJECT_ROOT / "ml"

# Allow deployment overrides without touching code.
if os.environ.get("CAPSTONE_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["CAPSTONE_PROJECT_ROOT"]).resolve()
    ML_DIR = PROJECT_ROOT / "ml"

# --- Master dataset ---------------------------------------------------
MASTER_DATASET = PROJECT_ROOT / "synthetic_master_dataset.csv"
DATA_DICTIONARY = PROJECT_ROOT / "DATA_DICTIONARY.md"

# --- Frozen model artifacts (read-only) -------------------------------
FLOOD_MODEL = ML_DIR / "flood_xgboost_model.json"
FLOOD_FEATURES = ML_DIR / "flood_features.json"
FLOOD_RESULTS = ML_DIR / "flood_model_results.json"

LANDSLIDE_MODEL = ML_DIR / "landslide_xgboost_model.json"
LANDSLIDE_FEATURES = ML_DIR / "landslide_features.json"
LANDSLIDE_RESULTS = ML_DIR / "landslide_model_results.json"

# --- TCDL implementation and its precomputed outputs ------------------
TCDL_MODULE = ML_DIR / "tcdl_v1.py"
TCDL_RESULTS = ML_DIR / "tcdl_results.json"
TCDL_TIMESERIES = ML_DIR / "tcdl_timeseries.csv"
TCDL_WARNINGS = ML_DIR / "tcdl_warnings.csv"
TCDL_LEAD_TIME = ML_DIR / "tcdl_lead_time.csv"

REQUIRED_ARTIFACTS = {
    "master_dataset": MASTER_DATASET,
    "flood_model": FLOOD_MODEL,
    "flood_features": FLOOD_FEATURES,
    "flood_results": FLOOD_RESULTS,
    "landslide_model": LANDSLIDE_MODEL,
    "landslide_features": LANDSLIDE_FEATURES,
    "landslide_results": LANDSLIDE_RESULTS,
    "tcdl_module": TCDL_MODULE,
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
    Report whether the pipeline is running on synthetic or real data.

    Derived from the dataset filename rather than hard-coded, so the flag
    flips by itself once a real master dataset is wired in -- the API
    contract does not change.
    """
    name = MASTER_DATASET.name.lower()
    return "synthetic" if "synthetic" in name or "dummy" in name else "real"


DATA_MODE = detect_data_mode()
SYNTHETIC_WARNING = (
    "SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS. Every probability, "
    "warning and metric served by this API is derived from a simulated dataset. "
    "These are not real-world hazard warnings and must not be acted upon or "
    "reported as research performance."
)
