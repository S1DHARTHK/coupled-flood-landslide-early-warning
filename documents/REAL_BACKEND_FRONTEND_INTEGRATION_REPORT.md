# Real XGBoost → FastAPI → Frontend Integration Report

**Date:** 12 September 2026 · **Scope:** connect the real-data Flood and Landslide XGBoost models (`models/real/`) to the existing FastAPI backend and React frontend. **TCDL has NOT been integrated** (see §9).

## 1. Architecture

```text
Frontend  (Model Performance page → "real-model prediction // by district" panel)
   │  GET  /districts                               district list (identifier only)
   │  GET  /districts/{district}/observation?date=  real row of the real Master Dataset
   │  POST /predict/flood      POST /predict/landslide
   ▼
FastAPI  ── REAL ModelService ──► models/real/flood_xgboost_model.json
   │                          └──► models/real/landslide_xgboost_model.json
   │
   └──── SYNTHETIC ModelService ─► ml/*_xgboost_model.json  (unchanged)
            used by TCDL (/predict/coupled), /current, /trends, /warnings,
            /leadtime, /evaluation and SHAP (/explain/*)
```

The backend now loads **two separate model sets**. TCDL still scores its inputs with the synthetic models, because its thresholds were fitted to the synthetic models' probabilities. Feeding it real-model output would amount to integrating TCDL, which is out of scope. The dashboard, warnings and analysis pages are therefore unchanged and stay labelled "synthetic".

## 2. Files changed

| File | Change |
|---|---|
| `backend/config.py` | Real model, real dataset and district-reference paths; `SYNTHETIC_MODEL_SET` / `REAL_MODEL_SET`; `PREDICTION_MODEL_SET = "real"`; real artifacts added to `REQUIRED_ARTIFACTS` |
| `backend/services/model_service.py` | One instance per model set (default = synthetic, i.e. the old behaviour). It checks the booster's feature names against the saved `feature_order`, reads the nullable features from the training results, and adds strict `validate_observations()`. The feature frame is cast to float so nulls become NaN |
| `backend/services/district_service.py` (new) | District names (CHIRPS spelling plus geoBoundaries alias, case-insensitive), representative points, and real observation lookup |
| `backend/schemas.py` | Additive optional fields only: `district` and `missing_features` per prediction; `model_set`, `model_artifact` and `model_note` per response; extra `/health` and `/models/status` fields |
| `backend/main.py` | Loads both model sets. `/predict/flood` and `/predict/landslide` now use the real set (with district resolution). New `GET /districts` and `GET /districts/{district}/observation`. `/models/performance` and `/models/features` default to the real set (`?model_set=synthetic` still available). `/health` and `/models/status` report both sets |
| `backend/tests/test_real_integration.py` (new) | 16 integration tests (§8) |
| `frontend/src/components/terminal/TermRealPrediction.tsx` (new) | District + date → real prediction panel, built from the existing terminal components |
| `frontend/src/pages/EarlyWarning/ModelPerformance.tsx` | Mounts the panel. Two previously hard-coded "synthetic" sentences now follow `data_mode` |
| `frontend/src/components/terminal/TerminalUI.tsx` | Banner has a `real` case ("real historical data"); the synthetic wording is unchanged |
| `frontend/src/services/api.ts`, `frontend/src/types/hazard.ts` | `getDistricts`, `getDistrictObservation`; typed `predictFlood` / `predictLandslide`; new and optional types |

**Not changed:** `ml/` (models, training code, TCDL, SHAP), `models/real/`, all datasets, `backend/services/tcdl_service.py`, `backend/services/shap_service.py`, and the Dashboard, Warnings, Analysis and Landing pages.

## 3. Real models connected

| Endpoint | Model file | Features (saved order) | Threshold |
|---|---|---|---|
| `POST /predict/flood` | `models/real/flood_xgboost_model.json` (9 trees) | rainfall 1/3/7/14/30 d, soil_moisture, temperature_c, humidity_percent, elevation_m, river_level_m, distance_to_river_km, drainage_density | 0.50 |
| `POST /predict/landslide` | `models/real/landslide_xgboost_model.json` (72 trees) | same 9 common + slope_degree, aspect_degree, curvature, land_cover, lithology | 0.50 |

At startup the backend refuses to serve a booster whose `feature_names` differ from the saved `feature_order`. The real feature order is identical to the synthetic one, so the request format is unchanged.

## 4. Request / response

Request (unchanged format; `district` is new and optional):
```json
{"observations": [{"date": "2018-08-16", "district": "Ernakulam", "latitude": 10.08408, "longitude": 76.5463,
  "rainfall_1d_mm": 82.91, "...": "...", "river_level_m": 11.4046, "distance_to_river_km": 1.116, "drainage_density": 0.4117}]}
```
Response: the same fields as before. The new fields are optional and marked ★:
```json
{"hazard": "flood", "model_version": "1.0", "decision_threshold": 0.5, "data_mode": "real", "synthetic_data_warning": null,
 "n_observations": 1, "calibration_note": "...", "model_set": "real"★, "model_artifact": "models/real/flood_xgboost_model.json"★, "model_note": "..."★,
 "predictions": [{"date": "2018-08-16", "latitude": 10.08408, "longitude": 76.5463, "probability": 0.671631, "prediction": 1,
                  "warning_status": "Warning", "district": "Ernakulam"★, "missing_features": []★}]}
```
The error format is unchanged: FastAPI `{"detail": ...}`. For invalid input, `detail` is `{"error": "Invalid input", "problems": [...], "required_features": [...], "nullable_features": [...]}`. A model that isn't loaded gives 503 `{"error": "... model not loaded", "details": [...]}`.

## 5. District handling

- `district` is an identifier. It is removed before the feature frame is built and is on each script's forbidden-feature list. A test confirms that three different districts with identical feature values give an identical probability.
- Names resolve to the CHIRPS spelling used by the dataset. `Pathanamthitta`, `pattanamtitta` and `  PATHANAMTHITTA ` all map to `Pattanamtitta`. An unknown district gives 422, listing the 14 valid names (404 on the observation endpoint).
- If `latitude`/`longitude` are omitted, the district's representative point from `updated_data/reference/Kerala_district_reference.csv` is used. That's the same point the real Master Dataset uses, so it's a lookup, not an estimate.
- The frontend gets real inputs with `GET /districts/{district}/observation`: a row of `updated_data/master_dataset/real_master_dataset.csv`, read unchanged. The response includes the recorded labels for that day and its split (train = in-sample for the models).

## 6. Missing river level

- The real flood model was trained with `river_level_m` missing on 38.9 % of district-days. The trees learned a default branch for missing values.
- The API accepts `river_level_m: null` **only** for features that were missing in that model's own training data. This is read from `missing_values_in_model_columns` in the results file, not hard-coded. The value is passed to XGBoost as NaN, so nothing is imputed or synthesised.
- Each prediction lists these features in `missing_features`, and the UI shows "null (missing) → scored by XGBoost's learned missing-value branch".
- A null in any other feature (e.g. `soil_moisture`) is rejected with 422.
- Verified: the API probability for Wayanad on 30 Jul 2024 (no gauge) equals the training script's saved test prediction (0.607012).

## 7. Frontend changes

- There was no existing path from the UI to the prediction endpoints (`predictFlood`/`predictLandslide` were defined but never called), so one panel was added to the Model Performance page, built from the existing terminal components (`Panel`, `Field`, `Meter`, `crt-select`/`crt-input`, `TermLoading`/`TermError`/`TermEmpty`).
- It shows:
  - both real probabilities with their Warning / No Warning status and the threshold
  - the model file used
  - the exact inputs, with missing values highlighted
  - the day's split and recorded event
  - the model and calibration notes
- The page's metrics now come from the real models, and the banner and two captions follow `data_mode`.
- Prediction calls are live-only. If the backend is down, the panel shows an error and never a fixture.

## 8. Tests performed

| Test | Result |
|---|---|
| `python backend/tests/test_real_integration.py` | **16/16 passed** |
| — startup: real + synthetic models, districts and TCDL load; `/health` = ok | pass |
| — feature order: `/models/status` = `features.json` = booster `feature_names`; no `district` | pass |
| — API probabilities = training script's saved test predictions (flood and landslide, 6 districts × 2 days, including missing river level), and = direct booster output within 5e-7 | pass |
| — real ≠ synthetic probability for the same input | pass |
| — all 14 districts predict (both models), district echoed, probabilities vary by district | pass |
| — district aliases and point lookup; district not a feature; invalid district → 422/404 | pass |
| — missing river level (no-gauge district, explicit null) → 200 with `missing_features`; null elsewhere → 422 | pass |
| — missing inputs, wrong types, strings, booleans, NaN literal, negative rainfall, humidity 150, soil moisture 1.5, aspect 400, bad category, bad date/latitude → 422 JSON, never 500 | pass |
| — missing / corrupt real model file → `/health` degraded, predictions 503, synthetic pipeline still 200 | pass |
| — `/models/performance` real by default, `?model_set=synthetic` still synthetic | pass |
| Regression: 14 synthetic endpoint responses (`/current`, `/trends`, `/warnings`, `/leadtime`, `/evaluation`, `/explain/*`, `/predict/coupled`, …) before vs after | **14/14 byte-identical** |
| Browser (Vite + uvicorn): Alappuzha 2024-12-31, Ernakulam 2018-08-16, Wayanad 2024-07-30 through the panel | values equal the saved model outputs; network log shows `/districts` → observation → `/predict/*`, all 200 |
| Browser: out-of-range date; backend stopped | clean error states; no fabricated prediction |
| `npm run build` (tsc + vite) | passes |

Example outputs (real models, shown in the UI):

| District · day | Split | Flood p | Landslide p | Recorded |
|---|---|---|---|---|
| Ernakulam · 2018-08-16 | train (in-sample) | 0.672 Warning | 0.915 Warning | flood 1, landslide 1 |
| Wayanad · 2024-07-30 (no gauge) | test | 0.607 Warning | 0.196 No Warning | flood 1, landslide 1 |
| Alappuzha · 2024-12-31 (no gauge) | test | 0.321 No Warning | 0.013 No Warning | 0, 0 |

## 9. Known limitations

- **TCDL not integrated.** `/predict/coupled`, `/current`, `/trends`, `/warnings`, `/leadtime` and `/evaluation` still run on the synthetic models and data. SHAP (`/explain/*`) is still synthetic.
- The landslide model rests on very few positive cases: 8 test positives on 4 dates. It misses the recorded 30 Jul 2024 Wayanad landslide (p = 0.196), one of its 3 test misses. The API serves the model as trained; this is a research-evaluation limitation, not an API fault.
- Probabilities are uncalibrated (inflated by `scale_pos_weight`) and the 0.50 threshold is untuned. For example, the flood model flags about 30 % of test days.
- Predictions are per district and day. Static terrain features have one value per district. Predictions on training-split days are in-sample; the UI states the split.
- Pre-existing, unchanged:
  - one ESLint error (`any` in the fixture cast, `frontend/src/services/api.ts`)
  - the >500 kB bundle warning
  - the Landing page's decorative "tcdl@v2 · uptime 99.98%" text
  - the offline fixture is synthetic, so if the backend is down the Model Performance metrics fall back to synthetic values (correctly labelled)

## 10. Confirmation

- FastAPI starts; the real Flood and Landslide models load from `models/real/`, and both prediction endpoints work for all 14 districts.
- The frontend talks to FastAPI and displays predictions produced by the **real** models.
- The synthetic models, TCDL, SHAP, the ML training code, the model files and all datasets are **unchanged**. No model was retrained and no synthetic data was created.
- **TCDL has not been integrated with the real models.**
