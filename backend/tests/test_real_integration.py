"""
Integration tests: REAL pipeline (XGBoost models, TCDL, SHAP) -> FastAPI.

Run from the project root (plain Python; pytest also collects the test_* functions):
    python backend/tests/test_real_integration.py

What is proven:
  * /predict/flood and /predict/landslide are scored by artifacts/models/*, not synthetic/*:
    API probabilities equal the probabilities the training script saved for the
    same test-split district-days (artifacts/results/*_test_predictions.csv).
  * the feature vector is built in the saved order; district is not a feature;
  * null river_level_m is passed to XGBoost as missing (same result as training);
  * invalid districts / inputs / numbers give clean 4xx JSON errors, never a 500;
  * a missing real model file degrades the API cleanly (503), nothing crashes;
  * the dashboard endpoints (/current, /trends, /locations, /warnings, /leadtime,
    /evaluation, /predict/coupled, /explain/*) serve the REAL pipeline: every
    value equals artifacts/tcdl/tcdl_* / dataset/real_master_dataset.csv read independently;
  * districts and location_ids are interchangeable identifiers; history replays
    by date (as_of); bad dates / locations give 4xx, never 500;
  * SHAP explains exactly the probability the dashboard shows;
  * CAPSTONE_DASHBOARD_DATA=synthetic switches the dashboard back to synthetic/*.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as bm  # noqa: E402

ART = ROOT / "artifacts"
REAL_MODELS, REAL_RESULTS, REAL_TCDL, REAL_SHAP = ART / "models", ART / "results", ART / "tcdl", ART / "shap"
DISTRICTS_UNDER_TEST = ["Ernakulam", "Idukki", "Wayanad", "Alappuzha", "Thiruvananthapuram",
                        "Kozhikode"]
_CLIENT: TestClient | None = None


def client() -> TestClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = TestClient(bm.app)
        _CLIENT.__enter__()                      # runs the lifespan (model loading)
    return _CLIENT


def observation(district: str, date: str | None = None) -> dict:
    r = client().get(f"/districts/{district}/observation", params={"date": date} if date else {})
    assert r.status_code == 200, r.text
    return r.json()


def predict(hazard: str, records: list) -> "requests.Response":  # noqa: F821
    return client().post(f"/predict/{hazard}", json={"observations": records})


def problems_of(r) -> list[str]:
    detail = r.json()["detail"]
    return detail.get("problems", []) if isinstance(detail, dict) else [str(detail)]


# ---------------------------------------------------------------------------------------------
# 1. startup and model loading
# ---------------------------------------------------------------------------------------------
def test_startup_loads_real_and_synthetic_models():
    h = client().get("/health").json()
    assert h["status"] == "ok", h
    assert h["ready_for_prediction"] and h["flood_model_loaded"] and h["landslide_model_loaded"]
    assert h["prediction_model_set"] == "real" and h["synthetic_models_loaded"]
    assert h["districts_loaded"] and h["tcdl_available"] and h["errors"] == []


def test_real_artifacts_and_feature_order():
    s = client().get("/models/status").json()
    for hazard in ("flood", "landslide"):
        block = s[f"{hazard}_model"]
        assert block["model_set"] == "real"
        assert block["artifact"] == f"artifacts/models/{hazard}_xgboost_model.json"
        saved = json.loads((REAL_MODELS / f"{hazard}_features.json").read_text())["feature_order"]
        booster = xgb.Booster()
        booster.load_model(str(REAL_MODELS / f"{hazard}_xgboost_model.json"))
        assert block["feature_order"] == saved == list(booster.feature_names), hazard
        assert "district" not in block["feature_order"]
    assert s["flood_model"]["nullable_features"] == ["river_level_m"]
    assert s["landslide_model"]["nullable_features"] == []
    assert s["tcdl"]["model_set"] == "real" and s["dashboard_model_set"] == "real"
    assert s["tcdl"]["thresholds_source"].startswith("artifacts/tcdl/tcdl_results.json")
    assert s["shap"]["available"] and s["shap"]["dataset"] == "real"


# ---------------------------------------------------------------------------------------------
# 2. predictions come from the REAL models (match the training script's saved output)
# ---------------------------------------------------------------------------------------------
def _match_saved_test_predictions(hazard: str):
    saved = pd.read_csv(REAL_RESULTS / f"{hazard}_test_predictions.csv")
    saved = saved[saved["district"].isin(DISTRICTS_UNDER_TEST)]
    # a spread of days per district: highest-probability day, a missing-river day, the first day
    picks = []
    for d, g in saved.groupby("district"):
        picks += [g.sort_values(f"{hazard}_probability").iloc[-1], g.iloc[0]]
    obs_rows, expected = [], []
    for p in picks:
        o = observation(p["district"], p["date"])
        obs_rows.append(o["observation"])
        expected.append(float(p[f"{hazard}_probability"]))
    r = predict(hazard, obs_rows)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_set"] == "real" and body["data_mode"] == "real"
    assert body["synthetic_data_warning"] is None
    got = [x["probability"] for x in body["predictions"]]
    # (a) the saved CSV was rounded to 6 dp from float32 separately, so a value
    #     sitting on a rounding boundary may differ by one unit in the last digit
    assert all(abs(a - b) <= 1e-6 + 1e-9 for a, b in zip(got, expected)), list(zip(got, expected))
    # (b) exact check against the real booster scored directly on the stored rows
    master = pd.read_csv(bm.config.REAL_MASTER_DATASET)
    feats = json.loads((REAL_MODELS / f"{hazard}_features.json").read_text())
    rows = pd.concat([master[(master["district"] == p["district"]) & (master["date"] == p["date"])]
                      for p in picks])[feats["feature_order"]]
    for col, cats in feats.get("categorical_handling", {}).get("category_schema", {}).items():
        rows[col] = rows[col].astype(pd.CategoricalDtype(categories=cats))
    booster = xgb.Booster()
    booster.load_model(str(REAL_MODELS / f"{hazard}_xgboost_model.json"))
    raw = booster.predict(xgb.DMatrix(rows, enable_categorical=hazard == "landslide"))
    assert all(abs(a - float(b)) <= 5e-7 for a, b in zip(got, raw)), list(zip(got, raw))
    return body


def test_flood_predictions_equal_training_output():
    body = _match_saved_test_predictions("flood")
    assert body["model_artifact"] == "artifacts/models/flood_xgboost_model.json"


def test_landslide_predictions_equal_training_output():
    body = _match_saved_test_predictions("landslide")
    assert body["model_artifact"] == "artifacts/models/landslide_xgboost_model.json"


def test_real_model_differs_from_synthetic_model():
    """Same input through the synthetic booster gives a different probability."""
    o = observation("Ernakulam", "2018-08-16")["observation"]
    real_p = predict("flood", [o]).json()["predictions"][0]["probability"]
    syn = bm.STATE["models"]
    syn_p = float(syn.predict_flood([o])[0])
    assert abs(real_p - syn_p) > 1e-3, (real_p, syn_p)


# ---------------------------------------------------------------------------------------------
# 3. districts
# ---------------------------------------------------------------------------------------------
def test_every_district_predicts():
    names = [d["district"] for d in client().get("/districts").json()["districts"]]
    assert len(names) == 14
    obs = [observation(n, "2024-07-30")["observation"] for n in names]
    for hazard in ("flood", "landslide"):
        r = predict(hazard, obs)
        assert r.status_code == 200, r.text
        preds = r.json()["predictions"]
        assert [p["district"] for p in preds] == names
        assert all(0.0 <= p["probability"] <= 1.0 for p in preds)
        assert len({p["probability"] for p in preds}) > 1          # varies by district


def test_district_aliases_and_point_lookup():
    o = observation("Pattanamtitta", "2023-06-01")["observation"]
    for alias in ("Pathanamthitta", "pattanamtitta", "  PATHANAMTHITTA "):
        rec = {k: v for k, v in o.items() if k not in ("latitude", "longitude")}
        rec["district"] = alias
        r = predict("flood", [rec])
        assert r.status_code == 200, r.text
        p = r.json()["predictions"][0]
        assert p["district"] == "Pattanamtitta"
        assert (p["latitude"], p["longitude"]) == (o["latitude"], o["longitude"])


def test_district_is_not_a_model_feature():
    o = observation("Idukki", "2019-08-08")["observation"]
    probs = set()
    for name in ("Idukki", "Wayanad", "Kasaragod"):
        r = predict("landslide", [dict(o, district=name)])
        assert r.status_code == 200
        probs.add(r.json()["predictions"][0]["probability"])
    assert len(probs) == 1, probs


def test_invalid_district():
    o = observation("Idukki", "2019-08-08")["observation"]
    r = predict("flood", [dict(o, district="Atlantis")])
    assert r.status_code == 422
    assert any("Unknown district" in p for p in problems_of(r))
    r = client().get("/districts/Atlantis/observation")
    assert r.status_code == 404 and r.json()["detail"]["valid_districts"]
    r = client().get("/districts/Idukki/observation", params={"date": "2030-01-01"})
    assert r.status_code == 404
    r = client().get("/districts/Idukki/observation", params={"date": "16-08-2018"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------------------------
# 4. missing river level and missing / invalid inputs
# ---------------------------------------------------------------------------------------------
def test_missing_river_level_is_passed_as_missing():
    o = observation("Wayanad", "2024-07-30")               # Wayanad has no CWC gauge
    assert o["observation"]["river_level_m"] is None and o["missing_features"] == ["river_level_m"]
    r = predict("flood", [o["observation"]])
    assert r.status_code == 200, r.text
    p = r.json()["predictions"][0]
    assert p["missing_features"] == ["river_level_m"]
    saved = pd.read_csv(REAL_RESULTS / "flood_test_predictions.csv")
    exp = saved[(saved["district"] == "Wayanad") & (saved["date"] == "2024-07-30")]
    assert abs(p["probability"] - float(exp["flood_probability"].iloc[0])) < 1e-6
    # an explicit null on a gauged district is scored through the same missing branch
    e = observation("Ernakulam", "2024-07-30")["observation"]
    r2 = predict("flood", [dict(e, river_level_m=None)])
    assert r2.status_code == 200 and r2.json()["predictions"][0]["missing_features"] == ["river_level_m"]
    # the landslide model does not use river_level_m at all
    r3 = predict("landslide", [o["observation"]])
    assert r3.status_code == 200 and r3.json()["predictions"][0]["missing_features"] == []


def test_missing_required_inputs():
    o = observation("Kollam", "2023-10-15")["observation"]
    no_rain = {k: v for k, v in o.items() if k != "rainfall_7d_mm"}
    r = predict("flood", [no_rain])
    assert r.status_code == 422 and any("rainfall_7d_mm" in p for p in problems_of(r))
    r = predict("flood", [dict(o, soil_moisture=None)])      # not nullable
    assert r.status_code == 422 and any("may not be null" in p for p in problems_of(r))
    r = predict("landslide", [dict(o, slope_degree=None)])
    assert r.status_code == 422
    no_date = {k: v for k, v in o.items() if k != "date"}
    assert predict("flood", [no_date]).status_code == 422
    no_place = {k: v for k, v in o.items() if k not in ("latitude", "longitude", "district")}
    r = predict("flood", [no_place])
    assert r.status_code == 422 and any("latitude" in p for p in problems_of(r))
    assert client().post("/predict/flood", json={"observations": []}).status_code == 422
    assert client().post("/predict/flood", json={"observations": "x"}).status_code == 422
    assert client().post("/predict/flood", json={}).status_code == 422
    assert predict("flood", [42]).status_code == 422


def test_invalid_numeric_values():
    o = observation("Kannur", "2022-07-01")["observation"]
    bad = [("rainfall_1d_mm", "abc"), ("rainfall_1d_mm", True), ("rainfall_3d_mm", -5.0),
           ("humidity_percent", 150.0), ("soil_moisture", 1.5), ("temperature_c", "25")]
    for field, value in bad:
        r = predict("flood", [dict(o, **{field: value})])
        assert r.status_code == 422, (field, value, r.status_code)
        assert any(field in p for p in problems_of(r)), (field, problems_of(r))
    for field, value in [("aspect_degree", 400.0), ("slope_degree", -1.0),
                         ("land_cover", "desert"), ("lithology", 3)]:
        r = predict("landslide", [dict(o, **{field: value})])
        assert r.status_code == 422, (field, value)
    # non-finite JSON literals
    body = json.dumps({"observations": [o]}).replace('"rainfall_1d_mm": ' + json.dumps(
        o["rainfall_1d_mm"]), '"rainfall_1d_mm": NaN')
    r = client().post("/predict/flood", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    for key, value in (("date", "2022/07/01"), ("latitude", 123.0)):
        assert predict("flood", [dict(o, **{key: value})]).status_code == 422


def test_errors_are_json_never_500():
    o = observation("Kannur", "2022-07-01")["observation"]
    for payload in ([dict(o, rainfall_1d_mm={"x": 1})], [dict(o, river_level_m="n/a")], [None]):
        r = predict("flood", payload)
        assert r.status_code == 422 and isinstance(r.json()["detail"], (dict, str))


# ---------------------------------------------------------------------------------------------
# 5. model-loading failure is reported, not fatal
# ---------------------------------------------------------------------------------------------
def test_missing_real_model_degrades_cleanly():
    global _CLIENT
    if _CLIENT is not None:                      # this test starts its own app instance
        _CLIENT.__exit__(None, None, None)
        _CLIENT = None
    original = copy.deepcopy(bm.config.REAL_MODEL_SET)
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken_model.json"
        broken.write_text("{not a model")
        bm.config.REAL_MODEL_SET["flood_model"] = Path(tmp) / "does_not_exist.json"
        bm.config.REAL_MODEL_SET["landslide_model"] = broken
        try:
            with TestClient(bm.app) as c:
                h = c.get("/health").json()
                assert h["status"] == "degraded" and not h["ready_for_prediction"]
                assert not h["flood_model_loaded"] and not h["landslide_model_loaded"]
                assert any("real flood model" in e for e in h["errors"])
                o = c.get("/districts/Idukki/observation", params={"date": "2019-08-08"}).json()
                for hazard in ("flood", "landslide"):
                    r = c.post(f"/predict/{hazard}", json={"observations": [o["observation"]]})
                    assert r.status_code == 503 and "not loaded" in r.json()["detail"]["error"]
                # the precomputed dashboard outputs keep being served
                assert c.get("/current").status_code == 200
        finally:
            bm.config.REAL_MODEL_SET.clear()
            bm.config.REAL_MODEL_SET.update(original)


# ---------------------------------------------------------------------------------------------
# 6. metadata endpoints and the untouched synthetic pipeline
# ---------------------------------------------------------------------------------------------
def test_performance_endpoints_serve_real_by_default():
    real = json.loads((REAL_RESULTS / "flood_model_results.json").read_text())
    syn = json.loads((ROOT / "synthetic" / "results" / "flood_model_results.json").read_text())
    p = client().get("/models/performance").json()
    assert p["data_mode"] == "real" and p["synthetic_data_warning"] is None
    assert p["models"]["flood"]["splits"]["test"] == real["metrics"]["test"]
    ps = client().get("/models/performance", params={"model_set": "synthetic"}).json()
    assert ps["data_mode"] == "synthetic" and ps["models"]["flood"]["splits"]["test"] == syn["metrics"]["test"]
    assert client().get("/models/performance", params={"model_set": "other"}).status_code == 422
    f = client().get("/models/features").json()
    assert f["feature_contract"]["flood"]["nullable_features"] == ["river_level_m"]


# ---------------------------------------------------------------------------------------------
# 7. the dashboard serves the REAL pipeline
# ---------------------------------------------------------------------------------------------
_TS: pd.DataFrame | None = None


def real_timeseries() -> pd.DataFrame:
    """artifacts/tcdl/tcdl_timeseries.csv, read independently of the backend."""
    global _TS
    if _TS is None:
        _TS = pd.read_csv(REAL_TCDL / "tcdl_timeseries.csv")
    return _TS


def test_dashboard_serves_real_pipeline():
    assert bm.STATE["dashboard"] is bm.STATE["real_models"]
    assert bm.STATE["tcdl"].models is bm.STATE["real_models"]
    for path, params in (("/current", {}), ("/warnings", {"limit": 5}), ("/evaluation", {}),
                         ("/explain/global", {"model": "flood"}), ("/locations", {}),
                         ("/trends", {"location_id": "Idukki", "limit": 5}),
                         ("/leadtime", {"limit": 5})):
        body = client().get(path, params=params).json()
        assert body["data_mode"] == "real", path
        assert body.get("synthetic_data_warning") is None, path
    h = client().get("/health").json()
    assert h["data_mode"] == "real" and h["dashboard_model_set"] == "real"
    assert h["shap_available"] and "REAL DATA" in h["data_notice"]
    saved = json.loads((REAL_TCDL / "tcdl_results.json").read_text())
    for scope in ("any_hazard", "flood", "landslide"):
        ev = client().get("/evaluation", params={"hazard_type": scope}).json()
        assert ev["systems"] == saved["results"][scope]
        assert ev["source"] == "artifacts/tcdl/tcdl_results.json"
    assert ev["parameters"]["resolved_thresholds"] == saved["parameters"]["resolved_thresholds"]


def test_current_equals_real_outputs():
    ts = real_timeseries()
    master = pd.read_csv(bm.config.REAL_MASTER_DATASET)
    for day in ("2018-08-16", "2024-07-30", None):
        body = client().get("/current", params={"as_of": day} if day else {}).json()
        want = day or ts["date"].max()
        assert body["as_of"] == want and body["n_locations"] == 14
        assert body["available_date_range"] == [ts["date"].min(), ts["date"].max()]
        for loc in body["locations"]:
            row = ts[(ts["location_id"] == loc["location_id"]) & (ts["date"] == want)].iloc[0]
            assert loc["district"] == row["district"] == loc["environment"]["district"]
            assert abs(loc["flood"]["probability"] - row["flood_probability"]) < 1e-6
            assert abs(loc["landslide"]["probability"] - row["landslide_probability"]) < 1e-6
            assert (loc["tcdl"]["warning_status"] == "Warning") == bool(row["tcdl_warning"])
            assert loc["recorded_labels"] == {"flood": int(row["flood"]),
                                              "landslide": int(row["landslide"])}
            m = master[(master["district"] == row["district"]) & (master["date"] == want)].iloc[0]
            env = loc["environment"]
            assert env["common_environmental"]["rainfall_1d_mm"] == round(m["rainfall_1d_mm"], 6)
            assert env["landslide_specific"]["lithology"] == m["lithology"]
            if pd.isna(m["river_level_m"]):
                assert env["flood_specific"]["river_level_m"] is None
                assert "river_level_m" in env["unavailable_fields"]
    # Wayanad has no CWC gauge: river level is reported missing, never filled
    w = client().get("/current", params={"location_id": "Wayanad", "as_of": "2024-07-30"}).json()
    assert w["n_locations"] == 1
    assert w["locations"][0]["environment"]["flood_specific"]["river_level_m"] is None


def test_districts_and_location_ids_are_interchangeable():
    locs = client().get("/locations").json()["locations"]
    assert len(locs) == 14 and all(loc["district"] for loc in locs)
    idukki = next(loc for loc in locs if loc["district"] == "Idukki")
    a = client().get("/trends", params={"location_id": "Idukki", "limit": 20}).json()
    b = client().get("/trends", params={"location_id": idukki["location_id"], "limit": 20}).json()
    c = client().get("/trends", params={"location_id": "  idukki ", "limit": 20}).json()
    assert a == b == c and a["district"] == "Idukki"
    # geoBoundaries spelling resolves to the CHIRPS spelling the data uses
    p1 = client().get("/current", params={"location_id": "Pathanamthitta"}).json()
    assert p1["locations"][0]["district"] == "Pattanamtitta"
    r = client().get("/trends", params={"location_id": "Atlantis"})
    assert r.status_code == 404 and "Idukki" in r.json()["detail"]["valid_locations"]
    assert client().get("/warnings/Atlantis/2018-08-16").status_code == 404
    assert client().get("/leadtime", params={"location_id": "Atlantis"}).status_code == 404


def test_history_replay_and_date_validation():
    t = client().get("/trends", params={"location_id": "Wayanad", "end": "2024-07-30",
                                        "limit": 10}).json()
    ts = real_timeseries()
    want = ts[(ts["district"] == "Wayanad") & (ts["date"] <= "2024-07-30")].tail(10)
    assert [x["date"] for x in t["series"]] == list(want["date"])
    assert [x["flood"] for x in t["series"]] == list(want["flood"])
    for path, params in (("/current", {"as_of": "2024/07/30"}),
                         ("/trends", {"location_id": "Idukki", "end": "30-07-2024"}),
                         ("/warnings", {"start": "yesterday"}),
                         ("/explain/current", {"location_id": "Idukki", "date": "2024-7-3x"})):
        r = client().get(path, params=params)
        assert r.status_code == 422, (path, r.status_code)
    assert client().get("/warnings/Idukki/16-08-2018").status_code == 422
    r = client().get("/current", params={"as_of": "2000-01-01"})
    assert r.status_code == 404 and r.json()["detail"]["available_date_range"]


def test_warnings_filters_match_outputs():
    ts = real_timeseries()
    body = client().get("/warnings", params={"location_id": "Idukki", "start": "2018-08-01",
                                             "end": "2018-08-31", "limit": 5000}).json()
    want = ts[(ts["district"] == "Idukki") & (ts["date"] >= "2018-08-01")
              & (ts["date"] <= "2018-08-31") & (ts["tcdl_warning"] == 1)]
    assert body["n_records"] == body["n_matching"] == len(want) and not body["truncated"]
    assert [r["date"] for r in body["records"]] == list(want["date"])
    assert {r["district"] for r in body["records"]} == {"Idukki"}
    assert body["n_location_days_in_range"] == 31
    every = client().get("/warnings", params={"location_id": "Idukki", "start": "2018-08-01",
                                              "end": "2018-08-31", "warning_only": False}).json()
    assert every["n_records"] == 31
    small = client().get("/warnings", params={"limit": 3}).json()
    assert small["n_records"] == 3 and small["truncated"]
    assert small["n_matching"] == int((ts["tcdl_warning"] == 1).sum())
    test_only = client().get("/warnings", params={"split": "test", "limit": 20000}).json()
    assert {r["split"] for r in test_only["records"]} == {"test"}


def test_leadtime_equals_outputs():
    lt = pd.read_csv(REAL_TCDL / "tcdl_lead_time.csv")
    body = client().get("/leadtime", params={"system": "tcdl_coupled__vs_flood",
                                             "hazard_type": "flood", "limit": 5000}).json()
    want = lt[(lt["system"] == "tcdl_coupled__vs_flood") & (lt["hazard_type"] == "flood")]
    assert len(want) > 0
    assert body["n_records"] == len(want)
    assert [r["district"] for r in body["records"]] == list(want["district"])
    assert [r["detected"] for r in body["records"]] == list(want["detected"])
    one = client().get("/leadtime", params={"location_id": "Kozhikode", "limit": 5000}).json()
    assert {r["district"] for r in one["records"]} == {"Kozhikode"}


def test_shap_explains_the_displayed_probability():
    for model, district, day in (("flood", "Ernakulam", "2018-08-16"),
                                 ("landslide", "Wayanad", "2024-07-30"),
                                 ("flood", "Kottayam", "2023-10-01")):
        cur = client().get("/current", params={"location_id": district, "as_of": day}).json()
        shown = cur["locations"][0][model]["probability"]
        e = client().get("/explain/current", params={"model": model, "location_id": district,
                                                     "date": day})
        assert e.status_code == 200, e.text
        e = e.json()
        assert e["data_mode"] == "real" and e["sample"]["district"] == district
        assert e["sample"]["date"] == day and abs(e["probability"] - shown) < 1e-6
        assert e["additivity_check"]["passed"]
    peak = client().get("/explain/current", params={"model": "landslide",
                                                    "location_id": "Idukki", "mode": "peak"}).json()
    assert peak["sample"]["split"] == "test"
    for model in ("flood", "landslide"):
        g = client().get("/explain/global", params={"model": model}).json()
        saved = json.loads((REAL_SHAP / f"{model}_shap_importance.json").read_text())
        assert g["importance"] == saved["global_importance"]
    r = client().get("/explain/current", params={"location_id": "Idukki", "date": "1999-01-01"})
    assert r.status_code == 404


def test_coupled_on_demand_equals_precomputed():
    """A real 11-day series through /predict/coupled reproduces the stored TCDL row."""
    ts = real_timeseries()
    obs: list = []
    for district, month, last in (("Wayanad", "2024-07", 30), ("Idukki", "2018-08", 16)):
        days = [f"{month}-{d:02d}" for d in range(last - 10, last + 1)]
        obs = [observation(district, d)["observation"] for d in days]
        r = client().post("/predict/coupled", json={"observations": obs})
        assert r.status_code == 200, r.text
        body = r.json()
        row = ts[(ts["district"] == district) & (ts["date"] == days[-1])].iloc[0]
        assert abs(body["flood_probability"] - row["flood_probability"]) < 1e-6
        assert abs(body["landslide_probability"] - row["landslide_probability"]) < 1e-6
        assert (body["warning_status"] == "Warning") == bool(row["tcdl_warning"])
        want_rules = [x for x in str(row["tcdl_triggered_rules"]).split("|")
                      if x and x != "nan"]
        assert sorted(body["explanation"]["triggered_rules"]) == sorted(want_rules), (
            body["explanation"]["triggered_rules"], row["tcdl_triggered_rules"])
    short = [observation("Idukki", f"2018-08-{d:02d}")["observation"] for d in (14, 15)]
    assert client().post("/predict/coupled", json={"observations": short}).status_code == 422
    bad = [dict(o, soil_moisture=None) for o in obs]
    assert client().post("/predict/coupled", json={"observations": bad}).status_code == 422


def test_synthetic_dashboard_switch():
    """CAPSTONE_DASHBOARD_DATA=synthetic serves the old pipeline; bad values fail fast."""
    code = ("import sys, json; sys.path.insert(0, '.')\n"
            "from fastapi.testclient import TestClient\n"
            "import backend.main as bm\n"
            "with TestClient(bm.app) as c:\n"
            "    cur = c.get('/current').json(); h = c.get('/health').json()\n"
            "    print(json.dumps([cur['data_mode'], cur['n_locations'],"
            " h['dashboard_model_set']]))\n")
    env = dict(os.environ, CAPSTONE_DASHBOARD_DATA="synthetic")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    mode, n, dash = json.loads(out.stdout.strip().splitlines()[-1])
    assert (mode, dash) == ("synthetic", "synthetic") and n == 9
    env["CAPSTONE_DASHBOARD_DATA"] = "bogus"
    bad = subprocess.run([sys.executable, "-c",
                          "import sys; sys.path.insert(0, '.'); import backend.config"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert bad.returncode != 0 and "CAPSTONE_DASHBOARD_DATA" in bad.stderr


# ---------------------------------------------------------------------------------------------
def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if _CLIENT is not None:
        _CLIENT.__exit__(None, None, None)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
