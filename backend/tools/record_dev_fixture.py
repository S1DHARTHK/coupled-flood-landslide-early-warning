"""
Re-record frontend/src/data/devFixture.json from the running backend code.

The frontend falls back to this file only when the live API is unreachable,
so it must hold RECORDED responses of this project's own backend -- never
hand-written values. Re-run it after retraining or re-running TCDL / SHAP:

    python backend/tools/record_dev_fixture.py

It starts the FastAPI app in-process (no server needed), records the GET
responses the dashboard reads, and writes them to the fixture. The data mode
recorded is whatever the backend serves (real by default; set
CAPSTONE_DASHBOARD_DATA=synthetic to record the synthetic pipeline).
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

import backend.main as bm  # noqa: E402

OUT = ROOT / "frontend" / "src" / "data" / "devFixture.json"
TRENDS_LIMIT = 200          # Analysis page requests 200, the dashboard 120
WARNINGS_LIMIT = 400        # most recent warning rows kept offline
LEADTIME_LIMIT = 2000


def get(client: TestClient, path: str, **params) -> dict:
    r = client.get(path, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} {params} -> HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def main() -> int:
    with TestClient(bm.app) as c:
        health = get(c, "/health")
        if health["status"] != "ok":
            print(f"backend is degraded, refusing to record: {health['errors']}")
            return 1
        locations = get(c, "/locations")
        ids = [loc["location_id"] for loc in locations["locations"]]

        fixture = {
            "_note": ("API-compatible development fixture. These are RECORDED RESPONSES "
                      "from the project's own FastAPI backend -- not hand-written values. "
                      "Used only when the live API is unreachable. Regenerate with "
                      "`python backend/tools/record_dev_fixture.py`."),
            "recorded_from": (f"GET responses of backend/main.py "
                              f"({health['data_mode']} data mode, dashboard model set "
                              f"'{health.get('dashboard_model_set')}')"),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "health": health,
            "models_status": get(c, "/models/status"),
            "locations": locations,
            "current": get(c, "/current"),
            "warnings": get(c, "/warnings", limit=WARNINGS_LIMIT),
            "evaluation": {scope: get(c, "/evaluation", hazard_type=scope)
                           for scope in ("any_hazard", "flood", "landslide")},
            "leadtime": get(c, "/leadtime", limit=LEADTIME_LIMIT),
            "models_performance": get(c, "/models/performance"),
            "models_features": get(c, "/models/features"),
            "trends": {loc: get(c, "/trends", location_id=loc, limit=TRENDS_LIMIT)
                       for loc in ids},
            "shap": {
                "_note": "Recorded /explain/* responses (latest day and test-period peak).",
                "global": {m: get(c, "/explain/global", model=m)
                           for m in ("flood", "landslide")},
                "current": {mode: {m: {loc: get(c, "/explain/current", model=m,
                                                location_id=loc, mode=mode)
                                       for loc in ids}
                                   for m in ("flood", "landslide")}
                            for mode in ("current", "peak")},
            },
        }

    # Compact: the fixture is bundled into the frontend build.
    OUT.write_text(json.dumps(fixture, separators=(",", ":"), ensure_ascii=False) + "\n",
                   encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"recorded {health['data_mode']} fixture -> {OUT.relative_to(ROOT)} "
          f"({size_kb:.0f} KB, {len(ids)} locations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
