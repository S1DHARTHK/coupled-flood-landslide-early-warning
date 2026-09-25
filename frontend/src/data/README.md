# Frontend data files

## `Kerala_districts.geojson`

Kerala district boundaries used by `components/KeralaMap/KeralaMap.tsx`.

- **Source:** [github.com/opendatakerala/kl_district](https://github.com/opendatakerala/kl_district)
  (`Kerala_districts.geojson`), derived from OpenStreetMap.
- **Licence:** ODbL 1.0 — © OpenStreetMap contributors. Attribution is rendered
  on the map itself via the Leaflet attribution control.
- **Processing applied** (no geometry was redrawn or invented):
  1. Filtered from 845 raw features to the **14 `admin_level=5` administrative
     polygons**. The raw export also contained hundreds of LineStrings (river
     and border segments) and Points (settlement labels), plus one unnamed
     coastline islet polygon — all irrelevant to district boundaries.
  2. Normalised the `name` property into `district`, stripping the trailing
     `" district"` suffix that OSM applies inconsistently (e.g.
     `"Thrissur district"` → `"Thrissur"`).
  3. Rounded coordinates to 5 decimal places (~1 m precision).
- **Result:** 4.3 MB → 553 KB.

District names in this file describe the **boundary polygons only**. The
frontend never infers a district from coordinates: a marker shows a district
only when the backend supplies `district` for that location.

### Note on marker placement

- **Real data (default):** one marker per Kerala district (14), drawn at the
  representative point the real Master Dataset uses for that district. The data
  are district-level, so a marker stands for the whole district, not one site.
- **Synthetic data** (`CAPSTONE_DASHBOARD_DATA=synthetic` on the backend): the
  nine simulated coordinates are drawn where they are; five fall east of the
  state boundary. They are not moved, because relocating them would fabricate
  data.

## `devFixture.json`

API-compatible development fixture: **recorded GET responses** from this
project's own FastAPI backend (`backend/main.py`), currently in real data mode
(`recorded_from` / `recorded_at` inside the file say which). Not hand-written
values.

Used only by `services/api.ts` when the live API is unreachable, so the UI can
be developed without the backend running. Every response is labelled
`sourceMode: "fixture"` and the banner says so explicitly. The recording holds
the latest day only; a request it cannot answer exactly (another day, another
filter) shows an error instead of substituted data.

To refresh it after retraining or re-running TCDL / SHAP (no server needed):

```bash
python backend/tools/record_dev_fixture.py
```
