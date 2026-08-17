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

District names in this file describe the **boundary polygons only**. They are
never attached to a monitoring point: the synthetic dataset has no district
field, and the frontend does not infer one from coordinates. A marker shows a
district only when the backend supplies `district` for that location.

### Note on marker placement

The synthetic monitoring coordinates are simulated, not surveyed Kerala sites.
**Four of the nine fall inside a Kerala district polygon; five fall east of the
state boundary.** They are drawn at their true coordinates rather than moved,
because relocating them would fabricate data. The dashboard states this beneath
the map.

## `devFixture.json`

API-compatible development fixture: **recorded GET responses** from this
project's own FastAPI backend (`backend/main.py`) running in synthetic data
mode. Not hand-written values.

Used only by `services/api.ts` when the live API is unreachable, so the UI can
be developed without the backend running. Every response is labelled
`sourceMode: "fixture"` and the dashboard banner says so explicitly.

To refresh it, start the backend and re-record the responses listed under the
`recorded_from` key inside the file.
