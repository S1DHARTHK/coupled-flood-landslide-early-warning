# Prompt: build the REAL Master Dataset (Kerala Flood–Landslide Early Warning)

Project root: `D:\work\capstone`. All real data is in `D:\work\capstone\updated_data\`.
Read these first (they are the source of truth for provenance and caveats):
`updated_data/documentation/NEW_DATA_SOURCES.md`, `updated_data/DATA_SOURCE_AND_CORRECTION_REPORT.docx`,
`updated_data/DATA_VERIFICATION_REPORT.docx`, `updated_data/river_level/STATUS_CWC_COLLECTED.md`,
`updated_data/flood_events/README.md`, `updated_data/landslide_events/README.md`,
`updated_data/lithology/processed/README.md`, and the ML code `ml/flood_xgboost.py`, `ml/landslide_xgboost.py`, `ml/tcdl_v1.py`, `DATA_DICTIONARY.md`.

## Task
Build the real Master Dataset from the existing real data, exactly in the schema the existing ML code expects.
Write a reproducible build script so it can be re-run when new data (e.g. a GSI Bhukosh landslide inventory) is added later.

## Hard rules
- Do NOT train models. Do NOT modify ML code (`ml/`), backend, frontend, or `synthetic_master_dataset.csv`.
- Do NOT modify or delete any existing data file. Write only new files.
- No synthetic, estimated, interpolated, forward-filled or imputed values. Missing stays NaN (the code passes NaN to XGBoost).
- Never assign a date to a record that does not state one. No random negative sampling.
- Any value computed from the data (e.g. a gauge median) must use the TRAINING period only.

## Grid and schema
- One row per (district, date): 14 Kerala districts × every day 2012-01-30 → 2024-12-31 = **66,080 rows**
  (2012-01-01..01-29 is the 29-day warm-up of rainfall_30d_mm; use it for the rolling sums, then drop it).
- Split used by the code (`chronological_split`, 70/15/15 on unique dates) on this grid:
  train 2012-01-30 → 2021-02-14 · validation 2021-02-15 → 2023-01-23 · test 2023-01-24 → 2024-12-31. Recompute and confirm.
- District vocabulary = CHIRPS names (Pattanamtitta, not Pathanamthitta). Map IFI/IMD/GSI spellings via
  `updated_data/reference/Kerala_district_reference.csv` (district_chirps ↔ district_geoboundaries) — IDUKKI / "Idukki (Devikulam Taluk)" → Idukki, Plakkad → Palakkad, Pathanamthitta → Pattanamtitta.
- Output columns, exactly this order and names (23 = the 22 synthetic columns + `district` after `longitude`):
  `date, latitude, longitude, district, rainfall_1d_mm, rainfall_3d_mm, rainfall_7d_mm, rainfall_14d_mm, rainfall_30d_mm, soil_moisture, temperature_c, humidity_percent, elevation_m, river_level_m, distance_to_river_km, drainage_density, slope_degree, aspect_degree, curvature, land_cover, lithology, flood, landslide`
- `district` (CHIRPS vocabulary, string) is an identifier only — never a model feature. This is safe: every ML script,
  TCDL, SHAP and the backend select features by explicit lists (`FLOOD_FEATURES`, `LANDSLIDE_FEATURES`, `feature_order`),
  so the extra column is ignored; `district` must NOT appear in any feature list. (The frontend types already declare an
  optional `district` field for future real data.)
- `land_cover` must be one of {forest, agriculture, shrubland, built_up, barren}; `lithology` one of {granite_gneiss, laterite, schist, sandstone, shale, alluvium}; `flood`, `landslide` integer 0/1 with no NaN.
- Put gauge used, raw published stage and provenance/flag columns in a separate companion "keys" file with the same row order (not in the master file).

## Column sources
| Column | Source file (under updated_data/) | Rule |
|---|---|---|
| date | CHIRPS | daily, gap-free |
| latitude, longitude | reference/Kerala_district_reference.csv (latitude, longitude) | district representative point; static per district |
| district | reference/Kerala_district_reference.csv (district_chirps) | CHIRPS district name; identifier only, not a feature |
| rainfall_1d_mm | existing_collected_data/Kerala_CHIRPS_District_Rainfall_2012.csv + _2013_2024.csv (date, district, rainfall) | concatenate; rename |
| rainfall_3d/7d/14d/30d_mm | same | trailing rolling sum per district, inclusive of day t, min_periods = window, sorted by date within district |
| soil_moisture | weather_soil/NASA_POWER_Kerala_District_Daily_2012_2024.csv → GWETROOT | 0–1 root-zone saturation fraction (NOT volumetric like the synthetic data — document) |
| temperature_c | same → T2M | °C |
| humidity_percent | same → RH2M | % |
| elevation_m | terrain/Kerala_District_Terrain_Summary.csv → elevation_m_mean | static per district |
| slope_degree | same → slope_degree_mean | static |
| aspect_degree | same → aspect_degree_circmean | static |
| curvature | same → curvature_mean | static |
| drainage_density | hydrology/Kerala_District_Drainage_Density.csv → drainage_density_km_per_km2 | static |
| distance_to_river_km | hydrology/Kerala_HydroRIVERS_v10.geojson + the district representative point | NOT yet computed: distance in km from the district point to the nearest HydroRIVERS reach (any Strahler order), computed in a metric projection (e.g. EPSG:32643). Document the definition. |
| land_cover | terrain/Kerala_District_LandCover_Summary.csv → dominant_project_land_cover | ("forest" in all 14 — near-constant, document) |
| lithology | lithology/Kerala_District_Lithology_CGWB.csv → lithology | (2 values only — document) |
| river_level_m | river_level/processed/District_Gauge_Assignment.csv (district → primary_gauge) + river_level/processed/CWC_Kerala_RiverLevel_Daily_Station.csv (station, date, level_mean_m, datum_segment) | see decision 1 |
| flood | flood_events/processed/IFI_Kerala_2012_2023_corrected.csv + flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv | see decision 2 |
| landslide | landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv + landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv | see decision 3 |

## Decisions (already agreed — apply exactly; edit here if they change)
1. **river_level_m** = primary CWC gauge of the district, daily `level_mean_m`, expressed as stage relative to that gauge's
   own median over the TRAINING period only (metres above/below the gauge's normal), because gauges sit on different
   datums (e.g. Vandiperiyar ~791 m MSL vs Ayilam ~1.7 m). Check the primary gauge has one `datum_segment` over
   2012–2024; if a break exists, set values after it to NaN rather than mixing references. Districts with no primary
   gauge (Alappuzha, Wayanad, Kottayam) → NaN. Record the gauge per district in the keys file. Also keep the raw
   published stage in the keys file for transparency.
2. **flood**: 1 on every (district, day) in an event's `event_days` (if empty, every day from `verified_start` to
   `verified_end`); use `verified_*`, never the original Start/End Date columns. Include the IMD extension events
   (`event_days`, `districts`). Keep LONG_PERIOD_SUMMARY records (they are the only coverage of the 2018 flood),
   EXCLUDE LANDSLIDE_ONLY_CAUSE records (12) and NO_DISTRICT records (2) — see the `flags` column. 0 elsewhere
   (document: "0 = no damaging flood/heavy-rain event reported by IMD for that district-day").
   Expected before exclusions: 3,132 positive district-days (train 2,764 / val 249 / test 119). Report the final count.
3. **landslide**: exact-date records only. GSI: rows of GSI_Kerala_Landslides_date_precision.csv with non-empty
   `exact_dates` (district from `district_project`; all-records file, filtered to the grid window). IMD: rows of
   IMD_DWE_Kerala_landslide_entries.csv with `date_precision == EXACT_DATE` and non-empty `landslide_districts`.
   Union, de-duplicated at district-day level. 1 on those district-days, 0 elsewhere (presence-only assumption —
   document). Expected ≈ 120 district-days (train 101 / val 11 / test 8). Exclude MONTH_YEAR, YEAR_ONLY, NO_DATE,
   PERIOD and multi-district unattributed records.
4. **Missing values** stay NaN. No imputation anywhere.

## Validation (must all pass, report results)
- 66,080 rows; 14 locations per date; no duplicate (date, latitude, longitude); every district has a gap-free daily series.
- Replicate `validate_columns()` of both training scripts: all required columns present, targets ⊂ {0,1}, categories inside the fixed schemas.
- Rolling-sum reconstruction test (as in `validate_synthetic_dataset.py`): rainfall_Nd_mm equals the sum of the last N rainfall_1d_mm values.
- Missing-value count per column, by split. river_level_m coverage by split (expected ≈ 54 % train / 75 % val / 78 % test of district-days).
- Positive counts per split for flood and landslide; confirm the landslide target has no NaN.
- Leakage check: gauge medians computed only from training dates; no centred/forward windows; no target-derived feature.
- Column order identical to `synthetic_master_dataset.csv` except the added `district` column directly after `longitude`;
  each district maps to exactly one (latitude, longitude) pair; `district` is in no feature list of the ML code.

## Outputs (new files only)
- `updated_data/master_dataset/real_master_dataset.csv` (23 columns: the 22 synthetic columns + `district`)
- `updated_data/master_dataset/real_master_dataset_keys.csv` (date, district, river gauge, raw published stage, label sources/flags)
- `updated_data/master_dataset/REAL_DATA_DICTIONARY.md` (same structure as DATA_DICTIONARY.md: every column, source, unit, rule, caveat; label assumptions; split counts)
- `updated_data/master_dataset/VALIDATION_RESULTS.csv`
- `updated_data/scripts/13_build_master_dataset.py` (rebuilds everything from the source files)
- `updated_data/master_dataset/MASTER_DATASET_REPORT.docx` (short: inputs, decisions, row/column counts, missing values, label counts per split, limitations)

## Known limitations to state in the report
Static terrain/land-cover/lithology features take one value per district (14 values; land_cover and lithology are near-constant).
Landslide target is sparse (≈8 test positives). River stage missing for 3 districts and before mid-2015 for 7.
Flood labels in 2018 include IMD multi-week summary periods. Soil moisture is a saturation fraction.
The ML scripts read the hard-coded `synthetic_master_dataset.csv`; training on the real file will later need a path change in
`ml/flood_xgboost.py`, `ml/landslide_xgboost.py`, `ml/tcdl_v1.py`, `ml/shap_explainability.py` and `backend/config.py` — do NOT change them in this task; just report it.

At the end, summarise: rows/columns, missing values per column, flood and landslide positives per split, any validation failures, and what must happen before training.
