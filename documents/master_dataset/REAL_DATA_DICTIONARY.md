# REAL DATA — Kerala district-day Master Dataset

**`updated_data/master_dataset/real_master_dataset.csv` holds observed / published data only.** Every value comes from a named public source in `updated_data/`; nothing is synthetic, estimated, interpolated, forward-filled or imputed. Missing values are left empty (NaN) and are passed to XGBoost's native missing-value handling by the training scripts.

Built by `updated_data/scripts/13_build_master_dataset.py` on 2026-09-12 (re-run it to rebuild everything, e.g. after adding a GSI Bhukosh landslide inventory). Specification: `updated_data/documentation/MASTER_DATASET_BUILD_PROMPT.md`. Validation: `VALIDATION_RESULTS.csv` (87 checks, 0 FAIL).

> Unlike `synthetic_master_dataset.csv`, this file describes the real world — but one row is a **whole district on one day**, with static features taken at district level (14 values each). Read §10 before interpreting any model result.

---

## 1. Size

| Property | Value |
|---|---|
| Rows | **66,080** |
| Columns | **23** (the 22 synthetic columns + `district` after `longitude`) |
| Locations | 14 Kerala districts (one representative point each) |
| Time steps | 4,720 consecutive days per district |
| Date range | 2012-01-30 → 2024-12-31 (2012-01-01..01-29 used only as the 29-day warm-up of `rainfall_30d_mm`, then dropped) |
| Row ordering | `date`, then `latitude`, `longitude` ascending (14 rows per date) — the order `chronological_split()` sorts to |
| Missing values | `river_level_m` 25,738 (38.9 %); all other columns complete |

Row identity is `(date, latitude, longitude)` — equivalently `(date, district)`. Companion file `real_master_dataset_keys.csv` has the same row order (§8).

---

## 2. Columns

Observed range and missing count are for this build (all 66,080 rows).

### Time and location
| Column | Type | Source | Rule / caveat |
|---|---|---|---|
| `date` | string `YYYY-MM-DD` | CHIRPS daily series | Daily and gap-free for every district. Split key only. |
| `latitude` | float | `reference/Kerala_district_reference.csv` → `latitude` | District representative point (geoBoundaries ADM2 polygon centroid, inside the polygon); static per district. Metadata only. |
| `longitude` | float | same → `longitude` | as above. |
| `district` | string | same → `district_chirps` | CHIRPS vocabulary (`Pattanamtitta`, not `Pathanamthitta`). **Identifier only — never a model feature**: every ML script, TCDL, SHAP and the backend select features by explicit lists, and `district` is in none of them (checked). |

### Common environmental features (inputs to **both** models)
| Column | Unit | Source (under `updated_data/`) | Rule / caveat | Observed range | Missing |
|---|---|---|---|---|---|
| `rainfall_1d_mm` | mm/day | `existing_collected_data/Kerala_CHIRPS_District_Rainfall_2012.csv` + `_2013_2024.csv` → `rainfall` | CHIRPS v2.0 district-mean daily rainfall; the two files concatenated. | 0 – 254.2 | 0 |
| `rainfall_3d_mm` | mm | same | Trailing 3-day sum per district, **inclusive of day t**, `min_periods = 3`, sorted by date within district; warm-up from 2012-01-01. Reconstructs exactly from `rainfall_1d_mm`. | 0 – 432.5 | 0 |
| `rainfall_7d_mm` | mm | same | Trailing 7-day sum per district, **inclusive of day t**, `min_periods = 7`, sorted by date within district; warm-up from 2012-01-01. Reconstructs exactly from `rainfall_1d_mm`. | 0 – 707.1 | 0 |
| `rainfall_14d_mm` | mm | same | Trailing 14-day sum per district, **inclusive of day t**, `min_periods = 14`, sorted by date within district; warm-up from 2012-01-01. Reconstructs exactly from `rainfall_1d_mm`. | 0 – 963.5 | 0 |
| `rainfall_30d_mm` | mm | same | Trailing 30-day sum per district, **inclusive of day t**, `min_periods = 30`, sorted by date within district; warm-up from 2012-01-01. Reconstructs exactly from `rainfall_1d_mm`. | 0 – 1635 | 0 |
| `soil_moisture` | fraction 0–1 | `weather_soil/NASA_POWER_Kerala_District_Daily_2012_2024.csv` → `GWETROOT` | NASA POWER (MERRA-2) **root-zone soil wetness = saturation fraction** (0 = dry, 1 = saturated). **Not** volumetric water content as in the synthetic data — same column name, different quantity. POWER grid cell (~0.5° × 0.625°) at the district point. | 0.24 – 1 | 0 |
| `temperature_c` | °C | same → `T2M` | Daily mean 2-m air temperature (POWER cell). | 18.18 – 33.02 | 0 |
| `humidity_percent` | % | same → `RH2M` | Daily mean 2-m relative humidity (POWER cell). | 28.36 – 96.85 | 0 |
| `elevation_m` | m a.s.l. | `terrain/Kerala_District_Terrain_Summary.csv` → `elevation_m_mean` | District mean of the DEM (AWS Terrarium z11, ~74 m). **Static per district** (14 values). | 8.4 – 976.9 | 0 |

### Flood-specific features (Model 1 only)
| Column | Unit | Source | Rule / caveat | Observed range | Missing |
|---|---|---|---|---|---|
| `river_level_m` | m, relative | `river_level/processed/District_Gauge_Assignment.csv` (district → `primary_gauge`) + `river_level/processed/CWC_Kerala_RiverLevel_Daily_Station.csv` (`level_mean_m`) | **Stage of the district's primary CWC gauge relative to that gauge's own median over the TRAINING period only** (2012-01-30..2021-02-14): metres above (+) / below (−) the gauge's normal level. Needed because gauges sit on different datums (e.g. Vandiperiyar ~791 m MSL vs Ayilam ~1.7 m). Every primary gauge has a single `datum_segment` over 2012–2024 (a later segment would be set to NaN, never converted). Alappuzha, Wayanad, Kottayam have no primary gauge → NaN. 7 gauges start 2015-06-01 → NaN before. Days without a reading → NaN. Rounded to 0.1 mm. A point gauge measures one river reach, not the whole district. | -9.018 – 11.4 | 25,738 (38.9 %) |
| `distance_to_river_km` | km | `hydrology/Kerala_HydroRIVERS_v10.geojson` + district point | **Computed here**: shortest distance from the district representative point to the nearest HydroRIVERS v1.0 reach (any Strahler order; all 10,230 reaches of the clipped file searched exhaustively), measured in EPSG:32643 (UTM 43N). HydroRIVERS contains rivers with ≥ 10 km² catchment or ≥ 0.1 m³/s mean flow, so smaller streams are not counted. Static per district. | 0.169 – 2.001 | 0 |
| `drainage_density` | km/km² | `hydrology/Kerala_District_Drainage_Density.csv` → `drainage_density_km_per_km2` | HydroRIVERS reach length inside the district / district area. Static. | 0.2964 – 0.488 | 0 |

### Landslide-specific features (Model 2 only)
| Column | Unit | Source | Rule / caveat | Observed range | Missing |
|---|---|---|---|---|---|
| `slope_degree` | ° | `terrain/Kerala_District_Terrain_Summary.csv` → `slope_degree_mean` | District mean slope from the DEM. Static. | 1.21 – 14.59 | 0 |
| `aspect_degree` | °, 0 = N, clockwise | same → `aspect_degree_circmean` | Circular mean aspect. Static. | 2.6 – 221 | 0 |
| `curvature` | dimensionless | same → `curvature_mean` | District mean curvature (negative = concave). District means are close to 0. Static. | -0.00134 – 0.00081 | 0 |
| `land_cover` | category | `terrain/Kerala_District_LandCover_Summary.csv` → `dominant_project_land_cover` | ESA WorldCover 2021 classes mapped to the 5 project classes; dominant class per district. **`forest` in all 14 districts — constant, carries no information.** | forest (66,080) | 0 |
| `lithology` | category | `lithology/Kerala_District_Lithology_CGWB.csv` → `lithology` | CGWB Major Principal Aquifer map, dominant class per district (independently reproduced 14/14). **Only 2 values**; an aquifer (hydrogeological) map used as a lithology source. | granite_gneiss (61,360)|alluvium (4,720) | 0 |

### Targets
| Column | Values | Source | Definition |
|---|---|---|---|
| `flood` | 0 / 1 (int, no NaN) | `flood_events/processed/IFI_Kerala_2012_2023_corrected.csv` + `flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv` | **1** = the district-day lies in a damaging flood / heavy-rain event reported by IMD (IFI-Impacts v3.0 records with IMD-verified dates, extended to 2024 from IMD DWE). **0 = no damaging flood/heavy-rain event reported by IMD for that district-day** — not a proof that no flooding occurred. See §5. |
| `landslide` | 0 / 1 (int, no NaN) | `landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv` + `landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv` | **1** = at least one landslide with an exact, stated date is recorded in that district on that day (GSI inventory or IMD DWE). **0 = no exact-dated landslide record** (presence-only data). See §5. |

---

## 3. Feature groups for the two models

Unchanged from `DATA_DICTIONARY.md` and the code — the real file plugs into the same lists:

```
COMMON = ["rainfall_1d_mm","rainfall_3d_mm","rainfall_7d_mm",
          "rainfall_14d_mm","rainfall_30d_mm","soil_moisture",
          "temperature_c","humidity_percent","elevation_m"]

FLOOD_SPECIFIC     = ["river_level_m","distance_to_river_km","drainage_density"]
LANDSLIDE_SPECIFIC = ["slope_degree","aspect_degree","curvature",
                      "land_cover","lithology"]

Model 1 (flood):     X = COMMON + FLOOD_SPECIFIC          y = flood       (12 features)
Model 2 (landslide): X = COMMON + LANDSLIDE_SPECIFIC      y = landslide   (14 features)
```

`date`, `latitude`, `longitude` and `district` are indexing/splitting columns, not model inputs.

---

## 4. Categorical encoding

Stored as strings, inside the fixed `CATEGORY_SCHEMA` of `ml/landslide_xgboost.py` (codes as in `DATA_DICTIONARY.md` §4).

**`land_cover`**
| Label | Code | Rows |
|---|---|---|
| `forest` | 0 | 66,080 |
| `agriculture` | 1 | 0 |
| `shrubland` | 2 | 0 |
| `built_up` | 3 | 0 |
| `barren` | 4 | 0 |

**`lithology`**
| Label | Code | Rows |
|---|---|---|
| `granite_gneiss` | 0 | 61,360 |
| `laterite` | 1 | 0 |
| `schist` | 2 | 0 |
| `sandstone` | 3 | 0 |
| `shale` | 4 | 0 |
| `alluvium` | 5 | 4,720 |

---

## 5. Target behaviour and label assumptions

| Split | Dates | District-days | `flood` = 1 | `landslide` = 1 |
|---|---|---|---|---|
| train | 2012-01-30 → 2021-02-14 | 46,256 | 2,760 (5.97 %) | 95 (0.205 %) |
| validation | 2021-02-15 → 2023-01-23 | 9,912 | 242 (2.44 %) | 11 (0.111 %) |
| test | 2023-01-24 → 2024-12-31 | 9,912 | 123 (1.24 %) | 8 (0.081 %) |
| **all** | | **66,080** | **3,125** (4.73 %) | **114** (0.173 %) |

`flood=1 & landslide=1`: 96 district-days. φ (Pearson) = 0.156.

**Flood label (decision 2).**

* 1 on every (district, day) listed in an event's `event_days`; where `event_days` is empty (UEI-IMD-FL-2014-0050), every day `verified_start`..`verified_end`. The `verified_*` dates are used — never IFI's original Start/End Date columns.
* IMD events from `IMD_DWE_Kerala_events_not_in_IFI.csv` (`event_days` × `districts`): the 24 events after IFI's last Kerala record and the OMITTED_FROM_IFI event (IMD:DWE_2023:p133:KERALA1 (2023-06-28; Ernakulam;Kannur;Kasaragod;Kozhikode)) — included (`INCLUDE_OMITTED_FROM_IFI` in the build script).
* **Kept:** LONG_PERIOD_SUMMARY records (UEI-IMD-FL-2013-0071, UEI-IMD-FL-2015-0063, UEI-IMD-FL-2018-0035, UEI-IMD-FL-2018-0036) — the only coverage of the 2018 flood. 1,416 positive district-days rest only on such multi-week summary periods (train 1,416 / val 0 / test 0); they are flagged `flood_long_period_only = 1` in the keys file for sensitivity runs.
* **Excluded:** 12 LANDSLIDE_ONLY_CAUSE records (UEI-IMD-FL-2017-0067, UEI-IMD-FL-2019-0059, UEI-IMD-FL-2019-0061, UEI-IMD-FL-2020-0053, UEI-IMD-FL-2021-0076, UEI-IMD-FL-2021-0117, UEI-IMD-FL-2022-0544, UEI-IMD-FL-2022-0547, UEI-IMD-FL-2022-0551, UEI-IMD-FL-2022-0563, UEI-IMD-FL-2022-0564, UEI-IMD-FL-2022-0576) and 2 NO_DISTRICT records (UEI-IMD-FL-2022-0567, UEI-IMD-FL-2023-0353).
* Before exclusions: 3,136 positive district-days (train 2,764 / val 249 / test 123) — reproduces the documented row of `Flood_label_coverage_summary.csv` (“IFI corrected + all IMD DWE Kerala entries missing from IFI”). The build prompt's 3,132 (2,764 / 249 / 119) is the extension-only row; the 4 test district-days in between are the omitted 2023-06-28 event. After exclusions: **3,125**.
* Caveat: the extension file has no cause flag. 4 extension entries mention a landslide but not a flood and are labelled flood = 1 as decided: IMD:DWE_2023:p134:KERALA6 (2023-11-05; Ernakulam;Idukki;Thrissur); IMD:DWE_2024:p152:KERALA12 (2024-07-14; Idukki); IMD:DWE_2024:p153:KERALA16 (2024-07-30; Wayanad); IMD:DWE_2024:p153:KERALA17 (2024-07-30; Kozhikode).

**Landslide label (decision 3).** Exact-date records only; presence-only.

* GSI: rows of the all-records date-precision file with non-empty `exact_dates`, district from `district_project`, filtered to the grid window. Excluded precision classes: YEAR_ONLY 621, NO_DATE 508, MONTH_YEAR 13 records.
* IMD: rows with `date_precision == EXACT_DATE` and non-empty `landslide_districts` (26 of 59). Excluded: PERIOD 17, WEEK_ONLY 1 entries and 15 exact-day multi-district entries that do not say where the landslide was.
* Union, de-duplicated at district-day level.
* **Stated-date guard:** 17 GSI record-dates in `exact_dates` are not stated in the record's own `History` text (the upstream parser split a 4-digit year into two day numbers). They are not used — a date may never be assigned to a record that does not state one. In the grid window: GSI Sl_No 15395, Pattanamtitta, 2018-08-18 (History: “July 2018, 15th August 2018”); GSI Sl_No 15395, Pattanamtitta, 2018-08-20 (History: “July 2018, 15th August 2018”); GSI Sl_No 15489, Wayanad, 2018-06-14 (History: “2014, June 2018,8 August 2018”); GSI Sl_No 15489, Wayanad, 2018-06-20 (History: “2014, June 2018,8 August 2018”); GSI Sl_No 15745, Idukki, 2018-08-16 (History: “2016, 15th August 2018”); GSI Sl_No 15745, Idukki, 2018-08-20 (History: “2016, 15th August 2018”); GSI Sl_No 15760, Idukki, 2018-08-13 (History: “2013 & 15th August 2018”); GSI Sl_No 15760, Idukki, 2018-08-20 (History: “2013 & 15th August 2018”); GSI Sl_No 15788, Idukki, 2018-08-16 (History: “2016 & 15th August 2018”); GSI Sl_No 15788, Idukki, 2018-08-20 (History: “2016 & 15th August 2018”); GSI Sl_No 15807, Idukki, 2018-08-12 (History: “2012 & 14th August 2018”); GSI Sl_No 15807, Idukki, 2018-08-20 (History: “2012 & 14th August 2018”); GSI Sl_No 15814, Idukki, 2018-08-19 (History: “1983 & 15th August 2018”); GSI Sl_No 15847, Idukki, 2018-08-20 (History: “2015 & 15th August 2018”); GSI Sl_No 16135, Palakkad, 2019-08-18 (History: “2018 and 09 August 2019”); GSI Sl_No 16135, Palakkad, 2019-08-20 (History: “2018 and 09 August 2019”).
* Result: **114** district-days (train 95 / val 11 / test 8); documented expectation ≈ 120 (101 / 11 / 8). Without the stated-date guard the build gives exactly those 120, so the guard accounts for the whole difference (see `VALIDATION_RESULTS.csv`).
* Presence-only assumption: a 0 means no exact-dated landslide is on record for that district-day. GSI's inventory is dominated by the 2018–2019 monsoons, and many real landslides have only a month or year, so true positives are certainly missed. No random negative sampling is done.

---

## 6. Temporal structure and split

`chronological_split()` (70/15/15 on unique dates, from `ml/flood_xgboost.py`) on 4,720 dates gives train 2012-01-30 → 2021-02-14, validation → 2023-01-23, test → 2024-12-31 — recomputed here and confirmed by running the scripts' own function on this file.

Rows are not i.i.d.: consecutive days of a district share rainfall windows and a slowly varying soil-moisture and river state, and neighbouring districts share storms. Always split chronologically.

---

## 7. Data leakage

* No column encodes a label; no feature file contains hazard information. No feature equals a target.
* Rolling sums are trailing windows ending on day t (verified on every row); nothing is centred or forward.
* The only statistic estimated from the data — each gauge's median — uses TRAINING dates only (last date used: 2021-02-14; train ends 2021-02-14); recomputed and matched in validation.
* `district`, `latitude`, `longitude` are identifiers, not features.
* Same-day features: rainfall, T2M, RH2M and soil wetness of day t are daily aggregates of day t, as in the synthetic design; a strict nowcast/lead-time experiment must lag them.

---

## 8. Companion keys file (`real_master_dataset_keys.csv`)

Same row order as the master file. Not a model input.

| Column | Meaning |
|---|---|
| `date, district` | row identity, identical order to the master file |
| `split` | train / validation / test from `chronological_split()` |
| `river_gauge` | primary CWC gauge of the district (empty = none) |
| `river_gauge_river` | river of that gauge |
| `river_gauge_status` | assignment status from `District_Gauge_Assignment.csv` |
| `river_stage_published_m` | raw published daily mean stage of the gauge (`level_mean_m`, gauge datum), for transparency |
| `river_datum_segment` | datum segment of that reading (all primary gauges: single segment 0) |
| `river_gauge_train_median_m` | the gauge's training-period median that is subtracted |
| `river_level_flag` | OBSERVED / NO_READING_THAT_DAY / NO_PRIMARY_GAUGE / AFTER_DATUM_BREAK |
| `flood_source_ids` | included flood records covering the day (IFI UEI or IMD:<report>:p<page>:<entry>) |
| `flood_n_sources` | number of included flood records covering the day |
| `flood_long_period_only` | 1 if the flood label rests only on LONG_PERIOD_SUMMARY records |
| `flood_excluded_source_ids` | excluded records (LANDSLIDE_ONLY_CAUSE) that cover the day — not used |
| `landslide_source` | GSI, IMD or GSI+IMD |
| `landslide_gsi_n_records` | number of GSI records dated to that district-day |
| `landslide_source_ids` | GSI:<Sl_No> / IMD:<report>:p<page>:s<sno> |
| `landslide_flags` | GSI_NUMERIC_DATE_DM_AMBIGUOUS if a contributing GSI date was numeric and read day-first |

---

## 9. Static values per district

| District | Lat | Lon | Elev. m | Slope ° | Aspect ° | Curv. | Drain. dens. | Dist. river km (HYRIV_ID, Strahler) | Land cover | Lithology | Primary gauge (river) | Gauge train median m (n obs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kasaragod | 12.45993 | 75.15314 | 113.5 | 7.38 | 210.3 | -0.00027 | 0.488 | 1.563 (41359360, 1) | forest | granite_gneiss | ERINJIPUZHA (Payaswani) | 14.624 (1,431) |
| Kannur | 12.001 | 75.52542 | 134.7 | 7.86 | 203.6 | 0.00029 | 0.4592 | 1.251 (41366701, 1) | forest | granite_gneiss | PERUMANNU (Valapatnam) | 7.484 (1,665) |
| Wayanad | 11.7109 | 76.08737 | 845.5 | 8.66 | 2.6 | 0.00081 | 0.3864 | 1.259 (41371674, 4) | forest | granite_gneiss | — (none) | — |
| Kozhikode | 11.48426 | 75.83435 | 174.0 | 8.64 | 218.6 | -0.00077 | 0.4677 | 0.169 (41374434, 1) | forest | granite_gneiss | KUTTYADI (Kuttyadi) | 1.319 (1,665) |
| Malappuram | 11.13281 | 76.15629 | 196.8 | 8.3 | 215.1 | 0.00029 | 0.4192 | 0.288 (41379116, 1) | forest | granite_gneiss | KARATHODU (Kadalundi) | 3.477 (1,659) |
| Palakkad | 10.79598 | 76.55899 | 369.4 | 8.94 | 221.0 | 6e-05 | 0.3865 | 0.214 (41382811, 5) | forest | granite_gneiss | MANKARA (Bharathapuzha) | 47.320 (1,947) |
| Thrissur | 10.47192 | 76.31575 | 137.2 | 6.21 | 212.4 | -0.00026 | 0.421 | 1.493 (41386363, 1) | forest | granite_gneiss | ARANGALI (Periyar) | 0.733 (1,917) |
| Ernakulam | 10.08408 | 76.5463 | 152.0 | 6.73 | 214.2 | 0.0001 | 0.4117 | 1.116 (41389353, 1) | forest | granite_gneiss | NEELEESWARAM (Periyar) | 0.673 (1,955) |
| Idukki | 9.83875 | 77.0588 | 976.9 | 14.59 | 193.6 | -0.00134 | 0.313 | 1.11 (41391230, 1) | forest | granite_gneiss | VANDIPERIYAR (Periyar) | 791.343 (3,249) |
| Kottayam | 9.63819 | 76.65185 | 95.2 | 6.57 | 215.1 | -4e-05 | 0.4315 | 0.432 (41392470, 2) | forest | granite_gneiss | — (none) | — |
| Alappuzha | 9.42945 | 76.44763 | 8.4 | 1.21 | 143.0 | -0.00018 | 0.3506 | 0.443 (41393829, 1) | forest | alluvium | — (none) | — |
| Pattanamtitta | 9.28512 | 76.92861 | 376.4 | 11.91 | 202.6 | -0.00071 | 0.3596 | 0.832 (41395004, 1) | forest | granite_gneiss | THUMPAMON (Pamba) | 6.720 (3,205) |
| Kollam | 8.96209 | 76.87268 | 160.6 | 7.68 | 189.4 | -0.00032 | 0.3442 | 2.001 (41396869, 1) | forest | granite_gneiss | PATTAZHY (Kallada) | 2.249 (3,238) |
| Thiruvananthapuram | 8.60859 | 77.01251 | 148.1 | 6.99 | 209.7 | -0.00013 | 0.2964 | 0.202 (41398937, 1) | forest | granite_gneiss | AYILAM (Vamanapuram) | 1.430 (3,215) |

---

## 10. Known limitations

1. Spatial unit = district (14 locations). Static terrain, drainage, distance-to-river, land-cover and lithology features take one value per district, so a tree model can memorise district identity through them; `land_cover` is constant (forest ×14) and `lithology` has 2 values (granite_gneiss ×13, alluvium ×1). Terrain feature importances from this file are not evidence about processes.
1. Landslide target is sparse (≈ 8 test positives) and presence-only; test metrics will have very wide uncertainty.
1. River stage is missing for Alappuzha, Wayanad and Kottayam (no usable CWC gauge) and before 2015-06-01 for 7 districts; one point gauge represents a whole district.
1. Flood labels in 2013, 2015 and 2018 include IMD multi-week summary periods (LONG_PERIOD_SUMMARY), which mark whole periods rather than flood days — including the 2018 flood season in all 14 districts.
1. Flood 0 = not reported by IMD, not proven absence; IMD records damaging events.
1. `soil_moisture` is a 0–1 saturation fraction (GWETROOT), not volumetric water content; POWER cells are ~50 km.
1. The ML scripts read the hard-coded `synthetic_master_dataset.csv`. Training on this file later needs a path change in `ml/flood_xgboost.py`, `ml/landslide_xgboost.py`, `ml/tcdl_v1.py`, `ml/shap_explainability.py` and `backend/config.py` (not changed here).

---

## 11. Regenerating

```bash
python updated_data/scripts/13_build_master_dataset.py
```

Deterministic (no randomness). Adding new landslide records: process them into `landslide_events/processed/` with the same columns (`district_project`, `exact_dates`, `History`), re-run, and compare `VALIDATION_RESULTS.csv`.
