# SYNTHETIC / DUMMY DATA — NOT FOR RESEARCH RESULTS

**`synthetic_master_dataset.csv` contains simulated values only.**
It exists so the coupled flood–landslide ML pipeline can be built and tested before
the real environmental datasets are collected and integrated.

> **This dataset is synthetic and must NOT be used to report research accuracy,
> scientific conclusions, or actual disaster lead-time results. It is only for
> developing and testing the ML pipeline.**

Coordinates, terrain, rainfall, river stages and both hazard labels are invented.
Any accuracy, AUC, feature-importance or lead-time number obtained from this file
describes the simulator, not the physical world.

---

## 1. Size

| Property | Value |
|---|---|
| Rows | **9,864** |
| Columns | **22** |
| Locations | 9 synthetic grid cells |
| Time steps | 1,096 consecutive days per location |
| Date range | 2018-01-01 → 2020-12-31 (three full calendar years) |
| Row ordering | `date` ascending (9 location rows per date) |
| Missing values | none |

Row identity is the `(date, latitude, longitude)` triple. There is no location-ID
column, since no columns beyond the agreed schema were added.

---

## 2. Columns

### Time and location
| Column | Type | Description |
|---|---|---|
| `date` | string `YYYY-MM-DD` | Observation date. Chronological; each date has 9 location rows. |
| `latitude` | float | Decimal degrees, 9.283 – 11.842. Fictitious cell centroid. |
| `longitude` | float | Decimal degrees, 76.104 – 77.463. Fictitious cell centroid. |

### Common environmental features (inputs to **both** models)
| Column | Unit / range | Description |
|---|---|---|
| `rainfall_1d_mm` | mm, 0 – 190.5 | Rainfall on the observation day. |
| `rainfall_3d_mm` | mm, 0 – 380.8 | Trailing 3-day rainfall total (inclusive of the current day). |
| `rainfall_7d_mm` | mm, 0 – 679.4 | Trailing 7-day rainfall total. |
| `rainfall_14d_mm` | mm, 0 – 937.5 | Trailing 14-day rainfall total. |
| `rainfall_30d_mm` | mm, 0 – 1305.5 | Trailing 30-day rainfall total (antecedent wetness). |
| `soil_moisture` | volumetric fraction, 0.03 – 0.60 | Modelled root-zone soil water content. |
| `temperature_c` | °C, 15.1 – 36.9 | Daily mean air temperature. |
| `humidity_percent` | %, 22 – 100 | Daily mean relative humidity. |
| `elevation_m` | m, 19 – 1268 | Cell elevation above sea level (static per location). |

The five rainfall columns are **true rolling sums of the same daily series**, so
`rainfall_3d_mm` equals the sum of the last three `rainfall_1d_mm` values, and so on.
A 30-day burn-in was simulated before 2018-01-01, so the accumulation columns are
already correct on the first exported row (they are not reconstructible from this
file alone for the first 29 rows of each location).

### Flood-specific features (Model 1 only)
| Column | Unit / range | Description |
|---|---|---|
| `river_level_m` | m, 0.59 – 5.47 | River stage at the nearest gauge; routed catchment response with recession. |
| `distance_to_river_km` | km, 0.16 – 6.35 | Distance to nearest channel (static per location). |
| `drainage_density` | km/km², 0.92 – 3.38 | Channel length per unit area (static per location). |

### Landslide-specific features (Model 2 only)
| Column | Unit / range | Description |
|---|---|---|
| `slope_degree` | °, 1.1 – 37.9 | Terrain slope (static per location). |
| `aspect_degree` | °, 0 – 360 | Slope azimuth, 0 = north, clockwise (static per location). |
| `curvature` | dimensionless, −0.77 – +0.34 | Profile curvature. **Negative = concave** (convergent hollow), **positive = convex** (ridge). |
| `land_cover` | categorical string | Dominant land cover (static per location). |
| `lithology` | categorical string | Dominant surface geology (static per location). |

### Targets
| Column | Values | Description |
|---|---|---|
| `flood` | 0 / 1 | 1 = flood hazard occurred at that cell on that date. |
| `landslide` | 0 / 1 | 1 = landslide hazard occurred at that cell on that date. |

---

## 3. Feature groups for the two models

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

`date`, `latitude` and `longitude` are indexing/splitting columns, not model inputs.

---

## 4. Categorical encoding

Both categorical columns are stored as **human-readable strings**. No pre-encoded
duplicate columns were added, to keep the schema exactly as specified. Use either
`astype("category")` with `enable_categorical=True`, or the integer mapping below.

**`land_cover`** — 5 classes
| Label | Code | Rows |
|---|---|---|
| `forest` | 0 | 1,096 |
| `agriculture` | 1 | 3,288 |
| `shrubland` | 2 | 1,096 |
| `built_up` | 3 | 2,192 |
| `barren` | 4 | 1,096 |

**`lithology`** — 6 classes
| Label | Code | Rows |
|---|---|---|
| `granite_gneiss` | 0 | 1,096 |
| `laterite` | 1 | 2,192 |
| `schist` | 2 | 1,096 |
| `sandstone` | 3 | 1,096 |
| `shale` | 4 | 1,096 |
| `alluvium` | 5 | 3,288 |

```python
LAND_COVER_CODES = {"forest":0,"agriculture":1,"shrubland":2,"built_up":3,"barren":4}
LITHOLOGY_CODES  = {"granite_gneiss":0,"laterite":1,"schist":2,
                    "sandstone":3,"shale":4,"alluvium":5}
```

The codes are **nominal identifiers, not an ordering** — do not treat them as ordinal.
In the simulator, landslide susceptibility increases roughly
`alluvium < granite_gneiss < sandstone < laterite < schist < shale` and
`forest < shrubland < built_up < agriculture < barren`, but that ordering is *not*
encoded in the integers.

---

## 5. Target behaviour

| Combination | Rows | Share |
|---|---|---|
| `flood=0, landslide=0` | 8,470 | 85.87 % |
| `flood=1, landslide=0` | 670 | 6.79 % |
| `flood=0, landslide=1` | 607 | 6.15 % |
| `flood=1, landslide=1` | 117 | 1.19 % |

Overall flood rate 7.98 %, landslide rate 7.34 %, φ (Pearson) = 0.085.

The two targets are **not** the same variable. The marginal correlation is low
because flood-prone cells and landslide-prone cells are largely *different* cells,
but within any single cell the coupling is strong — at the valley-side locations
`P(landslide=1 | flood=1)` runs roughly 8–12× the cell's baseline landslide rate,
because both hazards are driven by the same storm.

Neither target is a threshold rule. Both are Bernoulli draws from a logistic
function of the drivers, including interaction terms
(rainfall × soil moisture, river stage × river proximity, slope × soil moisture,
rainfall × lithology) plus an unobserved region-day shock and logistic noise.

---

## 6. Temporal structure

Rainfall is generated as a persistent daily process, not independent draws:
a season-dependent two-state (wet/dry) Markov chain, gamma wet-day depths,
injected multi-day storm episodes, and occasional cloudburst days.

- Lag-1 rainfall autocorrelation ≈ 0.64, decaying to ≈ 0.15 by lag 5.
- Daily regime mix: 59.2 % dry, 25.0 % light (<10 mm), 11.1 % moderate (10–35 mm),
  3.7 % heavy (35–75 mm), 1.0 % extreme (>75 mm).
- Clear bimodal monsoon seasonality; hazard rates track it (July: mean 15.2 mm/day,
  22.2 % flood, 19.7 % landslide — versus March: 0.8 mm/day, 0.8 % / 1.7 %).

Soil moisture is a bucket store with rainfall input, evapotranspiration loss and
drainage; river level is a routed catchment response with exponential recession.
Both therefore lag rainfall, which is what makes lead-time experiments meaningful.

---

## 7. Data leakage

No column encodes the answer. There is no `days_until_flood`, `flood_risk`,
`flood_severity`, `landslide_risk`, `days_until_landslide`, probability, warning,
risk-score, coupled-probability or TCDL column. Every feature is an environmental
or terrain quantity that would be observable before the event.

Note the one structural caveat: **rows are not i.i.d.** Consecutive days at the same
location share overlapping rainfall windows and a slowly-varying soil-moisture
state. A random train/test split therefore leaks across time — always split
chronologically, as the pipeline specifies.

---

## 8. Reference pipeline check

`validate_synthetic_dataset.py` runs the schema/constraint checks and a real
chronological-split XGBoost fit. Current result (70/15/15 chronological split):

| Model | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| Flood | 0.87 | 0.65 |
| Landslide | 0.90 | 0.63 |

Learnable but not trivial — which is the point. **These numbers are properties of
the simulator and must never be reported as project results.**

---

## 9. Regenerating

```bash
python generate_synthetic_master_dataset.py
```

Seeded (`SEED = 20260814`), so output is reproducible. Change `SEED` for a
different draw, or `N_DAYS` / `SITES` for a different size or geography.

## 10. Known limitations

1. Terrain columns (`elevation_m`, `slope_degree`, `aspect_degree`, `curvature`,
   `distance_to_river_km`, `drainage_density`, `land_cover`, `lithology`) are
   **static per location**, so only 9 distinct values exist for each. A tree model
   can partly memorise location identity through them. Fine for pipeline testing;
   the real dataset will need many more cells for trustworthy terrain
   feature-importance.
2. Hazard labels are frequent by real-world standards (the most exposed riverside
   cell floods on ~20 % of days). Read them as "hazard-present / warning-relevant
   days", not as major disaster events.
3. The 9 cells are grouped into 3 regional weather regimes, so rainfall is
   spatially correlated in blocks of 3 rather than through a realistic field.
