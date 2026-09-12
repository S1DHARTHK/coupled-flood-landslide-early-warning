"""
====================================================================
SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS
====================================================================
Generator for `synthetic_master_dataset.csv`.

Purpose: provide a schema-correct, internally-consistent placeholder
dataset so the coupled flood-landslide ML pipeline (data loading ->
feature selection -> chronological split -> XGBoost -> hazard
probability -> evaluation) can be developed and tested BEFORE the real
environmental datasets are collected and integrated.

Every value in the output is simulated. Coordinates, terrain, rainfall,
river stages and hazard labels are invented. Nothing here may be used to
report model accuracy, scientific findings, or disaster lead-time
results.

Design summary
--------------
* 9 synthetic locations (grid cells) x 1096 daily time steps
  (2018-01-01 .. 2020-12-31, three full calendar years) = 9864 rows.
* Locations are grouped into 3 regional weather regimes, so hazards can
  cluster in space as well as time.
* Rainfall is generated as a daily time series per location using a
  two-state (wet/dry) Markov chain with monsoon seasonality plus
  injected multi-day storm episodes. The 1/3/7/14/30-day columns are
  true rolling sums of that series (with a 30-day burn-in before the
  first exported date), so accumulation columns are mutually consistent.
* Soil moisture is a bucket store driven by rainfall, evapotranspiration
  and local drainage. River level is a routed/recessing catchment
  response to regional rainfall.
* Targets are Bernoulli draws from logistic functions of the drivers,
  including interaction terms and an unobserved latent shock, so neither
  target is a deterministic threshold on any single feature.
"""

import numpy as np
import pandas as pd

SEED = 20260814
rng = np.random.default_rng(SEED)

START_DATE = "2018-01-01"
N_DAYS = 1096           # three full calendar years
BURN_IN = 30            # extra days simulated before START_DATE for rolling sums

# Target prevalences (calibrated by bisection on the latent intercepts)
FLOOD_RATE = 0.080
LANDSLIDE_RATE = 0.075

LAND_COVER_CLASSES = ["forest", "agriculture", "shrubland", "built_up", "barren"]
LITHOLOGY_CLASSES = ["granite_gneiss", "laterite", "schist",
                     "sandstone", "shale", "alluvium"]

# --------------------------------------------------------------------
# 1. Static site descriptors (one row per synthetic location)
# --------------------------------------------------------------------
# region: shared regional weather regime (0,1,2)
# Sites deliberately span the terrain spectrum: low flat valley cells
# near rivers (flood-prone, landslide-safe), steep upland cells far from
# rivers (landslide-prone, flood-safe), and mid-slope valley-side cells
# where both hazards are possible.
SITES = [
    # name-ish comment           region lat     lon     elev  slope aspect curv  d_river dd    land_cover     lithology
    dict(region=0, latitude=11.842, longitude=76.104, elevation_m=  38.0, slope_degree= 1.6, aspect_degree= 172.0, curvature=-0.42, distance_to_river_km=0.22, drainage_density=3.05, land_cover="agriculture", lithology="alluvium"),
    dict(region=0, latitude=11.615, longitude=76.371, elevation_m= 287.0, slope_degree=12.5, aspect_degree= 238.0, curvature=-0.31, distance_to_river_km=0.78, drainage_density=2.44, land_cover="built_up",    lithology="laterite"),
    dict(region=0, latitude=11.394, longitude=76.688, elevation_m= 812.0, slope_degree=28.4, aspect_degree= 221.0, curvature=-0.61, distance_to_river_km=4.10, drainage_density=1.32, land_cover="forest",      lithology="schist"),
    dict(region=1, latitude=10.978, longitude=76.955, elevation_m=1268.0, slope_degree=37.9, aspect_degree= 249.0, curvature=-0.77, distance_to_river_km=6.35, drainage_density=0.92, land_cover="barren",      lithology="shale"),
    dict(region=1, latitude=10.712, longitude=77.208, elevation_m= 604.0, slope_degree=19.7, aspect_degree= 118.0, curvature= 0.34, distance_to_river_km=1.65, drainage_density=1.88, land_cover="agriculture", lithology="laterite"),
    dict(region=1, latitude=10.455, longitude=76.842, elevation_m=  76.0, slope_degree= 3.1, aspect_degree=  64.0, curvature= 0.12, distance_to_river_km=0.41, drainage_density=2.77, land_cover="built_up",    lithology="alluvium"),
    dict(region=2, latitude= 9.986, longitude=77.463, elevation_m= 955.0, slope_degree=32.6, aspect_degree= 292.0, curvature=-0.55, distance_to_river_km=2.95, drainage_density=1.51, land_cover="shrubland",   lithology="granite_gneiss"),
    dict(region=2, latitude= 9.641, longitude=77.119, elevation_m= 421.0, slope_degree=14.2, aspect_degree= 205.0, curvature=-0.28, distance_to_river_km=1.10, drainage_density=2.16, land_cover="agriculture", lithology="sandstone"),
    dict(region=2, latitude= 9.283, longitude=76.774, elevation_m=  19.0, slope_degree= 1.1, aspect_degree=  31.0, curvature= 0.05, distance_to_river_km=0.16, drainage_density=3.38, land_cover="agriculture", lithology="alluvium"),
]
N_SITES = len(SITES)
N_REGIONS = 3

TOTAL_DAYS = N_DAYS + BURN_IN
dates_full = pd.date_range(end=pd.Timestamp(START_DATE) + pd.Timedelta(days=N_DAYS - 1),
                           periods=TOTAL_DAYS, freq="D")
doy = dates_full.dayofyear.to_numpy()


# --------------------------------------------------------------------
# 2. Regional rainfall: seasonal Markov wet/dry chain + storm episodes
# --------------------------------------------------------------------
def seasonal_shape(doy_arr):
    """Bimodal monsoon climatology: SW monsoon (Jun-Sep) + NE monsoon (Oct-Nov)."""
    sw = np.exp(-0.5 * ((doy_arr - 196) / 38.0) ** 2)      # south-west peak ~mid-July
    ne = 0.55 * np.exp(-0.5 * ((doy_arr - 305) / 24.0) ** 2)  # north-east peak ~Nov
    return 0.08 + sw + ne                                   # dry-season floor


def regional_rainfall(n, doy_arr, wetness):
    """One regional daily rainfall series (mm/day)."""
    season = seasonal_shape(doy_arr)
    season_n = season / season.max()

    # Wet/dry persistence: wet spells are longer in the monsoon.
    p_wet_given_dry = np.clip(0.05 + 0.55 * season_n, 0.02, 0.72)
    p_wet_given_wet = np.clip(0.30 + 0.55 * season_n, 0.20, 0.90)

    wet = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        p = p_wet_given_wet[t] if state else p_wet_given_dry[t]
        state = rng.random() < p
        wet[t] = state

    # Wet-day depth: heavy-tailed gamma whose scale follows the season.
    scale = wetness * (1.5 + 11.0 * season_n)
    amounts = rng.gamma(shape=0.72, scale=scale, size=n)
    rain = np.where(wet, amounts, 0.0)

    # Injected multi-day storm episodes -> "prolonged rainfall" regime.
    n_years = n / 365.25
    n_episodes = int(round(rng.uniform(7, 11) * n_years))
    for _ in range(n_episodes):
        start = int(rng.integers(0, n - 12))
        # bias episode onset toward the monsoon
        for _try in range(6):
            if rng.random() < season_n[start] ** 0.6:
                break
            start = int(rng.integers(0, n - 12))
        dur = int(rng.integers(2, 10))
        peak = rng.uniform(16, 72) * wetness * (0.45 + 0.85 * season_n[start])
        profile = np.sin(np.linspace(0.35, np.pi - 0.35, dur))
        rain[start:start + dur] += peak * profile * rng.uniform(0.6, 1.4, size=dur)

    # A handful of extreme cloudburst days per record.
    n_extreme = int(round(1.5 * n_years))
    for _ in range(n_extreme):
        t = int(rng.integers(0, n))
        rain[t] += rng.uniform(55, 165) * season_n[t] ** 0.5

    return np.clip(rain, 0.0, 330.0)


region_rain = np.stack([
    regional_rainfall(TOTAL_DAYS, doy, wetness=w)
    for w in (0.78, 0.66, 0.88)
])


# --------------------------------------------------------------------
# 3. Per-site meteorology, soil moisture and river level
# --------------------------------------------------------------------
def rolling_sum(x, w):
    """Trailing w-day sum, inclusive of the current day."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(len(x), np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


LAND_COVER_RUNOFF = {"forest": -0.25, "shrubland": -0.05, "agriculture": 0.05,
                     "barren": 0.20, "built_up": 0.40}
LITHO_SUSCEPT = {"granite_gneiss": -0.20, "laterite": 0.35, "schist": 0.45,
                 "sandstone": 0.15, "shale": 0.60, "alluvium": -0.40}
LITHO_STORAGE = {"granite_gneiss": 0.80, "laterite": 1.10, "schist": 0.95,
                 "sandstone": 1.05, "shale": 0.85, "alluvium": 1.25}

frames = []

for site_id, site in enumerate(SITES):
    reg = site["region"]
    elev = site["elevation_m"]

    # --- rainfall: regional signal + orographic gain + local convection ---
    oro = 1.0 + 0.28 * (elev / 1200.0)
    local = region_rain[reg] * oro * rng.uniform(0.90, 1.10, size=TOTAL_DAYS)
    convective = (rng.random(TOTAL_DAYS) < 0.05 * seasonal_shape(doy) / seasonal_shape(doy).max()) \
        * rng.gamma(0.9, 9.0, size=TOTAL_DAYS)
    rain = np.clip(local + convective, 0.0, 460.0)
    rain = np.round(rain, 1)

    r1 = rain.copy()
    r3 = rolling_sum(rain, 3)
    r7 = rolling_sum(rain, 7)
    r14 = rolling_sum(rain, 14)
    r30 = rolling_sum(rain, 30)

    # --- temperature: seasonal cycle, lapse rate, rain-day cooling ---
    t_base = 30.5 - 5.8 * (elev / 1000.0)
    t_seasonal = 3.6 * np.cos(2 * np.pi * (doy - 128) / 365.25)
    temp = (t_base + t_seasonal
            - 3.4 * np.tanh(rain / 28.0)
            + rng.normal(0, 1.15, size=TOTAL_DAYS))
    temp = np.clip(temp, 2.0, 44.0)

    # --- soil moisture bucket ---
    cap = 0.52 * LITHO_STORAGE[site["lithology"]]
    drain_k = 0.055 + 0.030 * site["drainage_density"] + 0.0022 * site["slope_degree"]
    sm = np.empty(TOTAL_DAYS)
    s = 0.20
    for t in range(TOTAL_DAYS):
        infil = (rain[t] / 165.0) * (1.0 - 0.45 * LAND_COVER_RUNOFF[site["land_cover"]])
        et = 0.0016 * max(temp[t] - 8.0, 0.0) * (s / cap)
        s = s + infil - drain_k * max(s - 0.09, 0.0) - et
        s = float(np.clip(s, 0.045, cap))
        sm[t] = s
    soil_moisture = np.clip(sm + rng.normal(0, 0.006, size=TOTAL_DAYS), 0.03, 0.60)

    # --- humidity ---
    sm_n = (soil_moisture - 0.03) / (0.60 - 0.03)
    humidity = (52.0 + 34.0 * sm_n + 0.16 * np.minimum(rain, 90.0)
                - 0.55 * (temp - 26.0) + rng.normal(0, 3.2, size=TOTAL_DAYS))
    humidity = np.clip(humidity, 22.0, 100.0)

    # --- river level: routed catchment response with recession ---
    catch_rain = region_rain[reg]
    stage_base = 0.55 + 1.35 * np.exp(-elev / 520.0)      # low cells sit on bigger channels
    lvl = np.empty(TOTAL_DAYS)
    L = stage_base
    for t in range(TOTAL_DAYS):
        inflow = 0.0062 * catch_rain[t] + (0.0011 * catch_rain[t - 1] if t else 0.0)
        L = stage_base + 0.885 * (L - stage_base) + inflow * (0.6 + 0.5 * site["drainage_density"] / 3.4)
        lvl[t] = L
    river_level = np.clip(lvl + rng.normal(0, 0.035, size=TOTAL_DAYS), 0.02, None)

    df = pd.DataFrame({
        "date": dates_full,
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "rainfall_1d_mm": r1,
        "rainfall_3d_mm": r3,
        "rainfall_7d_mm": r7,
        "rainfall_14d_mm": r14,
        "rainfall_30d_mm": r30,
        "soil_moisture": soil_moisture,
        "temperature_c": temp,
        "humidity_percent": humidity,
        "elevation_m": elev,
        "river_level_m": river_level,
        "distance_to_river_km": site["distance_to_river_km"],
        "drainage_density": site["drainage_density"],
        "slope_degree": site["slope_degree"],
        "aspect_degree": site["aspect_degree"],
        "curvature": site["curvature"],
        "land_cover": site["land_cover"],
        "lithology": site["lithology"],
    })
    df["_site"] = site_id
    df["_stage_base"] = stage_base
    frames.append(df.iloc[BURN_IN:].reset_index(drop=True))   # drop burn-in

data = pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------
# 4. Hazard generation (probabilistic, with interactions and noise)
# --------------------------------------------------------------------
def z(series):
    return (series - series.mean()) / series.std()


n = len(data)

# Unobserved "regional shock" per (region, day): the part of the physics
# we pretend not to have measured. One small shared component (both
# hazards respond to the same unmeasured storm intensity) plus two large
# hazard-specific components (channel obstruction / bank state for
# floods; local weathering, root cohesion, prior disturbance for slides).
# This keeps the hazards coupled without making them near-duplicates.
region_of = np.array([SITES[s]["region"] for s in data["_site"]])
day_index = (data["date"] - data["date"].min()).dt.days.to_numpy()
shock_shared = rng.normal(0, 1.0, size=(N_REGIONS, N_DAYS))[region_of, day_index]
shock_f = rng.normal(0, 1.0, size=(N_REGIONS, N_DAYS))[region_of, day_index]
shock_l = rng.normal(0, 1.0, size=(N_REGIONS, N_DAYS))[region_of, day_index]

sm_n = z(data["soil_moisture"])
river_anom = data["river_level_m"] - data["_stage_base"]          # stage above local baseline
prox = np.exp(-data["distance_to_river_km"] / 1.4)                # 1 at the bank -> 0 far away
dd_n = z(data["drainage_density"])
elev_n = z(np.log1p(data["elevation_m"]))

r1n, r3n, r7n = z(data["rainfall_1d_mm"]), z(data["rainfall_3d_mm"]), z(data["rainfall_7d_mm"])
r14n, r30n = z(data["rainfall_14d_mm"]), z(data["rainfall_30d_mm"])

# ---- Flood latent score: common features + flood-specific only -------
# Exposure gate: a cell can only flood if it is close to a channel and
# low in the catchment. A ridge cell 6 km from the nearest river does not
# inundate no matter how hard it rains, so the meteorological trigger is
# multiplied by the gate rather than added to it.
flood_gate = np.exp(-data["distance_to_river_km"] / 3.5) * \
             np.exp(-data["elevation_m"] / 1800.0)

flood_trigger = (0.72 * r1n
                 + 0.92 * r3n
                 + 0.40 * r7n
                 + 0.24 * r30n                  # wet antecedent season
                 + 0.65 * sm_n
                 + 1.25 * z(river_anom)
                 + 0.35 * dd_n
                 + 0.50 * r3n * sm_n            # saturated catchment amplifies the storm
                 + 0.62 * z(river_anom) * prox  # stage matters most at the bank
                 + 0.30 * r1n * dd_n            # flashy, well-drained cells respond fast
                 - 0.26 * r30n * dd_n)

zf = (1.40 * flood_gate
      - 1.55 * (1.0 - flood_gate)
      + flood_gate * flood_trigger
      + 0.55 * shock_shared
      + 0.50 * shock_f
      + rng.logistic(0, 1.75, size=n))    # irreducible noise

# ---- Landslide latent score: common features + landslide-specific ----
slope = data["slope_degree"].to_numpy()
# Susceptibility peaks on steep-but-not-bare-rock slopes (~38 deg) and is
# near-zero on flat ground.
slope_eff = np.exp(-0.5 * ((slope - 38.0) / 15.0) ** 2) * (slope > 8.0)
curv_eff = -data["curvature"].to_numpy()          # concave (negative) hollows concentrate flow
aspect_eff = np.cos(np.deg2rad(data["aspect_degree"].to_numpy() - 225.0))  # monsoon-facing
lc_ls = data["land_cover"].map({"forest": -0.40, "shrubland": 0.10, "agriculture": 0.30,
                                "built_up": 0.22, "barren": 0.55}).to_numpy()
lith_ls = data["lithology"].map(LITHO_SUSCEPT).to_numpy()

# Terrain predisposition, and a slope gate playing the same role as the
# flood exposure gate: near-flat ground does not fail, however wet it is.
slope_gate = np.clip((slope - 1.0) / 20.0, 0.05, 1.0)

terrain_l = (1.90 * slope_eff
             + 0.45 * curv_eff
             + 0.22 * aspect_eff
             + 0.75 * lith_ls
             + 0.60 * lc_ls
             + 0.15 * elev_n)

land_trigger = (0.52 * r1n
                + 0.78 * r3n
                + 0.60 * r7n
                + 0.38 * r14n
                + 0.20 * r30n
                + 0.95 * sm_n
                + 1.05 * slope_eff * sm_n   # wet steep slopes are the real trigger
                + 0.55 * r3n * lith_ls      # weak lithology fails faster under a storm
                + 0.34 * slope_eff * r7n)

zl = (terrain_l
      - 1.95 * (1.0 - slope_gate)
      + slope_gate * land_trigger
      + 0.55 * shock_shared
      + 0.50 * shock_l
      + rng.logistic(0, 1.85, size=n))


def calibrate(scores, target_rate):
    """Find the intercept that yields the requested mean event probability."""
    lo, hi = -25.0, 25.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(scores + mid)))
        if p.mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


p_flood = 1.0 / (1.0 + np.exp(-(zf + calibrate(zf, FLOOD_RATE))))
p_land = 1.0 / (1.0 + np.exp(-(zl + calibrate(zl, LANDSLIDE_RATE))))

data["flood"] = (rng.random(n) < p_flood).astype(int)
data["landslide"] = (rng.random(n) < p_land).astype(int)


# --------------------------------------------------------------------
# 5. Constraints, rounding, ordering, export
# --------------------------------------------------------------------
COLUMNS = [
    "date", "latitude", "longitude",
    "rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm",
    "rainfall_14d_mm", "rainfall_30d_mm",
    "soil_moisture", "temperature_c", "humidity_percent", "elevation_m",
    "river_level_m", "distance_to_river_km", "drainage_density",
    "slope_degree", "aspect_degree", "curvature", "land_cover", "lithology",
    "flood", "landslide",
]

data = data.sort_values(["date", "latitude"], kind="mergesort").reset_index(drop=True)

for c in ["rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm",
          "rainfall_14d_mm", "rainfall_30d_mm"]:
    data[c] = np.round(np.clip(data[c], 0.0, None), 1)
data["soil_moisture"] = np.round(np.clip(data["soil_moisture"], 0.0, 1.0), 4)
data["temperature_c"] = np.round(np.clip(data["temperature_c"], -10.0, 55.0), 1)
data["humidity_percent"] = np.round(np.clip(data["humidity_percent"], 0.0, 100.0), 1)
data["elevation_m"] = np.round(np.clip(data["elevation_m"], 0.0, None), 1)
data["river_level_m"] = np.round(np.clip(data["river_level_m"], 0.0, None), 3)
data["distance_to_river_km"] = np.round(np.clip(data["distance_to_river_km"], 0.0, None), 2)
data["drainage_density"] = np.round(np.clip(data["drainage_density"], 0.0, None), 2)
data["slope_degree"] = np.round(np.clip(data["slope_degree"], 0.0, 90.0), 1)
data["aspect_degree"] = np.round(np.mod(data["aspect_degree"], 360.0), 1)
data["curvature"] = np.round(data["curvature"], 3)
data["date"] = data["date"].dt.strftime("%Y-%m-%d")

out = data[COLUMNS]
out.to_csv("synthetic_master_dataset.csv", index=False)

# --------------------------------------------------------------------
# 6. Console summary
# --------------------------------------------------------------------
print("SYNTHETIC / DUMMY DATA -- NOT FOR RESEARCH RESULTS")
print(f"rows={len(out)}  cols={out.shape[1]}  "
      f"dates {out['date'].min()} .. {out['date'].max()}  sites={N_SITES}")
print("\nTarget combinations:")
combo = out.groupby(["flood", "landslide"]).size()
for (f, l), c in combo.items():
    print(f"  flood={f}, landslide={l}: {c:6d}  ({100*c/len(out):5.2f}%)")
print(f"\nflood rate     : {out['flood'].mean():.4f}")
print(f"landslide rate : {out['landslide'].mean():.4f}")
print(f"phi correlation: {np.corrcoef(out['flood'], out['landslide'])[0,1]:.4f}")
print("\nRanges:")
print(out.describe().T[["min", "max", "mean"]].round(3).to_string())
print("\nCategoricals:")
print(out["land_cover"].value_counts().to_string())
print(out["lithology"].value_counts().to_string())
print("\nNulls:", int(out.isna().sum().sum()))
