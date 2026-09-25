/**
 * Centralized API service for the Kerala Flood-Landslide Early Warning System.
 *
 * ALL backend communication goes through this module. Components never call
 * fetch() directly, and they contain no ML or TCDL logic -- the frontend only
 * renders what the backend computes.
 *
 * DEVELOPMENT FALLBACK
 * --------------------
 * If the live API is unreachable, requests fall back to `devFixture.json`,
 * which holds RECORDED RESPONSES from this project's own FastAPI backend
 * (re-record with `python backend/tools/record_dev_fixture.py`). It is not
 * hand-written fake data, and it is confined to this module so the rest of
 * the app cannot depend on it. Every response carries `sourceMode` so the UI
 * can state plainly where the numbers came from. A request the recording
 * cannot answer exactly (another day, another filter) is NOT substituted --
 * the original error propagates instead.
 *
 * ENDPOINT MAPPING
 * ----------------
 * The backend already implements these routes; this client targets them
 * directly rather than inventing parallel names:
 *   GET  /health                  GET  /models/status
 *   POST /predict/flood           POST /predict/landslide
 *   POST /predict/coupled
 *   GET  /current                 GET  /warnings        GET /trends
 *   GET  /evaluation              -> lead-time + system comparison
 *   GET  /leadtime                -> per-event lead-time records
 *   GET  /models/performance      GET  /models/features
 *   GET  /locations
 *   GET  /districts               GET  /districts/{district}/observation
 *
 * MODEL SETS
 * ----------
 * Every route is served by the REAL pipeline (artifacts/: XGBoost models,
 * TCDL outputs, SHAP) unless the backend was started with
 * CAPSTONE_DASHBOARD_DATA=synthetic. Every response carries `data_mode`, so
 * each view labels its own source rather than assuming one.
 *
 * LOCATIONS AND DATES
 * -------------------
 * `locationId` accepts a backend location_id or a district name. `asOf` /
 * `end` replay history: the dashboard shows what the system produced on that
 * day, from the stored pipeline outputs.
 */

import fixture from "../data/devFixture.json";
import type {
  CurrentResponse,
  DistrictObservationResponse,
  DistrictsResponse,
  EvaluationResponse,
  FeaturesResponse,
  HazardPredictionResponse,
  HealthResponse,
  LeadTimeResponse,
  Observation,
  LocationsResponse,
  ModelsStatusResponse,
  PerformanceResponse,
  ShapExplanationResponse,
  ShapGlobalResponse,
  TrendsResponse,
  WarningsResponse,
} from "../types/hazard";

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

const REQUEST_TIMEOUT_MS = 12_000;

export type SourceMode = "live" | "fixture";

/** Every result states where it came from, so the UI never has to guess. */
export interface ApiResult<T> {
  data: T;
  sourceMode: SourceMode;
  /** Present when the live API failed and the fixture was used instead. */
  fallbackReason?: string;
}

export class ApiError extends Error {
  status?: number;
  detail?: unknown;
  constructor(message: string, status?: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
      let detail: unknown;
      try {
        detail = await res.json();
      } catch {
        detail = await res.text();
      }
      throw new ApiError(
        `Request to ${path} failed with HTTP ${res.status}`,
        res.status,
        detail
      );
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Try the live API; on failure fall back to the recorded fixture.
 *
 * `pickFixture` may return undefined when the fixture has no recording for
 * the requested query -- in that case the original error is rethrown rather
 * than substituting an unrelated response.
 */
async function withFallback<T>(
  path: string,
  pickFixture: () => T | undefined,
  init?: RequestInit
): Promise<ApiResult<T>> {
  try {
    const data = await request<T>(path, init);
    return { data, sourceMode: "live" };
  } catch (err) {
    const fallback = pickFixture();
    if (fallback === undefined) throw err;
    const reason =
      err instanceof ApiError
        ? err.message
        : `Live API unreachable at ${API_BASE_URL}`;
    return { data: fallback, sourceMode: "fixture", fallbackReason: reason };
  }
}

/** Shape of the recording written by backend/tools/record_dev_fixture.py. */
interface DevFixture {
  health?: HealthResponse;
  models_status?: ModelsStatusResponse;
  locations?: LocationsResponse;
  current?: CurrentResponse;
  warnings?: WarningsResponse;
  evaluation?: Partial<Record<HazardScope, EvaluationResponse>>;
  leadtime?: LeadTimeResponse;
  models_performance?: PerformanceResponse;
  models_features?: FeaturesResponse;
  trends?: Record<string, TrendsResponse>;
  shap?: {
    global?: Partial<Record<HazardModel, ShapGlobalResponse>>;
    current?: Partial<
      Record<ExplainMode, Partial<Record<HazardModel, Record<string, ShapExplanationResponse>>>>
    >;
  };
}

const fx = fixture as unknown as DevFixture;

// ---------------------------------------------------------------------
// System
// ---------------------------------------------------------------------
export const getHealth = () =>
  withFallback<HealthResponse>("/health", () => fx.health);

export const getModelsStatus = () =>
  withFallback<ModelsStatusResponse>("/models/status", () => fx.models_status);

// ---------------------------------------------------------------------
// Monitoring
// ---------------------------------------------------------------------
export const getLocations = () =>
  withFallback<LocationsResponse>("/locations", () => fx.locations);

const matchesLocation = (
  l: { location_id: string; district?: string | null },
  id: string
) => l.location_id === id || (l.district ?? "").toLowerCase() === id.toLowerCase();

export const getCurrent = (locationId?: string, asOf?: string) => {
  const p = new URLSearchParams();
  if (locationId) p.set("location_id", locationId);
  if (asOf) p.set("as_of", asOf);
  const qs = p.toString() ? `?${p.toString()}` : "";
  return withFallback<CurrentResponse>(`/current${qs}`, () => {
    const all = fx.current as CurrentResponse | undefined;
    if (!all) return undefined;
    // The recording holds one day only; never present it as another day.
    if (asOf && asOf !== all.as_of) return undefined;
    if (!locationId) return all;
    const one = all.locations.filter((l) => matchesLocation(l, locationId));
    return one.length ? { ...all, locations: one, n_locations: one.length } : undefined;
  });
};

export const getTrends = (locationId: string, limit = 200, end?: string) => {
  const p = new URLSearchParams({ location_id: locationId, limit: String(limit) });
  if (end) p.set("end", end);
  return withFallback<TrendsResponse>(`/trends?${p.toString()}`, () => {
    const rec = (fx.trends ?? {})[locationId] as TrendsResponse | undefined;
    if (!rec) return undefined;
    if (end && rec.series.length && rec.series[rec.series.length - 1].date !== end) {
      return undefined;
    }
    return { ...rec, series: rec.series.slice(-limit), n_records: Math.min(limit, rec.n_records) };
  });
};

/**
 * Empty trends payload for the render pass before a location is known.
 *
 * Without this the first render would request `/trends?location_id=`, which the
 * backend correctly rejects, and the resulting mount/unmount churn races
 * ApexCharts' async renderer. Resolving locally keeps the shape identical.
 */
export const emptyTrends = (locationId = ""): Promise<ApiResult<TrendsResponse>> =>
  Promise.resolve({
    data: {
      data_mode: "",
      location_id: locationId,
      n_records: 0,
      parameters: {},
      source: "not requested — no location selected yet",
      series: [],
    },
    sourceMode: "live",
  });

export interface WarningQuery {
  locationId?: string;
  start?: string;
  end?: string;
  split?: "train" | "validation" | "test";
  warningOnly?: boolean;
  limit?: number;
}

export const getWarnings = (q: WarningQuery = {}) => {
  const p = new URLSearchParams();
  if (q.locationId) p.set("location_id", q.locationId);
  if (q.start) p.set("start", q.start);
  if (q.end) p.set("end", q.end);
  if (q.split) p.set("split", q.split);
  if (q.warningOnly !== undefined) p.set("warning_only", String(q.warningOnly));
  p.set("limit", String(q.limit ?? 400));
  return withFallback<WarningsResponse>(`/warnings?${p.toString()}`, () => {
    const all = fx.warnings as WarningsResponse | undefined;
    if (!all) return undefined;
    // The recording holds the most recent warning rows only. A request for
    // every timestep, a split, or a range older than the recording cannot be
    // honoured offline, so signal that rather than mislead.
    if (q.warningOnly === false || q.split) return undefined;
    const oldest = all.records.length ? all.records[0].date : "";
    if (all.truncated && (!q.start || q.start < oldest)) return undefined;
    let recs = all.records;
    if (q.locationId) recs = recs.filter((r) => matchesLocation(r, q.locationId!));
    if (q.start) recs = recs.filter((r) => r.date >= q.start!);
    if (q.end) recs = recs.filter((r) => r.date <= q.end!);
    recs = recs.slice(-(q.limit ?? 400));
    return { ...all, records: recs, n_records: recs.length, n_matching: recs.length,
             truncated: false };
  });
};

// ---------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------
export type HazardScope = "any_hazard" | "flood" | "landslide";

export const getEvaluation = (hazardType: HazardScope = "any_hazard") =>
  withFallback<EvaluationResponse>(
    `/evaluation?hazard_type=${hazardType}`,
    () => (fx.evaluation ?? {})[hazardType]
  );

export const getLeadTime = (opts: {
  system?: string;
  hazardType?: HazardScope;
  detectedOnly?: boolean;
  limit?: number;
} = {}) => {
  const p = new URLSearchParams();
  if (opts.system) p.set("system", opts.system);
  if (opts.hazardType) p.set("hazard_type", opts.hazardType);
  if (opts.detectedOnly) p.set("detected_only", "true");
  p.set("limit", String(opts.limit ?? 2000));
  return withFallback<LeadTimeResponse>(`/leadtime?${p.toString()}`, () => {
    const all = fx.leadtime as LeadTimeResponse | undefined;
    if (!all) return undefined;
    let recs = all.records;
    if (opts.system) recs = recs.filter((r) => r.system === opts.system);
    if (opts.hazardType) recs = recs.filter((r) => r.hazard_type === opts.hazardType);
    if (opts.detectedOnly) recs = recs.filter((r) => r.detected === 1);
    return { ...all, records: recs, n_records: recs.length };
  });
};

export const getModelPerformance = () =>
  withFallback<PerformanceResponse>(
    "/models/performance",
    () => fx.models_performance
  );

export const getModelFeatures = () =>
  withFallback<FeaturesResponse>("/models/features", () => fx.models_features);

// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// SHAP explainability
//
// SHAP is computed in the ML layer and served by the backend. These
// functions only fetch it. The fixture fallback holds RECORDED backend
// responses (never hand-written SHAP values); if a recording is absent
// the original error propagates so the UI shows an unavailable state
// rather than substituting unrelated numbers.
// ---------------------------------------------------------------------
export type HazardModel = "flood" | "landslide";

export const getShapGlobal = (model: HazardModel) =>
  withFallback<ShapGlobalResponse>(
    `/explain/global?model=${model}`,
    () => (fx.shap?.global ?? {})[model]
  );

export type ExplainMode = "current" | "peak";

export const getShapExplanation = (
  model: HazardModel,
  locationId: string,
  mode: ExplainMode = "current",
  date?: string
) => {
  const p = new URLSearchParams({ model, location_id: locationId, mode });
  if (date && mode === "current") p.set("date", date);
  return withFallback<ShapExplanationResponse>(`/explain/current?${p.toString()}`, () => {
    const rec = (((fx.shap?.current ?? {})[mode] ?? {})[model] ?? {})[locationId] as
      | ShapExplanationResponse
      | undefined;
    if (rec && date && mode === "current" && rec.sample?.date !== date) return undefined;
    return rec;
  });
};


// Real district data (live only -- real observations are never substituted)
// ---------------------------------------------------------------------
export const getDistricts = () =>
  withFallback<DistrictsResponse>("/districts", () => undefined);

export const getDistrictObservation = (district: string, date?: string) => {
  const qs = date ? `?date=${encodeURIComponent(date)}` : "";
  return request<DistrictObservationResponse>(
    `/districts/${encodeURIComponent(district)}/observation${qs}`
  );
};

// ---------------------------------------------------------------------
// Prediction (live only -- no fixture, because a prediction for
// caller-supplied input cannot be faked without inventing results)
// ---------------------------------------------------------------------
export const predictFlood = (observations: (Observation | Record<string, unknown>)[]) =>
  request<HazardPredictionResponse>("/predict/flood", {
    method: "POST",
    body: JSON.stringify({ observations }),
  });

export const predictLandslide = (
  observations: (Observation | Record<string, unknown>)[]
) =>
  request<HazardPredictionResponse>("/predict/landslide", {
    method: "POST",
    body: JSON.stringify({ observations }),
  });

export const predictCoupled = (observations: Record<string, unknown>[]) =>
  request("/predict/coupled", {
    method: "POST",
    body: JSON.stringify({ observations }),
  });
