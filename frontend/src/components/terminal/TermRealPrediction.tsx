/**
 * Real-model prediction by district.
 *
 * Frontend -> GET /districts/{district}/observation (a real row of the real
 * Master Dataset, read unchanged) -> POST /predict/flood and /predict/landslide
 * (the REAL XGBoost models in artifacts/models/) -> rendered here.
 *
 * The frontend computes nothing: it picks a district and a day, forwards the
 * backend's own observation to the prediction endpoints and shows the result.
 * `district` is an identifier only; missing inputs (e.g. river level where a
 * district has no gauge) are shown as missing, never filled in. TCDL is not
 * applied to these predictions.
 */

import { useEffect, useMemo, useState } from "react";

import { pct } from "../hazard/hazardUtils";
import { useApi } from "../../hooks/useApi";
import {
  ApiError,
  getDistrictObservation,
  getDistricts,
  predictFlood,
  predictLandslide,
} from "../../services/api";
import type {
  DistrictObservationResponse,
  HazardPredictionResponse,
} from "../../types/hazard";
import { TERM } from "./termColors";
import { Field, Meter, Panel, TermEmpty, TermError, TermLoading } from "./TerminalUI";

interface Result {
  observation: DistrictObservationResponse;
  flood: HazardPredictionResponse;
  landslide: HazardPredictionResponse;
}

const SPLIT_LABEL: Record<string, string> = {
  train: "train (in-sample for the models)",
  validation: "validation",
  test: "test (unseen by training)",
};

const INPUT_GROUPS: { label: string; fields: string[] }[] = [
  {
    label: "common",
    fields: [
      "rainfall_1d_mm", "rainfall_3d_mm", "rainfall_7d_mm", "rainfall_14d_mm",
      "rainfall_30d_mm", "soil_moisture", "temperature_c", "humidity_percent", "elevation_m",
    ],
  },
  { label: "flood", fields: ["river_level_m", "distance_to_river_km", "drainage_density"] },
  {
    label: "landslide",
    fields: ["slope_degree", "aspect_degree", "curvature", "land_cover", "lithology"],
  },
];

/** Readable error text, including the backend's own validation problems. */
function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    const d = (err.detail as { detail?: unknown } | undefined)?.detail;
    if (d && typeof d === "object") {
      const o = d as { error?: string; detail?: string; problems?: string[] };
      return [err.message, o.error, o.detail, ...(o.problems ?? [])]
        .filter(Boolean)
        .join(" — ");
    }
    return typeof d === "string" ? `${err.message} — ${d}` : err.message;
  }
  return err instanceof Error ? err.message : "Unknown error";
}

function formatValue(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return v;
  return Math.abs(v) >= 100 ? v.toFixed(1) : Number(v.toFixed(4)).toString();
}

function HazardCard({
  title,
  res,
}: {
  title: string;
  res: HazardPredictionResponse;
}) {
  const p = res.predictions[0];
  const warn = p.prediction === 1;
  const color = warn ? TERM.amber : TERM.phosphor;
  return (
    <div className="crt-inset p-4">
      <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">{title}</p>
      <p
        className="mt-2 text-[28px] font-bold leading-none"
        style={{ color, textShadow: `0 0 12px ${color}66` }}
      >
        {pct(p.probability, 1)}
      </p>
      <div className="mt-3">
        <Meter value={p.probability} color={color} />
      </div>
      <p className="mt-3 text-[12px]" style={{ color }}>
        {p.warning_status}
        <span className="text-[#5f8d68]">
          {" "}
          · threshold {res.decision_threshold.toFixed(2)} · model_set {res.model_set ?? "—"}
        </span>
      </p>
      {(p.missing_features?.length ?? 0) > 0 && (
        <p className="mt-1.5 text-[11px] text-[#ffd24a]">
          missing input: {p.missing_features!.join(", ")} → scored by XGBoost&apos;s learned
          missing-value branch (not imputed)
        </p>
      )}
      <p className="mt-1.5 break-all text-[10px] text-[#3d6b47]">{res.model_artifact}</p>
    </div>
  );
}

export default function TermRealPrediction() {
  const districts = useApi(() => getDistricts(), []);
  const list = useMemo(() => districts.data?.districts ?? [], [districts.data]);

  const [district, setDistrict] = useState<string>("");
  const [date, setDate] = useState<string>("");
  const [runKey, setRunKey] = useState(0);
  const [result, setResult] = useState<Result | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const info = list.find((d) => d.district === district);

  // First district, latest day: a neutral default -- the user picks the rest.
  useEffect(() => {
    if (!district && list.length) {
      setDistrict(list[0].district);
      setDate(list[0].last_date);
      setRunKey(1);
    }
  }, [list, district]);

  useEffect(() => {
    if (!runKey || !district) return;
    let cancelled = false;
    setRunning(true);
    setError(null);
    (async () => {
      try {
        const observation = await getDistrictObservation(district, date || undefined);
        const obs = [observation.observation];
        const [flood, landslide] = await Promise.all([predictFlood(obs), predictLandslide(obs)]);
        if (!cancelled) setResult({ observation, flood, landslide });
      } catch (err) {
        if (!cancelled) {
          setError(errorText(err));
          setResult(null);
        }
      } finally {
        if (!cancelled) setRunning(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // district/date are read when the user presses run, not on every keystroke
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey]);

  const obs = result?.observation;

  return (
    <Panel
      title="real-model prediction // by district"
      desc="Pick a district and a day. The backend returns that day's real observed inputs from the real Master Dataset, then scores them with the REAL Flood and Landslide XGBoost models (artifacts/models/). District is an identifier only, never a model feature. TCDL is not applied here."
    >
      {districts.loading ? (
        <TermLoading label="loading districts…" height={120} />
      ) : districts.error ? (
        <TermError
          message={`${districts.error} — real-model prediction needs the live FastAPI backend.`}
          onRetry={districts.reload}
          height={120}
        />
      ) : !list.length ? (
        <TermEmpty message="no districts returned by the backend" height={120} />
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-4">
            <Field label="district">
              <select
                className="crt-select"
                value={district}
                onChange={(e) => {
                  setDistrict(e.target.value);
                  const d = list.find((x) => x.district === e.target.value);
                  if (d && (date < d.first_date || date > d.last_date)) setDate(d.last_date);
                }}
              >
                {list.map((d) => (
                  <option key={d.district} value={d.district}>
                    {d.district}
                    {d.river_level_days === 0 ? " (no river gauge)" : ""}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="date">
              <input
                type="date"
                className="crt-input"
                value={date}
                min={info?.first_date}
                max={info?.last_date}
                onChange={(e) => setDate(e.target.value)}
              />
            </Field>
            <button
              className="crt-chip crt-chip-on px-3 py-1.5 text-[12px]"
              onClick={() => setRunKey((k) => k + 1)}
              disabled={running}
            >
              {running ? "./predict …" : "./predict"}
            </button>
            {info && (
              <span className="text-[11px] text-[#3d6b47]">
                data {info.first_date} → {info.last_date} · river level on{" "}
                {info.river_level_days.toLocaleString()} of {info.n_days.toLocaleString()} days
              </span>
            )}
          </div>

          <div className="mt-4">
            {running && !result ? (
              <TermLoading label="scoring with the real models…" height={160} />
            ) : error ? (
              <TermError message={error} onRetry={() => setRunKey((k) => k + 1)} height={160} />
            ) : !result || !obs ? (
              <TermEmpty message="choose a district and a day, then ./predict" height={160} />
            ) : (
              <div className={running ? "opacity-60" : ""}>
                <p className="text-[12px] text-[#cfe9d5]">
                  {obs.district} · {obs.date}
                  <span className="text-[#5f8d68]">
                    {" "}
                    · split {obs.split ? SPLIT_LABEL[obs.split] ?? obs.split : "—"} · recorded
                    event: flood {obs.recorded_labels.flood ?? "—"}, landslide{" "}
                    {obs.recorded_labels.landslide ?? "—"}
                  </span>
                </p>

                <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
                  <HazardCard title="Flood XGBoost V1.0 — real" res={result.flood} />
                  <HazardCard title="Landslide XGBoost V1.0 — real" res={result.landslide} />
                </div>

                <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
                  {INPUT_GROUPS.map((g) => (
                    <div key={g.label} className="crt-inset p-3">
                      <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
                        {g.label} inputs
                      </p>
                      <dl className="mt-2 space-y-1 text-[11px]">
                        {g.fields.map((f) => {
                          const v = obs.observation[f];
                          const missing = v === null || v === undefined;
                          return (
                            <div key={f} className="flex justify-between gap-3">
                              <dt className="text-[#5f8d68]">{f}</dt>
                              <dd style={{ color: missing ? TERM.amber : "#cfe9d5" }}>
                                {missing ? "null (missing)" : formatValue(v)}
                              </dd>
                            </div>
                          );
                        })}
                      </dl>
                    </div>
                  ))}
                </div>

                <p className="mt-3 text-[10px] leading-relaxed text-[#3d6b47]">
                  {result.flood.model_note} Probabilities are not calibrated (training used
                  scale_pos_weight); compare them, do not read them as literal likelihoods.
                  Recorded event 0 means no event was reported, not that none occurred.
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </Panel>
  );
}
