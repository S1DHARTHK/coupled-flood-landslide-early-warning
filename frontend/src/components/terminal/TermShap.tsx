/**
 * SHAP explainability — a section of its own on the dashboard.
 *
 * This is deliberately kept separate from the TCDL warning readout above it:
 * TCDL decides whether a warning exists, SHAP only explains how each input
 * feature moved ONE XGBoost model's output for ONE observed row. Nothing here
 * computes SHAP values; they come from the backend (/explain/current,
 * /explain/global) and are presented verbatim.
 *
 * Wording stays non-causal throughout: features "push the model's prediction
 * higher/lower", never "cause" a hazard. If the explainability service is
 * unavailable the section degrades on its own and the rest of the dashboard
 * keeps working.
 *
 * Colour: amber = pushes the prediction higher, phosphor green = pushes it
 * lower, matching the severity ramp used across the terminal theme.
 */

import { useState } from "react";
import type {
  ShapContribution,
  ShapExplanationResponse,
  ShapImportanceRow,
} from "../../types/hazard";
import { num, pct } from "../hazard/hazardUtils";
import { useApi } from "../../hooks/useApi";
import { getShapExplanation, getShapGlobal } from "../../services/api";
import type { ExplainMode, HazardModel } from "../../services/api";
import {
  ChipButton,
  Panel,
  Prompt,
  SectionHead,
  TermEmpty,
  TermError,
  TermLoading,
} from "./TerminalUI";
import { TERM } from "./termColors";

const HIGHER = TERM.amber; // pushes the model's prediction higher
const LOWER = TERM.phosphor; // pushes it lower

const FEATURE_LABELS: Record<string, string> = {
  rainfall_1d_mm: "rainfall_1d",
  rainfall_3d_mm: "rainfall_3d",
  rainfall_7d_mm: "rainfall_7d",
  rainfall_14d_mm: "rainfall_14d",
  rainfall_30d_mm: "rainfall_30d",
  soil_moisture: "soil_moisture",
  temperature_c: "temperature",
  humidity_percent: "humidity",
  elevation_m: "elevation",
  river_level_m: "river_level",
  distance_to_river_km: "dist_to_river",
  drainage_density: "drainage_density",
  slope_degree: "slope",
  aspect_degree: "aspect",
  curvature: "curvature",
  land_cover: "land_cover",
  lithology: "lithology",
};

const label = (f: string) => FEATURE_LABELS[f] ?? f;

function valueStr(c: ShapContribution): string {
  if (c.value === null || c.value === undefined) return "—";
  if (typeof c.value === "string") return c.value;
  return num(c.value, 2);
}

/** Distinguish "backend has no SHAP" from a transient error. */
function isUnavailable(msg: string | null): boolean {
  if (!msg) return false;
  return /503|unavailable|not initialised/i.test(msg);
}

function UnavailableBlock({ height = 200 }: { height?: number }) {
  return (
    <div
      className="crt-inset flex flex-col items-center justify-center gap-2 px-6 text-center"
      style={{ minHeight: height }}
    >
      <p className="text-[12px] text-[#5f8d68]">
        <span className="text-[#ffd24a]">[!]</span> shap explanation unavailable
      </p>
      <p className="max-w-md text-[11px] text-[#3d6b47]">
        The explainability service did not respond. The rest of the dashboard is
        unaffected.
      </p>
    </div>
  );
}

/** One signed contribution row, drawn as a bar either side of a zero axis. */
function ContribRow({ c, maxAbs }: { c: ShapContribution; maxAbs: number }) {
  const higher = c.shap_value > 0;
  const color = higher ? HIGHER : LOWER;
  const w = maxAbs > 0 ? (Math.abs(c.shap_value) / maxAbs) * 50 : 0;
  return (
    <div className="flex items-center gap-3 py-1">
      <div className="w-[130px] shrink-0">
        <p className="truncate text-[11px] text-[#cfe9d5]">
          {label(c.feature)}
          {c.is_categorical && (
            <span className="ml-1 text-[9px] text-[#1c7a3c]">cat</span>
          )}
        </p>
        <p className="text-[10px] text-[#3d6b47]">= {valueStr(c)}</p>
      </div>

      {/* Diverging track: left half pushes lower, right half pushes higher. */}
      <div className="crt-inset relative h-[10px] flex-1">
        <span className="absolute inset-y-0 left-1/2 w-px bg-[#1f4d1f]" />
        <span
          className="absolute inset-y-[1px]"
          style={{
            width: `${w}%`,
            [higher ? "left" : "right"]: "50%",
            background: `linear-gradient(90deg, ${color}55, ${color})`,
            boxShadow: `0 0 10px ${color}70`,
          }}
        />
      </div>

      <span
        className="w-[64px] shrink-0 text-right text-[11px] tabular-nums"
        style={{ color }}
      >
        {higher ? "+" : ""}
        {c.shap_value.toFixed(3)}
      </span>
    </div>
  );
}

/** A directional column: what pushed the prediction up, and what pushed it down. */
function ContribColumn({
  title,
  hint,
  items,
  color,
  emptyText,
}: {
  title: string;
  hint: string;
  items: ShapContribution[];
  color: string;
  emptyText: string;
}) {
  return (
    <div className="crt-inset p-3">
      <p className="text-[11px]" style={{ color }}>
        {title}
      </p>
      <p className="mb-2 mt-0.5 text-[10px] text-[#3d6b47]">{hint}</p>
      {items.length === 0 ? (
        <p className="py-2 text-[11px] text-[#3d6b47]">{emptyText}</p>
      ) : (
        <ul className="space-y-1">
          {items.map((c) => (
            <li key={c.feature} className="flex items-baseline justify-between gap-2">
              <span className="min-w-0 truncate text-[11px] text-[#5f8d68]">
                {label(c.feature)}
                <span className="ml-1 text-[10px] text-[#1c7a3c]">
                  ({valueStr(c)})
                </span>
              </span>
              <span
                className="shrink-0 text-[11px] tabular-nums"
                style={{ color }}
              >
                {c.shap_value > 0 ? "+" : ""}
                {c.shap_value.toFixed(3)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** "Why this prediction?" — one row, one model, ranked contributions. */
function Explanation({
  e,
  topK = 8,
}: {
  e: ShapExplanationResponse;
  topK?: number;
}) {
  const hazard = e.model === "flood" ? "flood" : "landslide";
  const ranked = [...e.contributions].sort(
    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
  );
  const top = ranked.slice(0, topK);
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.shap_value)), 1e-9);
  const increasing = ranked.filter((c) => c.shap_value > 0).slice(0, 5);
  const decreasing = ranked.filter((c) => c.shap_value < 0).slice(0, 5);

  return (
    <div className="space-y-4">
      {/* Prediction headline */}
      <div className="crt-inset flex flex-wrap items-end justify-between gap-3 p-4">
        <div>
          <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
            {e.model_label} — predicted {hazard} risk
          </p>
          <p
            className="mt-1 text-[28px] font-bold leading-none text-[#eafff1]"
            style={{ textShadow: "0 0 14px rgba(57,255,122,0.35)" }}
          >
            {pct(e.probability, 1)}
          </p>
          <p className="mt-1 text-[11px] text-[#5f8d68]">
            {e.prediction === 1 ? "above" : "below"} the{" "}
            {pct(e.decision_threshold, 0)} decision threshold
            {e.sample ? ` · ${e.sample.date}` : ""}
            {e.sample?.district ? ` · ${e.sample.district}` : ""}
          </p>
          {e.sample && (
            <p className="mt-0.5 text-[10px] text-[#3d6b47]">
              recorded {hazard}:{" "}
              {e.sample.actual_label === 1
                ? "yes"
                : e.sample.actual_label === 0
                  ? "not reported"
                  : "—"}
              {e.sample.split ? ` · ${e.sample.split} period` : ""}
            </p>
          )}
        </div>
        <div className="text-right">
          <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
            model base value
          </p>
          <p className="mt-1 text-[15px] text-[#cfe9d5]">
            {pct(e.base_value_probability, 1)}
          </p>
          <p className="text-[10px] text-[#3d6b47]">
            average prediction before features
          </p>
        </div>
      </div>

      <p className="text-[11px] leading-relaxed text-[#5f8d68]">
        Starting from the model&apos;s base value of{" "}
        <span className="text-[#39ff7a]">{pct(e.base_value_probability, 1)}</span>,
        these features pushed the {hazard} model&apos;s prediction to{" "}
        <span className="text-[#39ff7a]">{pct(e.probability, 1)}</span>. A positive
        value pushes the prediction{" "}
        <span style={{ color: HIGHER }}>higher</span>; a negative value pushes it{" "}
        <span style={{ color: LOWER }}>lower</span>. These describe the model&apos;s
        behaviour, not physical causation.
      </p>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <ContribColumn
          title="[^] pushes prediction higher"
          hint="features increasing the model's predicted risk"
          items={increasing}
          color={HIGHER}
          emptyText="no feature pushed the prediction higher."
        />
        <ContribColumn
          title="[v] pushes prediction lower"
          hint="features decreasing the model's predicted risk"
          items={decreasing}
          color={LOWER}
          emptyText="no feature pushed the prediction lower."
        />
      </div>

      <div>
        <p className="mb-1.5 text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
          top contributing features
        </p>
        <div className="crt-inset px-3 py-2">
          {top.map((c) => (
            <ContribRow key={c.feature} c={c} maxAbs={maxAbs} />
          ))}
        </div>
        <p className="mt-2 text-[10px] leading-relaxed text-[#3d6b47]">
          SHAP values are in log-odds space and sum, with the base value, to the model
          output (additivity check{" "}
          <span
            style={{ color: e.additivity_check.passed ? TERM.phosphor : TERM.red }}
          >
            {e.additivity_check.passed ? "passed" : "failed"}
          </span>
          ). They explain the XGBoost model prediction shown above
          {e.data_mode === "synthetic"
            ? "; results shown here use synthetic development data."
            : " on the real Kerala district-day data; they describe model behaviour, not physical causation."}
        </p>
      </div>
    </div>
  );
}

/** Global mean|SHAP| importance, drawn as ASCII meters instead of a chart. */
function GlobalImportance({
  rows,
  categoricalFeatures = [],
}: {
  rows: ShapImportanceRow[];
  categoricalFeatures?: string[];
}) {
  const sorted = [...rows].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);
  const max = Math.max(...sorted.map((r) => r.mean_abs_shap), 1e-9);

  return (
    <div className="space-y-1.5">
      {sorted.map((r) => {
        const cat =
          r.is_categorical === true || categoricalFeatures.includes(r.feature);
        const color = cat ? TERM.amber : TERM.phosphor;
        return (
          <div key={r.feature} className="flex items-center gap-3">
            <span className="w-[130px] shrink-0 truncate text-[11px] text-[#5f8d68]">
              {label(r.feature)}
              {cat && <span className="ml-1 text-[9px] text-[#1c7a3c]">cat</span>}
            </span>
            <div className="crt-meter flex-1">
              <div
                className="crt-meter-fill"
                style={{
                  width: `${(r.mean_abs_shap / max) * 100}%`,
                  background: `linear-gradient(90deg, ${color}55, ${color})`,
                  boxShadow: `0 0 10px ${color}70`,
                }}
              />
            </div>
            <span className="w-[110px] shrink-0 text-right text-[10px] tabular-nums text-[#3d6b47]">
              {r.mean_abs_shap.toFixed(3)}
              <span className="ml-1.5 text-[#1c7a3c]">
                {((r.share_of_total ?? 0) * 100).toFixed(1)}%
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- section */

export default function TermShapSection({
  locationId,
  date = null,
}: {
  locationId: string | null;
  /** The day the page is showing (null = latest). Used by --current-day. */
  date?: string | null;
}) {
  const [model, setModel] = useState<HazardModel>("flood");
  const [mode, setMode] = useState<ExplainMode>("current");

  // Don't ask the backend to explain "no location": on the first render the
  // active point is not resolved yet, and `/explain/current?location_id=` is
  // correctly rejected. The UI shows an empty state for this case, so the
  // rejection below is never rendered. Mirrors `emptyTrends()` in the API layer.
  const explanation = useApi(
    () =>
      locationId
        ? getShapExplanation(model, locationId, mode, date ?? undefined)
        : Promise.reject(new Error("no location selected")),
    [model, locationId, mode, date]
  );
  const global = useApi(() => getShapGlobal(model), [model]);

  return (
    <section>
      <Prompt
        command={`./explain --shap --model=${model} --row=${mode}`}
        comment="model explainability"
      />
      <SectionHead
        label="prediction explainability // shap"
        title="Why the model predicted this"
        desc="A separate layer from the TCDL warning above. TCDL decides whether a warning exists; SHAP (SHapley Additive exPlanations) attributes one model's single prediction across its input features, showing how much each feature moved the output in log-odds space. It explains model behaviour, never physical causation, and it never changes a warning."
      />

      <Panel>
        {/* Which model, and which observed row to explain. */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
            model
          </span>
          <ChipButton on={model === "flood"} onClick={() => setModel("flood")}>
            flood_xgboost
          </ChipButton>
          <ChipButton on={model === "landslide"} onClick={() => setModel("landslide")}>
            landslide_xgboost
          </ChipButton>

          <span className="ml-4 text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
            explain
          </span>
          <ChipButton on={mode === "current"} onClick={() => setMode("current")}>
            --current-day
          </ChipButton>
          <ChipButton on={mode === "peak"} onClick={() => setMode("peak")}>
            --peak-risk-day
          </ChipButton>

          <span className="ml-auto text-[10px] text-[#3d6b47]">
            {mode === "peak"
              ? "the location's highest-probability day in the test period"
              : date
                ? `the day shown above (${date})`
                : "the latest available observation"}
          </span>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-6 xl:grid-cols-2">
          {/* Per-prediction explanation */}
          <div>
            <p className="mb-2 text-[12px] font-semibold text-[#39ff7a] crt-glow-soft">
              why this prediction?
            </p>
            {!locationId ? (
              <TermEmpty message="select a location to explain its prediction" />
            ) : explanation.loading ? (
              <TermLoading label="computing shap attribution…" height={260} />
            ) : explanation.error ? (
              isUnavailable(explanation.error) ? (
                <UnavailableBlock />
              ) : (
                <TermError
                  message={`unable to load SHAP explanation. ${explanation.error}`}
                  onRetry={explanation.reload}
                  height={220}
                />
              )
            ) : explanation.data ? (
              <Explanation e={explanation.data} />
            ) : (
              <UnavailableBlock />
            )}
          </div>

          {/* Global importance */}
          <div>
            <p className="mb-2 text-[12px] font-semibold text-[#39ff7a] crt-glow-soft">
              global shap importance
            </p>
            <p className="mb-3 text-[11px] leading-relaxed text-[#5f8d68]">
              Mean |SHAP value| per feature across the evaluation set — how much each
              feature moves this model&apos;s output on average. Amber marks
              categorical features.
            </p>
            {global.loading ? (
              <TermLoading label="loading shap importance…" height={260} />
            ) : global.error ? (
              isUnavailable(global.error) ? (
                <UnavailableBlock />
              ) : (
                <TermError
                  message={`unable to load SHAP importance. ${global.error}`}
                  onRetry={global.reload}
                  height={220}
                />
              )
            ) : global.data && global.data.importance.length > 0 ? (
              <GlobalImportance
                rows={global.data.importance}
                categoricalFeatures={global.data.categorical_features}
              />
            ) : (
              <TermEmpty message="no shap importance available" height={220} />
            )}
          </div>
        </div>
      </Panel>
    </section>
  );
}
