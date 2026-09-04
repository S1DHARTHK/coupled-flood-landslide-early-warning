/**
 * "Why this prediction?" -- SHAP explanation of a single model prediction.
 *
 * Renders the backend's /explain/current payload. It computes NO SHAP values;
 * it only presents them. SHAP describes how each feature moved THIS model's
 * output, in log-odds space. Wording is deliberately non-causal: features
 * "push the model's prediction higher/lower", never "cause" a hazard.
 *
 * Colour semantics reuse the dashboard's warning palette:
 *   increases risk -> error (red)      decreases risk -> success (green)
 */

import type { ShapContribution, ShapExplanationResponse } from "../../types/hazard";
import { num, pct } from "./hazardUtils";

const INCREASE = "#f04438"; // error-500  -- pushes prediction higher
const DECREASE = "#12b76a"; // success-500 -- pushes prediction lower

/** Human-friendly feature label; falls back to de-underscored name. */
const FEATURE_LABELS: Record<string, string> = {
  rainfall_1d_mm: "Rainfall (1 day)",
  rainfall_3d_mm: "Rainfall (3 day)",
  rainfall_7d_mm: "Rainfall (7 day)",
  rainfall_14d_mm: "Rainfall (14 day)",
  rainfall_30d_mm: "Rainfall (30 day)",
  soil_moisture: "Soil moisture",
  temperature_c: "Temperature",
  humidity_percent: "Humidity",
  elevation_m: "Elevation",
  river_level_m: "River level",
  distance_to_river_km: "Distance to river",
  drainage_density: "Drainage density",
  slope_degree: "Slope",
  aspect_degree: "Aspect",
  curvature: "Curvature",
  land_cover: "Land cover",
  lithology: "Lithology",
};

function label(feature: string): string {
  return FEATURE_LABELS[feature] ?? feature.replace(/_/g, " ");
}

function valueStr(c: ShapContribution): string {
  if (c.value === null || c.value === undefined) return "—";
  if (typeof c.value === "string") return c.value;
  return num(c.value, 2);
}

/** One ranked contribution row with a magnitude bar. */
function ContribRow({ c, maxAbs }: { c: ShapContribution; maxAbs: number }) {
  const positive = c.shap_value > 0;
  const color = positive ? INCREASE : DECREASE;
  const widthPct = maxAbs > 0 ? (Math.abs(c.shap_value) / maxAbs) * 100 : 0;
  return (
    <div className="flex items-center gap-3 py-1.5">
      <div className="w-32 shrink-0">
        <p className="truncate text-xs font-medium text-gray-800 dark:text-white/90">
          {label(c.feature)}
          {c.is_categorical && (
            <span className="ml-1 text-[9px] uppercase text-gray-400">cat</span>
          )}
        </p>
        <p className="text-[10px] text-gray-500 dark:text-gray-400">
          value {valueStr(c)}
        </p>
      </div>
      <div className="flex flex-1 items-center gap-2">
        <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-gray-100 dark:bg-white/5">
          <div
            className="h-full rounded-full"
            style={{ width: `${widthPct}%`, background: color }}
          />
        </div>
        <span
          className="w-16 shrink-0 text-right text-xs font-semibold tabular-nums"
          style={{ color }}
        >
          {positive ? "+" : ""}
          {c.shap_value.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

/** A titled column of contributors (increasing or decreasing). */
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
    <div className="rounded-xl border border-gray-100 p-3 dark:border-gray-800">
      <div className="mb-1 flex items-center gap-2">
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ background: color }}
        />
        <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
          {title}
        </p>
      </div>
      <p className="mb-2 text-[10px] text-gray-500 dark:text-gray-400">{hint}</p>
      {items.length === 0 ? (
        <p className="py-2 text-xs text-gray-400 dark:text-gray-500">{emptyText}</p>
      ) : (
        <ul className="space-y-1.5">
          {items.map((c) => (
            <li key={c.feature} className="flex items-baseline justify-between gap-2">
              <span className="min-w-0 truncate text-xs text-gray-700 dark:text-gray-300">
                {label(c.feature)}
                <span className="ml-1 text-[10px] text-gray-400">({valueStr(c)})</span>
              </span>
              <span
                className="shrink-0 text-xs font-semibold tabular-nums"
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

export default function SHAPExplanation({
  explanation,
  topK = 8,
}: {
  explanation: ShapExplanationResponse;
  topK?: number;
}) {
  const e = explanation;
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
      <div className="flex flex-wrap items-end justify-between gap-3 rounded-xl border border-gray-100 bg-gray-50 p-4 dark:border-gray-800 dark:bg-white/[0.02]">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
            {e.model_label} — predicted {hazard} risk
          </p>
          <p className="mt-0.5 text-3xl font-bold text-gray-800 dark:text-white/90">
            {pct(e.probability, 1)}
          </p>
          <p className="mt-0.5 text-[11px] text-gray-500 dark:text-gray-400">
            {e.prediction === 1 ? "Above" : "Below"} the {pct(e.decision_threshold, 0)}{" "}
            decision threshold
            {e.sample ? ` · ${e.sample.date}` : ""}
          </p>
        </div>
        <div className="text-right">
          <p className="text-[11px] text-gray-500 dark:text-gray-400">Model base value</p>
          <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            {pct(e.base_value_probability, 1)}
          </p>
          <p className="text-[10px] text-gray-400 dark:text-gray-500">
            average prediction before features
          </p>
        </div>
      </div>

      {/* Non-causal explanation of how features moved the prediction */}
      <p className="text-xs leading-relaxed text-gray-600 dark:text-gray-400">
        Starting from the model&apos;s base value of{" "}
        <span className="font-medium">{pct(e.base_value_probability, 1)}</span>, these
        features pushed the {hazard} model&apos;s prediction to{" "}
        <span className="font-medium">{pct(e.probability, 1)}</span>. A positive value
        pushes the prediction <span className="font-medium">higher</span>; a negative
        value pushes it <span className="font-medium">lower</span>. These describe the
        model&apos;s behaviour, not physical causation.
      </p>

      {/* Two directional columns */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <ContribColumn
          title="Pushes prediction higher"
          hint="Features increasing the model's predicted risk"
          items={increasing}
          color={INCREASE}
          emptyText="No feature pushed the prediction higher."
        />
        <ContribColumn
          title="Pushes prediction lower"
          hint="Features decreasing the model's predicted risk"
          items={decreasing}
          color={DECREASE}
          emptyText="No feature pushed the prediction lower."
        />
      </div>

      {/* Ranked contribution magnitudes */}
      <div>
        <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
          Top contributing features
        </p>
        <div className="rounded-xl border border-gray-100 px-3 py-2 dark:border-gray-800">
          {top.map((c) => (
            <ContribRow key={c.feature} c={c} maxAbs={maxAbs} />
          ))}
        </div>
        <p className="mt-2 text-[10px] leading-relaxed text-gray-400 dark:text-gray-500">
          SHAP values are in log-odds space and sum, with the base value, to the model
          output (additivity check {e.additivity_check.passed ? "passed" : "failed"}).
          They explain the current XGBoost model prediction; results shown here use
          synthetic development data.
        </p>
      </div>
    </div>
  );
}
