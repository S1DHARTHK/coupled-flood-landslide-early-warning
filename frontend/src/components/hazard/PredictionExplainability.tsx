/**
 * Prediction Explainability section for the Analysis page.
 *
 * Ties together the Flood/Landslide model toggle, the per-prediction
 * "Why this prediction?" card (SHAPExplanation) and the global mean|SHAP|
 * importance chart (SHAPImportanceChart). All SHAP data comes from the
 * backend (/explain/current, /explain/global); nothing is computed here.
 *
 * If SHAP is unavailable the section degrades gracefully -- the rest of the
 * Analysis page keeps working.
 */

import { useState } from "react";
import ComponentCard from "../common/ComponentCard";
import SHAPExplanation from "./SHAPExplanation";
import SHAPImportanceChart from "./SHAPImportanceChart";
import { EmptyBlock, ErrorBlock, LoadingBlock } from "./StateBlocks";
import { useApi } from "../../hooks/useApi";
import { getShapExplanation, getShapGlobal } from "../../services/api";
import type { ExplainMode, HazardModel } from "../../services/api";

const MODELS: { key: HazardModel; label: string; accent: string }[] = [
  { key: "flood", label: "Flood Model", accent: "#2e90fa" },
  { key: "landslide", label: "Landslide Model", accent: "#f79009" },
];

/** Distinguish "unavailable" (backend has no SHAP) from a transient error. */
function isUnavailable(msg: string | null): boolean {
  if (!msg) return false;
  return /503|unavailable|not initialised/i.test(msg);
}

function UnavailableBlock({ height = 200 }: { height?: number }) {
  return (
    <div
      className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-gray-200 bg-gray-50 px-6 text-center dark:border-gray-800 dark:bg-white/[0.02]"
      style={{ minHeight: height }}
    >
      <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
        SHAP explanation is currently unavailable.
      </p>
      <p className="max-w-md text-xs text-gray-500 dark:text-gray-400">
        The explainability service did not respond. The rest of the dashboard is
        unaffected.
      </p>
    </div>
  );
}

export default function PredictionExplainability({
  locationId,
}: {
  locationId: string | null;
}) {
  const [model, setModel] = useState<HazardModel>("flood");
  const [mode, setMode] = useState<ExplainMode>("current");

  const explanation = useApi(
    () => getShapExplanation(model, locationId ?? "", mode),
    [model, locationId, mode]
  );
  const global = useApi(() => getShapGlobal(model), [model]);

  return (
    <div className="mt-5">
      <ComponentCard
        title="Prediction Explainability"
        desc="SHAP-based explanation of the factors influencing the model prediction."
      >
        {/* Model toggle */}
        <div className="flex flex-wrap items-center gap-2">
          {MODELS.map((m) => (
            <button
              key={m.key}
              onClick={() => setModel(m.key)}
              className={`rounded-lg border px-4 py-1.5 text-sm font-medium transition ${
                model === m.key
                  ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-400"
                  : "border-gray-200 text-gray-600 hover:border-gray-300 dark:border-gray-700 dark:text-gray-300"
              }`}
            >
              {m.label}
            </button>
          ))}
          <span className="ml-auto text-[11px] text-gray-500 dark:text-gray-400">
            SHAP explains the XGBoost model prediction. Results use synthetic
            development data.
          </span>
        </div>

        {/* Which real row to explain: latest ("current") or the location's
            highest-probability day ("peak"). Both are actual observed rows. */}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
            Explain
          </span>
          {([
            { key: "current", label: "Current day" },
            { key: "peak", label: "Peak-risk day" },
          ] as { key: ExplainMode; label: string }[]).map((o) => (
            <button
              key={o.key}
              onClick={() => setMode(o.key)}
              className={`rounded-lg border px-3 py-1 text-xs font-medium transition ${
                mode === o.key
                  ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-400"
                  : "border-gray-200 text-gray-600 hover:border-gray-300 dark:border-gray-700 dark:text-gray-300"
              }`}
            >
              {o.label}
            </button>
          ))}
          <span className="text-[11px] text-gray-400 dark:text-gray-500">
            {mode === "peak"
              ? "the location's highest-probability observed day"
              : "the latest available observation (dry season in this dataset)"}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-5 xl:grid-cols-2">
          {/* Why this prediction? */}
          <div>
            <p className="mb-2 text-sm font-semibold text-gray-800 dark:text-white/90">
              Why this prediction?
            </p>
            {!locationId ? (
              <EmptyBlock message="Select a monitoring point to explain its prediction." />
            ) : explanation.loading ? (
              <LoadingBlock label="Loading explanation…" height={260} />
            ) : explanation.error ? (
              isUnavailable(explanation.error) ? (
                <UnavailableBlock />
              ) : (
                <ErrorBlock
                  message={`Unable to load SHAP explanation. ${explanation.error}`}
                  onRetry={explanation.reload}
                  height={220}
                />
              )
            ) : explanation.data ? (
              <SHAPExplanation explanation={explanation.data} />
            ) : (
              <UnavailableBlock />
            )}
          </div>

          {/* Global importance */}
          <div>
            <p className="mb-2 text-sm font-semibold text-gray-800 dark:text-white/90">
              Global SHAP importance
            </p>
            <p className="mb-2 text-[11px] text-gray-500 dark:text-gray-400">
              Mean |SHAP value| per feature across the evaluation set — how much each
              feature moves this model&apos;s output on average.
            </p>
            {global.loading ? (
              <LoadingBlock label="Loading SHAP importance…" height={260} />
            ) : global.error ? (
              isUnavailable(global.error) ? (
                <UnavailableBlock />
              ) : (
                <ErrorBlock
                  message={`Unable to load SHAP importance. ${global.error}`}
                  onRetry={global.reload}
                  height={220}
                />
              )
            ) : global.data && global.data.importance.length > 0 ? (
              <SHAPImportanceChart
                key={`shap-imp-${model}`}
                rows={global.data.importance}
                categoricalFeatures={global.data.categorical_features}
              />
            ) : (
              <EmptyBlock message="No SHAP importance available." height={220} />
            )}
          </div>
        </div>
      </ComponentCard>
    </div>
  );
}
