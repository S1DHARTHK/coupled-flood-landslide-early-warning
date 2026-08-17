/**
 * Model performance page.
 *
 * Shows the two independent XGBoost baselines side by side with their stored
 * evaluation metrics and feature importance. TCDL is deliberately absent from
 * the metric tables: it is a rule-based layer, not a trained classifier, so it
 * has no accuracy or ROC-AUC of its own — its evaluation lives on the Analysis
 * and Dashboard lead-time sections.
 */

import { useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import PageBreadcrumb from "../../components/common/PageBreadCrumb";
import ComponentCard from "../../components/common/ComponentCard";
import {
  CalibrationNote,
  ConfusionMatrix,
  FeatureImportanceChart,
  MetricTiles,
  SplitPeriodTable,
} from "../../components/hazard/ModelMetrics";
import {
  EmptyBlock,
  ErrorBlock,
  LoadingBlock,
  SyntheticDataBanner,
} from "../../components/hazard/StateBlocks";
import { useApi } from "../../hooks/useApi";
import { getModelFeatures, getModelPerformance } from "../../services/api";

type ModelKey = "flood" | "landslide";

const MODEL_META: Record<ModelKey, { title: string; subtitle: string; color: string }> = {
  flood: {
    title: "Flood XGBoost V1.0",
    subtitle: "9 common environmental + 3 flood-specific features",
    color: "#2e90fa",
  },
  landslide: {
    title: "Landslide XGBoost V1.0",
    subtitle: "9 common environmental + 5 landslide-specific features",
    color: "#f79009",
  },
};

export default function ModelPerformancePage() {
  const perf = useApi(() => getModelPerformance(), []);
  const feats = useApi(() => getModelFeatures(), []);
  const [active, setActive] = useState<ModelKey>("flood");

  const model = perf.data?.models?.[active] ?? null;
  const importance = feats.data?.feature_importance?.[active];
  const contract = feats.data?.feature_contract?.[active];

  return (
    <>
      <PageMeta
        title="Model Performance | Kerala Flood–Landslide Early Warning"
        description="Evaluation metrics and feature importance for the independent Flood and Landslide XGBoost baselines."
      />
      <PageBreadcrumb pageTitle="Model Performance" />
      <SyntheticDataBanner
        dataMode={perf.data?.data_mode}
        sourceMode={perf.sourceMode}
        fallbackReason={perf.fallbackReason}
      />

      {/* Architecture reminder */}
      <div className="mb-5 rounded-2xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-white/[0.03]">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
          Pipeline architecture
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded-lg bg-blue-light-50 px-2.5 py-1 font-medium text-blue-light-700 dark:bg-blue-light-500/15 dark:text-blue-light-400">
            Flood XGBoost
          </span>
          <span className="text-gray-400">+</span>
          <span className="rounded-lg bg-orange-50 px-2.5 py-1 font-medium text-orange-700 dark:bg-orange-500/15 dark:text-orange-400">
            Landslide XGBoost
          </span>
          <span className="text-gray-400">→</span>
          <span className="rounded-lg bg-success-50 px-2.5 py-1 font-medium text-success-700 dark:bg-success-500/15 dark:text-success-400">
            TCDL (rule-based, not a model)
          </span>
          <span className="text-gray-400">→</span>
          <span className="rounded-lg bg-gray-100 px-2.5 py-1 font-medium text-gray-700 dark:bg-white/5 dark:text-gray-300">
            Warning decision
          </span>
        </div>
        <p className="mt-2 text-[11px] text-gray-500 dark:text-gray-400">
          The two classifiers are trained independently — neither sees the other's
          target or output. Coupling happens only afterwards, inside TCDL.
        </p>
      </div>

      {/* Model tabs */}
      <div className="mb-5 flex gap-2">
        {(Object.keys(MODEL_META) as ModelKey[]).map((k) => (
          <button
            key={k}
            onClick={() => setActive(k)}
            className={`rounded-lg border px-4 py-2 text-sm font-medium transition ${
              active === k
                ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-400"
                : "border-gray-200 text-gray-600 hover:border-gray-300 dark:border-gray-700 dark:text-gray-300"
            }`}
          >
            {MODEL_META[k].title}
          </button>
        ))}
      </div>

      {perf.loading ? (
        <LoadingBlock label="Loading model metrics…" height={300} />
      ) : perf.error ? (
        <ErrorBlock message={perf.error} onRetry={perf.reload} height={300} />
      ) : !model ? (
        <EmptyBlock message="No stored metrics for this model." height={300} />
      ) : (
        <div className="space-y-5">
          <ComponentCard
            title={`${MODEL_META[active].title} — Test Set Metrics`}
            desc={`${MODEL_META[active].subtitle} · target “${model.target}” · decision threshold ${model.decision_threshold}`}
          >
            <MetricTiles metrics={model.splits.test} />
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <div>
                <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
                  Confusion matrix (test set, threshold {model.decision_threshold})
                </p>
                <ConfusionMatrix metrics={model.splits.test} />
              </div>
              <div className="space-y-3">
                <CalibrationNote model={model} />
                <div className="rounded-xl border border-gray-100 p-3 dark:border-gray-800">
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
                    Training configuration
                  </p>
                  <dl className="mt-2 space-y-1 text-xs">
                    <div className="flex justify-between">
                      <dt className="text-gray-500 dark:text-gray-400">Best iteration</dt>
                      <dd className="font-medium text-gray-800 dark:text-white/90">
                        {model.best_iteration ?? "—"}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500 dark:text-gray-400">
                        Class imbalance
                      </dt>
                      <dd className="font-medium text-gray-800 dark:text-white/90">
                        {String(
                          (model.class_imbalance_handling as Record<string, unknown>)
                            ?.method ?? "—"
                        )}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500 dark:text-gray-400">
                        scale_pos_weight
                      </dt>
                      <dd className="font-medium text-gray-800 dark:text-white/90">
                        {Number(
                          (model.class_imbalance_handling as Record<string, unknown>)
                            ?.scale_pos_weight ?? NaN
                        ).toFixed(2)}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500 dark:text-gray-400">Features</dt>
                      <dd className="font-medium text-gray-800 dark:text-white/90">
                        {contract?.n_features ?? "—"}
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>
            </div>
          </ComponentCard>

          <ComponentCard
            title="Performance Across Chronological Splits"
            desc="Train, validation and unseen test periods"
          >
            <SplitPeriodTable model={model} />
          </ComponentCard>

          <ComponentCard
            title="Feature Importance"
            desc="Gain-based importance computed during training. Not recalculated in the frontend."
          >
            {feats.loading ? (
              <LoadingBlock label="Loading feature importance…" height={300} />
            ) : feats.error ? (
              <ErrorBlock message={feats.error} onRetry={feats.reload} height={300} />
            ) : !importance?.importance?.length ? (
              <EmptyBlock message="No feature importance available." height={300} />
            ) : (
              <>
                <FeatureImportanceChart
                  rows={importance.importance}
                  categoricalFeatures={importance.categorical_features ?? []}
                />
                {(importance.categorical_features?.length ?? 0) > 0 && (
                  <p className="text-[11px] text-gray-500 dark:text-gray-400">
                    <span className="mr-1 inline-block h-2 w-2 rounded-full align-middle" style={{ background: "#d95f0e" }} />
                    Categorical features ({importance.categorical_features.join(", ")})
                    are handled natively by XGBoost, not one-hot encoded.
                  </p>
                )}
                <p className="text-[11px] leading-relaxed text-gray-500 dark:text-gray-400">
                  On synthetic data these rankings describe the data generator, not real
                  flood or landslide physics.
                </p>
              </>
            )}
          </ComponentCard>
        </div>
      )}
    </>
  );
}
