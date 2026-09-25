/**
 * Model performance page, rendered as a CRT terminal session.
 *
 * Shows the two independent XGBoost baselines side by side with their stored
 * evaluation metrics and feature importance. TCDL is deliberately absent from
 * the metric tables: it is a rule-based layer, not a trained classifier, so it
 * has no accuracy or ROC-AUC of its own — its evaluation lives on the Analysis
 * and Dashboard lead-time sections.
 */

import { useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import {
  ChipButton,
  Panel,
  Prompt,
  Rule,
  SectionHead,
  TerminalWindow,
  TermEmpty,
  TermError,
  TermLoading,
  TermSyntheticBanner,
} from "../../components/terminal/TerminalUI";
import { TERM } from "../../components/terminal/termColors";
import {
  TermCalibrationNote,
  TermConfusionMatrix,
  TermFeatureImportance,
  TermMetricTiles,
  TermSplitPeriodTable,
} from "../../components/terminal/TermModelMetrics";
import TermRealPrediction from "../../components/terminal/TermRealPrediction";
import { useApi } from "../../hooks/useApi";
import { getModelFeatures, getModelPerformance } from "../../services/api";

type ModelKey = "flood" | "landslide";

const MODEL_META: Record<
  ModelKey,
  { title: string; command: string; subtitle: string; color: string }
> = {
  flood: {
    title: "Flood XGBoost V1.0",
    command: "flood_xgboost",
    subtitle: "9 common environmental + 3 flood-specific features",
    color: TERM.dim,
  },
  landslide: {
    title: "Landslide XGBoost V1.0",
    command: "landslide_xgboost",
    subtitle: "9 common environmental + 5 landslide-specific features",
    color: TERM.amber,
  },
};

/** One stage in the pipeline strip. */
function Stage({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="rounded-[3px] px-2.5 py-1 text-[11px]"
      style={{ color, border: `1px solid ${color}55`, background: `${color}0d` }}
    >
      {label}
    </span>
  );
}

export default function ModelPerformancePage() {
  const perf = useApi(() => getModelPerformance(), []);
  const feats = useApi(() => getModelFeatures(), []);
  const [active, setActive] = useState<ModelKey>("flood");

  const model = perf.data?.models?.[active] ?? null;
  const importance = feats.data?.feature_importance?.[active];
  const contract = feats.data?.feature_contract?.[active];
  // Wording follows whichever data mode the response reports.
  const isReal = perf.data?.data_mode === "real";

  return (
    <>
      <PageMeta
        title="Model Performance | Kerala Flood–Landslide Early Warning"
        description="Evaluation metrics and feature importance for the independent Flood and Landslide XGBoost baselines."
      />

      <TerminalWindow
        path="~/model-performance"
        status={
          <span className="flex items-center gap-2 text-[13px] text-[#5f8d68]">
            <span className="text-[#39ff7a]">2</span> models loaded
          </span>
        }
      >
        <div className="px-4 py-8 sm:px-8 sm:py-10">
          <Prompt
            command="./models --metrics --feature-importance"
            comment="stored training results"
            cwd="~/model-performance"
          />
          <h1 className="mt-4 text-[26px] font-bold leading-tight tracking-tight text-[#eafff1] sm:text-[30px] [text-shadow:0_0_14px_rgba(57,255,122,0.35)]">
            Model Performance
          </h1>
          <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-[#5f8d68] sm:text-[13px]">
            Metrics are read from the backend&apos;s stored training results. Nothing
            is recomputed and no model is trained here.
          </p>

          <div className="mt-5">
            <TermSyntheticBanner
              dataMode={perf.data?.data_mode}
              sourceMode={perf.sourceMode}
              fallbackReason={perf.fallbackReason}
            />
          </div>

          {/* ----------------------------------- pipeline architecture */}
          <div className="mt-6">
            <Panel title="pipeline architecture">
              <div className="flex flex-wrap items-center gap-2">
                <Stage label="Flood XGBoost" color={TERM.dim} />
                <span className="text-[#1c7a3c]">+</span>
                <Stage label="Landslide XGBoost" color={TERM.amber} />
                <span className="text-[#1c7a3c]">-&gt;</span>
                <Stage label="TCDL (rule-based, not a model)" color={TERM.phosphor} />
                <span className="text-[#1c7a3c]">-&gt;</span>
                <Stage label="Warning decision" color={TERM.sage} />
              </div>
              <p className="mt-3 text-[11px] leading-relaxed text-[#5f8d68]">
                The two classifiers are trained independently — neither sees the
                other&apos;s target or output. Coupling happens only afterwards, inside
                TCDL, which is why TCDL has no accuracy or ROC-AUC of its own.
              </p>
            </Panel>
          </div>

          {/* ------------------------------- real-model prediction */}
          <div className="mt-6">
            <Prompt
              command="./predict --district <name> --date <YYYY-MM-DD>"
              comment="real XGBoost models via FastAPI"
              cwd="~/model-performance"
            />
            <div className="mt-3">
              <TermRealPrediction />
            </div>
          </div>

          {/* --------------------------------------------- model tabs */}
          <div className="mt-6 flex flex-wrap items-center gap-2">
            <span className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
              model
            </span>
            {(Object.keys(MODEL_META) as ModelKey[]).map((k) => (
              <ChipButton key={k} on={active === k} onClick={() => setActive(k)}>
                {MODEL_META[k].command}
              </ChipButton>
            ))}
          </div>

          <Rule />

          {perf.loading ? (
            <TermLoading label="loading model metrics…" height={300} />
          ) : perf.error ? (
            <TermError message={perf.error} onRetry={perf.reload} height={300} />
          ) : !model ? (
            <TermEmpty message="no stored metrics for this model" height={300} />
          ) : (
            <div className="space-y-8">
              {/* ------------------------------------ test set metrics */}
              <section>
                <Prompt
                  command={`cat ${MODEL_META[active].command}/test_metrics.json`}
                  comment={`target "${model.target}"`}
                  cwd="~/model-performance"
                />
                <SectionHead
                  label="test set metrics"
                  title={`${MODEL_META[active].title} — unseen test period`}
                  desc={`${MODEL_META[active].subtitle} · decision threshold ${model.decision_threshold}`}
                />

                <Panel>
                  <TermMetricTiles metrics={model.splits.test} />

                  <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
                    <div>
                      <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
                        confusion matrix (test set, threshold{" "}
                        {model.decision_threshold})
                      </p>
                      <TermConfusionMatrix metrics={model.splits.test} />
                    </div>

                    <div className="space-y-3">
                      <TermCalibrationNote model={model} />
                      <div className="crt-inset p-3">
                        <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
                          training configuration
                        </p>
                        <dl className="mt-2 space-y-1 text-[11px]">
                          {[
                            ["best_iteration", model.best_iteration ?? "—"],
                            [
                              "class_imbalance",
                              String(
                                (
                                  model.class_imbalance_handling as Record<
                                    string,
                                    unknown
                                  >
                                )?.method ?? "—"
                              ),
                            ],
                            [
                              "scale_pos_weight",
                              Number(
                                (
                                  model.class_imbalance_handling as Record<
                                    string,
                                    unknown
                                  >
                                )?.scale_pos_weight ?? NaN
                              ).toFixed(2),
                            ],
                            ["n_features", contract?.n_features ?? "—"],
                          ].map(([k, v]) => (
                            <div key={String(k)} className="flex justify-between">
                              <dt className="text-[#5f8d68]">{k}</dt>
                              <dd className="text-[#cfe9d5]">{String(v)}</dd>
                            </div>
                          ))}
                        </dl>
                      </div>
                    </div>
                  </div>
                </Panel>
              </section>

              {/* ------------------------------- chronological splits */}
              <section>
                <Prompt
                  command="cat split_metrics.tsv | column -t"
                  comment="train / validation / test"
                  cwd="~/model-performance"
                />
                <SectionHead
                  label="chronological splits"
                  title="Performance across train, validation and unseen test"
                  desc="Training always precedes validation, which precedes the unseen test period. No random shuffling."
                />
                <Panel>
                  <TermSplitPeriodTable model={model} />
                </Panel>
              </section>

              {/* --------------------------------- feature importance */}
              <section>
                <Prompt
                  command={`./importance --model=${MODEL_META[active].command} --sort desc`}
                  comment="gain-based, computed during training"
                  cwd="~/model-performance"
                />
                <SectionHead
                  label="feature importance"
                  title="What the model leaned on"
                  desc="Gain-based importance computed during training. Not recalculated in the frontend."
                />

                <Panel>
                  {feats.loading ? (
                    <TermLoading label="loading feature importance…" height={300} />
                  ) : feats.error ? (
                    <TermError
                      message={feats.error}
                      onRetry={feats.reload}
                      height={300}
                    />
                  ) : !importance?.importance?.length ? (
                    <TermEmpty
                      message="no feature importance available"
                      height={300}
                    />
                  ) : (
                    <>
                      <TermFeatureImportance
                        rows={importance.importance}
                        categoricalFeatures={importance.categorical_features ?? []}
                      />
                      {(importance.categorical_features?.length ?? 0) > 0 && (
                        <p className="mt-4 text-[10px] text-[#3d6b47]">
                          <span
                            className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                            style={{
                              background: TERM.amber,
                              boxShadow: `0 0 6px ${TERM.amber}`,
                            }}
                          />
                          categorical features (
                          {importance.categorical_features.join(", ")}) are handled
                          natively by XGBoost, not one-hot encoded.
                        </p>
                      )}
                      <p className="mt-2 text-[10px] leading-relaxed text-[#3d6b47]">
                        {isReal
                          ? "Static terrain features take one value per district (14 values), so their importance partly reflects district identity rather than physical process."
                          : "On synthetic data these rankings describe the data generator, not real flood or landslide physics."}
                      </p>
                    </>
                  )}
                </Panel>
              </section>
            </div>
          )}

          {/* ------------------------------------------------ footer */}
          <div className="crt-rule mt-10" />
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 text-[11px]">
            <p className="text-[#5f8d68]">
              <span className="text-[#39ff7a] crt-glow-soft">$</span> echo
              &quot;
              {isReal
                ? "real district-day data · research metrics, not operational skill"
                : "synthetic development data · rankings describe the generator"}
              &quot;{" "}
              <span className="crt-caret align-middle" />
            </p>
            <p className="text-[#1c7a3c]">
              active: {MODEL_META[active].command} · threshold{" "}
              {model?.decision_threshold ?? "—"}
            </p>
          </div>
        </div>
      </TerminalWindow>
    </>
  );
}
