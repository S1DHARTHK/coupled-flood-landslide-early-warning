/**
 * Detailed analysis page, rendered as a CRT terminal session: the full
 * temporal signal set for a chosen location (a district on real data), the
 * model inputs for the day shown, the lead-time comparison across hazard
 * scopes, TCDL rule activity, and the SHAP explainability section.
 *
 * District names come from the backend and are never inferred from
 * coordinates. Every number is served by the backend; nothing is recomputed.
 */

import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import PageMeta from "../../components/common/PageMeta";
import {
  Field,
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
  TermEvaluationNotes,
  TermEvaluationTable,
  TermLeadTimeChart,
  TermProbabilityChart,
  TermRainfallChart,
  TermRateOfIncreaseChart,
  TermSmoothedProbabilityChart,
  TermSoilMoistureChart,
} from "../../components/terminal/TermCharts";
import TermShapSection from "../../components/terminal/TermShap";
import { TermAsOfPicker } from "../../components/terminal/TermHazard";
import { num, pct } from "../../components/hazard/hazardUtils";
import { useApi } from "../../hooks/useApi";
import { emptyTrends, getCurrent, getEvaluation, getTrends } from "../../services/api";
import type { HazardScope } from "../../services/api";

export default function AnalysisPage() {
  // null = the latest day in the pipeline outputs.
  const [asOf, setAsOf] = useState<string | null>(null);
  const current = useApi(() => getCurrent(undefined, asOf ?? undefined), [asOf]);
  const locations = useMemo(() => current.data?.locations ?? [], [current.data]);
  const [locId, setLocId] = useState<string | null>(null);
  const [scope, setScope] = useState<HazardScope>("any_hazard");
  // District wording unless the backend says it is serving synthetic points.
  const isReal = current.data?.data_mode !== "synthetic";

  const activeId = locId ?? locations[0]?.location_id ?? null;
  const trends = useApi(
    () => (activeId ? getTrends(activeId, 200, asOf ?? undefined) : emptyTrends()),
    [activeId, asOf]
  );
  const evaluation = useApi(() => getEvaluation(scope), [scope]);

  const selected = locations.find((l) => l.location_id === activeId) ?? null;
  const env = selected?.environment;

  const envRows = useMemo(() => {
    if (!env) return [];
    return [
      ...Object.entries(env.common_environmental),
      ...Object.entries(env.flood_specific),
      ...Object.entries(env.landslide_specific),
    ];
  }, [env]);

  // Charts are keyed by location so a location change remounts them cleanly
  // rather than updating in place, which races ApexCharts' async renderer.
  const chartBlock = (title: string, desc: string, node: ReactNode) => (
    <Panel title={title} desc={desc} key={`${title}-${activeId}`}>
      {trends.loading ? (
        <TermLoading label="loading signals…" height={280} />
      ) : trends.error ? (
        <TermError message={trends.error} onRetry={trends.reload} height={280} />
      ) : !trends.data?.series.length ? (
        <TermEmpty message="no data for this location" height={280} />
      ) : (
        node
      )}
    </Panel>
  );

  return (
    <>
      <PageMeta
        title="Analysis | Kerala Flood–Landslide Early Warning"
        description="Detailed temporal analysis of hazard probabilities, environmental signals and lead-time performance."
      />

      <TerminalWindow
        path="~/analysis"
        status={
          <span className="flex items-center gap-2 text-[13px] text-[#5f8d68]">
            <span className="text-[#39ff7a]">{locations.length}</span>{" "}
            {isReal ? "districts" : "points"}
          </span>
        }
      >
        <div className="px-4 py-8 sm:px-8 sm:py-10">
          <Prompt
            command="./analyse --signals --lead-time"
            comment="full temporal signal set"
            cwd="~/analysis"
          />
          <h1 className="mt-4 text-[26px] font-bold leading-tight tracking-tight text-[#eafff1] sm:text-[30px] [text-shadow:0_0_14px_rgba(57,255,122,0.35)]">
            Analysis
          </h1>
          <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-[#5f8d68] sm:text-[13px]">
            Every signal the two XGBoost models and the TCDL rules consume, for one
            {isReal ? " district" : " monitoring point"} at a time, plus the evaluation
            the pipeline produced.
          </p>

          <div className="mt-5">
            <TermSyntheticBanner
              dataMode={current.data?.data_mode}
              sourceMode={current.sourceMode}
              fallbackReason={current.fallbackReason}
              notice={current.data?.data_notice}
            />
          </div>

          <div className="mt-4">
            <TermAsOfPicker
              value={asOf}
              range={current.data?.available_date_range}
              onChange={setAsOf}
            />
          </div>

          {/* --------------------------------------- location selector */}
          <div className="mt-6">
            <Panel>
              <div className="flex flex-wrap items-end gap-4">
                <Field label={isReal ? "district" : "monitoring point (latitude, longitude)"}>
                  <select
                    className="crt-select"
                    value={activeId ?? ""}
                    onChange={(e) => setLocId(e.target.value)}
                  >
                    {locations.map((l) => (
                      <option key={l.location_id} value={l.location_id}>
                        {l.district
                          ? `${l.district} — ${l.environment.latitude.toFixed(3)}, ${l.environment.longitude.toFixed(3)}`
                          : `${l.environment.latitude.toFixed(3)}, ${l.environment.longitude.toFixed(3)}`}
                      </option>
                    ))}
                  </select>
                </Field>

                {selected && (
                  <div className="flex flex-wrap gap-5 pb-1 text-[11px] text-[#3d6b47]">
                    <span>
                      flood{" "}
                      <span style={{ color: TERM.dim }}>
                        {pct(selected.flood.probability, 1)}
                      </span>
                    </span>
                    <span>
                      landslide{" "}
                      <span style={{ color: TERM.amber }}>
                        {pct(selected.landslide.probability, 1)}
                      </span>
                    </span>
                    <span>
                      tcdl{" "}
                      <span className="text-[#eafff1]">
                        {selected.tcdl.warning_type}
                      </span>
                    </span>
                  </div>
                )}
              </div>
            </Panel>
          </div>

          <Rule />

          {/* -------------------------------------------- ML signals */}
          <section>
            <Prompt
              command="cat signals.log | tail -200"
              comment={
                selected
                  ? `${selected.district ?? selected.location_id} · 200 days to ${current.data?.as_of ?? "latest"}`
                  : "no location"
              }
              cwd="~/analysis"
            />
            <SectionHead
              label="temporal signals"
              title="Raw output, smoothed input, and rate of change"
              desc="Nothing is smoothed or differenced here: the moving averages and rates are produced by TCDL V1.0 and merely plotted."
            />

            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              {chartBlock(
                "hazard probabilities over time",
                "Raw model output: flood, landslide and coupled probability",
                trends.data ? (
                  <TermProbabilityChart series={trends.data.series} />
                ) : null
              )}
              {chartBlock(
                "smoothed probabilities (TCDL input)",
                "3-day moving averages actually evaluated by the TCDL rules",
                trends.data ? (
                  <TermSmoothedProbabilityChart series={trends.data.series} />
                ) : null
              )}
              {chartBlock(
                "rate of increase",
                "3-day change in each smoothed signal — the trend evidence TCDL uses",
                trends.data ? (
                  <TermRateOfIncreaseChart series={trends.data.series} />
                ) : null
              )}
              {chartBlock(
                "rainfall",
                "3-day moving average and rate of increase",
                trends.data ? <TermRainfallChart series={trends.data.series} /> : null
              )}
              {chartBlock(
                "soil moisture",
                "3-day moving average of root-zone soil moisture",
                trends.data ? (
                  <TermSoilMoistureChart series={trends.data.series} />
                ) : null
              )}

              {/* Current model inputs */}
              <Panel
                title="environmental conditions"
                desc={env ? `Model inputs at ${env.date}` : "Model inputs"}
              >
                {current.loading ? (
                  <TermLoading label="loading conditions…" height={280} />
                ) : !env ? (
                  <TermEmpty
                    message="no environmental data available"
                    height={280}
                  />
                ) : (
                  <>
                    <div className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
                      {envRows.map(([k, v]) => (
                        <div
                          key={k}
                          className="flex items-baseline justify-between border-b border-[#0f2a12] py-1"
                        >
                          <span className="truncate text-[11px] text-[#5f8d68]">
                            {k}
                          </span>
                          <span className="ml-2 shrink-0 text-[11px] text-[#cfe9d5]">
                            {v === null
                              ? "—"
                              : typeof v === "number"
                              ? num(v, 2)
                              : String(v)}
                          </span>
                        </div>
                      ))}
                    </div>
                    {env.unavailable_fields.length > 0 && (
                      <p className="mt-3 text-[10px] text-[#3d6b47]">
                        no value for this day:{" "}
                        {env.unavailable_fields.join(", ")} — shown as “—” rather than
                        substituted.
                      </p>
                    )}
                  </>
                )}
              </Panel>
            </div>
          </section>

          <Rule />

          {/* ------------------------------- lead time / system compare */}
          <section>
            <Prompt
              command={`./evaluate --scope=${scope}`}
              comment="produced by the TCDL pipeline"
              cwd="~/analysis"
            />
            <SectionHead
              label="lead-time & system comparison"
              title="How much earlier does coupling warn?"
              desc="Evaluation results served by the backend — not recalculated here."
            />

            <Panel>
              <div className="flex flex-wrap items-end gap-4">
                <Field label="event scope">
                  <select
                    className="crt-select"
                    value={scope}
                    onChange={(e) => setScope(e.target.value as HazardScope)}
                  >
                    <option value="any_hazard">any hazard onsets</option>
                    <option value="flood">flood onsets</option>
                    <option value="landslide">landslide onsets</option>
                  </select>
                </Field>
                {evaluation.data && (
                  <span className="pb-1 text-[11px] text-[#3d6b47]">
                    evaluation period{" "}
                    <span className="text-[#5f8d68]">
                      {evaluation.data.evaluation_period.date_range.join(" → ")}
                    </span>{" "}
                    · {evaluation.data.evaluation_period.n_rows} rows
                  </span>
                )}
              </div>

              <div className="mt-4">
                {evaluation.loading ? (
                  <TermLoading label="loading evaluation…" height={300} />
                ) : evaluation.error ? (
                  <TermError
                    message={evaluation.error}
                    onRetry={evaluation.reload}
                    height={300}
                  />
                ) : evaluation.data ? (
                  <div className="space-y-5">
                    <TermLeadTimeChart
                      key={`lead-${scope}`}
                      evaluation={evaluation.data}
                    />
                    <TermEvaluationTable evaluation={evaluation.data} />
                    <TermEvaluationNotes evaluation={evaluation.data} />
                  </div>
                ) : (
                  <TermEmpty message="no evaluation results" height={300} />
                )}
              </div>
            </Panel>
          </section>

          {/* ------------------------------------------ rule activity */}
          {evaluation.data && (
            <>
              <Rule />
              <section>
                <Prompt
                  command="grep -c 'RULE_FIRED' tcdl.log"
                  comment="rule firing counts"
                  cwd="~/analysis"
                />
                <SectionHead
                  label="tcdl rule activity"
                  title="How often each rule fired"
                  desc="R1/R2 replicate the single-model baselines; R3–R7 are the coupling rules."
                />

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {Object.entries(evaluation.data.rule_firing_counts).map(
                    ([rule, n]) => {
                      const isBaseline =
                        evaluation.data!.rule_classification.baseline_replicating.includes(
                          rule
                        );
                      const total = evaluation.data!.evaluation_period.n_rows || 1;
                      const color = isBaseline ? TERM.faint : TERM.phosphor;
                      return (
                        <div key={rule} className="crt-panel p-3">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-[11px] text-[#cfe9d5]">{rule}</span>
                            <span
                              className="rounded-[2px] px-1.5 py-0.5 text-[10px]"
                              style={{ color, border: `1px solid ${color}55` }}
                            >
                              {isBaseline ? "baseline" : "coupling"}
                            </span>
                          </div>
                          <p className="mt-1.5 text-[11px] leading-snug text-[#5f8d68]">
                            {evaluation.data!.rules[rule]}
                          </p>
                          <div className="mt-2 flex items-center gap-2">
                            <div className="crt-meter flex-1">
                              <div
                                className="crt-meter-fill"
                                style={{
                                  width: `${Math.min(100, (n / total) * 100)}%`,
                                  background: `linear-gradient(90deg, ${color}55, ${color})`,
                                  boxShadow: `0 0 10px ${color}70`,
                                }}
                              />
                            </div>
                            <span className="shrink-0 text-[10px] tabular-nums text-[#3d6b47]">
                              {n} days ({((n / total) * 100).toFixed(1)}%)
                            </span>
                          </div>
                        </div>
                      );
                    }
                  )}
                </div>
              </section>
            </>
          )}

          <Rule />

          {/* -------------------------------------------------- shap */}
          <TermShapSection locationId={activeId} date={asOf} />

          {/* ------------------------------------------------ footer */}
          <div className="crt-rule mt-10" />
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 text-[11px]">
            <p className="text-[#5f8d68]">
              <span className="text-[#39ff7a] crt-glow-soft">$</span> echo
              &quot;
              {isReal
                ? "real historical data · research output, not an operational warning"
                : "synthetic development data · not a real-world Kerala warning"}
              &quot; <span className="crt-caret align-middle" />
            </p>
            <p className="text-[#1c7a3c]">
              scope: {scope} · {isReal ? "district" : "point"}:{" "}
              {selected?.district ?? activeId ?? "—"}
            </p>
          </div>
        </div>
      </TerminalWindow>
    </>
  );
}
