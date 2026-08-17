/**
 * Detailed analysis page: full temporal signal set for a chosen monitoring
 * point, plus the lead-time comparison across hazard scopes.
 * Location is selected by coordinates — no district is fabricated.
 */

import { useMemo, useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import PageBreadcrumb from "../../components/common/PageBreadCrumb";
import ComponentCard from "../../components/common/ComponentCard";
import {
  ProbabilityTrendChart,
  RainfallChart,
  RateOfIncreaseChart,
  SmoothedProbabilityChart,
  SoilMoistureChart,
} from "../../components/hazard/TrendCharts";
import {
  EvaluationNotes,
  EvaluationTable,
  LeadTimeBarChart,
} from "../../components/hazard/LeadTimeComparison";
import {
  EmptyBlock,
  ErrorBlock,
  LoadingBlock,
  SyntheticDataBanner,
} from "../../components/hazard/StateBlocks";
import { num, pct } from "../../components/hazard/hazardUtils";
import { useApi } from "../../hooks/useApi";
import { emptyTrends, getCurrent, getEvaluation, getTrends } from "../../services/api";
import type { HazardScope } from "../../services/api";

export default function AnalysisPage() {
  const current = useApi(() => getCurrent(), []);
  const locations = current.data?.locations ?? [];
  const [locId, setLocId] = useState<string | null>(null);
  const [scope, setScope] = useState<HazardScope>("any_hazard");

  const activeId = locId ?? locations[0]?.location_id ?? null;
  const trends = useApi(
    () => (activeId ? getTrends(activeId, 200) : emptyTrends()),
    [activeId]
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

  const selectCls =
    "rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-700 focus:border-brand-500 focus:outline-none dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200";

  // Charts are keyed by location so a location change remounts them cleanly
  // rather than updating in place, which races ApexCharts' async renderer.
  const chartBlock = (
    title: string,
    desc: string,
    node: React.ReactNode
  ) => (
    <ComponentCard title={title} desc={desc} key={`${title}-${activeId}`}>
      {trends.loading ? (
        <LoadingBlock label="Loading signals…" height={280} />
      ) : trends.error ? (
        <ErrorBlock message={trends.error} onRetry={trends.reload} height={280} />
      ) : !trends.data?.series.length ? (
        <EmptyBlock message="No data for this location." height={280} />
      ) : (
        node
      )}
    </ComponentCard>
  );

  return (
    <>
      <PageMeta
        title="Analysis | Kerala Flood–Landslide Early Warning"
        description="Detailed temporal analysis of hazard probabilities, environmental signals and lead-time performance."
      />
      <PageBreadcrumb pageTitle="Analysis" />
      <SyntheticDataBanner
        dataMode={current.data?.data_mode}
        sourceMode={current.sourceMode}
        fallbackReason={current.fallbackReason}
      />

      {/* Location selector by coordinates */}
      <div className="mb-5 flex flex-wrap items-end gap-3 rounded-2xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-white/[0.03]">
        <label className="flex flex-col gap-1">
          <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
            Monitoring point (latitude, longitude)
          </span>
          <select
            className={selectCls}
            value={activeId ?? ""}
            onChange={(e) => setLocId(e.target.value)}
          >
            {locations.map((l) => (
              <option key={l.location_id} value={l.location_id}>
                {l.environment.latitude.toFixed(3)}, {l.environment.longitude.toFixed(3)}
                {l.district ? ` — ${l.district}` : ""}
              </option>
            ))}
          </select>
        </label>
        {selected && (
          <div className="flex flex-wrap gap-4 text-xs">
            <span className="text-gray-500 dark:text-gray-400">
              Flood{" "}
              <span className="font-semibold text-gray-800 dark:text-white/90">
                {pct(selected.flood.probability, 1)}
              </span>
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              Landslide{" "}
              <span className="font-semibold text-gray-800 dark:text-white/90">
                {pct(selected.landslide.probability, 1)}
              </span>
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              TCDL{" "}
              <span className="font-semibold text-gray-800 dark:text-white/90">
                {selected.tcdl.warning_type}
              </span>
            </span>
          </div>
        )}
      </div>

      {/* ML probability signals */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {chartBlock(
          "Hazard Probabilities Over Time",
          "Raw model output: flood, landslide and coupled probability",
          trends.data ? <ProbabilityTrendChart series={trends.data.series} /> : null
        )}
        {chartBlock(
          "Smoothed Probabilities (TCDL input)",
          "3-day moving averages actually evaluated by the TCDL rules",
          trends.data ? <SmoothedProbabilityChart series={trends.data.series} /> : null
        )}
        {chartBlock(
          "Rate of Increase",
          "3-day change in each smoothed signal — the trend evidence TCDL uses",
          trends.data ? <RateOfIncreaseChart series={trends.data.series} /> : null
        )}
        {chartBlock(
          "Rainfall",
          "3-day moving average and rate of increase",
          trends.data ? <RainfallChart series={trends.data.series} /> : null
        )}
        {chartBlock(
          "Soil Moisture",
          "3-day moving average of root-zone soil moisture",
          trends.data ? <SoilMoistureChart series={trends.data.series} /> : null
        )}

        {/* Current environmental conditions */}
        <ComponentCard
          title="Environmental Conditions"
          desc={env ? `Model inputs at ${env.date}` : "Model inputs"}
        >
          {current.loading ? (
            <LoadingBlock label="Loading conditions…" height={280} />
          ) : !env ? (
            <EmptyBlock message="No environmental data available." height={280} />
          ) : (
            <>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-3">
                {envRows.map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-baseline justify-between border-b border-gray-50 py-1 dark:border-gray-800"
                  >
                    <span className="truncate text-[11px] text-gray-500 dark:text-gray-400">
                      {k}
                    </span>
                    <span className="ml-2 shrink-0 text-xs font-medium text-gray-800 dark:text-white/90">
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
                <p className="mt-3 text-[11px] text-gray-500 dark:text-gray-400">
                  Unavailable in this dataset: {env.unavailable_fields.join(", ")} —
                  shown as “—” rather than substituted.
                </p>
              )}
            </>
          )}
        </ComponentCard>
      </div>

      {/* Lead-time and system comparison */}
      <div className="mt-5">
        <ComponentCard
          title="Lead-Time & System Comparison"
          desc="Evaluation results produced by the TCDL pipeline — not recalculated here"
        >
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
                Event scope
              </span>
              <select
                className={selectCls}
                value={scope}
                onChange={(e) => setScope(e.target.value as HazardScope)}
              >
                <option value="any_hazard">Any hazard onsets</option>
                <option value="flood">Flood onsets</option>
                <option value="landslide">Landslide onsets</option>
              </select>
            </label>
            {evaluation.data && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Evaluation period{" "}
                {evaluation.data.evaluation_period.date_range.join(" → ")} ·{" "}
                {evaluation.data.evaluation_period.n_rows} rows
              </span>
            )}
          </div>

          {evaluation.loading ? (
            <LoadingBlock label="Loading evaluation…" height={300} />
          ) : evaluation.error ? (
            <ErrorBlock
              message={evaluation.error}
              onRetry={evaluation.reload}
              height={300}
            />
          ) : evaluation.data ? (
            <div className="space-y-5">
              <LeadTimeBarChart evaluation={evaluation.data} />
              <EvaluationTable evaluation={evaluation.data} />
              <EvaluationNotes evaluation={evaluation.data} />
            </div>
          ) : (
            <EmptyBlock message="No evaluation results." height={300} />
          )}
        </ComponentCard>
      </div>

      {/* TCDL rule firing */}
      {evaluation.data && (
        <div className="mt-5">
          <ComponentCard
            title="TCDL Rule Activity"
            desc="How often each rule fired across the evaluation period"
          >
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {Object.entries(evaluation.data.rule_firing_counts).map(([rule, n]) => {
                const isBaseline =
                  evaluation.data!.rule_classification.baseline_replicating.includes(
                    rule
                  );
                const total = evaluation.data!.evaluation_period.n_rows || 1;
                return (
                  <div
                    key={rule}
                    className="rounded-lg border border-gray-100 p-3 dark:border-gray-800"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[11px] font-semibold text-gray-700 dark:text-gray-300">
                        {rule}
                      </span>
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] ${
                          isBaseline
                            ? "bg-gray-100 text-gray-600 dark:bg-white/5 dark:text-gray-400"
                            : "bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400"
                        }`}
                      >
                        {isBaseline ? "baseline" : "coupling"}
                      </span>
                    </div>
                    <p className="mt-1.5 text-[11px] leading-snug text-gray-500 dark:text-gray-400">
                      {evaluation.data!.rules[rule]}
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-gray-100 dark:bg-white/5">
                        <div
                          className="h-full rounded-full bg-brand-500"
                          style={{ width: `${Math.min(100, (n / total) * 100)}%` }}
                        />
                      </div>
                      <span className="shrink-0 text-[11px] font-medium text-gray-700 dark:text-gray-300">
                        {n} days ({((n / total) * 100).toFixed(1)}%)
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </ComponentCard>
        </div>
      )}
    </>
  );
}
