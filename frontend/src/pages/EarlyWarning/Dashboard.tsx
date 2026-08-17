/**
 * Main early-warning dashboard.
 *
 * Layout: summary cards -> Kerala map + current warning panel -> trend
 * charts -> lead-time comparison. Every value originates from the backend.
 */

import { useMemo, useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import ComponentCard from "../../components/common/ComponentCard";
import KeralaMap from "../../components/KeralaMap/KeralaMap";
import HazardSummaryCards from "../../components/hazard/HazardSummaryCards";
import CurrentWarningPanel from "../../components/hazard/CurrentWarningPanel";
import {
  ProbabilityTrendChart,
  RainfallChart,
} from "../../components/hazard/TrendCharts";
import {
  LeadTimeBarChart,
  EvaluationTable,
} from "../../components/hazard/LeadTimeComparison";
import {
  EmptyBlock,
  ErrorBlock,
  LoadingBlock,
  SyntheticDataBanner,
} from "../../components/hazard/StateBlocks";
import { toMapLocation } from "../../components/hazard/hazardUtils";
import { useApi } from "../../hooks/useApi";
import { emptyTrends, getCurrent, getEvaluation, getTrends } from "../../services/api";
import type { MapLocation } from "../../types/hazard";

export default function EarlyWarningDashboard() {
  const current = useApi(() => getCurrent(), []);
  const evaluation = useApi(() => getEvaluation("any_hazard"), []);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const locations = current.data?.locations ?? [];

  // Default to the highest coupled risk so the dashboard opens on the point
  // that most needs attention.
  const activeId = useMemo(() => {
    if (selectedId) return selectedId;
    if (!locations.length) return null;
    return [...locations].sort(
      (a, b) =>
        (b.trends.coupled_probability ?? 0) - (a.trends.coupled_probability ?? 0)
    )[0].location_id;
  }, [selectedId, locations]);

  const selected = locations.find((l) => l.location_id === activeId) ?? null;
  const trends = useApi(
    () => (activeId ? getTrends(activeId, 120) : emptyTrends()),
    [activeId]
  );

  const mapLocations: MapLocation[] = useMemo(
    () => locations.map(toMapLocation),
    [locations]
  );
  const activeWarnings = locations.filter(
    (l) => l.tcdl.warning_status === "Warning"
  ).length;

  return (
    <>
      <PageMeta
        title="Kerala Flood–Landslide Early Warning | Dashboard"
        description="Coupled flood and landslide early warning dashboard with Kerala hazard map, TCDL warning status and lead-time analysis."
      />

      <div className="mb-5">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white/90">
          Kerala Flood–Landslide Early Warning
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Two independent XGBoost hazard models feeding a rule-based Temporal
          Coupled Decision Layer (TCDL V1.0)
          {current.data?.as_of ? ` · as of ${current.data.as_of}` : ""}
        </p>
      </div>

      <SyntheticDataBanner
        dataMode={current.data?.data_mode}
        sourceMode={current.sourceMode}
        fallbackReason={current.fallbackReason}
      />

      {/* ---- Summary cards ---- */}
      {current.loading ? (
        <LoadingBlock label="Loading current hazard status…" height={120} />
      ) : current.error ? (
        <ErrorBlock message={current.error} onRetry={current.reload} height={120} />
      ) : (
        <HazardSummaryCards
          location={selected}
          activeWarnings={activeWarnings}
          totalLocations={locations.length}
        />
      )}

      {/* ---- Map + warning panel ---- */}
      <div className="mt-5 grid grid-cols-1 gap-5 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <ComponentCard
            title="Kerala Hazard Map"
            desc="District boundaries from OpenStreetMap. Markers are synthetic monitoring points positioned by their coordinates — click one to inspect its status."
          >
            {current.loading ? (
              <LoadingBlock label="Loading map data…" height={420} />
            ) : current.error ? (
              <ErrorBlock message={current.error} onRetry={current.reload} height={420} />
            ) : mapLocations.length === 0 ? (
              <EmptyBlock message="No monitoring points available." height={420} />
            ) : (
              <>
                <KeralaMap
                  locations={mapLocations}
                  selectedId={activeId}
                  onSelect={(l) => setSelectedId(l.location_id)}
                  height={460}
                />
                <p className="mt-3 text-[11px] leading-relaxed text-gray-500 dark:text-gray-400">
                  The synthetic dataset carries no district field, so markers show
                  coordinates only — no district is inferred from position. Because
                  these coordinates are simulated rather than surveyed Kerala sites,
                  some points fall outside the state boundary.
                </p>
              </>
            )}
          </ComponentCard>
        </div>

        <div className="xl:col-span-1">
          {current.loading ? (
            <LoadingBlock label="Loading warning status…" height={420} />
          ) : (
            <CurrentWarningPanel
              location={selected}
              ruleDescriptions={evaluation.data?.rules}
            />
          )}
        </div>
      </div>

      {/* ---- Monitoring point selector ---- */}
      {locations.length > 0 && (
        <div className="mt-5 rounded-2xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-white/[0.03]">
          <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
            Monitoring points
          </p>
          <div className="flex flex-wrap gap-2">
            {locations.map((l) => {
              const on = l.location_id === activeId;
              const warn = l.tcdl.warning_status === "Warning";
              return (
                <button
                  key={l.location_id}
                  onClick={() => setSelectedId(l.location_id)}
                  className={`rounded-lg border px-3 py-1.5 text-xs transition ${
                    on
                      ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-400"
                      : "border-gray-200 text-gray-600 hover:border-gray-300 dark:border-gray-700 dark:text-gray-300"
                  }`}
                >
                  <span className="font-mono">
                    {l.environment.latitude.toFixed(3)},{" "}
                    {l.environment.longitude.toFixed(3)}
                  </span>
                  {warn && (
                    <span className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-error-500 align-middle" />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ---- Trends ---- */}
      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ComponentCard
          title="Hazard Probability Trend"
          desc="Flood, landslide and coupled probability over the evaluation period"
        >
          {trends.loading ? (
            <LoadingBlock label="Loading trends…" height={280} />
          ) : trends.error ? (
            <ErrorBlock message={trends.error} onRetry={trends.reload} height={280} />
          ) : !trends.data?.series.length ? (
            <EmptyBlock message="No trend data for this location." height={280} />
          ) : (
            <ProbabilityTrendChart key={`prob-${activeId}`} series={trends.data.series} />
          )}
        </ComponentCard>

        <ComponentCard
          title="Rainfall Signal"
          desc="3-day moving average and its rate of increase, as consumed by TCDL"
        >
          {trends.loading ? (
            <LoadingBlock label="Loading rainfall…" height={280} />
          ) : trends.error ? (
            <ErrorBlock message={trends.error} onRetry={trends.reload} height={280} />
          ) : !trends.data?.series.length ? (
            <EmptyBlock message="No rainfall data for this location." height={280} />
          ) : (
            <RainfallChart key={`rain-${activeId}`} series={trends.data.series} />
          )}
        </ComponentCard>
      </div>

      {/* ---- Lead time ---- */}
      <div className="mt-5">
        <ComponentCard
          title="Lead-Time Comparison"
          desc="Flood-only vs landslide-only vs TCDL coupled — the project's primary research objective"
        >
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
            </div>
          ) : (
            <EmptyBlock message="No evaluation results available." height={300} />
          )}
        </ComponentCard>
      </div>
    </>
  );
}
