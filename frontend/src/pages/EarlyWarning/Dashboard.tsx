/**
 * Main early-warning dashboard, rendered as a CRT terminal session.
 *
 * Same data and same features as before — summary cards -> Kerala map +
 * current warning panel -> monitoring-point selector -> trend charts ->
 * lead-time comparison — plus a separate SHAP explainability section. Only
 * the presentation changed: the page is its own terminal window rather than a
 * card layout inside AppLayout, so it carries its own nav.
 *
 * Every value still originates from the backend; nothing is computed here.
 */

import { useMemo, useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import KeralaMap from "../../components/KeralaMap/KeralaMap";
import {
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
  TermMapLegend,
  TermPointSelector,
  TermSummaryCards,
  TermWarningPanel,
} from "../../components/terminal/TermHazard";
import {
  TermEvaluationTable,
  TermLeadTimeChart,
  TermProbabilityChart,
  TermRainfallChart,
} from "../../components/terminal/TermCharts";
import TermShapSection from "../../components/terminal/TermShap";
import TermWeatherWidget from "../../components/terminal/TermWeatherWidget";
import { useApi } from "../../hooks/useApi";
import { emptyTrends, getCurrent, getEvaluation, getTrends } from "../../services/api";
import { toMapLocation } from "../../components/hazard/hazardUtils";
import type { MapLocation } from "../../types/hazard";

export default function EarlyWarningDashboard() {
  const current = useApi(() => getCurrent(), []);
  const evaluation = useApi(() => getEvaluation("any_hazard"), []);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Memoised so the derived lists below don't recompute on every render.
  const locations = useMemo(() => current.data?.locations ?? [], [current.data]);

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

  const statusColor = activeWarnings > 0 ? TERM.amber : TERM.phosphor;

  return (
    <>
      <PageMeta
        title="Kerala Flood–Landslide Early Warning | Dashboard"
        description="Coupled flood and landslide early warning dashboard with Kerala hazard map, TCDL warning status, lead-time analysis and SHAP model explainability."
      />

      <TerminalWindow
        path="~/early-warning"
        status={
          <span
            className="flex items-center gap-2 text-[13px]"
            style={{ color: statusColor }}
          >
            <span
              className="crt-blip h-2 w-2 rounded-full"
              style={{ background: statusColor, boxShadow: `0 0 8px ${statusColor}` }}
            />
            {activeWarnings} active warning{activeWarnings === 1 ? "" : "s"}
          </span>
        }
      >
        <div className="px-4 py-8 sm:px-8 sm:py-10">
          {/* ------------------------------------------------- header */}
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <Prompt
                command="./ews status --coupled"
                comment="two XGBoost models + TCDL V1.0"
              />
              <h1 className="mt-4 text-[26px] font-bold leading-tight tracking-tight text-[#eafff1] sm:text-[32px] [text-shadow:0_0_14px_rgba(57,255,122,0.35)]">
                Kerala Flood&ndash;Landslide Early Warning
              </h1>
              <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-[#5f8d68] sm:text-[13px]">
                Two independent XGBoost hazard models feeding a rule-based Temporal
                Coupled Decision Layer (TCDL V1.0)
                {current.data?.as_of ? ` · as of ${current.data.as_of}` : ""}
              </p>
            </div>

            {/* Live observed weather. The only real-world data on this page —
                display only, it never reaches the models or TCDL. */}
            <TermWeatherWidget />
          </div>

          <div className="mt-5">
            <TermSyntheticBanner
              dataMode={current.data?.data_mode}
              sourceMode={current.sourceMode}
              fallbackReason={current.fallbackReason}
            />
          </div>

          {/* ------------------------------------------ summary cards */}
          <div className="mt-6">
            {current.loading ? (
              <TermLoading label="reading current hazard status…" height={120} />
            ) : current.error ? (
              <TermError
                message={current.error}
                onRetry={current.reload}
                height={120}
              />
            ) : (
              <TermSummaryCards
                location={selected}
                activeWarnings={activeWarnings}
                totalLocations={locations.length}
              />
            )}
          </div>

          <Rule />

          {/* -------------------------------- map + warning decision */}
          <section>
            <Prompt
              command="./map --render kerala --markers monitoring_points"
              comment={`${mapLocations.length} points`}
            />
            <SectionHead
              label="live hazard map"
              title="Kerala Hazard Map // TCDL decision"
              desc="District boundaries from OpenStreetMap. Markers are synthetic monitoring points positioned by their coordinates — click one to inspect its status."
            />

            <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
              <div className="xl:col-span-2">
                <Panel right={<TermMapLegend />}>
                  {current.loading ? (
                    <TermLoading label="loading map data…" height={420} />
                  ) : current.error ? (
                    <TermError
                      message={current.error}
                      onRetry={current.reload}
                      height={420}
                    />
                  ) : mapLocations.length === 0 ? (
                    <TermEmpty
                      message="no monitoring points available"
                      height={420}
                    />
                  ) : (
                    <>
                      <div className="crt-map">
                        <KeralaMap
                          locations={mapLocations}
                          selectedId={activeId}
                          onSelect={(l) => setSelectedId(l.location_id)}
                          height={460}
                          showLegend={false}
                        />
                      </div>
                      <p className="mt-3 text-[10px] leading-relaxed text-[#3d6b47]">
                        The synthetic dataset carries no district field, so markers show
                        coordinates only — no district is inferred from position. Because
                        these coordinates are simulated rather than surveyed Kerala sites,
                        some points fall outside the state boundary.
                      </p>
                    </>
                  )}
                </Panel>
              </div>

              <div className="xl:col-span-1">
                {current.loading ? (
                  <TermLoading label="loading warning status…" height={420} />
                ) : (
                  <TermWarningPanel
                    location={selected}
                    ruleDescriptions={evaluation.data?.rules}
                  />
                )}
              </div>
            </div>

            {/* ---------------------------- monitoring point selector */}
            {locations.length > 0 && (
              <div className="mt-5">
                <Panel
                  title="ls ~/monitoring_points"
                  desc="Select a point to drive the warning panel, the trends below and the SHAP explanation."
                >
                  <TermPointSelector
                    locations={locations}
                    activeId={activeId}
                    onSelect={setSelectedId}
                  />
                </Panel>
              </div>
            )}
          </section>

          <Rule />

          {/* ------------------------------------------------ trends */}
          <section>
            <Prompt
              command="cat trends.log | tail -120"
              comment={activeId ? `location ${activeId}` : "no location selected"}
            />
            <SectionHead
              label="temporal signals"
              title="What the models and TCDL are watching"
              desc="Hazard probabilities and the rainfall signal over the evaluation period, exactly as served by /trends."
            />

            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <Panel
                title="hazard probability trend"
                desc="Flood, landslide and coupled probability over the evaluation period"
              >
                {trends.loading ? (
                  <TermLoading label="loading trends…" height={280} />
                ) : trends.error ? (
                  <TermError
                    message={trends.error}
                    onRetry={trends.reload}
                    height={280}
                  />
                ) : !trends.data?.series.length ? (
                  <TermEmpty
                    message="no trend data for this location"
                    height={280}
                  />
                ) : (
                  <TermProbabilityChart
                    key={`prob-${activeId}`}
                    series={trends.data.series}
                  />
                )}
              </Panel>

              <Panel
                title="rainfall signal"
                desc="3-day moving average and its rate of increase, as consumed by TCDL"
              >
                {trends.loading ? (
                  <TermLoading label="loading rainfall…" height={280} />
                ) : trends.error ? (
                  <TermError
                    message={trends.error}
                    onRetry={trends.reload}
                    height={280}
                  />
                ) : !trends.data?.series.length ? (
                  <TermEmpty
                    message="no rainfall data for this location"
                    height={280}
                  />
                ) : (
                  <TermRainfallChart
                    key={`rain-${activeId}`}
                    series={trends.data.series}
                  />
                )}
              </Panel>
            </div>
          </section>

          <Rule />

          {/* --------------------------------------------- lead time */}
          <section>
            <Prompt
              command="./evaluate --lead-time --systems all"
              comment="primary research objective"
            />
            <SectionHead
              label="lead-time comparison"
              title="Flood-only vs landslide-only vs TCDL coupled"
              desc="Time-in-warning and false-alarm rate sit beside every lead time, and the always-warn control stays visible: lead time alone can always be improved by warning more often."
            />

            <Panel>
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
                  <TermLeadTimeChart evaluation={evaluation.data} />
                  <TermEvaluationTable evaluation={evaluation.data} />
                </div>
              ) : (
                <TermEmpty message="no evaluation results available" height={300} />
              )}
            </Panel>
          </section>

          <Rule />

          {/* -------------------------------------------------- shap */}
          <TermShapSection locationId={activeId} />

          {/* ------------------------------------------------ footer */}
          <div className="crt-rule mt-10" />
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 text-[11px]">
            <p className="text-[#5f8d68]">
              <span className="text-[#39ff7a] crt-glow-soft">$</span> echo
              &quot;synthetic development data · not a real-world Kerala
              warning&quot; <span className="crt-caret align-middle" />
            </p>
            <p className="text-[#1c7a3c]">
              models: flood_xgboost@v1.0 landslide_xgboost@v1.0 · tcdl@v1.0 ·
              points: {locations.length}
            </p>
          </div>
        </div>
      </TerminalWindow>
    </>
  );
}
