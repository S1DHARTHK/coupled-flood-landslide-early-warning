/**
 * Terminal-themed charts and tables for the early-warning dashboard.
 *
 * Same ApexCharts library and the same backend series as the light-theme
 * components in `components/hazard` — only the palette, grid and typography
 * change. Nothing is smoothed, differenced or computed here: the moving
 * averages, rates and lead times all arrive from the ML/TCDL layer.
 */

import Chart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";
import type { EvaluationResponse, TrendPoint } from "../../types/hazard";
import { SYSTEM_LABELS, leadDays, pct } from "../hazard/hazardUtils";
import { TERM } from "./termColors";

const MONO = '"JetBrains Mono", ui-monospace, monospace';
const GRID = "#1e2e14";
const AXIS = "#6a8958";

/** Shared console styling for every chart on the page. */
function baseOptions(
  categories: string[],
  yTitle: string,
  opts: { percent?: boolean } = {}
): ApexOptions {
  return {
    chart: {
      type: "line",
      fontFamily: MONO,
      background: "transparent",
      foreColor: TERM.sage,
      toolbar: { show: false },
      animations: { enabled: false },
      zoom: { enabled: false },
    },
    theme: { mode: "dark" },
    stroke: { curve: "smooth", width: 2 },
    dataLabels: { enabled: false },
    legend: {
      show: true,
      position: "top",
      horizontalAlign: "left",
      fontSize: "11px",
      fontFamily: MONO,
      labels: { colors: TERM.sage },
      markers: { size: 5 },
    },
    grid: {
      borderColor: GRID,
      strokeDashArray: 3,
      xaxis: { lines: { show: false } },
      yaxis: { lines: { show: true } },
    },
    xaxis: {
      categories,
      tickAmount: 8,
      labels: { style: { fontSize: "10px", colors: AXIS, fontFamily: MONO }, rotate: 0 },
      axisBorder: { show: false },
      axisTicks: { show: false },
      tooltip: { enabled: false },
    },
    yaxis: {
      title: {
        text: yTitle,
        style: { fontSize: "10px", color: AXIS, fontWeight: 500, fontFamily: MONO },
      },
      labels: {
        style: { fontSize: "10px", colors: AXIS, fontFamily: MONO },
        formatter: (v: number) =>
          opts.percent ? `${Math.round(v * 100)}%` : v.toFixed(2),
      },
    },
    tooltip: {
      theme: "dark",
      shared: true,
      intersect: false,
      style: { fontFamily: MONO, fontSize: "11px" },
      y: {
        formatter: (v: number) =>
          v === null || v === undefined
            ? "no data"
            : opts.percent
            ? `${(v * 100).toFixed(1)}%`
            : v.toFixed(3),
      },
    },
  };
}

interface ChartProps {
  series: TrendPoint[];
  height?: number;
}

/** Flood / landslide / coupled probability over the evaluation period. */
export function TermProbabilityChart({ series, height = 300 }: ChartProps) {
  const defs = [
    { key: "flood_probability", name: "flood", color: TERM.dim },
    { key: "landslide_probability", name: "landslide", color: TERM.amber },
    { key: "coupled_probability", name: "coupled", color: TERM.phosphor },
  ] as const;

  return (
    <Chart
      options={baseOptions(
        series.map((p) => p.date),
        "probability",
        { percent: true }
      )}
      series={defs.map((d) => ({
        name: d.name,
        color: d.color,
        data: series.map((p) => {
          const v = p[d.key];
          return typeof v === "number" ? v : null;
        }),
      }))}
      type="line"
      height={height}
    />
  );
}

/** Rainfall 3-day moving average plus its rate of increase. */
export function TermRainfallChart({ series, height = 300 }: ChartProps) {
  const options: ApexOptions = {
    ...baseOptions(
      series.map((p) => p.date),
      "mm / day"
    ),
    stroke: { curve: "smooth", width: [0, 2] },
    plotOptions: { bar: { columnWidth: "60%", borderRadius: 1 } },
  };
  return (
    <Chart
      options={options}
      series={[
        {
          name: "rainfall (3-day MA)",
          type: "column",
          color: TERM.faint,
          data: series.map((p) => p.rainfall_ma ?? null),
        },
        {
          name: "rate of increase",
          type: "line",
          color: TERM.amber,
          data: series.map((p) => p.rainfall_rate ?? null),
        },
      ]}
      type="line"
      height={height}
    />
  );
}

/** Smoothed (3-day MA) probabilities — the series TCDL actually evaluates. */
export function TermSmoothedProbabilityChart({ series, height = 300 }: ChartProps) {
  const defs = [
    { key: "flood_prob_ma", name: "flood (3d MA)", color: TERM.dim },
    { key: "landslide_prob_ma", name: "landslide (3d MA)", color: TERM.amber },
    { key: "coupled_prob_ma", name: "coupled (3d MA)", color: TERM.phosphor },
  ] as const;
  return (
    <Chart
      options={baseOptions(
        series.map((p) => p.date),
        "smoothed probability",
        { percent: true }
      )}
      series={defs.map((d) => ({
        name: d.name,
        color: d.color,
        data: series.map((p) => {
          const v = p[d.key];
          return typeof v === "number" ? v : null;
        }),
      }))}
      type="line"
      height={height}
    />
  );
}

/** Rate-of-increase signals (3-day change in each smoothed series). */
export function TermRateOfIncreaseChart({ series, height = 300 }: ChartProps) {
  const defs = [
    { key: "flood_prob_rate", name: "flood rate", color: TERM.dim },
    { key: "landslide_prob_rate", name: "landslide rate", color: TERM.amber },
    { key: "coupled_prob_rate", name: "coupled rate", color: TERM.phosphor },
    { key: "soil_moisture_rate", name: "soil moisture rate", color: "#b2caa0" },
  ] as const;
  const options: ApexOptions = {
    ...baseOptions(
      series.map((p) => p.date),
      "3-day change"
    ),
    annotations: {
      yaxis: [
        {
          y: 0,
          borderColor: "#3d5d2d",
          strokeDashArray: 3,
          label: {
            text: "no change",
            style: { fontSize: "10px", color: AXIS, background: "#050805" },
          },
        },
      ],
    },
  };
  return (
    <Chart
      options={options}
      series={defs.map((d) => ({
        name: d.name,
        color: d.color,
        data: series.map((p) => {
          const v = p[d.key];
          return typeof v === "number" ? v : null;
        }),
      }))}
      type="line"
      height={height}
    />
  );
}

/** Root-zone soil moisture moving average. */
export function TermSoilMoistureChart({ series, height = 300 }: ChartProps) {
  return (
    <Chart
      options={baseOptions(
        series.map((p) => p.date),
        "volumetric fraction"
      )}
      series={[
        {
          name: "soil moisture (3d MA)",
          color: TERM.phosphor,
          data: series.map((p) => p.soil_moisture_ma ?? null),
        },
      ]}
      type="area"
      height={height}
    />
  );
}

/** Interpretation notes served by the backend, shown verbatim. */
export function TermEvaluationNotes({
  evaluation,
}: {
  evaluation: EvaluationResponse;
}) {
  const entries = Object.entries(evaluation.interpretation_notes ?? {}).filter(
    ([, v]) => Boolean(v)
  );
  if (!entries.length) return null;
  return (
    <div className="crt-inset p-4">
      <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-[#76cc5c]">
        how to read these results
      </p>
      <ul className="space-y-2">
        {entries.map(([k, v]) => (
          <li key={k} className="text-[11px] leading-relaxed text-[#8ea97e]">
            <span className="text-[#dbe9ce]">{k.replace(/_/g, " ")}:</span> {v}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Lead-time comparison across the evaluated warning systems.
 *
 * Shown in DAYS: the API's *_hours fields are date-quantised (whole days x 24)
 * because the data has daily resolution, so hours would imply false precision.
 */
export function TermLeadTimeChart({
  evaluation,
  height = 320,
}: {
  evaluation: EvaluationResponse;
  height?: number;
}) {
  const keys = Object.keys(evaluation.systems);
  const labels = keys.map((k) => SYSTEM_LABELS[k] ?? k);

  const options: ApexOptions = {
    chart: {
      type: "bar",
      fontFamily: MONO,
      background: "transparent",
      foreColor: TERM.sage,
      toolbar: { show: false },
      animations: { enabled: false },
    },
    theme: { mode: "dark" },
    plotOptions: { bar: { columnWidth: "45%", borderRadius: 1 } },
    colors: [TERM.phosphor, TERM.faint],
    dataLabels: {
      enabled: true,
      formatter: (v: number) => (v ? `${v.toFixed(1)}d` : ""),
      style: { fontSize: "10px", fontWeight: 600, fontFamily: MONO, colors: ["#f5fbe9"] },
    },
    legend: {
      show: true,
      position: "top",
      horizontalAlign: "left",
      fontSize: "11px",
      fontFamily: MONO,
      labels: { colors: TERM.sage },
    },
    xaxis: {
      categories: labels,
      labels: { style: { fontSize: "10px", colors: AXIS, fontFamily: MONO } },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      title: {
        text: "mean lead time (days, date-based)",
        style: { fontSize: "10px", color: AXIS, fontWeight: 500, fontFamily: MONO },
      },
      labels: { style: { fontSize: "10px", colors: AXIS, fontFamily: MONO } },
    },
    grid: { borderColor: GRID, strokeDashArray: 3 },
    tooltip: {
      theme: "dark",
      style: { fontFamily: MONO, fontSize: "11px" },
      y: { formatter: (v: number) => `${v?.toFixed(1)} days (daily-resolution data)` },
    },
  };

  return (
    <Chart
      options={options}
      series={[
        {
          name: "contiguous lead time",
          data: keys.map(
            (k) => (evaluation.systems[k].mean_lead_time_hours_contiguous ?? 0) / 24
          ),
        },
        {
          name: "earliest-warning lead time",
          data: keys.map(
            (k) => (evaluation.systems[k].mean_lead_time_hours_earliest ?? 0) / 24
          ),
        },
      ]}
      type="bar"
      height={height}
    />
  );
}

/**
 * Evaluation table. Time-in-warning and false-alarm rate stay beside every
 * lead time, and the always-warn control stays visible — lead time alone can
 * always be improved by warning more often.
 */
export function TermEvaluationTable({
  evaluation,
}: {
  evaluation: EvaluationResponse;
}) {
  const keys = Object.keys(evaluation.systems);
  return (
    <div className="overflow-x-auto">
      <table className="crt-table">
        <thead>
          <tr>
            {[
              "system",
              "events",
              "detected",
              "detection_rate",
              "contiguous_lead",
              "earliest_lead",
              "time_in_warning",
              "false_alarm_rate",
            ].map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => {
            const s = evaluation.systems[k];
            const isControl = k === "always_warn_control";
            const isCoupled = k === "tcdl_coupled";
            return (
              <tr
                key={k}
                className={isCoupled ? "crt-row-hi" : ""}
                style={isControl ? { opacity: 0.65 } : undefined}
              >
                <td>
                  <span className="flex items-center gap-2">
                    <span style={{ color: isCoupled ? TERM.phosphor : TERM.sage }}>
                      {SYSTEM_LABELS[k] ?? k}
                    </span>
                    {isControl && (
                      <span className="text-[10px] text-[#538f3d]">[reference]</span>
                    )}
                  </span>
                </td>
                <td>{s.n_events}</td>
                <td>{s.n_detected}</td>
                <td style={{ color: TERM.ink }}>{pct(s.detection_rate, 1)}</td>
                <td style={{ color: TERM.ink }}>
                  {leadDays(s.mean_lead_time_hours_contiguous)}
                </td>
                <td>{leadDays(s.mean_lead_time_hours_earliest)}</td>
                <td>{pct(s.time_in_warning_rate, 1)}</td>
                <td>{pct(s.false_alarm_day_rate, 1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[10px] leading-relaxed text-[#6a8958]">
        Lead times are mean days per detected onset and are date-based: the data has
        daily resolution, so they are not hour-level timings.
      </p>
    </div>
  );
}
