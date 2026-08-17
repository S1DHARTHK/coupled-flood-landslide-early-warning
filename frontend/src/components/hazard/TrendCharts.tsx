/**
 * Time-series charts for ML probabilities and environmental signals.
 *
 * Built on ApexCharts (react-apexcharts), the charting library already present
 * in the project. Reusing it keeps a single charting system rather than
 * introducing a second library.
 *
 * All series come from backend `/trends` output. Nothing is smoothed,
 * differenced or otherwise computed here -- the moving averages and rates of
 * increase are produced by TCDL V1.0 and merely plotted.
 */

import Chart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";
import type { TrendPoint } from "../../types/hazard";

interface SeriesDef {
  key: keyof TrendPoint;
  name: string;
  color: string;
}

function baseOptions(
  categories: string[],
  yTitle: string,
  opts: { percent?: boolean; zeroLine?: boolean } = {}
): ApexOptions {
  return {
    chart: {
      type: "line",
      fontFamily: "Outfit, sans-serif",
      toolbar: { show: false },
      animations: { enabled: false },
      zoom: { enabled: false },
    },
    stroke: { curve: "smooth", width: 2 },
    dataLabels: { enabled: false },
    legend: {
      show: true,
      position: "top",
      horizontalAlign: "left",
      fontSize: "12px",
      markers: { size: 5 },
    },
    grid: {
      borderColor: "#e5e7eb",
      strokeDashArray: 4,
      xaxis: { lines: { show: false } },
      yaxis: { lines: { show: true } },
    },
    xaxis: {
      categories,
      tickAmount: 8,
      labels: { style: { fontSize: "11px", colors: "#98a2b3" }, rotate: 0 },
      axisBorder: { show: false },
      axisTicks: { show: false },
      tooltip: { enabled: false },
    },
    yaxis: {
      title: { text: yTitle, style: { fontSize: "11px", color: "#98a2b3", fontWeight: 500 } },
      labels: {
        style: { fontSize: "11px", colors: "#98a2b3" },
        formatter: (v: number) =>
          opts.percent ? `${Math.round(v * 100)}%` : v.toFixed(2),
      },
    },
    tooltip: {
      shared: true,
      intersect: false,
      x: { show: true },
      y: {
        formatter: (v: number) =>
          v === null || v === undefined
            ? "no data"
            : opts.percent
            ? `${(v * 100).toFixed(1)}%`
            : v.toFixed(3),
      },
    },
    annotations: opts.zeroLine
      ? {
          yaxis: [
            {
              y: 0,
              borderColor: "#98a2b3",
              strokeDashArray: 3,
              label: { text: "no change", style: { fontSize: "10px" } },
            },
          ],
        }
      : undefined,
  };
}

function buildSeries(series: TrendPoint[], defs: SeriesDef[]) {
  return defs.map((d) => ({
    name: d.name,
    color: d.color,
    data: series.map((p) => {
      const v = p[d.key];
      return typeof v === "number" ? v : null;
    }),
  }));
}

interface ChartProps {
  series: TrendPoint[];
  height?: number;
}

/** Raw + smoothed hazard probabilities from both XGBoost models plus coupled. */
export function ProbabilityTrendChart({ series, height = 300 }: ChartProps) {
  const categories = series.map((p) => p.date);
  const defs: SeriesDef[] = [
    { key: "flood_probability", name: "Flood probability", color: "#2e90fa" },
    { key: "landslide_probability", name: "Landslide probability", color: "#f79009" },
    { key: "coupled_probability", name: "Coupled probability", color: "#f04438" },
  ];
  return (
    <Chart
      options={baseOptions(categories, "probability", { percent: true })}
      series={buildSeries(series, defs)}
      type="line"
      height={height}
    />
  );
}

/** The smoothed (moving-average) probabilities TCDL actually evaluates. */
export function SmoothedProbabilityChart({ series, height = 300 }: ChartProps) {
  const categories = series.map((p) => p.date);
  const defs: SeriesDef[] = [
    { key: "flood_prob_ma", name: "Flood (3-day MA)", color: "#2e90fa" },
    { key: "landslide_prob_ma", name: "Landslide (3-day MA)", color: "#f79009" },
    { key: "coupled_prob_ma", name: "Coupled (3-day MA)", color: "#f04438" },
  ];
  return (
    <Chart
      options={baseOptions(categories, "smoothed probability", { percent: true })}
      series={buildSeries(series, defs)}
      type="line"
      height={height}
    />
  );
}

/** Rate-of-increase signals (change in the smoothed series over 3 days). */
export function RateOfIncreaseChart({ series, height = 300 }: ChartProps) {
  const categories = series.map((p) => p.date);
  const defs: SeriesDef[] = [
    { key: "flood_prob_rate", name: "Flood prob. rate", color: "#2e90fa" },
    { key: "landslide_prob_rate", name: "Landslide prob. rate", color: "#f79009" },
    { key: "coupled_prob_rate", name: "Coupled prob. rate", color: "#f04438" },
    { key: "soil_moisture_rate", name: "Soil moisture rate", color: "#7a5af8" },
  ];
  return (
    <Chart
      options={baseOptions(categories, "3-day change", { zeroLine: true })}
      series={buildSeries(series, defs)}
      type="line"
      height={height}
    />
  );
}

/** Rainfall: daily moving average plus its rate of increase. */
export function RainfallChart({ series, height = 300 }: ChartProps) {
  const categories = series.map((p) => p.date);
  const options: ApexOptions = {
    ...baseOptions(categories, "mm / day"),
    chart: {
      type: "line",
      fontFamily: "Outfit, sans-serif",
      toolbar: { show: false },
      animations: { enabled: false },
      zoom: { enabled: false },
    },
    stroke: { curve: "smooth", width: [0, 2] },
    plotOptions: { bar: { columnWidth: "60%", borderRadius: 2 } },
  };
  return (
    <Chart
      options={options}
      series={[
        {
          name: "Rainfall (3-day MA)",
          type: "column",
          color: "#3182bd",
          data: series.map((p) => p.rainfall_ma ?? null),
        },
        {
          name: "Rainfall rate of increase",
          type: "line",
          color: "#f04438",
          data: series.map((p) => p.rainfall_rate ?? null),
        },
      ]}
      type="line"
      height={height}
    />
  );
}

/** Soil moisture moving average. */
export function SoilMoistureChart({ series, height = 300 }: ChartProps) {
  const categories = series.map((p) => p.date);
  const defs: SeriesDef[] = [
    { key: "soil_moisture_ma", name: "Soil moisture (3-day MA)", color: "#756bb1" },
  ];
  return (
    <Chart
      options={baseOptions(categories, "volumetric fraction")}
      series={buildSeries(series, defs)}
      type="area"
      height={height}
    />
  );
}
