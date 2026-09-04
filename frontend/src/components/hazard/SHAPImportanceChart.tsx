/**
 * Global SHAP importance -- horizontal bar chart of mean|SHAP| per feature.
 *
 * Data comes from the backend (/explain/global). This component only renders;
 * it computes no SHAP values. Built on ApexCharts, the charting library already
 * used across the dashboard -- no new chart dependency is introduced.
 */

import Chart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";
import type { ShapImportanceRow } from "../../types/hazard";

// Reuse the same numeric-vs-categorical palette as the model-performance
// feature-importance chart, so the two read as one system.
const NUMERIC_COLOR = "#2c7fb8";
const CATEGORICAL_COLOR = "#d95f0e";

export default function SHAPImportanceChart({
  rows,
  categoricalFeatures = [],
  height,
}: {
  rows: ShapImportanceRow[];
  categoricalFeatures?: string[];
  height?: number;
}) {
  const sorted = [...rows].sort((a, b) => a.mean_abs_shap - b.mean_abs_shap);
  const isCat = (f: string) =>
    categoricalFeatures.includes(f) ||
    sorted.find((r) => r.feature === f)?.is_categorical === true;

  const options: ApexOptions = {
    chart: {
      type: "bar",
      fontFamily: "Outfit, sans-serif",
      toolbar: { show: false },
      animations: { enabled: false },
    },
    plotOptions: {
      bar: { horizontal: true, barHeight: "68%", borderRadius: 3, distributed: true },
    },
    colors: sorted.map((r) => (isCat(r.feature) ? CATEGORICAL_COLOR : NUMERIC_COLOR)),
    legend: { show: false },
    dataLabels: {
      enabled: true,
      formatter: (v: number) => v.toFixed(3),
      style: { fontSize: "10px", colors: ["#667085"] },
      offsetX: 28,
    },
    xaxis: {
      categories: sorted.map((r) => r.feature),
      title: {
        text: "mean |SHAP value|  (log-odds impact on model output)",
        style: { fontSize: "11px", color: "#98a2b3", fontWeight: 500 },
      },
      labels: { style: { fontSize: "11px", colors: "#98a2b3" } },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: { labels: { style: { fontSize: "11px", colors: "#98a2b3" } } },
    grid: { borderColor: "#e5e7eb", strokeDashArray: 4 },
    tooltip: {
      y: {
        formatter: (v: number, ctx) => {
          const row = sorted[ctx?.dataPointIndex ?? 0];
          return `mean |SHAP| ${v.toFixed(4)} · ${(
            (row?.share_of_total ?? 0) * 100
          ).toFixed(1)}% of total`;
        },
      },
    },
  };

  return (
    <Chart
      options={options}
      series={[{ name: "mean |SHAP|", data: sorted.map((r) => r.mean_abs_shap) }]}
      type="bar"
      height={height ?? Math.max(280, sorted.length * 30)}
    />
  );
}
