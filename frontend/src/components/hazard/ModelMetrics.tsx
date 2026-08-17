/**
 * Model evaluation display: metric tiles, confusion matrix and feature
 * importance. All values are read from the backend's stored training results.
 * Nothing is recomputed and no model is trained here.
 */

import Chart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";
import type {
  FeatureImportanceRow,
  ModelPerformance,
  SplitMetrics,
} from "../../types/hazard";
import { num, pct } from "./hazardUtils";

export function MetricTiles({ metrics }: { metrics: SplitMetrics }) {
  const tiles = [
    { label: "Accuracy", value: num(metrics.accuracy, 4) },
    { label: "Precision", value: num(metrics.precision, 4) },
    { label: "Recall", value: num(metrics.recall, 4) },
    { label: "F1-score", value: num(metrics.f1, 4) },
    { label: "ROC-AUC", value: num(metrics.roc_auc, 4) },
    { label: "PR-AUC", value: num(metrics.pr_auc, 4) },
    { label: "Brier score", value: num(metrics.brier_score, 4) },
    { label: "Threshold", value: num(metrics.threshold, 2) },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {tiles.map((t) => (
        <div
          key={t.label}
          className="rounded-xl border border-gray-100 bg-gray-50 px-3 py-2.5 dark:border-gray-800 dark:bg-white/[0.02]"
        >
          <p className="text-[10px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
            {t.label}
          </p>
          <p className="mt-1 text-lg font-bold text-gray-800 dark:text-white/90">
            {t.value}
          </p>
        </div>
      ))}
    </div>
  );
}

export function ConfusionMatrix({ metrics }: { metrics: SplitMetrics }) {
  const cm = metrics.confusion_matrix;
  const cell = (
    label: string,
    value: number,
    tone: "good" | "bad" | "neutral"
  ) => (
    <div
      className={`rounded-lg p-3 text-center ${
        tone === "good"
          ? "bg-success-50 dark:bg-success-500/10"
          : tone === "bad"
          ? "bg-error-50 dark:bg-error-500/10"
          : "bg-gray-50 dark:bg-white/[0.02]"
      }`}
    >
      <p className="text-[10px] uppercase tracking-wide text-gray-500 dark:text-gray-400">
        {label}
      </p>
      <p
        className={`mt-0.5 text-xl font-bold ${
          tone === "good"
            ? "text-success-700 dark:text-success-400"
            : tone === "bad"
            ? "text-error-700 dark:text-error-400"
            : "text-gray-800 dark:text-white/90"
        }`}
      >
        {value}
      </p>
    </div>
  );

  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        {cell("True negatives", cm.true_negatives, "good")}
        {cell("False positives", cm.false_positives, "neutral")}
        {cell("False negatives", cm.false_negatives, "bad")}
        {cell("True positives", cm.true_positives, "good")}
      </div>
      <p className="mt-2.5 text-[11px] leading-relaxed text-gray-500 dark:text-gray-400">
        In an early-warning context the{" "}
        <span className="font-medium text-error-600 dark:text-error-400">
          {cm.false_negatives} false negatives
        </span>{" "}
        are the costly cell: hazard days the model did not flag. Observed positive
        rate {pct(metrics.observed_positive_rate, 2)}; mean predicted probability{" "}
        {pct(metrics.mean_predicted_probability, 2)}.
      </p>
    </div>
  );
}

export function CalibrationNote({ model }: { model: ModelPerformance }) {
  const test = model.calibration_summary?.test;
  if (!test) return null;
  return (
    <div className="rounded-xl border border-warning-200 bg-warning-50 p-3 dark:border-warning-500/30 dark:bg-warning-500/10">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-warning-700 dark:text-warning-400">
        Probability calibration
      </p>
      <p className="mt-1 text-xs leading-relaxed text-warning-800 dark:text-warning-200">
        Mean predicted probability {pct(test.mean_predicted_probability, 1)} against an
        observed event rate of {pct(test.observed_positive_rate, 1)} (Brier{" "}
        {num(test.brier_score, 4)}). The model was trained with{" "}
        <code className="font-mono">scale_pos_weight</code> to favour recall, which
        inflates probabilities. They rank correctly but are not literal likelihoods.
        Calibration is a known open item, deliberately out of scope for V1.0.
      </p>
    </div>
  );
}

export function FeatureImportanceChart({
  rows,
  categoricalFeatures = [],
  height,
}: {
  rows: FeatureImportanceRow[];
  categoricalFeatures?: string[];
  height?: number;
}) {
  const sorted = [...rows].sort((a, b) => a.importance_gain - b.importance_gain);
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
    colors: sorted.map((r) =>
      categoricalFeatures.includes(r.feature) ? "#d95f0e" : "#2c7fb8"
    ),
    legend: { show: false },
    dataLabels: {
      enabled: true,
      formatter: (v: number) => v.toFixed(1),
      style: { fontSize: "10px", colors: ["#667085"] },
      offsetX: 26,
    },
    xaxis: {
      categories: sorted.map((r) => r.feature),
      title: {
        text: "importance (gain)",
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
          return `gain ${v.toFixed(2)} · ${pct(row?.importance_normalised, 1)} of total · ${
            row?.split_count ?? 0
          } splits`;
        },
      },
    },
  };
  return (
    <Chart
      options={options}
      series={[{ name: "Importance (gain)", data: sorted.map((r) => r.importance_gain) }]}
      type="bar"
      height={height ?? Math.max(280, sorted.length * 30)}
    />
  );
}

export function SplitPeriodTable({ model }: { model: ModelPerformance }) {
  const rows: Array<["train" | "validation" | "test", string]> = [
    ["train", "Training"],
    ["validation", "Validation"],
    ["test", "Test (unseen)"],
  ];
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left dark:border-gray-800">
            {["Split", "Period", "Rows", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"].map(
              (h) => (
                <th
                  key={h}
                  className="whitespace-nowrap px-3 py-2.5 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400"
                >
                  {h}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label]) => {
            const m = model.splits[key];
            const period = model.split_periods?.[key];
            return (
              <tr
                key={key}
                className={`border-b border-gray-100 dark:border-gray-800 ${
                  key === "test" ? "bg-brand-50/40 dark:bg-brand-500/5 font-medium" : ""
                }`}
              >
                <td className="whitespace-nowrap px-3 py-2.5 text-gray-800 dark:text-white/90">
                  {label}
                </td>
                <td className="whitespace-nowrap px-3 py-2.5 text-xs text-gray-500 dark:text-gray-400">
                  {period ? `${period[0]} → ${period[1]}` : "—"}
                </td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">{m.n_rows}</td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">{num(m.accuracy, 4)}</td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">{num(m.precision, 4)}</td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">{num(m.recall, 4)}</td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">{num(m.f1, 4)}</td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">{num(m.roc_auc, 4)}</td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">{num(m.pr_auc, 4)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-gray-500 dark:text-gray-400">
        Chronological split — training always precedes validation, which precedes the
        unseen test period. No random shuffling.
      </p>
    </div>
  );
}
