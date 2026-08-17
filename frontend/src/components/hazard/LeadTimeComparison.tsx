/**
 * Lead-time comparison -- the project's primary research output.
 *
 * Every number here is read from the backend `/evaluation` response, which
 * serves the ML/TCDL evaluation verbatim. NO lead time is computed in the
 * frontend.
 *
 * Two honesty features are deliberate and should not be removed:
 *  1. The always-warn control is shown. A system that warns every day scores
 *     perfect detection and maximum lead time while being useless, so it is
 *     the reference against which the other systems must be read.
 *  2. Time-in-warning and false-alarm rate sit beside every lead time, because
 *     lead time alone can always be improved by warning more often.
 */

import Chart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";
import type { EvaluationResponse } from "../../types/hazard";
import { SYSTEM_COLORS, SYSTEM_LABELS, hours, pct } from "./hazardUtils";

const PRIMARY_SYSTEMS = ["flood_only", "landslide_only", "tcdl_coupled"];

export function LeadTimeBarChart({
  evaluation,
  includeControls = true,
  height = 320,
}: {
  evaluation: EvaluationResponse;
  includeControls?: boolean;
  height?: number;
}) {
  const keys = Object.keys(evaluation.systems).filter((k) =>
    includeControls ? true : PRIMARY_SYSTEMS.includes(k)
  );
  const labels = keys.map((k) => SYSTEM_LABELS[k] ?? k);

  const options: ApexOptions = {
    chart: {
      type: "bar",
      fontFamily: "Outfit, sans-serif",
      toolbar: { show: false },
      animations: { enabled: false },
    },
    plotOptions: {
      bar: { horizontal: false, columnWidth: "45%", borderRadius: 4, distributed: false },
    },
    colors: ["#12b76a", "#98a2b3"],
    dataLabels: {
      enabled: true,
      formatter: (v: number) => (v ? `${v.toFixed(0)}h` : ""),
      style: { fontSize: "11px", fontWeight: 600 },
    },
    legend: { show: true, position: "top", horizontalAlign: "left", fontSize: "12px" },
    xaxis: {
      categories: labels,
      labels: { style: { fontSize: "11px", colors: "#98a2b3" } },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      title: {
        text: "lead time (hours)",
        style: { fontSize: "11px", color: "#98a2b3", fontWeight: 500 },
      },
      labels: { style: { fontSize: "11px", colors: "#98a2b3" } },
    },
    grid: { borderColor: "#e5e7eb", strokeDashArray: 4 },
    tooltip: { y: { formatter: (v: number) => `${v?.toFixed(1)} hours` } },
  };

  return (
    <Chart
      options={options}
      series={[
        {
          name: "Contiguous lead time",
          data: keys.map(
            (k) => evaluation.systems[k].mean_lead_time_hours_contiguous ?? 0
          ),
        },
        {
          name: "Earliest-warning lead time",
          data: keys.map(
            (k) => evaluation.systems[k].mean_lead_time_hours_earliest ?? 0
          ),
        },
      ]}
      type="bar"
      height={height}
    />
  );
}

export function EvaluationTable({ evaluation }: { evaluation: EvaluationResponse }) {
  const keys = Object.keys(evaluation.systems);
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left dark:border-gray-800">
            {[
              "System",
              "Events",
              "Detected",
              "Detection rate",
              "Contiguous lead",
              "Earliest lead",
              "Time in warning",
              "False-alarm rate",
            ].map((h) => (
              <th
                key={h}
                className="whitespace-nowrap px-3 py-2.5 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400"
              >
                {h}
              </th>
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
                className={`border-b border-gray-100 dark:border-gray-800 ${
                  isCoupled ? "bg-success-50/40 dark:bg-success-500/5" : ""
                } ${isControl ? "opacity-70" : ""}`}
              >
                <td className="whitespace-nowrap px-3 py-2.5">
                  <span className="flex items-center gap-2">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ background: SYSTEM_COLORS[k] ?? "#98a2b3" }}
                    />
                    <span
                      className={`text-gray-800 dark:text-white/90 ${
                        isCoupled ? "font-semibold" : ""
                      }`}
                    >
                      {SYSTEM_LABELS[k] ?? k}
                    </span>
                    {isControl && (
                      <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500 dark:bg-white/5 dark:text-gray-400">
                        reference
                      </span>
                    )}
                  </span>
                </td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">
                  {s.n_events}
                </td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">
                  {s.n_detected}
                </td>
                <td className="px-3 py-2.5 font-medium text-gray-800 dark:text-white/90">
                  {pct(s.detection_rate, 1)}
                </td>
                <td className="px-3 py-2.5 font-medium text-gray-800 dark:text-white/90">
                  {hours(s.mean_lead_time_hours_contiguous)}
                </td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">
                  {hours(s.mean_lead_time_hours_earliest)}
                </td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">
                  {pct(s.time_in_warning_rate, 1)}
                </td>
                <td className="px-3 py-2.5 text-gray-600 dark:text-gray-300">
                  {pct(s.false_alarm_day_rate, 1)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Interpretation notes served by the backend, shown verbatim. */
export function EvaluationNotes({ evaluation }: { evaluation: EvaluationResponse }) {
  const notes = evaluation.interpretation_notes ?? {};
  const entries = Object.entries(notes).filter(([, v]) => Boolean(v));
  if (!entries.length) return null;
  return (
    <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 dark:border-gray-800 dark:bg-white/[0.02]">
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
        How to read these results
      </p>
      <ul className="space-y-2">
        {entries.map(([k, v]) => (
          <li key={k} className="text-xs leading-relaxed text-gray-600 dark:text-gray-400">
            <span className="font-medium capitalize text-gray-700 dark:text-gray-300">
              {k.replace(/_/g, " ")}:
            </span>{" "}
            {v}
          </li>
        ))}
      </ul>
    </div>
  );
}
