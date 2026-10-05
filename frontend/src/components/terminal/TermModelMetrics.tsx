/**
 * Terminal-themed model evaluation display: metric tiles, confusion matrix,
 * calibration note, feature importance and the chronological split table.
 *
 * Same contract as the light-theme `components/hazard/ModelMetrics`: every
 * value is read from the backend's stored training results. Nothing is
 * recomputed and no model is trained here.
 */

import type {
  FeatureImportanceRow,
  ModelPerformance,
  SplitMetrics,
} from "../../types/hazard";
import { num, pct } from "../hazard/hazardUtils";
import { TERM } from "./termColors";

export function TermMetricTiles({ metrics }: { metrics: SplitMetrics }) {
  const tiles = [
    { label: "accuracy", value: num(metrics.accuracy, 4) },
    { label: "precision", value: num(metrics.precision, 4) },
    { label: "recall", value: num(metrics.recall, 4) },
    { label: "f1_score", value: num(metrics.f1, 4) },
    { label: "roc_auc", value: num(metrics.roc_auc, 4) },
    { label: "pr_auc", value: num(metrics.pr_auc, 4) },
    { label: "brier_score", value: num(metrics.brier_score, 4) },
    { label: "threshold", value: num(metrics.threshold, 2) },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {tiles.map((t) => (
        <div key={t.label} className="crt-inset px-3 py-2.5">
          <p className="text-[10px] uppercase tracking-[0.16em] text-[#76cc5c]">
            {t.label}
          </p>
          <p className="mt-1 text-[17px] font-bold text-[#f5fbe9]">{t.value}</p>
        </div>
      ))}
    </div>
  );
}

export function TermConfusionMatrix({ metrics }: { metrics: SplitMetrics }) {
  const cm = metrics.confusion_matrix;

  const cell = (label: string, value: number, tone: "good" | "bad" | "neutral") => {
    const color =
      tone === "good" ? TERM.phosphor : tone === "bad" ? TERM.red : TERM.sage;
    return (
      <div
        className="rounded-[3px] p-3 text-center"
        style={{
          border: `1px solid ${color}44`,
          background: `${color}0d`,
        }}
      >
        <p className="text-[10px] uppercase tracking-wide text-[#6a8958]">{label}</p>
        <p className="mt-0.5 text-[20px] font-bold" style={{ color }}>
          {value}
        </p>
      </div>
    );
  };

  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        {cell("true negatives", cm.true_negatives, "good")}
        {cell("false positives", cm.false_positives, "neutral")}
        {cell("false negatives", cm.false_negatives, "bad")}
        {cell("true positives", cm.true_positives, "good")}
      </div>
      <p className="mt-2.5 text-[11px] leading-relaxed text-[#8ea97e]">
        In an early-warning context the{" "}
        <span style={{ color: TERM.red }}>
          {cm.false_negatives} false negatives
        </span>{" "}
        are the costly cell: hazard days the model did not flag. Observed positive
        rate {pct(metrics.observed_positive_rate, 2)}; mean predicted probability{" "}
        {pct(metrics.mean_predicted_probability, 2)}.
      </p>
    </div>
  );
}

export function TermCalibrationNote({ model }: { model: ModelPerformance }) {
  const test = model.calibration_summary?.test;
  if (!test) return null;
  return (
    <div
      className="rounded-[3px] p-3"
      style={{
        border: `1px solid ${TERM.amber}55`,
        background: `${TERM.amber}0d`,
      }}
    >
      <p
        className="crt-glow-amber text-[10px] font-bold uppercase tracking-[0.18em]"
        style={{ color: TERM.amber }}
      >
        [!] probability calibration
      </p>
      <p className="mt-1.5 text-[11px] leading-relaxed text-[#8a8460]">
        Mean predicted probability {pct(test.mean_predicted_probability, 1)} against an
        observed event rate of {pct(test.observed_positive_rate, 1)} (Brier{" "}
        {num(test.brier_score, 4)}). The model was trained with{" "}
        <span className="text-[#dbe9ce]">scale_pos_weight</span> to favour recall,
        which inflates probabilities. They rank correctly but are not literal
        likelihoods. Calibration is a known open item, deliberately out of scope for
        V1.0.
      </p>
    </div>
  );
}

/**
 * Gain-based feature importance as ASCII meters, matching the SHAP importance
 * block. Amber marks categorical features, which XGBoost handles natively.
 */
export function TermFeatureImportance({
  rows,
  categoricalFeatures = [],
}: {
  rows: FeatureImportanceRow[];
  categoricalFeatures?: string[];
}) {
  const sorted = [...rows].sort((a, b) => b.importance_gain - a.importance_gain);
  const max = Math.max(...sorted.map((r) => r.importance_gain), 1e-9);

  return (
    <div className="space-y-1.5">
      {sorted.map((r) => {
        const cat = categoricalFeatures.includes(r.feature);
        const color = cat ? TERM.amber : TERM.phosphor;
        return (
          <div key={r.feature} className="flex items-center gap-3">
            <span className="w-[140px] shrink-0 truncate text-[11px] text-[#8ea97e]">
              {r.feature}
              {cat && <span className="ml-1 text-[9px] text-[#538f3d]">cat</span>}
            </span>
            <div className="crt-meter flex-1">
              <div
                className="crt-meter-fill"
                style={{
                  width: `${(r.importance_gain / max) * 100}%`,
                  background: `linear-gradient(90deg, ${color}55, ${color})`,
                  boxShadow: `0 0 10px ${color}70`,
                }}
              />
            </div>
            <span className="w-[150px] shrink-0 text-right text-[10px] tabular-nums text-[#6a8958]">
              gain {r.importance_gain.toFixed(1)}
              <span className="ml-1.5 text-[#538f3d]">
                {pct(r.importance_normalised, 1)} · {r.split_count ?? 0} splits
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function TermSplitPeriodTable({ model }: { model: ModelPerformance }) {
  const rows: Array<["train" | "validation" | "test", string]> = [
    ["train", "training"],
    ["validation", "validation"],
    ["test", "test (unseen)"],
  ];
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="crt-table">
          <thead>
            <tr>
              {[
                "split",
                "period",
                "rows",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
              ].map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, label]) => {
              const m = model.splits[key];
              const period = model.split_periods?.[key];
              return (
                <tr key={key} className={key === "test" ? "crt-row-hi" : ""}>
                  <td style={{ color: key === "test" ? TERM.phosphor : TERM.sage }}>
                    {label}
                  </td>
                  <td className="text-[10px]">
                    {period ? `${period[0]} → ${period[1]}` : "—"}
                  </td>
                  <td>{m.n_rows}</td>
                  <td>{num(m.accuracy, 4)}</td>
                  <td>{num(m.precision, 4)}</td>
                  <td>{num(m.recall, 4)}</td>
                  <td>{num(m.f1, 4)}</td>
                  <td>{num(m.roc_auc, 4)}</td>
                  <td>{num(m.pr_auc, 4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[10px] text-[#6a8958]">
        Chronological split — training always precedes validation, which precedes the
        unseen test period. No random shuffling.
      </p>
    </div>
  );
}
