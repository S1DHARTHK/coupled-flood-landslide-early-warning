/** Top summary cards: flood risk, landslide risk, TCDL status, warning type. */

import type { CurrentLocation } from "../../types/hazard";
import { WARNING_TYPE_META, pct, trendArrow } from "./hazardUtils";

interface Props {
  location: CurrentLocation | null;
  /** How many of the monitored points are currently under a TCDL warning. */
  activeWarnings: number;
  totalLocations: number;
}

function Card({
  label,
  value,
  sub,
  accent,
  trend,
  glyph,
  glyphKind = "line",
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
  trend?: { glyph: string; cls: string; label: string };
  glyph?: React.ReactNode;
  glyphKind?: "line" | "emoji";
}) {
  return (
    <div className="ews-elevated relative overflow-hidden rounded-2xl border border-gray-200/90 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03]">
      {glyph && (
        <span
          className={`ews-watermark ${glyphKind === "emoji" ? "ews-emoji" : ""}`}
          style={
            glyphKind === "line" && accent ? { color: accent } : undefined
          }
        >
          {glyph}
        </span>
      )}
      <p className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
        {label}
      </p>
      <div className="mt-2 flex items-end gap-2">
        <span
          className="text-2xl font-bold leading-none text-gray-800 dark:text-white/90"
          style={accent ? { color: accent } : undefined}
        >
          {value}
        </span>
        {trend && (
          <span className={`text-sm font-semibold ${trend.cls}`} title={trend.label}>
            {trend.glyph}
          </span>
        )}
      </div>
      {sub && (
        <p className="mt-1.5 text-xs text-gray-500 dark:text-gray-400">{sub}</p>
      )}
    </div>
  );
}

export default function HazardSummaryCards({
  location,
  activeWarnings,
  totalLocations,
}: Props) {
  const t = location?.trends;
  const wtype = location?.tcdl.warning_type ?? "No Warning";
  const meta = WARNING_TYPE_META[wtype];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Card
        label="Flood Risk"
        value={pct(location?.flood.probability ?? null, 0)}
        sub={
          location
            ? `Flood XGBoost V1.0 · threshold 0.50`
            : "Awaiting data"
        }
        accent="#2e90fa"
        trend={trendArrow(t?.flood_prob_rate)}
      />
      <Card
        label="Landslide Risk"
        value={pct(location?.landslide.probability ?? null, 0)}
        sub={location ? "Landslide XGBoost V1.0 · threshold 0.50" : "Awaiting data"}
        accent="#f79009"
        trend={trendArrow(t?.landslide_prob_rate)}
      />
      <Card
        label="TCDL Status"
        value={(location?.tcdl.warning_status ?? "—").toUpperCase()}
        sub={`${activeWarnings} of ${totalLocations} monitoring points under warning`}
        accent={meta.color}
      />
      <Card
        label="Warning Type"
        value={wtype === "No Warning" ? "NONE" : wtype.replace(" Warning", "").toUpperCase()}
        sub={
          location?.tcdl.triggered_rules.length
            ? `${location.tcdl.triggered_rules.length} TCDL rule(s) triggered`
            : "No TCDL rule conditions met"
        }
        accent={meta.color}
      />
    </div>
  );
}
