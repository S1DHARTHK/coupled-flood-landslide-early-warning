/**
 * Current TCDL warning panel.
 *
 * Displays the warning decision, its severity, both hazard probabilities, the
 * coupled probability, the trend indicators TCDL consumes, and the exact rules
 * that fired. Everything is read from backend fields -- no rule is evaluated
 * here and no rule is invented.
 */

import { TCDL_RULES } from "../../types/hazard";
import type { CurrentLocation, TcdlRuleId } from "../../types/hazard";
import {
  LEVEL_META,
  WARNING_TYPE_META,
  formatCoord,
  hazardLevel,
  num,
  pct,
  ruleIsBaseline,
  trendArrow,
} from "./hazardUtils";
import { EmptyBlock } from "./StateBlocks";

interface Props {
  location: CurrentLocation | null;
  ruleDescriptions?: Record<string, string>;
}

function TrendRow({
  label,
  value,
  rate,
  unit = "",
}: {
  label: string;
  value: string;
  rate: number | null | undefined;
  unit?: string;
}) {
  const t = trendArrow(rate);
  return (
    <div className="flex items-center justify-between border-b border-gray-100 py-2 last:border-0 dark:border-gray-800">
      <span className="text-xs text-gray-600 dark:text-gray-400">{label}</span>
      <span className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-800 dark:text-white/90">
          {value}
          {unit}
        </span>
        <span className={`text-xs font-bold ${t.cls}`} title={`${t.label}`}>
          {t.glyph}
        </span>
      </span>
    </div>
  );
}

export default function CurrentWarningPanel({ location, ruleDescriptions }: Props) {
  if (!location) {
    return (
      <div className="rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03]">
        <EmptyBlock message="Select a monitoring point to view its warning status." />
      </div>
    );
  }

  const { tcdl, trends, flood, landslide, environment } = location;
  const wmeta = WARNING_TYPE_META[tcdl.warning_type];
  const level = hazardLevel(trends.coupled_probability, tcdl.warning_status);
  const lmeta = LEVEL_META[level];
  const fired = new Set<TcdlRuleId>(tcdl.triggered_rules ?? []);

  return (
    <div className="ews-elevated flex h-full flex-col rounded-2xl border border-gray-200/90 bg-white dark:border-gray-800 dark:bg-white/[0.03]">
      {/* Header: the warning state, made visually prominent */}
      <div
        className="rounded-t-2xl px-5 py-4"
        style={{ background: `${wmeta.color}14`, borderBottom: `2px solid ${wmeta.color}` }}
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">
              Current TCDL Decision
            </p>
            <p
              className="mt-1 text-xl font-bold leading-tight"
              style={{ color: wmeta.color }}
            >
              {tcdl.warning_type.toUpperCase()}
            </p>
            <p className="mt-0.5 text-xs text-gray-600 dark:text-gray-400">
              {formatCoord(environment.latitude, environment.longitude)}
              {location.district ? ` · District: ${location.district}` : ""}
            </p>
          </div>
          <span
            className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-bold ${lmeta.badge}`}
          >
            {lmeta.label}
          </span>
        </div>
        <p className="mt-2 text-[11px] text-gray-500 dark:text-gray-400">
          {tcdl.warning_timestamp
            ? `Warning timestamp: ${tcdl.warning_timestamp} (daily resolution)`
            : `No active warning · as of ${environment.date}`}
        </p>
      </div>

      <div className="space-y-4 p-5">
        {/* Probabilities */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { k: "Flood", v: flood.probability, c: "#2e90fa" },
            { k: "Landslide", v: landslide.probability, c: "#f79009" },
            { k: "Coupled", v: trends.coupled_probability, c: "#f04438" },
          ].map((x) => (
            <div
              key={x.k}
              className="rounded-lg border border-gray-100 p-2.5 text-center dark:border-gray-800"
            >
              <p className="text-[10px] uppercase tracking-wide text-gray-500 dark:text-gray-400">
                {x.k}
              </p>
              <p className="mt-0.5 text-lg font-bold" style={{ color: x.c }}>
                {pct(x.v, 0)}
              </p>
            </div>
          ))}
        </div>

        {/* Trend indicators consumed by TCDL */}
        <div>
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
            Temporal signals
          </p>
          <TrendRow
            label="Flood probability (3-day MA)"
            value={pct(trends.flood_prob_ma, 1)}
            rate={trends.flood_prob_rate}
          />
          <TrendRow
            label="Landslide probability (3-day MA)"
            value={pct(trends.landslide_prob_ma, 1)}
            rate={trends.landslide_prob_rate}
          />
          <TrendRow
            label="Rainfall (3-day MA)"
            value={num(trends.rainfall_ma, 1)}
            unit=" mm/day"
            rate={trends.rainfall_rate}
          />
          <TrendRow
            label="Soil moisture (3-day MA)"
            value={num(trends.soil_moisture_ma, 3)}
            rate={trends.soil_moisture_rate}
          />
        </div>

        {/* TCDL rules */}
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
            TCDL rules ({fired.size} of {TCDL_RULES.length} triggered)
          </p>
          <ul className="space-y-1">
            {TCDL_RULES.map((rule) => {
              const on = fired.has(rule);
              return (
                <li
                  key={rule}
                  className={`flex items-start gap-2 rounded-lg px-2 py-1.5 ${
                    on
                      ? "bg-error-50 dark:bg-error-500/10"
                      : "bg-gray-50 dark:bg-white/[0.02]"
                  }`}
                  title={ruleDescriptions?.[rule] ?? ""}
                >
                  <span
                    className={`mt-px text-xs font-bold ${
                      on ? "text-error-600 dark:text-error-400" : "text-gray-300 dark:text-gray-600"
                    }`}
                  >
                    {on ? "✓" : "○"}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span
                      className={`font-mono text-[11px] ${
                        on
                          ? "font-semibold text-error-700 dark:text-error-400"
                          : "text-gray-400 dark:text-gray-600"
                      }`}
                    >
                      {rule}
                    </span>
                    <span className="ml-1.5 text-[10px] text-gray-400 dark:text-gray-500">
                      {ruleIsBaseline(rule) ? "baseline" : "coupling"}
                    </span>
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="mt-2 text-[10px] leading-relaxed text-gray-400 dark:text-gray-500">
            Rules R1/R2 replicate the single-model baselines; R3–R7 are the coupling
            rules. Rule conditions and thresholds are defined by TCDL V1.0 in the ML
            layer and are not evaluated in this interface.
          </p>
        </div>
      </div>
    </div>
  );
}
