/**
 * Terminal-themed hazard readouts for the early-warning dashboard.
 *
 * These mirror the light-theme components in `components/hazard` feature for
 * feature (summary cards, TCDL warning panel, monitoring-point selector, map
 * legend) and read the same backend fields. No hazard is computed here: every
 * probability, warning status and triggered rule comes from the ML/TCDL layer.
 */

import { TCDL_RULES } from "../../types/hazard";
import type { CurrentLocation, HazardLevel, TcdlRuleId } from "../../types/hazard";
import {
  LEVEL_META,
  formatCoord,
  hazardLevel,
  num,
  pct,
  ruleIsBaseline,
} from "../hazard/hazardUtils";
import { Meter, TermEmpty } from "./TerminalUI";
import { TERM, levelColor, warningColor } from "./termColors";

/** Rate-of-change arrow in the terminal palette (rising reads as amber). */
function trend(rate: number | null | undefined): { glyph: string; color: string; label: string } {
  if (rate === null || rate === undefined || Number.isNaN(rate)) {
    return { glyph: "-", color: TERM.faint, label: "no data" };
  }
  if (rate > 0) return { glyph: "^", color: TERM.amber, label: "rising" };
  if (rate < 0) return { glyph: "v", color: TERM.phosphor, label: "falling" };
  return { glyph: "=", color: TERM.sage, label: "flat" };
}

/* --------------------------------------------------------- summary cards */

function Card({
  label,
  value,
  sub,
  color = TERM.phosphor,
  meter,
  rate,
}: {
  label: string;
  value: string;
  sub: string;
  color?: string;
  meter?: number | null;
  rate?: number | null;
}) {
  const t = rate !== undefined ? trend(rate) : null;
  return (
    <div className="crt-panel crt-panel-hover p-4">
      <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
        {label}
      </p>
      <div className="mt-2 flex items-end gap-2">
        <span
          className="text-[26px] font-bold leading-none"
          style={{ color, textShadow: `0 0 12px ${color}55` }}
        >
          {value}
        </span>
        {t && (
          <span
            className="text-[13px] font-bold leading-none"
            style={{ color: t.color }}
            title={t.label}
          >
            [{t.glyph}]
          </span>
        )}
      </div>
      {meter !== undefined && (
        <div className="mt-3">
          <Meter value={meter} color={color} />
        </div>
      )}
      <p className="mt-2 text-[11px] leading-relaxed text-[#5f8d68]">{sub}</p>
    </div>
  );
}

export function TermSummaryCards({
  location,
  activeWarnings,
  totalLocations,
}: {
  location: CurrentLocation | null;
  activeWarnings: number;
  totalLocations: number;
}) {
  const t = location?.trends;
  const wtype = location?.tcdl.warning_type ?? "No Warning";
  const wcolor = warningColor(wtype);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Card
        label="flood_risk"
        value={pct(location?.flood.probability ?? null, 0)}
        meter={location?.flood.probability ?? 0}
        rate={t?.flood_prob_rate ?? null}
        color={TERM.dim}
        sub={location ? "Flood XGBoost V1.0 · threshold 0.50" : "awaiting data"}
      />
      <Card
        label="landslide_risk"
        value={pct(location?.landslide.probability ?? null, 0)}
        meter={location?.landslide.probability ?? 0}
        rate={t?.landslide_prob_rate ?? null}
        color={TERM.amber}
        sub={location ? "Landslide XGBoost V1.0 · threshold 0.50" : "awaiting data"}
      />
      <Card
        label="tcdl_status"
        value={(location?.tcdl.warning_status ?? "—").toUpperCase()}
        color={wcolor}
        sub={`${activeWarnings} of ${totalLocations} monitoring points under warning`}
      />
      <Card
        label="warning_type"
        value={
          wtype === "No Warning"
            ? "NONE"
            : wtype.replace(" Warning", "").toUpperCase()
        }
        color={wcolor}
        sub={
          location?.tcdl.triggered_rules.length
            ? `${location.tcdl.triggered_rules.length} TCDL rule(s) triggered`
            : "no TCDL rule conditions met"
        }
      />
    </div>
  );
}

/* ---------------------------------------------------------- warning panel */

function SignalRow({
  label,
  value,
  unit = "",
  rate,
}: {
  label: string;
  value: string;
  unit?: string;
  rate: number | null | undefined;
}) {
  const t = trend(rate);
  return (
    <div className="flex items-center justify-between border-b border-[#0f2a12] py-2 last:border-0">
      <span className="text-[11px] text-[#5f8d68]">{label}</span>
      <span className="flex items-center gap-2">
        <span className="text-[11px] text-[#cfe9d5]">
          {value}
          {unit}
        </span>
        <span
          className="text-[11px] font-bold"
          style={{ color: t.color }}
          title={t.label}
        >
          [{t.glyph}]
        </span>
      </span>
    </div>
  );
}

export function TermWarningPanel({
  location,
  ruleDescriptions,
}: {
  location: CurrentLocation | null;
  ruleDescriptions?: Record<string, string>;
}) {
  if (!location) {
    return (
      <div className="crt-panel h-full p-4">
        <TermEmpty message="select a monitoring point to view its warning status" />
      </div>
    );
  }

  const { tcdl, trends, flood, landslide, environment } = location;
  const wcolor = warningColor(tcdl.warning_type);
  const level = hazardLevel(trends.coupled_probability, tcdl.warning_status);
  const lcolor = levelColor(level);
  const fired = new Set<TcdlRuleId>(tcdl.triggered_rules ?? []);

  return (
    <div className="crt-panel crt-panel-hover flex h-full flex-col">
      {/* Decision header */}
      <div
        className="px-4 py-3.5"
        style={{ background: `${wcolor}0f`, borderBottom: `1px solid ${wcolor}66` }}
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[10px] uppercase tracking-[0.2em] text-[#2bbf5c]">
              current tcdl decision
            </p>
            <p
              className="mt-1 text-[19px] font-bold leading-tight"
              style={{ color: wcolor, textShadow: `0 0 14px ${wcolor}55` }}
            >
              {tcdl.warning_type.toUpperCase()}
            </p>
            <p className="mt-1 text-[11px] text-[#5f8d68]">
              {formatCoord(environment.latitude, environment.longitude)}
              {location.district ? ` · district: ${location.district}` : ""}
            </p>
          </div>
          <span
            className="shrink-0 rounded-[3px] px-2 py-0.5 text-[10px] font-bold uppercase"
            style={{ color: lcolor, border: `1px solid ${lcolor}66` }}
          >
            {LEVEL_META[level as HazardLevel].label}
          </span>
        </div>
        <p className="mt-2 text-[11px] text-[#3d6b47]">
          {tcdl.warning_timestamp
            ? `warning timestamp: ${tcdl.warning_timestamp} (daily resolution)`
            : `no active warning · as of ${environment.date}`}
        </p>
      </div>

      <div className="space-y-4 p-4">
        {/* Probabilities */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { k: "flood", v: flood.probability, c: TERM.dim },
            { k: "landslide", v: landslide.probability, c: TERM.amber },
            { k: "coupled", v: trends.coupled_probability, c: TERM.phosphor },
          ].map((x) => (
            <div key={x.k} className="crt-inset px-2 py-2 text-center">
              <p className="text-[10px] uppercase tracking-wide text-[#3d6b47]">
                {x.k}
              </p>
              <p
                className="mt-0.5 text-[17px] font-bold"
                style={{ color: x.c, textShadow: `0 0 10px ${x.c}45` }}
              >
                {pct(x.v, 0)}
              </p>
            </div>
          ))}
        </div>

        {/* Temporal signals consumed by TCDL */}
        <div>
          <p className="mb-1 text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
            temporal signals
          </p>
          <SignalRow
            label="flood probability (3-day MA)"
            value={pct(trends.flood_prob_ma, 1)}
            rate={trends.flood_prob_rate}
          />
          <SignalRow
            label="landslide probability (3-day MA)"
            value={pct(trends.landslide_prob_ma, 1)}
            rate={trends.landslide_prob_rate}
          />
          <SignalRow
            label="rainfall (3-day MA)"
            value={num(trends.rainfall_ma, 1)}
            unit=" mm/day"
            rate={trends.rainfall_rate}
          />
          <SignalRow
            label="soil moisture (3-day MA)"
            value={num(trends.soil_moisture_ma, 3)}
            rate={trends.soil_moisture_rate}
          />
        </div>

        {/* TCDL rules */}
        <div>
          <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
            tcdl rules ({fired.size} of {TCDL_RULES.length} triggered)
          </p>
          <ul className="space-y-1">
            {TCDL_RULES.map((rule) => {
              const on = fired.has(rule);
              return (
                <li
                  key={rule}
                  className="flex items-start gap-2 rounded-[3px] px-2 py-1"
                  style={on ? { background: "rgba(255,210,74,0.08)" } : undefined}
                  title={ruleDescriptions?.[rule] ?? ""}
                >
                  <span
                    className="text-[11px] font-bold"
                    style={{ color: on ? TERM.amber : "#1c3a22" }}
                  >
                    [{on ? "x" : " "}]
                  </span>
                  <span className="min-w-0 flex-1">
                    <span
                      className="text-[11px]"
                      style={{ color: on ? TERM.amber : "#3d6b47" }}
                    >
                      {rule}
                    </span>
                    <span className="ml-1.5 text-[10px] text-[#1c7a3c]">
                      {ruleIsBaseline(rule) ? "baseline" : "coupling"}
                    </span>
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="mt-2 text-[10px] leading-relaxed text-[#3d6b47]">
            Rules R1/R2 replicate the single-model baselines; R3–R7 are the coupling
            rules. Rule conditions and thresholds are defined by TCDL V1.0 in the ML
            layer and are not evaluated in this interface.
          </p>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------- monitoring-point list */

export function TermPointSelector({
  locations,
  activeId,
  onSelect,
}: {
  locations: CurrentLocation[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {locations.map((l) => {
        const on = l.location_id === activeId;
        const warn = l.tcdl.warning_status === "Warning";
        return (
          <button
            key={l.location_id}
            onClick={() => onSelect(l.location_id)}
            className={`crt-chip px-2.5 py-1 text-[11px] ${on ? "crt-chip-on" : ""}`}
          >
            {l.environment.latitude.toFixed(3)},{" "}
            {l.environment.longitude.toFixed(3)}
            {warn && (
              <span
                className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle"
                style={{ background: TERM.amber, boxShadow: `0 0 6px ${TERM.amber}` }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

/** Terminal replacement for the map's built-in (light) legend. */
export function TermMapLegend() {
  const levels: HazardLevel[] = ["normal", "watch", "warning", "critical"];
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      <span className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
        hazard level
      </span>
      {levels.map((lvl) => (
        <span key={lvl} className="flex items-center gap-1.5 text-[11px] text-[#5f8d68]">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{
              background: LEVEL_META[lvl].hex,
              boxShadow: `0 0 6px ${LEVEL_META[lvl].hex}aa`,
            }}
          />
          {LEVEL_META[lvl].label}
        </span>
      ))}
    </div>
  );
}
