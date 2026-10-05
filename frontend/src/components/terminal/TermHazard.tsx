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
import { ChipButton, Field, Meter, TermEmpty } from "./TerminalUI";
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
      <p className="text-[10px] uppercase tracking-[0.18em] text-[#76cc5c]">
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
      <p className="mt-2 text-[11px] leading-relaxed text-[#8ea97e]">{sub}</p>
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
        sub={`${activeWarnings} of ${totalLocations} ${
          location?.district ? "districts" : "monitoring points"
        } under warning`}
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
    <div className="flex items-center justify-between border-b border-[#1e2e14] py-2 last:border-0">
      <span className="text-[11px] text-[#8ea97e]">{label}</span>
      <span className="flex items-center gap-2">
        <span className="text-[11px] text-[#dbe9ce]">
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
        <TermEmpty message="select a district to view its warning status" />
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
            <p className="text-[10px] uppercase tracking-[0.2em] text-[#76cc5c]">
              current tcdl decision
            </p>
            <p
              className="mt-1 text-[19px] font-bold leading-tight"
              style={{ color: wcolor, textShadow: `0 0 14px ${wcolor}55` }}
            >
              {tcdl.warning_type.toUpperCase()}
            </p>
            <p className="mt-1 text-[11px] text-[#8ea97e]">
              {location.district ? (
                <span className="text-[#dbe9ce]">{location.district} · </span>
              ) : null}
              {formatCoord(environment.latitude, environment.longitude)}
            </p>
          </div>
          <span
            className="shrink-0 rounded-[3px] px-2 py-0.5 text-[10px] font-bold uppercase"
            style={{ color: lcolor, border: `1px solid ${lcolor}66` }}
          >
            {LEVEL_META[level as HazardLevel].label}
          </span>
        </div>
        <p className="mt-2 text-[11px] text-[#6a8958]">
          {tcdl.warning_timestamp
            ? `warning timestamp: ${tcdl.warning_timestamp} (daily resolution)`
            : `no active warning · as of ${environment.date}`}
        </p>
        {location.recorded_labels && (
          <RecordedEvents
            labels={location.recorded_labels}
            split={location.split}
          />
        )}
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
              <p className="text-[10px] uppercase tracking-wide text-[#6a8958]">
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
          <p className="mb-1 text-[10px] uppercase tracking-[0.18em] text-[#76cc5c]">
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
          <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-[#76cc5c]">
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
                    style={{ color: on ? TERM.amber : "#304625" }}
                  >
                    [{on ? "x" : " "}]
                  </span>
                  <span className="min-w-0 flex-1">
                    <span
                      className="text-[11px]"
                      style={{ color: on ? TERM.amber : "#6a8958" }}
                    >
                      {rule}
                    </span>
                    <span className="ml-1.5 text-[10px] text-[#538f3d]">
                      {ruleIsBaseline(rule) ? "baseline" : "coupling"}
                    </span>
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="mt-2 text-[10px] leading-relaxed text-[#6a8958]">
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
            {l.district ??
              `${l.environment.latitude.toFixed(3)}, ${l.environment.longitude.toFixed(3)}`}
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

/**
 * What the dataset recorded for this district-day, shown beside the decision
 * so a replayed day can be checked against reality. 0 means "not reported",
 * not proven absence. `split` says whether the models saw the day in training.
 */
function RecordedEvents({
  labels,
  split,
}: {
  labels: { flood?: number; landslide?: number };
  split?: string;
}) {
  const hits = (["flood", "landslide"] as const).filter((h) => labels[h] === 1);
  return (
    <p className="mt-1 text-[11px] text-[#6a8958]">
      recorded event:{" "}
      <span style={{ color: hits.length ? TERM.red : TERM.faint }}>
        {hits.length ? hits.join(" + ") : "none reported"}
      </span>
      {split ? (
        <span title="train = seen by the models during training; test = unseen">
          {" "}
          · {split} period
        </span>
      ) : null}
    </p>
  );
}

/**
 * Notable days in the real dataset, offered as replay shortcuts. Each one has
 * recorded events in dataset/real_master_dataset.csv; a shortcut outside the served
 * date range is hidden rather than offered.
 */
const REPLAY_PRESETS: { date: string; label: string }[] = [
  { date: "2018-08-16", label: "2018 Kerala floods" },
  { date: "2019-08-08", label: "2019 Kavalappara / Puthumala" },
  { date: "2021-10-16", label: "2021 Kokkayar / Koottickal" },
  { date: "2024-07-30", label: "2024 Wayanad landslide" },
];

/**
 * History replay control. Picks the day the page shows; the backend serves
 * the pipeline outputs stored for that day. `value = null` means "latest".
 */
export function TermAsOfPicker({
  value,
  range,
  onChange,
}: {
  value: string | null;
  range?: [string, string];
  onChange: (date: string | null) => void;
}) {
  const presets = REPLAY_PRESETS.filter(
    (p) => !range || (p.date >= range[0] && p.date <= range[1])
  );
  return (
    <div className="flex flex-wrap items-end gap-3">
      <Field label="as of (replay a day)">
        <input
          type="date"
          className="crt-input"
          value={value ?? range?.[1] ?? ""}
          min={range?.[0]}
          max={range?.[1]}
          onChange={(e) => onChange(e.target.value || null)}
        />
      </Field>
      <div className="flex flex-wrap gap-2 pb-0.5">
        <ChipButton on={value === null} onClick={() => onChange(null)}>
          latest{range ? ` (${range[1]})` : ""}
        </ChipButton>
        {presets.map((p) => (
          <ChipButton key={p.date} on={value === p.date} onClick={() => onChange(p.date)}>
            {p.label}
          </ChipButton>
        ))}
      </div>
    </div>
  );
}

/** Terminal replacement for the map's built-in (light) legend. */
export function TermMapLegend() {
  const levels: HazardLevel[] = ["normal", "watch", "warning", "critical"];
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      <span className="text-[10px] uppercase tracking-[0.18em] text-[#76cc5c]">
        hazard level
      </span>
      {levels.map((lvl) => (
        <span key={lvl} className="flex items-center gap-1.5 text-[11px] text-[#8ea97e]">
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
