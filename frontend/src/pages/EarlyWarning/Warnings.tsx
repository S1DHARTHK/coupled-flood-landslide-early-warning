/**
 * Warning history page — filterable table of TCDL warning records, rendered
 * as a CRT terminal session.
 *
 * Records come from the backend; no warning is derived in the frontend. The
 * district, date and period filters are applied by the backend (the real
 * decision log holds tens of thousands of district-days, far more than one
 * page should load); the warning-type filter narrows the loaded rows. No
 * filter ever changes a decision.
 */

import { useMemo, useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import {
  Field,
  Panel,
  Prompt,
  SectionHead,
  TerminalWindow,
  TermEmpty,
  TermError,
  TermLoading,
  TermSyntheticBanner,
} from "../../components/terminal/TerminalUI";
import { TERM, warningColor } from "../../components/terminal/termColors";
import { pct } from "../../components/hazard/hazardUtils";
import { useApi } from "../../hooks/useApi";
import { getLocations, getWarnings } from "../../services/api";
import type { WarningType } from "../../types/hazard";

const WARNING_TYPES: WarningType[] = [
  "Flood Warning",
  "Landslide Warning",
  "Coupled Hazard Warning",
];

/** Most recent rows loaded per query; the header says when more matched. */
const PAGE_LIMIT = 2000;
type SplitFilter = "all" | "train" | "validation" | "test";

export default function WarningsPage() {
  const [typeFilter, setTypeFilter] = useState<WarningType | "all">("all");
  const [locationFilter, setLocationFilter] = useState<string>("all");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [split, setSplit] = useState<SplitFilter>("all");

  const warnings = useApi(
    () =>
      getWarnings({
        locationId: locationFilter === "all" ? undefined : locationFilter,
        start: startDate || undefined,
        end: endDate || undefined,
        split: split === "all" ? undefined : split,
        limit: PAGE_LIMIT,
      }),
    [locationFilter, startDate, endDate, split]
  );
  const locations = useApi(() => getLocations(), []);

  const records = useMemo(() => warnings.data?.records ?? [], [warnings.data]);
  const nMatching = warnings.data?.n_matching ?? records.length;
  const truncated = Boolean(warnings.data?.truncated);
  const isReal = warnings.data?.data_mode === "real";
  const hasDistrict = records.some((r) => r.district);

  const locationOptions = useMemo(
    () =>
      [...(locations.data?.locations ?? [])]
        .map((l) => ({
          id: l.location_id,
          label: l.district ?? `${l.latitude.toFixed(3)}, ${l.longitude.toFixed(3)}`,
        }))
        .sort((a, b) => a.label.localeCompare(b.label)),
    [locations.data]
  );

  // Newest first: the backend returns the most recent rows in date order.
  const filtered = useMemo(
    () =>
      records
        .filter((r) => typeFilter === "all" || r.warning_type === typeFilter)
        .reverse(),
    [records, typeFilter]
  );

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    records.forEach((r) => (c[r.warning_type] = (c[r.warning_type] ?? 0) + 1));
    return c;
  }, [records]);

  return (
    <>
      <PageMeta
        title="Warning History | Kerala Flood–Landslide Early Warning"
        description="Historical TCDL warning records with filtering by type, location and date."
      />

      <TerminalWindow
        path="~/warnings"
        status={
          <span className="flex items-center gap-2 text-[13px] text-[#8ea97e]">
            <span className="text-[#8ef075]">{nMatching}</span> matching ·{" "}
            {records.length} loaded
          </span>
        }
      >
        <div className="px-4 py-8 sm:px-8 sm:py-10">
          <Prompt
            command={`./warnings --history --limit ${PAGE_LIMIT}`}
            comment="TCDL decision log"
            cwd="~/warnings"
          />
          <h1 className="mt-4 text-[26px] font-bold leading-tight tracking-tight text-[#f5fbe9] sm:text-[30px] [text-shadow:0_0_14px_rgba(142, 240, 117,0.35)]">
            Warning History
          </h1>
          <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-[#8ea97e] sm:text-[13px]">
            Every row is a TCDL decision produced by the ML pipeline
            {isReal ? " on the real Kerala district-day data (2012–2024)" : ""}. Timestamps
            are daily — the dataset carries no time of day.
          </p>

          <div className="mt-5">
            <TermSyntheticBanner
              dataMode={warnings.data?.data_mode}
              sourceMode={warnings.sourceMode}
              fallbackReason={warnings.fallbackReason}
              notice={warnings.data?.data_notice}
            />
          </div>

          {/* ---------------------------------------- summary counts */}
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div className="crt-panel crt-panel-hover p-4">
              <p className="text-[10px] uppercase tracking-[0.18em] text-[#76cc5c]">
                matching_warning_days
              </p>
              <p
                className="mt-2 text-[26px] font-bold leading-none text-[#f5fbe9]"
                style={{ textShadow: "0 0 12px rgba(142, 240, 117,0.35)" }}
              >
                {nMatching}
              </p>
              {warnings.data?.n_location_days_in_range !== undefined && (
                <p className="mt-2 text-[11px] text-[#8ea97e]">
                  of {warnings.data.n_location_days_in_range}{" "}
                  {isReal ? "district" : "location"}-days in range
                </p>
              )}
            </div>
            {WARNING_TYPES.map((t) => {
              const color = warningColor(t);
              return (
                <div key={t} className="crt-panel crt-panel-hover p-4">
                  <p className="text-[10px] uppercase tracking-[0.18em] text-[#76cc5c]">
                    {t.toLowerCase().replace(/ /g, "_")}
                    {truncated ? " (loaded)" : ""}
                  </p>
                  <p
                    className="mt-2 text-[26px] font-bold leading-none"
                    style={{ color, textShadow: `0 0 12px ${color}55` }}
                  >
                    {counts[t] ?? 0}
                  </p>
                </div>
              );
            })}
          </div>

          {/* --------------------------------------- records + filters */}
          <div className="mt-8">
            <Prompt
              command="grep warnings.log"
              comment={`${filtered.length} of ${records.length} loaded rows`}
              cwd="~/warnings"
            />
            <SectionHead
              label="warning records"
              title="Filter the decision log"
              desc="District, dates and period are queried from the backend; warning type narrows the loaded rows. No filter re-evaluates a warning."
            />

            <Panel>
              <div className="flex flex-wrap items-end gap-3">
                <Field label="warning type">
                  <select
                    className="crt-select"
                    value={typeFilter}
                    onChange={(e) =>
                      setTypeFilter(e.target.value as WarningType | "all")
                    }
                  >
                    <option value="all">all types</option>
                    {WARNING_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </Field>

                <Field label={isReal ? "district" : "location"}>
                  <select
                    className="crt-select"
                    value={locationFilter}
                    onChange={(e) => setLocationFilter(e.target.value)}
                  >
                    <option value="all">{isReal ? "all districts" : "all locations"}</option>
                    {locationOptions.map((l) => (
                      <option key={l.id} value={l.id}>
                        {l.label}
                      </option>
                    ))}
                  </select>
                </Field>

                <Field label="period">
                  <select
                    className="crt-select"
                    value={split}
                    onChange={(e) => setSplit(e.target.value as SplitFilter)}
                  >
                    <option value="all">all periods</option>
                    <option value="train">train (seen by the models)</option>
                    <option value="validation">validation</option>
                    <option value="test">test (unseen, evaluated)</option>
                  </select>
                </Field>

                <Field label="from">
                  <input
                    type="date"
                    className="crt-input"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </Field>

                <Field label="to">
                  <input
                    type="date"
                    className="crt-input"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                  />
                </Field>

                <button
                  onClick={() => {
                    setTypeFilter("all");
                    setLocationFilter("all");
                    setStartDate("");
                    setEndDate("");
                    setSplit("all");
                  }}
                  className="crt-chip px-3 py-1.5 text-[12px]"
                >
                  --reset
                </button>

                <span className="ml-auto text-[11px] text-[#6a8958]">
                  showing{" "}
                  <span className="text-[#8ef075]">{filtered.length}</span> of{" "}
                  {records.length} loaded
                </span>
              </div>

              {truncated && (
                <p className="mt-3 text-[11px] leading-relaxed" style={{ color: TERM.amber }}>
                  [!] {nMatching} warning days match; the most recent {records.length} are
                  loaded. Pick a {isReal ? "district" : "location"} or narrow the dates to
                  see older records.
                </p>
              )}

              <div className="mt-4">
                {warnings.loading ? (
                  <TermLoading label="loading warning history…" height={260} />
                ) : warnings.error ? (
                  <TermError
                    message={warnings.error}
                    onRetry={warnings.reload}
                    height={260}
                  />
                ) : filtered.length === 0 ? (
                  <TermEmpty
                    message="no warnings match the selected filters"
                    height={220}
                  />
                ) : (
                  <div className="crt-inset crt-scroll max-h-[560px] overflow-auto">
                    <table className="crt-table crt-table-sticky">
                      <thead>
                        <tr>
                          {[
                            "date",
                            ...(hasDistrict ? ["district"] : ["latitude", "longitude"]),
                            "warning_type",
                            "flood",
                            "landslide",
                            "coupled",
                            "triggered_rules",
                            "actual",
                          ].map((h) => (
                            <th key={h}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filtered.map((r, i) => {
                          const color = warningColor(r.warning_type);
                          const hit =
                            r.actual.flood === 1 || r.actual.landslide === 1;
                          return (
                            <tr key={`${r.location_id}-${r.date}-${i}`}>
                              <td style={{ color: TERM.ink }}>{r.date}</td>
                              {hasDistrict ? (
                                <td>{r.district ?? "—"}</td>
                              ) : (
                                <>
                                  <td>{r.latitude.toFixed(3)}</td>
                                  <td>{r.longitude.toFixed(3)}</td>
                                </>
                              )}
                              <td>
                                <span
                                  className="rounded-[3px] px-2 py-0.5 text-[10px]"
                                  style={{
                                    color,
                                    border: `1px solid ${color}55`,
                                    background: `${color}0d`,
                                  }}
                                >
                                  {r.warning_type}
                                </span>
                              </td>
                              <td>{pct(r.flood_probability, 0)}</td>
                              <td>{pct(r.landslide_probability, 0)}</td>
                              <td>{pct(r.coupled_probability, 0)}</td>
                              <td>
                                <span className="flex flex-wrap gap-1">
                                  {r.triggered_rules.map((rule) => (
                                    <span
                                      key={rule}
                                      className="rounded-[2px] border border-[#3d5d2d] px-1.5 py-0.5 text-[10px] text-[#76cc5c]"
                                    >
                                      {rule}
                                    </span>
                                  ))}
                                </span>
                              </td>
                              <td>
                                {hit ? (
                                  <span style={{ color: TERM.red }}>
                                    {r.actual.flood ? "flood " : ""}
                                    {r.actual.landslide ? "landslide" : ""}
                                  </span>
                                ) : (
                                  <span className="text-[#304625]">—</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <p className="mt-3 text-[10px] leading-relaxed text-[#6a8958]">
                “actual” shows whether a hazard was recorded on that day
                {isReal ? " in that district (0 = not reported, not proven absence)" : " at that point"},
                for reference against the warning. Warning timestamps have daily
                resolution — the dataset carries no time of day.
              </p>
            </Panel>
          </div>

          {/* ---------------------------------------------- footer */}
          <div className="crt-rule mt-10" />
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 text-[11px]">
            <p className="text-[#8ea97e]">
              <span className="text-[#8ef075] crt-glow-soft">$</span> echo
              &quot;
              {isReal
                ? "real historical data · research output, not an operational warning"
                : "synthetic development data · not a real-world Kerala warning"}
              &quot; <span className="crt-caret align-middle" />
            </p>
            <p className="text-[#538f3d]">
              tcdl@v1.0 · matching: {nMatching} · loaded: {records.length} · shown:{" "}
              {filtered.length}
            </p>
          </div>
        </div>
      </TerminalWindow>
    </>
  );
}
