/**
 * Warning history page — filterable table of TCDL warning records, rendered
 * as a CRT terminal session.
 *
 * Records come from the backend; no warning is derived in the frontend. The
 * filters below only narrow what is displayed, they never change a decision.
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
import { getWarnings } from "../../services/api";
import type { WarningType } from "../../types/hazard";

const WARNING_TYPES: WarningType[] = [
  "Flood Warning",
  "Landslide Warning",
  "Coupled Hazard Warning",
];

export default function WarningsPage() {
  const warnings = useApi(() => getWarnings({ limit: 400 }), []);
  const [typeFilter, setTypeFilter] = useState<WarningType | "all">("all");
  const [locationFilter, setLocationFilter] = useState<string>("all");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const records = useMemo(() => warnings.data?.records ?? [], [warnings.data]);

  const locationOptions = useMemo(
    () => Array.from(new Set(records.map((r) => r.location_id))).sort(),
    [records]
  );

  const filtered = useMemo(
    () =>
      records.filter((r) => {
        if (typeFilter !== "all" && r.warning_type !== typeFilter) return false;
        if (locationFilter !== "all" && r.location_id !== locationFilter) return false;
        if (startDate && r.date < startDate) return false;
        if (endDate && r.date > endDate) return false;
        return true;
      }),
    [records, typeFilter, locationFilter, startDate, endDate]
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
          <span className="flex items-center gap-2 text-[13px] text-[#5f8d68]">
            <span className="text-[#39ff7a]">{records.length}</span> records loaded
          </span>
        }
      >
        <div className="px-4 py-8 sm:px-8 sm:py-10">
          <Prompt
            command="./warnings --history --limit 400"
            comment="TCDL decision log"
            cwd="~/warnings"
          />
          <h1 className="mt-4 text-[26px] font-bold leading-tight tracking-tight text-[#eafff1] sm:text-[30px] [text-shadow:0_0_14px_rgba(57,255,122,0.35)]">
            Warning History
          </h1>
          <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-[#5f8d68] sm:text-[13px]">
            Every row is a TCDL decision produced by the ML pipeline. Timestamps are
            daily — the dataset carries no time of day.
          </p>

          <div className="mt-5">
            <TermSyntheticBanner
              dataMode={warnings.data?.data_mode}
              sourceMode={warnings.sourceMode}
              fallbackReason={warnings.fallbackReason}
            />
          </div>

          {/* ---------------------------------------- summary counts */}
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div className="crt-panel crt-panel-hover p-4">
              <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
                total_warning_records
              </p>
              <p
                className="mt-2 text-[26px] font-bold leading-none text-[#eafff1]"
                style={{ textShadow: "0 0 12px rgba(57,255,122,0.35)" }}
              >
                {records.length}
              </p>
            </div>
            {WARNING_TYPES.map((t) => {
              const color = warningColor(t);
              return (
                <div key={t} className="crt-panel crt-panel-hover p-4">
                  <p className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
                    {t.toLowerCase().replace(/ /g, "_")}
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
              comment={`${filtered.length} of ${records.length} rows`}
              cwd="~/warnings"
            />
            <SectionHead
              label="warning records"
              title="Filter the decision log"
              desc="Filters narrow what is shown only — they never re-evaluate a warning."
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

                <Field label="location">
                  <select
                    className="crt-select"
                    value={locationFilter}
                    onChange={(e) => setLocationFilter(e.target.value)}
                  >
                    <option value="all">all locations</option>
                    {locationOptions.map((l) => (
                      <option key={l} value={l}>
                        {l.replace("_", ", ")}
                      </option>
                    ))}
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
                  }}
                  className="crt-chip px-3 py-1.5 text-[12px]"
                >
                  --reset
                </button>

                <span className="ml-auto text-[11px] text-[#3d6b47]">
                  showing{" "}
                  <span className="text-[#39ff7a]">{filtered.length}</span> of{" "}
                  {records.length}
                </span>
              </div>

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
                            "latitude",
                            "longitude",
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
                              <td>{r.latitude.toFixed(3)}</td>
                              <td>{r.longitude.toFixed(3)}</td>
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
                                      className="rounded-[2px] border border-[#1f4d1f] px-1.5 py-0.5 text-[10px] text-[#2bbf5c]"
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
                                  <span className="text-[#1c3a22]">—</span>
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

              <p className="mt-3 text-[10px] leading-relaxed text-[#3d6b47]">
                “actual” shows whether a hazard was recorded on that day at that point,
                for reference against the warning. Warning timestamps have daily
                resolution — the dataset carries no time of day.
              </p>
            </Panel>
          </div>

          {/* ---------------------------------------------- footer */}
          <div className="crt-rule mt-10" />
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 text-[11px]">
            <p className="text-[#5f8d68]">
              <span className="text-[#39ff7a] crt-glow-soft">$</span> echo
              &quot;synthetic development data · not a real-world Kerala
              warning&quot; <span className="crt-caret align-middle" />
            </p>
            <p className="text-[#1c7a3c]">
              tcdl@v1.0 · records: {records.length} · filtered: {filtered.length}
            </p>
          </div>
        </div>
      </TerminalWindow>
    </>
  );
}
