/**
 * Warning history page -- filterable table of TCDL warning records.
 * Records come from the backend; no warning is derived in the frontend.
 */

import { useMemo, useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import PageBreadcrumb from "../../components/common/PageBreadCrumb";
import ComponentCard from "../../components/common/ComponentCard";
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "../../components/ui/table";
import {
  EmptyBlock,
  ErrorBlock,
  LoadingBlock,
  SyntheticDataBanner,
} from "../../components/hazard/StateBlocks";
import { WARNING_TYPE_META, pct } from "../../components/hazard/hazardUtils";
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

  const records = warnings.data?.records ?? [];

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

  const selectCls =
    "rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-700 focus:border-brand-500 focus:outline-none dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200";

  return (
    <>
      <PageMeta
        title="Warning History | Kerala Flood–Landslide Early Warning"
        description="Historical TCDL warning records with filtering by type, location and date."
      />
      <PageBreadcrumb pageTitle="Warning History" />
      <SyntheticDataBanner
        dataMode={warnings.data?.data_mode}
        sourceMode={warnings.sourceMode}
        fallbackReason={warnings.fallbackReason}
      />

      {/* Summary counts */}
      <div className="mb-5 grid grid-cols-1 gap-4 sm:grid-cols-4">
        <div className="rounded-2xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-white/[0.03]">
          <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
            Total warning records
          </p>
          <p className="mt-1 text-2xl font-bold text-gray-800 dark:text-white/90">
            {records.length}
          </p>
        </div>
        {WARNING_TYPES.map((t) => (
          <div
            key={t}
            className="rounded-2xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-white/[0.03]"
          >
            <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
              {t}
            </p>
            <p
              className="mt-1 text-2xl font-bold"
              style={{ color: WARNING_TYPE_META[t].color }}
            >
              {counts[t] ?? 0}
            </p>
          </div>
        ))}
      </div>

      <ComponentCard
        title="Warning Records"
        desc="Every row is a TCDL decision produced by the ML pipeline. Timestamps are daily."
      >
        {/* Filters */}
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Warning type
            </span>
            <select
              className={selectCls}
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value as WarningType | "all")}
            >
              <option value="all">All types</option>
              {WARNING_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Location
            </span>
            <select
              className={selectCls}
              value={locationFilter}
              onChange={(e) => setLocationFilter(e.target.value)}
            >
              <option value="all">All locations</option>
              {locationOptions.map((l) => (
                <option key={l} value={l}>
                  {l.replace("_", ", ")}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              From
            </span>
            <input
              type="date"
              className={selectCls}
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              To
            </span>
            <input
              type="date"
              className={selectCls}
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </label>

          <button
            onClick={() => {
              setTypeFilter("all");
              setLocationFilter("all");
              setStartDate("");
              setEndDate("");
            }}
            className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-white/5"
          >
            Reset
          </button>

          <span className="ml-auto text-xs text-gray-500 dark:text-gray-400">
            Showing {filtered.length} of {records.length}
          </span>
        </div>

        {/* Table */}
        {warnings.loading ? (
          <LoadingBlock label="Loading warning history…" height={260} />
        ) : warnings.error ? (
          <ErrorBlock message={warnings.error} onRetry={warnings.reload} height={260} />
        ) : filtered.length === 0 ? (
          <EmptyBlock message="No warnings match the selected filters." height={220} />
        ) : (
          <div className="max-h-[560px] overflow-auto rounded-lg border border-gray-100 dark:border-gray-800">
            <Table>
              <TableHeader className="sticky top-0 z-10 border-b border-gray-100 bg-gray-50 dark:border-gray-800 dark:bg-gray-900">
                <TableRow>
                  {[
                    "Date",
                    "Latitude",
                    "Longitude",
                    "Warning Type",
                    "Flood",
                    "Landslide",
                    "Coupled",
                    "Triggered Rules",
                    "Actual",
                  ].map((h) => (
                    <TableCell
                      key={h}
                      isHeader
                      className="whitespace-nowrap px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400"
                    >
                      {h}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((r, i) => {
                  const meta = WARNING_TYPE_META[r.warning_type];
                  const hit = r.actual.flood === 1 || r.actual.landslide === 1;
                  return (
                    <TableRow
                      key={`${r.location_id}-${r.date}-${i}`}
                      className="border-b border-gray-50 hover:bg-gray-50 dark:border-gray-800 dark:hover:bg-white/[0.02]"
                    >
                      <TableCell className="whitespace-nowrap px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                        {r.date}
                      </TableCell>
                      <TableCell className="px-4 py-3 font-mono text-xs text-gray-600 dark:text-gray-400">
                        {r.latitude.toFixed(3)}
                      </TableCell>
                      <TableCell className="px-4 py-3 font-mono text-xs text-gray-600 dark:text-gray-400">
                        {r.longitude.toFixed(3)}
                      </TableCell>
                      <TableCell className="whitespace-nowrap px-4 py-3">
                        <span
                          className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${meta.badge}`}
                        >
                          {r.warning_type}
                        </span>
                      </TableCell>
                      <TableCell className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                        {pct(r.flood_probability, 0)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                        {pct(r.landslide_probability, 0)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                        {pct(r.coupled_probability, 0)}
                      </TableCell>
                      <TableCell className="px-4 py-3">
                        <div className="flex flex-wrap gap-1">
                          {r.triggered_rules.map((rule) => (
                            <span
                              key={rule}
                              className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[10px] text-gray-600 dark:bg-white/5 dark:text-gray-400"
                            >
                              {rule}
                            </span>
                          ))}
                        </div>
                      </TableCell>
                      <TableCell className="whitespace-nowrap px-4 py-3 text-xs">
                        {hit ? (
                          <span className="text-error-600 dark:text-error-400">
                            {r.actual.flood ? "flood " : ""}
                            {r.actual.landslide ? "landslide" : ""}
                          </span>
                        ) : (
                          <span className="text-gray-400 dark:text-gray-600">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}

        <p className="text-[11px] leading-relaxed text-gray-500 dark:text-gray-400">
          “Actual” shows whether a hazard was recorded on that day at that point, for
          reference against the warning. Warning timestamps have daily resolution —
          the dataset carries no time of day.
        </p>
      </ComponentCard>
    </>
  );
}
