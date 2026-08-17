/**
 * Presentation helpers for hazard display.
 *
 * IMPORTANT SCOPE LIMIT
 * ---------------------
 * Nothing here computes a hazard. These functions only map values the backend
 * already produced onto colours, labels and formats. The warning decision, the
 * probabilities and the TCDL rules all come from the ML/TCDL layer.
 *
 * The one derived quantity is the marker `level` (normal/watch/warning/
 * critical), which is a VISUAL banding of the coupled probability plus the
 * backend's own warning status. It is a display convention, not a new
 * threshold on the hazard decision, and it never changes whether a warning
 * exists.
 */

import type {
  CurrentLocation,
  HazardLevel,
  MapLocation,
  TcdlRuleId,
  WarningType,
} from "../../types/hazard";

/** Visual banding thresholds for map markers. Display only. */
export const LEVEL_BANDS = {
  critical: 0.75,
  warning: 0.5,
  watch: 0.3,
} as const;

export const LEVEL_META: Record<
  HazardLevel,
  { label: string; hex: string; badge: string; ring: string }
> = {
  normal: {
    label: "Normal",
    hex: "#12b76a",
    badge: "bg-success-50 text-success-700 dark:bg-success-500/15 dark:text-success-400",
    ring: "ring-success-500/30",
  },
  watch: {
    label: "Watch",
    hex: "#eab308",
    badge: "bg-warning-50 text-warning-700 dark:bg-warning-500/15 dark:text-warning-400",
    ring: "ring-warning-500/30",
  },
  warning: {
    label: "Warning",
    hex: "#f79009",
    badge: "bg-orange-50 text-orange-700 dark:bg-orange-500/15 dark:text-orange-400",
    ring: "ring-orange-500/30",
  },
  critical: {
    label: "High / Critical",
    hex: "#f04438",
    badge: "bg-error-50 text-error-700 dark:bg-error-500/15 dark:text-error-400",
    ring: "ring-error-500/30",
  },
};

export const WARNING_TYPE_META: Record<
  WarningType,
  { color: string; badge: string; dot: string }
> = {
  "No Warning": {
    color: "#12b76a",
    badge: "bg-success-50 text-success-700 dark:bg-success-500/15 dark:text-success-400",
    dot: "bg-success-500",
  },
  "Flood Warning": {
    color: "#2e90fa",
    badge:
      "bg-blue-light-50 text-blue-light-700 dark:bg-blue-light-500/15 dark:text-blue-light-400",
    dot: "bg-blue-light-500",
  },
  "Landslide Warning": {
    color: "#f79009",
    badge: "bg-orange-50 text-orange-700 dark:bg-orange-500/15 dark:text-orange-400",
    dot: "bg-orange-500",
  },
  "Coupled Hazard Warning": {
    color: "#f04438",
    badge: "bg-error-50 text-error-700 dark:bg-error-500/15 dark:text-error-400",
    dot: "bg-error-500",
  },
};

/** Visual band for a marker, from backend-supplied values only. */
export function hazardLevel(
  coupled: number | null,
  warningStatus: string
): HazardLevel {
  if (coupled === null || Number.isNaN(coupled)) {
    return warningStatus === "Warning" ? "warning" : "normal";
  }
  if (coupled >= LEVEL_BANDS.critical) return "critical";
  if (coupled >= LEVEL_BANDS.warning) return "warning";
  if (coupled >= LEVEL_BANDS.watch) return "watch";
  return "normal";
}

export function toMapLocation(loc: CurrentLocation): MapLocation {
  const coupled = loc.trends?.coupled_probability ?? null;
  return {
    location_id: loc.location_id,
    latitude: loc.environment.latitude,
    longitude: loc.environment.longitude,
    // Passed through only if the backend supplied it. Never inferred.
    district: loc.district ?? loc.environment.district ?? null,
    floodProbability: loc.flood.probability,
    landslideProbability: loc.landslide.probability,
    coupledProbability: coupled,
    warningStatus: loc.tcdl.warning_status,
    warningType: loc.tcdl.warning_type,
    triggeredRules: loc.tcdl.triggered_rules ?? [],
    level: hazardLevel(coupled, loc.tcdl.warning_status),
    date: loc.environment.date,
  };
}

/** Percentage string, or an em dash when the backend sent null. */
export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function hours(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)} h`;
}

/** Direction arrow for a rate-of-change signal. */
export function trendArrow(rate: number | null | undefined): {
  glyph: string;
  cls: string;
  label: string;
} {
  if (rate === null || rate === undefined || Number.isNaN(rate)) {
    return { glyph: "—", cls: "text-gray-400", label: "no data" };
  }
  if (rate > 0) return { glyph: "▲", cls: "text-error-500", label: "rising" };
  if (rate < 0) return { glyph: "▼", cls: "text-success-500", label: "falling" };
  return { glyph: "▬", cls: "text-gray-400", label: "flat" };
}

export function formatCoord(lat: number, lon: number): string {
  return `${lat.toFixed(3)}°N, ${lon.toFixed(3)}°E`;
}

export function ruleIsBaseline(rule: TcdlRuleId): boolean {
  return rule === "R1_FLOOD_LEVEL" || rule === "R2_LANDSLIDE_LEVEL";
}

/** Friendly labels for the evaluated warning systems. */
export const SYSTEM_LABELS: Record<string, string> = {
  flood_only: "Flood-only",
  landslide_only: "Landslide-only",
  tcdl_coupled: "TCDL Coupled",
  tcdl_coupling_rules_only: "TCDL coupling rules only",
  always_warn_control: "Always-warn control",
};

export const SYSTEM_COLORS: Record<string, string> = {
  flood_only: "#2e90fa",
  landslide_only: "#f79009",
  tcdl_coupled: "#12b76a",
  tcdl_coupling_rules_only: "#7a5af8",
  always_warn_control: "#98a2b3",
};
