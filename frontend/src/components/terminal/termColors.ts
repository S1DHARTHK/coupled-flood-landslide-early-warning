/**
 * Severity colours for the terminal theme.
 *
 * The warning TYPE is carried by its text label; colour carries severity only,
 * so the palette stays phosphor green -> amber -> red instead of inventing a
 * colour per hazard. These map backend values onto colours and nothing else —
 * no threshold or decision is made here.
 */

export const TERM = {
  phosphor: "#39ff7a",
  dim: "#2bbf5c",
  faint: "#1c7a3c",
  sage: "#5f8d68",
  ink: "#eafff1",
  amber: "#ffd24a",
  red: "#ff5f56",
  cyan: "#5cf6ff",
} as const;

export function warningColor(warningType: string): string {
  switch (warningType) {
    case "Coupled Hazard Warning":
      return TERM.red;
    case "Flood Warning":
    case "Landslide Warning":
      return TERM.amber;
    default:
      return TERM.phosphor;
  }
}

export function levelColor(level: string): string {
  switch (level) {
    case "critical":
      return TERM.red;
    case "warning":
      return TERM.amber;
    case "watch":
      return "#c9a13b";
    default:
      return TERM.phosphor;
  }
}
