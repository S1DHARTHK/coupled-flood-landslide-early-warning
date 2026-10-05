/**
 * Severity colours for the terminal theme.
 *
 * The warning TYPE is carried by its text label; colour carries severity only,
 * so the palette stays phosphor green -> amber -> red instead of inventing a
 * colour per hazard. These map backend values onto colours and nothing else —
 * no threshold or decision is made here.
 */

export const TERM = {
  phosphor: "#8ef075",
  dim: "#76cc5c",
  faint: "#538f3d",
  sage: "#8ea97e",
  ink: "#f5fbe9",
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
