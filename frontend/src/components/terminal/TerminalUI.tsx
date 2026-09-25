/**
 * Shared CRT-terminal primitives.
 *
 * Presentation only. Nothing here fetches, derives or thresholds a hazard —
 * these are the window frame, prompts, panels, meters and state blocks that
 * the terminal-themed pages (landing + early-warning dashboard) are built
 * from. The rest of the app keeps its original light theme.
 *
 * Palette: phosphor #39ff7a, dim #2bbf5c, faint #1c7a3c, sage body #5f8d68,
 * ink #eafff1, single amber #ffd24a for live status. Red #ff5f56 appears only
 * where severity genuinely demands it (a coupled/critical warning).
 */

import type { ReactNode } from "react";
import { Link, useLocation } from "react-router";

import { TERM } from "./termColors";

const NAV_LINKS = [
  { label: "~/dashboard", to: "/early-warning" },
  { label: "~/warnings", to: "/warnings" },
  { label: "~/analysis", to: "/analysis" },
  { label: "~/model", to: "/model-performance" },
];

/* ------------------------------------------------------------------ frame */

/**
 * The fake terminal window that frames a whole page: traffic lights, a mono
 * path label, nav links and a status cluster in the title bar.
 */
export function TerminalWindow({
  path,
  status,
  maxWidthClass = "max-w-[1400px]",
  children,
}: {
  path: string;
  status?: ReactNode;
  /** Landing runs narrower than the dashboard; both keep the same frame. */
  maxWidthClass?: string;
  children: ReactNode;
}) {
  const { pathname } = useLocation();
  return (
    <div className="crt-page min-h-screen px-3 py-4 sm:px-6 sm:py-8">
      <div className="crt-scanlines" />
      <div className="crt-vignette" />

      <div className={`crt-window relative z-10 mx-auto w-full overflow-hidden ${maxWidthClass}`}>
        <div className="crt-titlebar flex flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-5">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-[#ff5f56]" />
            <span className="h-3 w-3 rounded-full bg-[#ffbd2e]" />
            <span className="h-3 w-3 rounded-full bg-[#27c93f]" />
          </div>

          <p className="text-[13px]">
            <span className="text-[#eafff1]">ews</span>
            <span className="text-[#5f8d68]">@kerala: </span>
            <span className="text-[#eafff1]">{path}</span>
          </p>

          <div className="ml-auto flex flex-wrap items-center gap-x-5 gap-y-2">
            <nav className="hidden items-center gap-5 md:flex">
              <Link
                to="/"
                className="text-[13px] text-[#5f8d68] transition-colors hover:text-[#39ff7a]"
              >
                ~/home
              </Link>
              {NAV_LINKS.map((link) => {
                const on = pathname === link.to;
                return (
                  <Link
                    key={link.to}
                    to={link.to}
                    className={`text-[13px] transition-colors ${
                      on
                        ? "text-[#39ff7a] crt-glow-soft"
                        : "text-[#5f8d68] hover:text-[#39ff7a]"
                    }`}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </nav>
            {status}
          </div>
        </div>

        {children}
      </div>
    </div>
  );
}

/** `ews~/early-warning $ command` line that introduces every section. */
export function Prompt({
  command,
  comment,
  cwd = "~/early-warning",
}: {
  command: string;
  comment?: string;
  cwd?: string;
}) {
  return (
    <p className="text-[13px]">
      <span className="text-[#5cf6ff]">ews</span>
      <span className="text-[#5f8d68]">{cwd}</span>{" "}
      <span className="text-[#39ff7a] crt-glow-soft">$</span>{" "}
      <span className="text-[#eafff1]">{command}</span>
      {comment && <span className="text-[#1c7a3c]">{`  # ${comment}`}</span>}
    </p>
  );
}

/** Section heading below a prompt: a `>` label plus a sage description. */
export function SectionHead({
  label,
  title,
  desc,
  right,
}: {
  label: string;
  title: string;
  desc?: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-4 mt-4 flex flex-wrap items-end justify-between gap-3">
      <div>
        <p className="text-[11px] uppercase tracking-[0.22em] text-[#2bbf5c]">
          &gt; {label}
        </p>
        <h2 className="mt-1 text-[17px] font-semibold text-[#eafff1] sm:text-[19px]">
          {title}
        </h2>
        {desc && (
          <p className="mt-1 max-w-3xl text-[12px] leading-relaxed text-[#5f8d68]">
            {desc}
          </p>
        )}
      </div>
      {right}
    </div>
  );
}

/** A framed console panel with an optional title strip. */
export function Panel({
  title,
  desc,
  right,
  className = "",
  children,
}: {
  title?: string;
  desc?: string;
  right?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={`crt-panel crt-panel-hover ${className}`}>
      {(title || right) && (
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#0f2a12] px-4 py-3">
          <div>
            {title && (
              <p className="text-[13px] font-semibold text-[#39ff7a] crt-glow-soft">
                {title}
              </p>
            )}
            {desc && (
              <p className="mt-1 max-w-2xl text-[11px] leading-relaxed text-[#5f8d68]">
                {desc}
              </p>
            )}
          </div>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

/** Section divider. */
export function Rule() {
  return <div className="crt-rule my-8" />;
}

/* ------------------------------------------------------------- primitives */

/** ASCII-style bar meter. `value` is 0..1. */
export function Meter({
  value,
  color = TERM.phosphor,
}: {
  value: number | null | undefined;
  color?: string;
}) {
  const v = value === null || value === undefined || Number.isNaN(value) ? 0 : value;
  const w = Math.max(0, Math.min(1, v)) * 100;
  return (
    <div className="crt-meter">
      <div
        className="crt-meter-fill"
        style={{
          width: `${w}%`,
          background: `linear-gradient(90deg, ${color}66, ${color})`,
          boxShadow: `0 0 12px ${color}80`,
        }}
      />
    </div>
  );
}

/** Small outline tag. */
export function Tag({
  children,
  color,
}: {
  children: ReactNode;
  color?: string;
}) {
  return (
    <span
      className="crt-chip px-2 py-0.5 text-[10px]"
      style={color ? { color, borderColor: `${color}66` } : undefined}
    >
      {children}
    </span>
  );
}

/** Labelled form control, console style. */
export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-[0.18em] text-[#2bbf5c]">
        {label}
      </span>
      {children}
    </label>
  );
}

/** Toggle button used by the model / mode switches. */
export function ChipButton({
  on,
  onClick,
  children,
}: {
  on: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`crt-chip px-3 py-1.5 text-[12px] ${on ? "crt-chip-on" : ""}`}
    >
      {children}
    </button>
  );
}

/* ----------------------------------------------------------- state blocks */

export function TermLoading({
  label = "loading…",
  height = 180,
}: {
  label?: string;
  height?: number;
}) {
  return (
    <div
      className="crt-inset flex items-center justify-center px-4 text-[12px] text-[#2bbf5c]"
      style={{ minHeight: height }}
      role="status"
      aria-live="polite"
    >
      <span>
        <span className="text-[#39ff7a] crt-glow-soft">$</span> {label}{" "}
        <span className="crt-caret align-middle" />
      </span>
    </div>
  );
}

export function TermError({
  message,
  onRetry,
  height = 180,
}: {
  message: string;
  onRetry?: () => void;
  height?: number;
}) {
  return (
    <div
      className="crt-inset flex flex-col items-center justify-center gap-2 px-5 py-4 text-center"
      style={{ minHeight: height, borderColor: "rgba(255,95,86,0.45)" }}
      role="alert"
    >
      <p className="text-[12px] font-semibold text-[#ff5f56]">
        stderr: could not load data
      </p>
      <p className="max-w-lg break-words text-[11px] text-[#8f6a68]">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="crt-chip mt-1 px-3 py-1 text-[11px]">
          ./retry
        </button>
      )}
    </div>
  );
}

export function TermEmpty({
  message = "no records",
  height = 160,
}: {
  message?: string;
  height?: number;
}) {
  return (
    <div
      className="crt-inset flex items-center justify-center px-4 text-center text-[12px] text-[#3d6b47]"
      style={{ minHeight: height }}
    >
      {message}
    </div>
  );
}

/**
 * Data-status notice. This must stay visible: the dashboard must never read as
 * a live operational Kerala warning. On real data it shows the backend's own
 * `data_notice` (the caveat the served pipeline carries) when one is sent.
 */
export function TermSyntheticBanner({
  dataMode,
  sourceMode,
  fallbackReason,
  notice,
}: {
  dataMode?: string;
  sourceMode?: "live" | "fixture" | null;
  fallbackReason?: string;
  notice?: string | null;
}) {
  // No response yet (loading or failed): claim nothing about the data.
  const unknown = !dataMode;
  const isSynthetic = dataMode === "synthetic";
  const isReal = dataMode === "real";
  const accent = isSynthetic || unknown ? TERM.amber : TERM.phosphor;
  return (
    <div
      className="rounded-[4px] px-4 py-3"
      style={{
        border: `1px solid ${accent}55`,
        background: `${accent}0d`,
      }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span
          className="crt-glow-amber text-[11px] font-bold uppercase tracking-wide"
          style={{ color: accent }}
        >
          [!]{" "}
          {unknown
            ? "no data loaded"
            : isSynthetic
              ? "synthetic development data"
              : isReal
                ? "real historical data"
                : "live data"}
        </span>
        <p className="text-[11px] leading-relaxed text-[#8a8460]">
          {unknown
            ? "No response has been received for this view yet, so its data source cannot be stated. Nothing below is shown until one arrives."
            : isSynthetic
            ? "All probabilities, warnings and metrics shown are simulated outputs used for development and testing. They are NOT real-world Kerala predictions and must not be acted upon."
            : isReal
              ? (notice ??
                "Real XGBoost models and TCDL V1.0 on the real Kerala district-day dataset (2012–2024). Research outputs, not operational warnings; probabilities are not calibrated.")
              : "Connected to live data source."}
        </p>
      </div>
      {sourceMode && (
        <p className="mt-1.5 text-[11px] text-[#6f6a4a]">
          source={" "}
          <span style={{ color: accent }}>
            {sourceMode === "live"
              ? "live FastAPI backend"
              : "recorded API fixture (backend unreachable)"}
          </span>
          {sourceMode === "fixture" && fallbackReason ? ` — ${fallbackReason}` : ""}
        </p>
      )}
    </div>
  );
}
