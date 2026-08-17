/** Loading, error, empty and data-source states shared across every page. */

interface LoadingProps {
  label?: string;
  height?: number;
}

export function LoadingBlock({ label = "Loading…", height = 180 }: LoadingProps) {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-white/[0.02]"
      style={{ minHeight: height }}
      role="status"
      aria-live="polite"
    >
      <span className="h-6 w-6 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
      <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
    </div>
  );
}

export function ErrorBlock({
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
      className="flex flex-col items-center justify-center gap-3 rounded-xl border border-error-200 bg-error-50 p-6 text-center dark:border-error-500/30 dark:bg-error-500/10"
      style={{ minHeight: height }}
      role="alert"
    >
      <p className="text-sm font-semibold text-error-700 dark:text-error-400">
        Could not load data
      </p>
      <p className="max-w-md text-xs text-error-600 dark:text-error-300">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="rounded-lg bg-error-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-error-600"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyBlock({
  message = "No records match the current filters.",
  height = 160,
}: {
  message?: string;
  height?: number;
}) {
  return (
    <div
      className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-white/[0.02]"
      style={{ minHeight: height }}
    >
      <p className="text-sm text-gray-500 dark:text-gray-400">{message}</p>
    </div>
  );
}

/**
 * Persistent banner stating that everything on screen comes from synthetic
 * development data, and whether it arrived from the live API or the recorded
 * offline fixture. This must stay visible -- the dashboard must never read as
 * a live real-world Kerala warning.
 */
export function SyntheticDataBanner({
  dataMode = "synthetic",
  sourceMode,
  fallbackReason,
}: {
  dataMode?: string;
  sourceMode?: "live" | "fixture" | null;
  fallbackReason?: string;
}) {
  const isSynthetic = dataMode === "synthetic";
  return (
    <div
      className={`mb-5 rounded-xl border px-4 py-3 ${
        isSynthetic
          ? "border-warning-300 bg-warning-50 dark:border-warning-500/30 dark:bg-warning-500/10"
          : "border-success-300 bg-success-50 dark:border-success-500/30 dark:bg-success-500/10"
      }`}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="inline-flex items-center gap-2 rounded-full bg-warning-500 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wide text-white">
          {isSynthetic ? "Synthetic Development Data" : "Live Data"}
        </span>
        <p className="text-xs text-warning-800 dark:text-warning-200">
          {isSynthetic
            ? "All probabilities, warnings and metrics shown are simulated outputs used for development and testing. They are NOT real-world Kerala predictions and must not be acted upon."
            : "Connected to live data source."}
        </p>
      </div>
      {sourceMode && (
        <p className="mt-1.5 text-[11px] text-warning-700 dark:text-warning-300">
          Source:{" "}
          <span className="font-medium">
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
