/**
 * Generic data-loading hook with explicit loading / error / empty states.
 *
 * Keeps components free of fetch plumbing and guarantees every screen can
 * render the three non-happy paths the UI requires.
 */

import { useCallback, useEffect, useState } from "react";
import type { ApiResult, SourceMode } from "../services/api";

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  sourceMode: SourceMode | null;
  fallbackReason?: string;
  reload: () => void;
}

export function useApi<T>(
  loader: () => Promise<ApiResult<T>>,
  deps: unknown[] = []
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<SourceMode | null>(null);
  const [fallbackReason, setFallbackReason] = useState<string | undefined>();
  const [tick, setTick] = useState(0);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(loader, deps);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    run()
      .then((res) => {
        if (cancelled) return;
        setData(res.data);
        setSourceMode(res.sourceMode);
        setFallbackReason(res.fallbackReason);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unknown error");
        setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [run, tick]);

  return {
    data,
    loading,
    error,
    sourceMode,
    fallbackReason,
    reload: () => setTick((t) => t + 1),
  };
}
