/**
 * Data-fetching hooks.
 *
 * A small hand-rolled `useAsync` rather than TanStack Query: the app makes a
 * handful of read-only calls with no cache invalidation or mutation story, and
 * a dependency earning its place needs more work than that to do.
 */

import { useCallback, useEffect, useState } from 'react';
import { ApiError } from './client';

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  /** Refetch on demand, e.g. from a retry button. */
  reload: () => void;
}

export function useAsync<T>(fetcher: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce(n => n + 1), []);

  useEffect(() => {
    // Guards against a slow first response overwriting a newer one after the
    // user has already changed filters.
    let cancelled = false;

    setLoading(true);
    setError(null);

    fetcher()
      .then(result => {
        if (!cancelled) setData(result);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setData(null);
        if (err instanceof ApiError) {
          setError(err.message);
        } else if (err instanceof TypeError) {
          // fetch throws TypeError when it cannot reach the host at all, which
          // almost always means the API simply is not running.
          setError('Cannot reach the API. Is the server running on port 8000?');
        } else {
          setError(String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, loading, error, reload };
}
