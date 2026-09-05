import { useEffect, useRef, useState } from 'react'
import { classifyError, type ClassifiedError } from '../api/errors'

export type QueryState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: ClassifiedError }

/**
 * Fetch-on-deps-change with automatic cancellation of the previous in-flight request.
 *
 * `enabled: false` (the scope-incomplete case) skips fetching entirely and reports `idle` --
 * this is a client-side convenience, not a re-implementation of the API's own scope validation.
 */
export function useApiQuery<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  enabled: boolean,
): QueryState<T> {
  const [state, setState] = useState<QueryState<T>>({ status: 'idle' })
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  useEffect(() => {
    if (!enabled) {
      setState({ status: 'idle' })
      return
    }
    const controller = new AbortController()
    setState({ status: 'loading' })
    fetcherRef
      .current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setState({ status: 'success', data })
        }
      })
      .catch((err: unknown) => {
        const classified = classifyError(err)
        if (classified.kind === 'aborted') {
          return
        }
        setState({ status: 'error', error: classified })
      })
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps])

  return state
}
