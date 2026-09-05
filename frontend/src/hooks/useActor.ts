import { useCallback, useState } from 'react'

const STORAGE_KEY = 'sentinel.actor'

/** The `actor` field every write contract requires (Override, Adjustment, ExecutionEvent,
 * ReviewResolution all refuse a blank one -- an anonymous decision is exactly what an audit
 * trail exists to prevent). Sentinel has no authentication (a documented gap, not a silent
 * omission -- see the Sentinel API docs), so this is a plain, session-local, self-reported name
 * remembered in this browser only. It is NOT identity verification. */
export function useActor(): [string, (value: string) => void] {
  const [actor, setActorState] = useState<string>(() => {
    try {
      return window.localStorage.getItem(STORAGE_KEY) ?? ''
    } catch {
      return ''
    }
  })

  const setActor = useCallback((value: string) => {
    setActorState(value)
    try {
      window.localStorage.setItem(STORAGE_KEY, value)
    } catch {
      // Private browsing / storage disabled -- the name still works for this submission,
      // it just won't be remembered next time.
    }
  }, [])

  return [actor, setActor]
}
