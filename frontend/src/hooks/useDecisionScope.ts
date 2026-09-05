import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { DecisionScope } from '../api/types'

const SCOPE_KEYS: (keyof DecisionScope)[] = [
  'policy_id',
  'model_name',
  'fold_set',
  'fold_id',
  'k_name',
  'schedule_config_id',
  'planning_run_id',
  'replan_index',
]

/**
 * Decision scope lives in the URL query string -- shareable/bookmarkable, and the natural
 * mechanism for preserving scope when a table row links to the establishment detail page.
 */
export function useDecisionScope(): {
  scope: DecisionScope
  setScopeField: (field: keyof DecisionScope, value: string | undefined) => void
  setScopeFields: (fields: Partial<Record<keyof DecisionScope, string | undefined>>) => void
  missingFields: (required: (keyof DecisionScope)[]) => (keyof DecisionScope)[]
} {
  const [searchParams, setSearchParams] = useSearchParams()

  const scope = useMemo<DecisionScope>(() => {
    const raw: Record<string, string> = {}
    for (const key of SCOPE_KEYS) {
      const value = searchParams.get(key)
      if (value !== null && value !== '') raw[key] = value
    }
    const { replan_index, ...rest } = raw
    return {
      ...rest,
      ...(replan_index !== undefined ? { replan_index: Number(replan_index) } : {}),
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams.toString()])

  /** Sets exactly one field. Calling this more than once in the same synchronous block (e.g.
   * from one effect) is unsafe: react-router's `setSearchParams` updater receives the *current*
   * committed search params at the time each call is scheduled, not the result of an
   * as-yet-unapplied sibling call, so only the last of several rapid calls reliably survives.
   * Use `setScopeFields` for anything that sets more than one field at once. */
  function setScopeField(field: keyof DecisionScope, value: string | undefined) {
    setScopeFields({ [field]: value })
  }

  /** Sets any number of fields in a single search-params update, so every field survives
   * regardless of how many are set together (see `setScopeField`'s note). */
  function setScopeFields(fields: Partial<Record<keyof DecisionScope, string | undefined>>) {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        for (const [field, value] of Object.entries(fields)) {
          if (value === undefined || value === '') {
            next.delete(field)
          } else {
            next.set(field, value)
          }
        }
        return next
      },
      { replace: true },
    )
  }

  function missingFields(required: (keyof DecisionScope)[]): (keyof DecisionScope)[] {
    return required.filter((field) => scope[field] === undefined || scope[field] === '')
  }

  return { scope, setScopeField, setScopeFields, missingFields }
}
