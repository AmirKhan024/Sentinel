import { useEffect, useRef } from 'react'
import type { DecisionScope, ManifestJson } from '../api/types'
import { FOLD_TABLE } from '../api/folds'

/** The most recent fold in the (hardcoded, verified-real) fold table -- see api/folds.ts for
 * why this one dimension is not read from a manifest. Quarterly folds are listed oldest-first,
 * so the last quarterly entry is the most recent one. */
function latestQuarterlyFold(): { fold_set: string; fold_id: string } {
  const quarterly = FOLD_TABLE.filter((f) => f.fold_set === 'quarterly')
  return quarterly[quarterly.length - 1] ?? FOLD_TABLE[0]
}

/**
 * Fills in a full, valid decision scope automatically the first time real manifest data is
 * available, so a first-time visitor sees a working "Inspection plan" immediately instead of an
 * empty form. Every field is derived from something the API actually published:
 *
 * - `fold_set` / `fold_id`: the most recent entry in the real fold table.
 * - `policy_id`: the baseline policy ("prioritize by risk") -- the simplest one to explain.
 * - `model_name` / `k_name`: the policy manifest's own `selected_model` / `primary_k_level`.
 * - `schedule_config_id`: the scheduling manifest's own default configuration.
 *
 * Never overwrites a field the visitor (or a bookmarked URL) already set -- this only fills in
 * what is still empty, once, and never mixes a stale guess with a value the user chose.
 *
 * All fields are set in a single `setScopeFields` call. Setting them one at a time via repeated
 * `setScopeField` calls looked correct but silently dropped all but the last field: each call
 * to react-router's `setSearchParams` reads the *currently committed* URL, not the result of an
 * unapplied sibling call from earlier in the same tick, so only the final call reliably won.
 */
export function useDefaultScope(
  scope: DecisionScope,
  setScopeFields: (fields: Partial<Record<keyof DecisionScope, string | undefined>>) => void,
  manifests: Partial<Record<'policy' | 'scheduling', ManifestJson>>,
): void {
  const appliedRef = useRef(false)

  useEffect(() => {
    if (appliedRef.current) return
    const policyManifest = manifests.policy
    const schedulingManifest = manifests.scheduling
    if (!policyManifest) return

    const fold = latestQuarterlyFold()
    const defaultConfig = schedulingManifest?.config_grid?.find((c) => c.is_default)

    const updates: Partial<Record<keyof DecisionScope, string | undefined>> = {}
    if (!scope.fold_set) updates.fold_set = fold.fold_set
    if (!scope.fold_id) updates.fold_id = fold.fold_id
    if (!scope.policy_id) updates.policy_id = 'pure_risk'
    if (!scope.model_name && policyManifest.selected_model) {
      updates.model_name = policyManifest.selected_model
    }
    if (!scope.k_name && policyManifest.primary_k_level) {
      updates.k_name = policyManifest.primary_k_level
    }
    if (!scope.schedule_config_id && defaultConfig) {
      updates.schedule_config_id = defaultConfig.schedule_config_id
    }
    if (Object.keys(updates).length > 0) {
      setScopeFields(updates)
    }
    appliedRef.current = true
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifests.policy, manifests.scheduling])
}
