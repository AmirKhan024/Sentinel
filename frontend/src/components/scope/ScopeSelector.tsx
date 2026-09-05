import type { ManifestComponent } from '../../api/meta'
import type { DecisionScope, ManifestJson } from '../../api/types'
import { FoldIdSelect, FoldSetSelect } from './FoldSelect'
import { ManifestSelect } from './ManifestSelect'

type ManifestCache = Partial<Record<ManifestComponent, ManifestJson>>

/** One reusable scope form, configured per page by which DecisionScope fields it needs.
 * fold_set/fold_id come from the hardcoded static fold table; policy_id/model_name/k_name/
 * schedule_config_id come live from whichever manifest(s) are passed in `manifests`. */
export function ScopeSelector({
  scope,
  setScopeField,
  requiredFields,
  manifests,
  showAdvanced = false,
}: {
  scope: DecisionScope
  setScopeField: (field: keyof DecisionScope, value: string | undefined) => void
  requiredFields: (keyof DecisionScope)[]
  manifests: ManifestCache
  showAdvanced?: boolean
}) {
  const policyManifest = manifests.policy
  const schedulingManifest = manifests.scheduling
  const required = new Set(requiredFields)

  const policyOptions = (policyManifest?.policy_grid ?? []).map((p) => p.policy_id)
  const modelOptions = policyManifest?.candidate_models ?? []
  const kLevelOptions = policyManifest?.k_levels ?? schedulingManifest?.k_levels ?? []
  const scheduleConfigOptions = (schedulingManifest?.config_grid ?? []).map(
    (c) => c.schedule_config_id,
  )

  return (
    <fieldset className="scope-selector">
      <legend>Decision scope</legend>
      {required.has('fold_set') && (
        <FoldSetSelect value={scope.fold_set} onChange={(v) => setScopeField('fold_set', v)} />
      )}
      {required.has('fold_id') && (
        <FoldIdSelect
          foldSet={scope.fold_set}
          value={scope.fold_id}
          onChange={(v) => setScopeField('fold_id', v)}
        />
      )}
      {required.has('policy_id') && (
        <ManifestSelect
          label="policy_id"
          options={policyOptions}
          value={scope.policy_id}
          onChange={(v) => setScopeField('policy_id', v)}
        />
      )}
      {required.has('model_name') && (
        <ManifestSelect
          label="model_name"
          options={modelOptions}
          value={scope.model_name}
          onChange={(v) => setScopeField('model_name', v)}
        />
      )}
      {required.has('k_name') && (
        <ManifestSelect
          label="k_name"
          options={kLevelOptions}
          value={scope.k_name}
          onChange={(v) => setScopeField('k_name', v)}
        />
      )}
      {required.has('schedule_config_id') && (
        <ManifestSelect
          label="schedule_config_id"
          options={scheduleConfigOptions}
          value={scope.schedule_config_id}
          onChange={(v) => setScopeField('schedule_config_id', v)}
        />
      )}
      {showAdvanced && (
        <details className="scope-advanced">
          <summary>Advanced (planning_run_id / replan_index — default to latest replan)</summary>
          <label className="scope-field">
            planning_run_id
            <input
              type="text"
              value={scope.planning_run_id ?? ''}
              onChange={(e) => setScopeField('planning_run_id', e.target.value || undefined)}
            />
          </label>
          <label className="scope-field">
            replan_index
            <input
              type="number"
              value={scope.replan_index ?? ''}
              onChange={(e) => setScopeField('replan_index', e.target.value || undefined)}
            />
          </label>
        </details>
      )}
    </fieldset>
  )
}
