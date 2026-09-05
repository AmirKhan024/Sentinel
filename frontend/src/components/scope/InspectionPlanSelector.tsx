import type { DecisionScope, ManifestJson } from '../../api/types'
import { foldLabel, formatDateTime } from '../../lib/copy'
import { ScopeSelector } from './ScopeSelector'
import { TechnicalDetails } from '../common/TechnicalDetails'

type ManifestCache = Partial<Record<'policy' | 'scheduling', ManifestJson>>

/**
 * The primary, plain-language stand-in for the raw scope form every page used to lead with.
 * A first-time visitor sees the planning period and what kind of view this is (already filled
 * in by `useDefaultScope`) -- a completed, timestamped planning run, never described as "today"
 * or a live feed, per the product reality check. The full technical scope form -- policy, model,
 * fold, capacity level, schedule configuration -- is still there, just one click away under
 * "Advanced options", for an operator who genuinely needs to compare a different plan.
 */
export function InspectionPlanSelector({
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
  const ready = requiredFields.every((f) => scope[f] !== undefined && scope[f] !== '')
  const builtAt = manifests.policy?.built_at

  return (
    <section className="inspection-plan-selector">
      <div className="inspection-plan-summary">
        {ready ? (
          <>
            <p className="inspection-plan-period">
              Planning period: <strong>{foldLabel(scope.fold_id)}</strong>
            </p>
            <p className="hint">
              This view shows Sentinel's completed planning run for this period
              {builtAt ? `, generated ${formatDateTime(builtAt)}` : ''}. It is a decision-support
              record, not a live calendar of upcoming appointments — Sentinel is not currently
              connected to a live feed of new inspections.
            </p>
          </>
        ) : (
          <p>Choose an inspection plan below to get started.</p>
        )}
      </div>
      <TechnicalDetails summary="Advanced options">
        <p className="hint">
          Sentinel can compare different prioritization approaches and time periods. Most users
          never need to change these.
        </p>
        <ScopeSelector
          scope={scope}
          setScopeField={setScopeField}
          requiredFields={requiredFields}
          manifests={manifests}
          showAdvanced={showAdvanced}
        />
      </TechnicalDetails>
    </section>
  )
}
