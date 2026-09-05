import { useEffect, useState } from 'react'
import { submitOverride } from '../../api/overrides'
import type { DecisionScope, StagedRequestReceipt } from '../../api/types'
import { useActor } from '../../hooks/useActor'
import { useStagedSubmit } from '../../hooks/useStagedSubmit'
import { overrideActionLabel } from '../../lib/copy'
import { generateId } from '../../lib/ids'
import { ErrorState } from '../common/ErrorState'
import { StagedReceiptNotice } from '../common/StagedReceiptNotice'

/**
 * "Change the priority decision" -- Component 13's `Override` contract (force_include /
 * force_exclude), exposed exactly as the backend defines it. Only one action is offered: the one
 * that is not already a no-op given the establishment's current selection state (an establishment
 * already selected cannot be meaningfully force_included, and vice versa) -- see
 * `sentinel.policy.governance.apply_overrides`.
 */
export function OverrideForm({
  scope,
  targetInspectionId,
  isSelected,
  onStaged,
}: {
  scope: DecisionScope
  targetInspectionId: string
  isSelected: boolean
  onStaged?: (receipt: StagedRequestReceipt) => void
}) {
  const action = isSelected ? 'force_exclude' : 'force_include'
  const [reasonCode, setReasonCode] = useState('')
  const [actor, setActor] = useActor()
  const [wantsChange, setWantsChange] = useState(false)
  const { status, receipt, error, submit, reset } = useStagedSubmit((signal) =>
    submitOverride(
      {
        override_id: generateId('OVR'),
        policy_id: scope.policy_id ?? '',
        fold_id: scope.fold_id ?? '',
        k_name: scope.k_name ?? '',
        target_inspection_id: targetInspectionId,
        action,
        reason_code: reasonCode.trim(),
        actor: actor.trim(),
        decided_at: new Date().toISOString(),
      },
      signal,
    ),
  )

  useEffect(() => {
    if (status === 'success' && receipt) onStaged?.(receipt)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, receipt])

  if (status === 'success' && receipt) {
    return (
      <div className="action-form">
        <StagedReceiptNotice receipt={receipt} what="Your change to the priority decision" />
        <button
          type="button"
          className="link-button"
          onClick={() => {
            reset()
            setWantsChange(false)
            setReasonCode('')
          }}
        >
          Submit another change
        </button>
      </div>
    )
  }

  if (!wantsChange) {
    return (
      <div className="action-form">
        <p>
          <button type="button" onClick={() => setWantsChange(true)}>
            Change the priority decision
          </button>{' '}
          <span className="hint">
            or keep Sentinel's current decision -- no action needed to do that.
          </span>
        </p>
      </div>
    )
  }

  return (
    <form
      className="action-form"
      onSubmit={(e) => {
        e.preventDefault()
        void submit()
      }}
    >
      <p>
        <strong>{overrideActionLabel(action)}.</strong>{' '}
        <span className="hint">
          Capacity is fixed, so including this establishment will bump the lowest-ranked
          currently-selected establishment out of this plan; excluding it frees its slot without
          promoting anyone else into it.
        </span>
      </p>
      <label>
        Why are you changing Sentinel's decision? (required)
        <input
          type="text"
          value={reasonCode}
          onChange={(e) => setReasonCode(e.target.value)}
          placeholder="e.g. local complaint on file, known renovation in progress"
          required
        />
      </label>
      <label>
        Your name or id (required)
        <input type="text" value={actor} onChange={(e) => setActor(e.target.value)} required />
      </label>
      <p className="hint">This is what the audit record will show for this decision.</p>
      {status === 'error' && error && <ErrorState error={error} />}
      <div className="action-form-buttons">
        <button type="submit" disabled={status === 'submitting'}>
          {status === 'submitting' ? 'Submitting…' : 'Submit this decision'}
        </button>
        <button type="button" className="link-button" onClick={() => setWantsChange(false)}>
          Cancel
        </button>
      </div>
    </form>
  )
}
