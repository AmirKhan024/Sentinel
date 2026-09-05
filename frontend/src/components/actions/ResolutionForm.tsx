import { useEffect, useState } from 'react'
import { submitResolution } from '../../api/review'
import type { DecisionScope, StagedRequestReceipt } from '../../api/types'
import { useActor } from '../../hooks/useActor'
import { useStagedSubmit } from '../../hooks/useStagedSubmit'
import { resolutionActionLabel } from '../../lib/copy'
import { generateId } from '../../lib/ids'
import { ErrorState } from '../common/ErrorState'
import { StagedReceiptNotice } from '../common/StagedReceiptNotice'

const ACTIONS = ['acknowledge', 'refer_to_override', 'refer_to_adjustment', 'escalate'] as const

/**
 * The existing resolution vocabulary, exactly (Component 16's `ReviewResolutionAction`).
 * `refer_to_override`/`refer_to_adjustment` only *record a pointer* to a decision made through
 * Component 13's or Component 14's own contract -- they never create that override or adjustment
 * themselves (`REVIEW_CANNOT`). `prefillOverrideId`/`prefillAdjustmentId` let the establishment
 * page hand over an id from an override/adjustment just submitted in the same visit, so a
 * reviewer who used both forms in order doesn't have to retype it.
 */
export function ResolutionForm({
  scope,
  targetInspectionId,
  reviewId,
  prefillOverrideId,
  prefillAdjustmentId,
  onStaged,
}: {
  scope: DecisionScope
  targetInspectionId: string
  reviewId: string
  prefillOverrideId?: string
  prefillAdjustmentId?: string
  onStaged?: (receipt: StagedRequestReceipt) => void
}) {
  const [open, setOpen] = useState(false)
  const [action, setAction] = useState<(typeof ACTIONS)[number]>('acknowledge')
  const [referencedOverrideId, setReferencedOverrideId] = useState(prefillOverrideId ?? '')
  const [referencedAdjustmentId, setReferencedAdjustmentId] = useState(prefillAdjustmentId ?? '')
  const [escalationNote, setEscalationNote] = useState('')
  const [reasonCode, setReasonCode] = useState('')
  const [actor, setActor] = useActor()

  useEffect(() => {
    if (prefillOverrideId) setReferencedOverrideId(prefillOverrideId)
  }, [prefillOverrideId])
  useEffect(() => {
    if (prefillAdjustmentId) setReferencedAdjustmentId(prefillAdjustmentId)
  }, [prefillAdjustmentId])

  const { status, receipt, error, submit, reset } = useStagedSubmit((signal) =>
    submitResolution(
      {
        review_id: generateId('REV'),
        policy_id: scope.policy_id ?? '',
        fold_id: scope.fold_id ?? '',
        k_name: scope.k_name ?? '',
        target_inspection_id: targetInspectionId,
        resolution_action: action,
        reason_code: reasonCode.trim(),
        actor: actor.trim(),
        decided_at: new Date().toISOString(),
        referenced_override_id: action === 'refer_to_override' ? referencedOverrideId.trim() : null,
        referenced_adjustment_id:
          action === 'refer_to_adjustment' ? referencedAdjustmentId.trim() : null,
        escalation_note: escalationNote.trim() || null,
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
        <StagedReceiptNotice receipt={receipt} what="Your resolution of this review case" />
        <button
          type="button"
          className="link-button"
          onClick={() => {
            reset()
            setOpen(false)
            setReasonCode('')
          }}
        >
          Submit another resolution
        </button>
      </div>
    )
  }

  if (!open) {
    return (
      <div className="action-form">
        <button type="button" onClick={() => setOpen(true)}>
          Resolve this case
        </button>
      </div>
    )
  }

  return (
    <form
      className="action-form"
      aria-label={`Resolve review case ${reviewId}`}
      onSubmit={(e) => {
        e.preventDefault()
        void submit()
      }}
    >
      <label>
        What do you want to record about this case?
        <select value={action} onChange={(e) => setAction(e.target.value as (typeof ACTIONS)[number])}>
          {ACTIONS.map((a) => (
            <option key={a} value={a}>
              {resolutionActionLabel(a)}
            </option>
          ))}
        </select>
      </label>
      {action === 'refer_to_override' && (
        <label>
          Override id this refers to (required -- submit the override first, above, if you
          haven't yet)
          <input
            type="text"
            value={referencedOverrideId}
            onChange={(e) => setReferencedOverrideId(e.target.value)}
            required
          />
        </label>
      )}
      {action === 'refer_to_adjustment' && (
        <label>
          Adjustment id this refers to (required -- submit the adjustment first, above, if you
          haven't yet)
          <input
            type="text"
            value={referencedAdjustmentId}
            onChange={(e) => setReferencedAdjustmentId(e.target.value)}
            required
          />
        </label>
      )}
      {action === 'escalate' && (
        <label>
          Escalation note (optional)
          <input
            type="text"
            value={escalationNote}
            onChange={(e) => setEscalationNote(e.target.value)}
            placeholder="Who this is being escalated to, and why"
          />
        </label>
      )}
      <label>
        Reason (required)
        <input
          type="text"
          value={reasonCode}
          onChange={(e) => setReasonCode(e.target.value)}
          placeholder="e.g. reviewed, matches known local context"
          required
        />
      </label>
      <label>
        Your name or id (required)
        <input type="text" value={actor} onChange={(e) => setActor(e.target.value)} required />
      </label>
      {status === 'error' && error && <ErrorState error={error} />}
      <div className="action-form-buttons">
        <button type="submit" disabled={status === 'submitting'}>
          {status === 'submitting' ? 'Submitting…' : 'Submit this resolution'}
        </button>
        <button type="button" className="link-button" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
      <p className="hint">
        This resolves the review case you opened it from. Referring a case to an override or
        adjustment only records a pointer here -- it never creates that override or adjustment
        itself.
      </p>
    </form>
  )
}
