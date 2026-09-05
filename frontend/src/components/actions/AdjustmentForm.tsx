import { useEffect, useState } from 'react'
import { submitAdjustment } from '../../api/adjustments'
import type { DecisionScope, StagedRequestReceipt } from '../../api/types'
import { useActor } from '../../hooks/useActor'
import { useStagedSubmit } from '../../hooks/useStagedSubmit'
import { adjustmentActionLabel } from '../../lib/copy'
import { generateId } from '../../lib/ids'
import { ErrorState } from '../common/ErrorState'
import { StagedReceiptNotice } from '../common/StagedReceiptNotice'

const ACTIONS = ['defer_to_date', 'advance_to_date', 'cancel'] as const

/**
 * "Adjust this planned inspection" -- Component 14's `Adjustment` contract exactly
 * (defer_to_date / advance_to_date / cancel). `target_date` is required for a move and refused
 * by the backend if present on a cancel (`sentinel.scheduling.adjustments.parse_adjustments`),
 * so the field is cleared and disabled the moment "cancel" is chosen rather than left for the
 * API to reject.
 */
export function AdjustmentForm({
  scope,
  targetInspectionId,
  scheduleConfigId,
  onStaged,
}: {
  scope: DecisionScope
  targetInspectionId: string
  scheduleConfigId: string
  onStaged?: (receipt: StagedRequestReceipt) => void
}) {
  const [open, setOpen] = useState(false)
  const [action, setAction] = useState<(typeof ACTIONS)[number]>('defer_to_date')
  const [targetDate, setTargetDate] = useState('')
  const [reasonCode, setReasonCode] = useState('')
  const [actor, setActor] = useActor()
  const { status, receipt, error, submit, reset } = useStagedSubmit((signal) =>
    submitAdjustment(
      {
        adjustment_id: generateId('ADJ'),
        schedule_config_id: scheduleConfigId,
        policy_id: scope.policy_id ?? '',
        fold_id: scope.fold_id ?? '',
        k_name: scope.k_name ?? '',
        target_inspection_id: targetInspectionId,
        action,
        target_date: action === 'cancel' ? '' : targetDate,
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
        <StagedReceiptNotice receipt={receipt} what="Your schedule adjustment" />
        <button
          type="button"
          className="link-button"
          onClick={() => {
            reset()
            setOpen(false)
            setReasonCode('')
            setTargetDate('')
          }}
        >
          Submit another adjustment
        </button>
      </div>
    )
  }

  if (!open) {
    return (
      <div className="action-form">
        <button type="button" onClick={() => setOpen(true)}>
          Adjust this planned inspection
        </button>
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
      <p className="hint">
        Explain why this plan should change. Moving or cancelling this inspection frees its slot
        for the day it currently occupies without promoting anyone else into it; moving it into an
        occupied day displaces that day's lowest-ranked risk-priority row, which is logged too.
      </p>
      <label>
        What do you want to do?
        <select value={action} onChange={(e) => setAction(e.target.value as (typeof ACTIONS)[number])}>
          {ACTIONS.map((a) => (
            <option key={a} value={a}>
              {adjustmentActionLabel(a)}
            </option>
          ))}
        </select>
      </label>
      {action !== 'cancel' && (
        <label>
          New date (required)
          <input
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            required
          />
        </label>
      )}
      <label>
        Reason (required)
        <input
          type="text"
          value={reasonCode}
          onChange={(e) => setReasonCode(e.target.value)}
          placeholder="e.g. inspector reassigned, establishment requested reschedule"
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
          {status === 'submitting' ? 'Submitting…' : 'Submit this adjustment'}
        </button>
        <button type="button" className="link-button" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </form>
  )
}
