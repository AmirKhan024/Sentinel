import { useEffect, useState } from 'react'
import { submitPlanDecision } from '../../api/planReview'
import type { StagedRequestReceipt } from '../../api/types'
import { useActor } from '../../hooks/useActor'
import { useStagedSubmit } from '../../hooks/useStagedSubmit'
import { planDecisionActionLabel } from '../../lib/copy'
import { generateId } from '../../lib/ids'
import { ErrorState } from '../common/ErrorState'
import { StagedReceiptNotice } from '../common/StagedReceiptNotice'

const ACTIONS = [
  'keep_selected',
  'move_to_later_workday',
  'adjust_operational_priority',
  'do_not_proceed_as_planned',
] as const

/**
 * Component 21's decision vocabulary, exactly (`PlanDecisionAction`). Records the supervisor's
 * decision about one establishment in the proposed plan -- it never edits Sentinel's own
 * recommendation or Component 20's geographic placement, and `do_not_proceed_as_planned` does
 * not remove the establishment from the plan; both stay visible, side by side, always.
 */
export function PlanDecisionForm({
  planningDate,
  targetInspectionId,
  planAlreadyApproved,
  onStaged,
}: {
  planningDate: string
  targetInspectionId: string
  /** Whether this plan has already been approved -- when true, a decision recorded here does
   * not change the already-approved plan; it is only included the next time this plan is
   * reviewed and re-approved. Shown explicitly so this is never a silent surprise. */
  planAlreadyApproved?: boolean
  onStaged?: (receipt: StagedRequestReceipt) => void
}) {
  const [open, setOpen] = useState(false)
  const [action, setAction] = useState<(typeof ACTIONS)[number]>('keep_selected')
  const [revisedPlannedDate, setRevisedPlannedDate] = useState('')
  const [revisedOperationalPriority, setRevisedOperationalPriority] = useState('')
  const [reasonCode, setReasonCode] = useState('')
  const [actor, setActor] = useActor()

  const { status, receipt, error, submit, reset } = useStagedSubmit((signal) =>
    submitPlanDecision(
      {
        decision_id: generateId('DEC'),
        planning_date: planningDate,
        target_inspection_id: targetInspectionId,
        decision_action: action,
        reason_code: reasonCode.trim(),
        actor: actor.trim(),
        decided_at: new Date().toISOString(),
        revised_planned_date:
          action === 'move_to_later_workday' ? revisedPlannedDate || null : null,
        revised_operational_priority:
          action === 'adjust_operational_priority' && revisedOperationalPriority
            ? Number(revisedOperationalPriority)
            : null,
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
        <StagedReceiptNotice receipt={receipt} what="Your decision about this establishment" />
        <button
          type="button"
          className="link-button"
          onClick={() => {
            reset()
            setOpen(false)
            setReasonCode('')
          }}
        >
          Record another decision
        </button>
      </div>
    )
  }

  if (!open) {
    return (
      <div className="action-form">
        <button type="button" onClick={() => setOpen(true)}>
          Record a decision
        </button>
        {planAlreadyApproved && (
          <p className="hint">
            This plan has already been approved. A new decision here won't change the approved
            plan -- it will be included the next time this plan is reviewed and re-approved.
          </p>
        )}
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
      {planAlreadyApproved && (
        <p className="hint">
          This plan has already been approved. This decision will not change the approved plan
          -- it will be included the next time this plan is reviewed and re-approved.
        </p>
      )}
      <label>
        What do you want to record?
        <select value={action} onChange={(e) => setAction(e.target.value as (typeof ACTIONS)[number])}>
          {ACTIONS.map((a) => (
            <option key={a} value={a}>
              {planDecisionActionLabel(a)}
            </option>
          ))}
        </select>
      </label>
      {action === 'move_to_later_workday' && (
        <label>
          Intended later date (required)
          <input
            type="date"
            value={revisedPlannedDate}
            onChange={(e) => setRevisedPlannedDate(e.target.value)}
            required
          />
        </label>
      )}
      {action === 'adjust_operational_priority' && (
        <label>
          Revised field-work order (required)
          <input
            type="number"
            min={1}
            value={revisedOperationalPriority}
            onChange={(e) => setRevisedOperationalPriority(e.target.value)}
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
          placeholder="e.g. inspector already assigned elsewhere"
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
          {status === 'submitting' ? 'Submitting…' : 'Submit this decision'}
        </button>
        <button type="button" className="link-button" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
      <p className="hint">
        This never overwrites Sentinel's own recommendation -- both are kept, side by side, in
        the record. It also never creates a schedule change or a recommendation override by
        itself; those remain separate submissions through their own contracts. Adjusting the
        field-work order changes only the display order shown to field staff; it never changes
        Sentinel's own risk rank.
      </p>
    </form>
  )
}
