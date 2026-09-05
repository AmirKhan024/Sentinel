import { useEffect, useState } from 'react'
import { getPlanApproval, submitApproval } from '../../api/planReview'
import type { PlanApprovalOut, StagedRequestReceipt } from '../../api/types'
import { useActor } from '../../hooks/useActor'
import { useApiQuery } from '../../hooks/useApiQuery'
import { useStagedSubmit } from '../../hooks/useStagedSubmit'
import { generateId } from '../../lib/ids'
import { formatDateTime } from '../../lib/copy'
import { ErrorState } from '../common/ErrorState'
import { StagedReceiptNotice } from '../common/StagedReceiptNotice'
import { TechnicalDetails } from '../common/TechnicalDetails'
import { FieldRow } from '../common/FieldRow'

/**
 * Component 21's approval mechanism. Staging an approval here does not approve the plan --
 * see ADR 0049: an operator later runs `sentinel approve-plan`, which re-runs the full
 * readiness checklist (every row carries the machine recommendation, every decision has a
 * reason, ...) before writing the immutable `approved_operational_plan` artifact. This panel
 * never recomputes or previews that checklist itself.
 */
export function PlanApprovalPanel({
  planningDate,
  decisionsRecorded,
  totalEstablishments,
  onStaged,
}: {
  planningDate: string
  decisionsRecorded: number
  totalEstablishments: number
  onStaged?: (receipt: StagedRequestReceipt) => void
}) {
  const approvalQuery = useApiQuery<PlanApprovalOut>(
    (signal) => getPlanApproval(planningDate, signal),
    [planningDate],
    Boolean(planningDate),
  )

  const [open, setOpen] = useState(false)
  const [note, setNote] = useState('')
  const [actor, setActor] = useActor()

  const { status, receipt, error, submit, reset } = useStagedSubmit((signal) =>
    submitApproval(
      {
        approval_id: generateId('APPR'),
        planning_date: planningDate,
        approved_by: actor.trim(),
        approved_at: new Date().toISOString(),
        note: note.trim() || null,
      },
      signal,
    ),
  )

  useEffect(() => {
    if (status === 'success' && receipt) onStaged?.(receipt)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, receipt])

  if (approvalQuery.status === 'success') {
    const approval = approvalQuery.data
    return (
      <section className="plan-approval-panel plan-approval-panel-approved">
        <h2>Plan approved</h2>
        <p>
          Approved by {approval.approved_by} at {formatDateTime(approval.approved_at)}.
        </p>
        <p className="hint">
          {approval.final_active_count} active · {approval.final_deferred_count} deferred ·{' '}
          {approval.final_not_proceeding_count} not proceeding ·{' '}
          {approval.final_undecided_count} left to the machine recommendation ·{' '}
          {approval.final_selected_count} total.
        </p>
        {approval.note && <p className="hint">Note: {approval.note}</p>}
      </section>
    )
  }

  const undecided = totalEstablishments - decisionsRecorded

  if (status === 'success' && receipt) {
    return (
      <section className="plan-approval-panel">
        <StagedReceiptNotice receipt={receipt} what="Your plan approval" />
        <button
          type="button"
          className="link-button"
          onClick={() => {
            reset()
            setOpen(false)
            setNote('')
          }}
        >
          Stage another approval
        </button>
      </section>
    )
  }

  return (
    <section className="plan-approval-panel">
      <h2>Approve this plan</h2>
      <p className="hint">
        {decisionsRecorded} of {totalEstablishments} establishments have a recorded decision
        {undecided > 0 && ` (${undecided} left undecided -- these default to Sentinel's own recommendation)`}
        . Approving does not require every row to be decided.
      </p>
      {!open && (
        <button type="button" onClick={() => setOpen(true)}>
          Approve plan
        </button>
      )}
      {open && (
        <form
          className="action-form"
          onSubmit={(e) => {
            e.preventDefault()
            void submit()
          }}
        >
          <label>
            Optional note
            <input
              type="text"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g. reviewed with field supervisors"
            />
          </label>
          <label>
            Your name or id (required)
            <input type="text" value={actor} onChange={(e) => setActor(e.target.value)} required />
          </label>
          {status === 'error' && error && <ErrorState error={error} />}
          <div className="action-form-buttons">
            <button type="submit" disabled={status === 'submitting'}>
              {status === 'submitting' ? 'Submitting…' : 'Stage approval'}
            </button>
            <button type="button" className="link-button" onClick={() => setOpen(false)}>
              Cancel
            </button>
          </div>
          <p className="hint">
            Your approval is staged and recorded. It becomes the final, authoritative plan once
            it is confirmed -- Sentinel checks the plan is complete and consistent before that
            happens, so nothing is finalized silently.
          </p>
          <TechnicalDetails summary="Technical details">
            <FieldRow label="mechanism" value="staged, never applied directly (ADR 0049)" />
            <FieldRow label="commit command" value="sentinel approve-plan" />
            <p className="hint">
              An operator commits staged approvals through the CLI, which re-runs a 5-point
              readiness checklist (every row carries the machine recommendation, geographic
              provenance present, every recorded decision has a reason, ...) and refuses the
              whole approval outright if anything fails.
            </p>
          </TechnicalDetails>
        </form>
      )}
    </section>
  )
}
