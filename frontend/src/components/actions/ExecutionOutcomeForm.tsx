import { useEffect, useState } from 'react'
import { getExecutionContract, submitExecutionEvent } from '../../api/execution'
import type { DecisionScope, StagedRequestReceipt } from '../../api/types'
import { useActor } from '../../hooks/useActor'
import { useStagedSubmit } from '../../hooks/useStagedSubmit'
import { executionStatusLabel } from '../../lib/copy'
import { generateId } from '../../lib/ids'
import { ErrorState } from '../common/ErrorState'
import { StagedReceiptNotice } from '../common/StagedReceiptNotice'

/** Never invented: the option list comes from `/v1/execution/contract`, Component 14's own
 * data-driven description of what a person may report -- not a hardcoded enum. Falls back to the
 * three statuses defined in `sentinel.scheduling.definitions.ExecutionStatus` only if the
 * contract table can't be read, so the form still works but never silently disagrees with it. */
const FALLBACK_STATUSES = ['completed', 'not_performed', 'cancelled_in_field']

/**
 * "Record inspection outcome" -- Component 14's `ExecutionEvent` contract. This is the flow that
 * closes the loop the rest of the product only ever pointed at: after this is staged and later
 * applied, the establishment's row in `execution_log` stops being "no record yet."
 */
export function ExecutionOutcomeForm({
  scope,
  targetInspectionId,
  scheduleConfigId,
  defaultScheduledDate,
  onStaged,
}: {
  scope: DecisionScope
  targetInspectionId: string
  scheduleConfigId: string
  defaultScheduledDate: string | null
  onStaged?: (receipt: StagedRequestReceipt) => void
}) {
  const [open, setOpen] = useState(false)
  const [statuses, setStatuses] = useState<string[]>(FALLBACK_STATUSES)
  const [status, setStatusValue] = useState(FALLBACK_STATUSES[0])
  const [scheduledDate, setScheduledDate] = useState(defaultScheduledDate ?? '')
  const [reasonCode, setReasonCode] = useState('')
  const [actor, setActor] = useActor()

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    getExecutionContract(controller.signal)
      .then((rows) => {
        const field = rows.find((r) => r.field_name === 'execution_status')
        const allowed = field?.allowed_values?.split('|').filter(Boolean)
        if (allowed && allowed.length > 0) {
          setStatuses(allowed)
          setStatusValue(allowed[0])
        }
      })
      .catch(() => {
        // Contract table not available -- the fallback list (kept in sync with the backend
        // enum) still lets the form work.
      })
    return () => controller.abort()
  }, [open])

  const { status: submitStatus, receipt, error, submit, reset } = useStagedSubmit((signal) =>
    submitExecutionEvent(
      {
        execution_id: generateId('EXE'),
        schedule_config_id: scheduleConfigId,
        policy_id: scope.policy_id ?? '',
        fold_id: scope.fold_id ?? '',
        k_name: scope.k_name ?? '',
        target_inspection_id: targetInspectionId,
        scheduled_date: scheduledDate,
        execution_status: status,
        reason_code: reasonCode.trim(),
        actor: actor.trim(),
        observed_at: new Date().toISOString(),
      },
      signal,
    ),
  )

  useEffect(() => {
    if (submitStatus === 'success' && receipt) onStaged?.(receipt)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [submitStatus, receipt])

  if (submitStatus === 'success' && receipt) {
    return (
      <div className="action-form">
        <StagedReceiptNotice receipt={receipt} what="This inspection outcome" />
        <p className="hint">
          This does not remove the missing-outcome flag from Human Review immediately -- that
          only happens the next time the plan is rebuilt with this event applied. It is on record
          now.
        </p>
        <button
          type="button"
          className="link-button"
          onClick={() => {
            reset()
            setOpen(false)
            setReasonCode('')
          }}
        >
          Record another outcome
        </button>
      </div>
    )
  }

  if (!open) {
    return (
      <div className="action-form">
        <button type="button" onClick={() => setOpen(true)}>
          Record inspection outcome
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
      <label>
        What happened?
        <select value={status} onChange={(e) => setStatusValue(e.target.value)}>
          {statuses.map((s) => (
            <option key={s} value={s}>
              {executionStatusLabel(s)}
            </option>
          ))}
        </select>
      </label>
      <label>
        Scheduled date this refers to (required)
        <input
          type="date"
          value={scheduledDate}
          onChange={(e) => setScheduledDate(e.target.value)}
          required
        />
      </label>
      <label>
        Notes (required)
        <input
          type="text"
          value={reasonCode}
          onChange={(e) => setReasonCode(e.target.value)}
          placeholder="e.g. field report, inspector notes"
          required
        />
      </label>
      <label>
        Your name or id (required)
        <input type="text" value={actor} onChange={(e) => setActor(e.target.value)} required />
      </label>
      {submitStatus === 'error' && error && <ErrorState error={error} />}
      <div className="action-form-buttons">
        <button type="submit" disabled={submitStatus === 'submitting'}>
          {submitStatus === 'submitting' ? 'Submitting…' : 'Submit this outcome'}
        </button>
        <button type="button" className="link-button" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </form>
  )
}
