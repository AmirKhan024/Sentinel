import { listAdjustments } from '../../api/adjustments'
import { listExecutionEvents } from '../../api/execution'
import { listOverrides } from '../../api/overrides'
import { listResolutions } from '../../api/review'
import { listStagedRequests } from '../../api/stagedRequests'
import type { DecisionScope } from '../../api/types'
import { useApiQuery } from '../../hooks/useApiQuery'
import {
  adjustmentActionLabel,
  executionStatusLabel,
  formatDateTime,
  overrideActionLabel,
  resolutionActionLabel,
} from '../../lib/copy'
import { EmptyState } from '../common/EmptyState'
import { ErrorState } from '../common/ErrorState'
import { LoadingState } from '../common/LoadingState'

interface HistoryEntry {
  key: string
  when: string
  what: string
  actor: string
  reason: string
  state: 'committed' | 'pending'
}

/**
 * "Sentinel recommendation -> human override -> schedule adjustment -> inspection outcome
 * recorded" -- one establishment's real audit trail, read from the four existing logs plus
 * whatever is still only staged. Nothing here is invented: every row is either a committed log
 * row (`policy_override_log`, `schedule_adjustment_log`, `execution_log`,
 * `review_resolution_log`) or a pending entry from the staging store, reconciled against those
 * same logs by the existing `/v1/staged-requests` endpoint.
 */
export function DecisionHistory({
  scope,
  targetInspectionId,
  refreshKey,
}: {
  scope: DecisionScope
  targetInspectionId: string
  refreshKey: number
}) {
  const hasSchedule = Boolean(scope.schedule_config_id)

  const overridesQuery = useApiQuery(
    (signal) =>
      listOverrides(scope, { target_inspection_id: targetInspectionId }, { offset: 0, limit: 50, descending: false }, signal),
    [JSON.stringify(scope), targetInspectionId, refreshKey],
    Boolean(scope.policy_id && scope.fold_set && scope.fold_id && scope.k_name),
  )
  const adjustmentsQuery = useApiQuery(
    (signal) =>
      listAdjustments(scope, { target_inspection_id: targetInspectionId }, { offset: 0, limit: 50, descending: false }, signal),
    [JSON.stringify(scope), targetInspectionId, refreshKey],
    hasSchedule,
  )
  const executionQuery = useApiQuery(
    (signal) =>
      listExecutionEvents(scope, { target_inspection_id: targetInspectionId }, { offset: 0, limit: 50, descending: false }, signal),
    [JSON.stringify(scope), targetInspectionId, refreshKey],
    hasSchedule,
  )
  const resolutionsQuery = useApiQuery(
    (signal) =>
      listResolutions(scope, { offset: 0, limit: 50, descending: false }, signal, {
        target_inspection_id: targetInspectionId,
      }),
    [JSON.stringify(scope), targetInspectionId, refreshKey],
    Boolean(scope.policy_id && scope.fold_set && scope.fold_id && scope.k_name),
  )
  const pendingQuery = useApiQuery(
    (signal) => listStagedRequests({}, signal),
    [refreshKey],
    true,
  )

  const loading = [overridesQuery, adjustmentsQuery, executionQuery, resolutionsQuery, pendingQuery].some(
    (q) => q.status === 'loading' || q.status === 'idle',
  )
  const anyError = [overridesQuery, adjustmentsQuery, executionQuery, resolutionsQuery].find(
    (q) => q.status === 'error',
  )

  if (loading) return <LoadingState label="Loading decision history…" />
  if (anyError && anyError.status === 'error') return <ErrorState error={anyError.error} />

  const entries: HistoryEntry[] = []

  if (overridesQuery.status === 'success') {
    for (const row of overridesQuery.data.data) {
      entries.push({
        key: `override-${row.override_id}`,
        when: row.decided_at,
        what: `Priority decision changed: ${overrideActionLabel(row.action)} (${row.outcome})`,
        actor: row.actor,
        reason: row.reason_code,
        state: 'committed',
      })
    }
  }
  if (adjustmentsQuery.status === 'success') {
    for (const row of adjustmentsQuery.data.data) {
      entries.push({
        key: `adjustment-${row.adjustment_id}`,
        when: row.decided_at,
        what: `Schedule adjusted: ${adjustmentActionLabel(row.action)} (${row.outcome})`,
        actor: row.actor,
        reason: row.reason_code,
        state: 'committed',
      })
    }
  }
  if (executionQuery.status === 'success') {
    for (const row of executionQuery.data.data) {
      entries.push({
        key: `execution-${row.execution_id}`,
        when: row.observed_at,
        what: `Inspection outcome recorded: ${executionStatusLabel(row.execution_status)}`,
        actor: row.actor,
        reason: row.reason_code,
        state: 'committed',
      })
    }
  }
  if (resolutionsQuery.status === 'success') {
    for (const row of resolutionsQuery.data.data) {
      entries.push({
        key: `resolution-${row.review_id}`,
        when: row.decided_at,
        what: `Review case resolved: ${resolutionActionLabel(row.resolution_action)} (${row.outcome})`,
        actor: row.actor,
        reason: row.reason_code,
        state: 'committed',
      })
    }
  }
  if (pendingQuery.status === 'success') {
    for (const req of pendingQuery.data) {
      if (req.status !== 'pending') continue
      if (req.payload.target_inspection_id !== targetInspectionId) continue
      const label =
        req.kind === 'override'
          ? `Priority decision changed: ${overrideActionLabel(String(req.payload.action))}`
          : req.kind === 'adjustment'
            ? `Schedule adjusted: ${adjustmentActionLabel(String(req.payload.action))}`
            : req.kind === 'execution_event'
              ? `Inspection outcome recorded: ${executionStatusLabel(String(req.payload.execution_status))}`
              : `Review case resolved: ${resolutionActionLabel(String(req.payload.resolution_action))}`
      entries.push({
        key: `pending-${req.request_id}`,
        when: req.staged_at,
        what: label,
        actor: String(req.payload.actor ?? ''),
        reason: String(req.payload.reason_code ?? ''),
        state: 'pending',
      })
    }
  }

  entries.sort((a, b) => (a.when < b.when ? 1 : -1))

  if (entries.length === 0) {
    return <EmptyState message="No overrides, adjustments, outcomes or resolutions recorded for this establishment yet." />
  }

  return (
    <ol className="decision-history">
      {entries.map((entry) => (
        <li key={entry.key} className={entry.state === 'pending' ? 'decision-history-pending' : ''}>
          <span className={`chip ${entry.state === 'pending' ? 'chip-attention' : 'chip-positive'}`}>
            {entry.state === 'pending' ? 'Staged, not yet applied' : 'Applied'}
          </span>
          <p>{entry.what}</p>
          <p className="hint">
            {entry.reason ? `“${entry.reason}” — ` : ''}
            {entry.actor}, {formatDateTime(entry.when)}
          </p>
        </li>
      ))}
    </ol>
  )
}
