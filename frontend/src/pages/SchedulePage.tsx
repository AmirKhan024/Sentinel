import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  getCapacityUtilization,
  getPriorityPreservation,
  getScheduleSummary,
  listReplanningRuns,
  listSchedule,
} from '../api/schedule'
import { getExecutionSummary } from '../api/execution'
import type { ScheduleRowOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { useDecisionScope } from '../hooks/useDecisionScope'
import { useDefaultScope } from '../hooks/useDefaultScope'
import { useManifestOptions } from '../hooks/useManifestOptions'
import { capacityHonestyNote, formatDate, scheduleReasonLabel, scheduleStatusLabel } from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { InspectionPlanSelector } from '../components/scope/InspectionPlanSelector'
import { BacktestBanner } from '../components/common/BacktestBanner'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { DataTable, type Column } from '../components/common/DataTable'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { PaginationControls } from '../components/common/PaginationControls'
import { TechnicalDetails } from '../components/common/TechnicalDetails'

const REQUIRED_SCOPE = ['schedule_config_id', 'policy_id', 'fold_set', 'fold_id', 'k_name'] as const

function SmallTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (rows.length === 0) return <EmptyState message="Nothing to show for this plan." />
  const columns = Object.keys(rows[0])
  return (
    <table className="data-table">
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {columns.map((c) => (
              <td key={c}>{String(row[c] ?? '—')}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function SchedulePage() {
  const { scope, setScopeField, setScopeFields, missingFields } = useDecisionScope()
  const { manifests } = useManifestOptions(['policy', 'scheduling'])
  useDefaultScope(scope, setScopeFields, manifests)
  const navigate = useNavigate()

  const [establishmentId, setEstablishmentId] = useState('')
  const [offset, setOffset] = useState(0)
  const [limit] = useState(50)
  const [descending, setDescending] = useState(false)

  const missing = missingFields([...REQUIRED_SCOPE])
  const enabled = missing.length === 0

  const capacityMode = manifests.scheduling?.config_grid?.find(
    (c) => c.schedule_config_id === scope.schedule_config_id,
  )?.capacity_mode

  const query = useApiQuery(
    (signal) =>
      listSchedule(
        scope,
        { establishment_id: establishmentId || undefined },
        { offset, limit, descending },
        signal,
      ),
    [JSON.stringify(scope), establishmentId, offset, limit, descending],
    enabled,
  )

  const summaryQuery = useApiQuery((signal) => getScheduleSummary(scope, signal), [JSON.stringify(scope)], enabled)
  const capacityQuery = useApiQuery(
    (signal) => getCapacityUtilization(scope, signal),
    [JSON.stringify(scope)],
    enabled,
  )
  const priorityQuery = useApiQuery(
    (signal) => getPriorityPreservation(scope, signal),
    [JSON.stringify(scope)],
    enabled,
  )
  const replanQuery = useApiQuery((signal) => listReplanningRuns(scope, signal), [JSON.stringify(scope)], enabled)
  const executionQuery = useApiQuery((signal) => getExecutionSummary(scope, signal), [JSON.stringify(scope)], enabled)

  const columns: Column<ScheduleRowOut>[] = [
    {
      key: 'establishment_id',
      label: 'Establishment',
      render: (r) => (
        <EstablishmentIdentity
          name={r.establishment_name}
          address={r.establishment_address}
          establishmentId={r.establishment_id}
        />
      ),
    },
    {
      key: 'schedule_status',
      label: 'Status',
      render: (r) => (
        <span className={r.schedule_status === 'scheduled' ? 'chip chip-positive' : 'chip chip-neutral'}>
          {scheduleStatusLabel(r.schedule_status)}
        </span>
      ),
    },
    { key: 'scheduled_date', label: 'Planned date', render: (r) => formatDate(r.scheduled_date) },
    {
      key: 'slot_index',
      label: 'Order that day',
      // `slot_index` is already 1-based on disk (see data/data_contracts/schedule.md) -- the
      // first slot placed on a day is 1, not 0. Do not add 1 here again.
      render: (r) => (r.slot_index !== null ? `#${r.slot_index}` : '—'),
    },
    { key: 'why', label: 'Why', render: (r) => scheduleReasonLabel(r.schedule_reason) },
  ]

  function goToEstablishment(row: ScheduleRowOut) {
    const params = new URLSearchParams({
      policy_id: scope.policy_id ?? '',
      fold_set: scope.fold_set ?? '',
      fold_id: scope.fold_id ?? '',
      k_name: scope.k_name ?? '',
      schedule_config_id: scope.schedule_config_id ?? '',
    })
    navigate(`/establishments/${encodeURIComponent(row.establishment_id)}?${params.toString()}`)
  }

  return (
    <PageShell
      title="Inspection Schedule"
      description="These are the recommended inspections that fit into the currently available inspection capacity."
    >
      <p className="hint">
        <Link to="/schedule/day">View one day at a time →</Link>
      </p>

      <InspectionPlanSelector
        scope={scope}
        setScopeField={setScopeField}
        requiredFields={[...REQUIRED_SCOPE]}
        manifests={manifests}
        showAdvanced
      />

      <BacktestBanner foldId={scope.fold_id} />

      <section className="explainer-box">
        <h3>How this schedule is created</h3>
        <p>
          Sentinel places prioritized establishments into each day's available inspection
          capacity, in priority order, until that day is full. <strong>It does not calculate
          travel routes, assign specific inspectors, or account for the distance between
          establishments.</strong> Two establishments scheduled back-to-back may be on opposite
          sides of the city — treat this as a capacity-aware inspection plan, not an optimized
          route.
        </p>
        <p className="hint">
          Planned dates reflect the inspection days Sentinel is evaluating for this plan, which
          may be a past period rather than the coming week — this is a prioritization plan, not a
          live calendar of upcoming appointments.
        </p>
        <h4>How capacity is estimated</h4>
        <p className="hint">{capacityHonestyNote(capacityMode)}</p>
      </section>

      <div className="filters">
        <label>
          Search by establishment
          <input
            value={establishmentId}
            onChange={(e) => setEstablishmentId(e.target.value)}
            placeholder="Establishment ID"
          />
        </label>
      </div>

      {!enabled && <LoadingState label="Preparing an inspection plan…" />}

      {enabled && query.status === 'loading' && <LoadingState />}
      {enabled && query.status === 'error' && <ErrorState error={query.error} />}
      {enabled && query.status === 'success' && query.data.data.length === 0 && (
        <EmptyState message="Nothing is scheduled for this plan yet." />
      )}
      {enabled && query.status === 'success' && query.data.data.length > 0 && (
        <>
          <DataTable columns={columns} rows={query.data.data} rowKey={(r) => r.target_inspection_id} onRowClick={goToEstablishment} />
          <PaginationControls
            page={query.data.page}
            onOffsetChange={setOffset}
            descending={descending}
            onDescendingChange={(d) => {
              setDescending(d)
              setOffset(0)
            }}
          />
        </>
      )}

      <TechnicalDetails summary="Technical details and capacity metrics">
        <section>
          <h3>Summary</h3>
          {summaryQuery.status === 'error' && <ErrorState error={summaryQuery.error} />}
          {summaryQuery.status === 'success' && <SmallTable rows={summaryQuery.data} />}
        </section>
        <section>
          <h3>Capacity utilization</h3>
          {capacityQuery.status === 'error' && <ErrorState error={capacityQuery.error} />}
          {capacityQuery.status === 'success' && <SmallTable rows={capacityQuery.data} />}
        </section>
        <section>
          <h3>Priority preservation</h3>
          {priorityQuery.status === 'error' && <ErrorState error={priorityQuery.error} />}
          {priorityQuery.status === 'success' && <SmallTable rows={priorityQuery.data} />}
        </section>
        <section>
          <h3>Replanning runs</h3>
          {replanQuery.status === 'error' && <ErrorState error={replanQuery.error} />}
          {replanQuery.status === 'success' && (
            <SmallTable rows={replanQuery.data as unknown as Record<string, unknown>[]} />
          )}
        </section>
        <section>
          <h3>Execution summary</h3>
          {executionQuery.status === 'error' && executionQuery.error.kind === 'client' && (
            <p className="state state-empty">No execution record for this plan.</p>
          )}
          {executionQuery.status === 'error' && executionQuery.error.kind !== 'client' && (
            <ErrorState error={executionQuery.error} />
          )}
          {executionQuery.status === 'success' && (
            <>
              <p className="hint">
                Only shown because a record exists for this plan — its absence never implies an
                inspection did or did not happen.
              </p>
              <ul>
                <li>Scheduled: {executionQuery.data.n_scheduled}</li>
                <li>Completed: {executionQuery.data.n_completed}</li>
                <li>Not performed: {executionQuery.data.n_not_performed}</li>
                <li>Cancelled in the field: {executionQuery.data.n_cancelled_in_field}</li>
                <li>No record yet: {executionQuery.data.n_no_execution_record}</li>
                <li>Completion rate: {(executionQuery.data.completion_rate * 100).toFixed(0)}%</li>
              </ul>
            </>
          )}
        </section>
      </TechnicalDetails>
    </PageShell>
  )
}
