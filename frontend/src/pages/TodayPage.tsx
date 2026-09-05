import { Link, useNavigate } from 'react-router-dom'
import { getPlanSummary, listPlanRows } from '../api/planReview'
import type { PlanRowOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { currentOperationalDate } from '../lib/today'
import {
  approvalStatusLabel,
  decisionReasonLabel,
  formatPriorityScore,
  historyFactorSummary,
  planLabelForToday,
  planStalenessNote,
} from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { TechnicalDetails } from '../components/common/TechnicalDetails'
import { SummaryCard } from '../components/common/SummaryCard'

const FETCH_CAP = 500

/**
 * The live, operational "what should I inspect today" experience -- Components 17-21, scoped
 * by a real `planning_date`, never by a historical evaluation fold. Reads the same no-scope
 * "latest built plan" endpoints `GeographicPlanPage`/`SupervisorPlanReviewPage` already use;
 * this page adds no new backend call and computes nothing itself.
 *
 * "Today" is never assumed -- `planLabelForToday`/`planStalenessNote` compare the plan's own
 * `planning_date` against the real current date (`lib/today.ts`) and say plainly when they
 * don't match, rather than silently relabeling a stale plan as today's.
 */
export function TodayPage() {
  const navigate = useNavigate()

  const summaryQuery = useApiQuery((signal) => getPlanSummary(undefined, signal), [], true)
  const rowsQuery = useApiQuery(
    (signal) => listPlanRows(undefined, { offset: 0, limit: FETCH_CAP, descending: false }, signal),
    [],
    true,
  )

  const loading = summaryQuery.status === 'loading' || rowsQuery.status === 'loading'
  const errored = summaryQuery.status === 'error' ? summaryQuery : rowsQuery.status === 'error' ? rowsQuery : null

  const current = currentOperationalDate()
  const planningDate = summaryQuery.status === 'success' ? summaryQuery.data.planning_date : undefined
  const isToday = planningDate === current

  const rows: PlanRowOut[] =
    rowsQuery.status === 'success' ? [...rowsQuery.data.data].sort((a, b) => a.rank - b.rank) : []

  function goToEstablishment(row: PlanRowOut) {
    navigate(`/plan/establishments/${encodeURIComponent(row.target_inspection_id)}`)
  }

  return (
    <PageShell>
      {loading && <LoadingState label="Preparing today's inspection plan…" />}
      {errored && <ErrorState error={errored.error} />}

      {!loading && !errored && summaryQuery.status === 'success' && rowsQuery.status === 'success' && (
        <>
          <h1 className="today-headline">{planLabelForToday(planningDate, isToday)}</h1>

          <div className="summary-grid">
            <SummaryCard
              label="Establishments"
              value={summaryQuery.data.selected_inspection_workload}
              hint="Selected for this plan, in priority order below"
            />
            <SummaryCard
              label="Work areas"
              value={summaryQuery.data.work_block_count}
              hint="Geographic groupings of today's establishments"
              to="/geographic-plan"
            />
            <SummaryCard
              label="Awaiting your decision"
              value={Math.max(
                0,
                summaryQuery.data.selected_inspection_workload - summaryQuery.data.decisions_recorded,
              )}
              hint="Establishments in this plan with no supervisor decision recorded yet"
              to="/plan-review"
              variant="attention"
            />
            <SummaryCard
              label="Plan status"
              value={approvalStatusLabel(summaryQuery.data.approval_status)}
              hint="Whether a supervisor has confirmed this plan"
              to="/plan-review"
            />
          </div>

          {rows.length === 0 && <EmptyState message="No establishments in today's plan yet." />}

          {rows.length > 0 && (
            <ol className="today-list">
              {rows.map((row) => (
                <li key={row.target_inspection_id} className="today-row" onClick={() => goToEstablishment(row)}>
                  <span className="today-row-slot">#{row.rank}</span>
                  <span className="today-row-identity">
                    <EstablishmentIdentity
                      name={row.establishment_name}
                      address={row.establishment_address}
                      establishmentId={row.establishment_id}
                    />
                  </span>
                  <span className="today-row-why">
                    {decisionReasonLabel(row.selection_reason)}
                    {' — '}
                    {formatPriorityScore(row.calibrated_score)}
                    {row.history_factors && historyFactorSummary(row.history_factors)[0] && (
                      <> · {historyFactorSummary(row.history_factors)[0]}</>
                    )}
                  </span>
                </li>
              ))}
            </ol>
          )}

          <p className="next-actions">
            <Link to="/geographic-plan">See the full field plan by work area →</Link>
            {' · '}
            <Link to="/plan-review">Go to Plan Review →</Link>
          </p>

          <TechnicalDetails summary="About this plan">
            <p className="hint">{planStalenessNote(planningDate ?? current, current)}</p>
            <p className="hint">
              Sentinel is not connected to a live feed of new inspections -- this plan is built
              from the most recent Chicago inspection data ingest and a fixed prioritization
              model, run explicitly for one planning date at a time.
            </p>
          </TechnicalDetails>
        </>
      )}
    </PageShell>
  )
}
