import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { getPlanRow, getPlanSummary } from '../api/planReview'
import { useApiQuery } from '../hooks/useApiQuery'
import {
  approvalStatusLabel,
  decisionReasonLabel,
  formatDate,
  formatDateTime,
  planDecisionActionLabel,
  workAreaLabel,
} from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { TechnicalDetails } from '../components/common/TechnicalDetails'
import { FieldRow } from '../components/common/FieldRow'
import { PlanDecisionForm } from '../components/actions/PlanDecisionForm'

/**
 * The establishment detail view for the live operational plan (Components 17-21) -- reached
 * from Today, Field Plan, and Plan Review. Deliberately not the same page as
 * `EstablishmentDetailPage.tsx`: that page's data (`GET /v1/establishments/{id}`) comes from
 * Component 13's fold-scoped historical recommendation table, and an establishment selected for
 * a live `planning_date` has no guaranteed home there -- routing Side-B clicks through it is
 * exactly what caused "We couldn't find that record" for real establishments. This page reads
 * only `GET /v1/plan-review/rows/{target_inspection_id}` (Component 21's own single-row
 * endpoint), which is authoritative for every establishment in the current plan.
 */
export function EstablishmentPlanDetailPage() {
  const { targetInspectionId = '' } = useParams()
  const [refreshKey, setRefreshKey] = useState(0)

  const query = useApiQuery(
    (signal) => getPlanRow(targetInspectionId, undefined, signal),
    [targetInspectionId, refreshKey],
    Boolean(targetInspectionId),
  )
  // Read-only, so this page states the same plan-level approval status Today/Plan Review
  // show -- an establishment's own row carries no approval field of its own.
  const summaryQuery = useApiQuery((signal) => getPlanSummary(undefined, signal), [refreshKey], true)
  const isApproved = summaryQuery.status === 'success' && summaryQuery.data.approval_status === 'approved'

  return (
    <PageShell>
      {query.status === 'loading' && <LoadingState label="Loading this establishment's place in the plan…" />}
      {query.status === 'error' && <ErrorState error={query.error} />}

      {query.status === 'success' && (
        <>
          <h1>
            <EstablishmentIdentity
              name={query.data.establishment_name}
              address={query.data.establishment_address}
              establishmentId={query.data.establishment_id}
            />
          </h1>
          <p className="hint">
            Plan for {formatDate(query.data.planning_date)}.
            {summaryQuery.status === 'success' && (
              <> Plan status: {approvalStatusLabel(summaryQuery.data.approval_status)}.</>
            )}
          </p>

          <section>
            <h2>Sentinel priority</h2>
            <p>
              <strong>Priority #{query.data.rank}</strong> in today's plan
            </p>
            <p className="hint">{decisionReasonLabel(query.data.selection_reason)}</p>
            {query.data.operational_priority != null && query.data.operational_priority !== query.data.policy_rank && (
              <p className="hint">
                A supervisor set this establishment's field-work order to #
                {query.data.operational_priority}. Sentinel's own priority (#{query.data.rank}) is
                unchanged.
              </p>
            )}
          </section>

          <section>
            <h2>Current plan</h2>
            <p>Selected for today's plan.</p>
            <p className="hint">
              {query.data.location_status === 'location_available'
                ? `Grouped with: ${workAreaLabel(query.data.work_block_label || 'a nearby work area')}.`
                : 'Location unavailable -- not placed into a geographic work area.'}
            </p>
          </section>

          <section>
            <h2>Your decision</h2>
            {query.data.supervisor_decision_action ? (
              <p>
                <strong>Supervisor decision.</strong>{' '}
                {planDecisionActionLabel(query.data.supervisor_decision_action)}
                {query.data.supervisor_decision_reason_code && (
                  <> — {query.data.supervisor_decision_reason_code}</>
                )}
                {query.data.supervisor_decision_actor && (
                  <>
                    {' '}
                    (by {query.data.supervisor_decision_actor}
                    {query.data.supervisor_decision_decided_at &&
                      `, ${formatDateTime(query.data.supervisor_decision_decided_at)}`}
                    )
                  </>
                )}
              </p>
            ) : (
              <p className="hint">No decision recorded yet.</p>
            )}
            <PlanDecisionForm
              planningDate={query.data.planning_date}
              targetInspectionId={query.data.target_inspection_id}
              planAlreadyApproved={isApproved}
              onStaged={() => setRefreshKey((k) => k + 1)}
            />
          </section>

          <TechnicalDetails summary="More inspection history">
            <p className="hint">
              Detailed historical inspection factors and a feature-by-feature model explanation
              are not currently available for establishments in a live operational plan --
              Sentinel does not run its explainability step (Component 11) for live scoring, and
              the underlying feature values are not exposed through any API today. This is a
              genuine, honest limitation, not a hidden field.
            </p>
          </TechnicalDetails>

          <TechnicalDetails summary="Technical details">
            <FieldRow label="target_inspection_id" value={query.data.target_inspection_id} />
            <FieldRow label="establishment_id" value={query.data.establishment_id} />
            <FieldRow label="calibrated_score" value={query.data.calibrated_score.toFixed(4)} />
            <FieldRow label="base_score" value={query.data.base_score.toFixed(4)} />
            <FieldRow label="selection_mechanism" value={query.data.selection_mechanism} />
            <FieldRow label="organization_mode" value={query.data.organization_mode} />
          </TechnicalDetails>
        </>
      )}
    </PageShell>
  )
}
