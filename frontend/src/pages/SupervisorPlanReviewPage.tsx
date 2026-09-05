import { Fragment, useState } from 'react'
import { Link } from 'react-router-dom'
import { getPlanSummary, listPlanRows } from '../api/planReview'
import { getManifest } from '../api/meta'
import type { PlanRowOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import {
  approvalStatusLabel,
  decisionReasonLabel,
  formatDate,
  formatDateTime,
  operationalCoverageNote,
  planDecisionActionLabel,
  workAreaLabel,
  workBlockDisplayLabel,
} from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { TechnicalDetails } from '../components/common/TechnicalDetails'
import { PlanDecisionForm } from '../components/actions/PlanDecisionForm'
import { PlanApprovalPanel } from '../components/actions/PlanApprovalPanel'

const FETCH_CAP = 500

/**
 * Component 21: the supervisor's actual job -- see the proposed workload, understand why it
 * was organized this way, and record a decision that is kept *beside* Sentinel's own
 * recommendation, never in place of it.
 */
export function SupervisorPlanReviewPage() {
  const [openRowId, setOpenRowId] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  const summaryQuery = useApiQuery(
    (signal) => getPlanSummary(undefined, signal),
    [refreshKey],
    true,
  )
  const rowsQuery = useApiQuery(
    (signal) => listPlanRows(undefined, { offset: 0, limit: FETCH_CAP, descending: false }, signal),
    [refreshKey],
    true,
  )
  // Read-only, for the "what about everyone else" note below -- never recomputed here, only the
  // counts operational_selection's own manifest already wrote.
  const selectionManifestQuery = useApiQuery(
    (signal) => getManifest('operational_selection', signal),
    [],
    true,
  )

  const loading = summaryQuery.status === 'loading' || rowsQuery.status === 'loading'
  const errored = summaryQuery.status === 'error' ? summaryQuery : rowsQuery.status === 'error' ? rowsQuery : null

  const rowsByBlock = new Map<string, PlanRowOut[]>()
  if (rowsQuery.status === 'success') {
    for (const row of rowsQuery.data.data) {
      const list = rowsByBlock.get(row.work_block_id) ?? []
      list.push(row)
      rowsByBlock.set(row.work_block_id, list)
    }
    for (const list of rowsByBlock.values()) {
      list.sort((a, b) => (a.suggested_order_in_block ?? 0) - (b.suggested_order_in_block ?? 0))
    }
  }

  const planningDate = summaryQuery.status === 'success' ? summaryQuery.data.planning_date : undefined

  return (
    <PageShell
      title="Supervisor Plan Review"
      description="The proposed inspection workload, organized geographically. Review it, apply local knowledge, and record what you decide -- Sentinel's own recommendation is always kept alongside your decision, never replaced by it."
    >
      {loading && <LoadingState />}
      {errored && <ErrorState error={errored.error} />}
      {!loading && !errored && summaryQuery.status === 'success' && rowsQuery.status === 'success' && (
        <>
          {planningDate && <p className="hint">Plan for {formatDate(planningDate)}.</p>}

          <section className="plan-summary-card">
            <div>
              <span className="plan-summary-figure">{summaryQuery.data.selected_inspection_workload}</span>
              <span className="hint">establishments in the proposed workload</span>
            </div>
            <div>
              <span className="plan-summary-figure">
                {summaryQuery.data.location_available_count}/{summaryQuery.data.selected_inspection_workload}
              </span>
              <span className="hint">mapped to a geographic work block</span>
            </div>
            <div>
              <span className="plan-summary-figure">{summaryQuery.data.work_block_count}</span>
              <span className="hint">geographic work blocks</span>
            </div>
            <div>
              <span className="plan-summary-figure">{summaryQuery.data.decisions_recorded}</span>
              <span className="hint">supervisor decisions recorded</span>
            </div>
            <div>
              <span className="plan-summary-figure">{approvalStatusLabel(summaryQuery.data.approval_status)}</span>
            </div>
          </section>

          {selectionManifestQuery.status === 'success' &&
            selectionManifestQuery.data.ranked_candidate_count != null &&
            selectionManifestQuery.data.selectable_candidate_count != null &&
            selectionManifestQuery.data.selected_count != null && (
              <p className="hint">
                {operationalCoverageNote(
                  selectionManifestQuery.data.ranked_candidate_count,
                  selectionManifestQuery.data.selectable_candidate_count,
                  selectionManifestQuery.data.selected_count,
                )}
              </p>
            )}

          {planningDate && (
            <PlanApprovalPanel
              planningDate={planningDate}
              decisionsRecorded={summaryQuery.data.decisions_recorded}
              totalEstablishments={summaryQuery.data.selected_inspection_workload}
              onStaged={() => setRefreshKey((k) => k + 1)}
            />
          )}

          {rowsQuery.data.data.length === 0 && (
            <EmptyState message="No establishments in this plan yet." />
          )}

          <div className="work-block-list">
            {[...rowsByBlock.entries()].map(([blockId, rows], index) => (
              <section key={blockId} className="work-block-card">
                <header>
                  <h2>{workAreaLabel(workBlockDisplayLabel(rows[0]?.work_block_label, index))}</h2>
                  <span className="hint">
                    {rows.length} establishment{rows.length === 1 ? '' : 's'}
                    {rows[0]?.highest_sentinel_rank_in_block != null && (
                      <> · highest priority #{rows[0].highest_sentinel_rank_in_block}</>
                    )}
                  </span>
                </header>
                <table className="plan-row-table">
                  <tbody>
                    {rows.map((row) => (
                      <Fragment key={row.target_inspection_id}>
                        <tr
                          className={openRowId === row.target_inspection_id ? 'open' : ''}
                          onClick={() =>
                            setOpenRowId(
                              openRowId === row.target_inspection_id ? null : row.target_inspection_id,
                            )
                          }
                        >
                          <td>
                            {row.policy_rank != null ? `#${row.policy_rank}` : '—'}
                            {row.operational_priority != null &&
                              row.operational_priority !== row.policy_rank && (
                                <div className="hint">field-work order #{row.operational_priority}</div>
                              )}
                          </td>
                          <td>
                            <Link
                              to={`/plan/establishments/${encodeURIComponent(row.target_inspection_id)}`}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <EstablishmentIdentity
                                name={row.establishment_name}
                                address={row.establishment_address}
                                establishmentId={row.establishment_id}
                              />
                            </Link>
                          </td>
                          <td className="hint">{decisionReasonLabel(row.selection_reason)}</td>
                          <td>
                            {row.supervisor_decision_action ? (
                              <span className="supervisor-decision-badge">
                                {planDecisionActionLabel(row.supervisor_decision_action)}
                              </span>
                            ) : (
                              <span className="hint">
                                {summaryQuery.data.approval_status === 'approved'
                                  ? 'No decision was recorded before this plan was approved'
                                  : 'No decision recorded yet'}
                              </span>
                            )}
                          </td>
                        </tr>
                        {openRowId === row.target_inspection_id && (
                          <tr>
                            <td colSpan={4}>
                              <div className="plan-row-detail">
                                <p>
                                  <strong>Sentinel recommendation.</strong> Selected, rank #
                                  {row.policy_rank ?? '—'} of the plan. {decisionReasonLabel(row.selection_reason)}
                                </p>
                                {row.operational_priority != null &&
                                  row.operational_priority !== row.policy_rank && (
                                    <p>
                                      <strong>Field-work order.</strong> A supervisor set this
                                      establishment's field-work order to #{row.operational_priority}.
                                      Sentinel's own risk rank (#{row.policy_rank ?? '—'}) is
                                      unchanged.
                                    </p>
                                  )}
                                {row.supervisor_decision_action && (
                                  <p>
                                    <strong>Supervisor decision.</strong>{' '}
                                    {planDecisionActionLabel(row.supervisor_decision_action)}
                                    {row.supervisor_decision_reason_code && <> — {row.supervisor_decision_reason_code}</>}
                                    {row.supervisor_decision_actor && (
                                      <>
                                        {' '}
                                        (by {row.supervisor_decision_actor}
                                        {row.supervisor_decision_decided_at &&
                                          `, ${formatDateTime(row.supervisor_decision_decided_at)}`}
                                        )
                                      </>
                                    )}
                                  </p>
                                )}
                                {planningDate && (
                                  <PlanDecisionForm
                                    planningDate={planningDate}
                                    targetInspectionId={row.target_inspection_id}
                                    planAlreadyApproved={summaryQuery.data.approval_status === 'approved'}
                                    onStaged={() => setRefreshKey((k) => k + 1)}
                                  />
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </section>
            ))}
          </div>

          <TechnicalDetails summary="Machine recommendation vs. human decision">
            <p className="hint">
              Sentinel's own recommendation (rank, selection reason, geographic work block) is
              never edited by a supervisor decision -- both are kept, side by side, permanently.
              A decision is staged, not applied immediately (see ADR 0049): an operator applies
              staged decisions the next time this planning run is rebuilt. A
              "do not proceed as planned" decision does not remove the establishment from the
              plan; it only records the supervisor's intent.
            </p>
          </TechnicalDetails>
        </>
      )}
    </PageShell>
  )
}
