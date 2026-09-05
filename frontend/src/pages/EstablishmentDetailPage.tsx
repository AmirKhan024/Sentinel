import { useState, type ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getSelectionAllocation } from '../api/recommendations'
import { getEstablishment } from '../api/establishments'
import { getReviewCase } from '../api/review'
import type { EstablishmentHistoryOut, ReviewCaseOut, StagedRequestReceipt } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { useDecisionScope } from '../hooks/useDecisionScope'
import { useDefaultScope } from '../hooks/useDefaultScope'
import { useManifestOptions } from '../hooks/useManifestOptions'
import {
  HOW_TO_USE_PRIORITY,
  capacityHonestyNote,
  decisionReasonLabel,
  establishmentDisplayName,
  formatDate,
  historyFactorSummary,
  relativePriorityLabel,
  reviewStatusLabel,
  reviewTriggerLabels,
  scheduleReasonLabel,
  scheduleStatusLabel,
  selectionStatusHint,
  selectionStatusLabel,
  warningLabels,
} from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { InspectionPlanSelector } from '../components/scope/InspectionPlanSelector'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { FieldRow } from '../components/common/FieldRow'
import { TechnicalDetails } from '../components/common/TechnicalDetails'
import { OverrideForm } from '../components/actions/OverrideForm'
import { AdjustmentForm } from '../components/actions/AdjustmentForm'
import { ExecutionOutcomeForm } from '../components/actions/ExecutionOutcomeForm'
import { ResolutionForm } from '../components/actions/ResolutionForm'
import { DecisionHistory } from '../components/actions/DecisionHistory'

const REQUIRED_SCOPE = ['policy_id', 'fold_set', 'fold_id', 'k_name'] as const

/**
 * This page's data (`GET /v1/establishments/{id}`) comes from Component 13's historical,
 * fold-scoped recommendation table -- a genuinely smaller population than every real
 * establishment Sentinel knows about. An establishment can be real, resolved (Component 2), and
 * even part of *today's live plan* (Component 17-21) while never once appearing in any
 * historical backtest fold, because a fold only includes establishments with a qualifying
 * inspection during that fold's own historical test window. A "not found" here is therefore a
 * structural property of which population this page reads, not a transient bug -- confirmed
 * directly against real data: the same establishment can 404 here while resolving correctly at
 * `/plan/establishments/:targetInspectionId` (see `EstablishmentPlanDetailPage.tsx`). Do not
 * "fix" this by relaxing required scope -- verified that every scope field (including `k_name`)
 * has the *same* considered population for a given fold; only `is_selected` varies by `k_name`.
 */
function isRowNotFoundError(error: { kind: string; error?: string }): boolean {
  return error.kind === 'client' && (error.error === 'row_not_found' || error.error === 'artifact_not_found')
}

function JourneyStep({
  title,
  status,
  children,
}: {
  title: string
  status: 'done' | 'active' | 'waiting' | 'attention'
  children: ReactNode
}) {
  return (
    <li className={`journey-step journey-step-${status}`}>
      <span className="journey-step-marker" aria-hidden="true" />
      <div className="journey-step-body">
        <h3>{title}</h3>
        {children}
      </div>
    </li>
  )
}

export function EstablishmentDetailPage() {
  const { establishmentId = '' } = useParams()
  const { scope, setScopeField, setScopeFields, missingFields } = useDecisionScope()
  const { manifests } = useManifestOptions(['policy', 'scheduling'])
  useDefaultScope(scope, setScopeFields, manifests)

  const missing = missingFields([...REQUIRED_SCOPE])
  const enabled = missing.length === 0

  // Bumped after any staged write so the review-case lookup and the decision-history panel
  // both refetch -- they can't see their own change applied to the plan (ADR 0049 forbids
  // that), but they can see it appear as a new staged/pending entry.
  const [refreshKey, setRefreshKey] = useState(0)
  const [stagedOverrideId, setStagedOverrideId] = useState<string | undefined>(undefined)
  const [stagedAdjustmentId, setStagedAdjustmentId] = useState<string | undefined>(undefined)

  function noteStaged(receipt: StagedRequestReceipt) {
    if (receipt.kind === 'override') setStagedOverrideId(receipt.natural_id)
    if (receipt.kind === 'adjustment') setStagedAdjustmentId(receipt.natural_id)
    setRefreshKey((k) => k + 1)
  }

  const query = useApiQuery(
    (signal) => getEstablishment(establishmentId, scope, signal),
    [establishmentId, JSON.stringify(scope)],
    enabled,
  )

  // A later scope enrichment (e.g. `useDefaultScope` filling in `schedule_config_id` slightly
  // after the establishment itself has already loaded, so a fuller record can be re-fetched
  // with a schedule attached) makes `query` cycle back through 'loading'. Gating the journey on
  // `query.status === 'success'` directly would unmount it -- and every action form inside it,
  // discarding whatever a person had already typed -- for a background refresh they never asked
  // for. `data` is the last successful payload and only ever moves forward; the journey renders
  // from it, not from `query` directly, so a background refetch updates it in place instead of
  // tearing it down.
  const [data, setData] = useState<EstablishmentHistoryOut | null>(null)
  if (query.status === 'success' && query.data !== data) {
    // Adjusting state during render, not in an effect: React re-renders immediately with the
    // new value before committing to the screen, so this never paints a stale frame -- see
    // "Storing information from previous renders" in the React docs.
    setData(query.data)
  }

  const targetId = data?.recommendation.target_inspection_id
  const reviewQuery = useApiQuery(
    (signal) => getReviewCase(targetId!, scope, signal),
    [targetId, JSON.stringify(scope), refreshKey],
    enabled && Boolean(targetId),
  )
  // Same reasoning as `data` above: a background refetch (scope enrichment, or the deliberate
  // `refreshKey` bump after staging a write) must not unmount the ResolutionForm mid-use.
  const [reviewData, setReviewData] = useState<ReviewCaseOut | null>(null)
  const [reviewNotFlagged, setReviewNotFlagged] = useState(false)
  if (reviewQuery.status === 'success' && reviewQuery.data !== reviewData) {
    setReviewData(reviewQuery.data)
    setReviewNotFlagged(false)
  }
  const queryIsNotFlagged =
    reviewQuery.status === 'error' && reviewQuery.error.kind === 'client' && reviewQuery.error.status === 404
  if (queryIsNotFlagged && (!reviewNotFlagged || reviewData !== null)) {
    setReviewNotFlagged(true)
    setReviewData(null)
  }
  const notFlagged = reviewData === null && reviewNotFlagged

  const allocationQuery = useApiQuery(
    (signal) => getSelectionAllocation(scope, signal),
    [JSON.stringify(scope)],
    enabled,
  )
  const modelName = data?.recommendation.model_name
  const allocation =
    allocationQuery.status === 'success'
      ? (allocationQuery.data.find((a) => a.model_name === modelName) ?? allocationQuery.data[0])
      : undefined

  const capacityMode = manifests.scheduling?.config_grid?.find(
    (c) => c.schedule_config_id === scope.schedule_config_id,
  )?.capacity_mode

  return (
    <PageShell
      title={establishmentDisplayName(data?.establishment_name, establishmentId)}
      description="Where this establishment is in Sentinel's inspection process, and why."
    >
      {data && data.establishment_address && (
        <p className="hint establishment-detail-subtitle">{data.establishment_address}</p>
      )}

      <InspectionPlanSelector
        scope={scope}
        setScopeField={setScopeField}
        requiredFields={[...REQUIRED_SCOPE, 'schedule_config_id']}
        manifests={manifests}
      />

      {!enabled && <LoadingState label="Preparing an inspection plan…" />}
      {enabled && !data && query.status === 'loading' && <LoadingState />}
      {enabled && !data && query.status === 'error' && isRowNotFoundError(query.error) && (
        <div className="state state-error" role="alert">
          <p>
            This establishment doesn't have a recommendation for the selected historical
            period — it may not have had a qualifying inspection during that period's window.
          </p>
          <p className="hint">
            Try a different historical period above, or{' '}
            <Link to="/geographic-plan">look it up in today's live field plan</Link> or{' '}
            <Link to="/plan-review">plan review</Link> instead.
          </p>
        </div>
      )}
      {enabled && !data && query.status === 'error' && !isRowNotFoundError(query.error) && (
        <ErrorState error={query.error} />
      )}

      {enabled && data && (
        <ol className="journey">
          <JourneyStep title="1. Available information" status="done">
            <p>Sentinel evaluated the inspection information available for this establishment.</p>
          </JourneyStep>

          <JourneyStep title="2. Priority position and evidence" status="done">
            <p>
              <strong>
                {allocation
                  ? relativePriorityLabel(data.recommendation.model_rank, allocation.n_universe)
                  : `Ranked ${data.recommendation.model_rank}`}
              </strong>
            </p>
            <p className="hint">
              This position is fixed by the risk ranking alone, before any capacity cutoff is
              applied -- it does not change when this plan's capacity changes.
            </p>
            <div className="why-recommended">
              <h4>What Sentinel's history for this establishment shows</h4>
              {data.history_factors ? (
                historyFactorSummary(data.history_factors).length > 0 ? (
                  <ul className="journey-notes">
                    {historyFactorSummary(data.history_factors).map((note, i) => (
                      <li key={i}>{note}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint">
                    Sentinel has very little recorded history for this establishment — that
                    absence is itself part of what the model saw.
                  </p>
                )
              ) : (
                <p className="hint">
                  {data.history_factors_unavailable_reason ??
                    "The specific inspection-history facts behind this score aren't available for this establishment."}
                </p>
              )}
              <p className="hint">
                These are facts Sentinel considered, not a claim that any one of them caused this
                position — see "Model explanation" in Technical details for how much each factor
                measurably moved the score, when available for this establishment.
              </p>
            </div>
            {warningLabels(data.recommendation.warnings).length > 0 && (
              <ul className="journey-notes">
                {warningLabels(data.recommendation.warnings).map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
            <TechnicalDetails summary="How Sentinel prioritizes locations">
              <p>{HOW_TO_USE_PRIORITY}</p>
            </TechnicalDetails>
          </JourneyStep>

          <JourneyStep
            title="3. Selected for this plan?"
            status={data.recommendation.is_selected ? 'active' : 'waiting'}
          >
            <p>
              <span
                className={
                  data.recommendation.is_selected ? 'chip chip-positive' : 'chip chip-neutral'
                }
              >
                {selectionStatusLabel(data.recommendation.is_selected)}
              </span>
            </p>
            <p className="hint">{selectionStatusHint(data.recommendation.is_selected)}</p>
            <p>{decisionReasonLabel(data.recommendation.decision_reason)}</p>
            <div className="explainer-box">
              <h4>Human decision</h4>
              <p className="hint">
                Sentinel prioritizes; it does not decide alone. If local knowledge says this plan's
                decision should be different, record that here.
              </p>
              <OverrideForm
                scope={scope}
                targetInspectionId={data.recommendation.target_inspection_id}
                isSelected={data.recommendation.is_selected}
                onStaged={noteStaged}
              />
            </div>
          </JourneyStep>

          <JourneyStep
            title="4. Schedule"
            status={
              data.schedule?.schedule_status === 'scheduled'
                ? 'active'
                : data.schedule
                  ? 'waiting'
                  : 'waiting'
            }
          >
            {data.schedule ? (
              <>
                <p>
                  <span
                    className={
                      data.schedule.schedule_status === 'scheduled' ? 'chip chip-positive' : 'chip chip-neutral'
                    }
                  >
                    {scheduleStatusLabel(data.schedule.schedule_status)}
                  </span>
                </p>
                <p>{scheduleReasonLabel(data.schedule.schedule_reason)}</p>
                {data.schedule.scheduled_date && (
                  <p>Planned date: {formatDate(data.schedule.scheduled_date)}</p>
                )}
                <p className="hint">
                  This is a capacity-aware planning placement, not a geographically optimized
                  route. Sentinel does not calculate travel time or assign inspectors.
                </p>
                <p className="hint">{capacityHonestyNote(capacityMode)}</p>
                <div className="explainer-box">
                  <h4>Adjust this planned inspection</h4>
                  <AdjustmentForm
                    scope={scope}
                    targetInspectionId={data.recommendation.target_inspection_id}
                    scheduleConfigId={data.schedule.schedule_config_id}
                    onStaged={noteStaged}
                  />
                </div>
                <div className="explainer-box">
                  <h4>Record inspection outcome</h4>
                  <ExecutionOutcomeForm
                    scope={scope}
                    targetInspectionId={data.recommendation.target_inspection_id}
                    scheduleConfigId={data.schedule.schedule_config_id}
                    defaultScheduledDate={data.schedule.scheduled_date}
                    onStaged={noteStaged}
                  />
                </div>
              </>
            ) : (
              <p className="hint">
                Not scheduled — either no schedule was selected above, or this establishment has
                no schedule entry yet.
              </p>
            )}
          </JourneyStep>

          <JourneyStep
            title="5. Human review"
            status={reviewData ? (reviewData.review_status === 'resolved' ? 'done' : 'attention') : 'waiting'}
          >
            {reviewQuery.status === 'loading' && !reviewData && !notFlagged && <LoadingState />}
            {notFlagged && (
              <p className="hint">This establishment does not currently need human review.</p>
            )}
            {reviewQuery.status === 'error' && !notFlagged && !reviewData && (
              <ErrorState error={reviewQuery.error} />
            )}
            {reviewData && (
              <>
                <p>
                  <span
                    className={
                      reviewData.review_status === 'resolved' ? 'chip chip-positive' : 'chip chip-attention'
                    }
                  >
                    {reviewStatusLabel(reviewData.review_status)}
                  </span>
                </p>
                <ul className="journey-notes">
                  {reviewTriggerLabels(reviewData.trigger_reasons).map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
                <p className="hint">
                  A review flag is not a verdict that Sentinel made a wrong decision.
                </p>
                {reviewData.review_status !== 'resolved' && (
                  <div className="explainer-box">
                    <ResolutionForm
                      scope={scope}
                      targetInspectionId={data.recommendation.target_inspection_id}
                      reviewId={reviewData.review_id}
                      prefillOverrideId={stagedOverrideId}
                      prefillAdjustmentId={stagedAdjustmentId}
                      onStaged={noteStaged}
                    />
                  </div>
                )}
              </>
            )}
          </JourneyStep>

          <JourneyStep title="6. Decision history" status="done">
            <p className="hint">
              Every override, schedule adjustment, recorded outcome and review resolution for this
              establishment, applied and staged alike.
            </p>
            <DecisionHistory
              scope={scope}
              targetInspectionId={data.recommendation.target_inspection_id}
              refreshKey={refreshKey}
            />
          </JourneyStep>

          <JourneyStep title="7. Current field plan" status="done">
            <p className="hint">
              This is Sentinel's historical analysis for a past planning period. If this
              establishment is part of today's live operational plan, its geographic work area
              and any supervisor decision appear there instead.
            </p>
            <p>
              <Link to="/geographic-plan">See the current field plan →</Link>
              {' · '}
              <Link to="/plan-review">See the current plan review →</Link>
            </p>
          </JourneyStep>
        </ol>
      )}

      {enabled && data && (
        <TechnicalDetails summary="Technical details">
          <section>
            <h4>Model</h4>
            <FieldRow label="base_score" value={data.recommendation.base_score.toFixed(4)} />
            <FieldRow label="calibrated score" value={data.recommendation.score.toFixed(4)} />
            <FieldRow label="model_name" value={data.recommendation.model_name} />
            <FieldRow label="model_rank" value={data.recommendation.model_rank} />
            {allocation && <FieldRow label="n_universe" value={allocation.n_universe} />}
          </section>
          <section>
            <h4>Policy</h4>
            <FieldRow label="final_policy_rank" value={data.recommendation.final_policy_rank} />
            <FieldRow label="decision_mechanism" value={data.recommendation.decision_mechanism} />
            <FieldRow label="decision_reason" value={data.recommendation.decision_reason} />
            <FieldRow label="coverage_eligible" value={data.recommendation.coverage_eligible} />
            <FieldRow label="warnings" value={data.recommendation.warnings} />
          </section>
          <section>
            <h4>Inspection history factors (Component 4)</h4>
            {data.history_factors ? (
              <>
                <FieldRow
                  label="prior_canvass_count_code_era"
                  value={data.history_factors.prior_canvass_count_code_era}
                />
                <FieldRow
                  label="prior_canvass_priority_count"
                  value={data.history_factors.prior_canvass_priority_count}
                />
                <FieldRow
                  label="prior_canvass_priority_rate"
                  value={data.history_factors.prior_canvass_priority_rate?.toFixed(4)}
                />
                <FieldRow
                  label="prior_canvass_fail_rate"
                  value={data.history_factors.prior_canvass_fail_rate?.toFixed(4)}
                />
                <FieldRow label="fail_at_last_canvass" value={data.history_factors.fail_at_last_canvass} />
                <FieldRow
                  label="priority_at_last_canvass"
                  value={data.history_factors.priority_at_last_canvass}
                />
                <FieldRow label="days_since_last_canvass" value={data.history_factors.days_since_last_canvass} />
                <FieldRow
                  label="days_since_any_inspection"
                  value={data.history_factors.days_since_any_inspection}
                />
                <FieldRow
                  label="prior_inspection_count_any_type"
                  value={data.history_factors.prior_inspection_count_any_type}
                />
                <FieldRow
                  label="name_changed_since_last_canvass"
                  value={data.history_factors.name_changed_since_last_canvass}
                />
              </>
            ) : (
              <p className="state state-empty">
                {data.history_factors_unavailable_reason ?? 'No history factors available.'}
              </p>
            )}
          </section>
          <section>
            <h4>Model explanation (Component 11)</h4>
            {data.explanation ? (
              <>
                <FieldRow label="explanation_method" value={data.explanation.explanation_method} />
                <FieldRow label="is_exact" value={data.explanation.is_exact} />
                <FieldRow label="base_value" value={data.explanation.base_value.toFixed(4)} />
                <FieldRow label="prediction_value" value={data.explanation.prediction_value.toFixed(4)} />
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>feature</th>
                      <th>value</th>
                      <th>shap_value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.explanation.values
                      .slice()
                      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
                      .map((v) => (
                        <tr key={v.feature_name}>
                          <td>{v.feature_name}</td>
                          <td>{v.feature_value ?? '—'}</td>
                          <td>{v.shap_value.toFixed(4)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </>
            ) : (
              <p className="state state-empty">
                {data.explanation_unavailable_reason ?? 'No explanation available.'}
              </p>
            )}
          </section>
          <section>
            <h4>Schedule</h4>
            {data.schedule ? (
              <>
                <FieldRow label="planning_run_id" value={data.schedule.planning_run_id} />
                <FieldRow label="replan_index" value={data.schedule.replan_index} />
                <FieldRow label="day_index" value={data.schedule.day_index} />
                <FieldRow label="slot_index" value={data.schedule.slot_index} />
                <FieldRow label="is_scenario" value={data.schedule.is_scenario} />
              </>
            ) : (
              <p className="hint">No schedule row.</p>
            )}
          </section>
          <section>
            <h4>Human review</h4>
            {reviewData ? (
              <>
                <FieldRow label="trigger_reasons" value={reviewData.trigger_reasons} />
                <FieldRow label="review_status" value={reviewData.review_status} />
                <FieldRow label="review_id" value={reviewData.review_id} />
                <FieldRow label="resolution_action" value={reviewData.resolution_action} />
              </>
            ) : (
              <p className="hint">Not currently in the review queue.</p>
            )}
          </section>
          <section>
            <h4>Provenance</h4>
            <FieldRow label="establishment_id" value={establishmentId} />
            <FieldRow label="policy_definition_version" value={data.recommendation.policy_definition_version} />
            {data.schedule && (
              <FieldRow label="schedule_definition_version" value={data.schedule.schedule_definition_version} />
            )}
          </section>
        </TechnicalDetails>
      )}
    </PageShell>
  )
}
