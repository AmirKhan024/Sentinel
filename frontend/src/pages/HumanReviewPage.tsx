import { useNavigate } from 'react-router-dom'
import { listReviewQueue } from '../api/review'
import type { ReviewCaseOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { useDecisionScope } from '../hooks/useDecisionScope'
import { useDefaultScope } from '../hooks/useDefaultScope'
import { useManifestOptions } from '../hooks/useManifestOptions'
import { reviewStatusLabel } from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { InspectionPlanSelector } from '../components/scope/InspectionPlanSelector'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { TechnicalDetails } from '../components/common/TechnicalDetails'

const REQUIRED_SCOPE = ['policy_id', 'fold_set', 'fold_id', 'k_name'] as const

const FETCH_CAP = 500

interface AttentionItem {
  case: ReviewCaseOut
  reasons: ('warning' | 'missing_outcome')[]
}

export function HumanReviewPage() {
  const { scope, setScopeField, setScopeFields, missingFields } = useDecisionScope()
  const { manifests } = useManifestOptions(['policy', 'scheduling'])
  useDefaultScope(scope, setScopeFields, manifests)
  const navigate = useNavigate()

  const missing = missingFields([...REQUIRED_SCOPE])
  const enabled = missing.length === 0

  const warningQuery = useApiQuery(
    (signal) =>
      listReviewQueue(scope, { offset: 0, limit: FETCH_CAP, descending: false }, signal, {
        trigger: 'policy_warning_present',
      }),
    [JSON.stringify(scope)],
    enabled,
  )
  const missingOutcomeQuery = useApiQuery(
    (signal) =>
      listReviewQueue(scope, { offset: 0, limit: FETCH_CAP, descending: false }, signal, {
        trigger: 'no_execution_record_on_scheduled_row',
      }),
    [JSON.stringify(scope)],
    enabled,
  )

  function goToEstablishment(row: ReviewCaseOut) {
    const params = new URLSearchParams({
      policy_id: scope.policy_id ?? '',
      fold_set: scope.fold_set ?? '',
      fold_id: scope.fold_id ?? '',
      k_name: scope.k_name ?? '',
    })
    navigate(`/establishments/${encodeURIComponent(row.establishment_id)}?${params.toString()}`)
  }

  const loading = warningQuery.status === 'loading' || missingOutcomeQuery.status === 'loading'
  const errored = warningQuery.status === 'error' ? warningQuery : missingOutcomeQuery.status === 'error' ? missingOutcomeQuery : null

  let items: AttentionItem[] = []
  let truncated = false
  if (warningQuery.status === 'success' && missingOutcomeQuery.status === 'success') {
    const byId = new Map<string, AttentionItem>()
    for (const c of warningQuery.data.data) {
      byId.set(c.target_inspection_id, { case: c, reasons: ['warning'] })
    }
    for (const c of missingOutcomeQuery.data.data) {
      const existing = byId.get(c.target_inspection_id)
      if (existing) existing.reasons.push('missing_outcome')
      else byId.set(c.target_inspection_id, { case: c, reasons: ['missing_outcome'] })
    }
    items = [...byId.values()].filter((item) => item.case.review_status !== 'resolved')
    truncated =
      warningQuery.data.page.total > FETCH_CAP || missingOutcomeQuery.data.page.total > FETCH_CAP
  }

  return (
    <PageShell
      title="Needs Attention"
      description="Cases worth a human look before you treat the recommendation or schedule as final."
    >
      <InspectionPlanSelector
        scope={scope}
        setScopeField={setScopeField}
        requiredFields={[...REQUIRED_SCOPE]}
        manifests={manifests}
      />

      {!enabled && <LoadingState label="Preparing an inspection plan…" />}
      {enabled && loading && <LoadingState />}
      {enabled && errored && <ErrorState error={errored.error} />}
      {enabled && !loading && !errored && items.length === 0 && (
        <EmptyState message="Nothing needs attention right now." />
      )}

      {enabled && !loading && !errored && items.length > 0 && (
        <>
          {(() => {
            const decisionConcerns = items.filter((item) => item.reasons.includes('warning'))
            const missingOutcomes = items.filter((item) => item.reasons.includes('missing_outcome'))
            return (
              <>
                <section>
                  <h2>Decision review ({decisionConcerns.length})</h2>
                  <p className="hint">
                    Recommendations carrying a warning worth a second look before you treat them
                    as final. This is not a claim that Sentinel decided wrong.
                  </p>
                  {decisionConcerns.length === 0 ? (
                    <EmptyState message="No decisions currently need a second look." />
                  ) : (
                    <ol className="attention-list">
                      {decisionConcerns.map((item) => (
                        <li
                          key={item.case.target_inspection_id}
                          className="attention-row"
                          onClick={() => goToEstablishment(item.case)}
                        >
                          <EstablishmentIdentity
                            name={item.case.establishment_name}
                            address={item.case.establishment_address}
                            establishmentId={item.case.establishment_id}
                          />
                          <span className="attention-row-reason">Flagged for a policy warning</span>
                          <span className="attention-row-action">Review →</span>
                        </li>
                      ))}
                    </ol>
                  )}
                </section>

                <section>
                  <h2>Missing outcomes ({missingOutcomes.length})</h2>
                  <p className="hint">
                    Planned inspections nobody has logged the outcome of yet. This is a
                    record-keeping gap, not evidence anything went wrong with the recommendation
                    or the schedule.
                  </p>
                  {missingOutcomes.length === 0 ? (
                    <EmptyState message="No missing outcomes are currently recorded." />
                  ) : (
                    <ol className="attention-list">
                      {missingOutcomes.map((item) => (
                        <li
                          key={item.case.target_inspection_id}
                          className="attention-row"
                          onClick={() => goToEstablishment(item.case)}
                        >
                          <EstablishmentIdentity
                            name={item.case.establishment_name}
                            address={item.case.establishment_address}
                            establishmentId={item.case.establishment_id}
                          />
                          <span className="attention-row-reason">No outcome logged yet</span>
                          <span className="attention-row-action">Log outcome →</span>
                        </li>
                      ))}
                    </ol>
                  )}
                </section>
              </>
            )
          })()}
        </>
      )}
      {truncated && (
        <p className="hint">
          Showing the first {FETCH_CAP} cases. Narrow the plan (e.g. a smaller capacity level) to
          see the rest.
        </p>
      )}

      <TechnicalDetails summary="Why a case lands here">
        <p className="hint">
          A case appears for one of two independent, threshold-free reasons: a selected
          recommendation carries a policy warning (<code>policy_warning_present</code>), or a
          scheduled inspection has no matching execution record yet (
          <code>no_execution_record_on_scheduled_row</code>). Neither means Sentinel's
          recommendation was wrong, and a case can carry both reasons at once. Status shown as
          "{reviewStatusLabel('flagged')}" until a resolution is recorded on the establishment's
          own page.
        </p>
      </TechnicalDetails>
    </PageShell>
  )
}
