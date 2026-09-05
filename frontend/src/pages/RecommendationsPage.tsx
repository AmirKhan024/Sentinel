import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSelectionAllocation, listRecommendations } from '../api/recommendations'
import type { RecommendationOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { useDecisionScope } from '../hooks/useDecisionScope'
import { useDefaultScope } from '../hooks/useDefaultScope'
import { useManifestOptions } from '../hooks/useManifestOptions'
import {
  HOW_TO_USE_PRIORITY,
  decisionReasonLabel,
  relativePriorityLabel,
  selectionStatusLabel,
  warningLabels,
} from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { InspectionPlanSelector } from '../components/scope/InspectionPlanSelector'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { DataTable, type Column } from '../components/common/DataTable'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { PaginationControls } from '../components/common/PaginationControls'
import { TechnicalDetails } from '../components/common/TechnicalDetails'

const REQUIRED_SCOPE = ['policy_id', 'fold_set', 'fold_id', 'k_name'] as const

export function RecommendationsPage() {
  const { scope, setScopeField, setScopeFields, missingFields } = useDecisionScope()
  const { manifests } = useManifestOptions(['policy', 'scheduling'])
  useDefaultScope(scope, setScopeFields, manifests)
  const navigate = useNavigate()

  const [establishmentId, setEstablishmentId] = useState('')
  const [onlyRecommended, setOnlyRecommended] = useState(true)
  const [offset, setOffset] = useState(0)
  const [limit] = useState(50)
  const [descending, setDescending] = useState(false)

  const missing = missingFields([...REQUIRED_SCOPE])
  const enabled = missing.length === 0

  const query = useApiQuery(
    (signal) =>
      listRecommendations(
        scope,
        {
          establishment_id: establishmentId || undefined,
          is_selected: onlyRecommended ? true : undefined,
        },
        { offset, limit, descending },
        signal,
      ),
    [JSON.stringify(scope), establishmentId, onlyRecommended, offset, limit, descending],
    enabled,
  )

  const allocationQuery = useApiQuery(
    (signal) => getSelectionAllocation(scope, signal),
    [JSON.stringify(scope)],
    enabled,
  )
  const allocation =
    allocationQuery.status === 'success'
      ? (allocationQuery.data.find((a) => a.model_name === scope.model_name) ?? allocationQuery.data[0])
      : undefined

  const columns: Column<RecommendationOut>[] = [
    { key: 'rank', label: 'Priority order', render: (r) => r.final_policy_rank ?? '—' },
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
      key: 'relative_priority',
      label: 'Relative priority',
      render: (r) =>
        allocation ? (
          <span title="Fixed by the risk ranking alone, before any capacity cutoff -- does not change when this plan's capacity changes.">
            {relativePriorityLabel(r.model_rank, allocation.n_universe)}
          </span>
        ) : (
          `Ranked ${r.model_rank}`
        ),
    },
    {
      key: 'is_selected',
      label: 'Status',
      render: (r) => (
        <span className={r.is_selected ? 'chip chip-positive' : 'chip chip-neutral'}>
          {selectionStatusLabel(r.is_selected)}
        </span>
      ),
    },
    {
      key: 'why',
      label: 'Why',
      render: (r) => decisionReasonLabel(r.decision_reason),
    },
    {
      key: 'warnings',
      label: 'Notes',
      render: (r) => {
        const labels = warningLabels(r.warnings)
        return labels.length > 0 ? labels.join('; ') : '—'
      },
    },
  ]

  function goToEstablishment(row: RecommendationOut) {
    const params = new URLSearchParams({
      policy_id: scope.policy_id ?? '',
      fold_set: scope.fold_set ?? '',
      fold_id: scope.fold_id ?? '',
      k_name: scope.k_name ?? '',
    })
    navigate(`/establishments/${encodeURIComponent(row.establishment_id)}?${params.toString()}`)
  }

  return (
    <PageShell
      title="Inspection Priorities"
      description="Establishments ranked by available risk signals, in priority order. Selection into this specific plan also depends on today's capacity -- see each row's status."
    >
      <InspectionPlanSelector
        scope={scope}
        setScopeField={setScopeField}
        requiredFields={[...REQUIRED_SCOPE]}
        manifests={manifests}
      />

      <TechnicalDetails summary="How Sentinel prioritizes locations">
        <p>{HOW_TO_USE_PRIORITY}</p>
      </TechnicalDetails>

      <div className="filters">
        <label>
          Search by establishment
          <input
            value={establishmentId}
            onChange={(e) => setEstablishmentId(e.target.value)}
            placeholder="Establishment ID"
          />
        </label>
        <label className="filters-checkbox">
          <input
            type="checkbox"
            checked={onlyRecommended}
            onChange={(e) => setOnlyRecommended(e.target.checked)}
          />
          Only show establishments selected for this plan
        </label>
      </div>

      {!enabled && <LoadingState label="Preparing an inspection plan…" />}

      {enabled && query.status === 'loading' && <LoadingState />}
      {enabled && query.status === 'error' && <ErrorState error={query.error} />}
      {enabled && query.status === 'success' && query.data.data.length === 0 && (
        <EmptyState message="No establishments match this view." />
      )}
      {enabled && query.status === 'success' && query.data.data.length > 0 && (
        <>
          <p className="hint">
            Click an establishment to see why it was recommended, using Sentinel's actual
            inspection history for that establishment.
          </p>
          <DataTable
            columns={columns}
            rows={query.data.data}
            rowKey={(r) => r.target_inspection_id}
            onRowClick={goToEstablishment}
          />
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

      <TechnicalDetails summary="Technical details">
        <p className="hint">
          The raw calibrated score (0-1) behind each establishment's rank is not shown on this
          list -- a bare score reads as a probability or a fixed threshold, which the policy does
          not claim. It is shown, with the model name and fold it came from, on each
          establishment's own page.
        </p>
        {enabled && allocationQuery.status === 'success' && (
          <section className="allocation-panel">
            <h3>How today's capacity was allocated</h3>
            {allocationQuery.data.map((a) => (
              <ul key={`${a.policy_id}-${a.fold_id}-${a.k_name}`}>
                <li>Establishments considered: {a.n_universe}</li>
                <li>Prioritized by risk: {a.n_risk}</li>
                <li>Prioritized to maintain coverage: {a.n_reserve}</li>
                <li>Total recommended: {a.n_selected}</li>
                <li>Coverage set-aside mechanism: {a.reserve_mechanism}</li>
              </ul>
            ))}
          </section>
        )}
      </TechnicalDetails>
    </PageShell>
  )
}
