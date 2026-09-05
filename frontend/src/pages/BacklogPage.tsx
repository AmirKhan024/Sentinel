import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listBacklog } from '../api/schedule'
import type { BacklogRowOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { useDecisionScope } from '../hooks/useDecisionScope'
import { useDefaultScope } from '../hooks/useDefaultScope'
import { useManifestOptions } from '../hooks/useManifestOptions'
import { backlogReasonLabel, formatDate } from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { InspectionPlanSelector } from '../components/scope/InspectionPlanSelector'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { DataTable, type Column } from '../components/common/DataTable'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { PaginationControls } from '../components/common/PaginationControls'
import { TechnicalDetails } from '../components/common/TechnicalDetails'

const REQUIRED_SCOPE = ['schedule_config_id', 'policy_id', 'fold_set', 'fold_id', 'k_name'] as const

export function BacklogPage() {
  const { scope, setScopeField, setScopeFields, missingFields } = useDecisionScope()
  const { manifests } = useManifestOptions(['policy', 'scheduling'])
  useDefaultScope(scope, setScopeFields, manifests)
  const navigate = useNavigate()

  const [offset, setOffset] = useState(0)
  const [limit] = useState(50)
  const [descending, setDescending] = useState(false)

  const missing = missingFields([...REQUIRED_SCOPE])
  const enabled = missing.length === 0

  const query = useApiQuery(
    (signal) => listBacklog(scope, { offset, limit, descending }, signal),
    [JSON.stringify(scope), offset, limit, descending],
    enabled,
  )

  const columns: Column<BacklogRowOut>[] = [
    { key: 'backlog_position', label: 'Waiting position', render: (r) => r.backlog_position },
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
    { key: 'why', label: 'Why it is waiting', render: (r) => backlogReasonLabel(r.backlog_reason) },
    {
      key: 'first_available_date',
      label: 'Next available capacity',
      render: (r) => formatDate(r.first_available_date),
    },
  ]

  function goToEstablishment(row: BacklogRowOut) {
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
      title="Waiting for Capacity"
      description="These establishments were recommended for inspection but did not fit into the currently available inspection capacity. They have not disappeared or been rejected — they are waiting for future capacity."
    >
      <InspectionPlanSelector
        scope={scope}
        setScopeField={setScopeField}
        requiredFields={[...REQUIRED_SCOPE]}
        manifests={manifests}
        showAdvanced
      />

      {!enabled && <LoadingState label="Preparing an inspection plan…" />}

      {enabled && query.status === 'loading' && <LoadingState />}
      {enabled && query.status === 'error' && <ErrorState error={query.error} />}
      {enabled && query.status === 'success' && query.data.data.length === 0 && (
        <EmptyState message="Nothing is waiting — every recommended inspection fit within the currently available capacity." />
      )}
      {enabled && query.status === 'success' && query.data.data.length > 0 && (
        <>
          <p className="hint">
            {query.data.page.total} establishment{query.data.page.total === 1 ? ' is' : 's are'}{' '}
            waiting for future inspection capacity.
          </p>
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

      <TechnicalDetails summary="Technical details">
        <p className="hint">
          This is distinct from "not currently prioritized" (see Inspection Priorities) — a
          waiting establishment was recommended; it simply ranked below what today's capacity
          could reach.
        </p>
      </TechnicalDetails>
    </PageShell>
  )
}
