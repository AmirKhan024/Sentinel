import { useNavigate } from 'react-router-dom'
import { listPlanRows, listWorkBlocks } from '../api/planReview'
import type { PlanRowOut, WorkBlockOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { formatDate, organizationModeLabel, workAreaLabel, workBlockRationale } from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { TechnicalDetails } from '../components/common/TechnicalDetails'

const FETCH_CAP = 500

/**
 * Component 20's proposed field-work organization. Read-only: the geographic grouping and
 * suggested order are Sentinel's proposal, not a confirmed schedule -- adjustments belong on
 * the Plan Review page (Component 21), which is where a supervisor's decisions live.
 */
export function GeographicPlanPage() {
  const navigate = useNavigate()

  const blocksQuery = useApiQuery((signal) => listWorkBlocks(undefined, signal), [], true)
  const rowsQuery = useApiQuery(
    (signal) => listPlanRows(undefined, { offset: 0, limit: FETCH_CAP, descending: false }, signal),
    [],
    true,
  )

  const loading = blocksQuery.status === 'loading' || rowsQuery.status === 'loading'
  const errored = blocksQuery.status === 'error' ? blocksQuery : rowsQuery.status === 'error' ? rowsQuery : null

  function goToEstablishment(row: PlanRowOut) {
    navigate(`/plan/establishments/${encodeURIComponent(row.target_inspection_id)}`)
  }

  let blocks: WorkBlockOut[] = []
  const rowsByBlock = new Map<string, PlanRowOut[]>()
  if (blocksQuery.status === 'success' && rowsQuery.status === 'success') {
    blocks = blocksQuery.data
    for (const row of rowsQuery.data.data) {
      const list = rowsByBlock.get(row.work_block_id) ?? []
      list.push(row)
      rowsByBlock.set(row.work_block_id, list)
    }
    for (const list of rowsByBlock.values()) {
      list.sort((a, b) => (a.suggested_order_in_block ?? 0) - (b.suggested_order_in_block ?? 0))
    }
  }
  // Split by what's actually true of each block, rather than rendering 20+ near-identical
  // one-establishment card sections: a genuinely grouped area is the primary content; a
  // standalone establishment (real, correctly computed -- just not near another selected
  // establishment today) belongs in one compact list, not its own repeated card.
  const groupedBlocks = blocks.filter((b) => !b.is_unmapped && b.size > 1)
  const singletonBlocks = blocks.filter((b) => !b.is_unmapped && b.size === 1)
  const unmappedBlocks = blocks.filter((b) => b.is_unmapped)

  const organizationMode = rowsQuery.status === 'success' ? rowsQuery.data.data[0]?.organization_mode : undefined
  const planningDate = rowsQuery.status === 'success' ? rowsQuery.data.data[0]?.planning_date : undefined

  return (
    <PageShell
      title="Proposed Field-Work Organization"
      description="How today's selected inspection workload is grouped by geographic proximity."
    >
      {loading && <LoadingState />}
      {errored && <ErrorState error={errored.error} />}
      {!loading && !errored && blocks.length === 0 && (
        <EmptyState message="No field plan is available yet for this planning date. Check back once today's plan has been built." />
      )}

      {!loading && !errored && blocks.length > 0 && (
        <>
          {planningDate && <p className="hint">Field plan for {formatDate(planningDate)}.</p>}
          {organizationMode && (
            <p className="hint">Suggested order: {organizationModeLabel(organizationMode)}.</p>
          )}

          {groupedBlocks.length > 0 && (
            <div className="work-block-list">
              {groupedBlocks.map((block) => {
                const rows = rowsByBlock.get(block.work_block_id) ?? []
                return (
                  <section key={block.work_block_id} className="work-block-card">
                    <header>
                      <h2>{workAreaLabel(block.work_block_label)}</h2>
                      <span className="hint">
                        {block.size} establishments
                        {block.highest_sentinel_rank != null && <> · highest priority #{block.highest_sentinel_rank}</>}
                        {block.rank_range && block.rank_range[0] !== block.rank_range[1] && (
                          <> · ranks #{block.rank_range[0]}-#{block.rank_range[1]}</>
                        )}
                      </span>
                    </header>
                    <p className="hint">{workBlockRationale(block)}</p>
                    {rows.length > 0 && (
                      <ol className="work-block-order">
                        {rows.map((row) => (
                          <li key={row.target_inspection_id} onClick={() => goToEstablishment(row)}>
                            {row.policy_rank != null && <span className="hint">Rank #{row.policy_rank} — </span>}
                            <EstablishmentIdentity
                              name={row.establishment_name}
                              address={row.establishment_address}
                              establishmentId={row.establishment_id}
                            />
                          </li>
                        ))}
                      </ol>
                    )}
                  </section>
                )
              })}
            </div>
          )}

          {singletonBlocks.length > 0 && (
            <section className="work-block-card">
              <header>
                <h2>Not near another selected location today ({singletonBlocks.length})</h2>
              </header>
              <p className="hint">
                Each of these establishments is farther than the configured proximity threshold
                from every other selected establishment today -- real, correctly computed, just
                not part of a shared work area.
              </p>
              <ol className="work-block-order">
                {singletonBlocks.map((block) => {
                  const row = (rowsByBlock.get(block.work_block_id) ?? [])[0]
                  if (!row) return null
                  return (
                    <li key={block.work_block_id} onClick={() => goToEstablishment(row)}>
                      {row.policy_rank != null && <span className="hint">Rank #{row.policy_rank} — </span>}
                      <EstablishmentIdentity
                        name={row.establishment_name}
                        address={row.establishment_address}
                        establishmentId={row.establishment_id}
                      />
                    </li>
                  )
                })}
              </ol>
            </section>
          )}

          {unmappedBlocks.length > 0 && (
            <section className="work-block-card">
              <header>
                <h2>Location unavailable ({unmappedBlocks[0].size})</h2>
              </header>
              <p className="hint">
                Sentinel has no usable coordinates for these establishments -- shown here rather
                than fabricated a location.
              </p>
              <ol className="work-block-order">
                {(rowsByBlock.get(unmappedBlocks[0].work_block_id) ?? []).map((row) => (
                  <li key={row.target_inspection_id} onClick={() => goToEstablishment(row)}>
                    {row.policy_rank != null && <span className="hint">Rank #{row.policy_rank} — </span>}
                    <EstablishmentIdentity
                      name={row.establishment_name}
                      address={row.establishment_address}
                      establishmentId={row.establishment_id}
                    />
                  </li>
                ))}
              </ol>
            </section>
          )}

          <TechnicalDetails summary="What Sentinel does not know about travel">
            <p className="hint">
              Every distance here is straight-line (Haversine) between establishment coordinates.
              Sentinel has no road-network, traffic, or travel-time data source, no inspector
              start locations or working hours, and no confirmed staffing count. A geographic work
              block is not a workday -- whether these establishments fit into one inspector's day
              is a decision for a supervisor, not a claim Sentinel makes.
            </p>
          </TechnicalDetails>
        </>
      )}
    </PageShell>
  )
}
