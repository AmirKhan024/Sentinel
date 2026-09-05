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
  let mostBlocksAreSingletons = false
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
    const mapped = blocks.filter((b) => !b.is_unmapped)
    const singletons = mapped.filter((b) => b.size === 1)
    mostBlocksAreSingletons = mapped.length > 0 && singletons.length >= Math.round(0.7 * mapped.length)
  }

  const organizationMode = rowsQuery.status === 'success' ? rowsQuery.data.data[0]?.organization_mode : undefined
  const planningDate = rowsQuery.status === 'success' ? rowsQuery.data.data[0]?.planning_date : undefined

  return (
    <PageShell
      title="Proposed Field-Work Organization"
      description="How the selected inspection workload can be organized geographically -- based on straight-line distance between establishments, not driving time or traffic."
    >
      {loading && <LoadingState />}
      {errored && <ErrorState error={errored.error} />}
      {!loading && !errored && blocks.length === 0 && (
        <EmptyState message="No field plan is available yet for this planning date. Check back once today's plan has been built." />
      )}

      {!loading && !errored && blocks.length > 0 && (
        <>
          {planningDate && <p className="hint">Field plan for {formatDate(planningDate)}.</p>}
          <p className="page-description">
            {organizationMode && <>Suggested order: {organizationModeLabel(organizationMode)}. </>}
            Proximity is based on straight-line geographic distance; Sentinel does not currently
            estimate driving time or traffic.
          </p>

          {mostBlocksAreSingletons && (
            <div className="state state-ambiguous" role="note">
              <p>
                Most work blocks below contain a single establishment: the selected establishments
                are spatially dispersed at the current geographic proximity threshold. A broader
                threshold would merge more of them into shared blocks, at the cost of grouping
                establishments farther apart.
              </p>
            </div>
          )}

          <div className="work-block-list">
            {blocks.map((block) => {
              const rows = rowsByBlock.get(block.work_block_id) ?? []
              return (
                <section key={block.work_block_id} className="work-block-card">
                  <header>
                    <h2>{workAreaLabel(block.work_block_label)}</h2>
                    <span className="hint">
                      {block.size} establishment{block.size === 1 ? '' : 's'}
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
