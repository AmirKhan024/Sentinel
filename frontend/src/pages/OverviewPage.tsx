import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getHealthz } from '../api/meta'
import { listRecommendations } from '../api/recommendations'
import { listBacklog, listSchedule } from '../api/schedule'
import { listReviewQueue } from '../api/review'
import { classifyError, type ClassifiedError } from '../api/errors'
import { useApiQuery } from '../hooks/useApiQuery'
import { useDecisionScope } from '../hooks/useDecisionScope'
import { useDefaultScope } from '../hooks/useDefaultScope'
import { useManifestOptions } from '../hooks/useManifestOptions'
import { capacityHonestyNote } from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { WorkflowDiagram } from '../components/layout/WorkflowDiagram'
import { InspectionPlanSelector } from '../components/scope/InspectionPlanSelector'
import { SummaryCard } from '../components/common/SummaryCard'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { TechnicalDetails } from '../components/common/TechnicalDetails'
import { ManifestChecksPanel } from '../components/common/ManifestChecksPanel'

const REQUIRED_SCOPE = ['policy_id', 'fold_set', 'fold_id', 'k_name'] as const

function HealthBanner() {
  const [status, setStatus] = useState<'loading' | 'ok' | 'down'>('loading')

  useEffect(() => {
    const controller = new AbortController()
    getHealthz(controller.signal)
      .then(() => setStatus('ok'))
      .catch(() => setStatus('down'))
    return () => controller.abort()
  }, [])

  if (status === 'ok') return null // Say nothing when everything is working.
  if (status === 'loading') {
    return <div className="health-indicator health-loading">Checking connection…</div>
  }
  return (
    <div className="health-indicator health-down" role="alert">
      Sentinel's inspection data is not currently available. Is the Sentinel service running?
    </div>
  )
}

export function OverviewPage() {
  const { scope, setScopeField, setScopeFields, missingFields } = useDecisionScope()
  const { manifests, loading, errors } = useManifestOptions(['policy', 'scheduling', 'explanations', 'review'])
  useDefaultScope(scope, setScopeFields, manifests)

  const missing = missingFields([...REQUIRED_SCOPE])
  const enabled = missing.length === 0
  const scheduleEnabled = enabled && Boolean(scope.schedule_config_id)

  const consideredQuery = useApiQuery(
    (signal) => listRecommendations(scope, {}, { offset: 0, limit: 1, descending: false }, signal),
    [JSON.stringify(scope)],
    enabled,
  )
  const recommendedQuery = useApiQuery(
    (signal) =>
      listRecommendations(scope, { is_selected: true }, { offset: 0, limit: 1, descending: false }, signal),
    [JSON.stringify(scope)],
    enabled,
  )
  const scheduledQuery = useApiQuery(
    (signal) => listSchedule(scope, {}, { offset: 0, limit: 1, descending: false }, signal),
    [JSON.stringify(scope)],
    scheduleEnabled,
  )
  const backlogQuery = useApiQuery(
    (signal) => listBacklog(scope, { offset: 0, limit: 1, descending: false }, signal),
    [JSON.stringify(scope)],
    scheduleEnabled,
  )
  const decisionReviewQuery = useApiQuery(
    (signal) =>
      listReviewQueue(scope, { offset: 0, limit: 1, descending: false }, signal, {
        trigger: 'policy_warning_present',
      }),
    [JSON.stringify(scope)],
    enabled,
  )
  const missingOutcomesQuery = useApiQuery(
    (signal) =>
      listReviewQueue(scope, { offset: 0, limit: 1, descending: false }, signal, {
        trigger: 'no_execution_record_on_scheduled_row',
      }),
    [JSON.stringify(scope)],
    enabled,
  )

  function count(query: { status: string; data?: { page: { total: number } } }): string | number {
    if (query.status === 'loading' || query.status === 'idle') return '…'
    if (query.status === 'error') return '—'
    return query.data?.page.total ?? '—'
  }

  return (
    <PageShell>
      <section className="hero">
        <h1>Sentinel</h1>
        <p className="hero-lede">
          Sentinel helps inspection teams decide which establishments should be inspected first
          when inspection capacity is limited.
        </p>
      </section>

      <HealthBanner />

      <section>
        <h2>Today's field plan</h2>
        <p className="hint">
          The live, date-specific inspection plan -- separate from the historical analysis below,
          which looks back at a past planning period rather than today's actual workload.
        </p>
        <div className="summary-grid">
          <SummaryCard
            label="Field Plan"
            value="→"
            hint="See how the current inspection workload is organized into geographic work areas"
            to="/geographic-plan"
          />
          <SummaryCard
            label="Plan Review"
            value="→"
            hint="Review, adjust, and approve the current operational plan"
            to="/plan-review"
          />
        </div>
      </section>

      <InspectionPlanSelector
        scope={scope}
        setScopeField={setScopeField}
        requiredFields={[...REQUIRED_SCOPE, 'schedule_config_id']}
        manifests={manifests}
      />

      {!enabled && <LoadingState label="Preparing an inspection plan…" />}

      {enabled && (
        <section>
          <h2>This plan, at a glance</h2>
          <div className="summary-grid">
            <SummaryCard
              label="Establishments considered"
              value={count(consideredQuery)}
              hint="Every establishment Sentinel had enough information to evaluate"
            />
            <SummaryCard
              label="Selected for this plan"
              value={count(recommendedQuery)}
              hint="Ranked within this plan's capacity cutoff -- not a claim these establishments are unsafe"
              to={`/recommendations?${new URLSearchParams(scope as Record<string, string>).toString()}`}
            />
            <SummaryCard
              label="Fits in available capacity"
              value={scheduleEnabled ? count(scheduledQuery) : '—'}
              hint="Selected establishments that fit into the current schedule"
              to={
                scheduleEnabled
                  ? `/schedule?${new URLSearchParams(scope as Record<string, string>).toString()}`
                  : undefined
              }
            />
            <SummaryCard
              label="Waiting for capacity"
              value={scheduleEnabled ? count(backlogQuery) : '—'}
              hint="Selected, but not yet scheduled — not rejected"
              to={
                scheduleEnabled
                  ? `/backlog?${new URLSearchParams(scope as Record<string, string>).toString()}`
                  : undefined
              }
            />
            <SummaryCard
              label="Decision concerns"
              value={count(decisionReviewQuery)}
              hint="Recommendations carrying a warning worth a second look"
              to={`/review?${new URLSearchParams(scope as Record<string, string>).toString()}`}
              variant="attention"
            />
            <SummaryCard
              label="Missing outcomes"
              value={count(missingOutcomesQuery)}
              hint="Planned inspections nobody has logged the outcome of yet -- not a sign anything went wrong"
              to={`/review?${new URLSearchParams(scope as Record<string, string>).toString()}`}
              variant="attention"
            />
          </div>
          {[consideredQuery, recommendedQuery, decisionReviewQuery, missingOutcomesQuery].some(
            (q) => q.status === 'error',
          ) && (
            <ErrorState
              error={
                ([consideredQuery, recommendedQuery, decisionReviewQuery, missingOutcomesQuery].find(
                  (q) => q.status === 'error',
                ) as {
                  error: ClassifiedError
                })?.error ?? classifyError(new Error('unknown'))
              }
            />
          )}
          {scheduleEnabled && (
            <p className="hint">
              {capacityHonestyNote(
                manifests.scheduling?.config_grid?.find(
                  (c) => c.schedule_config_id === scope.schedule_config_id,
                )?.capacity_mode,
              )}
            </p>
          )}
        </section>
      )}

      {enabled && (
        <NextActions
          scope={scope}
          recommendedQuery={recommendedQuery}
          backlogQuery={backlogQuery}
          decisionReviewQuery={decisionReviewQuery}
          missingOutcomesQuery={missingOutcomesQuery}
          scheduleEnabled={scheduleEnabled}
        />
      )}

      <section>
        <h2>How Sentinel decides</h2>
        <WorkflowDiagram />
      </section>

      <section className="explainer-box">
        <h2>Why trust these recommendations?</h2>
        <ul>
          <li>Every recommendation is based on the inspection information Sentinel actually has on file — not a guess and not outside data.</li>
          <li>Sentinel prioritizes; it does not replace an inspector's or supervisor's judgment. Every recommendation can be reviewed, and a person can still act differently.</li>
          <li>Capacity constraints are explicit: the plan shows exactly what fits today and what is waiting, never silently.</li>
          <li>Limitations are disclosed rather than hidden — including where Sentinel cannot explain a specific score, and where the underlying data is thin.</li>
          <li>The reasoning behind each recommendation is preserved and available — see "Why this recommendation?" on an establishment's page, and "Technical details" throughout.</li>
        </ul>
      </section>

      <TechnicalDetails summary="Technical details for this run">
        <ManifestTechnicalPanel title="Prioritization run" loading={loading} error={errors.policy} manifest={manifests.policy} />
        <ManifestTechnicalPanel title="Scheduling run" loading={loading} error={errors.scheduling} manifest={manifests.scheduling} />
      </TechnicalDetails>
    </PageShell>
  )
}

type CountQuery = { status: string; data?: { page: { total: number } } }

/** Turns the summary counts already on screen into concrete next actions, rather than a fixed
 * checklist -- an action only appears when the real count behind it is greater than zero, and
 * links reuse the exact scope already selected so a click lands on the same plan. */
function NextActions({
  scope,
  recommendedQuery,
  backlogQuery,
  decisionReviewQuery,
  missingOutcomesQuery,
  scheduleEnabled,
}: {
  scope: import('../api/types').DecisionScope
  recommendedQuery: CountQuery
  backlogQuery: CountQuery
  decisionReviewQuery: CountQuery
  missingOutcomesQuery: CountQuery
  scheduleEnabled: boolean
}) {
  const qs = new URLSearchParams(scope as Record<string, string>).toString()
  const actions: { key: string; to: string; label: string }[] = []

  if (recommendedQuery.status === 'success' && (recommendedQuery.data?.page.total ?? 0) > 0) {
    actions.push({
      key: 'recommendations',
      to: `/recommendations?${qs}`,
      label: 'Review the recommended inspections and why each was prioritized',
    })
  }
  if (scheduleEnabled && backlogQuery.status === 'success' && (backlogQuery.data?.page.total ?? 0) > 0) {
    actions.push({
      key: 'backlog',
      to: `/backlog?${qs}`,
      label: `Check the ${backlogQuery.data?.page.total} establishment${backlogQuery.data?.page.total === 1 ? '' : 's'} waiting for future capacity`,
    })
  }
  if (decisionReviewQuery.status === 'success' && (decisionReviewQuery.data?.page.total ?? 0) > 0) {
    actions.push({
      key: 'decision-review',
      to: `/review?${qs}`,
      label: `Investigate the ${decisionReviewQuery.data?.page.total} decision concern${decisionReviewQuery.data?.page.total === 1 ? '' : 's'}`,
    })
  }
  if (missingOutcomesQuery.status === 'success' && (missingOutcomesQuery.data?.page.total ?? 0) > 0) {
    actions.push({
      key: 'missing-outcomes',
      to: `/review?${qs}`,
      label: `Record the ${missingOutcomesQuery.data?.page.total} missing inspection outcome${missingOutcomesQuery.data?.page.total === 1 ? '' : 's'}`,
    })
  }
  if (scheduleEnabled) {
    actions.push({ key: 'schedule', to: `/schedule?${qs}`, label: 'Review the capacity plan' })
  }

  if (actions.length === 0) return null

  return (
    <section>
      <h2>What should I do next?</h2>
      <ul className="next-actions">
        {actions.map((a) => (
          <li key={a.key}>
            <Link to={a.to}>{a.label} →</Link>
          </li>
        ))}
      </ul>
    </section>
  )
}

function ManifestTechnicalPanel({
  title,
  loading,
  error,
  manifest,
}: {
  title: string
  loading: boolean
  error: ClassifiedError | undefined
  manifest: import('../api/types').ManifestJson | undefined
}) {
  return (
    <div className="manifest-panel">
      <h3>{title}</h3>
      {error && <ErrorState error={error} />}
      {!error && loading && !manifest && <LoadingState />}
      {!error && manifest && (
        <>
          <p>
            Built at: <strong>{manifest.built_at ?? '—'}</strong>
          </p>
          {manifest.row_counts && (
            <ul className="row-counts">
              {Object.entries(manifest.row_counts).map(([table, cnt]) => (
                <li key={table}>
                  {table}: {cnt}
                </li>
              ))}
            </ul>
          )}
          {manifest.selected_model && (
            <p>
              Selected model: <strong>{manifest.selected_model}</strong>
            </p>
          )}
          {manifest.default_capacity_mode && (
            <p>
              Default capacity mode: <strong>{manifest.default_capacity_mode}</strong>
            </p>
          )}
          <ManifestChecksPanel checks={manifest.checks} />
          {manifest.advisories && manifest.advisories.length > 0 && (
            <>
              <h4>Advisories</h4>
              <ul>
                {manifest.advisories.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  )
}
