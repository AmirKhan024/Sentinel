import { http, HttpResponse } from 'msw'
import {
  backlogPageFixture,
  establishmentHistoryFixture,
  operationalSelectionManifestFixture,
  planRowsPageFixture,
  planSummaryFixture,
  policyManifestFixture,
  recommendationsPageFixture,
  reviewCaseFixture,
  reviewQueuePageFixture,
  scheduleDatesFixture,
  scheduleRowsPageFixture,
  schedulingManifestFixture,
  workBlocksFixture,
} from './fixtures'

const BASE = 'http://127.0.0.1:8000'

function requireFullScope(request: Request) {
  const url = new URL(request.url)
  const required = ['policy_id', 'fold_set', 'fold_id', 'k_name']
  const missing = required.filter((f) => !url.searchParams.get(f))
  if (missing.length > 0) {
    return HttpResponse.json(
      {
        error: 'ambiguous_scope',
        detail: `Decision scope is missing required field(s): ${missing.join(', ')}.`,
        missing_scope_fields: missing,
      },
      { status: 422 },
    )
  }
  return null
}

/** A minimal in-memory mirror of `StagingService` (see `sentinel.api.services.staging_service`):
 * a POST to one of the four write endpoints appends here, and `GET /v1/staged-requests` reads it
 * back -- so a test can submit an override and then see it reflected as "pending," the same
 * relationship the real API has between those two endpoints. Reset between tests by
 * `resetStagedRequests()`, called from `src/test/setup.ts`. */
interface StagedEntry {
  request_id: string
  kind: string
  natural_id: string
  status: string
  staged_at: string
  payload: Record<string, unknown>
}

let stagedRequests: StagedEntry[] = []

export function resetStagedRequests() {
  stagedRequests = []
}

function stage(kind: string, idField: string, body: Record<string, unknown>) {
  const naturalId = String(body[idField])
  const entry: StagedEntry = {
    request_id: `req-${naturalId}`,
    kind,
    natural_id: naturalId,
    status: 'pending',
    staged_at: '2026-08-28T00:00:00Z',
    payload: body,
  }
  stagedRequests.push(entry)
  return entry
}

export const handlers = [
  http.get(`${BASE}/healthz`, () => HttpResponse.json({ status: 'ok' })),

  http.get(`${BASE}/v1/runs`, () => HttpResponse.json([])),

  http.get(`${BASE}/v1/manifests/:component`, ({ params }) => {
    if (params.component === 'policy') return HttpResponse.json(policyManifestFixture)
    if (params.component === 'scheduling') return HttpResponse.json(schedulingManifestFixture)
    if (params.component === 'operational_selection')
      return HttpResponse.json(operationalSelectionManifestFixture)
    return HttpResponse.json(
      { error: 'artifact_not_found', detail: 'No manifest sidecar for the latest explanations run.' },
      { status: 404 },
    )
  }),

  http.get(`${BASE}/v1/staged-requests`, ({ request }) => {
    const url = new URL(request.url)
    const kind = url.searchParams.get('kind')
    const rows = kind ? stagedRequests.filter((r) => r.kind === kind) : stagedRequests
    return HttpResponse.json(rows)
  }),

  http.get(`${BASE}/v1/recommendations`, ({ request }) => {
    return requireFullScope(request) ?? HttpResponse.json(recommendationsPageFixture)
  }),

  http.get(`${BASE}/v1/policy/selection-allocation`, () => HttpResponse.json([])),

  http.get(`${BASE}/v1/establishments/:id`, () => HttpResponse.json(establishmentHistoryFixture)),

  http.get(`${BASE}/v1/schedule`, ({ request }) => {
    return requireFullScope(request) ?? HttpResponse.json(scheduleRowsPageFixture)
  }),

  http.get(`${BASE}/v1/schedule/dates`, ({ request }) => {
    return requireFullScope(request) ?? HttpResponse.json(scheduleDatesFixture)
  }),

  http.get(`${BASE}/v1/schedule/backlog`, ({ request }) => {
    return requireFullScope(request) ?? HttpResponse.json(backlogPageFixture)
  }),

  http.get(`${BASE}/v1/schedule/summary`, () => HttpResponse.json([])),
  http.get(`${BASE}/v1/schedule/capacity-utilization`, () => HttpResponse.json([])),
  http.get(`${BASE}/v1/schedule/priority-preservation`, () => HttpResponse.json([])),
  http.get(`${BASE}/v1/schedule/replanning-runs`, () => HttpResponse.json([])),
  http.get(`${BASE}/v1/execution/summary`, () =>
    HttpResponse.json(
      { error: 'artifact_not_found', detail: 'No execution record for this scope.' },
      { status: 404 },
    ),
  ),

  http.get(`${BASE}/v1/review/queue`, ({ request }) => {
    const scopeError = requireFullScope(request)
    if (scopeError) return scopeError
    const url = new URL(request.url)
    const trigger = url.searchParams.get('trigger')
    if (!trigger) return HttpResponse.json(reviewQueuePageFixture)
    const data = reviewQueuePageFixture.data.filter((row) => row.trigger_reasons.includes(trigger))
    return HttpResponse.json({
      ...reviewQueuePageFixture,
      data,
      page: { ...reviewQueuePageFixture.page, total: data.length },
    })
  }),

  http.get(`${BASE}/v1/review/queue/:id`, () => HttpResponse.json(reviewCaseFixture)),

  http.get(`${BASE}/v1/review/resolutions`, () =>
    HttpResponse.json({
      data: [],
      page: { offset: 0, limit: 50, total: 0 },
      run: { path: '/data/review_resolution_log.parquet', manifest_path: null, built_at: null },
    }),
  ),

  http.get(`${BASE}/v1/policy/overrides`, () =>
    HttpResponse.json({
      data: [],
      page: { offset: 0, limit: 50, total: 0 },
      run: { path: '/data/policy_override_log.parquet', manifest_path: null, built_at: null },
    }),
  ),
  http.post(`${BASE}/v1/policy/overrides`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>
    const entry = stage('override', 'override_id', body)
    return HttpResponse.json(
      {
        request_id: entry.request_id,
        kind: entry.kind,
        natural_id: entry.natural_id,
        status: entry.status,
        staged_at: entry.staged_at,
      },
      { status: 201 },
    )
  }),

  http.get(`${BASE}/v1/schedule/adjustments`, () =>
    HttpResponse.json({
      data: [],
      page: { offset: 0, limit: 50, total: 0 },
      run: { path: '/data/schedule_adjustment_log.parquet', manifest_path: null, built_at: null },
    }),
  ),
  http.post(`${BASE}/v1/schedule/adjustments`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>
    const entry = stage('adjustment', 'adjustment_id', body)
    return HttpResponse.json(
      {
        request_id: entry.request_id,
        kind: entry.kind,
        natural_id: entry.natural_id,
        status: entry.status,
        staged_at: entry.staged_at,
      },
      { status: 201 },
    )
  }),

  http.get(`${BASE}/v1/execution/events`, () =>
    HttpResponse.json({
      data: [],
      page: { offset: 0, limit: 50, total: 0 },
      run: { path: '/data/execution_log.parquet', manifest_path: null, built_at: null },
    }),
  ),
  http.get(`${BASE}/v1/execution/contract`, () =>
    HttpResponse.json([
      {
        contract_name: 'execution_event',
        field_name: 'execution_status',
        required: true,
        dtype: 'str',
        allowed_values: 'completed|not_performed|cancelled_in_field',
        meaning: 'what the field reported',
      },
    ]),
  ),
  http.post(`${BASE}/v1/execution/events`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>
    const entry = stage('execution_event', 'execution_id', body)
    return HttpResponse.json(
      {
        request_id: entry.request_id,
        kind: entry.kind,
        natural_id: entry.natural_id,
        status: entry.status,
        staged_at: entry.staged_at,
      },
      { status: 201 },
    )
  }),

  http.get(`${BASE}/v1/plan-review/summary`, () => HttpResponse.json(planSummaryFixture)),
  http.get(`${BASE}/v1/plan-review/rows`, () => HttpResponse.json(planRowsPageFixture)),
  http.get(`${BASE}/v1/plan-review/work-blocks`, () => HttpResponse.json(workBlocksFixture)),
  http.get(`${BASE}/v1/plan-review/approval`, () =>
    HttpResponse.json(
      { error: 'artifact_not_found', detail: 'No approved_operational_plan for this planning_date.' },
      { status: 404 },
    ),
  ),
  http.post(`${BASE}/v1/plan-review/decisions`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>
    const entry = stage('plan_decision', 'decision_id', body)
    return HttpResponse.json(
      {
        request_id: entry.request_id,
        kind: entry.kind,
        natural_id: entry.natural_id,
        status: entry.status,
        staged_at: entry.staged_at,
      },
      { status: 201 },
    )
  }),
  http.post(`${BASE}/v1/plan-review/approve`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>
    const entry = stage('plan_approval', 'approval_id', body)
    return HttpResponse.json(
      {
        request_id: entry.request_id,
        kind: entry.kind,
        natural_id: entry.natural_id,
        status: entry.status,
        staged_at: entry.staged_at,
      },
      { status: 201 },
    )
  }),

  http.post(`${BASE}/v1/review/resolutions`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>
    const entry = stage('review_resolution', 'review_id', body)
    return HttpResponse.json(
      {
        request_id: entry.request_id,
        kind: entry.kind,
        natural_id: entry.natural_id,
        status: entry.status,
        staged_at: entry.staged_at,
      },
      { status: 201 },
    )
  }),
]
