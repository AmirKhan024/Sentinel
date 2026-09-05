import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { server } from '../test/mocks/server'
import { planSummaryFixture } from '../test/mocks/fixtures'
import { SupervisorPlanReviewPage } from './SupervisorPlanReviewPage'

const BASE = 'http://127.0.0.1:8000'

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/plan-review']}>
      <SupervisorPlanReviewPage />
    </MemoryRouter>,
  )
}

describe('SupervisorPlanReviewPage', () => {
  it('shows the establishment by name in its geographic work block', async () => {
    renderPage()
    expect(await screen.findByText('Eat A Pita')).toBeInTheDocument()
    expect(screen.getByText('Work Area 1')).toBeInTheDocument()
  })

  it('shows a visible count when changes are staged and waiting for an operator, so a submit never looks like it did nothing', async () => {
    server.use(
      http.get(`${BASE}/v1/staged-requests`, ({ request }) => {
        const kind = new URL(request.url).searchParams.get('kind')
        if (kind === 'plan_decision') {
          return HttpResponse.json([
            { request_id: 'r1', kind: 'plan_decision', natural_id: 'DEC-1', status: 'pending', staged_at: '2026-09-05T00:00:00Z', payload: {} },
          ])
        }
        return HttpResponse.json([])
      }),
    )
    renderPage()
    expect(
      await screen.findByText(/1 change staged and waiting for an operator's next batch run/),
    ).toBeInTheDocument()
  })

  it('links the establishment name to the live-plan detail page', async () => {
    renderPage()
    const name = await screen.findByText('Eat A Pita')
    const link = name.closest('a')
    expect(link).toHaveAttribute(
      'href',
      '/plan/establishments/CANDIDATE%3A%3A2026-08-28%3A%3AE-1',
    )
  })

  it('offers an approval action when the plan is not yet approved', async () => {
    renderPage()
    expect(await screen.findByText('Approve this plan')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Approve plan' })).toBeInTheDocument()
  })

  it('shows the approved state, with counts, once the plan has been approved', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/summary`, () =>
        HttpResponse.json({ ...planSummaryFixture, approval_status: 'approved' }),
      ),
      http.get(`${BASE}/v1/plan-review/approval`, () =>
        HttpResponse.json({
          approval_id: 'APPR-1',
          planning_date: '2026-08-28',
          approved_by: 'supervisor.demo',
          approved_at: '2026-09-04T12:00:00Z',
          note: null,
          final_selected_count: 1,
          final_active_count: 1,
          final_deferred_count: 0,
          final_not_proceeding_count: 0,
          final_undecided_count: 1,
          source_plan_review_path: 'supervisor_plan_review_2026-08-28.parquet',
          source_plan_review_sha256: 'abc123',
        }),
      ),
    )
    renderPage()
    expect(await screen.findByText('Plan approved')).toBeInTheDocument()
    expect(screen.getByText(/Approved by supervisor.demo/)).toBeInTheDocument()
    // An undecided row reads differently once the plan is approved -- it did not proceed
    // undecided by accident, it was approved with the machine recommendation standing.
    expect(
      screen.getByText('No decision was recorded before this plan was approved'),
    ).toBeInTheDocument()

    // Opening the decision form for a row must not silently let a new decision look like it
    // changes the already-approved plan. Click the reason cell, not the establishment link
    // (which navigates instead of expanding the row).
    const user = userEvent.setup()
    await user.click(screen.getByText('No decision was recorded before this plan was approved'))
    expect(
      await screen.findByText(/This plan has already been approved/),
    ).toBeInTheDocument()
  })

  it('never removes an establishment from view -- both machine rank and decision are shown', async () => {
    renderPage()
    await screen.findByText('Eat A Pita')
    expect(screen.getByText('No decision recorded yet')).toBeInTheDocument()
  })

  it('shows a plain-language planning-date header', async () => {
    renderPage()
    expect(await screen.findByText(/Plan for/)).toBeInTheDocument()
  })

  it('shows how many establishments were eligible vs. selected, from real manifest counts', async () => {
    renderPage()
    expect(
      await screen.findByText(/Sentinel found 35,859 establishments eligible/),
    ).toBeInTheDocument()
    expect(screen.getByText(/1 fit within today's capacity/)).toBeInTheDocument()
  })

  it('falls back to a plain work-block label, never a raw id, if no label is present', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/rows`, () =>
        HttpResponse.json({
          data: [
            {
              planning_date: '2026-08-28',
              establishment_id: 'E-1',
              target_inspection_id: 'CANDIDATE::2026-08-28::E-1',
              canonical_name: 'Eat A Pita',
              canonical_address: '3155 N Halsted St',
              establishment_name: 'Eat A Pita',
              establishment_address: '3155 N Halsted St',
              calibrated_score: 0.55,
              base_score: 0.41,
              rank: 1,
              policy_rank: 1,
              selection_reason: 'selected_by_risk_rank',
              selection_mechanism: 'risk_priority',
              operational_priority: 1,
              location_status: 'location_available',
              work_block_id: 'AREA-1',
              work_block_label: '',
              suggested_order_in_block: 1,
              organization_mode: 'risk_first',
              highest_sentinel_rank_in_block: 1,
              supervisor_decision_id: null,
              supervisor_decision_action: null,
              supervisor_decision_reason_code: null,
              supervisor_decision_actor: null,
              supervisor_decision_decided_at: null,
              supervisor_revised_planned_date: null,
              supervisor_revised_work_block_id: null,
              supervisor_revised_operational_priority: null,
            },
          ],
          page: { offset: 0, limit: 500, total: 1 },
          run: { path: 'x', manifest_path: null, built_at: null },
        }),
      ),
    )
    renderPage()
    expect(await screen.findByText('Work block 1')).toBeInTheDocument()
    expect(screen.queryByText('AREA-1')).not.toBeInTheDocument()
  })
})
