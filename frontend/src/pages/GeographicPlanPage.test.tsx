import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { MemoryRouter, Route, Routes, useParams } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { server } from '../test/mocks/server'
import { GeographicPlanPage } from './GeographicPlanPage'

const BASE = 'http://127.0.0.1:8000'

function PlanEstablishmentProbe() {
  const { targetInspectionId } = useParams()
  return <p>NAVIGATED-TO:{targetInspectionId}</p>
}

function renderPage() {
  return render(
    <MemoryRouter>
      <GeographicPlanPage />
    </MemoryRouter>,
  )
}

function renderPageWithRouting() {
  return render(
    <MemoryRouter initialEntries={['/geographic-plan']}>
      <Routes>
        <Route path="/geographic-plan" element={<GeographicPlanPage />} />
        <Route path="/plan/establishments/:targetInspectionId" element={<PlanEstablishmentProbe />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('GeographicPlanPage', () => {
  it('shows a real standalone establishment in the compact singleton list, not its own repeated card', async () => {
    renderPage()
    expect(await screen.findByText('Eat A Pita')).toBeInTheDocument()
    expect(
      screen.getByText(/Not near another selected location today/),
    ).toBeInTheDocument()
    expect(screen.queryByText('Work Area 1')).not.toBeInTheDocument()
  })

  it('shows a genuinely grouped work area as its own card, with every establishment in it', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/work-blocks`, () =>
        HttpResponse.json([
          {
            work_block_id: 'AREA-1',
            work_block_label: 'Area 1',
            size: 2,
            highest_sentinel_rank: 1,
            rank_range: [1, 2],
            is_unmapped: false,
            decisions_recorded: 0,
          },
        ]),
      ),
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
              work_block_label: 'Area 1',
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
            {
              planning_date: '2026-08-28',
              establishment_id: 'E-2',
              target_inspection_id: 'CANDIDATE::2026-08-28::E-2',
              canonical_name: 'Second Spot',
              canonical_address: '10 W Lake St',
              establishment_name: 'Second Spot',
              establishment_address: '10 W Lake St',
              calibrated_score: 0.5,
              base_score: 0.4,
              rank: 2,
              policy_rank: 2,
              selection_reason: 'selected_by_risk_rank',
              selection_mechanism: 'risk_priority',
              operational_priority: 2,
              location_status: 'location_available',
              work_block_id: 'AREA-1',
              work_block_label: 'Area 1',
              suggested_order_in_block: 2,
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
          page: { offset: 0, limit: 500, total: 2 },
          run: { path: 'x', manifest_path: null, built_at: null },
        }),
      ),
    )
    renderPage()
    expect(await screen.findByText('Work Area 1')).toBeInTheDocument()
    expect(screen.getByText('Eat A Pita')).toBeInTheDocument()
    expect(screen.getByText('Second Spot')).toBeInTheDocument()
    expect(screen.queryByText(/Not near another selected location today/)).not.toBeInTheDocument()
  })

  it('shows a plain-language planning-date header, not a raw scope selector', async () => {
    renderPage()
    expect(await screen.findByText(/Field plan for/)).toBeInTheDocument()
  })

  it('never claims optimized routing or driving directions, and states the driving-time caveat once, not always-visible', async () => {
    renderPage()
    await screen.findByText('Eat A Pita')
    expect(screen.getByText(/grouped by geographic proximity/)).toBeInTheDocument()
    expect(screen.queryByText(/optimized route/)).not.toBeInTheDocument()
    // Collapsed (<details> without `open`), so still present in the DOM -- see the established
    // convention elsewhere in this test suite for why this asserts presence, not visibility.
    expect(screen.getByText(/no road-network, traffic, or travel-time data source/)).toBeInTheDocument()
  })

  it('clicking an establishment navigates to the live-plan detail page, not the broken historical-scope route', async () => {
    const user = userEvent.setup()
    renderPageWithRouting()
    const row = await screen.findByText('Eat A Pita')
    await user.click(row)
    expect(await screen.findByText(/NAVIGATED-TO:/)).toHaveTextContent(
      'NAVIGATED-TO:CANDIDATE::2026-08-28::E-1',
    )
  })

  it('shows a plain-language empty state with no raw CLI command', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/work-blocks`, () => HttpResponse.json([])),
      http.get(`${BASE}/v1/plan-review/rows`, () =>
        HttpResponse.json({
          data: [],
          page: { offset: 0, limit: 500, total: 0 },
          run: { path: 'x', manifest_path: null, built_at: null },
        }),
      ),
    )
    renderPage()
    expect(await screen.findByText(/No field plan is available yet/)).toBeInTheDocument()
    expect(screen.queryByText(/sentinel organize-geography/)).not.toBeInTheDocument()
  })

  it('shows an error state in plain language, not a raw error code', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/work-blocks`, () =>
        HttpResponse.json(
          { error: 'artifact_not_found', detail: 'No geographic plan found.' },
          { status: 404 },
        ),
      ),
    )
    renderPage()
    expect(
      await screen.findByText(/We couldn't find that information yet/),
    ).toBeInTheDocument()
  })
})
