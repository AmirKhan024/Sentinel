import { render, screen } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { server } from '../test/mocks/server'
import { planRowFixture } from '../test/mocks/fixtures'
import { EstablishmentPlanDetailPage } from './EstablishmentPlanDetailPage'

const BASE = 'http://127.0.0.1:8000'

function renderAt(targetInspectionId: string) {
  return render(
    <MemoryRouter initialEntries={[`/plan/establishments/${encodeURIComponent(targetInspectionId)}`]}>
      <Routes>
        <Route path="/plan/establishments/:targetInspectionId" element={<EstablishmentPlanDetailPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('EstablishmentPlanDetailPage', () => {
  it('resolves a real establishment from the live plan by target_inspection_id -- the exact case that was broken', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/rows/:id`, () => HttpResponse.json(planRowFixture)),
    )
    renderAt(planRowFixture.target_inspection_id)
    expect(await screen.findByText('Eat A Pita')).toBeInTheDocument()
    expect(screen.getByText('3155 N Halsted St')).toBeInTheDocument()
  })

  it('shows Sentinel priority, why it is here, and current plan status', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/rows/:id`, () => HttpResponse.json(planRowFixture)),
    )
    renderAt(planRowFixture.target_inspection_id)
    await screen.findByText('Eat A Pita')
    expect(screen.getByText(/Priority #1/)).toBeInTheDocument()
    expect(screen.getByText(/Ranked highly enough/)).toBeInTheDocument()
    expect(screen.getByText('Selected for today\'s plan.')).toBeInTheDocument()
  })

  it('shows the machine recommendation and human decision as separate, never overwritten', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/rows/:id`, () =>
        HttpResponse.json({
          ...planRowFixture,
          supervisor_decision_action: 'move_to_later_workday',
          supervisor_decision_reason_code: 'inspector_unavailable',
          supervisor_decision_actor: 'supervisor.demo',
        }),
      ),
    )
    renderAt(planRowFixture.target_inspection_id)
    await screen.findByText('Eat A Pita')
    // The Sentinel priority section is untouched -- still #1 -- while a separate section shows
    // the supervisor's decision.
    expect(screen.getByText(/Priority #1/)).toBeInTheDocument()
    expect(screen.getByText(/Supervisor decision\./)).toBeInTheDocument()
    expect(screen.getByText(/Move to a later workday/)).toBeInTheDocument()
  })

  it('honestly states that deeper inspection history is not available, rather than hiding the section', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/rows/:id`, () => HttpResponse.json(planRowFixture)),
    )
    renderAt(planRowFixture.target_inspection_id)
    await screen.findByText('Eat A Pita')
    expect(screen.getByText('More inspection history')).toBeInTheDocument()
    expect(
      screen.getByText(/not currently available for establishments in a live operational plan/),
    ).toBeInTheDocument()
  })

  it('warns that a new decision will not silently change an already-approved plan', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/rows/:id`, () => HttpResponse.json(planRowFixture)),
      http.get(`${BASE}/v1/plan-review/summary`, () =>
        HttpResponse.json({
          planning_date: '2026-08-28',
          selected_inspection_workload: 1,
          location_available_count: 1,
          location_unavailable_count: 0,
          work_block_count: 1,
          decisions_recorded: 0,
          approval_status: 'approved',
        }),
      ),
    )
    renderAt(planRowFixture.target_inspection_id)
    await screen.findByText('Eat A Pita')
    expect(screen.getByText(/Plan status: Approved/)).toBeInTheDocument()
    expect(
      screen.getByText(/A new decision here won't change the approved plan/),
    ).toBeInTheDocument()
  })

  it('shows a plain-language error, never a raw error code, when the establishment cannot be found', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/rows/:id`, () =>
        HttpResponse.json({ error: 'row_not_found', detail: 'no such row' }, { status: 404 }),
      ),
    )
    renderAt('CANDIDATE::2026-08-28::DOES-NOT-EXIST')
    expect(await screen.findByText(/We couldn't find that record/)).toBeInTheDocument()
  })
})
