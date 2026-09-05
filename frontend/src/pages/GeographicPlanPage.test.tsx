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
  it('shows the establishment by name inside its work block', async () => {
    renderPage()
    expect(await screen.findByText('Eat A Pita')).toBeInTheDocument()
    expect(screen.getByText('Work Area 1')).toBeInTheDocument()
  })

  it('shows a plain-language planning-date header, not a raw scope selector', async () => {
    renderPage()
    expect(await screen.findByText(/Field plan for/)).toBeInTheDocument()
  })

  it('never claims optimized routing or driving directions', async () => {
    renderPage()
    await screen.findByText('Eat A Pita')
    expect(screen.getByText(/not driving time or traffic/)).toBeInTheDocument()
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
