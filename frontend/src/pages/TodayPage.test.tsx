import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { MemoryRouter, Route, Routes, useParams } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { server } from '../test/mocks/server'
import { planSummaryFixture } from '../test/mocks/fixtures'
import * as today from '../lib/today'
import { TodayPage } from './TodayPage'

function PlanEstablishmentProbe() {
  const { targetInspectionId } = useParams()
  return <p>NAVIGATED-TO:{targetInspectionId}</p>
}

const BASE = 'http://127.0.0.1:8000'

/** `planSummaryFixture.planning_date` is `'2026-08-28'`. Mocking `currentOperationalDate`
 * directly (rather than `vi.useFakeTimers`) avoids fake timers freezing testing-library's own
 * `waitFor`-based polling, which `findByText` relies on. */
function mockCurrentDate(date: string) {
  vi.spyOn(today, 'currentOperationalDate').mockReturnValue(date)
}

function renderPage() {
  return render(
    <MemoryRouter>
      <TodayPage />
    </MemoryRouter>,
  )
}

function renderPageWithRouting() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/" element={<TodayPage />} />
        <Route path="/plan/establishments/:targetInspectionId" element={<PlanEstablishmentProbe />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('TodayPage', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('shows the establishment by name in priority order', async () => {
    renderPage()
    expect(await screen.findByText('Eat A Pita')).toBeInTheDocument()
  })

  it('honestly says "Today\'s inspection plan" when the plan\'s date is the real current date', async () => {
    mockCurrentDate('2026-08-28')
    renderPage()
    expect(await screen.findByText(/Today's inspection plan/)).toBeInTheDocument()
  })

  it('honestly says a plan is not today\'s when its date differs from the real current date', async () => {
    mockCurrentDate('2026-09-04')
    renderPage()
    expect(await screen.findByText(/not today's plan yet/)).toBeInTheDocument()
  })

  it('never silently claims a stale plan is today\'s -- says so in the technical details too', async () => {
    mockCurrentDate('2026-09-04')
    renderPage()
    await screen.findByText('Eat A Pita')
    expect(screen.getByText('About this plan')).toBeInTheDocument()
    expect(screen.getByText(/No plan has been built yet for 4 Sept 2026/)).toBeInTheDocument()
  })

  it('links to the field plan and plan review pages', async () => {
    renderPage()
    await screen.findByText('Eat A Pita')
    expect(screen.getByText(/See the full field plan by work area/)).toBeInTheDocument()
    expect(screen.getByText(/Go to Plan Review/)).toBeInTheDocument()
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

  it('shows a plain-language empty state, never a raw error, when no plan exists', async () => {
    server.use(
      http.get(`${BASE}/v1/plan-review/summary`, () =>
        HttpResponse.json({ ...planSummaryFixture, selected_inspection_workload: 0 }),
      ),
      http.get(`${BASE}/v1/plan-review/rows`, () =>
        HttpResponse.json({
          data: [],
          page: { offset: 0, limit: 500, total: 0 },
          run: { path: 'x', manifest_path: null, built_at: null },
        }),
      ),
    )
    renderPage()
    expect(await screen.findByText(/No establishments in today's plan yet/)).toBeInTheDocument()
  })
})
