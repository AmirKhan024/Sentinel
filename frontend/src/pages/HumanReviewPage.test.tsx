import { render, screen } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { server } from '../test/mocks/server'
import { HumanReviewPage } from './HumanReviewPage'

const BASE = 'http://127.0.0.1:8000'
const FULL_SCOPE = '?policy_id=pure_risk&fold_set=quarterly&fold_id=quarterly-2026Q1&k_name=k_1_day'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[`/review${path}`]}>
      <HumanReviewPage />
    </MemoryRouter>,
  )
}

describe('HumanReviewPage', () => {
  it('labels this page as a historical simulation, and its title no longer collides with the live "Needs attention" concept', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Backtest: Decision Review')).toBeInTheDocument()
    expect(screen.getByText(/Historical simulation/)).toBeInTheDocument()
  })

  it('shows the establishment by name, not just its raw id', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Eat A Pita')).toBeInTheDocument()
  })

  it('states in one line why the case is here, in plain language', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Flagged for a policy warning')).toBeInTheDocument()
  })

  it('never implies a flagged case is necessarily wrong', async () => {
    renderAt(FULL_SCOPE)
    expect(
      await screen.findByText(/Cases worth a human look before you treat the recommendation/),
    ).toBeInTheDocument()
  })

  it('keeps the raw trigger code available, in Technical details', async () => {
    renderAt(FULL_SCOPE)
    await screen.findByText('Eat A Pita')
    expect(screen.getByText('policy_warning_present')).toBeInTheDocument()
  })

  it('keeps decision review and missing outcomes visually separate, never merged into one list', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText(/Decision review \(1\)/)).toBeInTheDocument()
    expect(screen.getByText(/Missing outcomes \(0\)/)).toBeInTheDocument()
    expect(screen.getByText('No missing outcomes are currently recorded.')).toBeInTheDocument()
    // The fixture case is a decision concern only -- it must not appear to also be a missing
    // outcome, and a missing outcome must never read as evidence Sentinel decided wrong.
    expect(
      screen.getByText(/record-keeping gap, not evidence anything went wrong/),
    ).toBeInTheDocument()
  })

  it('shows an empty state when nothing needs attention', async () => {
    server.use(
      http.get(`${BASE}/v1/review/queue`, () =>
        HttpResponse.json({
          data: [],
          page: { offset: 0, limit: 50, total: 0 },
          run: { path: 'x', manifest_path: null, built_at: null },
        }),
      ),
    )
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Nothing needs attention right now.')).toBeInTheDocument()
  })
})
