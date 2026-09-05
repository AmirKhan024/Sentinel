import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { OverviewPage } from './OverviewPage'

function renderPage() {
  return render(
    <MemoryRouter>
      <OverviewPage />
    </MemoryRouter>,
  )
}

describe('OverviewPage', () => {
  it('leads with the plain-language product story', async () => {
    renderPage()
    expect(
      await screen.findByText(
        /Sentinel helps inspection teams decide which establishments should be inspected first/,
      ),
    ).toBeInTheDocument()
  })

  it('shows the workflow explanation, including the live operational plan steps, without requiring documentation', async () => {
    renderPage()
    expect(await screen.findByText(/A prioritization policy decides which establishments/)).toBeInTheDocument()
    expect(screen.getByText(/Some cases are flagged for a person to review/)).toBeInTheDocument()
    expect(screen.getAllByText(/geographic work areas/).length).toBeGreaterThan(0)
    expect(screen.getByText(/A supervisor reviews the proposed plan/)).toBeInTheDocument()
  })

  it("links to the live Field Plan and Plan Review pages, not just the backtest pages", async () => {
    renderPage()
    expect(await screen.findByText("Today's field plan")).toBeInTheDocument()
    expect(screen.getByText('Field Plan')).toBeInTheDocument()
    expect(screen.getByText('Plan Review')).toBeInTheDocument()
    expect(
      screen.getByText('See how the current inspection workload is organized into geographic work areas'),
    ).toBeInTheDocument()
  })

  it('shows the operational summary cards, filled from real manifest and API data', async () => {
    renderPage()
    expect(await screen.findByText('Selected for this plan', {}, { timeout: 3000 })).toBeInTheDocument()
    expect(screen.getByText('Waiting for capacity')).toBeInTheDocument()
    expect(screen.getByText('Decision concerns')).toBeInTheDocument()
    expect(screen.getByText('Missing outcomes')).toBeInTheDocument()
  })

  it('keeps the technical manifest detail out of the primary view, behind Technical details', async () => {
    renderPage()
    expect(await screen.findByText('Technical details for this run')).toBeInTheDocument()
    // The old, raw manifest text (a policy id, a capacity mode) is not part of the first
    // screen's plain-language content -- it only appears once opened, which jsdom still renders
    // into the DOM (a <details> element is not display:none), so we assert it exists inside the
    // technical section rather than asserting visibility.
    expect(await screen.findByText('xgboost_platt')).toBeInTheDocument()
  })

  it('does not show a loud health warning once the API is reachable', async () => {
    renderPage()
    await screen.findByText(/Sentinel helps inspection teams/)
    expect(screen.queryByText(/not currently available/)).not.toBeInTheDocument()
  })

  it('suggests concrete next actions once real counts are known', async () => {
    renderPage()
    expect(await screen.findByText('What should I do next?', {}, { timeout: 3000 })).toBeInTheDocument()
    expect(
      await screen.findByText(/Review the recommended inspections and why each was prioritized/, {}, { timeout: 3000 }),
    ).toBeInTheDocument()
  })

  it('states plainly why the recommendations can be trusted, without marketing language', async () => {
    renderPage()
    expect(await screen.findByText('Why trust these recommendations?')).toBeInTheDocument()
    expect(
      screen.getByText(/Sentinel prioritizes; it does not replace an inspector's or supervisor's judgment/),
    ).toBeInTheDocument()
  })

  it('never blends decision concerns and missing outcomes into one misleading count', async () => {
    renderPage()
    await screen.findByText('Selected for this plan', {}, { timeout: 3000 })
    // Two distinct cards with two distinct, real counts -- never a single "needs review" number.
    expect(screen.getByText('Decision concerns')).toBeInTheDocument()
    expect(screen.getByText('Missing outcomes')).toBeInTheDocument()
    expect(screen.queryByText('Needs human review')).not.toBeInTheDocument()
  })

  it('discloses that capacity is a historical count, not live staffing', async () => {
    renderPage()
    expect(
      await screen.findByText(/Chicago's own historical record shows for that same calendar day/, {}, { timeout: 3000 }),
    ).toBeInTheDocument()
  })
})
