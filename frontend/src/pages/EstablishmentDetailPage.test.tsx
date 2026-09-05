import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { server } from '../test/mocks/server'
import { establishmentHistoryFixture } from '../test/mocks/fixtures'
import { EstablishmentDetailPage } from './EstablishmentDetailPage'

const BASE = 'http://127.0.0.1:8000'
const FULL_SCOPE = 'policy_id=pure_risk&fold_set=quarterly&fold_id=quarterly-2026Q1&k_name=k_1_day'

function renderAt(query: string) {
  return render(
    <MemoryRouter initialEntries={[`/establishments/E-1?${query}`]}>
      <Routes>
        <Route path="/establishments/:establishmentId" element={<EstablishmentDetailPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('EstablishmentDetailPage', () => {
  it('shows the establishment journey as six separate, never-merged steps', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('1. Available information')).toBeInTheDocument()
    expect(screen.getByText('2. Priority position and evidence')).toBeInTheDocument()
    expect(screen.getByText('3. Selected for this plan?')).toBeInTheDocument()
    expect(screen.getByText('4. Schedule')).toBeInTheDocument()
    expect(screen.getByText('5. Human review')).toBeInTheDocument()
    expect(screen.getByText('6. Decision history')).toBeInTheDocument()
  })

  it('keeps decision_reason and schedule_reason separate, never concatenated', async () => {
    renderAt(FULL_SCOPE)
    // Appears twice by design: once in the plain-language journey, once verbatim in Technical
    // details -- the two views are never merged into one string.
    expect((await screen.findAllByText('top of risk-ranked queue', {}, { timeout: 3000 })).length).toBeGreaterThan(0)
    expect(screen.getAllByText('fit within horizon at rank 1').length).toBeGreaterThan(0)
    expect(screen.queryByText(/top of risk-ranked queue.*fit within horizon/)).not.toBeInTheDocument()
  })

  it('renders "not scheduled" in plain language rather than an error when schedule is null', async () => {
    server.use(
      http.get(`${BASE}/v1/establishments/:id`, () =>
        HttpResponse.json({ ...establishmentHistoryFixture, schedule: null }),
      ),
    )
    renderAt(FULL_SCOPE)
    expect(await screen.findByText(/Not scheduled/)).toBeInTheDocument()
  })

  it('renders explanation_unavailable_reason verbatim inside Technical details', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('model has no attribution support for this fold')).toBeInTheDocument()
  })

  it('shows the human review status when this establishment is flagged', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Needs review')).toBeInTheDocument()
    expect(
      screen.getByText('This recommendation includes a warning that should be reviewed by a person'),
    ).toBeInTheDocument()
  })

  it('shows "does not currently need human review" when the establishment is not flagged', async () => {
    server.use(http.get(`${BASE}/v1/review/queue/:id`, () => HttpResponse.json({ error: 'row_not_found', detail: 'no case' }, { status: 404 })))
    renderAt(FULL_SCOPE)
    expect(await screen.findByText(/does not currently need human review/)).toBeInTheDocument()
  })

  it('shows a preparing state, then real data, when scope is filled in automatically', async () => {
    renderAt('')
    expect(await screen.findByText(/Preparing an inspection plan/)).toBeInTheDocument()
    expect(await screen.findByText('1. Available information', {}, { timeout: 3000 })).toBeInTheDocument()
  })

  it('explains why this establishment was recommended using its actual inspection history', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText("What Sentinel's history for this establishment shows")).toBeInTheDocument()
    expect(
      screen.getByText('6 of its last 7 canvass inspections found a Priority or Priority Foundation violation.'),
    ).toBeInTheDocument()
    expect(screen.getByText('It has been 345 days since its last canvass inspection.')).toBeInTheDocument()
  })

  it('reports why history factors are unavailable rather than showing nothing', async () => {
    server.use(
      http.get(`${BASE}/v1/establishments/:id`, () =>
        HttpResponse.json({
          ...establishmentHistoryFixture,
          history_factors: null,
          history_factors_unavailable_reason: 'No feature table has been built yet.',
        }),
      ),
    )
    renderAt(FULL_SCOPE)
    // Appears twice by design: once in the plain-language journey step, once in Technical
    // details -- same pattern as explanation_unavailable_reason above.
    expect((await screen.findAllByText('No feature table has been built yet.')).length).toBeGreaterThan(0)
  })

  it('never calls the selection a risk verdict, and discloses the model’s limited accuracy', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Selected for this plan')).toBeInTheDocument()
    expect(
      screen.getByText(/Ranked within this plan's capacity cutoff -- not a claim that this establishment is unsafe/),
    ).toBeInTheDocument()
    expect(screen.getByText('How Sentinel prioritizes locations')).toBeInTheDocument()
    // Collapsed (<details> without `open`), so still present in the DOM.
    expect(screen.getByText(/ROC-AUC roughly 0.61-0.62/)).toBeInTheDocument()
  })

  it('lets a reviewer submit a human override, and shows it as staged, not applied', async () => {
    const user = userEvent.setup()
    renderAt(FULL_SCOPE)
    await screen.findByText('Selected for this plan')

    await user.click(screen.getByRole('button', { name: 'Change the priority decision' }))
    await user.type(await screen.findByLabelText(/Why are you changing Sentinel's decision/), 'local complaint')
    await user.type(screen.getByLabelText(/Your name or id/), 'jsmith')
    await user.click(screen.getByRole('button', { name: 'Submit this decision' }))

    expect(await screen.findByText('Sentinel recorded your decision.')).toBeInTheDocument()
    expect(screen.getAllByText(/staged/i).length).toBeGreaterThan(0)
  })

  it('lets a reviewer record an inspection outcome using the real execution contract', async () => {
    const user = userEvent.setup()
    renderAt(FULL_SCOPE)
    await user.click(await screen.findByRole('button', { name: 'Record inspection outcome' }))

    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'Completed' })).toBeInTheDocument()
    })
    await user.type(screen.getByLabelText(/Notes/), 'field report')
    await user.type(screen.getByLabelText(/Your name or id/), 'inspector1')
    await user.click(screen.getByRole('button', { name: 'Submit this outcome' }))

    expect(await screen.findByText('Sentinel recorded your decision.')).toBeInTheDocument()
    expect(
      screen.getByText(/does not remove the missing-outcome flag from Human Review immediately/),
    ).toBeInTheDocument()
  })

  it('lets a reviewer resolve a flagged case using the existing resolution vocabulary', async () => {
    const user = userEvent.setup()
    renderAt(FULL_SCOPE)
    await user.click(await screen.findByRole('button', { name: 'Resolve this case' }))
    await user.type(screen.getByLabelText(/^Reason/), 'reviewed, matches local context')
    await user.type(screen.getByLabelText(/Your name or id/), 'jsmith')
    await user.click(screen.getByRole('button', { name: 'Submit this resolution' }))

    expect(await screen.findByText('Sentinel recorded your decision.')).toBeInTheDocument()
  })

  it('shows a decision-history entry immediately after staging an override', async () => {
    const user = userEvent.setup()
    renderAt(FULL_SCOPE)
    await screen.findByText('No overrides, adjustments, outcomes or resolutions recorded for this establishment yet.')

    await user.click(screen.getByRole('button', { name: 'Change the priority decision' }))
    await user.type(screen.getByLabelText(/Why are you changing Sentinel's decision/), 'local complaint')
    await user.type(screen.getByLabelText(/Your name or id/), 'jsmith')
    await user.click(screen.getByRole('button', { name: 'Submit this decision' }))

    expect(await screen.findByText('Staged, not yet applied')).toBeInTheDocument()
  })
})
