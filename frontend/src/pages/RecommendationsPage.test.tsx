import { render, screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { server } from '../test/mocks/server'
import { RecommendationsPage } from './RecommendationsPage'

const BASE = 'http://127.0.0.1:8000'

const FULL_SCOPE = '?policy_id=pure_risk&fold_set=quarterly&fold_id=quarterly-2026Q1&k_name=k_1_day'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[`/recommendations${path}`]}>
      <RecommendationsPage />
    </MemoryRouter>,
  )
}

describe('RecommendationsPage', () => {
  it('shows a loading state, then automatically fills in a working inspection plan', async () => {
    renderAt('')
    expect(await screen.findByText(/Preparing an inspection plan/)).toBeInTheDocument()
    // useDefaultScope fills in the rest from the real manifests, so the page still ends up
    // showing real data rather than staying stuck on an empty form.
    expect(await screen.findByText('E-1', {}, { timeout: 3000 })).toBeInTheDocument()
  })

  it('shows a loading state, then the row once the scope is complete', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('E-1')).toBeInTheDocument()
    expect(screen.getByText('Selected for this plan')).toBeInTheDocument()
  })

  it('discloses the model’s limited accuracy once, collapsed behind Technical details, not per row or in primary text', async () => {
    renderAt(FULL_SCOPE)
    await screen.findByText('E-1')
    expect(screen.getByText('How Sentinel prioritizes locations')).toBeInTheDocument()
    // Collapsed (<details> without `open`), so still present in the DOM -- see the established
    // convention in OverviewPage.test.tsx for why this asserts presence, not visibility.
    expect(screen.getByText(/ROC-AUC roughly 0.61-0.62/)).toBeInTheDocument()
  })

  it('shows an empty state when the scoped page has no rows', async () => {
    server.use(
      http.get(`${BASE}/v1/recommendations`, () =>
        HttpResponse.json({
          data: [],
          page: { offset: 0, limit: 50, total: 0 },
          run: { path: 'x', manifest_path: null, built_at: null },
        }),
      ),
    )
    renderAt(FULL_SCOPE)
    expect(await screen.findByText(/No establishments match this view/)).toBeInTheDocument()
  })

  it('shows an error state on a server failure', async () => {
    server.use(
      http.get(`${BASE}/v1/recommendations`, () =>
        HttpResponse.json({ error: 'internal_error', detail: 'boom' }, { status: 500 }),
      ),
    )
    renderAt(FULL_SCOPE)
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
  })
})
