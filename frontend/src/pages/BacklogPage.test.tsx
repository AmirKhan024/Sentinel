import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { BacklogPage } from './BacklogPage'

const FULL_SCOPE =
  '?schedule_config_id=strict_priority__observed_calendar&policy_id=pure_risk&fold_set=quarterly&fold_id=quarterly-2026Q1&k_name=k_1_day'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[`/backlog${path}`]}>
      <BacklogPage />
    </MemoryRouter>,
  )
}

describe('BacklogPage', () => {
  it('renders the waiting establishment in plain language', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('E-2')).toBeInTheDocument()
    expect(screen.getByText('Waiting position')).toBeInTheDocument()
    expect(screen.getByText('Next available capacity')).toBeInTheDocument()
    // The fixture's backlog_reason code isn't one copy.ts recognizes, so it falls back to the
    // raw string verbatim rather than disappearing.
    expect(screen.getByText('horizon exhausted before this rank')).toBeInTheDocument()
  })

  it('explains that a waiting establishment was recommended, not rejected', async () => {
    renderAt(FULL_SCOPE)
    expect(
      await screen.findByText(/did not fit into the currently available inspection capacity/),
    ).toBeInTheDocument()
    expect(screen.getByText(/have not disappeared or been rejected/)).toBeInTheDocument()
  })

  it('shows a loading state, not an error, while scope is still being filled in automatically', async () => {
    renderAt('')
    expect(await screen.findByText(/Preparing an inspection plan/)).toBeInTheDocument()
  })
})
