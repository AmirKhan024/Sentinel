import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { ScheduleDayPage } from './ScheduleDayPage'

const FULL_SCOPE =
  '?schedule_config_id=strict_priority__observed_calendar&policy_id=pure_risk&fold_set=quarterly&fold_id=quarterly-2026Q1&k_name=k_1_day'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[`/schedule/day${path}`]}>
      <ScheduleDayPage />
    </MemoryRouter>,
  )
}

describe('ScheduleDayPage', () => {
  it('shows a loading state while scope is still being filled in automatically', async () => {
    renderAt('')
    expect(await screen.findByText(/Preparing an inspection plan/)).toBeInTheDocument()
  })

  it('shows the day heading and the scheduled establishment', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Eat A Pita')).toBeInTheDocument()
  })

  it('is labeled as historical, never as today', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('Historical Day View')).toBeInTheDocument()
    expect(screen.getByText(/not today's real operations/)).toBeInTheDocument()
  })

  it('has a Technical details section carrying the raw scope fields', async () => {
    renderAt(FULL_SCOPE)
    await screen.findByText('Eat A Pita')
    expect(screen.getByText('Technical details')).toBeInTheDocument()
    // <details> is not display:none, so jsdom still renders its (collapsed) children -- this
    // just confirms the field made it into the technical section, matching OverviewPage's own
    // test convention rather than asserting DOM-level visibility.
    expect(screen.getAllByText('schedule_config_id').length).toBeGreaterThan(0)
  })
})
