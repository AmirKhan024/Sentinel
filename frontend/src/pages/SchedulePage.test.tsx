import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { SchedulePage } from './SchedulePage'

const FULL_SCOPE =
  '?schedule_config_id=strict_priority__observed_calendar&policy_id=pure_risk&fold_set=quarterly&fold_id=quarterly-2026Q1&k_name=k_1_day'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[`/schedule${path}`]}>
      <SchedulePage />
    </MemoryRouter>,
  )
}

describe('SchedulePage', () => {
  it('labels this page as a historical simulation, not the current plan', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText(/Historical simulation/)).toBeInTheDocument()
  })

  it('links to the one-day-at-a-time historical view', async () => {
    renderAt(FULL_SCOPE)
    const link = await screen.findByText(/View one day at a time/)
    expect(link.closest('a')).toHaveAttribute('href', '/schedule/day')
  })

  it('explains what the schedule means before showing any row', async () => {
    renderAt(FULL_SCOPE)
    expect(
      await screen.findByText(/recommended inspections that fit into the currently available inspection capacity/),
    ).toBeInTheDocument()
  })

  it('shows the scheduled establishment with a plain-language status and planned date', async () => {
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('E-1')).toBeInTheDocument()
    expect(screen.getByText('Fits in available capacity')).toBeInTheDocument()
  })

  it('is honest that a planned date may be a past evaluation period, not a live calendar', async () => {
    renderAt(FULL_SCOPE)
    expect(
      await screen.findByText(/this is a prioritization plan, not a live calendar of upcoming appointments/),
    ).toBeInTheDocument()
  })

  it('shows a preparing state, then real data, when scope is filled in automatically', async () => {
    renderAt('')
    expect(await screen.findByText(/Preparing an inspection plan/)).toBeInTheDocument()
    expect(await screen.findByText('E-1', {}, { timeout: 3000 })).toBeInTheDocument()
  })

  it('shows the true 1-based order number, not one higher (regression: off-by-one display bug)', async () => {
    // scheduleRowFixture.slot_index is 1 -- the real, already-1-based value on disk. This used
    // to render as "#2" because the column added 1 to an already-1-based field.
    renderAt(FULL_SCOPE)
    expect(await screen.findByText('#1')).toBeInTheDocument()
    expect(screen.queryByText('#2')).not.toBeInTheDocument()
  })

  it('explains that this is a capacity plan, not an optimized travel route', async () => {
    renderAt(FULL_SCOPE)
    expect(
      await screen.findByText(/does not calculate travel routes, assign specific inspectors/),
    ).toBeInTheDocument()
  })

  it('discloses that daily capacity is a historical count, not a live staffing feed', async () => {
    renderAt(FULL_SCOPE)
    expect(
      await screen.findByText(/Chicago's own historical record shows for that same calendar day/),
    ).toBeInTheDocument()
  })
})
