import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { NavBar } from './NavBar'

const EXPECTED_LINKS: [string, string][] = [
  ['Today', '/'],
  ['Field Plan', '/geographic-plan'],
  ['Plan Review', '/plan-review'],
  ['Backtest Summary', '/plan'],
  ['Priority List', '/recommendations'],
  ['Full Schedule', '/schedule'],
  ['Waiting', '/backlog'],
  ['Needs Attention', '/review'],
]

describe('NavBar', () => {
  it('renders every expected link with its correct route', () => {
    render(
      <MemoryRouter>
        <NavBar />
      </MemoryRouter>,
    )
    for (const [label, to] of EXPECTED_LINKS) {
      const link = screen.getByRole('link', { name: label })
      expect(link).toHaveAttribute('href', to)
    }
  })

  it('puts the live operational plan pages right after Today, ahead of the analysis pages', () => {
    render(
      <MemoryRouter>
        <NavBar />
      </MemoryRouter>,
    )
    const labels = screen.getAllByRole('link').map((el) => el.textContent)
    expect(labels.indexOf('Today')).toBeLessThan(labels.indexOf('Field Plan'))
    expect(labels.indexOf('Field Plan')).toBeLessThan(labels.indexOf('Plan Review'))
    expect(labels.indexOf('Plan Review')).toBeLessThan(labels.indexOf('Priority List'))
  })
})
