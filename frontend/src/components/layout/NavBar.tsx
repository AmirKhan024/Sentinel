import { NavLink } from 'react-router-dom'

const LANDING_LINK = { to: '/', label: 'Today', end: true }

/** The current, live operational plan for a real planning date -- Components 20/21. */
const OPERATIONAL_LINKS = [
  { to: '/geographic-plan', label: 'Field Plan' },
  { to: '/plan-review', label: 'Plan Review' },
]

/** Historical/backtest analysis surfaces -- fold-and-policy-scoped, not a live plan. */
const ANALYSIS_LINKS = [
  { to: '/plan', label: 'Backtest Summary' },
  { to: '/recommendations', label: 'Priority List' },
  { to: '/schedule', label: 'Full Schedule' },
  { to: '/backlog', label: 'Waiting' },
  { to: '/review', label: 'Needs Attention' },
]

function NavLinks({ links }: { links: { to: string; label: string; end?: boolean }[] }) {
  return (
    <>
      {links.map((link) => (
        <NavLink key={link.to} to={link.to} end={link.end} className={({ isActive }) => (isActive ? 'active' : '')}>
          {link.label}
        </NavLink>
      ))}
    </>
  )
}

export function NavBar() {
  return (
    <nav className="nav-bar">
      <span className="nav-brand">Sentinel</span>
      <NavLinks links={[LANDING_LINK]} />
      <span className="nav-divider" aria-hidden="true" />
      <NavLinks links={OPERATIONAL_LINKS} />
      <span className="nav-divider" aria-hidden="true" />
      <NavLinks links={ANALYSIS_LINKS} />
    </nav>
  )
}
