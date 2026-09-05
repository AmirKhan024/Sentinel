const STEPS = [
  'Sentinel looks at available inspection information for every establishment.',
  'A prioritization policy decides which establishments are recommended for inspection first.',
  'Available inspection capacity decides which recommendations fit into the schedule.',
  'Recommendations that do not fit yet wait for future capacity.',
  'Some cases are flagged for a person to review before anyone acts on them.',
  'For a live planning date, Sentinel identifies which establishments are eligible and scores them with the same prioritization model.',
  'The eligible establishments that fit today’s capacity are grouped into geographic work areas.',
  'A supervisor reviews the proposed plan, can change individual decisions, and approves it before anyone acts on it in the field.',
]

/** The eight-step plain-language story every page implicitly assumes the visitor already knows
 * -- the first five describe the historical/backtest analysis pages, the last three describe the
 * live operational plan (Field Plan and Plan Review). Shown once, in full, on the Overview page
 * -- so "why is this establishment in the backlog" or "what does flagged for review mean" never
 * requires reading project documentation first. */
export function WorkflowDiagram() {
  return (
    <ol className="workflow-diagram">
      {STEPS.map((step, i) => (
        <li key={i}>
          <span className="workflow-step-number">{i + 1}</span>
          <span className="workflow-step-text">{step}</span>
        </li>
      ))}
    </ol>
  )
}
