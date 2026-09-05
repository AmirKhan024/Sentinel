import { foldLabel } from '../../lib/copy'

/**
 * The one thing every backtest-side page must say before showing any data: this is a
 * historical simulation, not the current live plan. Sentinel has two structurally separate
 * worlds -- a live, `planning_date`-scoped operational plan (Today, Field Plan, Plan Review) and
 * a historical, fold-scoped backtest (this page and its siblings) -- and nothing else on screen
 * makes that boundary obvious enough on its own. Rendered once, prominently, before any numbers,
 * so a reader never mistakes "28 selected" here for anything to do with today's real plan.
 */
export function BacktestBanner({ foldId }: { foldId: string | undefined }) {
  return (
    <div className="backtest-banner" role="note">
      <strong>Historical simulation</strong> — {foldLabel(foldId)}. Not a current plan.
    </div>
  )
}
