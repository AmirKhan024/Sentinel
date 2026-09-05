/**
 * The one place "today" is computed. Sentinel's operational plan (Components 17-21) is scoped
 * by a real `planning_date`, never by a historical evaluation fold -- this file is what lets
 * the UI honestly compare a plan's own date against the real current date, instead of assuming
 * they match.
 */

/** The real current date, computed live -- never a hardcoded literal. Local time, not UTC, so
 * "today" matches the browser's own calendar day rather than flipping at midnight UTC (7-8pm
 * Chicago time), which would be wrong for a Chicago operations app. Call-time-evaluated (no
 * module-level constant) so it stays correct in a tab left open overnight, and is fake-able in
 * tests via `vi.setSystemTime`. */
export function currentOperationalDate(): string {
  const d = new Date()
  const yyyy = d.getFullYear()
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  return `${yyyy}-${mm}-${dd}`
}

/** Whether a plan's own `planning_date` is the real current date -- the one check that decides
 * whether a plan may honestly be called "today's plan" anywhere in the UI. */
export function isPlanningDateToday(planningDate: string | undefined | null): boolean {
  return Boolean(planningDate) && planningDate === currentOperationalDate()
}
