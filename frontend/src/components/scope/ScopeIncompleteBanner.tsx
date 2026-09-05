/** Shown instead of firing a request, while required scope fields are still unset. Phrased
 * against the API's own `missing_scope_fields` vocabulary (ADR 0050: never guess a scope). */
export function ScopeIncompleteBanner({
  missing,
  what,
}: {
  missing: string[]
  what: string
}) {
  return (
    <div className="state state-scope-incomplete">
      Select {missing.join(', ')} to view {what}.
    </div>
  )
}
