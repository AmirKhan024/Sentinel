/** Generates a natural id for a staged write. The backend imposes no format on
 * override_id/adjustment_id/execution_id/review_id beyond "non-blank and unique" (see
 * `parse_overrides`/`parse_adjustments`/`parse_execution_events`/`parse_resolutions`) -- this
 * only needs to not collide, which `crypto.randomUUID()` guarantees for practical purposes. */
export function generateId(prefix: string): string {
  const uuid =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`
  return `${prefix}-${uuid.slice(0, 8)}`
}
