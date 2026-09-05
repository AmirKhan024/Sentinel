import { apiFetch, apiPost } from './client'
import type { PageParams } from './recommendations'
import type { DecisionScope, OverrideIn, OverrideLogRowOut, Page, StagedRequestReceipt } from './types'

export function listOverrides(
  scope: DecisionScope,
  filters: { target_inspection_id?: string },
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<OverrideLogRowOut>> {
  return apiFetch('/v1/policy/overrides', { ...scope, ...filters, ...page }, signal)
}

/** Stages a human override of a recommendation (force_include / force_exclude). Never applied
 * immediately -- see ADR 0049: this only appends to a pending file an operator later runs
 * through `sentinel decide --overrides`. */
export function submitOverride(payload: OverrideIn, signal?: AbortSignal): Promise<StagedRequestReceipt> {
  return apiPost('/v1/policy/overrides', payload, signal)
}
