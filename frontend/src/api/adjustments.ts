import { apiFetch, apiPost } from './client'
import type { PageParams } from './recommendations'
import type { AdjustmentIn, AdjustmentLogRowOut, DecisionScope, Page, StagedRequestReceipt } from './types'

export function listAdjustments(
  scope: DecisionScope,
  filters: { target_inspection_id?: string },
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<AdjustmentLogRowOut>> {
  return apiFetch('/v1/schedule/adjustments', { ...scope, ...filters, ...page }, signal)
}

/** Stages a human adjustment (defer_to_date / advance_to_date / cancel) of a planned inspection.
 * Never applied immediately -- see ADR 0049: an operator later runs it through
 * `sentinel schedule --adjustments`. */
export function submitAdjustment(
  payload: AdjustmentIn,
  signal?: AbortSignal,
): Promise<StagedRequestReceipt> {
  return apiPost('/v1/schedule/adjustments', payload, signal)
}
