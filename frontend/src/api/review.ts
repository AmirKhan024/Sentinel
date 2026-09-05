import { apiFetch, apiPost } from './client'
import type { PageParams } from './recommendations'
import type {
  DecisionScope,
  Page,
  ResolutionIn,
  ResolutionLogRowOut,
  ReviewCaseOut,
  StagedRequestReceipt,
} from './types'

export function listReviewQueue(
  scope: DecisionScope,
  page: PageParams,
  signal?: AbortSignal,
  filters: { trigger?: string } = {},
): Promise<Page<ReviewCaseOut>> {
  return apiFetch('/v1/review/queue', { ...scope, ...filters, ...page }, signal)
}

export function getReviewCase(
  targetInspectionId: string,
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<ReviewCaseOut> {
  return apiFetch(`/v1/review/queue/${encodeURIComponent(targetInspectionId)}`, { ...scope }, signal)
}

export function listResolutions(
  scope: DecisionScope,
  page: PageParams,
  signal?: AbortSignal,
  filters: { target_inspection_id?: string } = {},
): Promise<Page<ResolutionLogRowOut>> {
  return apiFetch('/v1/review/resolutions', { ...scope, ...filters, ...page }, signal)
}

/** Stages a reviewer's decision about a flagged case (acknowledge / refer_to_override /
 * refer_to_adjustment / escalate). Never applied immediately -- see ADR 0049: an operator
 * later runs it through `sentinel review --resolutions`. Never edits the case, the
 * recommendation or the schedule itself (ADR 0051) -- referring a case only records a pointer
 * to an override or adjustment submitted separately through those contracts. */
export function submitResolution(
  payload: ResolutionIn,
  signal?: AbortSignal,
): Promise<StagedRequestReceipt> {
  return apiPost('/v1/review/resolutions', payload, signal)
}
