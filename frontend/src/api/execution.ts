import { apiFetch, apiPost } from './client'
import type { PageParams } from './recommendations'
import type {
  DecisionScope,
  ExecutionContractField,
  ExecutionEventIn,
  ExecutionLogRowOut,
  ExecutionSummaryOut,
  Page,
  StagedRequestReceipt,
} from './types'

export function getExecutionSummary(
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<ExecutionSummaryOut> {
  return apiFetch('/v1/execution/summary', { ...scope }, signal)
}

export function listExecutionEvents(
  scope: DecisionScope,
  filters: { target_inspection_id?: string },
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<ExecutionLogRowOut>> {
  return apiFetch('/v1/execution/events', { ...scope, ...filters, ...page }, signal)
}

/** The authoritative, data-driven set of `execution_status` values a person may submit --
 * read from Component 14's own contract table rather than a hardcoded list. */
export function getExecutionContract(signal?: AbortSignal): Promise<ExecutionContractField[]> {
  return apiFetch('/v1/execution/contract', {}, signal)
}

/** Stages a report of what actually happened to a planned inspection. Never applied
 * immediately -- see ADR 0049: an operator later runs it through `sentinel schedule
 * --execution`. */
export function submitExecutionEvent(
  payload: ExecutionEventIn,
  signal?: AbortSignal,
): Promise<StagedRequestReceipt> {
  return apiPost('/v1/execution/events', payload, signal)
}
