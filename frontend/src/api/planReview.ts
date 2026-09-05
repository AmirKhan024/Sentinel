import { apiFetch, apiPost } from './client'
import type { PageParams } from './recommendations'
import type {
  Page,
  PlanApprovalIn,
  PlanApprovalOut,
  PlanDecisionIn,
  PlanDecisionLogRowOut,
  PlanRowOut,
  PlanSummaryOut,
  StagedRequestReceipt,
  WorkBlockOut,
} from './types'

/** Component 20/21 are scoped by `planning_date`, not the historical `DecisionScope`
 * (policy/fold/k) the rest of this app uses -- there is no backtest cell here, only a live
 * planning run. `planning_date` is optional everywhere below: omitting it asks the API for the
 * most recently built plan review. */

export function getPlanSummary(
  planningDate: string | undefined,
  signal?: AbortSignal,
): Promise<PlanSummaryOut> {
  return apiFetch('/v1/plan-review/summary', { planning_date: planningDate }, signal)
}

export function listPlanRows(
  planningDate: string | undefined,
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<PlanRowOut>> {
  return apiFetch('/v1/plan-review/rows', { planning_date: planningDate, ...page }, signal)
}

export function getPlanRow(
  targetInspectionId: string,
  planningDate: string | undefined,
  signal?: AbortSignal,
): Promise<PlanRowOut> {
  return apiFetch(
    `/v1/plan-review/rows/${encodeURIComponent(targetInspectionId)}`,
    { planning_date: planningDate },
    signal,
  )
}

export function listWorkBlocks(
  planningDate: string | undefined,
  signal?: AbortSignal,
): Promise<WorkBlockOut[]> {
  return apiFetch('/v1/plan-review/work-blocks', { planning_date: planningDate }, signal)
}

export function listPlanDecisions(
  planningDate: string | undefined,
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<PlanDecisionLogRowOut>> {
  return apiFetch('/v1/plan-review/decisions', { planning_date: planningDate, ...page }, signal)
}

/** Stages a supervisor's decision about one establishment in the proposed plan (keep_selected /
 * move_to_later_workday / do_not_proceed_as_planned). Never applied immediately -- see ADR 0049:
 * an operator later runs it through `sentinel review-plan --decisions`. Never edits Sentinel's
 * own recommendation or Component 20's geographic organization -- it only records the
 * supervisor's decision as an additional, auditable fact. */
export function submitPlanDecision(
  payload: PlanDecisionIn,
  signal?: AbortSignal,
): Promise<StagedRequestReceipt> {
  return apiPost('/v1/plan-review/decisions', payload, signal)
}

/** The latest committed approval for a planning date, if the plan has been approved. */
export function getPlanApproval(
  planningDate: string | undefined,
  signal?: AbortSignal,
): Promise<PlanApprovalOut> {
  return apiFetch('/v1/plan-review/approval', { planning_date: planningDate }, signal)
}

/** Stages a supervisor's approval of the whole plan (ADR 0049): never applied immediately --
 * an operator later runs it through `sentinel approve-plan`, which re-runs the full readiness
 * checklist (every row carries the machine recommendation, every decision has a reason, ...)
 * before writing the immutable `approved_operational_plan` artifact. */
export function submitApproval(
  payload: PlanApprovalIn,
  signal?: AbortSignal,
): Promise<StagedRequestReceipt> {
  return apiPost('/v1/plan-review/approve', payload, signal)
}
