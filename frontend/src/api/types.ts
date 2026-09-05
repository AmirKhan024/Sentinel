/**
 * TypeScript mirrors of the Sentinel API's Pydantic response/request schemas.
 *
 * Field names and nullability match `src/sentinel/api/schemas/*.py` exactly. This file
 * introduces no field the API does not return -- if something looks missing here, it is
 * missing from the API too.
 */

// --- Common shapes (src/sentinel/api/schemas/common.py) --------------------

export interface DecisionScope {
  policy_id?: string
  model_name?: string
  fold_set?: string
  fold_id?: string
  k_name?: string
  schedule_config_id?: string
  planning_run_id?: string
  replan_index?: number
}

export interface RunInfo {
  path: string
  manifest_path: string | null
  built_at: string | null
}

export interface PageMeta {
  offset: number
  limit: number
  total: number
}

export interface Page<T> {
  data: T[]
  page: PageMeta
  run: RunInfo
}

export interface StagedRequestStatus {
  request_id: string
  kind: string
  natural_id: string
  status: string
  staged_at: string
  payload: Record<string, unknown>
}

/** What every write endpoint returns. Never a recomputed artifact -- see ADR 0049: the request
 * is pending until an operator drains the staging store through the batch CLI. */
export interface StagedRequestReceipt {
  request_id: string
  kind: string
  natural_id: string
  status: string
  staged_at: string
}

// --- Policy (src/sentinel/api/schemas/policy.py) ----------------------------

export interface RecommendationOut {
  policy_id: string
  model_name: string
  fold_set: string
  fold_id: string
  k_name: string
  k: number
  target_inspection_id: string
  establishment_id: string
  establishment_name: string | null
  establishment_address: string | null
  inspection_date: string
  base_score: number
  score: number
  model_rank: number
  final_policy_rank: number | null
  is_selected: boolean
  decision_mechanism: string
  decision_reason: string
  coverage_eligible: boolean
  secondary_no_history: boolean
  warnings: string
  group_value: string
  group_status: string
  policy_definition_version: string
}

export interface AllocationOut {
  policy_id: string
  model_name: string
  fold_set: string
  fold_id: string
  k_name: string
  k: number
  n_universe: number
  reserve_mechanism: string
  reserve_share: number
  reserve_target: number
  n_eligible_available: number
  n_eligible_in_risk_top_k: number
  n_risk: number
  n_reserve: number
  n_selected: number
  reserve_inert: boolean
  policy_definition_version: string
}

/** Exactly Component 13's `Override` contract -- the API adds no field of its own. */
export interface OverrideIn {
  override_id: string
  policy_id: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  action: string
  reason_code: string
  actor: string
  decided_at: string
}

export interface OverrideLogRowOut {
  override_id: string
  policy_id: string
  fold_set: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  action: string
  reason_code: string
  actor: string
  decided_at: string
  original_is_selected: boolean | null
  original_mechanism: string
  original_reason: string
  original_policy_rank: number | null
  final_is_selected: boolean | null
  displaced_target_inspection_id: string
  outcome: string
  policy_definition_version: string
  status: string
}

// --- Scheduling (src/sentinel/api/schemas/scheduling.py) -------------------

export interface ScheduleRowOut {
  schedule_config_id: string
  policy_id: string
  model_name: string
  fold_set: string
  fold_id: string
  k_name: string
  k: number
  target_inspection_id: string
  establishment_id: string
  establishment_name: string | null
  establishment_address: string | null
  recommendation_date: string
  base_score: number
  score: number
  model_rank: number
  final_policy_rank: number
  decision_mechanism: string
  decision_reason: string
  coverage_eligible: boolean
  warnings: string
  recommendation_override_id: string
  policy_definition_version: string
  planning_run_id: string
  replan_index: number
  schedule_status: string
  schedule_reason: string
  inversion_reason: string
  scheduled_date: string | null
  day_index: number | null
  slot_index: number | null
  schedule_rank: number | null
  wait_operating_days: number | null
  original_scheduled_date: string | null
  original_schedule_rank: number | null
  adjustment_id: string
  is_scenario: boolean
  schedule_definition_version: string
}

export interface BacklogRowOut {
  schedule_config_id: string
  policy_id: string
  model_name: string
  fold_set: string
  fold_id: string
  k_name: string
  k: number
  target_inspection_id: string
  establishment_id: string
  establishment_name: string | null
  establishment_address: string | null
  final_policy_rank: number
  decision_mechanism: string
  decision_reason: string
  coverage_eligible: boolean
  backlog_position: number
  backlog_reason: string
  horizon_slots: number
  slots_short: number
  would_fit_on_day_index: number | null
  first_available_date: string | null
  planning_run_id: string
  replan_index: number
  is_scenario: boolean
  schedule_definition_version: string
}

export interface ReplanningRunOut {
  schedule_config_id: string
  policy_id: string
  fold_set: string
  fold_id: string
  k_name: string
  planning_run_id: string
  replan_index: number
  parent_replan_index: number | null
  replan_from_date: string | null
  trigger: string
  n_preserved_completed: number
  n_preserved_past: number
  n_returned_to_queue: number
  n_cancelled: number
  n_newly_scheduled: number
  n_still_backlog: number
  remaining_slots: number
  schedule_definition_version: string
}

export interface ExecutionSummaryOut {
  schedule_config_id: string
  policy_id: string
  model_name: string
  fold_set: string
  fold_id: string
  k_name: string
  n_scheduled: number
  n_completed: number
  n_not_performed: number
  n_cancelled_in_field: number
  n_no_execution_record: number
  completion_rate: number
  final_replan_index: number
  schedule_definition_version: string
}

/** Exactly Component 14's `Adjustment` contract. */
export interface AdjustmentIn {
  adjustment_id: string
  schedule_config_id: string
  policy_id: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  action: string
  target_date: string
  reason_code: string
  actor: string
  decided_at: string
}

export interface AdjustmentLogRowOut {
  adjustment_id: string
  schedule_config_id: string
  policy_id: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  action: string
  target_date: string
  reason_code: string
  actor: string
  decided_at: string
  original_status: string
  final_status: string
  displaced_target_inspection_id: string
  outcome: string
  planning_run_id: string
  replan_index: number
  schedule_definition_version: string
  status: string
}

/** Exactly Component 14's `ExecutionEvent` contract. */
export interface ExecutionEventIn {
  execution_id: string
  schedule_config_id: string
  policy_id: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  scheduled_date: string
  execution_status: string
  reason_code: string
  actor: string
  observed_at: string
}

export interface ExecutionLogRowOut {
  execution_id: string
  schedule_config_id: string
  policy_id: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  scheduled_date: string | null
  plan_scheduled_date: string | null
  execution_status: string
  reason_code: string
  actor: string
  observed_at: string
  outcome: string
  triggers_replan: boolean
  applied_at_replan_index: number
  schedule_definition_version: string
  status: string
}

/** One row of `/v1/execution/contract` -- the authoritative, data-driven source for which
 * `execution_status` values a person may submit (never hardcoded in the frontend). */
export interface ExecutionContractField {
  contract_name: string
  field_name: string
  required: boolean
  dtype: string
  allowed_values: string
  meaning: string
}

// --- Explanations (src/sentinel/api/schemas/explain.py) --------------------

export interface ExplanationValueOut {
  feature_name: string
  original_feature_name: string
  derived_from: string
  feature_kind: string
  feature_value: number | null
  transformed_value: number | null
  shap_value: number
  output_space: string
  is_exact: boolean
}

export interface ExplanationCaseOut {
  model_name: string
  model_version: string
  fold_set: string
  fold_id: string
  target_inspection_id: string
  output_space: string
  explanation_method: string
  is_exact: boolean
  base_value: number
  prediction_value: number
  reconstruction_value: number
  additivity_holds: boolean
  n_features: number
  base_score: number
  calibrated_probability: number | null
  base_model_trained_through: string | null
  sample_strategy: string
  values: ExplanationValueOut[]
}

export interface SupportOut {
  model_name: string
  explanation_status: string
  explanation_method: string | null
  output_space: string | null
  is_exact: boolean
  is_experimental: boolean
  rationale: string
  unsupported_reason: string | null
}

// --- Human review (src/sentinel/api/schemas/review.py) ---------------------

export interface ReviewCaseOut {
  policy_id: string
  model_name: string
  fold_set: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  establishment_id: string
  establishment_name: string | null
  establishment_address: string | null
  final_policy_rank: number | null
  decision_mechanism: string
  decision_reason: string
  warnings: string
  trigger_reasons: string
  schedule_config_id: string
  planning_run_id: string
  replan_index: number | null
  scheduled_date: string | null
  review_status: string
  review_id: string
  resolution_action: string
  review_definition_version: string
  status: string
}

/** Exactly Component 16's `ReviewResolution` contract. */
export interface ResolutionIn {
  review_id: string
  policy_id: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  resolution_action: string
  reason_code: string
  actor: string
  decided_at: string
  referenced_override_id?: string | null
  referenced_adjustment_id?: string | null
  escalation_note?: string | null
}

export interface ResolutionLogRowOut {
  review_id: string
  policy_id: string
  fold_id: string
  k_name: string
  target_inspection_id: string
  resolution_action: string
  reason_code: string
  actor: string
  decided_at: string
  referenced_override_id: string
  referenced_adjustment_id: string
  escalation_note: string
  original_status: string
  final_status: string
  outcome: string
  review_definition_version: string
  status: string
}

// --- Geographic plan / supervisor plan review (src/sentinel/api/schemas/plan_review.py) ---

export interface PlanRowOut {
  planning_date: string
  establishment_id: string
  target_inspection_id: string
  canonical_name: string | null
  canonical_address: string | null
  establishment_name: string | null
  establishment_address: string | null

  calibrated_score: number
  base_score: number
  rank: number
  policy_rank: number | null
  selection_reason: string
  selection_mechanism: string

  /** Display-only field-work ordering: the supervisor's `adjust_operational_priority` value
   * where recorded, else exactly `policy_rank`. Never a substitute for `rank`/`policy_rank`,
   * both of which are always present above, unedited. */
  operational_priority: number | null

  location_status: string
  work_block_id: string
  work_block_label: string
  suggested_order_in_block: number | null
  organization_mode: string
  highest_sentinel_rank_in_block: number | null

  supervisor_decision_id: string | null
  supervisor_decision_action: string | null
  supervisor_decision_reason_code: string | null
  supervisor_decision_actor: string | null
  supervisor_decision_decided_at: string | null
  supervisor_revised_planned_date: string | null
  supervisor_revised_work_block_id: string | null
  supervisor_revised_operational_priority: number | null

  /** Component 17's own as-of feature row for this candidate, reused verbatim -- `null` only
   * when the operational candidate table has no row for this `target_inspection_id`. */
  history_factors: RiskHistoryFactorsOut | null
}

export interface WorkBlockOut {
  work_block_id: string
  work_block_label: string
  size: number
  highest_sentinel_rank: number | null
  rank_range: [number, number] | null
  is_unmapped: boolean
  decisions_recorded: number
}

export interface PlanSummaryOut {
  planning_date: string
  selected_inspection_workload: number
  location_available_count: number
  location_unavailable_count: number
  work_block_count: number
  decisions_recorded: number
  approval_status: string
}

export interface PlanDecisionLogRowOut {
  decision_id: string
  planning_date: string
  target_inspection_id: string
  decision_action: string
  reason_code: string
  actor: string
  decided_at: string
  revised_planned_date: string | null
  revised_work_block_id: string | null
  outcome: string
  plan_review_definition_version: string
  status: string
}

/** Exactly Component 21's `PlanDecision` contract. */
export interface PlanDecisionIn {
  decision_id: string
  planning_date: string
  target_inspection_id: string
  decision_action: string
  reason_code: string
  actor: string
  decided_at: string
  revised_planned_date?: string | null
  revised_work_block_id?: string | null
  revised_operational_priority?: number | null
}

/** Exactly Component 21's `PlanApprovalRequest` contract. */
export interface PlanApprovalIn {
  approval_id: string
  planning_date: string
  approved_by: string
  approved_at: string
  note?: string | null
}

export interface PlanApprovalOut {
  approval_id: string
  planning_date: string
  approved_by: string
  approved_at: string
  note: string | null
  final_selected_count: number
  final_active_count: number
  final_deferred_count: number
  final_not_proceeding_count: number
  final_undecided_count: number
  source_plan_review_path: string
  source_plan_review_sha256: string
}

// --- Establishment bundle (src/sentinel/api/schemas/establishment.py) ------

export interface RiskHistoryFactorsOut {
  prior_canvass_count_code_era: number | null
  prior_canvass_priority_count: number | null
  prior_canvass_priority_rate: number | null
  prior_canvass_fail_rate: number | null
  fail_at_last_canvass: boolean | null
  priority_at_last_canvass: boolean | null
  days_since_last_canvass: number | null
  days_since_any_inspection: number | null
  prior_inspection_count_any_type: number | null
  name_changed_since_last_canvass: boolean | null
}

export interface EstablishmentHistoryOut {
  establishment_id: string
  establishment_name: string | null
  establishment_address: string | null
  recommendation: RecommendationOut
  schedule: ScheduleRowOut | null
  explanation: ExplanationCaseOut | null
  explanation_unavailable_reason: string | null
  history_factors: RiskHistoryFactorsOut | null
  history_factors_unavailable_reason: string | null
}

// --- Meta / manifests (src/sentinel/api/routers/meta.py) -------------------

export interface RunListEntry {
  component: string
  path: string
  name: string
}

export interface ManifestCheck {
  name: string
  detail: string
  passed: boolean
  severity: string
}

/** Manifests are arbitrary JSON dicts written by each pipeline component -- this names only
 * the fields the frontend actually reads; anything else present is ignored, not dropped. */
export interface ManifestJson {
  built_at?: string
  row_counts?: Record<string, number>
  checks?: ManifestCheck[]
  advisories?: string[]
  does_not_establish?: string[]
  policy_grid?: { policy_id: string; reserve_mechanism: string; reserve_share: number }[]
  k_levels?: string[]
  candidate_models?: string[]
  selected_model?: string
  primary_k_level?: string
  config_grid?: {
    schedule_config_id: string
    capacity_mode: string
    is_default: boolean
    is_scenario: boolean
    rationale: string
  }[]
  default_capacity_mode?: string
  ranked_candidate_count?: number
  selectable_candidate_count?: number
  selected_count?: number
  risk_selected_count?: number
  reserve_selected_count?: number
  coverage_eligible_selected_count?: number
  [key: string]: unknown
}
