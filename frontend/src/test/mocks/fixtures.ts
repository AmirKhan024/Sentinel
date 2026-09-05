import type {
  BacklogRowOut,
  EstablishmentHistoryOut,
  ManifestJson,
  Page,
  PlanRowOut,
  PlanSummaryOut,
  RecommendationOut,
  ReviewCaseOut,
  RiskHistoryFactorsOut,
  ScheduleRowOut,
  WorkBlockOut,
} from '../../api/types'

export const recommendationFixture: RecommendationOut = {
  policy_id: 'pure_risk',
  model_name: 'lightgbm_platt',
  fold_set: 'quarterly',
  fold_id: 'quarterly-2026Q1',
  k_name: 'k_1_day',
  k: 22,
  target_inspection_id: 'TI-1',
  establishment_id: 'E-1',
  establishment_name: 'Eat A Pita',
  establishment_address: '3155 N Halsted St',
  inspection_date: '2026-01-15',
  base_score: 0.41,
  score: 0.55,
  model_rank: 1,
  final_policy_rank: 1,
  is_selected: true,
  decision_mechanism: 'risk',
  decision_reason: 'top of risk-ranked queue',
  coverage_eligible: false,
  secondary_no_history: false,
  warnings: '',
  group_value: '__UNKNOWN__',
  group_status: 'unsupported',
  policy_definition_version: 'v1',
}

export const recommendationsPageFixture: Page<RecommendationOut> = {
  data: [recommendationFixture],
  page: { offset: 0, limit: 50, total: 1 },
  run: { path: '/data/inspection_recommendations.parquet', manifest_path: null, built_at: '2026-08-26T00:00:00Z' },
}

export const scheduleRowFixture: ScheduleRowOut = {
  schedule_config_id: 'strict_priority__observed_calendar',
  policy_id: 'pure_risk',
  model_name: 'lightgbm_platt',
  fold_set: 'quarterly',
  fold_id: 'quarterly-2026Q1',
  k_name: 'k_1_day',
  k: 22,
  target_inspection_id: 'TI-1',
  establishment_id: 'E-1',
  establishment_name: 'Eat A Pita',
  establishment_address: '3155 N Halsted St',
  recommendation_date: '2026-01-15',
  base_score: 0.41,
  score: 0.55,
  model_rank: 1,
  final_policy_rank: 1,
  decision_mechanism: 'risk',
  decision_reason: 'top of risk-ranked queue',
  coverage_eligible: false,
  warnings: '',
  recommendation_override_id: '',
  policy_definition_version: 'v1',
  planning_run_id: 'run-1',
  replan_index: 0,
  schedule_status: 'scheduled',
  schedule_reason: 'fit within horizon at rank 1',
  inversion_reason: '',
  scheduled_date: '2026-01-20',
  day_index: 1,
  // 1-based on disk (see docs/data_contracts/schedule.md): the first slot placed on a day is 1,
  // never 0. Kept realistic here deliberately -- an earlier, incorrect fixture value of 0 masked
  // the SchedulePage off-by-one display bug from this exact test suite.
  slot_index: 1,
  schedule_rank: 1,
  wait_operating_days: 5,
  original_scheduled_date: null,
  original_schedule_rank: null,
  adjustment_id: '',
  is_scenario: false,
  schedule_definition_version: 'v1',
}

export const backlogRowFixture: BacklogRowOut = {
  schedule_config_id: 'strict_priority__observed_calendar',
  policy_id: 'pure_risk',
  model_name: 'lightgbm_platt',
  fold_set: 'quarterly',
  fold_id: 'quarterly-2026Q1',
  k_name: 'k_1_day',
  k: 22,
  target_inspection_id: 'TI-2',
  establishment_id: 'E-2',
  establishment_name: 'La Gondola',
  establishment_address: '2914 N Ashland Ave',
  final_policy_rank: 22,
  decision_mechanism: 'risk',
  decision_reason: 'top of risk-ranked queue',
  coverage_eligible: false,
  backlog_position: 1,
  backlog_reason: 'horizon exhausted before this rank',
  horizon_slots: 20,
  slots_short: 2,
  would_fit_on_day_index: null,
  first_available_date: '2026-01-25',
  planning_run_id: 'run-1',
  replan_index: 0,
  is_scenario: false,
  schedule_definition_version: 'v1',
}

export const historyFactorsFixture: RiskHistoryFactorsOut = {
  prior_canvass_count_code_era: 7,
  prior_canvass_priority_count: 6,
  prior_canvass_priority_rate: 0.857143,
  prior_canvass_fail_rate: 0.333333,
  fail_at_last_canvass: false,
  priority_at_last_canvass: true,
  days_since_last_canvass: 345,
  days_since_any_inspection: 331,
  prior_inspection_count_any_type: 26,
  name_changed_since_last_canvass: true,
}

export const establishmentHistoryFixture: EstablishmentHistoryOut = {
  establishment_id: 'E-1',
  establishment_name: 'Eat A Pita',
  establishment_address: '3155 N Halsted St',
  recommendation: recommendationFixture,
  schedule: scheduleRowFixture,
  explanation: null,
  explanation_unavailable_reason: 'model has no attribution support for this fold',
  history_factors: historyFactorsFixture,
  history_factors_unavailable_reason: null,
}

export const scheduleDatesFixture: { scheduled_date: string; n_establishments: number }[] = [
  { scheduled_date: '2026-01-20', n_establishments: 1 },
]

export const scheduleRowsPageFixture: Page<ScheduleRowOut> = {
  data: [scheduleRowFixture],
  page: { offset: 0, limit: 50, total: 1 },
  run: { path: '/data/inspection_schedule.parquet', manifest_path: null, built_at: '2026-08-26T00:00:00Z' },
}

export const backlogPageFixture: Page<BacklogRowOut> = {
  data: [backlogRowFixture],
  page: { offset: 0, limit: 50, total: 1 },
  run: { path: '/data/schedule_backlog.parquet', manifest_path: null, built_at: '2026-08-26T00:00:00Z' },
}

export const reviewCaseFixture: ReviewCaseOut = {
  policy_id: 'pure_risk',
  model_name: 'lightgbm_platt',
  fold_set: 'quarterly',
  fold_id: 'quarterly-2026Q1',
  k_name: 'k_1_day',
  target_inspection_id: 'TI-1',
  establishment_id: 'E-1',
  establishment_name: 'Eat A Pita',
  establishment_address: '3155 N Halsted St',
  final_policy_rank: 1,
  decision_mechanism: 'risk_priority',
  decision_reason: 'selected_by_risk_rank',
  warnings: 'limited_history',
  trigger_reasons: 'policy_warning_present',
  schedule_config_id: '',
  planning_run_id: '',
  replan_index: null,
  scheduled_date: null,
  review_status: 'flagged',
  review_id: '',
  resolution_action: '',
  review_definition_version: 'v1',
  status: 'committed',
}

export const reviewQueuePageFixture: Page<ReviewCaseOut> = {
  data: [reviewCaseFixture],
  page: { offset: 0, limit: 50, total: 1 },
  run: { path: '/data/human_review_queue.parquet', manifest_path: null, built_at: '2026-08-27T00:00:00Z' },
}

export const policyManifestFixture: ManifestJson = {
  built_at: '2026-08-26T07:58:12Z',
  row_counts: { inspection_recommendations: 1453760 },
  checks: [{ name: 'tables_are_deterministically_sorted', detail: '0 issues', passed: true, severity: 'error' }],
  advisories: ['a_winner_was_determined: the data does not determine the correct policy'],
  does_not_establish: ['that the recommended queue is the correct queue'],
  policy_grid: [{ policy_id: 'pure_risk', reserve_mechanism: 'none', reserve_share: 0 }],
  k_levels: ['k_1_day', 'k_1_week'],
  candidate_models: ['lightgbm_platt', 'xgboost_platt'],
  selected_model: 'xgboost_platt',
  primary_k_level: 'k_1_day',
}

export const schedulingManifestFixture: ManifestJson = {
  built_at: '2026-08-26T13:14:16Z',
  row_counts: { inspection_schedule: 100000 },
  checks: [],
  advisories: [],
  config_grid: [
    {
      schedule_config_id: 'strict_priority__observed_calendar',
      capacity_mode: 'observed_calendar',
      is_default: true,
      is_scenario: false,
      rationale: 'the only configuration with contact with a real calendar',
    },
  ],
  k_levels: ['k_1_day', 'k_1_week'],
  default_capacity_mode: 'observed_calendar',
}

export const planRowFixture: PlanRowOut = {
  planning_date: '2026-08-28',
  establishment_id: 'E-1',
  target_inspection_id: 'CANDIDATE::2026-08-28::E-1',
  canonical_name: 'Eat A Pita',
  canonical_address: '3155 N Halsted St',
  establishment_name: 'Eat A Pita',
  establishment_address: '3155 N Halsted St',
  calibrated_score: 0.55,
  base_score: 0.41,
  rank: 1,
  policy_rank: 1,
  selection_reason: 'selected_by_risk_rank',
  selection_mechanism: 'risk_priority',
  operational_priority: 1,
  location_status: 'location_available',
  work_block_id: 'AREA-1',
  work_block_label: 'Area 1',
  suggested_order_in_block: 1,
  organization_mode: 'risk_first',
  highest_sentinel_rank_in_block: 1,
  supervisor_decision_id: null,
  supervisor_decision_action: null,
  supervisor_decision_reason_code: null,
  supervisor_decision_actor: null,
  supervisor_decision_decided_at: null,
  supervisor_revised_planned_date: null,
  supervisor_revised_work_block_id: null,
  supervisor_revised_operational_priority: null,
  history_factors: historyFactorsFixture,
}

export const workBlockFixture: WorkBlockOut = {
  work_block_id: 'AREA-1',
  work_block_label: 'Area 1',
  size: 1,
  highest_sentinel_rank: 1,
  rank_range: [1, 1],
  is_unmapped: false,
  decisions_recorded: 0,
}

export const workBlocksFixture: WorkBlockOut[] = [workBlockFixture]

export const planRowsPageFixture: Page<PlanRowOut> = {
  data: [planRowFixture],
  page: { offset: 0, limit: 500, total: 1 },
  run: { path: '/data/supervisor_plan_review.parquet', manifest_path: null, built_at: '2026-08-28T00:00:00Z' },
}

export const operationalSelectionManifestFixture: ManifestJson = {
  built_at: '2026-08-28T16:30:12Z',
  ranked_candidate_count: 35859,
  selectable_candidate_count: 35859,
  selected_count: 1,
  risk_selected_count: 1,
  reserve_selected_count: 0,
  coverage_eligible_selected_count: 1,
}

export const planSummaryFixture: PlanSummaryOut = {
  planning_date: '2026-08-28',
  selected_inspection_workload: 1,
  location_available_count: 1,
  location_unavailable_count: 0,
  work_block_count: 1,
  decisions_recorded: 0,
  approval_status: 'draft',
}
