/**
 * The real, static (fold_set, fold_id) table -- verified directly against
 * `data/processed/evaluation/evaluation_folds_*.parquet`, not invented.
 *
 * No Sentinel API endpoint enumerates these values (there is no "list distinct scope values"
 * route), so this is the one scope dimension the frontend hardcodes rather than reading from a
 * manifest. Every other scope dropdown (`policy_id`, `model_name`, `k_name`,
 * `schedule_config_id`) is populated live from `GET /v1/manifests/{policy,scheduling}` --
 * see `useManifestOptions`.
 */
export interface FoldOption {
  fold_set: string
  fold_id: string
}

const QUARTERLY_FOLD_IDS = [
  '2022Q2',
  '2022Q3',
  '2022Q4',
  '2023Q1',
  '2023Q2',
  '2023Q3',
  '2023Q4',
  '2024Q1',
  '2024Q2',
  '2024Q3',
  '2024Q4',
  '2025Q1',
  '2025Q2',
  '2025Q3',
  '2025Q4',
  '2026Q1',
  '2026Q2',
]

export const FOLD_TABLE: FoldOption[] = [
  ...QUARTERLY_FOLD_IDS.map((q) => ({ fold_set: 'quarterly', fold_id: `quarterly-${q}` })),
  { fold_set: 'covid_shift', fold_id: 'covid_shift-2020H2-2021' },
]

export const FOLD_SETS: string[] = [...new Set(FOLD_TABLE.map((f) => f.fold_set))]

export function foldIdsForSet(foldSet: string | undefined): string[] {
  if (!foldSet) return []
  return FOLD_TABLE.filter((f) => f.fold_set === foldSet).map((f) => f.fold_id)
}
