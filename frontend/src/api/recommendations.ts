import { apiFetch } from './client'
import type { AllocationOut, DecisionScope, Page, RecommendationOut } from './types'

export interface RecommendationFilters {
  establishment_id?: string
  is_selected?: boolean
}

export interface PageParams {
  offset: number
  limit: number
  descending: boolean
}

export function listRecommendations(
  scope: DecisionScope,
  filters: RecommendationFilters,
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<RecommendationOut>> {
  return apiFetch('/v1/recommendations', { ...scope, ...filters, ...page }, signal)
}

export function getRecommendation(
  targetInspectionId: string,
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<RecommendationOut> {
  return apiFetch(`/v1/recommendations/${encodeURIComponent(targetInspectionId)}`, { ...scope }, signal)
}

export function getSelectionAllocation(
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<AllocationOut[]> {
  return apiFetch('/v1/policy/selection-allocation', { ...scope }, signal)
}
