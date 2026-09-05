import { apiFetch } from './client'
import type { DecisionScope, ExplanationCaseOut, SupportOut } from './types'

export function listExplanationSupport(signal?: AbortSignal): Promise<SupportOut[]> {
  return apiFetch('/v1/explanations/support', {}, signal)
}

export function getExplanation(
  targetInspectionId: string,
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<ExplanationCaseOut> {
  return apiFetch(`/v1/explanations/${encodeURIComponent(targetInspectionId)}`, { ...scope }, signal)
}
