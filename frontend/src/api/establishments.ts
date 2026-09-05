import { apiFetch } from './client'
import type { DecisionScope, EstablishmentHistoryOut } from './types'

export function getEstablishment(
  establishmentId: string,
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<EstablishmentHistoryOut> {
  return apiFetch(`/v1/establishments/${encodeURIComponent(establishmentId)}`, { ...scope }, signal)
}
