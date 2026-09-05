import { apiFetch } from './client'
import type { StagedRequestStatus } from './types'

export function listStagedRequests(
  filters: { kind?: string; status?: string },
  signal?: AbortSignal,
): Promise<StagedRequestStatus[]> {
  return apiFetch('/v1/staged-requests', { ...filters }, signal)
}
