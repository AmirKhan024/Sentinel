import { apiFetch } from './client'
import type { ManifestJson, RunListEntry } from './types'

export type ManifestComponent = 'policy' | 'scheduling' | 'explanations' | 'review' | 'operational_selection'

export function getHealthz(signal?: AbortSignal): Promise<{ status: string }> {
  return apiFetch('/healthz', {}, signal)
}

export function getManifest(component: ManifestComponent, signal?: AbortSignal): Promise<ManifestJson> {
  return apiFetch(`/v1/manifests/${component}`, {}, signal)
}

export function listRuns(component: string | undefined, signal?: AbortSignal): Promise<RunListEntry[]> {
  return apiFetch('/v1/runs', { component }, signal)
}
