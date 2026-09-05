import { apiFetch } from './client'
import type { PageParams } from './recommendations'
import type {
  BacklogRowOut,
  DecisionScope,
  Page,
  ReplanningRunOut,
  ScheduleRowOut,
} from './types'

export interface ScheduleFilters {
  establishment_id?: string
  schedule_status?: string
  scheduled_date?: string
}

export interface ScheduleDateOut {
  scheduled_date: string
  n_establishments: number
}

export function listSchedule(
  scope: DecisionScope,
  filters: ScheduleFilters,
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<ScheduleRowOut>> {
  return apiFetch('/v1/schedule', { ...scope, ...filters, ...page }, signal)
}

/** Every date this plan actually has inspections on, each with a count -- the real,
 * establishment-backed set a day picker should offer, never a generic calendar the user could
 * pick an empty day from. */
export function listScheduleDates(
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<ScheduleDateOut[]> {
  return apiFetch('/v1/schedule/dates', { ...scope }, signal)
}

export function listBacklog(
  scope: DecisionScope,
  page: PageParams,
  signal?: AbortSignal,
): Promise<Page<BacklogRowOut>> {
  return apiFetch('/v1/schedule/backlog', { ...scope, ...page }, signal)
}

export function getScheduleSummary(
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<Record<string, unknown>[]> {
  return apiFetch('/v1/schedule/summary', { ...scope }, signal)
}

export function getCapacityUtilization(
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<Record<string, unknown>[]> {
  return apiFetch('/v1/schedule/capacity-utilization', { ...scope }, signal)
}

export function getPriorityPreservation(
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<Record<string, unknown>[]> {
  return apiFetch('/v1/schedule/priority-preservation', { ...scope }, signal)
}

export function listReplanningRuns(
  scope: DecisionScope,
  signal?: AbortSignal,
): Promise<ReplanningRunOut[]> {
  return apiFetch('/v1/schedule/replanning-runs', { ...scope }, signal)
}
