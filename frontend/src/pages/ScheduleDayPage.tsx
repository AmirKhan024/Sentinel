import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { listSchedule, listScheduleDates, type ScheduleDateOut } from '../api/schedule'
import { listReviewQueue } from '../api/review'
import type { ScheduleRowOut } from '../api/types'
import { useApiQuery } from '../hooks/useApiQuery'
import { useDecisionScope } from '../hooks/useDecisionScope'
import { useDefaultScope } from '../hooks/useDefaultScope'
import { useManifestOptions } from '../hooks/useManifestOptions'
import { scheduleReasonLabel, scheduleStatusLabel } from '../lib/copy'
import { PageShell } from '../components/layout/PageShell'
import { InspectionPlanSelector } from '../components/scope/InspectionPlanSelector'
import { LoadingState } from '../components/common/LoadingState'
import { ErrorState } from '../components/common/ErrorState'
import { EmptyState } from '../components/common/EmptyState'
import { EstablishmentIdentity } from '../components/common/EstablishmentIdentity'
import { TechnicalDetails } from '../components/common/TechnicalDetails'
import { FieldRow } from '../components/common/FieldRow'

const REQUIRED_SCOPE = ['schedule_config_id', 'policy_id', 'fold_set', 'fold_id', 'k_name'] as const

function formatDayHeading(iso: string): string {
  const d = new Date(`${iso}T00:00:00`)
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
}

/** A day picker built only from dates this plan actually has inspections on -- never a generic
 * calendar a user could pick an empty day from. Prev/next steps through that real list; the
 * dropdown is the same list, for jumping further than one day. */
function DaySelector({
  dates,
  selected,
  onSelect,
}: {
  dates: ScheduleDateOut[]
  selected: string | null
  onSelect: (date: string) => void
}) {
  const index = dates.findIndex((d) => d.scheduled_date === selected)

  return (
    <div className="day-selector">
      <button
        type="button"
        className="link-button"
        disabled={index <= 0}
        onClick={() => index > 0 && onSelect(dates[index - 1].scheduled_date)}
      >
        ← Previous day
      </button>
      <select
        className="day-selector-current"
        value={selected ?? ''}
        onChange={(e) => onSelect(e.target.value)}
      >
        {dates.map((d) => (
          <option key={d.scheduled_date} value={d.scheduled_date}>
            {formatDayHeading(d.scheduled_date)} ({d.n_establishments})
          </option>
        ))}
      </select>
      <button
        type="button"
        className="link-button"
        disabled={index === -1 || index >= dates.length - 1}
        onClick={() => index < dates.length - 1 && onSelect(dates[index + 1].scheduled_date)}
      >
        Next day →
      </button>
    </div>
  )
}

/**
 * A single simulated day within a historical evaluation fold (Component 14's backtest
 * schedule) -- for backtest analysis, never a substitute for the live operational "Today" page
 * at `/`. `scheduled_date` here comes from a fold's own simulated calendar (e.g. a quarter's
 * test window) and can be months or years in the past; it is never today's real date. See
 * `pages/TodayPage.tsx` for the genuinely live, `planning_date`-scoped experience.
 */
export function ScheduleDayPage() {
  const { scope, setScopeField, setScopeFields, missingFields } = useDecisionScope()
  const { manifests } = useManifestOptions(['policy', 'scheduling', 'review'])
  useDefaultScope(scope, setScopeFields, manifests)
  const navigate = useNavigate()

  const [selectedDate, setSelectedDate] = useState<string | null>(null)

  const missing = missingFields([...REQUIRED_SCOPE])
  const enabled = missing.length === 0

  const datesQuery = useApiQuery(
    (signal) => listScheduleDates(scope, signal),
    [JSON.stringify(scope)],
    enabled,
  )

  useEffect(() => {
    if (datesQuery.status !== 'success') return
    if (selectedDate && datesQuery.data.some((d) => d.scheduled_date === selectedDate)) return
    setSelectedDate(datesQuery.data[0]?.scheduled_date ?? null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datesQuery.status, datesQuery.status === 'success' ? datesQuery.data : null])

  const dayQuery = useApiQuery(
    (signal) =>
      listSchedule(
        scope,
        { scheduled_date: selectedDate ?? undefined },
        { offset: 0, limit: 200, descending: false },
        signal,
      ),
    [JSON.stringify(scope), selectedDate],
    enabled && Boolean(selectedDate),
  )
  const dayRows =
    dayQuery.status === 'success'
      ? [...dayQuery.data.data].sort((a, b) => (a.slot_index ?? 0) - (b.slot_index ?? 0))
      : []

  const decisionReviewQuery = useApiQuery(
    (signal) =>
      listReviewQueue(scope, { offset: 0, limit: 1, descending: false }, signal, {
        trigger: 'policy_warning_present',
      }),
    [JSON.stringify(scope)],
    enabled,
  )
  const missingOutcomesQuery = useApiQuery(
    (signal) =>
      listReviewQueue(scope, { offset: 0, limit: 1, descending: false }, signal, {
        trigger: 'no_execution_record_on_scheduled_row',
      }),
    [JSON.stringify(scope)],
    enabled,
  )
  const attentionCount =
    (decisionReviewQuery.status === 'success' ? decisionReviewQuery.data.page.total : 0) +
    (missingOutcomesQuery.status === 'success' ? missingOutcomesQuery.data.page.total : 0)

  function goToEstablishment(row: ScheduleRowOut) {
    const params = new URLSearchParams({
      policy_id: scope.policy_id ?? '',
      fold_set: scope.fold_set ?? '',
      fold_id: scope.fold_id ?? '',
      k_name: scope.k_name ?? '',
      schedule_config_id: scope.schedule_config_id ?? '',
    })
    navigate(`/establishments/${encodeURIComponent(row.establishment_id)}?${params.toString()}`)
  }

  return (
    <PageShell
      title="Historical Day View"
      description="A single simulated day within a historical evaluation fold -- for backtest analysis, not today's real operations. See Today for the live operational plan."
    >
      <InspectionPlanSelector
        scope={scope}
        setScopeField={setScopeField}
        requiredFields={[...REQUIRED_SCOPE]}
        manifests={manifests}
      />

      {!enabled && <LoadingState label="Preparing an inspection plan…" />}

      {enabled && datesQuery.status === 'loading' && <LoadingState />}
      {enabled && datesQuery.status === 'error' && <ErrorState error={datesQuery.error} />}
      {enabled && datesQuery.status === 'success' && datesQuery.data.length === 0 && (
        <EmptyState message="This plan has no scheduled inspection days yet." />
      )}

      {enabled && datesQuery.status === 'success' && datesQuery.data.length > 0 && selectedDate && (
        <>
          <DaySelector dates={datesQuery.data} selected={selectedDate} onSelect={setSelectedDate} />

          <h2 className="today-heading">{formatDayHeading(selectedDate)}</h2>

          {dayQuery.status === 'loading' && <LoadingState />}
          {dayQuery.status === 'error' && <ErrorState error={dayQuery.error} />}
          {dayQuery.status === 'success' && dayRows.length === 0 && (
            <EmptyState message="Nothing scheduled for this day." />
          )}
          {dayQuery.status === 'success' && dayRows.length > 0 && (
            <ol className="today-list">
              {dayRows.map((row) => (
                <li
                  key={row.target_inspection_id}
                  className="today-row"
                  onClick={() => goToEstablishment(row)}
                >
                  <span className="today-row-slot">
                    {row.slot_index !== null ? `#${row.slot_index}` : '—'}
                  </span>
                  <span className="today-row-identity">
                    <EstablishmentIdentity
                      name={row.establishment_name}
                      address={row.establishment_address}
                      establishmentId={row.establishment_id}
                    />
                  </span>
                  <span
                    className={row.schedule_status === 'scheduled' ? 'chip chip-positive' : 'chip chip-neutral'}
                  >
                    {scheduleStatusLabel(row.schedule_status)}
                  </span>
                  <span className="today-row-why">{scheduleReasonLabel(row.schedule_reason)}</span>
                </li>
              ))}
            </ol>
          )}
        </>
      )}

      {enabled && attentionCount > 0 && (
        <p className="today-attention-note">
          <Link to={`/review?${new URLSearchParams(scope as Record<string, string>).toString()}`}>
            {attentionCount} item{attentionCount === 1 ? '' : 's'} need attention →
          </Link>
        </p>
      )}

      {enabled && selectedDate && (
        <TechnicalDetails summary="Technical details">
          <FieldRow label="schedule_config_id" value={scope.schedule_config_id} />
          <FieldRow label="policy_id" value={scope.policy_id} />
          <FieldRow label="fold_id" value={scope.fold_id} />
          <FieldRow label="k_name" value={scope.k_name} />
          <FieldRow label="scheduled_date" value={selectedDate} />
        </TechnicalDetails>
      )}
    </PageShell>
  )
}
