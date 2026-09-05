/**
 * Plain-language translations of Sentinel's internal vocabulary.
 *
 * Every technical code (a decision mechanism, a schedule status, a warning token, a review
 * trigger) still exists in the API response untouched -- this module only supplies a second,
 * human-readable string to show *beside* it. Nothing here invents a claim the underlying
 * contract doesn't make: each mapping was checked against the real value the API returns
 * (`docs/data_contracts/*.md`) before being written, and the raw code is always still shown in
 * a "Technical details" section so nothing is hidden, only re-explained.
 */

/** A one-line label for an establishment, for contexts (page titles, browser tabs) that can't
 * show name and address on separate lines. Falls back to the raw id only when Component 2's
 * entity-resolution artifact hasn't produced a name for this establishment -- see
 * `EstablishmentIdentity` for the two-line table/header form used everywhere else. */
export function establishmentDisplayName(
  name: string | null | undefined,
  establishmentId: string,
): string {
  return name && name.trim().length > 0 ? name : establishmentId
}

export function mechanismLabel(mechanism: string): string {
  switch (mechanism) {
    case 'risk_priority':
      return 'Prioritized by risk'
    case 'coverage_reserve':
      return 'Prioritized to maintain inspection coverage'
    case 'not_selected':
      return 'Not currently prioritized'
    default:
      return mechanism
  }
}

export function decisionReasonLabel(reason: string): string {
  switch (reason) {
    case 'selected_by_risk_rank':
      return 'Ranked highly enough on available risk signals to be recommended'
    case 'selected_by_coverage_reserve':
      return 'Set aside capacity for establishments with little inspection history'
    case 'not_selected_capacity_exhausted':
      return "Ranked below the establishments that fit today's inspection capacity"
    case 'not_selected_reserve_exhausted':
      return 'Would have needed a coverage set-aside that was already used or unavailable'
    default:
      return reason
  }
}

/** `warnings` is a sorted, pipe-joined set of tokens, or the literal string "none". */
export function warningLabels(warnings: string): string[] {
  if (!warnings || warnings === 'none') return []
  return warnings.split('|').map((code) => {
    switch (code) {
      case 'limited_history':
        return 'Little inspection history is available for this establishment'
      case 'no_prior_inspection':
        return 'No inspection has ever been recorded for this establishment'
      case 'insufficient_group_audit_support':
        return "There isn't enough data yet to audit how this recommendation compares across neighborhoods"
      case 'unknown_geography':
        return "This establishment's neighborhood could not be determined"
      default:
        return code
    }
  })
}

export function scheduleStatusLabel(status: string): string {
  switch (status) {
    case 'scheduled':
      return 'Fits in available capacity'
    case 'backlog':
      return 'Waiting for capacity'
    case 'deferred':
      return 'Moved to a later day'
    case 'cancelled':
      return 'Cancelled'
    default:
      return status
  }
}

export function scheduleReasonLabel(reason: string): string {
  switch (reason) {
    case 'placed_in_priority_order':
      return 'Placed in priority order within available capacity'
    case 'capacity_exhausted_in_horizon':
      return 'Available capacity was used up by higher-priority establishments'
    case 'deferred_by_adjustment':
      return 'A supervisor moved this to a later day'
    case 'advanced_by_adjustment':
      return 'A supervisor moved this to an earlier day'
    case 'displaced_by_adjustment':
      return 'A supervisor moved another establishment into this slot'
    case 'rescheduled_by_replan':
      return 'The plan was updated after a reported change'
    case 'cancelled_by_adjustment':
      return 'A supervisor removed this from the plan'
    case 'cancelled_in_field':
      return 'The field reported this inspection as cancelled'
    default:
      return reason
  }
}

export function backlogReasonLabel(reason: string): string {
  switch (reason) {
    case 'capacity_exhausted_in_horizon':
      return "It was recommended, but today's available capacity was used up by higher-priority establishments first"
    case 'displaced_by_adjustment':
      return 'A supervisor moved another establishment ahead of it'
    default:
      return reason
  }
}

/** `trigger_reasons` on a human-review case is a sorted, pipe-joined set of codes. */
export function reviewTriggerLabels(triggerReasons: string): string[] {
  if (!triggerReasons || triggerReasons === 'none') return []
  return triggerReasons.split('|').map((code) => {
    switch (code) {
      case 'policy_warning_present':
        return 'This recommendation includes a warning that should be reviewed by a person'
      case 'no_execution_record_on_scheduled_row':
        return 'This scheduled inspection does not currently have a matching record of what happened'
      default:
        return code
    }
  })
}

/** Two structurally different triggers currently write into the same review queue (Component
 * 16): a judgment flag on the recommendation itself, and a bookkeeping gap in field reporting.
 * They are kept visually distinct here rather than merged into one generic "flagged" reading. */
export type ReviewTriggerCategory = 'decision' | 'record_keeping'

export function reviewTriggerCategory(code: string): ReviewTriggerCategory {
  return code === 'policy_warning_present' ? 'decision' : 'record_keeping'
}

/** What a person should actually do about one trigger, in plain language -- distinct from
 * `reviewTriggerLabels`, which explains *why* a case is here; this answers *what next*. */
export function reviewTriggerActionLabel(code: string): string {
  switch (code) {
    case 'policy_warning_present':
      return 'Read the warning on the recommendation before treating this priority as final.'
    case 'no_execution_record_on_scheduled_row':
      return "Confirm what actually happened with this inspection and log the outcome. This does not mean the recommendation or schedule was wrong."
    default:
      return 'Review this case before acting on it.'
  }
}

export function resolutionActionLabel(action: string): string {
  switch (action) {
    case 'acknowledge':
      return 'Reviewed, no further action needed'
    case 'refer_to_override':
      return 'Referred to a recommendation change'
    case 'refer_to_adjustment':
      return 'Referred to a schedule change'
    case 'escalate':
      return 'Escalated for further attention'
    default:
      return action
  }
}

export function reviewStatusLabel(status: string): string {
  switch (status) {
    case 'flagged':
      return 'Needs review'
    case 'resolved':
      return 'Reviewed'
    default:
      return status
  }
}

/** A calibrated risk score in [0, 1] is shown as a plain percentage-like priority score, never
 * relabelled "High Risk" / "Low Risk" -- the policy contract makes no such categorical claim,
 * only a rank. See docs/data_contracts/policy_decisions.md and ADR 0037.
 *
 * Kept for Technical details only -- see `relativePriorityLabel` for the primary, plain-language
 * framing. A bare "72/100" reads as a calibrated probability or a fixed risk threshold to most
 * readers, and Sentinel's policy makes neither claim: `is_selected` is a function of *this run's*
 * capacity cutoff, not of the score alone, and the model's discriminative power is limited (see
 * `MODEL_LIMITATION_NOTE`). */
export function formatPriorityScore(score: number): string {
  return `${Math.round(score * 100)} / 100`
}

/** Whether an establishment made the cut for *this specific plan* -- deliberately not called
 * "recommended," which reads as a verdict about the establishment. The same establishment, same
 * score, same history, can flip between these two states purely because a capacity number
 * changed; the label says so rather than implying a fixed judgment. */
export function selectionStatusLabel(isSelected: boolean): string {
  return isSelected ? 'Selected for this plan' : 'Not selected for this plan'
}

export function selectionStatusHint(isSelected: boolean): string {
  return isSelected
    ? "Ranked within this plan's capacity cutoff -- not a claim that this establishment is unsafe."
    : "Ranked below this plan's capacity cutoff for this run. A different capacity, or a different plan, could place it above the line without anything about the establishment changing."
}

/** Rank and percentile within the full evaluated population, computed from `model_rank` /
 * `n_universe` -- both fixed *before* any capacity cutoff is applied (Component 13's risk
 * ranking runs over the whole eligible universe; the cutoff is a later step). This is what
 * "relative priority" means in Sentinel: a stable position among everyone evaluated, independent
 * of how much capacity this particular plan happens to have. */
export function relativePriorityLabel(modelRank: number, nUniverse: number): string {
  if (nUniverse <= 0) return `Ranked ${modelRank.toLocaleString()}`
  const percentile = Math.max(1, Math.round((modelRank / nUniverse) * 100))
  return `Ranked ${modelRank.toLocaleString()} of ${nUniverse.toLocaleString()} evaluated establishments (top ${percentile}%)`
}

/** The honest, once-per-page disclosure Part 2 of the actionability pass asks for: concise,
 * not a per-row warning, and using the project's own measured evaluation numbers (Components
 * 6-8's ROC-AUC 0.6163-0.6241 on the quarterly folds; see STATUS.md and
 * docs/analysis/baseline_models_findings.md) rather than a vaguer "limited accuracy" claim. */
export const HOW_TO_USE_PRIORITY =
  "Sentinel ranks establishments using available historical inspection signals. Its ability to " +
  'tell a future violation apart from a future clean inspection is measurably better than chance ' +
  'but modest by ordinary statistical standards (ROC-AUC roughly 0.61-0.62 in this project’s ' +
  'own evaluation). Use this ranking as a starting point alongside inspector knowledge and local ' +
  'context -- not as a verdict about any one establishment.'

export function overrideActionLabel(action: string): string {
  switch (action) {
    case 'force_include':
      return 'Include in this plan, despite the current capacity cutoff'
    case 'force_exclude':
      return 'Remove from this plan'
    default:
      return action
  }
}

export function adjustmentActionLabel(action: string): string {
  switch (action) {
    case 'defer_to_date':
      return 'Move to a later date'
    case 'advance_to_date':
      return 'Move to an earlier date'
    case 'cancel':
      return 'Cancel this planned inspection'
    default:
      return action
  }
}

export function executionStatusLabel(status: string): string {
  switch (status) {
    case 'completed':
      return 'Completed'
    case 'not_performed':
      return 'Not performed'
    case 'cancelled_in_field':
      return 'Cancelled in the field'
    default:
      return status
  }
}

/** What Component 14's `observed_calendar` capacity mode (the default) actually is: the count
 * of inspections Chicago's own historical record shows for that calendar day, replayed as this
 * plan's slot count -- not a live staffing feed. `flat_median` is a labelled scenario and never
 * claims to be an observed fact at all. See `CAPACITY_IS_INHERITED` / `CALENDAR_IS_OBSERVED` /
 * `CAPACITY_MODE_SCENARIO_CLAIM` in `src/sentinel/scheduling/definitions.py`. */
export function capacityHonestyNote(capacityMode: string | undefined): string {
  if (capacityMode === 'flat_median') {
    return "This plan uses a labelled scenario: every day is assigned the same flat median inspection rate for the period, rather than any single day's real capacity. It describes what a smoothed schedule could look like, not an operational fact about any specific day."
  }
  return "This plan's daily capacity is the number of inspections Chicago's own historical record shows for that same calendar day -- not a live staffing count or a confirmed availability number for this period."
}

const QUARTER_MONTHS: Record<string, string> = {
  Q1: 'Jan–Mar',
  Q2: 'Apr–Jun',
  Q3: 'Jul–Sep',
  Q4: 'Oct–Dec',
}

/** "quarterly-2026Q2" -> "Apr-Jun 2026". "covid_shift-2020H2-2021" -> a plain description of
 * the distribution-shift period. Falls back to the raw id for any shape not recognized, so an
 * unexpected fold never renders as blank. */
export function foldLabel(foldId: string | undefined): string {
  if (!foldId) return 'an available period'
  const quarterly = /^quarterly-(\d{4})(Q[1-4])$/.exec(foldId)
  if (quarterly) {
    const [, year, quarter] = quarterly
    return `${QUARTER_MONTHS[quarter]} ${year}`
  }
  if (foldId.startsWith('covid_shift')) {
    return 'the 2020-2021 distribution-shift period'
  }
  return foldId
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return 'Not yet known'
  try {
    return new Date(`${value}T00:00:00`).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return value
  }
}

/** For a full ISO timestamp (a manifest's `built_at`), not a date-only value -- shown so a
 * reader can tell a completed run from a live feed ("generated Aug 26, 2026, 7:58 AM"). */
export function formatDateTime(value: string | null | undefined): string {
  if (!value) return 'an unknown time'
  try {
    return new Date(value).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    })
  } catch {
    return value
  }
}

/** Plain-language sentences built from a curated slice of Component 4's own as-of feature row
 * (see `RiskHistoryFactorsOut`) -- never a new computation, only a re-statement of values the
 * model that produced the recommendation already saw. Each sentence is independently true and
 * omitted individually when its underlying value is null, rather than hiding the whole list. */
export function historyFactorSummary(factors: {
  prior_canvass_count_code_era: number | null
  prior_canvass_priority_count: number | null
  prior_canvass_priority_rate: number | null
  days_since_last_canvass: number | null
  days_since_any_inspection: number | null
  fail_at_last_canvass: boolean | null
  name_changed_since_last_canvass: boolean | null
}): string[] {
  const notes: string[] = []
  const { prior_canvass_count_code_era: canvasses, prior_canvass_priority_count: priorityHits } = factors

  if (canvasses === 0) {
    notes.push('Sentinel has no canvass inspection on record for this establishment since 2018.')
  } else if (canvasses != null && priorityHits != null) {
    notes.push(
      `${priorityHits} of its last ${canvasses} canvass inspection${canvasses === 1 ? '' : 's'} found a Priority or Priority Foundation violation.`,
    )
  }

  if (factors.days_since_last_canvass != null) {
    notes.push(`It has been ${factors.days_since_last_canvass} days since its last canvass inspection.`)
  } else if (factors.days_since_any_inspection != null) {
    notes.push(`It has been ${factors.days_since_any_inspection} days since any inspection of any kind.`)
  }

  if (factors.fail_at_last_canvass) {
    notes.push('Its most recent canvass inspection resulted in a Fail.')
  }

  if (factors.name_changed_since_last_canvass) {
    notes.push('The business name on record has changed since its last canvass inspection.')
  }

  return notes
}

const CHECK_LABELS: Record<string, string> = {
  reserve_is_not_inert: "Confirms the coverage set-aside is actually being used, not a rule that never fires",
  coverage_is_not_free: 'Confirms the coverage set-aside was priced in citations forgone, not assumed to be costless',
  group_representation_is_stable: 'Confirms recommendations are not skewed toward or away from any one neighborhood group in a way the data cannot support',
  a_winner_was_determined: 'Records whether the data alone was enough to pick one policy over the others',
  tables_are_deterministically_sorted: 'Confirms re-running this step produces byte-identical output',
  inputs_were_not_modified: "Confirms this step did not alter any earlier component's data",
}

// --- Operational date (today vs. the latest built plan) --------------------

/** The primary, always-visible statement of which date an operational plan is for -- honest in
 * both directions: never claims a stale plan is "today's" when it isn't, and never hides that
 * the current plan genuinely is today's when it is. `isToday` comes from
 * `lib/today.ts::isPlanningDateToday` -- this function only renders the comparison's result,
 * it never computes it, so there is exactly one place "today" is decided. */
export function planLabelForToday(planningDate: string | undefined | null, isToday: boolean): string {
  if (!planningDate) return 'No operational plan is available yet.'
  return isToday
    ? `Today's inspection plan — ${formatDate(planningDate)}.`
    : `Plan for ${formatDate(planningDate)} — not today's plan yet.`
}

/** Secondary, collapsible context explaining the relationship between a plan's own date and the
 * real current date -- never blocks the primary plan content, per the product principle that a
 * supervisor should see the plan first and the caveat second. */
export function planStalenessNote(planningDate: string, current: string): string {
  if (planningDate === current) {
    return (
      'This plan was built for today. It reflects the most recent Chicago inspection data ' +
      'ingest, not a live feed of new inspections -- see "How Sentinel prioritizes locations" ' +
      'for what that means.'
    )
  }
  return (
    `This is the most recently built operational plan, for ${formatDate(planningDate)}. No ` +
    `plan has been built yet for ${formatDate(current)} -- Sentinel never builds a plan ` +
    'automatically; an operator runs it explicitly for a given date.'
  )
}

// --- Geographic plan / supervisor plan review (Components 20-21) -----------

/** Component 21's `PlanDecisionAction` vocabulary, exactly. */
export function planDecisionActionLabel(action: string): string {
  switch (action) {
    case 'keep_selected':
      return 'Keep as proposed'
    case 'move_to_later_workday':
      return 'Move to a later workday'
    case 'do_not_proceed_as_planned':
      return 'Do not proceed as planned'
    case 'adjust_operational_priority':
      return 'Adjust field-work order'
    default:
      return action
  }
}

export function organizationModeLabel(mode: string): string {
  switch (mode) {
    case 'risk_first':
      return 'Risk-first (Sentinel priority order, unchanged by geography)'
    case 'geography_assisted':
      return 'Geography-assisted (priority balanced with proximity)'
    default:
      return mode
  }
}

/** Plain-language text for a raw API error code -- shown as the primary message in `ErrorState`;
 * the raw code itself is kept, de-emphasized, alongside it rather than dropped, so a support
 * conversation can still reference it. Covers every `error_code` an `ApiError` subclass in
 * `src/sentinel/api/errors.py` defines, plus `unknown_component` from the manifest/runs routes;
 * `ambiguous_scope` is handled separately by `ErrorState` itself, not through this function. */
export function apiErrorCodeLabel(code: string): string {
  switch (code) {
    case 'artifact_not_found':
      return "We couldn't find that information yet. It may not have been generated for this plan."
    case 'row_not_found':
      return "We couldn't find that record."
    case 'validation_refused':
      return "We couldn't save this -- some of the information doesn't meet the requirements."
    case 'duplicate_key':
      return 'This has already been recorded.'
    case 'unknown_component':
      return "That information isn't available."
    default:
      return "Something went wrong and we couldn't complete this."
  }
}

/** "Area 7" (Component 20's raw label) -> "Work Area 7" for display. Passes any label that
 * doesn't match the plain "Area N" shape through unchanged, so an unexpected label format is
 * never mangled -- e.g. the "unmapped" label already reads fine on its own. */
export function workAreaLabel(label: string): string {
  const match = /^Area (\d+)$/.exec(label)
  return match ? `Work Area ${match[1]}` : label
}

/** Falls back to a 1-based position ("Work block 3") instead of a raw `work_block_id` when
 * Component 20 has not produced a label for a block -- rare, but the id itself is never a
 * meaningful thing for a supervisor to read. */
export function workBlockDisplayLabel(label: string | undefined | null, index: number): string {
  return label && label.trim().length > 0 ? label : `Work block ${index + 1}`
}

/** How many establishments Sentinel looked at for a live operational plan vs. how many actually
 * made it in, worded the same honest way `capacityHonestyNote` states the schedule's capacity --
 * a plain fact about this run's numbers, never a claim that an unselected establishment is safe.
 * Only ever called with counts read from `operational_selection`'s own manifest -- never derived
 * or estimated here. */
export function operationalCoverageNote(ranked: number, selectable: number, selected: number): string {
  return (
    `Sentinel found ${ranked.toLocaleString()} establishment${ranked === 1 ? '' : 's'} eligible for ` +
    `this planning date. ${selectable.toLocaleString()} had enough information to be scored, and ` +
    `${selected.toLocaleString()} fit within today's capacity and were selected for this plan. The ` +
    `rest are not flagged as lower risk -- they simply did not fit in this plan.`
  )
}

export function approvalStatusLabel(status: string): string {
  switch (status) {
    case 'draft':
      return 'Draft — not yet reviewed'
    case 'under_supervisor_review':
      return 'Under supervisor review'
    case 'adjusted':
      return 'Every establishment reviewed'
    case 'approved':
      return 'Approved'
    default:
      return status
  }
}

/** Why establishments in a work block are grouped together, derived from fields already on
 * `WorkBlockOut` -- never a new computation, a re-statement of Component 20's own grouping
 * rule (see `organization.build_work_blocks`'s `rationale` string). */
export function workBlockRationale(block: { size: number; is_unmapped: boolean }): string {
  if (block.is_unmapped) {
    return 'Coordinates are unavailable for these establishments, so Sentinel cannot place them into a geographic work block.'
  }
  if (block.size === 1) {
    return 'A single selected establishment with no other selected establishment within the configured geographic proximity threshold.'
  }
  return `These ${block.size} selected establishments are within the configured geographic proximity threshold and can be considered as one field-work area. Proximity is based on straight-line geographic distance; Sentinel does not currently estimate driving time or traffic.`
}

/** Falls back to a title-cased version of the raw snake_case check name so an unmapped check
 * still reads as a sentence fragment rather than raw code -- this list is deliberately not
 * exhaustive (dozens of checks exist); see `docs/data_contracts/*.md` for the authoritative list. */
export function checkLabel(name: string): string {
  if (name in CHECK_LABELS) return CHECK_LABELS[name]
  return name.replace(/_/g, ' ')
}
