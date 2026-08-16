"""Eligibility rules and target construction.

Pure functions over already-joined rows; no I/O. The target is:

    For each establishment-date on which at least one routine canvass occurred,
    did that day's canvassing find at least one Priority or Priority Foundation
    violation?

Four things this module deliberately does **not** do:

* It does not re-derive establishment identity. That is Component 2's contract.
* It does not read ``results`` to decide the *label*. The label comes from the
  violation text alone; using the result would make the target partly circular,
  since the result is itself a summary of what was found. ``results`` is read
  only to decide whether a row is *labellable at all* (findings §10).
* It does not compute any historical quantity. Chronology is used only to group
  same-day canvasses; counts, gaps and prior outcomes belong to Component 4.
* It does not touch ``inspection_date`` beyond eligibility and grouping — no
  windowing, no lookahead to a later inspection.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

from sentinel.target.models import (
    ADOPTION_PHASE_END,
    CANVASS_TYPE,
    CLEAN_PASS_RESULT,
    CODE_ERA_START,
    INSPECTED_RESULTS,
    CodeEraPhase,
    InspectionOutcome,
    TargetRow,
    TargetStatus,
)
from sentinel.target.violations import (
    first_priority_evidence,
    has_priority_violation,
    parse_violations,
)

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def normalize_inspection_type(raw: str | None) -> str:
    """Uppercase and collapse whitespace, so casing variants cannot split a type.

    Findings §4: in the code era this is identical to the literal ``'Canvass'``
    comparison, but the column contains ``CANVASS`` and ``CANVAS`` elsewhere, so
    normalizing costs nothing and removes a class of silent failure.
    """
    if raw is None:
        return ""
    return _WHITESPACE.sub(" ", raw.strip().upper())


def is_canvass(raw: str | None) -> bool:
    """Whether an inspection type is a routine canvass.

    Excludes ``Canvass Re-Inspection`` (16,998 code-era rows). A re-inspection
    exists only *because* an earlier inspection failed, so including it would
    condition the target on past failure and inflate the base rate.
    """
    return normalize_inspection_type(raw) == CANVASS_TYPE


def in_code_era(inspection_date: str | None) -> bool:
    """Whether the date falls on or after the 2018-07-01 food-code cutover.

    Dates are ISO-8601 strings (``2022-03-14T00:00:00.000``), so a lexicographic
    comparison against a ``YYYY-MM-DD`` prefix is correct and avoids parsing.
    """
    if not inspection_date:
        return False
    return inspection_date[:10] >= CODE_ERA_START


def code_era_phase(inspection_date: str | None) -> CodeEraPhase:
    """Which regulatory phase a date falls in.

    ``adoption`` covers 2018-07-01 to 2018-12-31, where the positive rate is
    87.4% against 38.9% in 2026 (findings §15). Flagged rather than excluded, so
    Component 5 can decide whether to hold it out.
    """
    if not in_code_era(inspection_date):
        return CodeEraPhase.PRE_CODE
    assert inspection_date is not None
    if inspection_date[:10] < ADOPTION_PHASE_END:
        return CodeEraPhase.ADOPTION
    return CodeEraPhase.STABLE


def classify_inspection(
    *,
    inspection_id: str,
    establishment_id: str,
    inspection_date: str,
    inspection_type: str,
    results: str,
    violations: str | None,
) -> InspectionOutcome:
    """Decide one inspection's eligibility and, if eligible, its label.

    Gates are evaluated in a fixed order so that a row's exclusion reason is the
    *first* thing that disqualified it, which keeps the status counts
    interpretable.
    """
    entries = ()
    if not in_code_era(inspection_date):
        status = TargetStatus.INELIGIBLE_ERA
    elif not is_canvass(inspection_type):
        status = TargetStatus.INELIGIBLE_TYPE
    elif results not in INSPECTED_RESULTS:
        # Out of Business, No Entry, Not Ready, Business Not Located: no
        # inspection took place, so there is no outcome to label (findings §9).
        status = TargetStatus.INELIGIBLE_RESULT
    else:
        parsed = parse_violations(violations)
        has_text = bool(violations and violations.strip())
        if not has_text and results != CLEAN_PASS_RESULT:
            # The result asserts violations were found; the text records none.
            # Nothing can be concluded, so it is not guessed (findings §10).
            status = TargetStatus.UNKNOWN_VIOLATIONS
        else:
            positive = has_priority_violation(parsed)
            return InspectionOutcome(
                inspection_id=inspection_id,
                establishment_id=establishment_id,
                inspection_date=inspection_date,
                inspection_type=inspection_type,
                results=results,
                status=TargetStatus.ELIGIBLE,
                label=positive,
                entries=tuple(parsed),
                evidence=first_priority_evidence(parsed) if positive else None,
            )

    return InspectionOutcome(
        inspection_id=inspection_id,
        establishment_id=establishment_id,
        inspection_date=inspection_date,
        inspection_type=inspection_type,
        results=results,
        status=status,
        label=None,
        entries=entries,
    )


def _inspection_sort_key(outcome: InspectionOutcome) -> tuple[int, str]:
    """Order inspections within a day deterministically, numeric ids first."""
    raw = outcome.inspection_id
    return (int(raw), raw) if raw.isdigit() else (0, raw)


def collapse_same_day(outcomes: Sequence[InspectionOutcome]) -> TargetRow:
    """Collapse one establishment-date's eligible canvasses into a single row.

    Findings §11: 582 establishment-dates carry more than one eligible canvass
    and 160 of those disagree on the outcome, so this rule changes real labels.

    ``target`` is the OR over the day's canvasses. The reasoning is the decision
    problem: "inspect establishment E on date D" is one scheduling decision, and
    the risk question — would an inspector visiting that day find a priority
    violation? — is answered yes if any of them did. Emitting two rows would give
    them an identical as-of boundary and possibly opposite labels, injecting
    irreducible noise by construction.

    The representative inspection is the first positive one in id order, or the
    lowest id when none is positive, so provenance points at the evidence.
    """
    if not outcomes:
        raise ValueError("Cannot collapse an empty set of inspections")

    ordered = sorted(outcomes, key=_inspection_sort_key)
    positives = [o for o in ordered if o.label]
    representative = positives[0] if positives else ordered[0]

    target = 1 if positives else 0
    return TargetRow(
        establishment_id=representative.establishment_id,
        inspection_date=representative.inspection_date,
        target_inspection_id=representative.inspection_id,
        inspection_type=representative.inspection_type,
        results=representative.results,
        target=target,
        target_status=TargetStatus.ELIGIBLE,
        has_priority=any(o.n_priority > 0 for o in ordered),
        has_priority_foundation=any(o.n_priority_foundation > 0 for o in ordered),
        n_priority_entries=sum(o.n_priority for o in ordered),
        n_priority_foundation_entries=sum(o.n_priority_foundation for o in ordered),
        n_violation_entries=sum(len(o.entries) for o in ordered),
        evidence=representative.evidence,
        n_contributing_inspections=len(ordered),
        contributing_inspection_ids=" ".join(o.inspection_id for o in ordered),
        code_era_phase=code_era_phase(representative.inspection_date),
    )


def _excluded_row(outcome: InspectionOutcome) -> TargetRow:
    """Represent an excluded inspection, so the exclusion stays countable."""
    return TargetRow(
        establishment_id=outcome.establishment_id,
        inspection_date=outcome.inspection_date,
        target_inspection_id=outcome.inspection_id,
        inspection_type=outcome.inspection_type,
        results=outcome.results,
        target=None,
        target_status=outcome.status,
        has_priority=False,
        has_priority_foundation=False,
        n_priority_entries=0,
        n_priority_foundation_entries=0,
        n_violation_entries=0,
        evidence=None,
        n_contributing_inspections=1,
        contributing_inspection_ids=outcome.inspection_id,
        code_era_phase=code_era_phase(outcome.inspection_date),
    )


def build_target_rows(
    outcomes: Iterable[InspectionOutcome],
    *,
    include_excluded: bool = True,
) -> list[TargetRow]:
    """Turn classified inspections into target rows.

    Eligible inspections are grouped by (establishment, date) and collapsed.
    Excluded inspections are emitted one-to-one when ``include_excluded`` is set,
    so every raw inspection is accounted for and no exclusion is silent.

    Rows are returned sorted by (establishment_id, inspection_date), so the
    output does not depend on input order.
    """
    eligible: dict[tuple[str, str], list[InspectionOutcome]] = {}
    excluded: list[InspectionOutcome] = []

    for outcome in outcomes:
        if outcome.status is TargetStatus.ELIGIBLE:
            key = (outcome.establishment_id, outcome.inspection_date)
            eligible.setdefault(key, []).append(outcome)
        else:
            excluded.append(outcome)

    rows = [collapse_same_day(group) for group in eligible.values()]
    if include_excluded:
        rows.extend(_excluded_row(o) for o in excluded)

    rows.sort(key=lambda r: (r.establishment_id, r.inspection_date, r.target_inspection_id))
    logger.info(
        "Built %d target rows (%d eligible decision points, %d excluded inspections)",
        len(rows),
        len(eligible),
        len(excluded),
    )
    return rows
