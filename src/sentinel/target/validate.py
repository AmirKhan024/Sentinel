"""Post-construction checks over the target table.

Same two severities as Component 2, for the same reason. ``error`` checks assert
things that cannot be true of a correct build and fail the command; ``warn``
checks report distributions that are informative but legitimate — the positive
rate ranges from 87% to 39% across years, so a check that failed on drift would
fail on correct data.

The most important checks are the ones that would catch a *silently wrong*
label: an eligible row outside the code era, an eligible row that is not a
canvass, a positive with no evidence, or a labelled row whose status says it
should not have a label.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

from sentinel.target.construct import normalize_inspection_type
from sentinel.target.models import (
    CANVASS_TYPE,
    CODE_ERA_START,
    INSPECTED_RESULTS,
    TARGET_DEFINITION_VERSION,
    TargetRow,
    TargetStatus,
    ValidationCheck,
)

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

MAX_OFFENDERS = 20


def _check(
    name: str,
    passed: bool,
    severity: str,
    detail: str,
    offenders: Sequence[str] = (),
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail,
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def validate_targets(
    rows: Sequence[TargetRow],
    *,
    known_establishment_ids: frozenset[str],
    known_inspection_ids: frozenset[str],
    source_row_count: int,
) -> list[ValidationCheck]:
    """Run every structural and distributional check over a completed build."""
    checks: list[ValidationCheck] = []
    eligible = [r for r in rows if r.target_status is TargetStatus.ELIGIBLE]

    # -- structural ---------------------------------------------------------
    unknown_est = [
        r.establishment_id for r in rows if r.establishment_id not in known_establishment_ids
    ]
    checks.append(
        _check(
            "establishment_id_from_component_2",
            not unknown_est,
            SEVERITY_ERROR,
            f"{len(unknown_est)} rows reference an establishment_id absent from the "
            "Component 2 assignments",
            unknown_est,
        )
    )

    unknown_insp = [
        r.target_inspection_id for r in rows if r.target_inspection_id not in known_inspection_ids
    ]
    checks.append(
        _check(
            "target_inspection_exists",
            not unknown_insp,
            SEVERITY_ERROR,
            f"{len(unknown_insp)} rows reference an inspection_id absent from the raw snapshot",
            unknown_insp,
        )
    )

    key_counts = Counter((r.establishment_id, r.inspection_date) for r in eligible)
    duplicate_keys = [f"{e}@{d}" for (e, d), n in key_counts.items() if n > 1]
    checks.append(
        _check(
            "one_eligible_row_per_establishment_date",
            not duplicate_keys,
            SEVERITY_ERROR,
            f"{len(duplicate_keys)} establishment-dates have more than one eligible target row",
            duplicate_keys,
        )
    )

    accounted = sum(r.n_contributing_inspections for r in rows)
    checks.append(
        _check(
            "every_inspection_accounted_for",
            accounted == source_row_count,
            SEVERITY_ERROR,
            f"{accounted} inspections represented across target rows, "
            f"{source_row_count} in the source",
        )
    )

    mislabelled = [
        r.target_inspection_id
        for r in rows
        if (r.target is None) is (r.target_status is TargetStatus.ELIGIBLE)
    ]
    checks.append(
        _check(
            "label_present_exactly_when_eligible",
            not mislabelled,
            SEVERITY_ERROR,
            f"{len(mislabelled)} rows have a label without being eligible, or vice versa",
            mislabelled,
        )
    )

    bad_values = [
        r.target_inspection_id for r in rows if r.target is not None and r.target not in (0, 1)
    ]
    checks.append(
        _check(
            "target_is_binary",
            not bad_values,
            SEVERITY_ERROR,
            f"{len(bad_values)} rows have a target outside {{0, 1}}",
            bad_values,
        )
    )

    # -- eligibility invariants --------------------------------------------
    wrong_era = [
        r.target_inspection_id for r in eligible if r.inspection_date[:10] < CODE_ERA_START
    ]
    checks.append(
        _check(
            "eligible_rows_are_in_the_code_era",
            not wrong_era,
            SEVERITY_ERROR,
            f"{len(wrong_era)} eligible rows fall before {CODE_ERA_START}, where "
            "Priority violations are not defined",
            wrong_era,
        )
    )

    wrong_type = [
        r.target_inspection_id
        for r in eligible
        if normalize_inspection_type(r.inspection_type) != CANVASS_TYPE
    ]
    checks.append(
        _check(
            "eligible_rows_are_canvasses",
            not wrong_type,
            SEVERITY_ERROR,
            f"{len(wrong_type)} eligible rows are not routine canvasses",
            wrong_type,
        )
    )

    wrong_result = [r.target_inspection_id for r in eligible if r.results not in INSPECTED_RESULTS]
    checks.append(
        _check(
            "eligible_rows_describe_a_real_inspection",
            not wrong_result,
            SEVERITY_ERROR,
            f"{len(wrong_result)} eligible rows have a result meaning no inspection occurred",
            wrong_result,
        )
    )

    unevidenced = [
        r.target_inspection_id for r in eligible if r.target == 1 and not (r.evidence or "").strip()
    ]
    checks.append(
        _check(
            "positives_carry_evidence",
            not unevidenced,
            SEVERITY_ERROR,
            f"{len(unevidenced)} positive rows carry no evidence span, so their label "
            "cannot be traced to the violation text",
            unevidenced,
        )
    )

    inconsistent = [
        r.target_inspection_id
        for r in eligible
        if (r.target == 1) != (r.has_priority or r.has_priority_foundation)
    ]
    checks.append(
        _check(
            "label_matches_priority_flags",
            not inconsistent,
            SEVERITY_ERROR,
            f"{len(inconsistent)} rows disagree with their own priority flags",
            inconsistent,
        )
    )

    versions = {r.target_definition_version for r in rows}
    checks.append(
        _check(
            "single_target_definition_version",
            versions <= {TARGET_DEFINITION_VERSION},
            SEVERITY_ERROR,
            f"target definition versions present: {sorted(versions) or ['<none>']}",
        )
    )

    # -- distributional (reported, never fatal) -----------------------------
    positives = sum(1 for r in eligible if r.target == 1)
    rate = 100.0 * positives / len(eligible) if eligible else 0.0
    checks.append(
        _check(
            "positive_rate",
            True,
            SEVERITY_WARN,
            f"{positives} of {len(eligible)} eligible rows are positive ({rate:.1f}%)",
        )
    )

    status_counts = Counter(r.target_status.value for r in rows)
    checks.append(
        _check(
            "status_breakdown",
            True,
            SEVERITY_WARN,
            ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
        )
    )

    by_year: Counter[str] = Counter()
    pos_by_year: Counter[str] = Counter()
    for row in eligible:
        year = row.inspection_date[:4]
        by_year[year] += 1
        if row.target == 1:
            pos_by_year[year] += 1
    drift = [
        f"{y}: {100.0 * pos_by_year[y] / by_year[y]:.1f}% of {by_year[y]}" for y in sorted(by_year)
    ]
    checks.append(
        _check(
            "positive_rate_by_year",
            True,
            SEVERITY_WARN,
            "base rate drifts materially across years; Component 5 must account for it",
            drift,
        )
    )

    collapsed = sum(1 for r in eligible if r.n_contributing_inspections > 1)
    checks.append(
        _check(
            "collapsed_multi_canvass_days",
            True,
            SEVERITY_WARN,
            f"{collapsed} establishment-dates collapsed more than one eligible canvass",
        )
    )

    phases = Counter(r.code_era_phase.value for r in eligible)
    checks.append(
        _check(
            "code_era_phase_breakdown",
            True,
            SEVERITY_WARN,
            ", ".join(f"{k}={v}" for k, v in sorted(phases.items())),
        )
    )

    return checks


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """Render the checks as a plain text block for the CLI."""
    lines = ["", "Target validation report", "------------------------"]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
        if check.offenders and not (check.passed and check.severity == SEVERITY_ERROR):
            for offender in check.offenders:
                lines.append(f"           - {offender}")
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """Whether any error-severity check failed."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)
