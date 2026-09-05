"""Post-review checks. Every one independently re-derived, not trusted.

The first check in this module is the most important one in Component 21: the set of
establishments in Component 20's plan must be byte-identical before and after plan review.
Everything else here is secondary to that invariant.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.features.models import ValidationCheck
from sentinel.geographic_organization.validate import IMMUTABLE_FIELDS as GEO_IMMUTABLE_FIELDS

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

#: Every field Component 21 must carry through unchanged from Component 20 -- Component 20's
#: own immutable risk/policy fields, plus the geographic fields Component 21 must also never
#: rewrite (it only reads them to render the review screen).
IMMUTABLE_FIELDS: tuple[str, ...] = (
    *GEO_IMMUTABLE_FIELDS,
    "geographic_group_id",
    "work_block_id",
)


def _check(name: str, passed: bool, severity: str, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=passed, severity=severity, detail=detail)


def check_plan_rows_unchanged(
    plan_frame: pl.DataFrame, review_frame: pl.DataFrame
) -> ValidationCheck:
    """The one invariant this component exists to protect: plan review adds no row, drops
    none, and substitutes no establishment for another."""
    plan_ids = set(plan_frame["target_inspection_id"].to_list())
    review_ids = set(review_frame["target_inspection_id"].to_list())
    passed = plan_ids == review_ids
    return _check(
        "plan_rows_unchanged_by_review",
        passed,
        SEVERITY_ERROR,
        "Component 20's and Component 21's establishment sets are identical"
        if passed
        else f"{len(plan_ids - review_ids)} establishment(s) missing from the plan review, "
        f"{len(review_ids - plan_ids)} extra -- plan review altered the plan, which must "
        "never happen",
    )


def check_immutable_fields_unchanged(
    plan_frame: pl.DataFrame, review_frame: pl.DataFrame
) -> ValidationCheck:
    """Component 21 may add decision columns; it may never rewrite a risk, policy, or
    geographic field."""
    left = plan_frame.select(["target_inspection_id", *IMMUTABLE_FIELDS]).sort(
        "target_inspection_id"
    )
    right = review_frame.select(["target_inspection_id", *IMMUTABLE_FIELDS]).sort(
        "target_inspection_id"
    )
    passed = left.equals(right)
    return _check(
        "immutable_fields_unchanged",
        passed,
        SEVERITY_ERROR,
        f"{', '.join(IMMUTABLE_FIELDS)} are byte-identical to Component 20's output"
        if passed
        else "Component 21 altered a risk, policy, or geographic field -- this must never happen",
    )


def check_original_recommendation_never_overwritten(review_frame: pl.DataFrame) -> ValidationCheck:
    """A decided row must still carry Sentinel's own recommendation fields, unedited, beside
    the supervisor's decision -- never replaced by it."""
    decided = review_frame.filter(pl.col("supervisor_decision_action").is_not_null())
    if decided.is_empty():
        return _check(
            "original_recommendation_never_overwritten",
            True,
            SEVERITY_WARN,
            "no decided rows in this plan review",
        )
    missing_recommendation = decided.filter(
        pl.col("policy_rank").is_null() | pl.col("selection_reason").is_null()
    )
    passed = missing_recommendation.is_empty()
    return _check(
        "original_recommendation_never_overwritten",
        passed,
        SEVERITY_ERROR,
        "every decided row still carries Sentinel's own policy_rank/selection_reason"
        if passed
        else f"{missing_recommendation.height} decided row(s) are missing Sentinel's own "
        "recommendation fields -- a supervisor decision must never overwrite them",
    )


def check_no_duplicate_decision_per_establishment(review_frame: pl.DataFrame) -> ValidationCheck:
    duplicates = review_frame.height - review_frame["target_inspection_id"].n_unique()
    return _check(
        "no_duplicate_decision_per_establishment",
        duplicates == 0,
        SEVERITY_ERROR,
        f"{duplicates} establishment(s) appear more than once in the plan review",
    )


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


def format_report(checks: Sequence[ValidationCheck]) -> str:
    lines = ["", "Plan review validation report", "------------------------------"]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def run_all_checks(plan_frame: pl.DataFrame, review_frame: pl.DataFrame) -> list[ValidationCheck]:
    return [
        check_plan_rows_unchanged(plan_frame, review_frame),
        check_immutable_fields_unchanged(plan_frame, review_frame),
        check_original_recommendation_never_overwritten(review_frame),
        check_no_duplicate_decision_per_establishment(review_frame),
    ]


__all__ = [
    "IMMUTABLE_FIELDS",
    "check_immutable_fields_unchanged",
    "check_no_duplicate_decision_per_establishment",
    "check_original_recommendation_never_overwritten",
    "check_plan_rows_unchanged",
    "format_report",
    "has_failures",
    "run_all_checks",
]
