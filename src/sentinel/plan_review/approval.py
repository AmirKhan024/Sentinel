"""Plan approval: an explicit, auditable supervisor act. Pure -- no filesystem, no clock.

Approval is deliberately not a side effect of decision completeness. A plan with every row
decided is not automatically approved -- that would make approval a consequence of data
entry rather than a supervisor's own act. It requires a separate, minimal
:class:`~sentinel.plan_review.models.PlanApprovalRequest`, and it is refused outright,
never partially applied, if the plan is not ready.

Approval immutability means the *written artifact* is never rewritten in place -- true here
exactly the way it is true for every other Sentinel artifact, by never opening one to edit
it. It does not mean supervisor decisions become impossible after an approval: a supervisor
who amends the plan afterwards produces a new ``supervisor_plan_review`` snapshot and, if
they choose, a new approval event: the original approved artifact is untouched and remains
a permanent record of exactly what was handed to Component 22 at that moment, named by its
own source checksum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.features.models import ValidationCheck
from sentinel.plan_review.definitions import PLAN_APPROVAL_REQUIRED_FIELDS
from sentinel.plan_review.models import PlanApprovalRequest
from sentinel.plan_review.validate import SEVERITY_ERROR, SEVERITY_WARN


class PlanApprovalGovernanceError(ValueError):
    """Raised when an approval request cannot be trusted enough to apply."""


def parse_approval(record: Mapping[str, object]) -> PlanApprovalRequest:
    """Validate one approval request at the boundary. All-or-nothing, like a decision."""
    try:
        request = PlanApprovalRequest.model_validate(dict(record))
    except Exception as exc:  # pydantic raises its own error type
        raise PlanApprovalGovernanceError(str(exc)) from exc

    blank = [
        name
        for name in PLAN_APPROVAL_REQUIRED_FIELDS
        if not str(getattr(request, name, "")).strip()
    ]
    if blank:
        raise PlanApprovalGovernanceError(
            f"approval {request.approval_id or '?'}: {', '.join(sorted(blank))} is blank. "
            "Every required approval field is required"
        )
    return request


def _check(name: str, passed: bool, severity: str, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=passed, severity=severity, detail=detail)


def check_readiness(review_frame: pl.DataFrame) -> list[ValidationCheck]:
    """The approval-readiness checklist, run before a plan may be approved.

    Every check here is independently re-derived from the plan-review frame itself, not
    trusted from an upstream flag -- the same posture ``plan_review.validate`` already takes
    toward Component 20's output. A failing check blocks approval outright; nothing here
    ever approves "most of" a plan.
    """
    checks: list[ValidationCheck] = []

    if review_frame.is_empty():
        return [
            _check(
                "plan_has_establishments",
                False,
                SEVERITY_ERROR,
                "the plan review has zero rows -- there is nothing to approve",
            )
        ]

    duplicates = review_frame.height - review_frame["target_inspection_id"].n_unique()
    checks.append(
        _check(
            "no_duplicate_establishments",
            duplicates == 0,
            SEVERITY_ERROR,
            f"{duplicates} establishment(s) appear more than once in the plan",
        )
    )

    missing_recommendation = review_frame.filter(
        pl.col("policy_rank").is_null() | pl.col("selection_reason").is_null()
    )
    checks.append(
        _check(
            "every_row_carries_the_machine_recommendation",
            missing_recommendation.is_empty(),
            SEVERITY_ERROR,
            f"{missing_recommendation.height} row(s) are missing Sentinel's own "
            "policy_rank/selection_reason -- the machine recommendation must be present "
            "on every row",
        )
    )

    missing_geo = review_frame.filter(
        pl.col("work_block_id").is_null() | pl.col("location_status").is_null()
    )
    checks.append(
        _check(
            "geographic_provenance_present",
            missing_geo.is_empty(),
            SEVERITY_ERROR,
            f"{missing_geo.height} row(s) are missing Component 20's geographic "
            "organization fields",
        )
    )

    decided = review_frame.filter(pl.col("supervisor_decision_action").is_not_null())
    missing_reason = decided.filter(
        pl.col("supervisor_decision_reason_code").is_null()
        | (pl.col("supervisor_decision_reason_code") == "")
    )
    checks.append(
        _check(
            "every_recorded_decision_has_a_reason",
            missing_reason.is_empty(),
            SEVERITY_ERROR,
            f"{missing_reason.height} recorded decision(s) have no reason_code -- refused "
            "at the parser already, re-checked here",
        )
    )

    undecided = review_frame.height - decided.height
    checks.append(
        _check(
            "undecided_rows_default_to_the_machine_recommendation",
            True,
            SEVERITY_WARN,
            f"{undecided} of {review_frame.height} row(s) carry no supervisor decision and "
            "will proceed exactly as Sentinel proposed",
        )
    )

    return checks


def has_blocking_failures(checks: Sequence[ValidationCheck]) -> bool:
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


def format_readiness_report(checks: Sequence[ValidationCheck]) -> str:
    lines = ["", "Plan approval readiness", "------------------------"]
    for check in checks:
        status = (
            "note" if check.severity == SEVERITY_WARN else ("READY" if check.passed else "BLOCKED")
        )
        lines.append(f"  [{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


__all__ = [
    "PlanApprovalGovernanceError",
    "check_readiness",
    "format_readiness_report",
    "has_blocking_failures",
    "parse_approval",
]
