"""The human resolution layer. Pure -- no filesystem, no clock.

Mirrors Component 13's ``governance.py`` and Component 14's ``adjustments.py``/``execution.py``:
an all-or-nothing parser at the boundary, and an apply step that never edits the queue it is
resolving cases in -- it only decides each resolution's outcome and the case's final status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sentinel.review.definitions import (
    POINTER_FIELD_FOR_ACTION,
    REVIEW_REQUIRED_FIELDS,
    ReviewCaseStatus,
    ReviewResolutionAction,
)
from sentinel.review.models import ResolutionOutcome, ReviewCase, ReviewResolution

#: Outcomes a resolution can have. Mirrors ``REVIEW_RESOLUTION_OUTCOMES`` in ``definitions.py``.
OUTCOME_APPLIED = "applied"
OUTCOME_NO_OP_ALREADY_RESOLVED = "no_op_already_resolved"
OUTCOME_CASE_NOT_IN_QUEUE = "case_not_in_queue"


class ReviewGovernanceError(ValueError):
    """Raised when a review resolution cannot be trusted enough to apply."""


def parse_resolutions(records: Sequence[Mapping[str, object]]) -> list[ReviewResolution]:
    """Validate a decoded resolution file at the boundary, or refuse the whole run.

    Every required field must be present and non-blank, the action must be a known
    ``ReviewResolutionAction``, and the pointer field ``POINTER_FIELD_FOR_ACTION`` names for that
    action must be present and non-blank while the other pointer field is blank. Refusing the
    whole file rather than skipping bad rows is deliberate, matching Component 13's and Component
    14's own parsers: a partially applied resolution file produces a review state nobody
    authorised.
    """
    actions = {a.value for a in ReviewResolutionAction}
    resolutions: list[ReviewResolution] = []
    for position, raw in enumerate(records):
        try:
            resolution = ReviewResolution.model_validate(dict(raw))
        except Exception as exc:  # pydantic raises its own error type
            raise ReviewGovernanceError(f"resolution {position}: {exc}") from exc

        blank = [
            name
            for name in REVIEW_REQUIRED_FIELDS
            if not str(getattr(resolution, name, "")).strip()
        ]
        if blank:
            raise ReviewGovernanceError(
                f"resolution {resolution.review_id or position}: {', '.join(sorted(blank))} is "
                "blank. Every required review field is required"
            )
        if resolution.resolution_action not in actions:
            raise ReviewGovernanceError(
                f"resolution {resolution.review_id}: unknown resolution_action "
                f"{resolution.resolution_action!r}; known: {', '.join(sorted(actions))}"
            )

        required_pointer = POINTER_FIELD_FOR_ACTION[resolution.resolution_action]
        for pointer_field in ("referenced_override_id", "referenced_adjustment_id"):
            value = getattr(resolution, pointer_field)
            present = bool(value and value.strip())
            if pointer_field == required_pointer and not present:
                raise ReviewGovernanceError(
                    f"resolution {resolution.review_id}: {resolution.resolution_action} "
                    f"requires a non-blank {pointer_field}"
                )
            if pointer_field != required_pointer and present:
                raise ReviewGovernanceError(
                    f"resolution {resolution.review_id}: {resolution.resolution_action} must "
                    f"not carry {pointer_field}"
                )
        resolutions.append(resolution)

    ids = [r.review_id for r in resolutions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ReviewGovernanceError(
            f"duplicate review_id: {', '.join(duplicates)}. The id is how a decision is "
            "referred to afterwards, so two decisions cannot share one"
        )
    return resolutions


def apply_resolutions(
    cases: Sequence[ReviewCase], resolutions: Sequence[ReviewResolution]
) -> tuple[dict[str, ResolutionOutcome], dict[tuple[str, str, str, str], str]]:
    """Apply every resolution addressed to this queue, in ``review_id`` order.

    Applied in id order rather than file order, so re-serialising the file cannot change which
    resolution "wins" a case two resolutions both address -- matching ADR 0047's rule for
    Component 13's overrides and Component 14's adjustments.

    Returns a map from a resolution's own scope key to its ``ResolutionOutcome``, and a map from
    every case's scope key to its final status. Both are computed here rather than mutating
    ``cases`` in place: a review case is a fact about the current queue, and the resolution log
    is a fact about what a human decided, and the two are kept as separate return values for the
    same reason they are separate tables.
    """
    case_by_key = {
        (case.policy_id, case.fold_id, case.k_name, case.target_inspection_id): case
        for case in cases
    }
    final_status: dict[tuple[str, str, str, str], str] = {
        key: ReviewCaseStatus.FLAGGED for key in case_by_key
    }
    outcomes: dict[str, ResolutionOutcome] = {}

    for resolution in sorted(resolutions, key=lambda r: r.review_id):
        key = (
            resolution.policy_id,
            resolution.fold_id,
            resolution.k_name,
            resolution.target_inspection_id,
        )
        if key not in case_by_key:
            outcomes[resolution.review_id] = ResolutionOutcome(
                resolution=resolution,
                outcome=OUTCOME_CASE_NOT_IN_QUEUE,
            )
            continue
        original_status = final_status[key]
        if original_status == ReviewCaseStatus.RESOLVED:
            outcomes[resolution.review_id] = ResolutionOutcome(
                resolution=resolution,
                outcome=OUTCOME_NO_OP_ALREADY_RESOLVED,
                original_status=original_status,
                final_status=original_status,
            )
            continue
        final_status[key] = ReviewCaseStatus.RESOLVED
        outcomes[resolution.review_id] = ResolutionOutcome(
            resolution=resolution,
            outcome=OUTCOME_APPLIED,
            original_status=original_status,
            final_status=ReviewCaseStatus.RESOLVED,
        )

    return outcomes, final_status


__all__ = [
    "OUTCOME_APPLIED",
    "OUTCOME_CASE_NOT_IN_QUEUE",
    "OUTCOME_NO_OP_ALREADY_RESOLVED",
    "ReviewGovernanceError",
    "apply_resolutions",
    "parse_resolutions",
]
