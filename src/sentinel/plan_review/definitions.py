"""Frozen contracts for Component 21. Every plan-review constant lives here and nowhere else.

**The separation this module exists to hold.** Component 19 decides *who* is selected for
inspection attention. Component 20 decides *how* that already-selected workload can be
geographically organized. Neither ever asks a human supervisor to look at the proposed
workload and decide whether to proceed with it as planned. This component is that step: a
supervisor-facing summary of Component 20's plan, plus an optional, auditable log of what a
supervisor decided about it. It never re-ranks a row, never re-groups it, and never edits a
score, a rank, or a geographic assignment.

**This is a fifth human-decision layer, and it must never collide with the other four.**
Component 13 owns "who is in the queue" (``OverrideAction``). Component 14 owns "when an
approved row is worked" (``AdjustmentAction``) and "what happened"
(``ExecutionStatus``). Component 16 owns "a human looked at a flagged case"
(``ReviewResolutionAction``). This component owns "a supervisor decided what to do with an
already-organized proposed workload" -- a distinct fact from all four, on distinct keys
(``planning_date`` + ``target_inspection_id``, not a backtest cell). ``_guard_registry`` below
checks this component's verbs against the other three and additionally refuses the literal
substring ``"defer"``, matching the discipline Component 16 already established.

**Why this is not Component 16.** Component 16's triggers read Component 13 (historical
policy recommendations) and Component 14 (scheduling) -- an entirely different pipeline from
Component 18/19/20's live, planning-date-scoped operational plan. Extending Component 16's
keys (``policy_id/fold_id/k_name``) to also mean "a live planning run" would blur a backtest
cell with a real plan. This component is new, small, and deliberately does not reimplement a
trigger engine: every establishment in Component 20's plan is automatically in scope for
supervisor review -- a supervisor reviews the whole proposed workload, not a second flagged
subset.
"""

from __future__ import annotations

from enum import StrEnum

from sentinel.policy.definitions import OverrideAction
from sentinel.review.definitions import ReviewResolutionAction
from sentinel.scheduling.definitions import AdjustmentAction

#: Bumped whenever the decision vocabulary, the required-fields list, or the output schema
#: changes in a way that makes two runs incomparable.
PLAN_REVIEW_DEFINITION_VERSION = "v1"


class PlanReviewDefinitionError(ValueError):
    """Raised when the frozen plan-review contracts contradict each other."""


# --- 1. what this component is not ------------------------------------------------

#: What a supervisor plan decision is not. Restated so it cannot be conflated with the
#: layers that already exist.
FIVE_HUMAN_LAYERS = (
    "a recommendation override is Component 13's and changes who is in the approved queue; a "
    "scheduling adjustment is Component 14's and changes when an approved row is worked; an "
    "execution event is Component 14's and records what a person reports actually happened; a "
    "review resolution is Component 16's and records that a human looked at a flagged "
    "historical-policy case; a supervisor plan decision is Component 21's and records what a "
    "supervisor decided about an already geographically organized, already selected proposed "
    "workload. A plan decision never creates an override, an adjustment, or a review "
    "resolution itself -- those remain separate submissions through their own contracts"
)

#: What Component 21 must never do, printed in every manifest.
PLAN_REVIEW_CANNOT = (
    "a supervisor plan decision changes only the plan decision log. It never edits "
    "calibrated_score, base_score, rank, policy_rank, selection_reason, selection_mechanism, "
    "geographic_group_id, or work_block_id. It never creates a Component 13 override or a "
    "Component 14 adjustment itself -- a decision to move an inspection or not proceed with "
    "it as planned is a distinct, additional, audited fact about the plan, not an edit to "
    "Component 19 or Component 20's own output. Turning that stated intent into a real "
    "capacity/schedule change remains a separate submission through Component 13's or "
    "Component 14's own contract"
)


# --- 2. the decision vocabulary -----------------------------------------------------


class PlanDecisionAction(StrEnum):
    """What a supervisor may decide about one establishment in a Component 20 plan.

    Four verbs, deliberately smaller than Component 16's four: there is no "escalate" or
    "refer" here because a plan decision is not routing a flagged exception -- it is the
    supervisor's own decision about the proposed workload itself.

    Disjoint from Component 13's ``OverrideAction``, Component 14's ``AdjustmentAction``, and
    Component 16's ``ReviewResolutionAction`` by construction and by import-time guard.
    """

    #: The supervisor reviewed the establishment and is proceeding with Sentinel's
    #: recommendation and Component 20's geographic placement exactly as proposed.
    KEEP_SELECTED = "keep_selected"

    #: The supervisor is not proceeding with this establishment on the date/work block
    #: Component 20 proposed, and states an intended later date. This records intent only --
    #: it does not itself move the establishment; a Component 14 adjustment, submitted
    #: separately, is what would actually change a schedule.
    MOVE_TO_LATER_WORKDAY = "move_to_later_workday"

    #: The supervisor is not proceeding with this establishment as planned at all (e.g. local
    #: knowledge makes the inspection currently inappropriate, redundant, or infeasible).
    #: This does not remove the establishment from Component 19's selected set or Component
    #: 20's plan -- both remain visible and unedited; only the decision log changes.
    DO_NOT_PROCEED_AS_PLANNED = "do_not_proceed_as_planned"

    #: The supervisor wants field work to visit this establishment earlier or later than
    #: Sentinel's own ``policy_rank`` implies, for operational reasons (e.g. two selected
    #: establishments share a building). This sets ``operational_priority`` -- a display-only
    #: ordering field -- and never touches ``rank`` or ``policy_rank``, which remain exactly
    #: Component 18/19's own values. See ``INHERITED_LIMITATIONS`` and ``PLAN_REVIEW_CANNOT``.
    ADJUST_OPERATIONAL_PRIORITY = "adjust_operational_priority"


#: Which optional field, if any, each decision action expects. Checked by the parser: a
#: MOVE_TO_LATER_WORKDAY decision with no ``revised_planned_date``, or an
#: ADJUST_OPERATIONAL_PRIORITY decision with no ``revised_operational_priority``, gives a
#: supervisor's stated intent with nothing to act on.
REQUIRED_FIELD_FOR_ACTION: dict[str, str | None] = {
    PlanDecisionAction.KEEP_SELECTED: None,
    PlanDecisionAction.MOVE_TO_LATER_WORKDAY: "revised_planned_date",
    PlanDecisionAction.DO_NOT_PROCEED_AS_PLANNED: None,
    PlanDecisionAction.ADJUST_OPERATIONAL_PRIORITY: "revised_operational_priority",
}

#: Every field a plan decision must carry. Absent or blank is a validation error, not a
#: default: a decision without an actor is an anonymous change to a supervisor's own record,
#: which is the precise thing an audit trail exists to prevent.
PLAN_REVIEW_REQUIRED_FIELDS: tuple[str, ...] = (
    "decision_id",
    "planning_date",
    "target_inspection_id",
    "decision_action",
    "reason_code",
    "actor",
    "decided_at",
)

#: Outcomes a decision can have. Same shape as Component 16's resolution outcomes: a decision
#: that changed nothing is logged as loudly as one that did.
PLAN_DECISION_OUTCOMES: tuple[str, ...] = (
    "applied",
    "no_op_already_decided",
    "establishment_not_in_plan",
)

# --- 3. derived approval status ------------------------------------------------------


class PlanApprovalStatus(StrEnum):
    """A plan's review status.

    The first three are derived at read time from the decision log alone, never stored, so
    they can never drift from the log that backs them. ``APPROVED`` is the one exception: it
    is not derived from decision completeness (a plan with every row decided is not
    automatically "approved" -- that would make approval a side effect of data entry rather
    than a supervisor's own act), and it can only be reached through an explicit,
    separately-validated ``PlanApprovalRecord`` (see ``approval.py``). Once an approval record
    exists for a given source plan, that fact is permanent for that exact source; it is never
    inferred, and it is never silently revoked.
    """

    #: No supervisor decision has been recorded for any establishment in the plan yet.
    DRAFT = "draft"

    #: At least one, but not all, establishments in the plan have a recorded decision.
    UNDER_SUPERVISOR_REVIEW = "under_supervisor_review"

    #: Every establishment in the plan has a recorded decision, but the plan has not been
    #: explicitly approved.
    ADJUSTED = "adjusted"

    #: A supervisor explicitly approved the plan (see ``approval.py``). The plan is now the
    #: authoritative ``ApprovedOperationalPlan`` Component 22 consumes.
    APPROVED = "approved"


def derive_plan_approval_status(
    *, total: int, decided: int, is_approved: bool = False
) -> PlanApprovalStatus:
    if is_approved:
        return PlanApprovalStatus.APPROVED
    if total == 0 or decided == 0:
        return PlanApprovalStatus.DRAFT
    if decided >= total:
        return PlanApprovalStatus.ADJUSTED
    return PlanApprovalStatus.UNDER_SUPERVISOR_REVIEW


# --- 3b. plan approval -----------------------------------------------------------------

#: Every field a supervisor's approval *request* must carry -- exactly ``PlanApprovalRequest``'s
#: own required fields, never a field only ``ApprovedPlanManifest`` carries (like
#: ``source_plan_review_sha256``, which is computed from the plan review file at commit time,
#: not typed by a supervisor). Same discipline as ``PLAN_REVIEW_REQUIRED_FIELDS``: an approval
#: without an approver is an anonymous decision to hand a plan to execution, which is precisely
#: what an audit trail exists to prevent.
PLAN_APPROVAL_REQUIRED_FIELDS: tuple[str, ...] = (
    "approval_id",
    "planning_date",
    "approved_by",
    "approved_at",
)

#: The artifact family name Component 22 reads. Stated once so the handoff contract has one
#: name, not one the CLI slug happens to agree with by convention.
APPROVED_PLAN_DATASET_SLUG = "approved_operational_plan"


# --- 4. inherited boundaries ----------------------------------------------------------

#: Written into every manifest and printed on every run.
DOES_NOT_ESTABLISH: tuple[str, ...] = (
    "that Component 20's geographic organization is a driving route, a confirmed schedule, "
    "or a travel-time estimate. See Component 20's own NON_GOALS",
    "that the selected inspection workload represents confirmed staffing or inspector "
    "appointments. Capacity here is Component 19's historical-activity-based estimate, not "
    "confirmed future staffing",
    "that a KEEP_SELECTED decision is a supervisor's endorsement of Sentinel's model. It "
    "means only that no change was requested",
    "that a DO_NOT_PROCEED_AS_PLANNED decision removes the establishment from Component 19's "
    "selected set or Component 20's plan. Both remain visible and unedited; only the "
    "decision log records the supervisor's intent",
    "that ADJUST_OPERATIONAL_PRIORITY changes Sentinel's model-derived risk rank. It sets a "
    "separate, display-only operational_priority field; rank and policy_rank remain exactly "
    "Component 18/19's own values",
    "that an APPROVED plan represents confirmed inspector staffing, a driving route, or a "
    "legal/regulatory sign-off. It records only that a named supervisor reviewed the proposed "
    "workload and elected to proceed with it, with whatever decisions were on record at that "
    "moment",
)

#: Limitations inherited whole from upstream components.
INHERITED_LIMITATIONS: tuple[str, ...] = (
    "Component 19: capacity is derived from historical inspection activity, not confirmed "
    "future staffing; this component inherits that boundary and does not sharpen it",
    "Component 20: geographic organization uses straight-line (Haversine) distance only; it "
    "does not know road networks, traffic, or travel time",
)


def _guard_registry() -> None:
    """Check the frozen constants against each other, and against the other three layers."""
    if set(REQUIRED_FIELD_FOR_ACTION) != set(PlanDecisionAction):
        raise PlanReviewDefinitionError(
            "REQUIRED_FIELD_FOR_ACTION must declare exactly one entry per PlanDecisionAction"
        )

    if not PLAN_REVIEW_REQUIRED_FIELDS:
        raise PlanReviewDefinitionError("the required-fields list is empty")
    if len(set(PLAN_REVIEW_REQUIRED_FIELDS)) != len(PLAN_REVIEW_REQUIRED_FIELDS):
        raise PlanReviewDefinitionError("the plan-review contract repeats a field")
    for required in ("actor", "reason_code"):
        if required not in PLAN_REVIEW_REQUIRED_FIELDS:
            raise PlanReviewDefinitionError(
                f"the plan-review contract has no {required!r} field. An anonymous plan "
                "decision is the precise failure an audit trail exists to prevent"
            )

    # Mechanical disjointness: a plan-decision verb must never be spellable as an override,
    # adjustment, or review-resolution verb.
    for other_name, other_enum in (
        ("OverrideAction", OverrideAction),
        ("AdjustmentAction", AdjustmentAction),
        ("ReviewResolutionAction", ReviewResolutionAction),
    ):
        clash = {str(a) for a in PlanDecisionAction} & {str(o) for o in other_enum}
        if clash:
            raise PlanReviewDefinitionError(
                f"plan decision verbs collide with {other_name} on {', '.join(sorted(clash))}. "
                "A plan decision, a recommendation override, a scheduling adjustment, and a "
                "review resolution must never be confusable"
            )

    for value in (*PlanDecisionAction, *PlanApprovalStatus):
        if "defer" in str(value):
            raise PlanReviewDefinitionError(
                f"{value!r} reuses 'defer', reserved by Component 14's ScheduleStatus.DEFERRED"
            )

    for name, value in (
        ("DOES_NOT_ESTABLISH", DOES_NOT_ESTABLISH),
        ("INHERITED_LIMITATIONS", INHERITED_LIMITATIONS),
    ):
        if not value:
            raise PlanReviewDefinitionError(
                f"{name} is empty. It travels in every manifest so the boundary arrives with "
                "the artifact; an empty list silently drops it"
            )

    if len(set(PLAN_APPROVAL_REQUIRED_FIELDS)) != len(PLAN_APPROVAL_REQUIRED_FIELDS):
        raise PlanReviewDefinitionError("the plan-approval contract repeats a field")
    # `source_plan_review_sha256` is deliberately not in this tuple: it identifies which plan
    # review file was approved, but a supervisor's *request* doesn't carry a checksum of a file
    # they never open directly -- ``build_approved_plan`` computes it independently from
    # ``review_path`` at commit time (see ``ApprovedPlanManifest``), so "unsourced" approval
    # is structurally impossible regardless of what the request itself carries.
    for required in ("approved_by", "approved_at"):
        if required not in PLAN_APPROVAL_REQUIRED_FIELDS:
            raise PlanReviewDefinitionError(
                f"the plan-approval contract has no {required!r} field. An anonymous "
                "approval is the precise failure an audit trail exists to prevent"
            )


_guard_registry()


__all__ = [
    "APPROVED_PLAN_DATASET_SLUG",
    "DOES_NOT_ESTABLISH",
    "FIVE_HUMAN_LAYERS",
    "INHERITED_LIMITATIONS",
    "PLAN_APPROVAL_REQUIRED_FIELDS",
    "PLAN_DECISION_OUTCOMES",
    "PLAN_REVIEW_CANNOT",
    "PLAN_REVIEW_DEFINITION_VERSION",
    "PLAN_REVIEW_REQUIRED_FIELDS",
    "REQUIRED_FIELD_FOR_ACTION",
    "PlanApprovalStatus",
    "PlanDecisionAction",
    "PlanReviewDefinitionError",
    "derive_plan_approval_status",
]
