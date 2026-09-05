"""Frozen contracts for Component 16. Every review constant lives here and nowhere else.

**The separation this module exists to hold.** Component 13 decides *who* to inspect and
Component 14 decides *when*. Neither ever stops and says "a human should look at this before it
is treated as automatically sufficient." This component is that stop: a deterministic,
rule-based gate that flags a case for human review, and nothing more. It never re-ranks a row,
never re-dates it, never edits a score, and never retrains a model.

**There is no confidence threshold anywhere in this file, and there is no flag to add one.**
ADR 0040 established Sentinel's ``ABSTENTION_POLICY``: the project has never built a predictive
interval, a conformal set or an ensemble spread, and manufacturing one to gate on would be
exactly the fabrication ADR 0040 refuses. Both triggers below are boolean facts already written
by an upstream component -- a warning column, or a matching row's absence in a log -- not a
score compared against a cutoff. See ADR 0051.

**This is the fourth human layer, and it must never collide with the other three.** Component
13 owns "who is in the queue" (``OverrideAction``); Component 14 owns "when an approved row is
worked" (``AdjustmentAction``) and "what happened" (``ExecutionStatus``). Component 14 also
already uses ``ScheduleStatus.DEFERRED`` to mean "moved to a later operating day" -- a
structurally different idea from "sent to a human for review." ``_guard_registry`` below checks
this component's verbs against the other three and additionally refuses the literal substring
``"defer"`` in any Component 16 vocabulary value, so the two concepts can never be confused in
code or in data.
"""

from __future__ import annotations

from enum import StrEnum

from sentinel.policy.definitions import OverrideAction
from sentinel.scheduling.definitions import AdjustmentAction

#: Bumped whenever the trigger contract, the resolution vocabulary or the required-fields list
#: changes in a way that makes two runs incomparable.
REVIEW_DEFINITION_VERSION = "v1"


class ReviewDefinitionError(ValueError):
    """Raised when the frozen review contracts contradict each other."""


# --- 1. the layer separation ----------------------------------------------------

#: The fourth human layer, stated beside the three ADR 0047 already named. Printed in the
#: manifest because the whole component is an argument that these are different things.
FOUR_HUMAN_LAYERS = (
    "a recommendation override is Component 13's and changes who is in the approved queue; a "
    "scheduling adjustment is Component 14's and changes when an approved row is worked; an "
    "execution event is Component 14's and records what a person reports actually happened; a "
    "review resolution is Component 16's and records that a human looked at a flagged case and "
    "what they decided to do about it. A review resolution never creates an override or an "
    "adjustment itself -- it records a pointer to one, submitted separately through its own "
    "contract"
)

#: What a review case is not. The sentence a reader reaches for when tempted to conflate this
#: with Component 14's scheduling status of the same rough shape.
DEFERRAL_IS_NOT_SCHEDULING_DEFERRAL = (
    "a Component 16 review case and a Component 14 ScheduleStatus.DEFERRED row are unrelated "
    "facts about the same establishment. The schedule status means a scheduled inspection was "
    "moved to a later operating day. A review case means a human should look at the row before "
    "it continues to be treated as automatically sufficient. Neither implies the other, and "
    "no Component 16 vocabulary value contains the word 'defer'"
)


# --- 2. the two deterministic triggers -------------------------------------------


class ReviewTriggerReason(StrEnum):
    """Why a case was flagged. Deterministic codes, never generated prose.

    No language model writes any value in this enum. A reason code that varied between runs
    would make the audit trail unreproducible, and one a reader cannot enumerate is one nobody
    can check. Component 13 and Component 14 make the same argument about their own codes.

    Exactly two members, and the count is deliberate. Both are boolean facts an upstream
    component already computed; neither is a probability compared against a threshold. Other
    candidate triggers -- an explanation Component 11 refused to produce, a validation check
    that failed without stopping the run -- were investigated and found to correspond to no
    real row: the one unsupported model is already excluded from every policy's candidate list,
    and the pipeline's severity split is binary at the run level, not the row level. A trigger
    with no row that could ever satisfy it is a code path no run takes, which is
    indistinguishable from one that is broken.
    """

    #: A selected recommendation carries at least one Component 13 policy warning. Component 13
    #: never escalates a warning past annotation (ADR 0040); this is that escalation.
    POLICY_WARNING_PRESENT = "policy_warning_present"

    #: A row occupying a schedule slot (scheduled or deferred, in Component 14's sense) has no
    #: matching row in the accumulated execution log. The row-level version of a count Component
    #: 14 already computes at cell grain (``NO_EXECUTION_RECORD``).
    NO_EXECUTION_RECORD_ON_SCHEDULED_ROW = "no_execution_record_on_scheduled_row"


#: The value written when a row triggers no review. A token rather than a null, for the reason
#: Component 13's ``NO_WARNING`` is: an empty cell is ambiguous between "not flagged" and
#: "triggers were not computed". Never written to a real queue row -- a queue row always has at
#: least one trigger, by construction.
NO_TRIGGER = "none"

#: Multiple triggers are joined with this, sorted, so the column is a deterministic set. Same
#: convention as Component 13's ``WARNING_SEPARATOR``.
TRIGGER_SEPARATOR = "|"

#: What a warning-trigger case is scoped to. Restricted to selected rows: a warning on a row
#: nobody was going to inspect was never an operational decision, and flagging it would blur
#: "not selected" with "needs review".
WARNING_TRIGGER_REQUIRES_SELECTION = True

#: No threshold, no flag, and the reason stated so it cannot be reintroduced quietly. Restated
#: here because three docstrings elsewhere in this project (Component 9's calibration
#: definitions, Component 5's evaluation metrics and evaluation build) each say "a threshold is
#: genuinely needed only by the Component 16 deferral gate" -- and ADR 0051 records, on the
#: record, that this component reads "threshold" as the flag/no-flag boundary those two triggers
#: already draw, not as a probability cutoff.
NO_THRESHOLD = (
    "no numeric score, probability or confidence threshold exists anywhere in this component "
    "and there is no flag to add one. Both triggers are boolean facts already written by "
    "Component 13 (warnings != none on a selected row) or derivable by an exact-match anti-join "
    "against Component 14's own execution log (no matching execution_id). Neither reads score, "
    "base_score or final_policy_rank to decide queue membership"
)


# --- 3. the human resolution layer -----------------------------------------------


class ReviewCaseStatus(StrEnum):
    """What the **queue** says about a flagged case. Two values.

    ``FLAGGED`` and ``RESOLVED`` are the only values a committed ``human_review_queue`` row
    carries. The API's "pending" presentation value (a flagged case with a staged but not yet
    committed resolution) is never written to this artifact -- it is computed the same way
    ``GET /v1/staged-requests`` already reconciles pending against applied for the other three
    human-input contracts.
    """

    FLAGGED = "flagged"
    RESOLVED = "resolved"


class ReviewResolutionAction(StrEnum):
    """What a human reviewer may do about a flagged case. Four verbs, all auditable.

    Disjoint from Component 13's ``OverrideAction`` and Component 14's ``AdjustmentAction`` by
    construction and by import-time guard. A resolution records what the reviewer decided about
    the *review case*; it never performs the override or the adjustment itself. Referring a case
    to either is a pointer to a decision made through that component's own contract, submitted
    separately.
    """

    #: The reviewer looked at the case and is taking no further action through this component.
    ACKNOWLEDGE = "acknowledge"

    #: The reviewer's decision is (or will be) recorded as a Component 13 override. This
    #: resolution carries the ``override_id`` as a pointer; it does not create the override.
    REFER_TO_OVERRIDE = "refer_to_override"

    #: The reviewer's decision is (or will be) recorded as a Component 14 adjustment. This
    #: resolution carries the ``adjustment_id`` as a pointer; it does not create the adjustment.
    REFER_TO_ADJUSTMENT = "refer_to_adjustment"

    #: The reviewer is escalating the case beyond what this component's contracts represent.
    ESCALATE = "escalate"


#: Which pointer field, if any, each resolution action requires. Checked by the parser: a
#: ``REFER_TO_OVERRIDE`` resolution with no ``referenced_override_id`` is not a request the
#: system can trace back to the decision it is referring to.
POINTER_FIELD_FOR_ACTION: dict[str, str | None] = {
    ReviewResolutionAction.ACKNOWLEDGE: None,
    ReviewResolutionAction.REFER_TO_OVERRIDE: "referenced_override_id",
    ReviewResolutionAction.REFER_TO_ADJUSTMENT: "referenced_adjustment_id",
    ReviewResolutionAction.ESCALATE: None,
}

#: Every field a review resolution must carry. Absent or blank is a validation error, not a
#: default: a resolution without an actor is an anonymous decision about a flagged case, which is
#: the precise thing an audit trail exists to prevent.
REVIEW_REQUIRED_FIELDS: tuple[str, ...] = (
    "review_id",
    "policy_id",
    "fold_id",
    "k_name",
    "target_inspection_id",
    "resolution_action",
    "reason_code",
    "actor",
    "decided_at",
)

#: Outcomes a resolution can have. One-to-one in shape with Component 13's override outcomes and
#: Component 14's adjustment outcomes: a resolution that changed nothing is logged as loudly as
#: one that did.
REVIEW_RESOLUTION_OUTCOMES: tuple[str, ...] = (
    "applied",
    "no_op_already_resolved",
    "case_not_in_queue",
)

#: What a review resolution may never do.
REVIEW_CANNOT = (
    "a review resolution changes only the review case's own status. It never edits a score, a "
    "rank, a decision mechanism, a decision reason, a schedule status, a policy override log or "
    "a scheduling adjustment log. Referring a case to override or adjustment records the human's "
    "stated intent as a pointer; it never creates the override or adjustment record itself -- "
    "that remains a separate submission through Component 13's or Component 14's own contract"
)

#: Whether a pointer to an as-yet-unsubmitted override or adjustment is permitted. Chosen
#: deliberately: a human may state intent ("I am going to override this") before the override
#: itself exists, and the pointer's validity is therefore an advisory finding, never a refusal.
POINTER_MAY_FORWARD_REFERENCE = True

#: The scope of the reproducibility claim, stated precisely because overstating it would be the
#: easiest lie in this component.
DETERMINISM_SCOPE = (
    "the review-queue computation is deterministic: identical Component 13/14 inputs produce a "
    "byte-identical queue, and shuffling the input rows changes nothing. Resolutions are "
    "external human decisions, so a run is byte-identical only given the identical resolutions "
    "file, and the manifest pins that file by checksum rather than claiming a human decision is "
    "reproducible computation"
)


# --- 4. inherited boundaries ------------------------------------------------------

#: The line between a review flag and an abstention, restated so it cannot be blurred later.
#: Inherited unchanged from ADR 0040.
ABSTENTION_POLICY_INHERITED = (
    "this component inherits ADR 0040 unchanged: Sentinel never abstains, and flagging a case "
    "for review is not an abstention. Every flagged row already carries a rank and a "
    "recommendation from Component 13; review annotates it and can route it to a human "
    "decision, but the recommendation itself is never withheld, blanked or replaced with a "
    "'no decision' placeholder while review is pending"
)

#: Written into every manifest and printed on every run.
DOES_NOT_ESTABLISH: tuple[str, ...] = (
    "that a flagged case is wrong, risky, or should be overridden. A warning or an execution "
    "gap is a fact about what is known, not a verdict",
    "that an unflagged case needed no review. The trigger set is deliberately narrow -- two "
    "boolean facts -- and a real department may have review criteria this component does not "
    "represent",
    "that a referenced override_id or adjustment_id is valid, applied, or even yet submitted. "
    "The pointer records a human's stated intent; the validator checks it only as an advisory",
    "any claim about how quickly a flagged case should be resolved. There is no SLA, no clock "
    "and no queue-age threshold anywhere in this component",
)

#: Experiments this component is not permitted to run, each with the ADR that blocked it.
BLOCKED: tuple[str, ...] = (
    "introducing a probability, score or confidence threshold of any kind (ADR 0040, ADR 0051)",
    "re-ranking, re-selecting or re-scoring any row (ADR 0037, ADR 0042)",
    "creating an override or adjustment record directly. A REFER_TO_OVERRIDE or "
    "REFER_TO_ADJUSTMENT resolution records a pointer only; the override or adjustment itself "
    "is a separate submission through its own contract (ADR 0047)",
    "reusing 'defer' or 'deferred' as a status or verb token. Component 14's "
    "ScheduleStatus.DEFERRED means something structurally different and the two must never be "
    "confusable (ADR 0051)",
)

#: Limitations inherited whole from upstream components.
INHERITED_LIMITATIONS: tuple[str, ...] = (
    "Component 13: a policy warning is a fact about what is known, not an estimate of risk; "
    "this component inherits that boundary and does not sharpen it into a score",
    "Component 14: an execution gap is external, unverified field reporting; a case flagged for "
    "a missing execution record says nothing about whether the inspection happened",
)


def _guard_registry() -> None:
    """Check the frozen constants against each other, and against the other three layers.

    Every one of these has a way of drifting during an edit, and every one of them would fail
    silently and plausibly. A verb that collides with an override or adjustment, a pointer field
    with no action, an empty boundary list -- each produces a run that finishes green and blurs
    a distinction this component exists to hold.
    """
    if TRIGGER_SEPARATOR in NO_TRIGGER:
        raise ReviewDefinitionError("the no-trigger token is not separable from a trigger list")
    for trigger in ReviewTriggerReason:
        if TRIGGER_SEPARATOR in trigger:
            raise ReviewDefinitionError(f"trigger {trigger!r} contains the separator")
    if not list(ReviewTriggerReason):
        raise ReviewDefinitionError("no trigger reason is declared")

    if set(POINTER_FIELD_FOR_ACTION) != set(ReviewResolutionAction):
        raise ReviewDefinitionError(
            "POINTER_FIELD_FOR_ACTION must declare exactly one entry per ReviewResolutionAction"
        )

    if not REVIEW_REQUIRED_FIELDS:
        raise ReviewDefinitionError("the required-fields list is empty")
    if len(set(REVIEW_REQUIRED_FIELDS)) != len(REVIEW_REQUIRED_FIELDS):
        raise ReviewDefinitionError("the review contract repeats a field")
    for required in ("actor", "reason_code"):
        if required not in REVIEW_REQUIRED_FIELDS:
            raise ReviewDefinitionError(
                f"the review contract has no {required!r} field. An anonymous review decision "
                "is the precise failure an audit trail exists to prevent"
            )

    # The disjointness this component depends on mechanically, not just by argument: a
    # resolution verb must never be spellable as an override or adjustment verb.
    for other_name, other_enum in (
        ("OverrideAction", OverrideAction),
        ("AdjustmentAction", AdjustmentAction),
    ):
        clash = {str(a) for a in ReviewResolutionAction} & {str(o) for o in other_enum}
        if clash:
            raise ReviewDefinitionError(
                f"review resolution verbs collide with {other_name} on "
                f"{', '.join(sorted(clash))}. A review resolution, a recommendation override "
                "and a scheduling adjustment must never be confusable"
            )

    # The explicit, named guarantee that "defer"/"deferred" never leaks into this component's
    # vocabulary -- stronger than mere enum-value disjointness, because it also catches a value
    # that happens not to collide with the other two enums but still reuses the reserved word.
    for value in (*ReviewCaseStatus, *ReviewTriggerReason, *ReviewResolutionAction):
        if "defer" in str(value):
            raise ReviewDefinitionError(
                f"{value!r} reuses 'defer', reserved by Component 14's ScheduleStatus.DEFERRED"
            )

    for name, value in (
        ("DOES_NOT_ESTABLISH", DOES_NOT_ESTABLISH),
        ("BLOCKED", BLOCKED),
        ("INHERITED_LIMITATIONS", INHERITED_LIMITATIONS),
    ):
        if not value:
            raise ReviewDefinitionError(
                f"{name} is empty. It travels in every manifest so the boundary arrives with "
                "the artifact; an empty list silently drops it"
            )


_guard_registry()


__all__ = [
    "ABSTENTION_POLICY_INHERITED",
    "BLOCKED",
    "DEFERRAL_IS_NOT_SCHEDULING_DEFERRAL",
    "DETERMINISM_SCOPE",
    "DOES_NOT_ESTABLISH",
    "FOUR_HUMAN_LAYERS",
    "INHERITED_LIMITATIONS",
    "NO_THRESHOLD",
    "NO_TRIGGER",
    "POINTER_FIELD_FOR_ACTION",
    "POINTER_MAY_FORWARD_REFERENCE",
    "REVIEW_CANNOT",
    "REVIEW_DEFINITION_VERSION",
    "REVIEW_REQUIRED_FIELDS",
    "REVIEW_RESOLUTION_OUTCOMES",
    "TRIGGER_SEPARATOR",
    "WARNING_TRIGGER_REQUIRES_SELECTION",
    "ReviewCaseStatus",
    "ReviewDefinitionError",
    "ReviewResolutionAction",
    "ReviewTriggerReason",
]
