"""Warnings, and the human override layer. Pure -- no filesystem, no clock.

Two responsibilities, and they are next to each other because they are the same idea seen from
either side: what the system tells a reviewer, and what a reviewer is allowed to tell the
system back.

**A warning annotates a recommendation. It never withholds one.** Every row in the prediction
universe gets a rank and a decision; a warning says what is thin about the evidence behind it.
Sentinel does not abstain, and the reason is not squeamishness -- an abstention needs a per-row
confidence estimate, and this project has built no predictive interval, no conformal set and no
ensemble spread. Emitting an abstention category anyway would mean manufacturing the statistic
that justifies it. ADR 0040 holds that line.

**No warning is a score.** Each is a deterministic fact about what is known: this establishment
has no code-era history; this one has no inspection history at all; the audit could not measure
this establishment's neighbourhood; no neighbourhood could be recovered for it. None of them
moves a probability or a rank, and ``validate.py`` proves it by rebuilding the queue with the
warning inputs absent and comparing the ranks exactly.

**Overrides are external inputs, and the component is careful never to pretend otherwise.** A
human decision is not reproducible computation. The deterministic queue is written unchanged,
the override log is a separate table beside it, and the manifest pins the override file by
checksum and scopes the determinism claim to "identical given identical inputs including this
file". Claiming byte-identity across runs for something a person typed would be the easiest
lie available in this component.

**A freed slot is not backfilled.** When a reviewer removes an establishment from the queue,
the policy does not quietly promote the next one. Backfilling would be the system making a
second decision on the back of a human one, and the reviewer who struck a row did not ask for a
replacement. A ``force_include`` does displace -- capacity is fixed, so something has to go --
and the displaced establishment is named in the log rather than absorbed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sentinel.policy.definitions import (
    NO_WARNING,
    WARNING_SEPARATOR,
    DecisionMechanism,
    OverrideAction,
    PolicyWarning,
)
from sentinel.policy.models import Allocation, Override, PolicyWindow

#: Component 12's absence token, carried through unchanged so the two components name the same
#: population with the same string.
UNKNOWN_GROUP = "__UNKNOWN__"

#: Component 12's status value for a group it could not measure.
INSUFFICIENT_SUPPORT = "insufficient_support"

#: What happened to one override. A controlled vocabulary, because "the override did nothing"
#: and "the override was applied" must be distinguishable in an audit years later.
OUTCOME_APPLIED = "applied"
OUTCOME_NO_OP_ALREADY_SELECTED = "no_op_already_selected"
OUTCOME_NO_OP_NOT_SELECTED = "no_op_not_selected"
OUTCOME_ROW_NOT_IN_WINDOW = "row_not_in_window"


class GovernanceError(ValueError):
    """Raised when an override cannot be trusted enough to apply."""


def warnings_for(
    *,
    eligible: bool,
    secondary_no_history: bool,
    group_value: str | None,
    group_status: str | None,
) -> str:
    """The warning set for one row, as a sorted, separator-joined string.

    A set rather than a precedence. Choosing one warning to display would mean choosing which
    fact about an establishment a reviewer is allowed to see, and the four facts here are not
    ordered by importance -- "we have never inspected this place" and "we could not measure how
    the model behaves in this neighbourhood" are different problems for different readers.

    Sorted, so the column is deterministic and two runs produce the same bytes.
    """
    codes: set[str] = set()
    if eligible:
        codes.add(PolicyWarning.LIMITED_HISTORY)
    if secondary_no_history:
        codes.add(PolicyWarning.NO_PRIOR_INSPECTION)
    if group_value == UNKNOWN_GROUP:
        codes.add(PolicyWarning.UNKNOWN_GEOGRAPHY)
    if group_status == INSUFFICIENT_SUPPORT:
        codes.add(PolicyWarning.INSUFFICIENT_GROUP_AUDIT_SUPPORT)
    if not codes:
        return NO_WARNING
    return WARNING_SEPARATOR.join(sorted(codes))


def parse_overrides(payload: Sequence[Mapping[str, object]]) -> list[Override]:
    """Validate a decoded override file at the boundary, or refuse the whole run.

    Every field is required and none has a default. An override with no actor is an anonymous
    change to who gets inspected, and an override with no reason code is a change nobody can
    review -- both are the precise failures an audit trail exists to prevent, so neither is
    something to fill in helpfully.

    Refusing the whole file rather than skipping bad rows is deliberate. A partially applied
    override file produces a queue nobody authorised: the reviewer believes they made five
    changes and four happened.
    """
    actions = {a.value for a in OverrideAction}
    overrides: list[Override] = []
    for position, raw in enumerate(payload):
        try:
            override = Override.model_validate(dict(raw))
        except Exception as exc:  # pydantic raises its own error type
            raise GovernanceError(f"override {position}: {exc}") from exc
        blank = [
            field
            for field in Override.model_fields
            if not str(getattr(override, field, "")).strip()
        ]
        if blank:
            raise GovernanceError(
                f"override {override.override_id or position}: {', '.join(sorted(blank))} "
                "is blank. Every override field is required"
            )
        if override.action not in actions:
            raise GovernanceError(
                f"override {override.override_id}: unknown action {override.action!r}; "
                f"known: {', '.join(sorted(actions))}"
            )
        overrides.append(override)

    ids = [o.override_id for o in overrides]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise GovernanceError(
            f"duplicate override_id: {', '.join(duplicates)}. The id is how a decision is "
            "referred to afterwards, so two decisions cannot share one"
        )
    return overrides


def apply_overrides(
    window: PolicyWindow,
    allocation: Allocation,
    overrides: Sequence[Override],
    *,
    mechanisms: Sequence[str],
    reasons: Sequence[str],
    ranks: Sequence[int | None],
    definition_version: str,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    """Apply the overrides addressed to this cell, and return the log and the final decisions.

    Applied in ``override_id`` order rather than file order, so re-serialising the file cannot
    change the queue. Two reviewers' decisions that interact -- one including, one excluding --
    must resolve the same way whichever order the file happened to list them in.

    Returns the log rows and a map from ``target_inspection_id`` to its **final** selected
    state. The deterministic recommendation artifact is written from ``mechanisms`` and
    ``ranks``, untouched; this map is the operations layer's answer, and it lives beside the
    policy layer's rather than overwriting it.
    """
    index_of = {row_id: position for position, row_id in enumerate(window.ids)}
    selected: dict[int, bool] = {
        position: mechanisms[position] != DecisionMechanism.NOT_SELECTED
        for position in range(window.n)
    }
    addressed = [
        override
        for override in overrides
        if override.policy_id == allocation.policy_id
        and override.fold_id == allocation.fold_id
        and override.k_name == allocation.k_name
    ]

    log: list[dict[str, object]] = []
    for override in sorted(addressed, key=lambda o: o.override_id):
        position = index_of.get(override.target_inspection_id)
        if position is None:
            log.append(
                _log_row(
                    override,
                    allocation,
                    definition_version,
                    original_selected=None,
                    original_mechanism="",
                    original_reason="",
                    original_rank=None,
                    final_selected=None,
                    displaced="",
                    outcome=OUTCOME_ROW_NOT_IN_WINDOW,
                )
            )
            continue

        was_selected = selected[position]
        displaced = ""
        if override.action == OverrideAction.FORCE_INCLUDE:
            if was_selected:
                outcome = OUTCOME_NO_OP_ALREADY_SELECTED
            else:
                # Capacity is fixed, so an inclusion costs an exclusion. The lowest-ranked
                # risk selection still standing is the one that goes, and it is named.
                victim = _lowest_standing_risk(allocation, selected, ranks)
                if victim is None:
                    raise GovernanceError(
                        f"override {override.override_id}: nothing left to displace at "
                        f"{allocation.fold_id}/{allocation.k_name}. Including without "
                        "displacing would raise capacity, which is the one thing this "
                        "project's simulation never does"
                    )
                selected[victim] = False
                selected[position] = True
                displaced = window.ids[victim]
                outcome = OUTCOME_APPLIED
        else:
            if not was_selected:
                outcome = OUTCOME_NO_OP_NOT_SELECTED
            else:
                # No backfill. The freed slot stays free.
                selected[position] = False
                outcome = OUTCOME_APPLIED

        log.append(
            _log_row(
                override,
                allocation,
                definition_version,
                original_selected=was_selected,
                original_mechanism=mechanisms[position],
                original_reason=reasons[position],
                original_rank=ranks[position],
                final_selected=selected[position],
                displaced=displaced,
                outcome=outcome,
            )
        )

    final = {window.ids[position]: state for position, state in selected.items()}
    return log, final


def _lowest_standing_risk(
    allocation: Allocation, selected: Mapping[int, bool], ranks: Sequence[int | None]
) -> int | None:
    """The risk selection with the worst policy rank that has not already been removed.

    The reserve is not raided. A reviewer including an establishment is exercising judgement
    the risk ranking did not have; taking the slot from the coverage allocation would quietly
    convert every override into a coverage cut, which is a policy change nobody made.
    """
    standing = [i for i in allocation.risk_indices if selected.get(i, False)]
    if not standing:
        return None
    return max(standing, key=lambda i: (ranks[i] or 0, i))


def _log_row(
    override: Override,
    allocation: Allocation,
    definition_version: str,
    *,
    original_selected: bool | None,
    original_mechanism: str,
    original_reason: str,
    original_rank: int | None,
    final_selected: bool | None,
    displaced: str,
    outcome: str,
) -> dict[str, object]:
    """One audit row. The original recommendation and the final decision, side by side.

    Both are kept because the question an audit asks is never "what happened" alone -- it is
    "what would have happened, what happened instead, and who decided".
    """
    return {
        "override_id": override.override_id,
        "policy_id": override.policy_id,
        "fold_set": allocation.fold_set,
        "fold_id": override.fold_id,
        "k_name": override.k_name,
        "target_inspection_id": override.target_inspection_id,
        "action": override.action,
        "reason_code": override.reason_code,
        "actor": override.actor,
        "decided_at": override.decided_at,
        "original_is_selected": original_selected,
        "original_mechanism": original_mechanism,
        "original_reason": original_reason,
        "original_policy_rank": original_rank,
        "final_is_selected": final_selected,
        "displaced_target_inspection_id": displaced,
        "outcome": outcome,
        "policy_definition_version": definition_version,
    }


__all__ = [
    "INSUFFICIENT_SUPPORT",
    "OUTCOME_APPLIED",
    "OUTCOME_NO_OP_ALREADY_SELECTED",
    "OUTCOME_NO_OP_NOT_SELECTED",
    "OUTCOME_ROW_NOT_IN_WINDOW",
    "UNKNOWN_GROUP",
    "GovernanceError",
    "apply_overrides",
    "parse_overrides",
    "warnings_for",
]
