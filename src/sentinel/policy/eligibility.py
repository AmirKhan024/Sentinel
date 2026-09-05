"""The coverage-eligibility contract. Pure -- no filesystem, no clock.

One predicate, over one column, from one component. That narrowness is the design.

An eligibility rule decides who a public agency reserves inspection capacity for, so it has to
survive being read aloud in a room where somebody disagrees with it. "Establishments with no
canvass since the 2018 code came into force" survives that. "Establishments the model finds
hard" does not, because it is unfalsifiable, and "establishments in neighbourhoods we could
not geocode" does not, because it is a data-quality artifact wearing a policy's clothes.

**Temporal safety is inherited, not re-derived.** ``prior_canvass_count_code_era`` is a
Component 4 as-of feature built under ADR 0010: it counts canvasses strictly before the row's
own reference date, from records that existed by then. This module reads that column and adds
nothing to it, which is the only way to be sure eligibility cannot see further than the model
did. ``validate.py`` re-derives the flag from the feature table and compares, so the claim is
checked on every run rather than argued once here.

**A null is never eligible.** The column carries ``NullRule.NEVER`` in Component 4, so a zero
is a positive observation that no code-era canvass exists. A null would mean the count itself
is missing, and treating the two as the same would reserve capacity for rows about which
nothing at all is known -- including, one snapshot later, rows a join quietly failed to match.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.policy.definitions import (
    ELIGIBILITY_COLUMN,
    FORBIDDEN_POLICY_COLUMNS,
    SECONDARY_FLAG_COLUMN,
)

#: The column names this module writes onto a frame.
ELIGIBLE_FLAG = "coverage_eligible"
SECONDARY_FLAG = "secondary_no_history"


class EligibilityError(ValueError):
    """Raised when eligibility cannot be decided from the columns offered."""


def refuse_forbidden(columns: Sequence[str]) -> None:
    """Reject any attempt to decide eligibility from a label.

    Called before the predicate is built rather than after it is applied. A rule that read
    ``target`` would allocate inspections using the answer to the question the inspection is
    supposed to settle, and it would do so while producing entirely plausible-looking numbers
    -- the failure mode this whole component is arranged to make impossible.
    """
    offenders = [c for c in columns if c in FORBIDDEN_POLICY_COLUMNS]
    if offenders:
        raise EligibilityError(
            f"eligibility may not read {', '.join(sorted(offenders))}: these are outcome "
            "columns, and a policy that reads the label is not a policy"
        )


def eligible_expr(column: str = ELIGIBILITY_COLUMN) -> pl.Expr:
    """The coverage-eligibility predicate: the count is exactly zero.

    ``fill_null(-1)`` rather than a null-safe comparison, so the intent is legible: a missing
    count is mapped to a value the predicate cannot match, and never to zero.
    """
    refuse_forbidden([column])
    return pl.col(column).fill_null(-1) == 0


def secondary_expr(column: str = SECONDARY_FLAG_COLUMN) -> pl.Expr:
    """The strict no-history flag: no inspection of any type was ever recorded.

    Reported, never allocated against. Profile 1 measured 401 of 57,727 rows, which at one day
    of real capacity is a reserve of zero or one slot -- a mechanism nobody could measure and
    therefore nobody could defend.
    """
    refuse_forbidden([column])
    return pl.col(column).fill_null(-1) == 0


def annotate(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach both history flags to a frame, or say which column is missing.

    Deciding eligibility once, here, is what makes it checkable later. A predicate scattered
    through the allocator could not be re-derived and compared by the validator, and a rule
    nobody can re-derive is a rule nobody can audit.
    """
    for column in (ELIGIBILITY_COLUMN, SECONDARY_FLAG_COLUMN):
        if column not in frame.columns:
            raise EligibilityError(
                f"eligibility needs {column!r}, which the frame does not carry. It comes from "
                "Component 4's as-of feature table; a prediction artifact alone cannot decide "
                "eligibility, and inferring it from what is present would be inventing history"
            )
    return frame.with_columns(
        eligible_expr().alias(ELIGIBLE_FLAG),
        secondary_expr().alias(SECONDARY_FLAG),
    )


def summarize(
    frame: pl.DataFrame,
    *,
    grain: str,
    fold_set: str,
    fold_id: str,
    definition_version: str,
) -> dict[str, object]:
    """One row of ``policy_coverage_eligibility``: how large is this population, and how risky?

    Both halves matter and the second is the one that surprises. The eligible population is
    not merely under-served or over-served -- profile 2 measured that its outcome rate runs
    *above* the window's in every quarterly fold, so a risk model that prioritises it is
    tracking something real rather than misfiring. A summary that reported only counts would
    have let a reader assume the opposite.
    """
    if ELIGIBLE_FLAG not in frame.columns:
        frame = annotate(frame)
    eligible = frame.filter(pl.col(ELIGIBLE_FLAG))
    n_rows = frame.height
    n_eligible = eligible.height
    positives = int(frame["target"].sum()) if n_rows else 0
    eligible_positives = int(eligible["target"].sum()) if n_eligible else 0
    return {
        "grain": grain,
        "fold_set": fold_set,
        "fold_id": fold_id,
        "n_rows": n_rows,
        "n_eligible": n_eligible,
        "eligible_share": (n_eligible / n_rows) if n_rows else None,
        "n_secondary_no_history": frame.filter(pl.col(SECONDARY_FLAG)).height,
        "n_positive": positives,
        "n_eligible_positive": eligible_positives,
        "base_rate": (positives / n_rows) if n_rows else None,
        "eligible_base_rate": (eligible_positives / n_eligible) if n_eligible else None,
        "eligible_share_of_positives": (eligible_positives / positives) if positives else None,
        "eligibility_column": ELIGIBILITY_COLUMN,
        "policy_definition_version": definition_version,
    }


__all__ = [
    "ELIGIBLE_FLAG",
    "SECONDARY_FLAG",
    "EligibilityError",
    "annotate",
    "eligible_expr",
    "refuse_forbidden",
    "secondary_expr",
    "summarize",
]
