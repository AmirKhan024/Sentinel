"""The coverage-eligibility contract: one column, one predicate, and what it refuses.

The most consequential rule in Component 13, because it decides who a public agency reserves
inspection capacity for. It is also the smallest, which is the design: a rule that fits in one
sentence is a rule somebody can disagree with in a meeting.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.policy import eligibility
from sentinel.policy.definitions import ELIGIBILITY_COLUMN, SECONDARY_FLAG_COLUMN
from sentinel.policy.eligibility import (
    ELIGIBLE_FLAG,
    SECONDARY_FLAG,
    EligibilityError,
    annotate,
    refuse_forbidden,
    summarize,
)
from tests.conftest import make_model_feature_row, model_feature_scenario


def _features(histories: list[str]) -> pl.DataFrame:
    return model_feature_scenario(
        [
            make_model_feature_row(index, history=history, target=index % 2)
            for index, history in enumerate(histories)
        ]
    )


# --- 1. the predicate --------------------------------------------------------------


def test_a_row_with_no_code_era_canvass_is_eligible() -> None:
    """The frozen rule: zero code-era canvasses, which is when the priority features are NULL."""
    frame = annotate(_features(["no_code_era", "full"]))
    assert frame[ELIGIBLE_FLAG].to_list() == [True, False]


def test_a_row_with_ordinary_history_is_not_eligible() -> None:
    frame = annotate(_features(["full", "full", "full"]))
    assert frame[ELIGIBLE_FLAG].sum() == 0


def test_a_row_with_no_history_at_all_is_eligible_and_carries_the_strict_flag() -> None:
    """The strict flag names a distinct population; it does not gate the reserve."""
    frame = annotate(_features(["none"]))
    assert frame[ELIGIBLE_FLAG].to_list() == [True]
    assert frame[SECONDARY_FLAG].to_list() == [True]


def test_the_strict_flag_is_a_subset_not_the_gate() -> None:
    """Profile 1 measured 401 of 57,727 rows: a reserve scaled to it is zero or one slot."""
    frame = annotate(_features(["none", "no_code_era", "full"]))
    strict = frame[SECONDARY_FLAG].to_list()
    eligible = frame[ELIGIBLE_FLAG].to_list()
    assert strict == [True, False, False]
    assert all(e for s, e in zip(strict, eligible, strict=True) if s)


def test_a_null_count_is_never_eligible() -> None:
    """The branch that must not become ``fill_null(0)`` during an edit.

    A null means the count itself is missing -- a join that failed, a column that moved. A
    zero means the establishment genuinely has no code-era canvass. Reserving capacity for the
    first would be reserving it for rows about which nothing at all is known.
    """
    frame = _features(["full"]).with_columns(pl.lit(None, dtype=pl.Int32).alias(ELIGIBILITY_COLUMN))
    assert annotate(frame)[ELIGIBLE_FLAG].to_list() == [False]


# --- 2. what eligibility refuses to read --------------------------------------------


@pytest.mark.parametrize("column", ["target", "target_status"])
def test_an_outcome_column_is_refused_as_an_eligibility_input(column: str) -> None:
    """A policy that reads the label allocates inspections using the answer.

    The failure mode this whole component is arranged to make impossible, and the one that
    would produce entirely plausible-looking numbers if it ever happened.
    """
    with pytest.raises(EligibilityError, match="not a policy"):
        refuse_forbidden([column])


def test_the_eligibility_expression_refuses_a_label_column() -> None:
    with pytest.raises(EligibilityError, match="outcome columns"):
        eligibility.eligible_expr("target")


def test_ordinary_history_columns_pass_the_refusal_check() -> None:
    refuse_forbidden([ELIGIBILITY_COLUMN, SECONDARY_FLAG_COLUMN, "prior_canvass_count"])


# --- 3. a frame that cannot decide eligibility ---------------------------------------


def test_a_frame_without_the_history_column_says_which_one_is_missing() -> None:
    """A prediction artifact alone cannot decide eligibility, and must not guess."""
    frame = pl.DataFrame({"target_inspection_id": ["T1"], "score": [0.5]})
    with pytest.raises(EligibilityError, match=ELIGIBILITY_COLUMN):
        annotate(frame)


# --- 4. the summary -------------------------------------------------------------------


def test_the_summary_reports_the_eligible_base_rate_beside_the_window_rate() -> None:
    """The pair that refutes the intuition this component was built on.

    Profile 2 measured the eligible population's outcome rate *above* the window's in every
    quarterly fold. A summary that reported only counts would let a reader assume the opposite
    and conclude that a reserve is obviously warranted.
    """
    frame = annotate(_features(["no_code_era", "no_code_era", "full", "full"]))
    row = summarize(
        frame,
        grain="fold",
        fold_set="quarterly",
        fold_id="quarterly-2022Q2",
        definition_version="v1",
    )
    assert row["n_rows"] == 4
    assert row["n_eligible"] == 2
    assert row["eligible_share"] == pytest.approx(0.5)
    assert row["base_rate"] is not None
    assert row["eligible_base_rate"] is not None
    assert row["eligibility_column"] == ELIGIBILITY_COLUMN


def test_the_summary_annotates_a_frame_that_has_not_been_annotated() -> None:
    row = summarize(
        _features(["no_code_era"]),
        grain="fold_set",
        fold_set="quarterly",
        fold_id="",
        definition_version="v1",
    )
    assert row["n_eligible"] == 1


def test_a_window_with_no_positives_reports_a_null_rate_rather_than_a_zero() -> None:
    """Zero would read as 'nothing was ever cited here', which is a different claim."""
    frame = model_feature_scenario(
        [make_model_feature_row(i, history="full", target=0) for i in range(3)]
    )
    row = summarize(frame, grain="fold", fold_set="quarterly", fold_id="f", definition_version="v1")
    assert row["base_rate"] == 0.0
    assert row["eligible_share_of_positives"] is None
