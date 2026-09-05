"""The deterministic baselines: the yardstick Component 6 has to beat.

Nothing here is fitted, so these tests are about two things: that each rule
orders rows the way it says it does, and that its **missing-value rule** is the
one written down. The second matters more than it looks. Roughly a quarter of
rows have no prior code-era canvass, so where the nulls sort largely decides
what these baselines score -- and choosing that placement by looking at the
outcome would be fitting to the label under another name.
"""

from __future__ import annotations

import pytest

from sentinel.evaluation.rankers import (
    RANKER_VERSION,
    RANKERS,
    RANKERS_BY_NAME,
    constant_scores,
    days_since_last_canvass_scores,
    prior_canvass_priority_rate_scores,
    priority_at_last_canvass_scores,
    random_scores,
    required_columns,
    score,
)
from tests.conftest import feature_scenario, make_feature_row


def _frame(**columns: list[object]):
    length = len(next(iter(columns.values())))
    rows = []
    for i in range(length):
        overrides = {name: values[i] for name, values in columns.items()}
        rows.append(make_feature_row(i, **overrides))
    return feature_scenario(rows)


# --- 1. days since last canvass --------------------------------------------


def test_a_longer_gap_scores_higher() -> None:
    """The spec's first heuristic: the most overdue goes first."""
    frame = _frame(days_since_last_canvass=[100, 900, 400])
    assert days_since_last_canvass_scores(frame) == [100.0, 900.0, 400.0]


def test_a_never_canvassed_row_sorts_first() -> None:
    """The limiting case of "a long time since the last one".

    Chosen on that argument, before measuring. The measured consequence is
    recorded in the module docstring: the never-canvassed group's base rate is
    above average, so the choice happens to help.
    """
    frame = _frame(days_since_last_canvass=[100, None, 400])
    scores = days_since_last_canvass_scores(frame)
    assert scores[1] > max(scores[0], scores[2])


def test_an_all_null_column_still_produces_finite_scores() -> None:
    frame = _frame(days_since_last_canvass=[None, None])
    scores = days_since_last_canvass_scores(frame)
    assert scores == [1.0, 1.0]


# --- 2. priority at last canvass -------------------------------------------


def test_a_prior_priority_citation_scores_above_a_clean_one() -> None:
    frame = _frame(priority_at_last_canvass=[True, False])
    assert priority_at_last_canvass_scores(frame) == [1.0, 0.0]


def test_an_unknown_prior_sits_between_known_clean_and_known_cited() -> None:
    """Collapsing "we know nothing" into "we know it was clean" would assert
    evidence that does not exist."""
    frame = _frame(priority_at_last_canvass=[True, None, False])
    assert priority_at_last_canvass_scores(frame) == [1.0, 0.5, 0.0]


# --- 3. prior priority rate -------------------------------------------------


def test_the_rate_ranker_passes_the_rate_through_unchanged() -> None:
    frame = _frame(prior_canvass_priority_rate=[0.0, 0.25, 1.0])
    assert prior_canvass_priority_rate_scores(frame) == [0.0, 0.25, 1.0]


def test_an_undefined_rate_scores_a_half() -> None:
    frame = _frame(prior_canvass_priority_rate=[0.0, None, 1.0])
    assert prior_canvass_priority_rate_scores(frame) == [0.0, 0.5, 1.0]


# --- 4. random and constant -------------------------------------------------


def test_random_scores_are_reproducible_from_their_seed() -> None:
    assert random_scores(20, seed=7) == random_scores(20, seed=7)
    assert random_scores(20, seed=7) != random_scores(20, seed=8)


def test_random_scores_are_bounded_and_the_right_length() -> None:
    scores = random_scores(50, seed=1)
    assert len(scores) == 50
    assert all(0.0 <= s < 1.0 for s in scores)


def test_constant_scores_are_all_equal() -> None:
    """The diagnostic: with every score tied, only the tie-break decides."""
    assert constant_scores(4) == [0.0, 0.0, 0.0, 0.0]


# --- 5. the registry --------------------------------------------------------


def test_every_declared_ranker_can_be_scored() -> None:
    frame = _frame(
        days_since_last_canvass=[100, 200, None],
        priority_at_last_canvass=[True, False, None],
        prior_canvass_priority_rate=[0.1, 0.9, None],
    )
    for spec in RANKERS:
        scores = score(spec.name, frame, seed=1)
        assert len(scores) == frame.height
        assert all(isinstance(s, float) for s in scores)


def test_every_declared_ranker_states_a_null_rule() -> None:
    for spec in RANKERS:
        assert spec.null_rule
        assert spec.description


def test_no_ranker_claims_to_emit_a_calibrated_probability() -> None:
    """None of these is a probability, so none may be scored on Brier or ECE."""
    assert all(not spec.is_probability for spec in RANKERS)


def test_an_unknown_ranker_is_an_error_not_a_silent_default() -> None:
    with pytest.raises(KeyError, match="unknown ranker"):
        score("does_not_exist", _frame(days_since_last_canvass=[1]))


def test_required_columns_are_deduplicated_and_declared() -> None:
    columns = required_columns(list(RANKERS_BY_NAME))
    assert len(columns) == len(set(columns))
    assert "days_since_last_canvass" in columns
    assert "priority_at_last_canvass" in columns
    assert "target" not in columns  # a baseline never reads the label


def test_the_ranker_version_is_recorded() -> None:
    assert RANKER_VERSION == "v1"


# --- 6. scoring is independent of row order --------------------------------


@pytest.mark.parametrize(
    "name",
    ["days_since_last_canvass", "priority_at_last_canvass", "prior_canvass_priority_rate"],
)
def test_a_ranker_scores_a_row_from_that_rows_values_alone(name: str) -> None:
    """Reversing the frame reverses the scores and changes nothing else."""
    frame = _frame(
        days_since_last_canvass=[100, 200, 300],
        priority_at_last_canvass=[True, False, None],
        prior_canvass_priority_rate=[0.1, 0.5, 0.9],
    )
    forward = score(name, frame)
    backward = score(name, frame.reverse())
    assert forward == list(reversed(backward))
