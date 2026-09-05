"""The support decision: the gate every other number in this component passes through.

Support is decided before any metric is computed, and that order is the discipline the whole
audit rests on. A group below the floor produces a row with real counts, a status and a stated
reason -- never an absent row -- which is what makes "we measured 51 of 78 community areas"
sayable and what stops "equal performance across groups" from resting silently on the ones
that were dropped.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.fairness import support
from sentinel.fairness.definitions import (
    CALIBRATION_MIN_ROWS,
    SUPPORT_MIN_NEGATIVE,
    SUPPORT_MIN_POSITIVE,
    SUPPORT_MIN_ROWS,
    Grain,
    GroupStatus,
    MetricKind,
    group_definition_for,
)

COMMUNITY_AREA = group_definition_for("community_area")


def _frame(counts: dict[str, tuple[int, int]]) -> pl.DataFrame:
    """A frame with, per group value, ``(n_rows, n_positive)`` laid out explicitly."""
    rows: list[dict[str, object]] = []
    index = 0
    for value, (n_rows, n_positive) in counts.items():
        for position in range(n_rows):
            rows.append(
                {
                    "target_inspection_id": f"T{index:05d}",
                    "community_area": value,
                    "target": 1 if position < n_positive else 0,
                }
            )
            index += 1
    return pl.DataFrame(rows)


# --- 1. the classification rule ------------------------------------------------


def test_a_group_clearing_every_floor_is_supported_for_both_families() -> None:
    ranking, calibration, reason = support.classify(CALIBRATION_MIN_ROWS, 100, 200)
    assert ranking is GroupStatus.SUPPORTED
    assert calibration is GroupStatus.SUPPORTED
    assert reason == ""


def test_a_group_between_the_two_floors_supports_ranking_but_not_calibration() -> None:
    """The two floors differ for an arithmetic reason: a binned statistic spends its rows."""
    rows = (SUPPORT_MIN_ROWS + CALIBRATION_MIN_ROWS) // 2
    ranking, calibration, reason = support.classify(rows, 100, rows - 100)
    assert ranking is GroupStatus.SUPPORTED
    assert calibration is GroupStatus.INSUFFICIENT_SUPPORT
    assert str(CALIBRATION_MIN_ROWS) in reason


def test_a_group_below_the_row_floor_supports_neither() -> None:
    ranking, calibration, reason = support.classify(50, 25, 25)
    assert ranking is GroupStatus.INSUFFICIENT_SUPPORT
    assert calibration is GroupStatus.INSUFFICIENT_SUPPORT
    assert f"50 rows < {SUPPORT_MIN_ROWS}" in reason


def test_zero_positives_blocks_support_however_many_rows_there_are() -> None:
    """ROC-AUC is undefined on a single-class group; more rows do not change that."""
    ranking, _, reason = support.classify(5000, 0, 5000)
    assert ranking is GroupStatus.INSUFFICIENT_SUPPORT
    assert f"0 positives < {SUPPORT_MIN_POSITIVE}" in reason


def test_zero_negatives_blocks_support_too() -> None:
    ranking, _, reason = support.classify(5000, 5000, 0)
    assert ranking is GroupStatus.INSUFFICIENT_SUPPORT
    assert f"0 negatives < {SUPPORT_MIN_NEGATIVE}" in reason


def test_the_reason_names_every_floor_that_was_missed_not_just_the_first() -> None:
    """A group that is both too small and single-class has two problems.

    Reporting one would send a reader looking for more rows when what is missing is a
    second class.
    """
    _, _, reason = support.classify(30, 0, 30)
    assert "30 rows" in reason
    assert "0 positives" in reason


# --- 2. measuring a frame ---------------------------------------------------------


def test_every_observed_group_gets_a_record_including_the_tiny_ones() -> None:
    """The property the small-group policy is made of."""
    frame = _frame({"0": (400, 200), "1": (12, 6), "2": (250, 125)})
    records = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD_SET, fold_set="quarterly")
    assert [r.group_value for r in records] == ["0", "1", "2"]
    tiny = next(r for r in records if r.group_value == "1")
    assert tiny.ranking_status is GroupStatus.INSUFFICIENT_SUPPORT
    assert tiny.n_rows == 12
    assert tiny.insufficient_reason


def test_counts_are_real_even_for_an_unsupported_group() -> None:
    """Support gates the reading, never the arithmetic."""
    frame = _frame({"1": (12, 5)})
    record = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD, fold_set="quarterly")[0]
    assert (record.n_rows, record.n_positive, record.n_negative) == (12, 5, 7)
    assert record.base_rate == pytest.approx(5 / 12)


def test_representation_share_sums_to_one_across_the_groups() -> None:
    frame = _frame({"0": (400, 200), "1": (100, 50), "2": (500, 250)})
    records = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD_SET, fold_set="quarterly")
    assert sum(r.representation_share for r in records) == pytest.approx(1.0)


def test_records_are_sorted_by_group_value_not_by_size() -> None:
    """So two runs over the same data emit the rows in the same order."""
    frame = _frame({"9": (400, 200), "1": (900, 450), "5": (300, 150)})
    records = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD_SET, fold_set="quarterly")
    assert [r.group_value for r in records] == ["1", "5", "9"]


def test_a_single_class_group_is_flagged_as_such() -> None:
    frame = _frame({"0": (400, 400)})
    record = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD, fold_set="quarterly")[0]
    assert record.is_single_class


def test_the_fold_id_is_empty_at_the_pooled_grain() -> None:
    """Empty rather than null, so the column's meaning never depends on the grain."""
    frame = _frame({"0": (400, 200)})
    pooled = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD_SET, fold_set="quarterly")[0]
    per_fold = support.measure(
        frame, COMMUNITY_AREA, grain=Grain.FOLD, fold_set="quarterly", fold_id="quarterly-2024Q2"
    )[0]
    assert pooled.fold_id == ""
    assert per_fold.fold_id == "quarterly-2024Q2"


def test_a_missing_group_column_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(KeyError, match="community_area"):
        support.measure(
            pl.DataFrame({"target": [1]}),
            COMMUNITY_AREA,
            grain=Grain.FOLD,
            fold_set="quarterly",
        )


# --- 3. reading the records back ------------------------------------------------


def test_the_metric_family_decides_which_status_gates_a_metric() -> None:
    frame = _frame({"0": (SUPPORT_MIN_ROWS + 10, 100)})
    record = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD_SET, fold_set="quarterly")[0]
    assert support.status_for(record, MetricKind.RANKING) is GroupStatus.SUPPORTED
    assert support.status_for(record, MetricKind.PROBABILITY) is GroupStatus.INSUFFICIENT_SUPPORT


def test_supported_values_returns_only_the_qualifying_groups_sorted() -> None:
    frame = _frame({"0": (400, 200), "1": (12, 6), "2": (500, 250)})
    records = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD_SET, fold_set="quarterly")
    assert support.supported_values(records, kind=MetricKind.RANKING) == ("0", "2")


def test_the_summary_reports_both_floors_because_they_disagree() -> None:
    """Reporting only the stricter number understates what was measured; only the looser
    one overstates it. The artifact carries both per row and the summary carries both here.
    """
    frame = _frame({"0": (400, 200), "1": (12, 6), "2": (250, 125)})
    records = support.measure(frame, COMMUNITY_AREA, grain=Grain.FOLD_SET, fold_set="quarterly")
    summary = support.summarise(records)
    assert summary["observed"] == 3
    assert summary["supported_ranking"] == 2
    assert summary["supported_calibration"] == 1
    assert summary["insufficient"] == 1


def test_the_index_is_keyed_by_definition_value_grain_and_fold() -> None:
    frame = _frame({"0": (400, 200)})
    records = support.measure(
        frame, COMMUNITY_AREA, grain=Grain.FOLD, fold_set="quarterly", fold_id="quarterly-2024Q2"
    )
    index = support.index(records)
    assert ("community_area", "0", "fold", "quarterly-2024Q2") in index
