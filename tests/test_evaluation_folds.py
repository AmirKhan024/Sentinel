"""Fold construction: the shape of the split, before any data touches it.

These tests are about the calendar arithmetic and the ordering invariants. The
question "does the split leak?" is asked separately, against real rows, in
``test_evaluation_leakage.py`` -- this file only proves the windows are where
they claim to be.

The invariant that matters most is that a fold **cannot be constructed** with
overlapping windows. ``FoldSpec`` raises in ``__post_init__`` rather than
leaving it to a checker, so there is no code path that produces a leaky fold and
reports it afterwards.
"""

from __future__ import annotations

from datetime import date

import pytest

from sentinel.evaluation.folds import (
    CODE_ERA_ANCHOR,
    MIN_TRAIN_QUARTERS,
    QUARTERLY,
    add_quarters,
    assign_split,
    covid_shift_fold,
    excluded_partial_windows,
    fold_stats,
    quarter_end,
    quarter_key,
    quarter_start,
    quarterly_folds,
    window_frame,
)
from sentinel.evaluation.models import FoldError, FoldSpec
from tests.conftest import spanning_features

# --- 1. calendar arithmetic ------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2022, 1, 1), date(2022, 1, 1)),
        (date(2022, 2, 14), date(2022, 1, 1)),
        (date(2022, 3, 31), date(2022, 1, 1)),
        (date(2022, 4, 1), date(2022, 4, 1)),
        (date(2022, 12, 31), date(2022, 10, 1)),
    ],
)
def test_quarter_start_snaps_to_the_containing_quarter(day: date, expected: date) -> None:
    assert quarter_start(day) == expected


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (date(2022, 1, 1), date(2022, 3, 31)),
        (date(2022, 4, 1), date(2022, 6, 30)),
        (date(2020, 1, 1), date(2020, 3, 31)),  # leap year
        (date(2022, 10, 1), date(2022, 12, 31)),
    ],
)
def test_quarter_end_is_the_day_before_the_next_quarter(start: date, expected: date) -> None:
    assert quarter_end(start) == expected


def test_add_quarters_crosses_year_boundaries_in_both_directions() -> None:
    assert add_quarters(date(2022, 10, 1), 1) == date(2023, 1, 1)
    assert add_quarters(date(2022, 1, 1), -1) == date(2021, 10, 1)
    assert add_quarters(date(2018, 7, 1), MIN_TRAIN_QUARTERS) == date(2022, 1, 1)


def test_the_anchor_plus_min_train_quarters_reproduces_the_spec_fold_one() -> None:
    """The project spec's Fold 1 is train through Dec 2021, calibrate on 2022Q1."""
    first_calibration = add_quarters(quarter_start(CODE_ERA_ANCHOR), MIN_TRAIN_QUARTERS)
    assert first_calibration == date(2022, 1, 1)
    assert quarter_key(first_calibration) == "2022Q1"


# --- 2. fold generation ----------------------------------------------------


def test_first_fold_matches_the_specified_structure() -> None:
    folds = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14))
    first = folds[0]
    assert first.train_start == date(2018, 7, 1)
    assert first.train_end == date(2021, 12, 31)
    assert first.calibration_start == date(2022, 1, 1)
    assert first.calibration_end == date(2022, 3, 31)
    assert first.test_start == date(2022, 4, 1)
    assert first.test_end == date(2022, 6, 30)


def test_fold_count_is_derived_from_the_data_not_hardcoded() -> None:
    """A longer snapshot must produce more folds without a code change."""
    short = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2023, 12, 31))
    long = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14))
    assert len(long) > len(short)
    assert [f.fold_id for f in short] == [f.fold_id for f in long[: len(short)]]


def test_a_partial_test_quarter_is_excluded_rather_than_shortened() -> None:
    """The snapshot ends mid-2026Q3, so 2026Q2 must be the last fold.

    Including a two-thirds-length window would compare it against full ones as
    if the two were the same measurement.
    """
    folds = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14))
    assert folds[-1].test_end == date(2026, 6, 30)
    assert all(f.test_end <= date(2026, 8, 14) for f in folds)


def test_the_excluded_partial_window_is_reported_not_silently_dropped() -> None:
    folds = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14))
    excluded = excluded_partial_windows(data_end=date(2026, 8, 14), folds=folds)
    assert len(excluded) == 1
    assert "2026Q3" in excluded[0]


def test_a_snapshot_ending_exactly_on_a_quarter_boundary_excludes_nothing() -> None:
    folds = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 6, 30))
    assert excluded_partial_windows(data_end=date(2026, 6, 30), folds=folds) == []


def test_too_little_data_produces_no_folds_rather_than_a_fabricated_one() -> None:
    assert quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2019, 12, 31)) == []


def test_data_start_after_data_end_is_rejected() -> None:
    with pytest.raises(ValueError, match="after data_end"):
        quarterly_folds(data_start=date(2026, 1, 1), data_end=date(2020, 1, 1))


# --- 3. the ordering invariants --------------------------------------------


def test_every_fold_orders_train_then_calibration_then_test() -> None:
    for fold in quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14)):
        assert fold.train_end < fold.calibration_start
        assert fold.calibration_start <= fold.calibration_end
        assert fold.calibration_end < fold.test_start
        assert fold.test_start <= fold.test_end


def test_the_training_window_expands_and_never_moves_its_anchor() -> None:
    folds = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14))
    anchor = folds[0].train_start
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.train_start == anchor
        assert later.train_end > earlier.train_end


def test_test_windows_are_disjoint_and_strictly_increasing() -> None:
    folds = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14))
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test_start > earlier.test_end


def test_a_fold_whose_calibration_overlaps_training_cannot_be_constructed() -> None:
    with pytest.raises(FoldError, match="strictly before"):
        FoldSpec(
            fold_set="broken",
            fold_id="broken-1",
            train_start=date(2018, 7, 1),
            train_end=date(2022, 1, 15),
            calibration_start=date(2022, 1, 1),
            calibration_end=date(2022, 3, 31),
            test_start=date(2022, 4, 1),
            test_end=date(2022, 6, 30),
        )


def test_a_fold_whose_calibration_overlaps_test_cannot_be_constructed() -> None:
    with pytest.raises(FoldError, match="strictly"):
        FoldSpec(
            fold_set="broken",
            fold_id="broken-2",
            train_start=date(2018, 7, 1),
            train_end=date(2021, 12, 31),
            calibration_start=date(2022, 1, 1),
            calibration_end=date(2022, 4, 15),
            test_start=date(2022, 4, 1),
            test_end=date(2022, 6, 30),
        )


def test_a_fold_with_reversed_bounds_cannot_be_constructed() -> None:
    with pytest.raises(FoldError):
        FoldSpec(
            fold_set="broken",
            fold_id="broken-3",
            train_start=date(2022, 1, 1),
            train_end=date(2018, 7, 1),
            calibration_start=date(2022, 4, 1),
            calibration_end=date(2022, 6, 30),
            test_start=date(2022, 7, 1),
            test_end=date(2022, 9, 30),
        )


# --- 4. the distribution-shift fold ----------------------------------------


def test_covid_shift_fold_trains_before_the_disruption_and_tests_through_it() -> None:
    (fold,) = covid_shift_fold(data_end=date(2026, 8, 14))
    assert fold.train_end == date(2020, 2, 29)
    assert fold.test_start == date(2020, 6, 1)
    assert fold.test_end == date(2021, 12, 31)
    assert fold.fold_set != QUARTERLY


def test_covid_shift_fold_is_withheld_when_the_snapshot_stops_short() -> None:
    assert covid_shift_fold(data_end=date(2021, 6, 30)) == []


# --- 5. splitting real rows ------------------------------------------------


def test_assign_split_labels_each_row_exactly_once() -> None:
    """Four labels, one per row, and the counts add back up to the whole frame."""
    import polars as pl

    frame = spanning_features().with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
    fold = quarterly_folds(data_start=date(2018, 7, 2), data_end=date(2026, 6, 30))[0]
    labelled = assign_split(frame, fold)

    assert labelled.height == frame.height
    counts = dict(
        labelled.group_by("split").len().iter_rows()  # type: ignore[arg-type]
    )
    assert sum(counts.values()) == frame.height
    assert set(counts) <= {"train", "calibration", "test", "outside"}


def test_assign_split_puts_every_row_on_the_correct_side_of_each_boundary() -> None:
    import polars as pl

    frame = spanning_features().with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
    fold = quarterly_folds(data_start=date(2018, 7, 2), data_end=date(2026, 6, 30))[0]
    labelled = assign_split(frame, fold)

    train = labelled.filter(pl.col("split") == "train")
    calibration = labelled.filter(pl.col("split") == "calibration")
    test = labelled.filter(pl.col("split") == "test")

    assert train["rd"].max() <= fold.train_end
    assert calibration["rd"].min() >= fold.calibration_start
    assert calibration["rd"].max() <= fold.calibration_end
    assert test["rd"].min() >= fold.test_start
    assert test["rd"].max() <= fold.test_end


def test_fold_stats_measure_capacity_from_the_data() -> None:
    import polars as pl

    frame = spanning_features(per_day=4).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    fold = quarterly_folds(data_start=date(2018, 7, 2), data_end=date(2026, 6, 30))[0]
    stats = fold_stats(frame, fold)
    assert stats.test_rows > 0
    assert stats.test_median_daily_capacity == 4.0
    assert stats.train_rows > stats.test_rows


def test_window_frame_is_sorted_chronologically_then_by_id() -> None:
    import polars as pl

    frame = spanning_features().with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
    fold = quarterly_folds(data_start=date(2018, 7, 2), data_end=date(2026, 6, 30))[0]
    window = window_frame(frame, fold)
    pairs = list(zip(window["rd"].to_list(), window["target_inspection_id"].to_list(), strict=True))
    assert pairs == sorted(pairs)


def test_split_of_reports_outside_for_a_date_in_no_window() -> None:
    from sentinel.evaluation.models import Split

    fold = quarterly_folds(data_start=date(2018, 7, 3), data_end=date(2026, 8, 14))[0]
    assert fold.split_of(date(2019, 1, 1)) is Split.TRAIN
    assert fold.split_of(date(2022, 2, 1)) is Split.CALIBRATION
    assert fold.split_of(date(2022, 5, 1)) is Split.TEST
    assert fold.split_of(date(2025, 1, 1)) is Split.OUTSIDE
    assert fold.split_of(date(2010, 1, 1)) is Split.OUTSIDE
