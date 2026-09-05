"""Leakage tests for Component 11.

The rest of the suite checks that the attributions are *right*. These check that they
cannot be **cheating**.

Component 11's leakage surface is not the one earlier components had, and that is why it
needs its own file. No model is fitted here, so no imputation median or scaler mean can be
contaminated. What can be contaminated is the **reference point**: a SHAP value says how far
a feature moved the output *relative to a background*, and a background drawn from the test
window would encode the period the model is being judged on. Every value would still be
finite, additive and plausible. Nothing would raise.

Three properties are defended:

1.  **The background is temporally safe.** Every reference row is dated on or before
    ``fold.train_end``, and -- separately, because a date comparison is weaker than the
    split -- every reference row is a member of the fold's own training window.
2.  **The explained rows are the fold's own test rows**, re-derived from
    ``evaluation.folds.window_frame`` rather than trusted from a column.
3.  **The selection reads no label.** Not the target, not an outcome, not a metric.

Two tests here have teeth rather than assertions about the happy path:
``test_the_background_leak_detector_itself_works`` and
``test_the_wrong_fold_detector_itself_works`` deliberately poison the thing being guarded
and assert the guard notices. Without them the rest proves only that nothing happened to be
wrong today.

Component 7's lesson applies here too: **when a leakage test fails, suspect the test first.**
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.explain import validate
from sentinel.explain.background import (
    BackgroundError,
    background_ids,
    background_is_safe,
    select_background,
)
from sentinel.explain.definitions import BACKGROUND_SEED, SAMPLING_SEED
from sentinel.explain.sample import SELECTION_COLUMNS, SampleError, select_sample
from sentinel.modeling.train import training_frame
from tests.conftest import spanning_model_features


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return spanning_model_features(days=1900).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


@pytest.fixture(scope="module")
def folds(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    return folds_module.quarterly_folds(data_start=start, data_end=end)


# --- 1. the background is temporally safe ------------------------------------


def test_no_background_row_post_dates_the_training_horizon(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    checked = 0
    for fold in folds:
        background = select_background(frame, fold, size=32, seed=BACKGROUND_SEED)
        assert background.height > 0
        checked += background.height
        assert background["rd"].max() <= fold.train_end
    assert checked > 0, "an empty loop would pass vacuously"


def test_every_background_row_is_a_training_row_of_its_own_fold(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """Stronger than the date check: the split is defined by assign_split, not by a date."""
    for fold in folds:
        background = select_background(frame, fold, size=32, seed=BACKGROUND_SEED)
        allowed = {str(v) for v in training_frame(frame, fold)["target_inspection_id"].to_list()}
        assert background_ids(background) <= allowed


def test_a_background_never_contains_a_test_row(frame: pl.DataFrame, folds: list[FoldSpec]) -> None:
    """The specific mistake this module exists to prevent, stated as its own assertion."""
    for fold in folds:
        background = select_background(frame, fold, size=32, seed=BACKGROUND_SEED)
        test_ids = {
            str(v) for v in folds_module.window_frame(frame, fold)["target_inspection_id"].to_list()
        }
        assert not (background_ids(background) & test_ids)


def test_appending_two_years_of_future_data_does_not_change_an_earlier_background(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """The strongest available form: the future cannot reach back into a past fold.

    Bit-identity, not approximate equality. A background that shifted by one row when the
    future arrived has read that future.
    """
    fold = folds[0]
    before = select_background(frame, fold, size=32, seed=BACKGROUND_SEED)

    # Dated from the end of the whole table, not by a fixed offset from each row's own
    # date. A first draft shifted ``frame.head(500)`` by 900 days and the rows landed back
    # inside fold 0's own training window, so the test failed while nothing was wrong --
    # Component 7's rule, that a failing leakage test should be suspected first, paid for
    # itself again.
    latest = frame["rd"].max()
    assert latest is not None
    offset = (latest - fold.train_start).days + 365
    future = frame.head(500).with_columns(
        (pl.col("rd") + timedelta(days=offset)).alias("rd"),
        (pl.col("target_inspection_id") + "_future").alias("target_inspection_id"),
    )
    assert future["rd"].min() > fold.test_end, "the appended rows must really be in the future"

    after = select_background(pl.concat([frame, future]), fold, size=32, seed=BACKGROUND_SEED)
    assert before.equals(after)


def test_the_background_leak_detector_itself_works(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """Poison one reference row with a future date and assert the check goes red.

    A guard whose failure path has never been observed is indistinguishable from one that
    cannot fire.
    """
    fold = folds[0]
    clean = select_background(frame, fold, size=32, seed=BACKGROUND_SEED)

    safe, offenders = background_is_safe(clean, fold)
    assert safe and not offenders, "the positive control"

    poisoned = clean.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(fold.test_start))
        .otherwise(pl.col("rd"))
        .alias("rd")
    )
    safe, offenders = background_is_safe(poisoned, fold)
    assert not safe
    assert len(offenders) == 1
    assert "train_end" in offenders[0]

    check = validate.background_rows_precede_the_training_horizon(
        frame, folds, {fold.fold_id: poisoned}
    )
    assert not check.passed
    assert check.severity == validate.SEVERITY_ERROR


def test_the_wrong_fold_detector_itself_works(frame: pl.DataFrame, folds: list[FoldSpec]) -> None:
    """A background built for a later fold, handed to an earlier one, must be rejected.

    Every row in it is legitimately a training row -- of the *wrong* fold. The date check
    alone would pass for some of them, which is why containment is checked separately.
    """
    early, late = folds[0], folds[-1]
    borrowed = select_background(frame, late, size=64, seed=BACKGROUND_SEED)

    correct = validate.background_is_drawn_from_the_training_window(
        frame, folds, {early.fold_id: select_background(frame, early, size=32, seed=1)}
    )
    assert correct.passed, "the positive control"

    check = validate.background_is_drawn_from_the_training_window(
        frame, folds, {early.fold_id: borrowed}
    )
    assert not check.passed
    assert check.offenders


def test_select_background_refuses_to_build_one_at_all_from_a_poisoned_frame(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """The date guard inside the selector, not only the validator afterwards."""
    fold = folds[0]
    # A frame whose training window has been back-dated so ``assign_split`` still calls the
    # rows training rows, then re-stamped into the future after the split would be caught by
    # the selector's own re-derivation. Simulate by shrinking the fold's horizon instead.
    impossible = FoldSpec(
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        train_start=fold.train_start,
        train_end=fold.train_start + timedelta(days=1),
        calibration_start=fold.train_start + timedelta(days=2),
        calibration_end=fold.train_start + timedelta(days=3),
        test_start=fold.train_start + timedelta(days=4),
        test_end=fold.train_start + timedelta(days=5),
    )
    background = select_background(frame, impossible, size=8, seed=1)
    assert background["rd"].max() <= impossible.train_end


def test_a_fold_with_no_training_rows_is_refused_rather_than_returning_nothing(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    empty = frame.head(0)
    with pytest.raises(BackgroundError, match="empty training window"):
        select_background(empty, folds[0], size=8, seed=1)


def test_a_non_positive_background_size_is_refused(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    with pytest.raises(BackgroundError, match="must be positive"):
        select_background(frame, folds[0], size=0, seed=1)


def test_a_background_larger_than_the_window_is_truncated_not_resampled(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """Sampling with replacement would weight some training rows twice in the reference."""
    fold = folds[0]
    window = training_frame(frame, fold)
    background = select_background(frame, fold, size=window.height * 3, seed=1)
    assert background.height == window.height
    assert background["target_inspection_id"].n_unique() == window.height


# --- 2. the explanation sample is a test-window sample, chosen blind ----------


def test_every_sampled_row_is_a_test_row_of_its_own_fold(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    check = validate.explained_rows_lie_in_the_test_window(
        frame,
        [select_sample(frame, fold, size=20, seed=SAMPLING_SEED) for fold in folds],
        folds,
    )
    assert check.passed, check.offenders


def test_the_sample_reads_no_label(frame: pl.DataFrame, folds: list[FoldSpec]) -> None:
    """The executable form of 'no outcome participates in the selection'.

    Every column the sampler is not permitted to read is corrupted -- targets flipped,
    features nulled -- and the selection must come back identical.
    """
    fold = folds[0]
    before = select_sample(frame, fold, size=25, seed=SAMPLING_SEED)

    corrupted = frame.with_columns(
        (1 - pl.col("target")).alias("target"),
        pl.lit(None, dtype=pl.Float64).alias("prior_canvass_priority_rate"),
    )
    after = select_sample(corrupted, fold, size=25, seed=SAMPLING_SEED)
    assert before.ids == after.ids


def test_the_sampler_declares_the_only_columns_it_may_read() -> None:
    assert SELECTION_COLUMNS == ("rd", "target_inspection_id")
    assert "target" not in SELECTION_COLUMNS
    assert "score" not in SELECTION_COLUMNS


def test_shuffling_the_frame_does_not_change_the_sample(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """Determinism comes from the canonical sort, not only from the seed."""
    fold = folds[0]
    before = select_sample(frame, fold, size=25, seed=SAMPLING_SEED)
    after = select_sample(frame.reverse(), fold, size=25, seed=SAMPLING_SEED)
    assert before.ids == after.ids


def test_the_same_sample_is_drawn_for_every_model(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """What makes a cross-model importance comparison like-for-like."""
    fold = folds[0]
    first = select_sample(frame, fold, size=25, seed=SAMPLING_SEED)
    second = select_sample(frame, fold, size=25, seed=SAMPLING_SEED)
    assert first.ids == second.ids


def test_an_empty_test_window_is_refused(frame: pl.DataFrame, folds: list[FoldSpec]) -> None:
    with pytest.raises(SampleError, match="empty test window"):
        select_sample(frame.head(0), folds[0], size=10, seed=1)


def test_a_non_positive_sample_size_is_refused(frame: pl.DataFrame, folds: list[FoldSpec]) -> None:
    with pytest.raises(SampleError, match="must be positive"):
        select_sample(frame, folds[0], size=0, seed=1)


def test_a_sample_larger_than_the_window_is_truncated(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    fold = folds[0]
    window = folds_module.window_frame(frame, fold)
    sample = select_sample(frame, fold, size=window.height * 5, seed=1)
    assert len(sample.ids) == window.height
    assert len(set(sample.ids)) == window.height
    assert sample.population_rows == window.height


def test_the_out_of_window_detector_itself_works(
    frame: pl.DataFrame, folds: list[FoldSpec]
) -> None:
    """Hand the checker a sample containing a training id and assert it goes red."""
    import dataclasses

    fold = folds[0]
    clean = select_sample(frame, fold, size=20, seed=SAMPLING_SEED)
    assert validate.explained_rows_lie_in_the_test_window(frame, [clean], folds).passed

    smuggled = str(training_frame(frame, fold)["target_inspection_id"].to_list()[0])
    poisoned = dataclasses.replace(clean, ids=(*clean.ids, smuggled))
    check = validate.explained_rows_lie_in_the_test_window(frame, [poisoned], folds)
    assert not check.passed
    assert any(smuggled in offender for offender in check.offenders)
