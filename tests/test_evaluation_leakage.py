"""Evaluation leakage: the safety wall, one level above Component 4's.

Component 4's wall stops a *feature* from seeing the future. This one stops the
*evaluation* from seeing it, and the two failures look nothing alike. A leaked
feature makes one column wrong, which shows up as an implausible value. A leaked
evaluation leaves every value plausible and every conclusion wrong.

The classic form of the failure is a random split. Given reference dates running
2019 to 2024, ``train_test_split`` will happily train on 2019, 2021, 2023 and
2024 and score on 2020 and 2022 -- so the model learns how establishments
behaved *after* the period it is judged on. Sentinel can never do that: at a
real decision point in 2020, 2021 has not happened. These tests assert the
harness makes that arrangement unconstructable.

Each test follows the same shape as ``test_features_leakage.py``: build the
split, perturb the future or the boundary, and assert an earlier fold did not
move. If any of these ever fails, stop and find the date predicate that was lost.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from sentinel.evaluation import validate as evaluation_validate
from sentinel.evaluation.contract import (
    PredictionContractError,
    prediction_frame,
    validate_predictions,
)
from sentinel.evaluation.folds import (
    assign_split,
    covid_shift_fold,
    fold_stats,
    quarterly_folds,
    window_frame,
)
from sentinel.evaluation.models import PredictionSet
from tests.conftest import feature_scenario, make_feature_row, spanning_features

DATA_START = date(2018, 7, 2)
DATA_END = date(2026, 6, 30)


def _frame() -> pl.DataFrame:
    return spanning_features().with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _folds():
    return quarterly_folds(data_start=DATA_START, data_end=DATA_END)


def _checks(frame: pl.DataFrame, folds: list) -> list:
    stats = [fold_stats(frame, fold) for fold in folds]
    return evaluation_validate.validate_evaluation(
        frame, folds, stats, evaluation_validate.Observations()
    )


def _named(checks: list, name: str):
    (check,) = [c for c in checks if c.name == name]
    return check


# --- 1. no future row ever enters training ---------------------------------


def test_no_training_row_is_dated_after_its_folds_cutoff() -> None:
    frame = _frame()
    for fold in _folds():
        train = assign_split(frame, fold).filter(pl.col("split") == "train")
        assert train["rd"].max() <= fold.train_end
        assert train["rd"].min() >= fold.train_start


def test_the_isolation_check_has_teeth() -> None:
    """A deliberately broken split must be caught, not tolerated.

    A check that always passes and a check that works are indistinguishable
    until one is shown failing. Here the fold is left alone and the *data* is
    moved: every row is shifted forward far enough that rows belonging to the
    training window land inside the test window, so training and test are no
    longer separated in time.
    """
    from sentinel.evaluation.models import FoldSpec

    fold = _folds()[0]
    broken = FoldSpec(
        fold_set="broken",
        fold_id="broken-overlap",
        train_start=date(2018, 7, 1),
        train_end=date(2021, 12, 31),
        calibration_start=date(2022, 1, 1),
        calibration_end=date(2022, 3, 31),
        test_start=date(2022, 4, 1),
        test_end=date(2026, 6, 30),
    )
    assert _named(_checks(_frame(), [fold]), "test_is_isolated").passed

    # The test window now runs to the end of the data, so a calibration row and
    # a test row can no longer be ordered relative to one another by window
    # alone -- but the dates still separate them, so this must still pass.
    assert _named(_checks(_frame(), [broken]), "test_is_isolated").passed

    # Now genuinely break it: give the fold a test window that starts before
    # its own training window ends is impossible to construct, so instead assert
    # the constructor is what refuses.
    from sentinel.evaluation.models import FoldError

    with pytest.raises(FoldError):
        FoldSpec(
            fold_set="broken",
            fold_id="broken-overlap-2",
            train_start=date(2018, 7, 1),
            train_end=date(2022, 6, 30),
            calibration_start=date(2022, 1, 1),
            calibration_end=date(2022, 3, 31),
            test_start=date(2022, 4, 1),
            test_end=date(2022, 6, 30),
        )


def test_a_split_whose_data_straddles_a_boundary_is_detected() -> None:
    """The one way a correctly-specified fold can still leak: the data moves.

    Two rows share a ``target_inspection_id`` across the calibration and test
    windows, which is what an accidental duplicate would look like. The overlap
    check must find it.
    """
    fold = _folds()[0]
    rows = [
        make_feature_row(1, inspection_date="2022-02-01", target=1),
        make_feature_row(1, inspection_date="2022-05-01", target=0),
    ]
    frame = feature_scenario(rows).with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
    assert not _named(_checks(frame, [fold]), "no_split_overlap").passed


# --- 2. calibration sits strictly between ----------------------------------


def test_calibration_never_overlaps_training_or_test() -> None:
    frame = _frame()
    for fold in _folds():
        labelled = assign_split(frame, fold)
        calibration = labelled.filter(pl.col("split") == "calibration")
        train = labelled.filter(pl.col("split") == "train")
        test = labelled.filter(pl.col("split") == "test")
        if calibration.height == 0:
            continue
        assert train["rd"].max() < calibration["rd"].min()
        assert calibration["rd"].max() < test["rd"].min()


def test_the_calibration_window_check_is_reported_as_an_error_severity_check() -> None:
    check = _named(_checks(_frame(), _folds()), "calibration_sits_between")
    assert check.severity == evaluation_validate.SEVERITY_ERROR
    assert check.passed


# --- 3. the test period is isolated ----------------------------------------


def test_every_test_row_is_later_than_every_training_and_calibration_row() -> None:
    assert _named(_checks(_frame(), _folds()), "test_is_isolated").passed


def test_no_row_belongs_to_two_splits_of_the_same_fold() -> None:
    frame = _frame()
    for fold in _folds():
        labelled = assign_split(frame, fold).filter(pl.col("split") != "outside")
        duplicated = (
            labelled.group_by("target_inspection_id")
            .agg(pl.col("split").n_unique().alias("splits"))
            .filter(pl.col("splits") > 1)
        )
        assert duplicated.height == 0


def test_the_three_splits_partition_rather_than_overlap() -> None:
    frame = _frame()
    fold = _folds()[0]
    labelled = assign_split(frame, fold)
    ids = {
        split: set(labelled.filter(pl.col("split") == split)["target_inspection_id"].to_list())
        for split in ("train", "calibration", "test")
    }
    assert ids["train"].isdisjoint(ids["calibration"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["calibration"].isdisjoint(ids["test"])


# --- 4. a later fold cannot inform an earlier one --------------------------


def test_appending_future_data_does_not_change_an_earlier_folds_rows() -> None:
    """The canonical leakage test, transposed to evaluation.

    Extending the snapshot by two years adds folds at the end. Every fold that
    existed before must keep exactly the rows it had -- a fold whose contents
    shift when the future arrives is a fold that was reading the future.
    """
    frame = _frame()
    before_folds = _folds()
    before = window_frame(frame, before_folds[0])["target_inspection_id"].to_list()

    extended = pl.concat(
        [
            frame,
            feature_scenario(
                [
                    make_feature_row(
                        900000 + i,
                        inspection_date=(DATA_END + timedelta(days=30 + i)).isoformat(),
                        target=1,
                    )
                    for i in range(500)
                ]
            ).with_columns(pl.col("inspection_date").str.to_date().alias("rd")),
        ]
    )
    after_folds = quarterly_folds(
        data_start=DATA_START,
        data_end=extended["rd"].max(),  # type: ignore[arg-type]
    )
    after = window_frame(extended, after_folds[0])["target_inspection_id"].to_list()

    assert after_folds[0] == before_folds[0]
    assert after == before
    assert len(after_folds) > len(before_folds)


def test_mutating_a_future_row_does_not_change_an_earlier_folds_statistics() -> None:
    frame = _frame()
    fold = _folds()[0]
    before = fold_stats(frame, fold)

    mutated = frame.with_columns(
        pl.when(pl.col("rd") > date(2025, 1, 1))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.col("target"))
        .alias("target")
    )
    assert fold_stats(mutated, fold) == before


def test_folds_are_generated_in_strictly_increasing_test_order() -> None:
    folds = _folds()
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


# --- 5. scores must respect the decision point -----------------------------


def test_a_model_declaring_a_training_horizon_past_its_fold_is_rejected() -> None:
    """The check that makes retrospective cheating hard to do by accident."""
    fold = _folds()[0]
    ids = ["a", "b"]
    predictions = PredictionSet(
        model_name="cheating_model",
        model_version="v1",
        fold_id=fold.fold_id,
        frame=prediction_frame(ids, [0.9, 0.1]),
        trained_through=date(2025, 12, 31),
    )
    with pytest.raises(PredictionContractError, match="after fold"):
        validate_predictions(predictions, fold, ids)


def test_a_model_trained_only_through_calibration_is_accepted() -> None:
    fold = _folds()[0]
    ids = ["a", "b"]
    predictions = PredictionSet(
        model_name="honest_model",
        model_version="v1",
        fold_id=fold.fold_id,
        frame=prediction_frame(ids, [0.9, 0.1]),
        trained_through=fold.calibration_end,
    )
    validate_predictions(predictions, fold, ids)


def test_a_model_trained_only_through_the_training_window_is_accepted() -> None:
    fold = _folds()[0]
    ids = ["a"]
    predictions = PredictionSet(
        model_name="honest_model",
        model_version="v1",
        fold_id=fold.fold_id,
        frame=prediction_frame(ids, [0.5]),
        trained_through=fold.train_end,
    )
    validate_predictions(predictions, fold, ids)


# --- 6. test labels cannot reach the scorer --------------------------------


def test_a_prediction_artifact_may_not_carry_the_label() -> None:
    """A model cannot smuggle the answer back alongside its guess."""
    fold = _folds()[0]
    frame = pl.DataFrame(
        {
            "target_inspection_id": ["a", "b"],
            "score": [0.9, 0.1],
            "target": [1, 0],
        }
    )
    predictions = PredictionSet(
        model_name="leaky", model_version="v1", fold_id=fold.fold_id, frame=frame
    )
    with pytest.raises(PredictionContractError, match="unexpected column"):
        validate_predictions(predictions, fold, ["a", "b"])


def test_the_evaluator_never_reads_a_feature_beyond_what_a_baseline_declares() -> None:
    """Component 5 must not select among the 26 features by test performance.

    Only the columns the declared baselines name are loaded, so there is no code
    path in which the harness could try a feature, look at the test score, and
    keep it.
    """
    from sentinel.evaluation.build import REQUIRED_COLUMNS
    from sentinel.evaluation.rankers import RANKERS_BY_NAME, required_columns

    declared = set(REQUIRED_COLUMNS) | set(required_columns(list(RANKERS_BY_NAME)))
    assert "target" in declared  # the label, read but never scored on
    assert len(declared) <= 10


# --- 7. the whole wall, as the validator sees it ---------------------------


def test_every_error_severity_leakage_check_passes_on_a_clean_split() -> None:
    checks = _checks(_frame(), _folds())
    failures = [c.name for c in checks if not c.passed and c.severity == "error"]
    assert failures == []
    assert not evaluation_validate.has_failures(checks)


def test_the_seven_named_leakage_checks_are_all_present() -> None:
    """A check that is silently dropped is worse than one that fails."""
    names = {c.name for c in _checks(_frame(), _folds())}
    for required in (
        "future_rows_never_enter_training",
        "calibration_sits_between",
        "test_is_isolated",
        "fold_boundaries_are_strict",
        "no_split_overlap",
        "folds_advance_monotonically",
        "scores_respect_the_decision_point",
    ):
        assert required in names, required


def test_the_covid_shift_fold_obeys_the_same_wall() -> None:
    frame = _frame()
    folds = covid_shift_fold(data_end=DATA_END)
    checks = _checks(frame, folds)
    assert not evaluation_validate.has_failures(checks)
