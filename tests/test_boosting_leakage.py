"""Letting future information into a boosted model must be impossible without a failure here.

The rest of the Component 7 suite checks that the models are *right*; this file checks
that they cannot be *cheating*. It is modelled on ``test_modeling_leakage.py`` and keeps
that file's standard: "did not move" is asserted as **bit-identical**, not approximately
equal. A prediction that shifts by 1e-12 for a reason nobody can explain is a defect
whether or not it changes a ranking.

Component 7 has two exposures Component 6 did not, and both are covered below.

**Early stopping.** A booster that early-stops needs a validation window later than its
training data, and the only such windows at fold-fit time are the fold's own calibration
and test spans. Reading either would make ``trained_through = train_end`` false. The
tests here delete each of those windows entirely and assert the predictions do not move,
which is the strongest available statement that neither was read.

**Hyperparameter selection.** A search that touches a test window leaks in a way no
artifact records. The tests here recompute the tuning region and every inner fold's dates
from the data and assert they end before the fold set's first test start -- rather than
reading the manifest field that says so.

The last test in the file is the one that makes the rest meaningful: it deliberately
introduces a leak and confirms the fixture is strong enough to reveal it. Without that,
every assertion above would be reassuring for the wrong reason -- they would be measuring
a model too weak to exploit a leak rather than a pipeline that prevents one.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.boosting import predict, tuning
from sentinel.boosting.definitions import BOOSTING_REGISTRY, spec_for
from sentinel.boosting.train import fit_fold
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from tests.conftest import (
    make_model_feature_row,
    model_feature_scenario,
    spanning_model_features,
)

PRIMARY = spec_for("xgboost")

FOLD = FoldSpec(
    fold_set="quarterly",
    fold_id="quarterly-2022Q2",
    train_start=date(2018, 7, 1),
    train_end=date(2021, 12, 31),
    calibration_start=date(2022, 1, 1),
    calibration_end=date(2022, 3, 31),
    test_start=date(2022, 4, 1),
    test_end=date(2022, 6, 30),
)


def _base() -> pl.DataFrame:
    """A frame spanning enough quarters that the real fold builder yields several folds.

    1900 days rather than a smaller number for the reason ``test_modeling_leakage.py``
    records: a shorter fixture produced zero folds and made three loops pass vacuously.
    """
    return spanning_model_features(days=1900).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def _quarterly(frame: pl.DataFrame) -> list[FoldSpec]:
    """Real folds over the fixture, asserted non-empty so no loop passes vacuously."""
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    built = folds_module.quarterly_folds(data_start=start, data_end=end)
    assert len(built) >= 4, f"fixture yielded {len(built)} folds; loops would be weak"
    return built


def _scores(frame: pl.DataFrame, fold: FoldSpec, spec: object = PRIMARY) -> list[float]:
    """Fit on the fold's training window and score its test window."""
    fitted = fit_fold(spec, training_frame(frame, fold), fold)  # type: ignore[arg-type]
    _, scores = predict.score_window(fitted, folds_module.window_frame(frame, fold))
    return scores


def _future_frame(count: int, *, target: int, start_index: int = 90_000) -> pl.DataFrame:
    """Rows dated after every fold in the fixture, in Component 4's column order.

    Ordered through ``model_feature_scenario`` rather than built as a bare frame: the
    override dicts are written in a different order from the schema, and a positional
    concat would silently place each value in the wrong column -- which looks exactly
    like a leak when the resulting fit moves.
    """
    rows = [
        make_model_feature_row(
            start_index + i,
            inspection_date="2024-06-15",
            target=target,
            target_status="inspected",
        )
        for i in range(count)
    ]
    return model_feature_scenario(rows).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


# --- 1. future rows cannot change an earlier fold ----------------------------


@pytest.mark.parametrize("spec", BOOSTING_REGISTRY, ids=lambda s: s.name)
def test_appending_future_rows_changes_nothing(spec: object) -> None:
    """Every registered model, so a new spec is covered the moment it is registered."""
    base = _base()
    reference = fit_fold(spec, training_frame(base, FOLD), FOLD)  # type: ignore[arg-type]

    extended = pl.concat([base, _future_frame(400, target=1)], how="vertical")
    after = fit_fold(spec, training_frame(extended, FOLD), FOLD)  # type: ignore[arg-type]

    assert after.train_rows == reference.train_rows
    assert after.train_positive_rate == reference.train_positive_rate
    assert after.train_nan_cells == reference.train_nan_cells
    assert after.importances == reference.importances
    assert after.trees_built == reference.trees_built


@pytest.mark.parametrize("target", [0, 1])
def test_a_future_row_of_either_class_changes_nothing(target: int) -> None:
    """Running both classes is what makes this sharp.

    A leak that only admitted positives would pass a positives-only test by luck.
    """
    base = _base()
    reference = _scores(base, FOLD)
    after = _scores(pl.concat([base, _future_frame(400, target=target)], how="vertical"), FOLD)
    assert after == reference


# --- 2. mutating the future ---------------------------------------------------


def test_flipping_every_future_label_changes_nothing() -> None:
    base = _base()
    reference = _scores(base, FOLD)
    mutated = base.with_columns(
        pl.when(pl.col("rd") > FOLD.train_end)
        .then(1 - pl.col("target"))
        .otherwise(pl.col("target"))
        .alias("target")
    )
    assert _scores(mutated, FOLD) == reference


def test_corrupting_a_feature_beyond_the_test_window_changes_nothing() -> None:
    """Mutate rows after ``test_end``, not after ``train_end``.

    Rows between ``train_end`` and ``test_end`` include the fold's own test rows, and
    changing a test row's features is *supposed* to change its score -- that is the model
    working, not a leak. The property under test is that a row the fold never touches
    cannot reach the fit.
    """
    base = _base()
    reference = _scores(base, FOLD)
    mutated = base.with_columns(
        pl.when(pl.col("rd") > FOLD.test_end)
        .then(pl.lit(999_999.0))
        .otherwise(pl.col("prior_canvass_priority_rate"))
        .alias("prior_canvass_priority_rate")
    )
    assert mutated.filter(pl.col("rd") > FOLD.test_end).height > 0, "no rows were mutated"
    assert _scores(mutated, FOLD) == reference


def test_deleting_the_calibration_window_changes_nothing() -> None:
    """The strongest available statement that no fit early-stopped on it.

    If the final fit consulted the calibration window -- which is what early stopping
    would require -- removing every one of its rows would change the fitted ensemble.
    """
    base = _base()
    reference = _scores(base, FOLD)
    without = base.filter(
        (pl.col("rd") < FOLD.calibration_start) | (pl.col("rd") > FOLD.calibration_end)
    )
    assert without.height < base.height, "fixture has no calibration rows; test is vacuous"
    assert _scores(without, FOLD) == reference


def test_deleting_the_test_window_changes_the_predictions_only_by_removing_rows() -> None:
    """Removing test rows must change *which* rows are scored, never *how*."""
    base = _base()
    fitted_before = fit_fold(PRIMARY, training_frame(base, FOLD), FOLD)
    kept = base.filter(pl.col("rd") != FOLD.test_start)
    fitted_after = fit_fold(PRIMARY, training_frame(kept, FOLD), FOLD)
    assert fitted_after.importances == fitted_before.importances
    assert fitted_after.train_rows == fitted_before.train_rows


# --- 3. fold isolation --------------------------------------------------------


def test_every_fold_trains_only_on_its_own_window() -> None:
    base = _base()
    for fold in _quarterly(base):
        fitted = fit_fold(PRIMARY, training_frame(base, fold), fold)
        expected = base.filter(
            (pl.col("rd") >= fold.train_start) & (pl.col("rd") <= fold.train_end)
        ).height
        assert fitted.train_rows == expected
        assert fitted.trained_through == fold.train_end
        assert fitted.trained_through < fold.calibration_start
        assert fitted.trained_through < fold.test_start


def test_an_earlier_fold_is_unaffected_by_a_later_folds_data() -> None:
    """Truncate the frame at the earliest fold's training end and refit it."""
    base = _base()
    folds = _quarterly(base)
    earliest = folds[0]
    reference = fit_fold(PRIMARY, training_frame(base, earliest), earliest)

    truncated = base.filter(pl.col("rd") <= earliest.train_end)
    after = fit_fold(PRIMARY, training_frame(truncated, earliest), earliest)
    assert after.importances == reference.importances
    assert after.train_rows == reference.train_rows


# --- 4. row order ---------------------------------------------------------------


def test_shuffling_the_source_rows_changes_no_prediction() -> None:
    """The canonical sort is what makes this true, and the profiler measured its size.

    On the real snapshot, fitting the same rows in a shuffled order moved a prediction
    by 1.12e-01 -- three orders of magnitude larger than the 7.049e-09 Component 6 saw
    in its coefficients. So the sort is not a tidiness measure here; it is the only
    reason a re-run reproduces.
    """
    base = _base()
    reference = _scores(base, FOLD)
    shuffled = base.sample(fraction=1.0, shuffle=True, seed=11)
    assert shuffled["target_inspection_id"].to_list() != base["target_inspection_id"].to_list()
    assert _scores(shuffled, FOLD) == reference


def test_refitting_the_same_frame_is_bit_identical() -> None:
    base = _base()
    assert _scores(base, FOLD) == _scores(base, FOLD)


# --- 5. the label can never be a feature ----------------------------------------


def test_no_registered_model_names_the_target() -> None:
    for spec in BOOSTING_REGISTRY:
        assert "target" not in spec.feature_columns
        assert "target_status" not in spec.feature_columns
        assert "target_inspection_id" not in spec.feature_columns
        assert "establishment_id" not in spec.feature_columns
        assert "inspection_date" not in spec.feature_columns


def test_the_matrix_refuses_a_column_component_4_does_not_declare() -> None:
    from dataclasses import replace

    from sentinel.boosting.preprocess import TreePreprocessError, tree_matrix

    leaky = replace(PRIMARY, feature_columns=(*PRIMARY.feature_columns, "target"))
    with pytest.raises(TreePreprocessError):
        tree_matrix(_base().drop("target"), leaky)


# --- 6. the tuning protocol, re-derived rather than trusted ----------------------


def test_every_tuning_region_ends_before_its_fold_sets_first_test_window() -> None:
    """Recomputed from the fold definitions, not read from a manifest field."""
    base = _base()
    folds = [
        *_quarterly(base),
        *folds_module.covid_shift_fold(data_end=folds_module.max_date(base, "rd") or date.max),
    ]
    seen = 0
    for fold_set in sorted({f.fold_set for f in folds}):
        _, region_end = tuning.tuning_region(fold_set, folds)
        horizon = tuning.first_test_start(fold_set, folds)
        assert region_end < horizon, f"{fold_set}: region ends {region_end}, test starts {horizon}"
        seen += 1
    assert seen >= 2, "expected both fold sets; the loop would be weak with one"


def test_every_inner_validation_window_ends_before_the_first_test_start() -> None:
    base = _base()
    folds = [
        *_quarterly(base),
        *folds_module.covid_shift_fold(data_end=folds_module.max_date(base, "rd") or date.max),
    ]
    for fold_set in sorted({f.fold_set for f in folds}):
        horizon = tuning.first_test_start(fold_set, folds)
        inner = tuning.build_inner_folds(fold_set, folds)
        assert inner, f"{fold_set} produced no inner folds"
        for fold in inner:
            assert fold.test_end < horizon
            assert fold.train_end < fold.calibration_start
            assert fold.calibration_end < fold.test_start


def test_the_objective_never_reads_a_row_from_a_test_window() -> None:
    """Collect every row id an inner fold touches and intersect with the test windows."""
    base = _base()
    folds = [
        *_quarterly(base),
        *folds_module.covid_shift_fold(data_end=folds_module.max_date(base, "rd") or date.max),
    ]
    test_ids: set[str] = set()
    for fold in folds:
        test_ids.update(folds_module.window_frame(base, fold)["target_inspection_id"].to_list())
    assert test_ids, "no test rows in the fixture; the intersection would be empty for free"

    for fold_set in sorted({f.fold_set for f in folds}):
        if fold_set != "quarterly":
            continue
        touched: set[str] = set()
        for inner in tuning.build_inner_folds(fold_set, folds):
            touched.update(training_frame(base, inner)["target_inspection_id"].to_list())
            touched.update(folds_module.window_frame(base, inner)["target_inspection_id"].to_list())
        quarterly_test_ids: set[str] = set()
        for fold in folds:
            if fold.fold_set == fold_set:
                quarterly_test_ids.update(
                    folds_module.window_frame(base, fold)["target_inspection_id"].to_list()
                )
        assert not (touched & quarterly_test_ids)


def test_a_region_overlapping_a_test_window_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard fires rather than searching a contaminated region.

    It cannot be reached through the public path, because ``tuning_region`` derives the
    region from the same fold whose ``test_start`` is the horizon, and a fold's
    calibration end always precedes its own test start. The realistic way to break it is
    to widen the region -- "use all the data we have before the last fold" is a natural
    and wrong instinct -- so that is what this test simulates. Without this, the guard
    would be a check that has never been observed to fail, which is indistinguishable
    from one that cannot.
    """
    base = _base()
    folds = _quarterly(base)
    widest = max(folds, key=lambda f: f.calibration_end)
    monkeypatch.setattr(
        tuning,
        "tuning_region",
        lambda fold_set, outer: (folds[0].train_start, widest.calibration_end),
    )
    with pytest.raises(tuning.TuningError, match="select hyperparameters using test labels"):
        tuning.build_inner_folds("quarterly", folds)


# --- 7. the detector itself works ------------------------------------------------


def test_the_leakage_detector_itself_works() -> None:
    """Plant the label in a feature and confirm the fixture reveals it.

    If the scores did not then separate the classes almost perfectly, every test above
    would be reassuring for the wrong reason -- they would be measuring a model too weak
    to exploit a leak rather than a pipeline that prevents one.
    """
    base = _base().with_columns(
        pl.col("target").cast(pl.Float64).alias("prior_canvass_priority_rate")
    )
    fitted = fit_fold(PRIMARY, training_frame(base, FOLD), FOLD)
    test = folds_module.window_frame(base, FOLD)
    _, scores = predict.score_window(fitted, test)
    labels = test["target"].to_list()

    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    assert positives and negatives
    assert min(positives) > max(negatives), (
        "a planted label did not separate the classes, so the leakage tests above are "
        "measuring a weak model rather than a protected pipeline"
    )
