"""Temporal leakage: the safety wall for fitted models.

This file exists so that letting future information into a model is impossible without a
test failing. It is separate from the other modeling tests deliberately -- the rest check
that the models are *right*, these check that they cannot be *cheating*.

The distinction that matters: Component 4's wall catches a feature that saw the future,
and that makes one column wrong. This wall catches a *model* that saw the future, and
that makes every number right and every conclusion wrong. The second is far harder to
notice, because nothing looks broken.

Each test follows the same shape: fit a fold, perturb the future or the present, refit,
and assert the earlier fold did not move. If any of these fails, stop and find the filter
that lost its date predicate.

Note that "did not move" is asserted as **bit-identical**, not approximately equal. A
fitted coefficient that shifts by 1e-12 for a reason nobody can explain is a defect
whether or not it changes a ranking.
"""

from __future__ import annotations

import dataclasses
import random
from datetime import date

import polars as pl
import pytest

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.features.definitions import FEATURE_COLUMNS, LABEL_COLUMNS
from sentinel.modeling import predict, train
from sentinel.modeling.definitions import MODEL_REGISTRY, MODELS_BY_NAME
from tests.conftest import make_model_feature_row, model_feature_scenario, spanning_model_features

PRIMARY = MODELS_BY_NAME["logistic_regression"]

EARLY_FOLD = FoldSpec(
    fold_set="quarterly",
    fold_id="quarterly-2022Q2",
    train_start=date(2018, 7, 1),
    train_end=date(2021, 12, 31),
    calibration_start=date(2022, 1, 1),
    calibration_end=date(2022, 3, 31),
    test_start=date(2022, 4, 1),
    test_end=date(2022, 6, 30),
)


def _prepared(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _base() -> pl.DataFrame:
    """Long enough that ``quarterly_folds`` actually returns folds.

    1,400 days ends 2022-05-02, before the first fold's test window closes on
    2022-06-30, so the fold list would be empty and every loop over it would pass
    vacuously. 1,900 days reaches 2023-08 and yields five real folds; the helper
    below asserts the count rather than trusting it.
    """
    return _prepared(spanning_model_features(days=1900))


def _quarterly(frame: pl.DataFrame) -> list[FoldSpec]:
    """Real folds over the fixture, asserted non-empty so no loop passes vacuously."""
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    folds = folds_module.quarterly_folds(data_start=start, data_end=end)
    assert len(folds) >= 4, f"fixture yielded {len(folds)} folds; loops would be weak"
    return folds


def _fit(frame: pl.DataFrame, fold: FoldSpec = EARLY_FOLD, spec=PRIMARY):  # type: ignore[no-untyped-def]
    return train.fit_fold(spec, train.training_frame(frame, fold), fold)


def _future_rows(count: int, *, start: str, target: int) -> list[dict[str, object]]:
    """Rows dated well after the early fold's test window."""
    first = date.fromisoformat(start)
    return [
        make_model_feature_row(
            900_000 + index,
            inspection_date=(first.replace(day=1 + index % 27)).isoformat(),
            target=target,
        )
        for index in range(count)
    ]


# --- 1. a future row cannot change an earlier fold --------------------------


def test_appending_future_rows_changes_no_coefficient() -> None:
    """The canonical leakage test. Add two years of 2024-25 data; the 2022Q2 fit must
    not move by a single bit."""
    base = _base()
    before = _fit(base)

    future = model_feature_scenario(
        [*base.drop("rd").to_dicts(), *_future_rows(400, start="2024-06-01", target=1)]
    )
    after = _fit(_prepared(future))

    assert after.train_rows == before.train_rows
    assert after.coefficients == before.coefficients
    assert after.intercept == before.intercept
    assert after.imputed_values == before.imputed_values
    assert after.scaler_mean == before.scaler_mean
    assert after.scaler_scale == before.scaler_scale


def test_appending_future_rows_changes_no_score() -> None:
    base = _base()
    test_window = folds_module.window_frame(base, EARLY_FOLD)
    before = predict.score_window(_fit(base), test_window)

    future = model_feature_scenario(
        [*base.drop("rd").to_dicts(), *_future_rows(400, start="2025-01-01", target=0)]
    )
    prepared = _prepared(future)
    after = predict.score_window(_fit(prepared), folds_module.window_frame(prepared, EARLY_FOLD))

    assert after == before


def test_a_future_row_of_the_opposite_class_changes_nothing() -> None:
    """Appending only positives and only negatives must both leave the fit untouched.

    If the training filter leaked, these two would move the coefficients in opposite
    directions -- so running both is what makes the test sharp.
    """
    base = _base()
    reference = _fit(base)
    for target in (0, 1):
        future = model_feature_scenario(
            [*base.drop("rd").to_dicts(), *_future_rows(300, start="2024-09-01", target=target)]
        )
        assert _fit(_prepared(future)).coefficients == reference.coefficients


@pytest.mark.parametrize("spec", MODEL_REGISTRY, ids=lambda s: s.name)
def test_every_registered_model_is_immune_to_future_rows(spec: object) -> None:
    base = _base()
    before = _fit(base, spec=spec)
    future = model_feature_scenario(
        [*base.drop("rd").to_dicts(), *_future_rows(250, start="2024-03-01", target=1)]
    )
    assert _fit(_prepared(future), spec=spec).coefficients == before.coefficients


# --- 2. mutating a future outcome cannot change an earlier fold -------------


def test_flipping_every_future_label_changes_nothing() -> None:
    """A model that reached forward for its labels would move a lot here."""
    base = _base()
    before = _fit(base)

    mutated = base.with_columns(
        pl.when(pl.col("rd") > EARLY_FOLD.train_end)
        .then(1 - pl.col("target"))
        .otherwise(pl.col("target"))
        .cast(pl.Int8)
        .alias("target")
    )
    after = _fit(mutated)
    assert after.coefficients == before.coefficients
    assert after.intercept == before.intercept


def test_mutating_a_future_feature_changes_nothing() -> None:
    """Preprocessing statistics must come from the training window only, so a wild
    value in the calibration or test period cannot move an imputation median."""
    base = _base()
    before = _fit(base)

    mutated = base.with_columns(
        pl.when(pl.col("rd") > EARLY_FOLD.train_end)
        .then(pl.lit(999_999))
        .otherwise(pl.col("days_since_last_canvass"))
        .cast(pl.Int32)
        .alias("days_since_last_canvass")
    )
    after = _fit(mutated)
    assert after.imputed_values == before.imputed_values
    assert after.scaler_mean == before.scaler_mean
    assert after.coefficients == before.coefficients


def test_removing_the_calibration_window_entirely_changes_nothing() -> None:
    """Component 6 must not read the calibration window at all. If it did, deleting
    that window would move the fit."""
    base = _base()
    before = _fit(base)

    without = base.filter(
        ~pl.col("rd").is_between(EARLY_FOLD.calibration_start, EARLY_FOLD.calibration_end)
    )
    assert without.height < base.height
    after = _fit(without)
    assert after.coefficients == before.coefficients
    assert after.trained_through == EARLY_FOLD.train_end


def test_removing_the_test_window_entirely_changes_nothing() -> None:
    base = _base()
    before = _fit(base)
    without = base.filter(~pl.col("rd").is_between(EARLY_FOLD.test_start, EARLY_FOLD.test_end))
    assert without.height < base.height
    assert _fit(without).coefficients == before.coefficients


# --- 3. fold isolation, asserted directly ----------------------------------


def test_training_rows_never_overlap_calibration_or_test() -> None:
    frame = _base()
    for fold in _quarterly(frame):
        assigned = folds_module.assign_split(frame, fold)
        training = set(
            assigned.filter(pl.col("split") == "train")["target_inspection_id"].to_list()
        )
        later = set(
            assigned.filter(pl.col("split").is_in(["calibration", "test"]))[
                "target_inspection_id"
            ].to_list()
        )
        assert training & later == set(), fold.fold_id


def test_fold_windows_are_strictly_ordered() -> None:
    frame = _base()
    for fold in _quarterly(frame):
        assert fold.train_end < fold.calibration_start
        assert fold.calibration_end < fold.test_start


def test_a_later_fold_trains_on_strictly_more_rows() -> None:
    """The expanding window is what makes a fold sequence a retraining simulation."""
    frame = _base()
    folds = _quarterly(frame)
    counts = [train.training_frame(frame, fold).height for fold in folds]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)
    assert all(fold.train_start == folds[0].train_start for fold in folds)


def test_an_earlier_folds_fit_is_unaffected_by_a_later_folds_existence() -> None:
    """Fitting fold 2 must not disturb fold 1 -- no shared mutable state."""
    frame = _base()
    folds = _quarterly(frame)
    first = _fit(frame, folds[0])
    _ = _fit(frame, folds[-1])
    again = _fit(frame, folds[0])
    assert again.coefficients == first.coefficients


# --- 4. the label can never be a feature -----------------------------------


@pytest.mark.parametrize("column", LABEL_COLUMNS)
def test_label_columns_are_not_features(column: str) -> None:
    assert column not in FEATURE_COLUMNS
    for spec in MODEL_REGISTRY:
        assert column not in spec.feature_columns


def test_a_spec_that_names_the_label_cannot_be_fitted_at_all() -> None:
    """Three independent barriers stand between a leaky spec and an artifact.

    The registry guard rejects it at import; this covers the path where a spec is
    constructed directly and handed straight to ``fit_fold``. It fails while building
    the preprocessor, because a missing-value strategy can only be resolved for a
    declared Component 4 feature -- so the leak is refused before anything is fitted,
    not merely reported afterwards.
    """
    leaky = dataclasses.replace(
        PRIMARY, name="leaky", feature_columns=(*PRIMARY.feature_columns, "target")
    )
    with pytest.raises(KeyError, match="Unknown feature: target"):
        _fit(_base(), spec=leaky)


def test_validate_also_reports_a_leaky_spec() -> None:
    """The third barrier, for a fit that somehow got past the first two.

    ``FittedModel`` is frozen, so the only way to reach this state is to substitute the
    spec after the fact -- which is exactly the scenario the check exists for.
    """
    from sentinel.modeling import validate as modeling_validate

    frame = _base()
    honest = _fit(frame)
    tampered = dataclasses.replace(
        honest,
        spec=dataclasses.replace(
            PRIMARY, name="leaky", feature_columns=(*PRIMARY.feature_columns, "target")
        ),
    )
    checks = modeling_validate.validate_baselines(
        frame,
        [EARLY_FOLD],
        [tampered],
        pl.DataFrame(schema={"model_name": pl.Utf8}),
        expected_models=["leaky"],
    )
    forbidden = next(c for c in checks if c.name == "features_exclude_forbidden_columns")
    assert not forbidden.passed
    assert any("target" in offender for offender in forbidden.offenders)


def test_the_leakage_detector_itself_works() -> None:
    """A sanity check on this suite: prove that leakage *would* be visible if present.

    Overwrite a legitimate feature with the label, so the model is fitted on a column
    that is the answer. If the scores did not then separate the classes perfectly, the
    tests above would be reassuring for the wrong reason -- they would be measuring a
    model too weak to exploit a leak rather than a pipeline that prevents one.
    """
    frame = _base().with_columns(
        pl.col("target").cast(pl.Float64).alias("prior_canvass_priority_rate")
    )
    fitted = _fit(frame)
    test_window = folds_module.window_frame(frame, EARLY_FOLD)
    _, scores = predict.score_window(fitted, test_window)
    labels = test_window["target"].to_list()
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    assert min(positives) > max(negatives)


# --- 5. determinism and order invariance ------------------------------------


def test_shuffling_the_whole_input_table_changes_nothing() -> None:
    """Row order of the source table must not reach a coefficient or a score."""
    base = _base()
    order = list(range(base.height))
    random.Random(20260817).shuffle(order)
    shuffled = base[order]

    reference = _fit(base)
    scrambled = _fit(shuffled)
    assert scrambled.coefficients == reference.coefficients
    assert scrambled.intercept == reference.intercept

    expected = predict.score_window(reference, folds_module.window_frame(base, EARLY_FOLD))
    actual = predict.score_window(scrambled, folds_module.window_frame(shuffled, EARLY_FOLD))
    assert actual == expected


def test_two_identical_runs_agree_exactly() -> None:
    base = _base()
    first = _fit(base)
    second = _fit(base)
    assert first.coefficients == second.coefficients
    assert first.n_iter == second.n_iter
