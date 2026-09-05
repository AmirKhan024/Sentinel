"""The validators, driven into failure as well as into success.

The rule this file enforces is Component 6's, restated: *a check that has never been
observed to fail is indistinguishable from a check that cannot fail.* Component 5 shipped
exactly that defect once -- ``scores_respect_the_decision_point``, declared and
unreachable, fixed in ADR 0014 -- so every error-severity check here is driven both ways.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from sentinel.neural import predict, train, validate, writer
from sentinel.neural.definitions import spec_for
from sentinel.neural.models import ValidationCheck
from tests.conftest import neural_categoricals_for, spanning_model_features

PRIMARY = spec_for("neural_embeddings")


def _base() -> pl.DataFrame:
    return spanning_model_features(days=1600, per_day=2).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def _fold(frame: pl.DataFrame) -> FoldSpec:
    built = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    assert built
    return built[0]


def _named(checks: list[ValidationCheck], name: str) -> ValidationCheck:
    match = [c for c in checks if c.name == name]
    assert match, f"no check named {name}; the suite is asserting against nothing"
    return match[0]


def _run(frame: pl.DataFrame, fold: FoldSpec, cats: pl.DataFrame) -> tuple:  # type: ignore[type-arg]
    fitted = train.fit_fold(
        PRIMARY, training_frame(frame, fold), fold, categoricals=cats, max_epochs=2
    )
    window = folds_module.window_frame(frame, fold)
    ids, scores = predict.score_window(fitted, window, categoricals=cats)
    rows = [
        {
            "target_inspection_id": i,
            "score": s,
            "model_name": PRIMARY.name,
            "model_version": PRIMARY.version,
            "fold_set": fold.fold_set,
            "fold_id": fold.fold_id,
            "trained_through": fitted.trained_through,
            "is_probability": True,
            "neural_definition_version": "v1",
        }
        for i, s in zip(ids, scores, strict=True)
    ]
    predictions = writer.finalize(rows, "neural_predictions")
    checks = validate.validate_neural(
        frame, [fold], [fitted], [], predictions, cats, expected_models=[PRIMARY.name]
    )
    return fitted, predictions, checks


# --- 1. a clean run passes ---------------------------------------------------


def test_a_clean_run_passes_every_error_severity_check() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    _, _, checks = _run(frame, fold, cats)
    failed = [c.name for c in checks if not c.passed and c.severity == validate.SEVERITY_ERROR]
    assert not failed, f"a clean run failed: {', '.join(failed)}"
    assert not validate.has_failures(checks)


def test_the_expected_checks_are_present() -> None:
    """A renamed check would silently stop being asserted anywhere."""
    frame = _base()
    fold = _fold(frame)
    _, _, checks = _run(frame, fold, neural_categoricals_for(frame))
    names = {c.name for c in checks}
    for required in (
        "features_exclude_forbidden_columns",
        "entity_columns_are_never_identity",
        "early_stopping_window_is_inside_training",
        "preprocessing_comes_from_inner_train",
        "vocabularies_contain_no_future_category",
        "chain_membership_is_fold_local",
        "trained_through_is_the_training_end",
        "embeddings_came_from_the_same_fold",
        "predictions_cover_every_fold_exactly",
        "every_model_scored_the_same_rows",
        "scores_are_probabilities",
    ):
        assert required in names, f"{required} is missing from the validator"


# --- 2. every check is driven into failure -----------------------------------


def test_a_missing_prediction_row_is_caught() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, predictions, _ = _run(frame, fold, cats)
    short = predictions.head(predictions.height - 1)
    checks = validate.validate_neural(
        frame, [fold], [fitted], [], short, cats, expected_models=[PRIMARY.name]
    )
    assert not _named(checks, "predictions_cover_every_fold_exactly").passed


def test_a_duplicated_prediction_row_is_caught() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, predictions, _ = _run(frame, fold, cats)
    doubled = pl.concat([predictions, predictions.head(1)])
    checks = validate.validate_neural(
        frame, [fold], [fitted], [], doubled, cats, expected_models=[PRIMARY.name]
    )
    assert not _named(checks, "no_duplicate_prediction_rows").passed


def test_an_out_of_range_score_is_caught() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, predictions, _ = _run(frame, fold, cats)
    broken = predictions.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(1.5).otherwise(pl.col("score")).alias("score")
    )
    checks = validate.validate_neural(
        frame, [fold], [fitted], [], broken, cats, expected_models=[PRIMARY.name]
    )
    assert not _named(checks, "scores_are_probabilities").passed


def test_a_null_in_a_metadata_column_is_caught() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, predictions, _ = _run(frame, fold, cats)
    broken = predictions.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("is_probability"))
        .alias("is_probability")
    )
    checks = validate.validate_neural(
        frame, [fold], [fitted], [], broken, cats, expected_models=[PRIMARY.name]
    )
    assert not _named(checks, "prediction_metadata_is_complete").passed


def test_two_models_scoring_different_rows_is_caught() -> None:
    """The fair-comparison rule, mechanised.

    A comparison table over two different populations is the failure least visible in a
    metric and most damaging to the component's conclusion.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, predictions, _ = _run(frame, fold, cats)
    second = predictions.head(predictions.height - 3).with_columns(
        pl.lit("neural_numeric_only").alias("model_name")
    )
    combined = pl.concat([predictions, second])
    checks = validate.validate_neural(
        frame,
        [fold],
        [fitted],
        [],
        combined,
        cats,
        expected_models=[PRIMARY.name, "neural_numeric_only"],
    )
    assert not _named(checks, "every_model_scored_the_same_rows").passed


def test_more_than_one_feature_definition_version_is_caught() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, predictions, _ = _run(frame, fold, cats)
    mixed = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < 5)
        .then(pl.lit("v2"))
        .otherwise(pl.col("feature_definition_version"))
        .alias("feature_definition_version")
    )
    checks = validate.validate_neural(
        frame, [fold], [fitted], [], predictions, cats, expected_models=[PRIMARY.name]
    )
    assert _named(checks, "feature_definition_version_is_single").passed
    checks = validate.validate_neural(
        mixed, [fold], [fitted], [], predictions, cats, expected_models=[PRIMARY.name]
    )
    assert not _named(checks, "feature_definition_version_is_single").passed


# --- 3. the categorical layer ------------------------------------------------


def test_the_categorical_checks_pass_on_a_clean_table() -> None:
    frame = _base()
    checks = validate.validate_categoricals(frame, neural_categoricals_for(frame))
    assert not validate.has_failures(checks)


def test_an_uncovered_feature_row_is_caught() -> None:
    frame = _base()
    cats = neural_categoricals_for(frame).head(10)
    checks = validate.validate_categoricals(frame, cats)
    assert not _named(checks, "categoricals_cover_every_row").passed


def test_a_null_category_is_caught() -> None:
    """Absence must be the UNKNOWN token, never a null."""
    frame = _base()
    cats = neural_categoricals_for(frame).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("facility_type"))
        .alias("facility_type")
    )
    checks = validate.validate_categoricals(frame, cats)
    assert not _named(checks, "categoricals_are_never_null").passed


def test_a_label_column_in_the_categorical_layer_is_caught() -> None:
    frame = _base()
    cats = neural_categoricals_for(frame).with_columns(pl.lit(1).alias("target"))
    checks = validate.validate_categoricals(frame, cats)
    assert not _named(checks, "categoricals_carry_no_label").passed


# --- 4. the sweep ------------------------------------------------------------


def _sweep(region_end: date, inner: tuple[str, ...], rate: float = 1e-3):  # type: ignore[no-untyped-def]
    from sentinel.neural.models import SweepPoint, SweepResult

    return SweepResult(
        study="s",
        model_name="neural_embeddings",
        fold_set="quarterly",
        region_start=date(2018, 7, 1),
        region_end=region_end,
        inner_folds=inner,
        points=(SweepPoint("a", rate, 10, 5, 0.6, 3),),
        scores=((rate, 0.6),),
        best_learning_rate=rate,
        selection_reason="r",
        seed=1,
        seconds=1.0,
    )


def test_a_sweep_reaching_a_test_window_is_caught() -> None:
    frame = _base()
    outer = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    from sentinel.boosting.tuning import build_inner_folds, first_test_start

    horizon = first_test_start("quarterly", outer)
    inner = tuple(f.fold_id for f in build_inner_folds("quarterly", outer))

    good = validate.validate_sweep([_sweep(date(2018, 7, 2), inner)], outer)
    assert _named(good, "sweep_never_reached_a_test_window").passed

    bad = validate.validate_sweep([_sweep(horizon, inner)], outer)
    assert not _named(bad, "sweep_never_reached_a_test_window").passed


def test_a_sweep_using_the_wrong_inner_folds_is_caught() -> None:
    frame = _base()
    outer = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    checks = validate.validate_sweep([_sweep(date(2018, 7, 2), ("made", "up"))], outer)
    assert not _named(checks, "sweep_inner_folds_are_ordered_and_disjoint").passed


def test_a_selected_rate_outside_the_scored_set_is_caught() -> None:
    from sentinel.neural.models import SweepPoint, SweepResult

    frame = _base()
    outer = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    from sentinel.boosting.tuning import build_inner_folds

    inner = tuple(f.fold_id for f in build_inner_folds("quarterly", outer))
    result = SweepResult(
        study="s",
        model_name="m",
        fold_set="quarterly",
        region_start=date(2018, 7, 1),
        region_end=date(2018, 7, 2),
        inner_folds=inner,
        points=(SweepPoint("a", 1e-3, 10, 5, 0.6, 3),),
        scores=((1e-3, 0.6),),
        best_learning_rate=0.42,
        selection_reason="r",
        seed=1,
        seconds=1.0,
    )
    checks = validate.validate_sweep([result], outer)
    assert not _named(checks, "selected_rate_is_in_the_grid").passed


# --- 5. reporting ------------------------------------------------------------


def test_has_failures_ignores_warnings() -> None:
    warn = ValidationCheck("w", False, validate.SEVERITY_WARN, "a warning")
    error = ValidationCheck("e", False, validate.SEVERITY_ERROR, "an error")
    assert not validate.has_failures([warn])
    assert validate.has_failures([warn, error])


def test_the_report_marks_failures_warnings_and_passes_distinctly() -> None:
    checks = [
        ValidationCheck("p", True, validate.SEVERITY_ERROR, "fine"),
        ValidationCheck("w", False, validate.SEVERITY_WARN, "hmm"),
        ValidationCheck("e", False, validate.SEVERITY_ERROR, "bad", offenders=("x",)),
    ]
    report = validate.format_report(checks)
    assert "[PASS] p" in report
    assert "[WARN] w" in report
    assert "[FAIL] e" in report
    assert "- x" in report


def test_failures_are_reported_before_passes() -> None:
    checks = [
        ValidationCheck("p", True, validate.SEVERITY_ERROR, "fine"),
        ValidationCheck("e", False, validate.SEVERITY_ERROR, "bad"),
    ]
    report = validate.format_report(checks)
    assert report.index("[FAIL] e") < report.index("[PASS] p")
