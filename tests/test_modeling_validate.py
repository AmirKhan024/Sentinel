"""Every Component 6 validation check, shown passing on good input and failing on bad.

A check that has never been observed to fail is indistinguishable from a check that
cannot fail -- Component 5 shipped one of those (``scores_respect_the_decision_point``,
declared and unreachable), which is the precedent this file exists to avoid repeating.
So each error-severity check below gets a deliberately broken input and an assertion that
it reports the specific problem.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import polars as pl
import pytest

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling import train, validate, writer
from sentinel.modeling.definitions import MODELS_BY_NAME
from sentinel.modeling.models import FittedModel, ValidationCheck
from tests.conftest import make_model_feature_row, model_feature_scenario, spanning_model_features

PRIMARY = MODELS_BY_NAME["logistic_regression"]

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


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return spanning_model_features(days=1600).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


@pytest.fixture(scope="module")
def fitted(frame: pl.DataFrame) -> FittedModel:
    return train.fit_fold(PRIMARY, train.training_frame(frame, FOLD), FOLD)


def _predictions(frame: pl.DataFrame, fitted: FittedModel, **overrides: object) -> pl.DataFrame:
    """A contract-shaped prediction table covering FOLD's test window exactly."""
    from sentinel.modeling import predict
    from sentinel.modeling.definitions import MODEL_DEFINITION_VERSION

    ids, scores = predict.score_window(fitted, folds_module.window_frame(frame, FOLD))
    rows: list[dict[str, object]] = [
        {
            "target_inspection_id": row_id,
            "score": score,
            "model_name": fitted.spec.name,
            "model_version": fitted.spec.version,
            "fold_set": FOLD.fold_set,
            "fold_id": FOLD.fold_id,
            "trained_through": fitted.trained_through,
            "is_probability": True,
            "model_definition_version": MODEL_DEFINITION_VERSION,
            **overrides,
        }
        for row_id, score in zip(ids, scores, strict=True)
    ]
    return writer.finalize(rows, "baseline_predictions")


def _run(
    frame: pl.DataFrame,
    fitted: list[FittedModel],
    predictions: pl.DataFrame,
    models: list[str] | None = None,
) -> dict[str, ValidationCheck]:
    checks = validate.validate_baselines(
        frame,
        [FOLD],
        fitted,
        predictions,
        expected_models=models or [PRIMARY.name],
    )
    return {check.name: check for check in checks}


# --- the happy path ---------------------------------------------------------


def test_a_clean_run_passes_every_error_check(frame: pl.DataFrame, fitted: FittedModel) -> None:
    checks = _run(frame, [fitted], _predictions(frame, fitted))
    failed = [
        c.name for c in checks.values() if not c.passed and c.severity == validate.SEVERITY_ERROR
    ]
    assert failed == []
    assert not validate.has_failures(list(checks.values()))


def test_advisory_checks_are_present_and_always_pass(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    checks = _run(frame, [fitted], _predictions(frame, fitted))
    for name in (
        "saturated_scores",
        "approximation_models",
        "probabilities_are_uncalibrated",
        "train_base_rate_drift",
    ):
        assert checks[name].severity == validate.SEVERITY_WARN
        assert checks[name].passed


def test_the_uncalibrated_note_is_always_emitted(frame: pl.DataFrame, fitted: FittedModel) -> None:
    """Component 9 owns calibration; nothing here may read as calibrated."""
    checks = _run(frame, [fitted], _predictions(frame, fitted))
    assert "uncalibrated" in checks["probabilities_are_uncalibrated"].detail
    assert "Component 9" in checks["probabilities_are_uncalibrated"].detail


# --- each error check, failing ----------------------------------------------


def test_forbidden_column_check_fails(frame: pl.DataFrame, fitted: FittedModel) -> None:
    tampered = dataclasses.replace(
        fitted,
        spec=dataclasses.replace(PRIMARY, feature_columns=(*PRIMARY.feature_columns, "target")),
    )
    check = _run(frame, [tampered], _predictions(frame, fitted))[
        "features_exclude_forbidden_columns"
    ]
    assert not check.passed
    assert any("target" in o for o in check.offenders)


def test_feature_definition_version_check_fails_on_two_versions(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    mixed = frame.with_columns(
        pl.when(pl.col("rd") > date(2020, 1, 1))
        .then(pl.lit("v2"))
        .otherwise(pl.col("feature_definition_version"))
        .alias("feature_definition_version")
    )
    check = _run(mixed, [fitted], _predictions(frame, fitted))[
        "feature_definition_version_is_single"
    ]
    assert not check.passed
    assert "2 versions" in check.detail


def test_feature_definition_version_check_fails_when_absent(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    check = _run(frame.drop("feature_definition_version"), [fitted], _predictions(frame, fitted))[
        "feature_definition_version_is_single"
    ]
    assert not check.passed


def test_training_window_check_fails_on_an_inflated_row_count(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """The row count is the observable proxy for "the fit saw these rows and no others"."""
    tampered = dataclasses.replace(fitted, train_rows=fitted.train_rows + 500)
    check = _run(frame, [tampered], _predictions(frame, fitted))["training_rows_respect_the_fold"]
    assert not check.passed
    assert any("rows" in o for o in check.offenders)


def test_training_window_check_fails_on_a_shifted_window(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    tampered = dataclasses.replace(fitted, train_end=date(2022, 3, 31))
    check = _run(frame, [tampered], _predictions(frame, fitted))["training_rows_respect_the_fold"]
    assert not check.passed


def test_training_window_check_fails_on_an_unknown_fold(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    tampered = dataclasses.replace(fitted, fold_id="quarterly-2099Q4")
    check = _run(frame, [tampered], _predictions(frame, fitted))["training_rows_respect_the_fold"]
    assert not check.passed
    assert any("not in the fold set" in o for o in check.offenders)


def test_preprocessing_check_fails_on_a_median_from_outside_the_window(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """This is the check that would catch preprocessing fitted before splitting."""
    whole_table_median = float(frame["days_since_last_canvass"].cast(pl.Float64).median() or 0.0)
    tampered = dataclasses.replace(
        fitted,
        imputed_values={
            **fitted.imputed_values,
            "days_since_last_canvass": whole_table_median + 77,
        },
    )
    check = _run(frame, [tampered], _predictions(frame, fitted))["preprocessing_comes_from_train"]
    assert not check.passed
    assert any("median" in o for o in check.offenders)


def test_preprocessing_check_fails_on_a_boolean_filled_with_something_other_than_zero(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    tampered = dataclasses.replace(
        fitted, imputed_values={**fitted.imputed_values, "priority_at_last_canvass": 1.0}
    )
    check = _run(frame, [tampered], _predictions(frame, fitted))["preprocessing_comes_from_train"]
    assert not check.passed
    assert any("constant fill" in o for o in check.offenders)


def test_trained_through_check_fails_on_the_calibration_end(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """The contract permits calibration_end; Component 6 does not. ADR 0014."""
    tampered = dataclasses.replace(fitted, trained_through=FOLD.calibration_end)
    check = _run(frame, [tampered], _predictions(frame, fitted))[
        "trained_through_is_the_training_end"
    ]
    assert not check.passed
    assert any("expected train_end" in o for o in check.offenders)


def test_trained_through_check_fails_on_a_future_horizon(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    tampered = dataclasses.replace(fitted, trained_through=date(2026, 1, 1))
    check = _run(frame, [tampered], _predictions(frame, fitted))[
        "trained_through_is_the_training_end"
    ]
    assert not check.passed


def test_null_mask_check_fails_when_a_family_member_diverges(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """The four-indicator design depends on within-family mask identity."""
    broken = frame.with_columns(
        pl.when(pl.col("fail_at_last_canvass").is_null())
        .then(pl.lit(value=False))
        .otherwise(pl.col("fail_at_last_canvass"))
        .alias("fail_at_last_canvass")
    )
    check = _run(broken, [fitted], _predictions(frame, fitted))[
        "null_masks_are_identical_within_family"
    ]
    assert not check.passed
    assert any("fail_at_last_canvass" in o for o in check.offenders)


def test_indicator_completeness_check_fails_when_one_is_dropped(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    tampered = dataclasses.replace(
        fitted,
        matrix_columns=tuple(
            c for c in fitted.matrix_columns if c != "missing_no_code_era_canvass"
        ),
    )
    check = _run(frame, [tampered], _predictions(frame, fitted))["indicator_columns_are_complete"]
    assert not check.passed


def test_convergence_check_fails(frame: pl.DataFrame, fitted: FittedModel) -> None:
    tampered = dataclasses.replace(fitted, converged=False, n_iter=1000)
    check = _run(frame, [tampered], _predictions(frame, fitted))["every_fit_converged"]
    assert not check.passed


def test_coefficient_labelling_check_fails_on_a_length_mismatch(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """A mismatch would mislabel every term while predicting identically."""
    tampered = dataclasses.replace(fitted, coefficients=fitted.coefficients[:-1])
    check = _run(frame, [tampered], _predictions(frame, fitted))["coefficients_are_labelled"]
    assert not check.passed


def test_coverage_check_fails_on_a_dropped_row(frame: pl.DataFrame, fitted: FittedModel) -> None:
    predictions = _predictions(frame, fitted).head(-1)
    check = _run(frame, [fitted], predictions)["predictions_cover_every_fold_exactly"]
    assert not check.passed
    assert any("unscored" in o for o in check.offenders)


def test_coverage_check_fails_on_an_out_of_window_row(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    predictions = _predictions(frame, fitted)
    extra = predictions.head(1).with_columns(pl.lit("not-a-test-row").alias("target_inspection_id"))
    check = _run(frame, [fitted], pl.concat([predictions, extra]))[
        "predictions_cover_every_fold_exactly"
    ]
    assert not check.passed
    assert any("outside the test window" in o for o in check.offenders)


def test_coverage_check_fails_on_no_predictions(frame: pl.DataFrame, fitted: FittedModel) -> None:
    check = _run(frame, [fitted], writer.empty("baseline_predictions"))[
        "predictions_cover_every_fold_exactly"
    ]
    assert not check.passed


def test_duplicate_check_fails(frame: pl.DataFrame, fitted: FittedModel) -> None:
    predictions = _predictions(frame, fitted)
    doubled = pl.concat([predictions, predictions.head(3)])
    check = _run(frame, [fitted], doubled)["no_duplicate_prediction_rows"]
    assert not check.passed
    assert "duplicated" in check.detail


def test_every_model_covers_every_fold_check_fails_on_a_missing_model(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """The covid_shift case: read_predictions returns [] silently for a missing fold."""
    check = _run(
        frame,
        [fitted],
        _predictions(frame, fitted),
        models=[PRIMARY.name, "cdph_2015_approximation"],
    )["every_model_covers_every_fold"]
    assert not check.passed
    assert any("cdph_2015_approximation" in o for o in check.offenders)


def test_score_range_check_fails_above_one(frame: pl.DataFrame, fitted: FittedModel) -> None:
    predictions = _predictions(frame, fitted).with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(1.5).otherwise(pl.col("score")).alias("score")
    )
    check = _run(frame, [fitted], predictions)["scores_are_probabilities"]
    assert not check.passed


def test_score_range_check_fails_below_zero(frame: pl.DataFrame, fitted: FittedModel) -> None:
    predictions = _predictions(frame, fitted).with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(-0.1).otherwise(pl.col("score")).alias("score")
    )
    check = _run(frame, [fitted], predictions)["scores_are_probabilities"]
    assert not check.passed


def test_score_range_check_accepts_exact_zero_and_one(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """Closed interval, not open: predict_proba can saturate legitimately, and that is
    reported as a warning rather than rejected."""
    predictions = _predictions(frame, fitted).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(0.0)
        .when(pl.int_range(pl.len()) == 1)
        .then(1.0)
        .otherwise(pl.col("score"))
        .alias("score")
    )
    checks = _run(frame, [fitted], predictions)
    assert checks["scores_are_probabilities"].passed
    assert "2 score(s) sit exactly" in checks["saturated_scores"].detail


def test_metadata_check_fails_on_a_null_trained_through(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """A null horizon makes the evaluator skip its horizon check entirely."""
    predictions = _predictions(frame, fitted).with_columns(
        pl.lit(None).cast(pl.Date).alias("trained_through")
    )
    check = _run(frame, [fitted], predictions)["prediction_metadata_is_complete"]
    assert not check.passed
    assert any("trained_through" in o for o in check.offenders)


def test_metadata_check_fails_on_a_null_is_probability(
    frame: pl.DataFrame, fitted: FittedModel
) -> None:
    """read_predictions coerces a null to False, silently downgrading the model."""
    predictions = _predictions(frame, fitted).with_columns(
        pl.lit(None).cast(pl.Boolean).alias("is_probability")
    )
    check = _run(frame, [fitted], predictions)["prediction_metadata_is_complete"]
    assert not check.passed


def test_approximation_note_names_the_labelled_model(frame: pl.DataFrame) -> None:
    spec = MODELS_BY_NAME["cdph_2015_approximation"]
    fitted = train.fit_fold(spec, train.training_frame(frame, FOLD), FOLD)
    checks = _run(frame, [fitted], _predictions(frame, fitted), models=[spec.name])
    note = checks["approximation_models"]
    assert "cdph_2015_approximation" in note.detail
    assert "never" in note.detail


# --- reporting --------------------------------------------------------------


def test_has_failures_only_counts_error_severity() -> None:
    warn = ValidationCheck(name="w", passed=False, severity=validate.SEVERITY_WARN, detail="")
    error = ValidationCheck(name="e", passed=False, severity=validate.SEVERITY_ERROR, detail="")
    assert not validate.has_failures([warn])
    assert validate.has_failures([warn, error])


def test_format_report_marks_failures_and_notes() -> None:
    checks = [
        ValidationCheck(name="ok", passed=True, severity=validate.SEVERITY_ERROR, detail="fine"),
        ValidationCheck(
            name="bad",
            passed=False,
            severity=validate.SEVERITY_ERROR,
            detail="broken",
            offenders=("row-1",),
        ),
        ValidationCheck(name="fyi", passed=True, severity=validate.SEVERITY_WARN, detail="note"),
    ]
    report = validate.format_report(checks)
    assert "[PASS] ok" in report
    assert "[FAIL] bad" in report
    assert "[note] fyi" in report
    assert "row-1" in report


def test_offenders_are_capped(frame: pl.DataFrame, fitted: FittedModel) -> None:
    """A failing check must not print thousands of lines."""
    predictions = _predictions(frame, fitted).with_columns(pl.lit(2.0).alias("score"))
    check = _run(frame, [fitted], predictions)["scores_are_probabilities"]
    assert not check.passed
    assert len(check.offenders) <= validate.MAX_OFFENDERS


def test_validate_handles_a_frame_with_no_matching_fold(fitted: FittedModel) -> None:
    """Robustness: an empty table must produce failures, not an exception."""
    empty = model_feature_scenario([make_model_feature_row(0)]).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    checks = _run(empty, [fitted], writer.empty("baseline_predictions"))
    assert validate.has_failures(list(checks.values()))
