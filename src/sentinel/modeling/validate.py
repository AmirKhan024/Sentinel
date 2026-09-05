"""Post-training checks over the models and their predictions.

Component 4 has a safety wall against features that can see the future, and Component 5
has one against an evaluation that can. This is the third: a wall against a *model* that
can.

Following the precedent set by both, the important checks **re-derive** their answer from
the data rather than reading back what the orchestrator reported. Re-deriving the
imputation median from the training frame and comparing it to the fitted imputer is the
mechanical proof that preprocessing statistics came from the training window; asking the
orchestrator whether it did the right thing would only prove the code agrees with itself.

The checks that matter most, and what each would catch:

1. ``features_exclude_forbidden_columns``     -- an id or the label used as a predictor
2. ``training_rows_respect_the_fold``         -- a fit that saw a row past its horizon
3. ``calibration_and_test_never_trained_on``  -- overlap at row-identity level
4. ``preprocessing_comes_from_train``         -- a median computed over the whole table
5. ``trained_through_is_the_training_end``    -- a model overstating what it may know
6. ``predictions_cover_every_fold_exactly``   -- a dropped or duplicated test row
7. ``every_model_covers_every_fold``          -- a silently missing fold, e.g. covid_shift
8. ``every_fit_converged``                    -- coefficients that depend on the iteration cap
9. ``scores_are_probabilities``               -- a score that cannot be P(target = 1)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import polars as pl

from sentinel.evaluation.folds import assign_split
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling import preprocess
from sentinel.modeling.definitions import (
    FORBIDDEN_COLUMNS,
    MissingStrategy,
    columns_in_family,
    indicator_columns,
    null_families,
)
from sentinel.modeling.models import FittedModel, ValidationCheck

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

MAX_OFFENDERS = 20

#: Tolerance for comparing a re-derived median against a fitted one. Both come from the
#: same rows, so they agree to the last bit in practice; the tolerance guards against a
#: float representation difference rather than against a real discrepancy.
MEDIAN_TOLERANCE = 1e-9


def _check(
    name: str,
    passed: bool,
    severity: str,
    detail: str,
    offenders: Sequence[str] = (),
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail,
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def validate_baselines(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    fitted: Sequence[FittedModel],
    predictions: pl.DataFrame,
    *,
    expected_models: Sequence[str],
    date_column: str = "rd",
) -> list[ValidationCheck]:
    """Every assertion Component 6 makes about its own output."""
    checks: list[ValidationCheck] = [
        _features_exclude_forbidden_columns(fitted),
        _feature_definition_version_is_single(frame),
        _training_rows_respect_the_fold(frame, folds, fitted, date_column),
        _calibration_and_test_never_trained_on(frame, folds, fitted, date_column),
        _preprocessing_comes_from_train(frame, folds, fitted, date_column),
        _trained_through_is_the_training_end(folds, fitted),
        _null_masks_are_identical_within_family(frame),
        _indicator_columns_are_complete(fitted),
        _every_fit_converged(fitted),
        _coefficients_are_labelled(fitted),
        _predictions_cover_every_fold_exactly(frame, folds, predictions, date_column),
        _no_duplicate_prediction_rows(predictions),
        _every_model_covers_every_fold(folds, predictions, expected_models),
        _scores_are_probabilities(predictions),
        _prediction_metadata_is_complete(predictions),
    ]
    checks.extend(_advisory(fitted, predictions))
    return checks


# --- 1. the feature contract -------------------------------------------------


def _features_exclude_forbidden_columns(fitted: Sequence[FittedModel]) -> ValidationCheck:
    """No identifier, label or provenance column may be a model input.

    Re-checked here even though the registry guard runs at import time, because a spec
    could be constructed directly and handed to ``fit_fold``.
    """
    offenders: list[str] = []
    for model in fitted:
        leaked = sorted(set(model.spec.feature_columns) & FORBIDDEN_COLUMNS)
        for column in leaked:
            offenders.append(f"{model.spec.name}/{model.fold_id}: {column}")
    return _check(
        "features_exclude_forbidden_columns",
        not offenders,
        SEVERITY_ERROR,
        f"{len({m.spec.name for m in fitted})} model(s) use only declared features"
        if not offenders
        else f"{len(offenders)} forbidden column(s) used as model inputs",
        offenders,
    )


def _feature_definition_version_is_single(frame: pl.DataFrame) -> ValidationCheck:
    """One feature definition version across the table.

    Two versions in one file would mean rows built by different feature formulas trained
    together, and the manifest could only record one of them.
    """
    if "feature_definition_version" not in frame.columns:
        return _check(
            "feature_definition_version_is_single",
            False,
            SEVERITY_ERROR,
            "feature table has no feature_definition_version column",
        )
    versions = sorted({str(v) for v in frame["feature_definition_version"].unique().to_list()})
    return _check(
        "feature_definition_version_is_single",
        len(versions) == 1,
        SEVERITY_ERROR,
        f"feature_definition_version = {versions[0]}"
        if len(versions) == 1
        else f"{len(versions)} versions present: {', '.join(versions)}",
        [] if len(versions) == 1 else versions,
    )


# --- 2. temporal integrity ---------------------------------------------------


def _training_rows_respect_the_fold(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    fitted: Sequence[FittedModel],
    date_column: str,
) -> ValidationCheck:
    """Re-derive each fold's training window and confirm the fit's row count matches.

    The row count is the observable proxy for "the fit saw these rows and no others": a
    fit that reached past its horizon would have more rows than the window contains.
    """
    by_id = {fold.fold_id: fold for fold in folds}
    offenders: list[str] = []
    for model in fitted:
        fold = by_id.get(model.fold_id)
        if fold is None:
            offenders.append(f"{model.spec.name}: fold {model.fold_id} is not in the fold set")
            continue
        window = frame.filter(pl.col(date_column).is_between(fold.train_start, fold.train_end))
        if window.height != model.train_rows:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: fitted on {model.train_rows} rows, "
                f"window {fold.train_start}..{fold.train_end} holds {window.height}"
            )
        if model.train_end != fold.train_end or model.train_start != fold.train_start:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: recorded window "
                f"{model.train_start}..{model.train_end} != fold's "
                f"{fold.train_start}..{fold.train_end}"
            )
    return _check(
        "training_rows_respect_the_fold",
        not offenders,
        SEVERITY_ERROR,
        f"{len(fitted)} fit(s) trained on exactly their fold's training window"
        if not offenders
        else f"{len(offenders)} fit(s) disagree with their fold's training window",
        offenders,
    )


def _calibration_and_test_never_trained_on(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    fitted: Sequence[FittedModel],
    date_column: str,
) -> ValidationCheck:
    """No training row id appears in the same fold's calibration or test split.

    Row identity rather than date arithmetic, so a mis-parsed date cannot make this pass.
    """
    by_id = {fold.fold_id: fold for fold in folds}
    offenders: list[str] = []
    for fold_id in sorted({m.fold_id for m in fitted}):
        fold = by_id.get(fold_id)
        if fold is None:
            continue
        assigned = assign_split(frame, fold, date_column=date_column)
        train_ids = set(
            assigned.filter(pl.col("split") == "train")["target_inspection_id"].to_list()
        )
        later_ids = set(
            assigned.filter(pl.col("split").is_in(["calibration", "test"]))[
                "target_inspection_id"
            ].to_list()
        )
        overlap = sorted(train_ids & later_ids)
        for row_id in overlap[:MAX_OFFENDERS]:
            offenders.append(f"{fold_id}: {row_id} is both training and later")
    return _check(
        "calibration_and_test_never_trained_on",
        not offenders,
        SEVERITY_ERROR,
        "no training row appears in a calibration or test split"
        if not offenders
        else f"{len(offenders)} row(s) appear in training and in a later split",
        offenders,
    )


def _preprocessing_comes_from_train(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    fitted: Sequence[FittedModel],
    date_column: str,
) -> ValidationCheck:
    """Re-derive every imputation median from the training window and compare.

    This is the check that would catch preprocessing fitted before splitting -- the one
    form of leakage a fold boundary cannot see, because the boundary is respected by the
    fit while the transform already knows the future.
    """
    by_id = {fold.fold_id: fold for fold in folds}
    offenders: list[str] = []
    compared = 0
    for model in fitted:
        fold = by_id.get(model.fold_id)
        if fold is None:
            continue
        window = frame.filter(pl.col(date_column).is_between(fold.train_start, fold.train_end))
        for column, fitted_value in sorted(model.imputed_values.items()):
            strategy = preprocess.strategy_for(column)
            if strategy is MissingStrategy.CONSTANT_FALSE:
                if abs(fitted_value - preprocess.BOOLEAN_FILL) > MEDIAN_TOLERANCE:
                    offenders.append(
                        f"{model.spec.name}/{model.fold_id}/{column}: constant fill "
                        f"{fitted_value} != {preprocess.BOOLEAN_FILL}"
                    )
                continue
            if strategy is not MissingStrategy.MEDIAN or column not in window.columns:
                continue
            expected = window[column].cast(pl.Float64).median()
            compared += 1
            # polars aggregates are typed as a wide union; the cast above guarantees a
            # float, but the narrowing has to be explicit under strict mode.
            if not isinstance(expected, (int, float)):
                continue
            if abs(float(expected) - fitted_value) > MEDIAN_TOLERANCE:
                offenders.append(
                    f"{model.spec.name}/{model.fold_id}/{column}: fitted median "
                    f"{fitted_value} != training-window median {float(expected)}"
                )
    return _check(
        "preprocessing_comes_from_train",
        not offenders,
        SEVERITY_ERROR,
        f"{compared} imputation median(s) re-derived from the training window and matched"
        if not offenders
        else f"{len(offenders)} preprocessing statistic(s) did not come from training rows",
        offenders,
    )


def _trained_through_is_the_training_end(
    folds: Sequence[FoldSpec], fitted: Sequence[FittedModel]
) -> ValidationCheck:
    """Every model declares its fold's training end, and nothing later.

    Stricter than the evaluator's contract, which permits up to the calibration end.
    Component 6 fits no calibrator, so declaring the calibration end would claim a
    horizon it did not use and would disable this check. ADR 0014.
    """
    by_id = {fold.fold_id: fold for fold in folds}
    offenders: list[str] = []
    for model in fitted:
        fold = by_id.get(model.fold_id)
        if fold is None:
            continue
        if model.trained_through != fold.train_end:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: trained_through="
                f"{model.trained_through}, expected train_end {fold.train_end}"
            )
        elif model.trained_through >= fold.calibration_start:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: trained_through="
                f"{model.trained_through} reaches the calibration window"
            )
    return _check(
        "trained_through_is_the_training_end",
        not offenders,
        SEVERITY_ERROR,
        f"{len(fitted)} fit(s) declare their training end and never the calibration window"
        if not offenders
        else f"{len(offenders)} fit(s) declare a horizon other than their training end",
        offenders,
    )


# --- 3. the missingness design ----------------------------------------------


def _null_masks_are_identical_within_family(frame: pl.DataFrame) -> ValidationCheck:
    """Every column in a null-rule family is null on exactly the same rows.

    The four-indicator design depends on this. If Component 4 ever breaks the property,
    one indicator would stop representing its family and this fails rather than the
    encoding quietly becoming wrong.
    """
    offenders: list[str] = []
    for rule in null_families():
        members = [c for c in columns_in_family(rule) if c in frame.columns]
        if len(members) < 2:
            continue
        reference = frame[members[0]].is_null()
        for member in members[1:]:
            if not bool(frame[member].is_null().eq(reference).all()):
                offenders.append(f"{rule.value}: {member} null mask differs from {members[0]}")
    return _check(
        "null_masks_are_identical_within_family",
        not offenders,
        SEVERITY_ERROR,
        f"{len(null_families())} null-rule family/families have internally identical masks"
        if not offenders
        else f"{len(offenders)} column(s) break their family's null mask",
        offenders,
    )


def _indicator_columns_are_complete(fitted: Sequence[FittedModel]) -> ValidationCheck:
    """Every fit carries all four family indicators, whatever its feature subset."""
    expected = set(indicator_columns())
    offenders: list[str] = []
    for model in fitted:
        absent = sorted(expected - set(model.matrix_columns))
        if absent:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: missing indicator(s) {', '.join(absent)}"
            )
    return _check(
        "indicator_columns_are_complete",
        not offenders,
        SEVERITY_ERROR,
        f"all {len(expected)} family indicators present in every fit"
        if not offenders
        else f"{len(offenders)} fit(s) are missing a family indicator",
        offenders,
    )


# --- 4. the fits -------------------------------------------------------------


def _every_fit_converged(fitted: Sequence[FittedModel]) -> ValidationCheck:
    """No fit stopped at the iteration cap.

    Installed while it passes with two orders of magnitude of headroom (68-83 of 1000).
    Coefficients that depend on the cap are not comparable across folds, and noticing
    that after publishing a number is worse than noticing it here.
    """
    offenders = [
        f"{m.spec.name}/{m.fold_id}: {m.n_iter} iterations, converged={m.converged}"
        for m in fitted
        if not m.converged
    ]
    iterations = [m.n_iter for m in fitted] or [0]
    return _check(
        "every_fit_converged",
        not offenders,
        SEVERITY_ERROR,
        f"{len(fitted)} fit(s) converged; iterations {min(iterations)}-{max(iterations)}"
        if not offenders
        else f"{len(offenders)} fit(s) did not converge",
        offenders,
    )


def _coefficients_are_labelled(fitted: Sequence[FittedModel]) -> ValidationCheck:
    """One coefficient per matrix column, and one scaler statistic per coefficient.

    A length mismatch would mislabel every term while producing identical predictions,
    which is the kind of defect that survives a long time.
    """
    offenders: list[str] = []
    for model in fitted:
        widths = {
            "coefficients": len(model.coefficients),
            "matrix_columns": len(model.matrix_columns),
            "scaler_mean": len(model.scaler_mean),
            "scaler_scale": len(model.scaler_scale),
        }
        if len(set(widths.values())) != 1:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: "
                + ", ".join(f"{k}={v}" for k, v in widths.items())
            )
    return _check(
        "coefficients_are_labelled",
        not offenders,
        SEVERITY_ERROR,
        "every coefficient has a matrix column and a scaler statistic"
        if not offenders
        else f"{len(offenders)} fit(s) have mismatched coefficient labelling",
        offenders,
    )


# --- 5. the predictions ------------------------------------------------------


def _predictions_cover_every_fold_exactly(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    predictions: pl.DataFrame,
    date_column: str,
) -> ValidationCheck:
    """Re-derive each fold's test window and compare id sets, both directions.

    The evaluator performs this check too, per prediction set. Doing it here means a
    coverage defect is reported by the component that caused it.
    """
    if predictions.height == 0:
        return _check(
            "predictions_cover_every_fold_exactly",
            False,
            SEVERITY_ERROR,
            "no predictions were produced",
        )
    offenders: list[str] = []
    for fold in folds:
        expected = set(
            frame.filter(pl.col(date_column).is_between(fold.test_start, fold.test_end))[
                "target_inspection_id"
            ].to_list()
        )
        if not expected:
            continue
        for model_name in sorted(predictions["model_name"].unique().to_list()):
            got = set(
                predictions.filter(
                    (pl.col("fold_id") == fold.fold_id) & (pl.col("model_name") == model_name)
                )["target_inspection_id"].to_list()
            )
            if got != expected:
                offenders.append(
                    f"{model_name}/{fold.fold_id}: {len(expected - got)} unscored, "
                    f"{len(got - expected)} outside the test window"
                )
    return _check(
        "predictions_cover_every_fold_exactly",
        not offenders,
        SEVERITY_ERROR,
        f"{predictions.height} prediction row(s) cover every test window exactly"
        if not offenders
        else f"{len(offenders)} (model, fold) pair(s) do not cover their test window",
        offenders,
    )


def _no_duplicate_prediction_rows(predictions: pl.DataFrame) -> ValidationCheck:
    """One score per (model, fold, row). A duplicate would be counted twice at k."""
    if predictions.height == 0:
        return _check("no_duplicate_prediction_rows", False, SEVERITY_ERROR, "no predictions")
    keys = ["model_name", "model_version", "fold_id", "target_inspection_id"]
    duplicated = predictions.group_by(keys).len().filter(pl.col("len") > 1).sort(keys)
    offenders = [
        f"{r['model_name']}/{r['fold_id']}/{r['target_inspection_id']}: {r['len']} rows"
        for r in duplicated.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "no_duplicate_prediction_rows",
        duplicated.height == 0,
        SEVERITY_ERROR,
        "every (model, fold, row) has exactly one score"
        if duplicated.height == 0
        else f"{duplicated.height} duplicated (model, fold, row) key(s)",
        offenders,
    )


def _every_model_covers_every_fold(
    folds: Sequence[FoldSpec],
    predictions: pl.DataFrame,
    expected_models: Sequence[str],
) -> ValidationCheck:
    """Every requested model scored every fold that has test rows.

    ``read_predictions`` returns an empty list for a fold it finds nothing for, without
    error, so a silently missing fold would simply not appear in the results -- and
    ``covid_shift`` is the one most likely to be missed, while being the fold the
    Component 5 findings say reverses the baseline ordering.
    """
    scored = (
        {
            (row["model_name"], row["fold_id"])
            for row in predictions.select(["model_name", "fold_id"]).unique().iter_rows(named=True)
        }
        if predictions.height
        else set()
    )
    offenders = [
        f"{name}: no predictions for {fold.fold_id}"
        for name in sorted(expected_models)
        for fold in folds
        if (name, fold.fold_id) not in scored
    ]
    return _check(
        "every_model_covers_every_fold",
        not offenders,
        SEVERITY_ERROR,
        f"{len(expected_models)} model(s) x {len(folds)} fold(s) all scored"
        if not offenders
        else f"{len(offenders)} (model, fold) pair(s) missing",
        offenders,
    )


def _scores_are_probabilities(predictions: pl.DataFrame) -> ValidationCheck:
    """Every score is finite and within [0, 1].

    Closed interval, not open: ``predict_proba`` can return exactly 0.0 or 1.0 by
    rounding when the linear predictor is extreme, which is legitimate and is reported
    as a warning instead.
    """
    if predictions.height == 0:
        return _check("scores_are_probabilities", False, SEVERITY_ERROR, "no predictions")
    scores = predictions["score"].to_list()
    offenders: list[str] = []
    for score, row_id, model in zip(
        scores,
        predictions["target_inspection_id"].to_list(),
        predictions["model_name"].to_list(),
        strict=True,
    ):
        if score is None or not math.isfinite(score) or score < 0.0 or score > 1.0:
            offenders.append(f"{model}/{row_id}: {score}")
            if len(offenders) >= MAX_OFFENDERS:
                break
    finite = [s for s in scores if s is not None and math.isfinite(s)]
    return _check(
        "scores_are_probabilities",
        not offenders,
        SEVERITY_ERROR,
        f"all {len(scores)} scores finite in [0, 1]; observed {min(finite):.6g}..{max(finite):.6g}"
        if not offenders and finite
        else f"{len(offenders)} score(s) are not usable probabilities",
        offenders,
    )


def _prediction_metadata_is_complete(predictions: pl.DataFrame) -> ValidationCheck:
    """No null in the columns that make a prediction file self-describing.

    A null ``trained_through`` would make the evaluator skip its horizon check; a null
    ``is_probability`` is coerced to False by ``read_predictions``, silently downgrading
    the model to ranking-only and suppressing the probability metrics.
    """
    if predictions.height == 0:
        return _check("prediction_metadata_is_complete", False, SEVERITY_ERROR, "no predictions")
    required = (
        "model_name",
        "model_version",
        "fold_id",
        "trained_through",
        "is_probability",
        "score",
        "target_inspection_id",
    )
    offenders = [
        f"{column}: {predictions[column].null_count()} null(s)"
        for column in required
        if column in predictions.columns and predictions[column].null_count()
    ]
    absent = [c for c in required if c not in predictions.columns]
    offenders.extend(f"{c}: column absent" for c in absent)
    return _check(
        "prediction_metadata_is_complete",
        not offenders,
        SEVERITY_ERROR,
        "every prediction row is self-describing"
        if not offenders
        else f"{len(offenders)} metadata problem(s)",
        offenders,
    )


# --- 6. advisory -------------------------------------------------------------


def _advisory(fitted: Sequence[FittedModel], predictions: pl.DataFrame) -> list[ValidationCheck]:
    """Warning-severity notes. Always pass; they exist to surface a distribution."""
    notes: list[ValidationCheck] = []

    saturated = 0
    if predictions.height:
        scores = predictions["score"].to_list()
        saturated = sum(1 for s in scores if s in (0.0, 1.0))
    notes.append(
        _check(
            "saturated_scores",
            True,
            SEVERITY_WARN,
            f"{saturated} score(s) sit exactly at 0.0 or 1.0. Harmless for ranking and "
            "log_loss clamps, but it means the probability lost resolution there.",
        )
    )

    approximations = sorted({m.spec.name for m in fitted if m.spec.approximation_note is not None})
    notes.append(
        _check(
            "approximation_models",
            True,
            SEVERITY_WARN,
            f"{len(approximations)} model(s) are labelled approximations and must never "
            f"be presented as the model they approximate: {', '.join(approximations)}"
            if approximations
            else "no approximation models in this run",
        )
    )

    notes.append(
        _check(
            "probabilities_are_uncalibrated",
            True,
            SEVERITY_WARN,
            "Scores are raw predict_proba output. Component 9 owns calibration; Brier, "
            "log-loss, ECE and MCE from this run are diagnostics of an uncalibrated "
            "model and must not be described otherwise.",
        )
    )

    if fitted:
        rates = [m.train_positive_rate for m in fitted if m.train_positive_rate is not None]
        if rates:
            notes.append(
                _check(
                    "train_base_rate_drift",
                    True,
                    SEVERITY_WARN,
                    f"training-window positive rate spans {min(rates):.4f}..{max(rates):.4f} "
                    "across folds. Expanding windows dilute early years, so a later fold "
                    "trains on a different base rate than an earlier one.",
                )
            )
    return notes


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """Render the checks as a plain text block for the CLI."""
    lines = ["", "Baseline model validation report", "-------------------------------"]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
        if check.offenders and not (check.passed and check.severity == SEVERITY_ERROR):
            for offender in check.offenders:
                lines.append(f"           - {offender}")
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """Whether any error-severity check failed."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


__all__ = [
    "MAX_OFFENDERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "format_report",
    "has_failures",
    "validate_baselines",
]
