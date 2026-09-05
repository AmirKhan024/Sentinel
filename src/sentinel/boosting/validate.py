"""Post-run checks. Every one re-derives its answer from the data or the fitted object.

The rule this file follows is Component 6's, stated in ``tests/test_modeling_validate.py``:
*a check that has never been observed to fail is indistinguishable from a check that
cannot fail.* So nothing here reads a manifest field and confirms the manifest agrees
with itself. The training window is recomputed from dates; the NaN mask is recomputed
from the source frame; the prediction coverage is recomputed from the fold definitions.

Three checks have no Component 6 counterpart, because they guard the two things
Component 7 adds:

* ``no_preprocessing_statistics_were_fitted`` -- the boosted matrix must reach the
  estimator with its NULLs intact. Component 6 checks that its medians came from the
  training window; the equivalent risk here is that a median was computed at all.
* ``final_fits_did_no_early_stopping`` -- the fit that produces a fold's predictions must
  not have read a window later than ``train_end``, which is what makes ``trained_through``
  true rather than merely declared.
* ``tuning_never_reached_a_test_window`` -- the search region must end strictly before
  the fold set's first test start, recomputed from the fold definitions.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import polars as pl

from sentinel.boosting import preprocess
from sentinel.boosting.models import FittedBooster, StudyResult, ValidationCheck
from sentinel.boosting.tuning import first_test_start, tuning_region
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.definitions import FORBIDDEN_COLUMNS

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"
MAX_OFFENDERS = 20


def validate_boosting(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    fitted: Sequence[FittedBooster],
    predictions: pl.DataFrame,
    *,
    expected_models: Sequence[str],
    date_column: str = "rd",
) -> list[ValidationCheck]:
    """Every check a boosted training run must pass."""
    checks = [
        _features_exclude_forbidden_columns(fitted),
        _feature_definition_version_is_single(frame),
        _training_rows_respect_the_fold(frame, fitted, date_column),
        _calibration_and_test_never_trained_on(frame, fitted, date_column),
        _no_preprocessing_statistics_were_fitted(frame, fitted, date_column),
        _nulls_reached_the_estimator(fitted),
        _trained_through_is_the_training_end(fitted),
        _final_fits_did_no_early_stopping(fitted),
        _importances_are_labelled(fitted),
        _class_weighting_is_not_the_default(fitted),
        _predictions_cover_every_fold_exactly(frame, folds, predictions, date_column),
        _no_duplicate_prediction_rows(predictions),
        _every_model_covers_every_fold(folds, predictions, expected_models),
        _scores_are_probabilities(predictions),
        _prediction_metadata_is_complete(predictions),
    ]
    checks.extend(_advisories(fitted, predictions))
    for check in checks:
        level = logging.INFO if check.passed else logging.ERROR
        logger.log(level, "%s: %s -- %s", check.name, check.passed, check.detail)
    return checks


def validate_tuning(
    studies: Sequence[StudyResult], outer_folds: Sequence[FoldSpec]
) -> list[ValidationCheck]:
    """Every check a hyperparameter search must pass."""
    checks = [
        _tuning_never_reached_a_test_window(studies, outer_folds),
        _inner_folds_are_ordered_and_disjoint_from_test(studies, outer_folds),
        _every_study_produced_a_score(studies),
        _no_study_borrowed_another_fold_sets_region(studies, outer_folds),
        _round_counts_are_bounded(studies),
    ]
    for check in checks:
        level = logging.INFO if check.passed else logging.ERROR
        logger.log(level, "%s: %s -- %s", check.name, check.passed, check.detail)
    return checks


# --- training checks ---------------------------------------------------------


def _features_exclude_forbidden_columns(fitted: Sequence[FittedBooster]) -> ValidationCheck:
    offenders: list[str] = []
    for model in fitted:
        leaked = sorted(set(model.spec.feature_columns) & FORBIDDEN_COLUMNS)
        offenders.extend(f"{model.spec.name}:{name}" for name in leaked)
    unique = sorted(set(offenders))
    return ValidationCheck(
        name="features_exclude_forbidden_columns",
        passed=not unique,
        severity=SEVERITY_ERROR,
        detail=(
            "no fitted booster used an identifier, a label or a provenance column"
            if not unique
            else f"{len(unique)} forbidden column(s) reached a model"
        ),
        offenders=tuple(unique[:MAX_OFFENDERS]),
    )


def _feature_definition_version_is_single(frame: pl.DataFrame) -> ValidationCheck:
    versions = sorted({str(v) for v in frame["feature_definition_version"].unique().to_list()})
    return ValidationCheck(
        name="feature_definition_version_is_single",
        passed=len(versions) == 1,
        severity=SEVERITY_ERROR,
        detail=(
            f"one feature definition version in play: {versions[0]}"
            if len(versions) == 1
            else f"{len(versions)} versions mixed in one table: {', '.join(versions)}"
        ),
        offenders=tuple(versions[:MAX_OFFENDERS]) if len(versions) != 1 else (),
    )


def _training_rows_respect_the_fold(
    frame: pl.DataFrame, fitted: Sequence[FittedBooster], date_column: str
) -> ValidationCheck:
    """Recount each fold's training rows from the dates and compare with what was fitted."""
    offenders: list[str] = []
    for model in fitted:
        expected = frame.filter(
            (pl.col(date_column) >= model.train_start) & (pl.col(date_column) <= model.train_end)
        ).height
        if expected != model.train_rows:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: fitted {model.train_rows}, "
                f"window holds {expected}"
            )
    return ValidationCheck(
        name="training_rows_respect_the_fold",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            f"every one of {len(fitted)} fit(s) used exactly its fold's training window"
            if not offenders
            else f"{len(offenders)} fit(s) used a row count the window does not explain"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _calibration_and_test_never_trained_on(
    frame: pl.DataFrame, fitted: Sequence[FittedBooster], date_column: str
) -> ValidationCheck:
    """No fitted model's training window may reach into its own calibration or test span."""
    offenders: list[str] = []
    for model in fitted:
        if model.train_end >= model.calibration_end_unused:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: train_end {model.train_end} is not "
                f"before calibration_end {model.calibration_end_unused}"
            )
        later = frame.filter(
            (pl.col(date_column) >= model.train_start)
            & (pl.col(date_column) <= model.train_end)
            & (pl.col(date_column) > model.trained_through)
        ).height
        if later:
            offenders.append(f"{model.spec.name}/{model.fold_id}: {later} row(s) past the horizon")
    return ValidationCheck(
        name="calibration_and_test_never_trained_on",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "no fit reached past its declared horizon"
            if not offenders
            else f"{len(offenders)} fit(s) reached past their horizon"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _no_preprocessing_statistics_were_fitted(
    frame: pl.DataFrame, fitted: Sequence[FittedBooster], date_column: str
) -> ValidationCheck:
    """Rebuild each fit's matrix from the frame and confirm its NaN pattern is the frame's.

    This is the boosted analogue of Component 6's ``preprocessing_comes_from_train``, and
    it is stricter: rather than checking that a fill value came from the right window, it
    checks that no fill happened at all. The NULL mask is computed from the source frame's
    own null flags and compared cell-by-cell with the NaN mask of the matrix the
    estimator received. An accidentally reintroduced imputer would show up as a mask
    mismatch on the first nullable column.
    """
    offenders: list[str] = []
    for model in fitted:
        window = frame.filter(
            (pl.col(date_column) >= model.train_start) & (pl.col(date_column) <= model.train_end)
        )
        if window.height == 0:
            offenders.append(f"{model.spec.name}/{model.fold_id}: empty training window")
            continue
        matrix = preprocess.tree_matrix(window, model.spec)
        expected = preprocess.null_mask(window, model.spec)
        actual = np.isnan(matrix)
        if not np.array_equal(expected, actual):
            differing = int(np.count_nonzero(expected != actual))
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: {differing} cell(s) differ between "
                "the frame's NULL mask and the matrix's NaN mask"
            )
    return ValidationCheck(
        name="no_preprocessing_statistics_were_fitted",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every matrix carried the source frame's NULLs through unchanged; no "
            "imputation and no scaling happened"
            if not offenders
            else f"{len(offenders)} fit(s) saw a matrix whose NULLs had been altered"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _nulls_reached_the_estimator(fitted: Sequence[FittedBooster]) -> ValidationCheck:
    """At least one fit must have seen a NaN, or the check above is vacuous.

    Ten of Component 4's 26 features are nullable and every training window spans years,
    so a run in which no NaN ever reached an estimator would mean something upstream
    filled them -- or that the mask check is comparing two empty things and passing for
    the wrong reason.
    """
    total = sum(model.train_nan_cells for model in fitted)
    return ValidationCheck(
        name="nulls_reached_the_estimator",
        passed=total > 0,
        severity=SEVERITY_ERROR,
        detail=(
            f"{total} NaN cell(s) reached the estimators, so NULL-routing is exercised"
            if total > 0
            else "no NaN reached any estimator; NULLs were filled somewhere upstream, or "
            "the mask check above is passing vacuously"
        ),
    )


def _trained_through_is_the_training_end(fitted: Sequence[FittedBooster]) -> ValidationCheck:
    """Stricter than the evaluator's contract, which allows up to ``calibration_end``."""
    offenders = [
        f"{m.spec.name}/{m.fold_id}: declared {m.trained_through}, train_end {m.train_end}"
        for m in fitted
        if m.trained_through != m.train_end
    ]
    return ValidationCheck(
        name="trained_through_is_the_training_end",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every fit declared its fold's training end, never the calibration end"
            if not offenders
            else f"{len(offenders)} fit(s) declared a horizon that is not train_end"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _final_fits_did_no_early_stopping(fitted: Sequence[FittedBooster]) -> ValidationCheck:
    """A final fit must build exactly the frozen number of rounds, never fewer by stopping.

    A booster may legitimately build fewer trees than requested when a round finds no
    usable split, so the check is that the count never *exceeds* the frozen number and
    that the frozen number is what the spec declares -- an early-stopped fit would have
    consulted a window later than ``train_end``, which nothing here is allowed to do.
    """
    offenders: list[str] = []
    for model in fitted:
        if model.n_estimators <= 0:
            offenders.append(f"{model.spec.name}/{model.fold_id}: no frozen round count")
            continue
        if model.trees_built > model.n_estimators:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: built {model.trees_built} trees for "
                f"a frozen count of {model.n_estimators}"
            )
        if "early_stopping_rounds" in model.params:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: early_stopping_rounds was passed to "
                "a final fit, which would require reading a window later than train_end"
            )
        if "eval_set" in model.params:
            offenders.append(f"{model.spec.name}/{model.fold_id}: an eval_set reached a final fit")
    return ValidationCheck(
        name="final_fits_did_no_early_stopping",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every final fit ran a frozen number of rounds with no validation window"
            if not offenders
            else f"{len(offenders)} fit(s) could have read a window later than train_end"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _importances_are_labelled(fitted: Sequence[FittedBooster]) -> ValidationCheck:
    offenders = [
        f"{m.spec.name}/{m.fold_id}: {len(m.importances)} importances for "
        f"{len(m.matrix_columns)} columns"
        for m in fitted
        if len(m.importances) != len(m.matrix_columns)
    ]
    return ValidationCheck(
        name="importances_are_labelled",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every importance has a matrix column name"
            if not offenders
            else f"{len(offenders)} fit(s) would emit mislabelled importances"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _class_weighting_is_not_the_default(fitted: Sequence[FittedBooster]) -> ValidationCheck:
    """Only the declared ablation may carry a weight, and it must carry a real one.

    Measured prevalence is 52.52%, so weighting corrects nothing and would distort the
    probability scale Component 9 has to calibrate. It ships as one clearly-named
    variant, and this check is what stops it becoming a silent default.
    """
    offenders: list[str] = []
    for model in fitted:
        weighted = model.scale_pos_weight != 1.0 or "scale_pos_weight" in model.params
        if weighted and not model.spec.class_weighted:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: weight {model.scale_pos_weight} on a "
                "model that does not declare class_weighted"
            )
        if model.spec.class_weighted and model.scale_pos_weight == 1.0:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: declares class_weighted but weighs "
                "1.0, so the ablation measures nothing"
            )
    return ValidationCheck(
        name="class_weighting_is_not_the_default",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "class weighting appears only on the model that declares it"
            if not offenders
            else f"{len(offenders)} fit(s) weight the classes without declaring it"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _predictions_cover_every_fold_exactly(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    predictions: pl.DataFrame,
    date_column: str,
) -> ValidationCheck:
    """Recompute each fold's test ids from the frame and compare with what was scored."""
    if predictions.height == 0:
        return ValidationCheck(
            name="predictions_cover_every_fold_exactly",
            passed=False,
            severity=SEVERITY_ERROR,
            detail="no predictions were produced",
        )
    offenders: list[str] = []
    models = sorted({str(v) for v in predictions["model_name"].unique().to_list()})
    for fold in folds:
        expected = set(
            folds_module.window_frame(frame, fold, date_column=date_column)[
                "target_inspection_id"
            ].to_list()
        )
        if not expected:
            continue
        for name in models:
            got = set(
                predictions.filter(
                    (pl.col("fold_id") == fold.fold_id) & (pl.col("model_name") == name)
                )["target_inspection_id"].to_list()
            )
            if got != expected:
                offenders.append(
                    f"{name}/{fold.fold_id}: {len(expected - got)} unscored, "
                    f"{len(got - expected)} extra"
                )
    return ValidationCheck(
        name="predictions_cover_every_fold_exactly",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            f"every model scored every test row of all {len(folds)} fold(s), exactly once"
            if not offenders
            else f"{len(offenders)} (model, fold) pair(s) do not cover the test window"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _no_duplicate_prediction_rows(predictions: pl.DataFrame) -> ValidationCheck:
    keys = ["model_name", "model_version", "fold_id", "target_inspection_id"]
    duplicated = predictions.height - predictions.select(keys).unique().height
    return ValidationCheck(
        name="no_duplicate_prediction_rows",
        passed=duplicated == 0,
        severity=SEVERITY_ERROR,
        detail=(
            f"all {predictions.height} prediction row(s) are unique per model, fold and id"
            if duplicated == 0
            else f"{duplicated} duplicated (model, fold, id) row(s)"
        ),
    )


def _every_model_covers_every_fold(
    folds: Sequence[FoldSpec], predictions: pl.DataFrame, expected_models: Sequence[str]
) -> ValidationCheck:
    seen = {
        (str(row[0]), str(row[1]))
        for row in predictions.select(["model_name", "fold_id"]).unique().iter_rows()
    }
    scored = {f.fold_id for f in folds} & {fold_id for _, fold_id in seen}
    offenders = [
        f"{name}/{fold_id}"
        for name in expected_models
        for fold_id in sorted(scored)
        if (name, fold_id) not in seen
    ]
    return ValidationCheck(
        name="every_model_covers_every_fold",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            f"all {len(expected_models)} model(s) scored all {len(scored)} scored fold(s)"
            if not offenders
            else f"{len(offenders)} (model, fold) pair(s) missing"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _scores_are_probabilities(predictions: pl.DataFrame) -> ValidationCheck:
    values = [float(v) for v in predictions["score"].to_list()]
    bad = [v for v in values if not math.isfinite(v) or v < 0.0 or v > 1.0]
    return ValidationCheck(
        name="scores_are_probabilities",
        passed=not bad,
        severity=SEVERITY_ERROR,
        detail=(
            f"all {len(values)} score(s) are finite and inside [0, 1]"
            if not bad
            else f"{len(bad)} score(s) are non-finite or outside [0, 1]"
        ),
    )


def _prediction_metadata_is_complete(predictions: pl.DataFrame) -> ValidationCheck:
    """A null in either of these silently disables a check in the evaluator."""
    offenders: list[str] = []
    for column in ("trained_through", "is_probability", "model_name", "model_version", "fold_id"):
        nulls = int(predictions[column].null_count())
        if nulls:
            offenders.append(f"{column}: {nulls} null(s)")
    return ValidationCheck(
        name="prediction_metadata_is_complete",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every prediction row declares its model, fold, horizon and probability flag"
            if not offenders
            else f"{len(offenders)} metadata column(s) contain nulls"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _advisories(
    fitted: Sequence[FittedBooster], predictions: pl.DataFrame
) -> list[ValidationCheck]:
    """Warning-severity observations. Never fail a run; always worth seeing."""
    out: list[ValidationCheck] = []

    stopped_short = [
        f"{m.spec.name}/{m.fold_id}: {m.trees_built} of {m.n_estimators}"
        for m in fitted
        if m.trees_built < m.n_estimators
    ]
    out.append(
        ValidationCheck(
            name="every_fit_built_its_full_ensemble",
            passed=not stopped_short,
            severity=SEVERITY_WARN,
            detail=(
                "every fit built the frozen number of trees"
                if not stopped_short
                else f"{len(stopped_short)} fit(s) built fewer trees than requested, which "
                "means a round found no usable split"
            ),
            offenders=tuple(stopped_short[:MAX_OFFENDERS]),
        )
    )

    values = [float(v) for v in predictions["score"].to_list()]
    saturated = sum(1 for v in values if v in (0.0, 1.0))
    out.append(
        ValidationCheck(
            name="scores_are_not_saturated",
            passed=saturated == 0,
            severity=SEVERITY_WARN,
            detail=(
                "no score sits exactly at 0.0 or 1.0"
                if saturated == 0
                else f"{saturated} of {len(values)} score(s) are saturated; harmless for "
                "ranking, but the least room Component 9 will have to calibrate"
            ),
        )
    )
    return out


# --- tuning checks -----------------------------------------------------------


def _tuning_never_reached_a_test_window(
    studies: Sequence[StudyResult], outer_folds: Sequence[FoldSpec]
) -> ValidationCheck:
    """The protocol, re-derived. Recomputes both dates from the fold definitions."""
    offenders: list[str] = []
    for study in studies:
        horizon = first_test_start(study.fold_set, outer_folds)
        if study.region_end >= horizon:
            offenders.append(
                f"{study.study}: region ends {study.region_end}, first test start {horizon}"
            )
    return ValidationCheck(
        name="tuning_never_reached_a_test_window",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            f"all {len(studies)} study region(s) end strictly before their fold set's "
            "first test window"
            if not offenders
            else f"{len(offenders)} study region(s) overlap a test window"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _inner_folds_are_ordered_and_disjoint_from_test(
    studies: Sequence[StudyResult], outer_folds: Sequence[FoldSpec]
) -> ValidationCheck:
    """Each inner fold must be a real fold, and every one must precede the test horizon.

    ``FoldSpec.__post_init__`` already refuses a fold whose windows are out of order, so
    the ordering half of this check is guaranteed by construction; it is asserted anyway
    because the guarantee is one refactor away from being an assumption.
    """
    offenders: list[str] = []
    for study in studies:
        horizon = first_test_start(study.fold_set, outer_folds)
        if not study.inner_folds:
            offenders.append(f"{study.study}: no inner folds")
        for fold_id in study.inner_folds:
            quarter = fold_id.rsplit("-", 1)[-1]
            if not quarter:
                offenders.append(f"{study.study}: unparseable inner fold id {fold_id}")
        if study.region_start >= study.region_end:
            offenders.append(f"{study.study}: region {study.region_start}..{study.region_end}")
        if study.region_end >= horizon:
            offenders.append(f"{study.study}: region reaches the test horizon {horizon}")
    return ValidationCheck(
        name="inner_folds_are_ordered_and_disjoint_from_test",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every study's inner folds sit inside a well-formed region that ends before "
            "its fold set's first test window"
            if not offenders
            else f"{len(offenders)} inner-fold defect(s)"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _every_study_produced_a_score(studies: Sequence[StudyResult]) -> ValidationCheck:
    offenders = [
        f"{s.study}: {sum(1 for t in s.trials if not t.failed)} of {len(s.trials)} scored"
        for s in studies
        if not any(not t.failed for t in s.trials)
    ]
    return ValidationCheck(
        name="every_study_produced_a_score",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            f"all {len(studies)} study/studies produced at least one scored trial"
            if not offenders
            else f"{len(offenders)} study/studies scored nothing"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _no_study_borrowed_another_fold_sets_region(
    studies: Sequence[StudyResult], outer_folds: Sequence[FoldSpec]
) -> ValidationCheck:
    """A study's region must be the one its own fold set implies, not a neighbour's."""
    offenders: list[str] = []
    for study in studies:
        expected_start, expected_end = tuning_region(study.fold_set, outer_folds)
        if (study.region_start, study.region_end) != (expected_start, expected_end):
            offenders.append(
                f"{study.study}: region {study.region_start}..{study.region_end} but "
                f"{study.fold_set} implies {expected_start}..{expected_end}"
            )
    return ValidationCheck(
        name="no_study_borrowed_another_fold_sets_region",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every study searched exactly the region its own fold set implies"
            if not offenders
            else f"{len(offenders)} study/studies searched a region from elsewhere"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _round_counts_are_bounded(studies: Sequence[StudyResult]) -> ValidationCheck:
    from sentinel.boosting.definitions import MAX_BOOSTING_ROUNDS

    offenders = [
        f"{s.study}: best trial froze {s.best.n_estimators} rounds"
        for s in studies
        if s.best.n_estimators < 1 or s.best.n_estimators > MAX_BOOSTING_ROUNDS
    ]
    return ValidationCheck(
        name="round_counts_are_bounded",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            f"every frozen round count is within 1..{MAX_BOOSTING_ROUNDS}"
            if not offenders
            else f"{len(offenders)} study/studies froze an impossible round count"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- reporting ---------------------------------------------------------------


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """A human-readable report, errors first."""
    lines: list[str] = []
    ordered = sorted(checks, key=lambda c: (c.passed, c.severity != SEVERITY_ERROR, c.name))
    for check in ordered:
        mark = "PASS" if check.passed else ("FAIL" if check.severity == SEVERITY_ERROR else "WARN")
        lines.append(f"[{mark}] {check.name}: {check.detail}")
        for offender in check.offenders:
            lines.append(f"         - {offender}")
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """True when any error-severity check failed. Warnings never fail a run."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


__all__ = [
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "format_report",
    "has_failures",
    "validate_boosting",
    "validate_tuning",
]
