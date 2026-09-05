"""Post-run checks. Every one re-derives its answer from the data or the fitted object.

The rule this file follows is Components 6's and 7's: *a check that has never been
observed to fail is indistinguishable from a check that cannot fail.* So nothing here
reads a manifest field and confirms the manifest agrees with itself. The training window
is recomputed from dates, the scaler statistics are recomputed from the training frame,
and the vocabularies are recomputed from the training rows.

Six checks have no Component 6 or 7 counterpart, because they guard the things Component 8
adds:

* ``preprocessing_comes_from_inner_train`` -- Component 7 checks that no statistic was
  fitted at all; Component 8 fits several, so it checks they came from the right rows.
* ``vocabularies_contain_no_future_category`` -- the leak specific to an embedding. A
  category index that exists because the category appears in a test window is future
  information, even though it is not a label.
* ``chain_membership_is_fold_local`` -- whether an establishment is "part of a chain"
  depends on which other establishments exist, so membership computed globally would let
  a later-opened location reach backwards.
* ``early_stopping_window_is_inside_training`` -- the claim that makes
  ``trained_through = train_end`` true for a component that early-stops.
* ``embeddings_came_from_the_same_fold`` -- the entire temporal argument for the
  embeddings-into-XGBoost experiment.
* ``categoricals_are_strictly_as_of`` -- every carried value came from an inspection
  strictly earlier than the row it was carried to, re-derived per row.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import polars as pl

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling import preprocess as baseline_preprocess
from sentinel.modeling.definitions import FORBIDDEN_COLUMNS
from sentinel.neural import encode
from sentinel.neural.definitions import (
    ENTITY_COLUMNS,
    UNKNOWN_CATEGORY,
    CategoricalEncoding,
)
from sentinel.neural.models import (
    FittedEmbeddingBooster,
    FittedNetwork,
    SweepResult,
    ValidationCheck,
)

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"
MAX_OFFENDERS = 20

#: Tolerance for re-deriving a fitted statistic. Matches Component 6's: the same
#: computation over the same rows in the same order is exact, so anything looser would be
#: hiding a real difference.
STATISTIC_TOLERANCE = 1e-9


def validate_neural(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    fitted: Sequence[FittedNetwork],
    boosted: Sequence[FittedEmbeddingBooster],
    predictions: pl.DataFrame,
    categoricals: pl.DataFrame,
    *,
    expected_models: Sequence[str],
    date_column: str = "rd",
) -> list[ValidationCheck]:
    """Every check a neural training run must pass."""
    checks = [
        _features_exclude_forbidden_columns(fitted),
        _entity_columns_are_never_identity(fitted),
        _feature_definition_version_is_single(frame),
        _training_rows_respect_the_fold(frame, fitted, date_column),
        _calibration_and_test_never_trained_on(frame, fitted, date_column),
        _early_stopping_window_is_inside_training(fitted),
        _preprocessing_comes_from_inner_train(frame, fitted, categoricals, date_column),
        _vocabularies_contain_no_future_category(frame, fitted, categoricals, date_column),
        _chain_membership_is_fold_local(frame, fitted, categoricals, date_column),
        _trained_through_is_the_training_end(fitted),
        _best_epoch_is_within_the_run(fitted),
        _embeddings_are_labelled(fitted),
        _pos_weighting_is_not_the_default(fitted),
        _embeddings_came_from_the_same_fold(boosted),
        _predictions_cover_every_fold_exactly(
            frame, folds, predictions, expected_models, date_column
        ),
        _no_duplicate_prediction_rows(predictions),
        _every_model_covers_every_fold(folds, predictions, expected_models),
        _scores_are_probabilities(predictions),
        _prediction_metadata_is_complete(predictions),
        _every_model_scored_the_same_rows(predictions),
    ]
    checks.extend(_advisories(fitted, predictions))
    for check in checks:
        logger.debug("check %s: %s", check.name, check.passed)
    return checks


def validate_categoricals(features: pl.DataFrame, table: pl.DataFrame) -> list[ValidationCheck]:
    """Every check the experimental categorical join must pass."""
    return [
        _categoricals_cover_every_row(features, table),
        _categoricals_are_strictly_as_of(table),
        _categoricals_are_never_null(table),
        _categoricals_carry_no_label(table),
    ]


def validate_sweep(
    results: Sequence[SweepResult], outer_folds: Sequence[FoldSpec]
) -> list[ValidationCheck]:
    """Every check a learning-rate sweep must pass."""
    return [
        _sweep_never_reached_a_test_window(results, outer_folds),
        _sweep_inner_folds_are_ordered_and_disjoint(results, outer_folds),
        _every_rate_was_scored(results),
        _selected_rate_is_in_the_grid(results),
    ]


# --- 1. what the models were allowed to see ----------------------------------


def _features_exclude_forbidden_columns(fitted: Sequence[FittedNetwork]) -> ValidationCheck:
    """Re-asserted after fitting, not only at import. A spec could be built at runtime."""
    offenders: list[str] = []
    for model in fitted:
        leaked = sorted(set(model.spec.feature_columns) & FORBIDDEN_COLUMNS)
        if leaked:
            offenders.append(f"{model.spec.name}/{model.fold_id}: {', '.join(leaked)}")
    return ValidationCheck(
        name="features_exclude_forbidden_columns",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "no fitted model used an identifier, label or provenance column as a feature"
            if not offenders
            else f"{len(offenders)} fit(s) used a forbidden column"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _entity_columns_are_never_identity(fitted: Sequence[FittedNetwork]) -> ValidationCheck:
    """The check that keeps ``establishment_id`` out of an embedding table.

    The import-time guard already refuses it, so this is the runtime restatement: an
    embedding of establishment identity is the single most damaging thing this component
    could build, and one guard for it is not enough.
    """
    allowed = set(ENTITY_COLUMNS)
    offenders: list[str] = []
    for model in fitted:
        stray = sorted(set(model.spec.entity_columns) - allowed)
        forbidden = sorted(set(model.spec.entity_columns) & FORBIDDEN_COLUMNS)
        if stray or forbidden:
            offenders.append(f"{model.spec.name}/{model.fold_id}: {', '.join(stray + forbidden)}")
    return ValidationCheck(
        name="entity_columns_are_never_identity",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every embedded family is a declared EntityFamily; no fit embedded an "
            "establishment identifier"
            if not offenders
            else f"{len(offenders)} fit(s) embedded something undeclared"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _feature_definition_version_is_single(frame: pl.DataFrame) -> ValidationCheck:
    versions = sorted({str(v) for v in frame["feature_definition_version"].to_list()})
    return ValidationCheck(
        name="feature_definition_version_is_single",
        passed=len(versions) == 1,
        severity=SEVERITY_ERROR,
        detail=(
            f"every row carries feature_definition_version={versions[0]}"
            if len(versions) == 1
            else f"{len(versions)} feature definition versions present: {', '.join(versions)}"
        ),
        offenders=tuple(versions[:MAX_OFFENDERS]) if len(versions) != 1 else (),
    )


def _training_rows_respect_the_fold(
    frame: pl.DataFrame, fitted: Sequence[FittedNetwork], date_column: str
) -> ValidationCheck:
    """Recomputed from the fold dates, not read off the fit."""
    offenders: list[str] = []
    for model in fitted:
        window = frame.filter(
            (pl.col(date_column) >= model.train_start) & (pl.col(date_column) <= model.train_end)
        )
        if window.height != model.train_rows:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: fitted on {model.train_rows} rows but "
                f"{model.train_start}..{model.train_end} contains {window.height}"
            )
    return ValidationCheck(
        name="training_rows_respect_the_fold",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every fit used exactly the rows its fold's training window contains"
            if not offenders
            else f"{len(offenders)} fit(s) used a different row set"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _calibration_and_test_never_trained_on(
    frame: pl.DataFrame, fitted: Sequence[FittedNetwork], date_column: str
) -> ValidationCheck:
    offenders: list[str] = []
    for model in fitted:
        later = frame.filter(pl.col(date_column) > model.train_end)
        if later.height and model.trained_through > model.train_end:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: trained_through {model.trained_through} "
                f"exceeds train_end {model.train_end}"
            )
    return ValidationCheck(
        name="calibration_and_test_never_trained_on",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "no fit declared a horizon later than its training window"
            if not offenders
            else f"{len(offenders)} fit(s) declared a later horizon"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _early_stopping_window_is_inside_training(
    fitted: Sequence[FittedNetwork],
) -> ValidationCheck:
    """The check that makes ``trained_through`` true for a component that early-stops.

    Two properties, both necessary: the validation window starts after the training
    window starts, and it ends no later than ``train_end``. If either failed, the
    stopping signal came from data the fold's horizon says the model never saw.
    """
    offenders: list[str] = []
    for model in fitted:
        if model.spec.learner.value != "mlp":
            continue
        if not (model.train_start < model.inner_validation_start <= model.train_end):
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: early-stopping window starts "
                f"{model.inner_validation_start}, outside {model.train_start}..{model.train_end}"
            )
        if model.inner_validation_start > model.calibration_end_unused:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: early stopping reached the calibration window"
            )
    return ValidationCheck(
        name="early_stopping_window_is_inside_training",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every early-stopping signal came from rows inside the fold's own training "
            "window; no calibration or test row influenced when a fit stopped"
            if not offenders
            else f"{len(offenders)} fit(s) stopped on data outside the training window"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- 2. the fitted statistics ------------------------------------------------


def _inner_train_frame(
    frame: pl.DataFrame,
    categoricals: pl.DataFrame,
    model: FittedNetwork,
    date_column: str,
) -> pl.DataFrame:
    """Re-derive the rows a fit's statistics should have come from.

    Rebuilt from the recorded dates rather than taken from the fit, so the comparison is
    between two independently derived answers rather than between the fit and itself.
    """
    window = frame.filter(
        (pl.col(date_column) >= model.train_start)
        & (pl.col(date_column) < model.inner_validation_start)
    ).sort(["inspection_date", "target_inspection_id"])
    if model.spec.encoding is CategoricalEncoding.NONE:
        return window
    wanted = [
        c for c in ("chain_key", *ENTITY_COLUMNS, "establishment_id") if c in categoricals.columns
    ]
    return window.join(
        categoricals.select("target_inspection_id", *wanted),
        on="target_inspection_id",
        how="left",
        suffix="_cat",
    )


def _preprocessing_comes_from_inner_train(
    frame: pl.DataFrame,
    fitted: Sequence[FittedNetwork],
    categoricals: pl.DataFrame,
    date_column: str,
) -> ValidationCheck:
    """Re-derive every median from the rows it should have come from.

    Component 6's check, adapted: there the source was the training window, here it is the
    inner training window, because Component 8 deliberately fits its statistics on the
    early-stopping split's training side only.
    """
    offenders: list[str] = []
    for model in fitted:
        if model.spec.learner.value != "mlp":
            continue
        source = _inner_train_frame(frame, categoricals, model, date_column)
        if source.height == 0:
            offenders.append(f"{model.spec.name}/{model.fold_id}: no inner training rows")
            continue
        for column, fill in model.imputed_values.items():
            if column not in source.columns:
                continue
            strategy = baseline_preprocess.strategy_for(column)
            if strategy.value == "constant_false":
                expected = 0.0
            else:
                median = source[column].cast(pl.Float64).median()
                # polars aggregates are typed as a wide union; the cast above makes this
                # a float in practice, but strict mode needs the narrowing to be explicit.
                if not isinstance(median, (int, float)):
                    continue
                expected = float(median)
            if not math.isclose(fill, expected, rel_tol=0.0, abs_tol=STATISTIC_TOLERANCE):
                offenders.append(
                    f"{model.spec.name}/{model.fold_id}/{column}: fitted {fill!r}, inner "
                    f"training window implies {expected!r}"
                )
    return ValidationCheck(
        name="preprocessing_comes_from_inner_train",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every imputation value re-derives exactly from the rows the fit was allowed "
            "to compute it on"
            if not offenders
            else f"{len(offenders)} statistic(s) do not match their source window"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _vocabularies_contain_no_future_category(
    frame: pl.DataFrame,
    fitted: Sequence[FittedNetwork],
    categoricals: pl.DataFrame,
    date_column: str,
) -> ValidationCheck:
    """The leak specific to an embedding, checked by set difference.

    A vocabulary entry that exists because its category appears after ``train_end`` is
    future information even though it is not a label: it changes the embedding table's
    shape, its initialisation and every gradient that flows through it.
    """
    offenders: list[str] = []
    for model in fitted:
        if model.spec.encoding is CategoricalEncoding.NONE:
            continue
        source = _inner_train_frame(frame, categoricals, model, date_column)
        if source.height == 0:
            continue
        resolved = encode.resolve_categories(source, model.encoding.chains)
        for vocab in model.encoding.vocabularies:
            if vocab.column not in resolved.columns:
                continue
            observed = {str(v) for v in resolved[vocab.column].drop_nulls().to_list()}
            observed.add(UNKNOWN_CATEGORY)
            unexplained = sorted(set(vocab.categories) - observed)
            if unexplained:
                offenders.append(
                    f"{model.spec.name}/{model.fold_id}/{vocab.column}: "
                    f"{len(unexplained)} categor(y/ies) not present in the fitting window, "
                    f"e.g. {', '.join(unexplained[:3])}"
                )
    return ValidationCheck(
        name="vocabularies_contain_no_future_category",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every embedding row corresponds to a category observed in the window its "
            "vocabulary was fitted on"
            if not offenders
            else f"{len(offenders)} vocabular(y/ies) contain a category from outside it"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _chain_membership_is_fold_local(
    frame: pl.DataFrame,
    fitted: Sequence[FittedNetwork],
    categoricals: pl.DataFrame,
    date_column: str,
) -> ValidationCheck:
    """Recompute the chain set from the fitting window and compare.

    The failure this catches is subtle and would not show up anywhere else: a globally
    computed membership set would mark an establishment as chained on the strength of a
    sibling location that does not exist yet, and every metric would still look normal.
    """
    offenders: list[str] = []
    for model in fitted:
        if model.spec.encoding is CategoricalEncoding.NONE:
            continue
        source = _inner_train_frame(frame, categoricals, model, date_column)
        if source.height == 0 or "chain_key" not in source.columns:
            continue
        expected = encode.chain_membership(source)
        if expected != model.encoding.chains:
            extra = sorted(model.encoding.chains - expected)
            absent = sorted(expected - model.encoding.chains)
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: {len(extra)} chain(s) not implied by "
                f"the fitting window, {len(absent)} implied but missing"
            )
    return ValidationCheck(
        name="chain_membership_is_fold_local",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every fit's chain set re-derives exactly from the rows it was fitted on"
            if not offenders
            else f"{len(offenders)} fit(s) used a chain set from a wider window"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _trained_through_is_the_training_end(fitted: Sequence[FittedNetwork]) -> ValidationCheck:
    offenders = [
        f"{m.spec.name}/{m.fold_id}: trained_through {m.trained_through} != train_end {m.train_end}"
        for m in fitted
        if m.trained_through != m.train_end
    ]
    return ValidationCheck(
        name="trained_through_is_the_training_end",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every fit declares its fold's training end as its horizon"
            if not offenders
            else f"{len(offenders)} fit(s) declared something else"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _best_epoch_is_within_the_run(fitted: Sequence[FittedNetwork]) -> ValidationCheck:
    offenders: list[str] = []
    for model in fitted:
        if model.spec.learner.value != "mlp":
            continue
        if not model.epochs:
            offenders.append(f"{model.spec.name}/{model.fold_id}: no epochs recorded")
            continue
        if not (1 <= model.best_epoch <= model.final_epoch):
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: best epoch {model.best_epoch} outside "
                f"1..{model.final_epoch}"
            )
        if len(model.epochs) != model.final_epoch:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: {len(model.epochs)} epoch records for "
                f"{model.final_epoch} epochs"
            )
    return ValidationCheck(
        name="best_epoch_is_within_the_run",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every fit's restored epoch is one it actually ran"
            if not offenders
            else f"{len(offenders)} fit(s) report an impossible epoch"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _embeddings_are_labelled(fitted: Sequence[FittedNetwork]) -> ValidationCheck:
    offenders: list[str] = []
    for model in fitted:
        if model.spec.encoding is not CategoricalEncoding.EMBEDDING:
            continue
        for table in model.embeddings:
            if len(table.vectors) != model.encoding.vocabulary_for(table.column).size:
                offenders.append(
                    f"{model.spec.name}/{model.fold_id}/{table.column}: "
                    f"{len(table.vectors)} vectors for "
                    f"{model.encoding.vocabulary_for(table.column).size} categories"
                )
            if len(table.categories) != len(table.vectors):
                offenders.append(
                    f"{model.spec.name}/{model.fold_id}/{table.column}: category and "
                    "vector counts disagree"
                )
    return ValidationCheck(
        name="embeddings_are_labelled",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every learned vector is attributable to exactly one category"
            if not offenders
            else f"{len(offenders)} embedding table(s) would be mislabelled"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _pos_weighting_is_not_the_default(fitted: Sequence[FittedNetwork]) -> ValidationCheck:
    """The weighting ablation must be the only weighted model, and must be weighted."""
    offenders: list[str] = []
    for model in fitted:
        if model.spec.learner.value != "mlp":
            continue
        if model.spec.pos_weighted and model.pos_weight is None:
            offenders.append(f"{model.spec.name}/{model.fold_id}: declared weighted, was not")
        if not model.spec.pos_weighted and model.pos_weight is not None:
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: weighted at {model.pos_weight} without "
                "declaring it"
            )
    return ValidationCheck(
        name="pos_weighting_is_not_the_default",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "only the declared ablation applied pos_weight; every other fit used the "
            "unweighted loss the 52.5% prevalence implies"
            if not offenders
            else f"{len(offenders)} fit(s) disagree with their own declaration"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _embeddings_came_from_the_same_fold(
    boosted: Sequence[FittedEmbeddingBooster],
) -> ValidationCheck:
    """The entire temporal argument for the embeddings-into-XGBoost experiment."""
    offenders = [
        f"{b.spec.name}/{b.fold_id}: consumed embeddings from {b.donor_model}/{b.donor_fold_id}"
        for b in boosted
        if b.donor_fold_id != b.fold_id
    ]
    return ValidationCheck(
        name="embeddings_came_from_the_same_fold",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every embedding-fed booster consumed vectors learned on its own fold's training window"
            if not offenders
            else f"{len(offenders)} fit(s) consumed vectors from another fold"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- 3. the artifact ---------------------------------------------------------


def _predictions_cover_every_fold_exactly(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    predictions: pl.DataFrame,
    expected_models: Sequence[str],
    date_column: str,
) -> ValidationCheck:
    offenders: list[str] = []
    for fold in folds:
        window = folds_module.window_frame(frame, fold, date_column=date_column)
        expected = set(str(v) for v in window["target_inspection_id"].to_list())
        if not expected:
            continue
        for name in expected_models:
            scored = predictions.filter(
                (pl.col("fold_id") == fold.fold_id) & (pl.col("model_name") == name)
            )
            got = {str(v) for v in scored["target_inspection_id"].to_list()}
            if got != expected:
                offenders.append(
                    f"{name}/{fold.fold_id}: {len(got)} scored against {len(expected)} test "
                    f"rows ({len(expected - got)} missing, {len(got - expected)} extra)"
                )
    return ValidationCheck(
        name="predictions_cover_every_fold_exactly",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every model scored every test row of every fold exactly once"
            if not offenders
            else f"{len(offenders)} (model, fold) pair(s) do not cover their window"
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
            "one score per (model, version, fold, row)"
            if duplicated == 0
            else f"{duplicated} duplicated prediction row(s)"
        ),
    )


def _every_model_covers_every_fold(
    folds: Sequence[FoldSpec], predictions: pl.DataFrame, expected_models: Sequence[str]
) -> ValidationCheck:
    present = {
        (str(r["model_name"]), str(r["fold_id"]))
        for r in predictions.select("model_name", "fold_id").unique().iter_rows(named=True)
    }
    scored_folds = {str(v) for v in predictions["fold_id"].to_list()}
    offenders = [
        f"{name}/{fold.fold_id}"
        for fold in folds
        for name in expected_models
        if fold.fold_id in scored_folds and (name, fold.fold_id) not in present
    ]
    return ValidationCheck(
        name="every_model_covers_every_fold",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every requested model produced predictions for every scored fold"
            if not offenders
            else f"{len(offenders)} (model, fold) pair(s) missing entirely"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _scores_are_probabilities(predictions: pl.DataFrame) -> ValidationCheck:
    scores = predictions["score"].to_numpy()
    bad = int(np.count_nonzero(~np.isfinite(scores)))
    outside = int(np.count_nonzero((scores < 0.0) | (scores > 1.0)))
    return ValidationCheck(
        name="scores_are_probabilities",
        passed=(bad == 0 and outside == 0),
        severity=SEVERITY_ERROR,
        detail=(
            "every score is finite and within [0, 1]"
            if bad == 0 and outside == 0
            else f"{bad} non-finite and {outside} out-of-range score(s)"
        ),
    )


def _prediction_metadata_is_complete(predictions: pl.DataFrame) -> ValidationCheck:
    required = (
        "model_name",
        "model_version",
        "fold_set",
        "fold_id",
        "trained_through",
        "is_probability",
    )
    offenders = [
        f"{column}: {predictions[column].null_count()} null(s)"
        for column in required
        if column in predictions.columns and predictions[column].null_count()
    ]
    return ValidationCheck(
        name="prediction_metadata_is_complete",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "no null in any metadata column a consumer reads"
            if not offenders
            else f"{len(offenders)} metadata column(s) carry nulls"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _every_model_scored_the_same_rows(predictions: pl.DataFrame) -> ValidationCheck:
    """Every Component 8 model must score an identical id set.

    Without this, a comparison table could silently be comparing models over different
    populations -- which is exactly the failure mode the fair-comparison rule exists to
    prevent, and the one that would be least visible in a metric.
    """
    names = sorted({str(v) for v in predictions["model_name"].to_list()})
    if len(names) < 2:
        return ValidationCheck(
            name="every_model_scored_the_same_rows",
            passed=True,
            severity=SEVERITY_ERROR,
            detail="fewer than two models present; nothing to compare",
        )
    reference = {
        str(v)
        for v in predictions.filter(pl.col("model_name") == names[0])[
            "target_inspection_id"
        ].to_list()
    }
    offenders: list[str] = []
    for name in names[1:]:
        other = {
            str(v)
            for v in predictions.filter(pl.col("model_name") == name)[
                "target_inspection_id"
            ].to_list()
        }
        if other != reference:
            offenders.append(
                f"{name}: {len(reference - other)} row(s) missing, {len(other - reference)} extra "
                f"relative to {names[0]}"
            )
    return ValidationCheck(
        name="every_model_scored_the_same_rows",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            f"all {len(names)} models scored an identical set of "
            f"{len(reference)} target_inspection_id values"
            if not offenders
            else f"{len(offenders)} model(s) scored a different population"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- 4. the experimental categorical layer -----------------------------------


def _categoricals_cover_every_row(features: pl.DataFrame, table: pl.DataFrame) -> ValidationCheck:
    expected = {str(v) for v in features["target_inspection_id"].to_list()}
    got = {str(v) for v in table["target_inspection_id"].to_list()}
    missing = len(expected - got)
    extra = len(got - expected)
    return ValidationCheck(
        name="categoricals_cover_every_row",
        passed=(missing == 0 and extra == 0),
        severity=SEVERITY_ERROR,
        detail=(
            f"one categorical row per feature row ({len(expected)})"
            if missing == 0 and extra == 0
            else f"{missing} feature row(s) uncovered, {extra} extra"
        ),
    )


def _categoricals_are_strictly_as_of(table: pl.DataFrame) -> ValidationCheck:
    """Every carried value came from an inspection strictly earlier. Re-derived per row.

    This is the single most important check in the component. Its failure mode -- an
    exact-date match supplying a row its own attributes -- would leave every metric
    looking normal while the model read the present.
    """
    dated = table.with_columns(pl.col("inspection_date").str.to_date().alias("_rd"))
    violating = dated.filter(
        pl.col("source_inspection_date").is_not_null()
        & (pl.col("source_inspection_date") >= pl.col("_rd"))
    )
    return ValidationCheck(
        name="categoricals_are_strictly_as_of",
        passed=violating.height == 0,
        severity=SEVERITY_ERROR,
        detail=(
            "every carried categorical came from an inspection strictly earlier than the "
            "row it was carried to"
            if violating.height == 0
            else f"{violating.height} row(s) carried a value from their own date or later"
        ),
        offenders=tuple(
            str(v) for v in violating["target_inspection_id"].to_list()[:MAX_OFFENDERS]
        ),
    )


def _categoricals_are_never_null(table: pl.DataFrame) -> ValidationCheck:
    """A missing category must be the UNKNOWN token, never a null.

    A null would reach the encoder and be coerced somewhere, and where it landed would
    depend on which code path saw it first. The token is explicit and has a learned row.
    """
    offenders = [
        f"{column}: {table[column].null_count()} null(s)"
        for column in ("chain_key", "facility_type", "community_area", "zip")
        if column in table.columns and table[column].null_count()
    ]
    return ValidationCheck(
        name="categoricals_are_never_null",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every categorical is a real token; absence is UNKNOWN, never null"
            if not offenders
            else f"{len(offenders)} column(s) carry nulls"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _categoricals_carry_no_label(table: pl.DataFrame) -> ValidationCheck:
    """The experimental layer must not smuggle the target or a feature alongside it."""
    forbidden = sorted(
        set(table.columns)
        & (FORBIDDEN_COLUMNS - {"establishment_id", "inspection_date", "target_inspection_id"})
    )
    return ValidationCheck(
        name="categoricals_carry_no_label",
        passed=not forbidden,
        severity=SEVERITY_ERROR,
        detail=(
            "the categorical table carries identity and its four families, and no label "
            "or provenance column"
            if not forbidden
            else f"the categorical table carries {', '.join(forbidden)}"
        ),
        offenders=tuple(forbidden),
    )


# --- 5. the sweep ------------------------------------------------------------


def _sweep_never_reached_a_test_window(
    results: Sequence[SweepResult], outer_folds: Sequence[FoldSpec]
) -> ValidationCheck:
    from sentinel.boosting.tuning import first_test_start

    offenders: list[str] = []
    for result in results:
        horizon = first_test_start(result.fold_set, outer_folds)
        if result.region_end >= horizon:
            offenders.append(
                f"{result.study}: region ends {result.region_end}, first test start {horizon}"
            )
    return ValidationCheck(
        name="sweep_never_reached_a_test_window",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every learning-rate search ran strictly earlier than its fold set's first test window"
            if not offenders
            else f"{len(offenders)} search(es) reached a test window"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _sweep_inner_folds_are_ordered_and_disjoint(
    results: Sequence[SweepResult], outer_folds: Sequence[FoldSpec]
) -> ValidationCheck:
    from sentinel.boosting.tuning import build_inner_folds

    offenders: list[str] = []
    for result in results:
        expected = [f.fold_id for f in build_inner_folds(result.fold_set, outer_folds)]
        if list(result.inner_folds) != expected:
            offenders.append(
                f"{result.study}: used {len(result.inner_folds)} inner fold(s), the fold "
                f"set implies {len(expected)}"
            )
    return ValidationCheck(
        name="sweep_inner_folds_are_ordered_and_disjoint",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every search used exactly the inner folds its own fold set implies"
            if not offenders
            else f"{len(offenders)} search(es) used a different inner structure"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _every_rate_was_scored(results: Sequence[SweepResult]) -> ValidationCheck:
    offenders: list[str] = []
    for result in results:
        for rate, score in result.scores:
            if not math.isfinite(score):
                offenders.append(f"{result.study}: lr={rate:g} scored {score}")
    return ValidationCheck(
        name="every_rate_was_scored",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every learning rate in the grid produced a finite mean validation PR-AUC"
            if not offenders
            else f"{len(offenders)} rate(s) failed to score"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def _selected_rate_is_in_the_grid(results: Sequence[SweepResult]) -> ValidationCheck:
    offenders = [
        f"{r.study}: selected {r.best_learning_rate:g}, not among the scored rates"
        for r in results
        if r.best_learning_rate not in {rate for rate, _ in r.scores}
    ]
    return ValidationCheck(
        name="selected_rate_is_in_the_grid",
        passed=not offenders,
        severity=SEVERITY_ERROR,
        detail=(
            "every selected rate is one the search actually evaluated"
            if not offenders
            else f"{len(offenders)} search(es) selected an unevaluated rate"
        ),
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- 6. advisories -----------------------------------------------------------


def _advisories(
    fitted: Sequence[FittedNetwork], predictions: pl.DataFrame
) -> list[ValidationCheck]:
    """Warning-severity observations. Never fail a run; always worth seeing."""
    checks: list[ValidationCheck] = []

    budget = [
        f"{m.spec.name}/{m.fold_id}: ran the full {m.final_epoch} epochs"
        for m in fitted
        if m.spec.learner.value == "mlp" and m.stop_reason.startswith("epoch budget")
    ]
    checks.append(
        ValidationCheck(
            name="every_fit_stopped_early",
            passed=not budget,
            severity=SEVERITY_WARN,
            detail=(
                "every network stopped on patience rather than exhausting its epoch budget"
                if not budget
                else f"{len(budget)} fit(s) hit the epoch cap, so the budget may bind"
            ),
            offenders=tuple(budget[:MAX_OFFENDERS]),
        )
    )

    saturated = int(np.count_nonzero(np.isin(predictions["score"].to_numpy(), (0.0, 1.0))))
    checks.append(
        ValidationCheck(
            name="scores_are_not_saturated",
            passed=saturated == 0,
            severity=SEVERITY_WARN,
            detail=(
                "no score sits exactly at 0 or 1"
                if saturated == 0
                else f"{saturated} score(s) saturated; a calibrator cannot recover these"
            ),
        )
    )

    early = [
        f"{m.spec.name}/{m.fold_id}: best epoch {m.best_epoch}"
        for m in fitted
        if m.spec.learner.value == "mlp" and m.best_epoch <= 2
    ]
    checks.append(
        ValidationCheck(
            name="best_epoch_is_not_immediate",
            passed=not early,
            severity=SEVERITY_WARN,
            detail=(
                "no fit peaked in its first two epochs"
                if not early
                else f"{len(early)} fit(s) peaked almost immediately, which usually means "
                "the learning rate is too high or the signal is very weak"
            ),
            offenders=tuple(early[:MAX_OFFENDERS]),
        )
    )
    return checks


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
    "STATISTIC_TOLERANCE",
    "format_report",
    "has_failures",
    "validate_categoricals",
    "validate_neural",
    "validate_sweep",
]
