"""Output schemas for Component 9's nine tables.

Column order is part of the data contract; changing it is a contract change.

The tables land in three directories, because ADR 0014 and ADR 0018 each named a home for
their grain in advance and ADR 0024 honours both:

    calibrated_predictions          -> data/processed/predictions/   (ADR 0014)
    calibrator_selection            -> data/processed/tuning/        (ADR 0018)
    everything else                 -> data/processed/calibration/   (ADR 0024)

``calibrated_predictions`` is deliberately shaped to be readable by
``evaluation.contract.read_predictions`` without translation. It carries the contract's two
columns plus the metadata every prediction file carries, and the extra Component 9 columns
are invisible to ``validate_predictions`` -- which selects only ``PREDICTION_COLUMNS`` into
a ``PredictionSet``. So ``sentinel evaluate --predictions <this file>`` works with **no
change to Component 5**.

No table carries a timestamp or a duration, matching ``modeling/writer.py``'s rule -- wall
clock in a Parquet file would mean two runs over identical inputs producing different bytes.
The one exception is ``calibrator_selection.seconds``, under the narrow exception ADR 0018
already granted the tuning layer, which makes no determinism claim about its bytes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: One row per (calibrated model, fold, scored test row). The artifact Component 5 consumes
#: and Component 10 starts from.
#:
#: ``model_name`` is ``"<base>_<method>"``, never a bare base name, so a calibrated row and
#: its uncalibrated ancestor can sit in one results table without either being mistaken for
#: the other.
#:
#: The three horizon columns exist because one of them alone would be misleading.
#: ``trained_through`` is the contract field and is the *maximum* of the other two, which is
#: the only honest single number; writing ``train_end`` there would be false, because the
#: calibrator really did read the calibration window.
CALIBRATED_PREDICTIONS_SCHEMA: dict[str, pl.DataType] = {
    "target_inspection_id": pl.Utf8(),
    "score": pl.Float64(),
    "model_name": pl.Utf8(),
    "model_version": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    # == fold.calibration_end. At the contract's ceiling, not past it. See ADR 0024.
    "trained_through": pl.Date(),
    "is_probability": pl.Boolean(),
    "base_model_name": pl.Utf8(),
    "base_model_version": pl.Utf8(),
    # The uncalibrated probability, carried so the correction is always visible and a
    # consumer never has to join back to Component 6/7/8 to see what changed.
    "base_score": pl.Float64(),
    # == fold.train_end: what the estimator's weights learned from.
    "base_model_trained_through": pl.Date(),
    # == fold.calibration_end: what the two-parameter correction learned from.
    "calibrator_fitted_through": pl.Date(),
    # == fold.test_start: the first date this score could have been produced in operation.
    "calibrated_prediction_available_from": pl.Date(),
    "method": pl.Utf8(),
    "is_experimental": pl.Boolean(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (base model, fold, window, scored row): the scores Component 9 had to
#: regenerate because no component persisted them (ADR 0026).
#:
#: ``reproduces_committed_artifact`` is null on calibration rows -- there is nothing to
#: compare them against, which is the whole reason this component exists.
BASE_SCORES_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "split": pl.Utf8(),
    "inner_portion": pl.Utf8(),
    "target_inspection_id": pl.Utf8(),
    "rd": pl.Date(),
    "base_score": pl.Float64(),
    "base_logit": pl.Float64(),
    # Null where the family's native margin is not reachable without widening a closed
    # component's API -- see basescores._neural_embedding_booster.
    "native_margin": pl.Float64(),
    "target": pl.Int8(),
    "reproduces_committed_artifact": pl.Boolean(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (model, fold, method, parameter). Long form, mirroring the ``__intercept__``
#: convention ``modeling/writer.py`` chose for coefficients rather than widening.
CALIBRATOR_PARAMETERS_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "method": pl.Utf8(),
    "term": pl.Utf8(),
    "value": pl.Float64(),
    "fit_rows": pl.Int32(),
    "fit_positive_rate": pl.Float64(),
    "fit_start": pl.Date(),
    "fit_end": pl.Date(),
    "input_transform": pl.Utf8(),
    "was_selected": pl.Boolean(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (model, fold, breakpoint). With ``np.interp`` and the clip bounds these two
#: columns reproduce the isotonic map exactly, which is what makes the calibrator an
#: artifact rather than a black box.
#:
#: Written for every fold where isotonic was **fitted**, not only where it won, so the
#: counterfactual stays answerable without a re-run.
ISOTONIC_BREAKPOINTS_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "breakpoint_index": pl.Int32(),
    "x_threshold": pl.Float64(),
    "y_threshold": pl.Float64(),
    "x_min": pl.Float64(),
    "x_max": pl.Float64(),
    "breakpoint_count": pl.Int32(),
    "was_selected": pl.Boolean(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (model, fold, method): what each candidate scored on the inner-select window,
#: and which was frozen. ADR 0018's layer, and ADR 0018's ``seconds`` exception.
#:
#: **No number in this table is a result.** Every one was measured on a window carved out of
#: the calibration period, and only ``inner_select_log_loss`` decides anything.
SELECTION_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "fold_index": pl.Int32(),
    "method": pl.Utf8(),
    "inner_fit_rows": pl.Int32(),
    "inner_select_rows": pl.Int32(),
    "inner_split_date": pl.Date(),
    "inner_fit_positive_rate": pl.Float64(),
    "inner_select_positive_rate": pl.Float64(),
    "inner_select_log_loss": pl.Float64(),
    "inner_select_brier": pl.Float64(),
    "inner_select_ece": pl.Float64(),
    "inner_select_mce": pl.Float64(),
    "prefix_mean_log_loss": pl.Float64(),
    "per_fold_winner": pl.Boolean(),
    "prefix_winner": pl.Boolean(),
    "gap_to_other": pl.Float64(),
    "tie_threshold": pl.Float64(),
    "declared_tie": pl.Boolean(),
    "selection_reason": pl.Utf8(),
    "seconds": pl.Float64(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (model, fold, stage). The quarter-by-quarter calibration drift the project
#: specification asks for, and the table the retraining trigger is defined against.
#:
#: All four stages are written -- uncalibrated, platt, isotonic, selected -- so "would
#: isotonic have been better on this quarter?" is answerable from the artifact instead of by
#: re-running with a different flag, which is how a selection becomes a test-set selection.
DRIFT_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "fold_index": pl.Int32(),
    "stage": pl.Utf8(),
    "test_start": pl.Date(),
    "test_end": pl.Date(),
    "test_rows": pl.Int32(),
    "test_positive_rate": pl.Float64(),
    "calibration_positive_rate": pl.Float64(),
    "prior_shift": pl.Float64(),
    "ece": pl.Float64(),
    "mce": pl.Float64(),
    "brier": pl.Float64(),
    "log_loss": pl.Float64(),
    "calibration_slope": pl.Float64(),
    "calibration_intercept": pl.Float64(),
    "mean_predicted": pl.Float64(),
    "observed_rate": pl.Float64(),
    "n_bins": pl.Int32(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (model, fold, stage). Whether the calibrator reordered anything.
#:
#: ``inversions`` and ``new_ties_created`` are separate columns on purpose. A monotone map
#: cannot invert, but isotonic's plateaus create ties, and ``top_k_indices`` settles ties by
#: ``target_inspection_id`` -- so top-k membership can move with zero inversions. That is a
#: tie, not a ranking inversion, and conflating them would misreport a correct calibrator as
#: a broken one.
RANKING_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "stage": pl.Utf8(),
    "spearman_rho": pl.Float64(),
    "kendall_tau_b": pl.Float64(),
    "inversions": pl.Int64(),
    "distinct_scores_before": pl.Int32(),
    "distinct_scores_after": pl.Int32(),
    "new_ties_created": pl.Int32(),
    "is_strictly_monotone": pl.Boolean(),
    "top_k": pl.Int32(),
    "top_k_name": pl.Utf8(),
    "top_k_membership_changed": pl.Int32(),
    "precision_at_k_before": pl.Float64(),
    "precision_at_k_after": pl.Float64(),
    "roc_auc_before": pl.Float64(),
    "roc_auc_after": pl.Float64(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (model, fold, stage). Murphy's decomposition over 15 equal-mass bins.
#:
#: ``within_bin_variance`` is the residual, and it is a column rather than a rounding error:
#: ``BS = REL - RES + UNC`` is exact only for a forecast constant within each bin, and
#: reporting the recomposition as "the Brier score" would be a fabrication.
DECOMPOSITION_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "stage": pl.Utf8(),
    "n_bins": pl.Int32(),
    "binning": pl.Utf8(),
    "brier": pl.Float64(),
    "reliability": pl.Float64(),
    "resolution": pl.Float64(),
    "uncertainty": pl.Float64(),
    "recomposed": pl.Float64(),
    "within_bin_variance": pl.Float64(),
    "calibration_definition_version": pl.Utf8(),
}

#: One row per (model, fold, stage, metric, scheme). Within-fold percentile intervals.
#:
#: ``scheme`` is not a detail: rows are not independent within a window, so the row and
#: establishment-block schemes are both run and both reported rather than one being chosen
#: and the other left as a caveat.
BOOTSTRAP_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "stage": pl.Utf8(),
    "metric": pl.Utf8(),
    "scheme": pl.Utf8(),
    "point_estimate": pl.Float64(),
    "replications": pl.Int32(),
    "seed": pl.Int32(),
    "bootstrap_mean": pl.Float64(),
    "bootstrap_sd": pl.Float64(),
    "ci_lower": pl.Float64(),
    "ci_upper": pl.Float64(),
    "ci_level": pl.Float64(),
    "degenerate_replications": pl.Int32(),
    "calibration_definition_version": pl.Utf8(),
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "calibrated_predictions": CALIBRATED_PREDICTIONS_SCHEMA,
    "calibration_base_scores": BASE_SCORES_SCHEMA,
    "calibrator_parameters": CALIBRATOR_PARAMETERS_SCHEMA,
    "calibrator_isotonic_breakpoints": ISOTONIC_BREAKPOINTS_SCHEMA,
    "calibrator_selection": SELECTION_SCHEMA,
    "calibration_drift": DRIFT_SCHEMA,
    "calibration_ranking_preservation": RANKING_SCHEMA,
    "calibration_brier_decomposition": DECOMPOSITION_SCHEMA,
    "calibration_bootstrap": BOOTSTRAP_SCHEMA,
}

#: Which directory each table belongs to. ADR 0024.
LAYERS: dict[str, str] = {
    "calibrated_predictions": "predictions",
    "calibrator_selection": "tuning",
    "calibration_base_scores": "calibration",
    "calibrator_parameters": "calibration",
    "calibrator_isotonic_breakpoints": "calibration",
    "calibration_drift": "calibration",
    "calibration_ranking_preservation": "calibration",
    "calibration_brier_decomposition": "calibration",
    "calibration_bootstrap": "calibration",
}

#: The anchor artifact each manifest is keyed to, per directory.
DATASET_SLUG = "calibrated_predictions"
SELECTION_SLUG = "calibrator_selection"
DIAGNOSTICS_SLUG = "calibration_base_scores"

#: Sort keys per table, so two runs over the same data produce byte-identical files. Row
#: order is the last place non-determinism could hide.
SORT_KEYS: dict[str, list[str]] = {
    "calibrated_predictions": ["model_name", "fold_set", "fold_id", "target_inspection_id"],
    "calibration_base_scores": [
        "model_name",
        "fold_set",
        "fold_id",
        "split",
        "target_inspection_id",
    ],
    "calibrator_parameters": ["model_name", "fold_set", "fold_id", "method", "term"],
    "calibrator_isotonic_breakpoints": ["model_name", "fold_set", "fold_id", "breakpoint_index"],
    "calibrator_selection": ["model_name", "fold_set", "fold_id", "method"],
    "calibration_drift": ["model_name", "fold_set", "fold_id", "stage"],
    "calibration_ranking_preservation": ["model_name", "fold_set", "fold_id", "stage"],
    "calibration_brier_decomposition": ["model_name", "fold_set", "fold_id", "stage"],
    "calibration_bootstrap": [
        "model_name",
        "fold_set",
        "fold_id",
        "stage",
        "metric",
        "scheme",
    ],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards:
    ``calibration_slope`` is null on a single-class window and ``native_margin`` is null for
    one family, so inference would type those columns from whichever kind of value happened
    to come first.
    """
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    schema = SCHEMAS[table]
    if not rows:
        return empty(table)
    missing = [c for c in schema if c not in rows[0]]
    if missing:
        raise ValueError(f"{table} rows are missing columns: {', '.join(missing)}")
    extra = [c for c in rows[0] if c not in schema]
    if extra:
        raise ValueError(f"{table} rows carry unknown columns: {', '.join(extra)}")
    return pl.DataFrame(rows, schema=schema).sort(SORT_KEYS[table])


def empty(table: str) -> pl.DataFrame:
    """A correctly typed zero-row frame, so a reader meets the right columns."""
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    return pl.DataFrame(schema=SCHEMAS[table])


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write a table as zstd Parquet, matching project convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows, %d columns)", path, frame.height, frame.width)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    """Column name to dtype string, for recording in the manifest."""
    return {name: str(dtype) for name, dtype in frame.schema.items()}


__all__ = [
    "BASE_SCORES_SCHEMA",
    "BOOTSTRAP_SCHEMA",
    "CALIBRATED_PREDICTIONS_SCHEMA",
    "CALIBRATOR_PARAMETERS_SCHEMA",
    "DATASET_SLUG",
    "DECOMPOSITION_SCHEMA",
    "DIAGNOSTICS_SLUG",
    "DRIFT_SCHEMA",
    "ISOTONIC_BREAKPOINTS_SCHEMA",
    "LAYERS",
    "RANKING_SCHEMA",
    "SCHEMAS",
    "SELECTION_SCHEMA",
    "SELECTION_SLUG",
    "SORT_KEYS",
    "empty",
    "finalize",
    "schema_of",
    "write_table",
]
