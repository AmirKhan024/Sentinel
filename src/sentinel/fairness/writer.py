"""Parquet schemas and deterministic writing for Component 12.

Column order is part of the data contract; changing it is a contract change. See
``docs/data_contracts/fairness_audit.md``.

Ten tables in one new processed layer (ADR 0032). The split follows the grain of the question
each answers, and two of the splits are load-bearing rather than tidy.

**Support is its own table** because it is model-independent. Rows, positives and base rate
are properties of the fold and the group; repeating them on every (model, stage, metric) row
would copy one measured fact across roughly two hundred rows and invite the copies to
disagree.

**Calibration before-and-after is its own table** because the question section 18 of the brief
asks -- *did the global improvement reach this group?* -- should be one column rather than a
pivot a reader has to construct correctly.

**Every metric row carries its support counts anyway.** That is the one deliberate
denormalisation here, and it is the component's central discipline: a metric value without the
row count behind it cannot be read, and the easiest way for a fairness table to mislead is a
dramatic ratio from a group of twelve rows quoted without its ``n_rows``. zstd stores a
repeated integer for almost nothing; a reader who quotes an unsupported number costs more.

**Unsupported groups are rows with a null value and a stated reason, never absent rows.**
Nothing in this module filters on ``group_status``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: The table the manifest is keyed to, matching the convention every component follows.
DATASET_SLUG = "fairness_group_metrics"


#: One row per candidate group definition -- audited **and refused**. The refusals are data
#: rather than prose so that a reader who opens the Parquet instead of ADR 0033 still finds
#: out why there is no ward breakdown, and finds the measurement that decided it.
GROUP_DEFINITIONS_SCHEMA: dict[str, pl.DataType] = {
    "group_definition": pl.Utf8(),
    "status": pl.Utf8(),
    "source_column": pl.Utf8(),
    "provenance": pl.Utf8(),
    "rationale": pl.Utf8(),
    # False for every row, and emitted anyway: it is the fact that stops "the model does not
    # use community area, therefore it is fair" being read into the artifact.
    "is_model_feature": pl.Boolean(),
    "refusal_reason": pl.Utf8(),
    "distinct_values": pl.Int64(),
    "unknown_rows": pl.Int64(),
    "audited_rows": pl.Int64(),
    "fairness_definition_version": pl.Utf8(),
}

#: One row per (group definition, group, grain, fold). Model-independent.
GROUP_SUPPORT_SCHEMA: dict[str, pl.DataType] = {
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    # Empty string at fold_set grain rather than null, so the column's meaning never depends
    # on the grain -- the convention Component 5 uses for ``k_name``.
    "fold_id": pl.Utf8(),
    "n_rows": pl.Int64(),
    "n_positive": pl.Int64(),
    "n_negative": pl.Int64(),
    # Null on a zero-row group. 0.0 is a legitimate base rate and would not mean the same.
    "base_rate": pl.Float64(),
    "representation_share": pl.Float64(),
    "ranking_status": pl.Utf8(),
    "calibration_status": pl.Utf8(),
    "insufficient_reason": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: The long grain: one row per (model, stage, definition, group, grain, fold, metric, k).
GROUP_METRICS_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    # 'base' or 'calibrated'. The two probabilities live side by side on Component 9's
    # artifact, so this column is the only thing separating them and confusing it would
    # produce entirely plausible numbers answering the wrong question.
    "stage": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "metric": pl.Utf8(),
    "metric_kind": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    # Null when the group missed its floor or the metric is undefined on its rows. Never a
    # substituted 0.5 or 0.0.
    "value": pl.Float64(),
    "n_rows": pl.Int64(),
    "n_positive": pl.Int64(),
    "n_negative": pl.Int64(),
    "group_status": pl.Utf8(),
    "insufficient_reason": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: Component 9's Platt map, per group. ``improved`` is null rather than false when either
#: side is missing: "we could not tell" and "it got worse" are different answers.
GROUP_CALIBRATION_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "metric": pl.Utf8(),
    "base_value": pl.Float64(),
    "calibrated_value": pl.Float64(),
    "delta": pl.Float64(),
    "improved": pl.Boolean(),
    "n_rows": pl.Int64(),
    "n_positive": pl.Int64(),
    "group_status": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: Representation in the top k, and what that prioritisation captured. Both, never combined.
PRIORITY_AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "stage": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "k_name": pl.Utf8(),
    "k": pl.Int64(),
    "n_rows": pl.Int64(),
    "n_positive": pl.Int64(),
    "population_share": pl.Float64(),
    "n_selected": pl.Int64(),
    "selected_share": pl.Float64(),
    "selection_rate": pl.Float64(),
    # Null rather than infinite on a zero denominator, so "nothing to select from" and
    # "selected at an enormous rate" stay distinguishable.
    "selection_rate_ratio": pl.Float64(),
    "positives_selected": pl.Int64(),
    "precision_in_selected": pl.Float64(),
    # Null when the group has no positives -- never 0.0, which would read as total failure
    # rather than as the absence of anything to capture.
    "capture_rate": pl.Float64(),
    "overall_capture_rate": pl.Float64(),
    "group_status": pl.Utf8(),
    "insufficient_reason": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: Data availability by group, and inside the priority set.
GROUP_MISSINGNESS_SCHEMA: dict[str, pl.DataType] = {
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "indicator": pl.Utf8(),
    "source_column": pl.Utf8(),
    "n_rows": pl.Int64(),
    "n_missing": pl.Int64(),
    "missing_rate": pl.Float64(),
    "overall_missing_rate": pl.Float64(),
    "deviation": pl.Float64(),
    "missing_rate_in_top_k": pl.Float64(),
    "k_name": pl.Utf8(),
    "group_status": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: Component 11's attributions, grouped. Descriptive; an attribution is not a quality
#: measure (ADR 0030) and a profile difference is not evidence of discrimination.
ATTRIBUTION_PROFILES_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "feature_name": pl.Utf8(),
    "mean_abs_shap": pl.Float64(),
    "mean_shap": pl.Float64(),
    "rank": pl.Int64(),
    "overall_rank": pl.Int64(),
    "rank_delta": pl.Int64(),
    "n_rows": pl.Int64(),
    "profile_spearman": pl.Float64(),
    # False for the network. A consumer that ignores this column over-trusts its per-row
    # values by roughly one percent of the largest attribution.
    "is_exact": pl.Boolean(),
    "group_status": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: Four measures per comparable cell, never one score.
DISPARITY_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "stage": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "fold_id": pl.Utf8(),
    "metric": pl.Utf8(),
    "k_name": pl.Utf8(),
    "measure": pl.Utf8(),
    "value": pl.Float64(),
    # The pooled population value, never a nominated group.
    "reference_value": pl.Float64(),
    "max_value": pl.Float64(),
    "max_group": pl.Utf8(),
    # The extremes carry their row counts so a dramatic ratio can never be quoted without
    # its support visible in the same record.
    "max_group_rows": pl.Int64(),
    "min_value": pl.Float64(),
    "min_group": pl.Utf8(),
    "min_group_rows": pl.Int64(),
    "n_groups_supported": pl.Int64(),
    # A spread over 51 of 78 groups is a different claim from one over all 78, and the two
    # are indistinguishable without this column.
    "n_groups_unsupported": pl.Int64(),
    "undefined_reason": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: Whether the disparity itself moved across a fold set's folds.
FAIRNESS_DRIFT_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "stage": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "metric": pl.Utf8(),
    "k_name": pl.Utf8(),
    "measure": pl.Utf8(),
    # Measured vs available. Usually far apart, because per-fold group cells are thin.
    "folds_measured": pl.Int64(),
    "folds_total": pl.Int64(),
    "mean_spread": pl.Float64(),
    # A fold-to-fold spread, NOT a confidence interval: the folds overlap and share
    # establishments on a 358-day median canvass cycle.
    "sd_spread": pl.Float64(),
    "min_spread": pl.Float64(),
    "max_spread": pl.Float64(),
    "first_fold_id": pl.Utf8(),
    "first_spread": pl.Float64(),
    "last_fold_id": pl.Utf8(),
    "last_spread": pl.Float64(),
    "relative_change": pl.Float64(),
    "trend": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}

#: Deterministic intervals, for the two metrics where sampling variability changes a reading.
FAIRNESS_BOOTSTRAP_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "stage": pl.Utf8(),
    "group_definition": pl.Utf8(),
    "group_value": pl.Utf8(),
    "grain": pl.Utf8(),
    "fold_set": pl.Utf8(),
    "metric": pl.Utf8(),
    "k_name": pl.Utf8(),
    "point_estimate": pl.Float64(),
    "lower": pl.Float64(),
    "upper": pl.Float64(),
    "replications": pl.Int64(),
    "level": pl.Float64(),
    # Derived from the candidate's registry position, never from hash() of a string --
    # Python salts str hashing per process, which is what made Component 9's bootstrap
    # non-reproducible until the key changed. MEMORY invariant 92.
    "seed": pl.Int64(),
    "n_rows": pl.Int64(),
    # 'row' or 'establishment_block'. Both are run for every group: establishments recur
    # inside a neighbourhood and their rows share an as-of history, so an i.i.d. row
    # bootstrap understates the standard error. Running both settles that with a measurement.
    "scheme": pl.Utf8(),
    "fairness_definition_version": pl.Utf8(),
}


SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "fairness_group_definitions": GROUP_DEFINITIONS_SCHEMA,
    "fairness_group_support": GROUP_SUPPORT_SCHEMA,
    "fairness_group_metrics": GROUP_METRICS_SCHEMA,
    "fairness_group_calibration": GROUP_CALIBRATION_SCHEMA,
    "fairness_priority_audit": PRIORITY_AUDIT_SCHEMA,
    "fairness_group_missingness": GROUP_MISSINGNESS_SCHEMA,
    "fairness_attribution_profiles": ATTRIBUTION_PROFILES_SCHEMA,
    "fairness_disparity": DISPARITY_SCHEMA,
    "fairness_drift": FAIRNESS_DRIFT_SCHEMA,
    "fairness_bootstrap": FAIRNESS_BOOTSTRAP_SCHEMA,
}

#: Every table lands in the one new layer. Unlike Component 9, which had to write to three,
#: a group metric is not a prediction and not a trial, so there is nowhere else it could go.
LAYERS: dict[str, str] = dict.fromkeys(SCHEMAS, "fairness")

#: Full keys, so a sort is a total order and two runs over identical inputs produce
#: byte-identical files. A partial key would leave ties resolved by whatever order the rows
#: happened to be appended in, which is not a contract.
SORT_KEYS: dict[str, list[str]] = {
    "fairness_group_definitions": ["group_definition"],
    "fairness_group_support": [
        "group_definition",
        "grain",
        "fold_set",
        "fold_id",
        "group_value",
    ],
    "fairness_group_metrics": [
        "model_name",
        "stage",
        "group_definition",
        "grain",
        "fold_set",
        "fold_id",
        "metric",
        "k_name",
        "group_value",
    ],
    "fairness_group_calibration": [
        "model_name",
        "group_definition",
        "grain",
        "fold_set",
        "fold_id",
        "metric",
        "group_value",
    ],
    "fairness_priority_audit": [
        "model_name",
        "stage",
        "group_definition",
        "grain",
        "fold_set",
        "fold_id",
        "k_name",
        "group_value",
    ],
    "fairness_group_missingness": [
        "group_definition",
        "grain",
        "fold_set",
        "fold_id",
        "indicator",
        "group_value",
    ],
    "fairness_attribution_profiles": [
        "model_name",
        "group_definition",
        "fold_set",
        "group_value",
        "rank",
        "feature_name",
    ],
    "fairness_disparity": [
        "model_name",
        "stage",
        "group_definition",
        "grain",
        "fold_set",
        "fold_id",
        "metric",
        "k_name",
        "measure",
    ],
    "fairness_drift": [
        "model_name",
        "stage",
        "group_definition",
        "fold_set",
        "metric",
        "k_name",
        "measure",
    ],
    "fairness_bootstrap": [
        "model_name",
        "stage",
        "group_definition",
        "grain",
        "fold_set",
        "metric",
        "k_name",
        "group_value",
        # Two rows per group, one per resampling scheme, so the scheme is part of the key.
        "scheme",
    ],
}


def finalize(rows: list[dict[str, object]], table: str) -> pl.DataFrame:
    """Cast to the contract schema and order deterministically.

    The schema is passed to the ``DataFrame`` constructor rather than cast afterwards. This
    component emits more nulls than any other -- an unsupported group's value, a ratio with a
    vanished denominator, a capture rate for a group with no positives -- so inference would
    type a column from whichever kind of value happened to come first, and a column that is
    null for the first two hundred rows would arrive as ``Null`` rather than ``Float64``.
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
    """A correctly typed zero-row frame, so a reader meets the right columns.

    A real outcome here rather than a defensive one: a run restricted to one model may write
    an empty attribution table because Component 11 could not explain it, and the reader
    should still find the schema rather than a missing file.
    """
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
    "DATASET_SLUG",
    "LAYERS",
    "SCHEMAS",
    "SORT_KEYS",
    "empty",
    "finalize",
    "schema_of",
    "write_table",
]
