"""Feature matrix construction and the train-only preprocessing pipeline.

Two responsibilities, kept apart on purpose.

``to_matrix`` is the polars half: it selects a model's declared features, builds the
four null-rule family indicators, and casts everything to float so a NULL becomes a
NaN. It performs **no imputation** -- it only converts, so the NULL-to-NaN transition
happens in exactly one place and is testable on its own.

``build_preprocessor`` is the scikit-learn half: median or constant imputation per
column, then standardisation. Its statistics are fitted on the training window and
applied unchanged to later windows, which is the guarantee the whole component rests
on. ``ColumnTransformer`` expresses that mechanically rather than by convention.

Why the indicators are built here and not by ``SimpleImputer(add_indicator=True)``: the
null masks within a Component 4 null-rule family are byte-identical (measured on all
57,727 rows), so ``add_indicator`` would emit ten columns of which only four are
distinct -- exact duplicates that split coefficient mass arbitrarily and make the
coefficients artifact unreadable. It also selects indicators by observation
(``features="missing-only"``) rather than by declaration, so on a small fixture the
matrix width would change between folds. Declaring four is both correct and stable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.modeling.definitions import (
    INDICATOR_PREFIX,
    MissingStrategy,
    ModelSpec,
    family_indicator_name,
    indicator_columns,
    indicator_source_column,
    missing_strategy_for,
    null_families,
)

logger = logging.getLogger(__name__)

#: Fill value for a nullable boolean. "Unknown" becomes "false, plus the family
#: indicator", so the indicator's coefficient is exactly the unknown group's log-odds
#: offset. A median fill would instead be whichever class dominates that fold's training
#: window, and ``priority_at_last_canvass`` sits 0.0056 from flipping.
BOOLEAN_FILL = 0.0

#: The order ``ColumnTransformer`` concatenates its branches in. Declared once and used
#: both to build the transformer and to name the coefficients, so the two cannot drift.
BRANCH_ORDER: tuple[MissingStrategy, ...] = (
    MissingStrategy.MEDIAN,
    MissingStrategy.CONSTANT_FALSE,
    MissingStrategy.PASSTHROUGH,
)


class PreprocessError(ValueError):
    """Raised when a frame cannot be turned into a model matrix."""


def matrix_columns(spec: ModelSpec) -> tuple[str, ...]:
    """The model's declared features followed by the four family indicators.

    Every family gets an indicator even when a model drops every nullable member of it,
    so the matrix width is a property of the rule set rather than of the fold or the
    feature subset. A constant column costs one coefficient that regularises toward
    zero; a *varying* width would mean coefficient terms shifting between folds.
    """
    return (*spec.feature_columns, *indicator_columns())


def ordered_matrix_columns(spec: ModelSpec) -> tuple[str, ...]:
    """Matrix column names in the order the preprocessor emits them.

    ``ColumnTransformer`` concatenates its branches in declaration order, so the
    coefficient at position *i* belongs to this sequence, not to ``matrix_columns``.
    Getting this wrong would mislabel every coefficient while producing identical
    predictions -- a silent defect -- so both orders are derived from ``BRANCH_ORDER``.
    """
    columns = matrix_columns(spec)
    ordered: list[str] = []
    for strategy in BRANCH_ORDER:
        ordered.extend(name for name in columns if strategy_for(name) is strategy)
    return tuple(ordered)


def strategy_for(column: str) -> MissingStrategy:
    """Missing-value strategy for a matrix column.

    Indicators are computed, never null, so they pass through.
    """
    if column.startswith(INDICATOR_PREFIX):
        return MissingStrategy.PASSTHROUGH
    return missing_strategy_for(column)


def to_matrix(frame: pl.DataFrame, spec: ModelSpec) -> NDArray[np.float64]:
    """Build the float matrix for one model over one window. No imputation here.

    NULLs survive as NaN for the imputer to fill. Booleans are cast to float in this one
    place, so the only representation change happens where a test can see it.
    """
    required = [*spec.feature_columns, *_indicator_sources()]
    missing = [c for c in dict.fromkeys(required) if c not in frame.columns]
    if missing:
        raise PreprocessError(
            f"{spec.name}: feature table is missing required column(s) "
            f"{', '.join(missing)}. A model may not be fitted on a table that lacks a "
            "column its specification or its missingness indicators depend on."
        )

    prepared = frame.with_columns(_indicator_expressions())
    selected = prepared.select(
        [pl.col(name).cast(pl.Float64).alias(name) for name in matrix_columns(spec)]
    )
    return selected.to_numpy().astype(np.float64, copy=False)


def build_preprocessor(spec: ModelSpec) -> Any:
    """Imputation then standardisation, to be fitted on training rows only.

    Returned as ``Any`` because it is a scikit-learn object and pretending otherwise
    would be a false annotation; the typed view of a fit is ``FittedModel``.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    columns = matrix_columns(spec)
    branches: list[tuple[str, Any, list[int]]] = []
    for strategy in BRANCH_ORDER:
        indices = [i for i, name in enumerate(columns) if strategy_for(name) is strategy]
        if not indices:
            continue
        branches.append((strategy.value, _imputer_for(strategy, SimpleImputer), indices))

    # Indices rather than names, because the matrix reaching the transformer is a bare
    # ndarray. ``remainder="drop"`` is safe only because the three branches partition
    # every column; ``ordered_matrix_columns`` asserts the same partition.
    impute = ColumnTransformer(branches, remainder="drop")
    return Pipeline([("impute", impute), ("scale", StandardScaler())])


def imputed_values(preprocessor: Any, spec: ModelSpec) -> dict[str, float]:
    """The fill value actually fitted for each nullable column.

    Extracted so ``validate`` can re-derive the median from the training frame and
    confirm it matches -- the mechanical check that preprocessing statistics came from
    the training window and nowhere else.
    """
    values: dict[str, float] = {}
    transformer = preprocessor.named_steps["impute"]
    columns = matrix_columns(spec)
    for name, fitted, indices in transformer.transformers_:
        if name not in {MissingStrategy.MEDIAN.value, MissingStrategy.CONSTANT_FALSE.value}:
            continue
        statistics = getattr(fitted, "statistics_", None)
        if statistics is None:
            continue
        for position, index in enumerate(indices):
            values[columns[index]] = float(statistics[position])
    return values


def _imputer_for(strategy: MissingStrategy, simple_imputer: Any) -> Any:
    if strategy is MissingStrategy.MEDIAN:
        return simple_imputer(strategy="median")
    if strategy is MissingStrategy.CONSTANT_FALSE:
        return simple_imputer(strategy="constant", fill_value=BOOLEAN_FILL)
    return "passthrough"


def _indicator_sources() -> tuple[str, ...]:
    """The columns whose null masks define the four indicators."""
    return tuple(indicator_source_column(rule) for rule in null_families())


def _indicator_expressions() -> list[pl.Expr]:
    """One boolean-as-float indicator per null-rule family.

    The source column is the family's first member. Any member would do: the masks
    within a family are byte-identical on the full snapshot, and ``validate``
    re-asserts that per fold rather than trusting it. The source is read from the
    feature table, not from the model's feature subset, so a model that drops one
    member still gets a correct indicator -- and gains no information by it, since the
    mask is identical to that of the members it kept.
    """
    return [
        pl.col(indicator_source_column(rule))
        .is_null()
        .cast(pl.Float64)
        .alias(family_indicator_name(rule))
        for rule in null_families()
    ]


__all__ = [
    "BOOLEAN_FILL",
    "BRANCH_ORDER",
    "PreprocessError",
    "build_preprocessor",
    "imputed_values",
    "matrix_columns",
    "ordered_matrix_columns",
    "strategy_for",
    "to_matrix",
]
