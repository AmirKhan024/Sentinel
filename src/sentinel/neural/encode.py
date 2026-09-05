"""Per-fold categorical vocabularies. Fitted on training rows only, every time.

A vocabulary is a fitted statistic in exactly the sense a scaler mean is, and it leaks in
exactly the same way. If the index for ``"TACO BELL"`` exists because Taco Bell appears
in the test window, the network has been told something about the future -- not the
label, but the *existence* of the category, which is enough to change what every
embedding row learns. So a vocabulary is built inside the fold, from the fold's training
rows, and rebuilt from scratch for the next one.

**Index 0 is UNKNOWN and it is a learned row, not a masked one.**

The usual objection to a learned out-of-vocabulary row is that it never receives
gradient, so it stays at its initialisation and quietly acts as a constant. That is not
the situation here. ``categoricals`` emits ``UNKNOWN`` for any row whose establishment has
no prior inspection to carry a value forward from -- 401 rows on the current snapshot,
and the same 401 rows Component 4 marks with a null ``days_since_any_inspection``. Those
rows are in the training window, they carry labels, and index 0 learns from them. What it
learns is the "we have never seen this place before" offset, which is a real thing to
know at scheduling time. Categories that appear only later map onto that same row, which
is the honest default: the model has no basis to say anything else about them.

**Chain membership is derived per fold, and that is not a detail.**

A chain is a set of establishments sharing a normalised name. Whether an establishment is
"part of a chain" therefore depends on which *other* establishments exist -- so computing
membership over the whole snapshot would let a second location, opened in 2025, change
what a 2022 fold's model saw about the first one. The membership set is computed from the
fold's training rows and nothing else, and an establishment whose name is unshared within
that window is ``__INDEPENDENT__``: a real category, because "not part of a chain" is a
fact worth conditioning on, not a null.

Vocabulary order is sorted, never insertion-ordered. Insertion order depends on row
order, and this project's determinism claim is bit-identity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.neural.definitions import (
    INDEPENDENT_CHAIN,
    UNKNOWN_CATEGORY,
    CategoricalEncoding,
    EntityFamily,
    NeuralSpec,
)

logger = logging.getLogger(__name__)

#: Reserved index for any category not present in a fold's training rows.
UNKNOWN_INDEX = 0


class EncodeError(ValueError):
    """Raised when a categorical frame cannot be encoded."""


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """One entity column's fitted category list. Index 0 is always UNKNOWN."""

    column: str
    categories: tuple[str, ...]

    @property
    def size(self) -> int:
        """Number of embedding rows, UNKNOWN included."""
        return len(self.categories)

    def index_of(self, value: str) -> int:
        """The index for one raw value, or ``UNKNOWN_INDEX`` if it was never trained on."""
        return self._lookup.get(value, UNKNOWN_INDEX)

    @property
    def _lookup(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.categories)}


@dataclass(frozen=True, slots=True)
class FoldEncoding:
    """Everything one fold's categorical encoding needs, fitted on training rows.

    ``chains`` is carried alongside the vocabularies because it is a second fitted
    object: the chain vocabulary is derived *from* it, and reproducing an encoding
    requires both.
    """

    vocabularies: tuple[Vocabulary, ...]
    chains: frozenset[str]
    train_rows: int

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(v.column for v in self.vocabularies)

    def vocabulary_for(self, column: str) -> Vocabulary:
        for vocab in self.vocabularies:
            if vocab.column == column:
                return vocab
        raise EncodeError(f"no vocabulary fitted for {column!r}")

    @property
    def sizes(self) -> dict[str, int]:
        return {v.column: v.size for v in self.vocabularies}


def chain_membership(train_categoricals: pl.DataFrame) -> frozenset[str]:
    """Normalised names shared by more than one establishment **in this window**.

    The whole temporal argument for the chain family sits in the word "window". A name
    counts as a chain only if two distinct ``establishment_id`` values carry it among the
    fold's training rows; a name that becomes shared later is not retroactively a chain.
    """
    if "chain_key" not in train_categoricals.columns:
        raise EncodeError("categorical frame has no chain_key column")
    grouped = (
        train_categoricals.filter(pl.col("chain_key") != UNKNOWN_CATEGORY)
        .group_by("chain_key")
        .agg(pl.col("establishment_id").n_unique().alias("establishments"))
        .filter(pl.col("establishments") > 1)
    )
    return frozenset(str(v) for v in grouped["chain_key"].to_list())


def chain_expression(chains: frozenset[str]) -> pl.Expr:
    """Map ``chain_key`` to the chain category, given one fold's membership set.

    Three outcomes, all real categories: UNKNOWN when nothing could be carried forward,
    the name itself when it is shared, and ``__INDEPENDENT__`` when it is not.
    """
    return (
        pl.when(pl.col("chain_key") == UNKNOWN_CATEGORY)
        .then(pl.lit(UNKNOWN_CATEGORY))
        .when(pl.col("chain_key").is_in(list(chains)))
        .then(pl.col("chain_key"))
        .otherwise(pl.lit(INDEPENDENT_CHAIN))
        .alias(EntityFamily.CHAIN.value)
    )


def resolve_categories(frame: pl.DataFrame, chains: frozenset[str]) -> pl.DataFrame:
    """Add the derived ``chain`` column to a categorical frame.

    The other three families are already their own categories; only chain needs the
    membership set to become one.
    """
    return frame.with_columns(chain_expression(chains))


def fit_encoding(train_categoricals: pl.DataFrame, spec: NeuralSpec) -> FoldEncoding:
    """Fit vocabularies and chain membership on one fold's training rows.

    ``train_categoricals`` must already be restricted to the fold's training window.
    Nothing in this function looks at a date, precisely so it cannot be given a wider
    frame and quietly do the right-looking thing: the caller's restriction is the
    guarantee, and ``validate`` re-derives it.
    """
    if spec.encoding is CategoricalEncoding.NONE:
        return FoldEncoding(
            vocabularies=(), chains=frozenset(), train_rows=train_categoricals.height
        )
    if train_categoricals.height == 0:
        raise EncodeError(f"{spec.name}: cannot fit a vocabulary on an empty training window")

    chains = chain_membership(train_categoricals)
    resolved = resolve_categories(train_categoricals, chains)

    vocabularies: list[Vocabulary] = []
    for column in spec.entity_columns:
        if column not in resolved.columns:
            raise EncodeError(f"{spec.name}: categorical frame has no column {column!r}")
        observed = {str(v) for v in resolved[column].drop_nulls().to_list()}
        observed.discard(UNKNOWN_CATEGORY)
        # UNKNOWN first, then everything else sorted. Sorted rather than insertion
        # ordered because insertion order is row order, and row order is exactly the
        # thing this project refuses to let reach a fitted artifact.
        categories = (UNKNOWN_CATEGORY, *sorted(observed))
        vocabularies.append(Vocabulary(column=column, categories=categories))

    encoding = FoldEncoding(
        vocabularies=tuple(vocabularies),
        chains=chains,
        train_rows=train_categoricals.height,
    )
    logger.info(
        "%s: fitted vocabularies on %d training rows: %s (%d chains)",
        spec.name,
        train_categoricals.height,
        ", ".join(f"{k}={v}" for k, v in encoding.sizes.items()),
        len(chains),
    )
    return encoding


def index_matrix(
    frame: pl.DataFrame, spec: NeuralSpec, encoding: FoldEncoding
) -> NDArray[np.int64]:
    """Integer codes for one window, shape ``(rows, len(entity_columns))``.

    Column order follows ``spec.entity_columns``, which
    ``definitions._entities`` guarantees is the declared family order rather than the
    order an ablation happened to name them in.
    """
    if spec.encoding is CategoricalEncoding.NONE:
        return np.zeros((frame.height, 0), dtype=np.int64)

    resolved = resolve_categories(frame, encoding.chains)
    columns: list[NDArray[np.int64]] = []
    for column in spec.entity_columns:
        vocab = encoding.vocabulary_for(column)
        lookup = {name: i for i, name in enumerate(vocab.categories)}
        values = [str(v) if v is not None else UNKNOWN_CATEGORY for v in resolved[column].to_list()]
        columns.append(np.asarray([lookup.get(v, UNKNOWN_INDEX) for v in values], dtype=np.int64))
    if not columns:
        return np.zeros((frame.height, 0), dtype=np.int64)
    return np.column_stack(columns)


def one_hot_columns(spec: NeuralSpec, encoding: FoldEncoding) -> tuple[str, ...]:
    """Indicator column names for the one-hot control, in a deterministic order.

    UNKNOWN is included as its own indicator rather than dropped. Dropping it would make
    "unseen category" indistinguishable from "all indicators zero", which is what a
    dropped reference level means -- and the two are different statements.
    """
    if spec.encoding is not CategoricalEncoding.ONE_HOT:
        return ()
    names: list[str] = []
    for column in spec.entity_columns:
        vocab = encoding.vocabulary_for(column)
        names.extend(f"{column}={category}" for category in vocab.categories)
    return tuple(names)


def one_hot_matrix(
    frame: pl.DataFrame, spec: NeuralSpec, encoding: FoldEncoding
) -> NDArray[np.float64]:
    """Dense indicator matrix for the one-hot control.

    Built from the same integer codes the embedding path uses, so the two representations
    are provably over the same categories and the B-versus-A comparison isolates the
    representation rather than the vocabulary.
    """
    if spec.encoding is not CategoricalEncoding.ONE_HOT:
        return np.zeros((frame.height, 0), dtype=np.float64)

    codes = index_matrix(frame, spec, encoding)
    blocks: list[NDArray[np.float64]] = []
    for position, column in enumerate(spec.entity_columns):
        vocab = encoding.vocabulary_for(column)
        block = np.zeros((frame.height, vocab.size), dtype=np.float64)
        block[np.arange(frame.height), codes[:, position]] = 1.0
        blocks.append(block)
    if not blocks:
        return np.zeros((frame.height, 0), dtype=np.float64)
    return np.hstack(blocks)


def unseen_rate(frame: pl.DataFrame, spec: NeuralSpec, encoding: FoldEncoding) -> dict[str, float]:
    """Fraction of rows mapping to UNKNOWN, per family.

    A diagnostic worth carrying: a test window where 40% of chains are unseen is one
    where the chain embedding is mostly a single learned row, and that is an explanation
    for a flat result rather than something to tune away.
    """
    if spec.encoding is CategoricalEncoding.NONE or frame.height == 0:
        return {}
    codes = index_matrix(frame, spec, encoding)
    return {
        column: float(np.count_nonzero(codes[:, position] == UNKNOWN_INDEX)) / frame.height
        for position, column in enumerate(spec.entity_columns)
    }


__all__ = [
    "UNKNOWN_INDEX",
    "EncodeError",
    "FoldEncoding",
    "Vocabulary",
    "chain_expression",
    "chain_membership",
    "fit_encoding",
    "index_matrix",
    "one_hot_columns",
    "one_hot_matrix",
    "resolve_categories",
    "unseen_rate",
]
