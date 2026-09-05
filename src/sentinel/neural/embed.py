"""The specification's embeddings-into-XGBoost experiment, and the vectors it feeds on.

The question is narrow and worth stating precisely: *does the representation Component 8
learned help the estimator that currently wins?* Component 7's XGBoost is the reference
model of the project so far. If a learned chain vector carries information the 26 as-of
features do not, the cleanest way to find out is to hand it to that model and change
nothing else.

So nothing else changes. Component 7's frozen parameters, Component 7's tree matrix,
Component 7's NaN-native handling -- and 16 extra columns. Re-tuning for the wider matrix
would confound "the embeddings helped" with "a second search helped", and a negative
result from an unfair comparison is worth nothing.

**The temporal argument, which is the whole risk here.**

An embedding table is a fitted artifact. Fitted on the wrong rows it is the most direct
leak available in this project: a chain's vector would encode how that chain's
inspections turned out, and handing it to a model scoring those same inspections is
circular. Three properties keep it honest, and each has a test:

1. The donor network is fitted on **the same fold's training window** and nothing else,
   by ``train.fit_fold``. There is no cross-fold embedding table anywhere in this
   component and no code path that could build one.
2. The vocabulary is the donor's, so a chain absent from that fold's training rows maps to
   the learned UNKNOWN row rather than acquiring a vector of its own.
3. The vectors are read from the donor *after* it is fitted and are then constant. XGBoost
   receives numbers, not parameters; nothing here backpropagates into the table.

What this experiment cannot rule out is the ordinary version of the same concern: the
donor network saw the fold's training labels, so its chain vectors are a supervised
summary of training-window outcomes per chain. That is not leakage -- it is target
encoding, and it is confined to the training window exactly as a training-window median
is -- but it does mean the vectors are more informative about training rows than about
test rows, and a gap between the two is the expected behaviour rather than a surprise.
Stated here because it is the thing a reader should suspect first.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sentinel.boosting import preprocess as tree_preprocess
from sentinel.boosting.definitions import (
    BoostingSpec,
    Estimator,
    estimator_params,
    n_estimators_of,
)
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import TRAIN_SORT_KEYS
from sentinel.neural import encode
from sentinel.neural.definitions import (
    NeuralSpec,
)
from sentinel.neural.models import FittedEmbeddingBooster, FittedNetwork

logger = logging.getLogger(__name__)

#: The Component 7 model whose frozen parameters this experiment borrows. Named rather
#: than reconstructed so a reader can check that the comparison really is like-for-like.
BOOSTER_DONOR = "xgboost"


class EmbedError(ValueError):
    """Raised when an embedding-fed matrix cannot be built."""


def _as_boosting_spec(spec: NeuralSpec) -> BoostingSpec:
    """Adapt a :class:`NeuralSpec` to the shape Component 7's code reads.

    Borrows the *donor's* name so ``estimator_params`` and ``n_estimators_of`` return
    Component 7's frozen values rather than raising on an unregistered model. That is the
    intent made mechanical: this experiment is XGBoost with its published parameters, and
    the only way to get them is to ask for them under the name they were frozen under.
    """
    return BoostingSpec(
        name=BOOSTER_DONOR,
        version=spec.version,
        description=spec.description,
        estimator=Estimator.XGBOOST,
        feature_columns=spec.feature_columns,
        is_probability=spec.is_probability,
        seed=spec.seed,
        class_weighted=False,
    )


def embedding_columns(donor: FittedNetwork, column: str) -> tuple[str, ...]:
    """Names for the appended dimensions, e.g. ``chain_emb_00 .. chain_emb_15``.

    Zero-padded so a lexicographic sort of the importances table matches the numeric
    order; without the padding ``chain_emb_10`` would sort before ``chain_emb_2``.
    """
    table = donor.embedding_for(column)
    return tuple(f"{column}_emb_{i:02d}" for i in range(table.dim))


def lookup_vectors(frame: pl.DataFrame, donor: FittedNetwork, column: str) -> NDArray[np.float64]:
    """One row per input row: that row's category's learned vector.

    Rows whose category is absent from the donor's vocabulary receive the UNKNOWN row,
    which is the same vector the network itself would have used for them. Substituting
    zeros instead would hand XGBoost a value the network never assigned to anything.
    """
    table = donor.embedding_for(column)
    vectors = np.asarray(table.vectors, dtype=np.float64)
    if vectors.size == 0:
        raise EmbedError(f"donor {donor.spec.name} has an empty embedding table for {column!r}")

    resolved = encode.resolve_categories(frame, donor.encoding.chains)
    lookup = {name: i for i, name in enumerate(table.categories)}
    codes = np.asarray(
        [lookup.get(str(v), encode.UNKNOWN_INDEX) for v in resolved[column].to_list()],
        dtype=np.int64,
    )
    return vectors[codes]


def augmented_columns(spec: NeuralSpec, donor: FittedNetwork) -> tuple[str, ...]:
    """The widened matrix's column names, in order: tree matrix, then embeddings."""
    base = tree_preprocess.matrix_columns(_as_boosting_spec(spec))
    extra: list[str] = []
    for column in spec.entity_columns:
        extra.extend(embedding_columns(donor, column))
    return (*base, *extra)


def augmented_matrix(
    frame: pl.DataFrame, spec: NeuralSpec, donor: FittedNetwork
) -> NDArray[np.float64]:
    """Component 7's tree matrix widened by the donor's learned vectors.

    The base half keeps its NaNs: this is still XGBoost, and routing a NULL to a learned
    default direction is the behaviour Component 7's findings identified as one reason a
    booster suits this data. Only the appended columns are dense.
    """
    base = tree_preprocess.tree_matrix(frame, _as_boosting_spec(spec))
    blocks: list[NDArray[np.float64]] = [base]
    for column in spec.entity_columns:
        blocks.append(lookup_vectors(frame, donor, column))
    matrix = np.hstack(blocks).astype(np.float64, copy=False)
    expected = len(augmented_columns(spec, donor))
    if matrix.shape[1] != expected:
        raise EmbedError(
            f"{spec.name}: augmented matrix has {matrix.shape[1]} columns for {expected} "
            "declared names; every importance would be mislabelled"
        )
    return matrix


def fit_fold(
    spec: NeuralSpec,
    train: pl.DataFrame,
    fold: FoldSpec,
    *,
    donor: FittedNetwork,
    categoricals: pl.DataFrame,
) -> FittedEmbeddingBooster:
    """Fit XGBoost on the widened matrix for one fold.

    ``donor`` must be a network fitted on **this same fold**; the check is not a courtesy,
    it is the experiment's entire temporal guarantee.
    """
    if donor.fold_id != fold.fold_id or donor.fold_set != fold.fold_set:
        raise EmbedError(
            f"{spec.name}: donor {donor.spec.name} was fitted on fold {donor.fold_id} but "
            f"is being used for {fold.fold_id}. An embedding table from another fold "
            "carries that fold's training window into this one."
        )
    if train.height == 0:
        raise EmbedError(f"{spec.name}: fold {fold.fold_id} has no training rows")

    from xgboost import XGBClassifier

    ordered = train.sort(list(TRAIN_SORT_KEYS))
    joined = _join(ordered, categoricals, spec)
    labels = np.asarray(joined["target"].to_list(), dtype=np.int64)
    if len(set(labels.tolist())) < 2:
        raise EmbedError(
            f"{spec.name}: fold {fold.fold_id} training window contains a single class"
        )

    boosting_spec = _as_boosting_spec(spec)
    params = estimator_params(boosting_spec, fold.fold_set)
    matrix = augmented_matrix(joined, spec, donor)
    columns = augmented_columns(spec, donor)

    kwargs: Any = params
    estimator = XGBClassifier(**kwargs)
    estimator.fit(matrix, labels)

    importances = _importances(estimator, len(columns), spec.name, fold.fold_id)
    mean = joined["target"].mean()

    fitted = FittedEmbeddingBooster(
        spec=spec,
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        matrix_columns=columns,
        params=params,
        n_estimators=n_estimators_of(boosting_spec, fold.fold_set),
        trees_built=_trees_built(estimator),
        importances=importances,
        embedding_columns=tuple(
            name for column in spec.entity_columns for name in embedding_columns(donor, column)
        ),
        donor_model=donor.spec.name,
        donor_fold_id=donor.fold_id,
        train_rows=joined.height,
        train_positive_rate=float(mean) if isinstance(mean, (int, float)) else None,
        train_nan_cells=int(np.count_nonzero(np.isnan(matrix))),
        train_start=fold.train_start,
        train_end=fold.train_end,
        trained_through=fold.train_end,
        calibration_end_unused=fold.calibration_end,
        seed=spec.seed,
    )
    _stash(fitted, estimator)
    logger.info(
        "Fitted %s on %s: %d rows, %d columns (%d from embeddings)",
        spec.name,
        fold.fold_id,
        joined.height,
        len(columns),
        len(fitted.embedding_columns),
    )
    return fitted


def score_window(
    fitted: FittedEmbeddingBooster,
    test: pl.DataFrame,
    *,
    donor: FittedNetwork,
    categoricals: pl.DataFrame,
) -> tuple[list[str], list[float]]:
    """Score one test window with the embedding-fed booster."""
    if test.height == 0:
        raise EmbedError(f"{fitted.spec.name}: fold {fitted.fold_id} test window is empty")
    estimator = _scorer_for(fitted)
    joined = _join(test, categoricals, fitted.spec)
    matrix = augmented_matrix(joined, fitted.spec, donor)
    proba = estimator.predict_proba(matrix)
    scores = [float(v) for v in proba[:, 1]]
    ids = [str(v) for v in joined["target_inspection_id"].to_list()]
    if len(ids) != len(scores):
        raise EmbedError(f"{fitted.spec.name}: {len(ids)} ids but {len(scores)} scores")
    return ids, scores


def _join(frame: pl.DataFrame, categoricals: pl.DataFrame, spec: NeuralSpec) -> pl.DataFrame:
    wanted = [
        c
        for c in ("chain_key", "facility_type", "community_area", "zip", "establishment_id")
        if c in categoricals.columns
    ]
    joined = frame.join(
        categoricals.select("target_inspection_id", *wanted),
        on="target_inspection_id",
        how="left",
        suffix="_cat",
    )
    if joined.height != frame.height:
        raise EmbedError(
            f"{spec.name}: categorical join changed the row count from {frame.height} to "
            f"{joined.height}"
        )
    return joined


def _importances(estimator: Any, width: int, model_name: str, fold_id: str) -> tuple[float, ...]:
    raw = getattr(estimator, "feature_importances_", None)
    if raw is None:
        raise EmbedError(f"{model_name}: fold {fold_id} produced no importances")
    values = [float(v) for v in raw]
    if len(values) != width:
        raise EmbedError(
            f"{model_name}: fold {fold_id} produced {len(values)} importances for {width} "
            "matrix columns"
        )
    return tuple(values)


def _trees_built(estimator: Any) -> int:
    booster = estimator.get_booster()
    return len(booster.get_dump())


_ESTIMATORS: dict[int, Any] = {}


def _stash(fitted: FittedEmbeddingBooster, estimator: Any) -> None:
    _ESTIMATORS[id(fitted)] = estimator


def _scorer_for(fitted: FittedEmbeddingBooster) -> Any:
    entry = _ESTIMATORS.get(id(fitted))
    if entry is None:
        raise EmbedError(
            "this FittedEmbeddingBooster has no live estimator: it was not produced by "
            "fit_fold in this process, so it cannot score a window"
        )
    return entry


def embedding_rows(fitted: FittedNetwork) -> list[dict[str, object]]:
    """One row per (family, category, dimension). The artifact the visualisation reads.

    Written out rather than kept in memory because the t-SNE projection, the nearest-
    neighbour tables and any later inspection all need the same vectors, and re-fitting a
    network to look at its embeddings again would be absurd.
    """
    rows: list[dict[str, object]] = []
    for table in fitted.embeddings:
        for index, (category, vector) in enumerate(
            zip(table.categories, table.vectors, strict=True)
        ):
            for dimension, value in enumerate(vector):
                rows.append(
                    {
                        "model_name": fitted.spec.name,
                        "fold_set": fitted.fold_set,
                        "fold_id": fitted.fold_id,
                        "family": table.column,
                        "category": category,
                        "category_index": index,
                        "dimension": dimension,
                        "value": float(value),
                    }
                )
    return rows


__all__ = [
    "BOOSTER_DONOR",
    "EmbedError",
    "augmented_columns",
    "augmented_matrix",
    "embedding_columns",
    "embedding_rows",
    "fit_fold",
    "lookup_vectors",
    "score_window",
]
