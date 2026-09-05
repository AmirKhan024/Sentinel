"""The embeddings-into-XGBoost experiment.

The temporal guarantee -- that a booster only ever consumes vectors learned on its own
fold -- is driven in ``test_neural_leakage.py``. This file establishes the rest: that the
comparison against Component 7 is genuinely like-for-like, that the widened matrix is the
tree matrix plus a labelled block, and that an unseen category gets the vector the network
itself would have used rather than an invented one.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sentinel.boosting import preprocess as tree_preprocess
from sentinel.boosting.definitions import spec_for as boosting_spec_for
from sentinel.boosting.definitions import tuned_params
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from sentinel.neural import embed, train
from sentinel.neural.definitions import EMBEDDING_DONOR, spec_for
from tests.conftest import neural_categoricals_for, spanning_model_features

DONOR = spec_for("neural_embeddings")
BOOSTED = spec_for("xgboost_chain_embeddings")


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


def _donor(frame: pl.DataFrame, fold: FoldSpec, cats: pl.DataFrame):  # type: ignore[no-untyped-def]
    return train.fit_fold(DONOR, training_frame(frame, fold), fold, categoricals=cats, max_epochs=3)


# --- 1. the comparison is like-for-like --------------------------------------


def test_the_experiment_borrows_component_7s_frozen_parameters_unchanged() -> None:
    """Re-tuning would confound "the embeddings helped" with "a second search helped"."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    fitted = embed.fit_fold(
        BOOSTED, training_frame(frame, fold), fold, donor=donor, categoricals=cats
    )
    expected = tuned_params(boosting_spec_for("xgboost"), fold.fold_set)
    for key, value in expected.items():
        assert fitted.params[key] == value, f"{key} differs from Component 7's frozen value"


def test_the_donor_is_the_component_7_xgboost_model() -> None:
    assert embed.BOOSTER_DONOR == "xgboost"


def test_the_base_half_of_the_matrix_is_component_7s_tree_matrix() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)

    columns = embed.augmented_columns(BOOSTED, donor)
    base = tree_preprocess.matrix_columns(boosting_spec_for("xgboost"))
    assert columns[: len(base)] == base


def test_the_base_half_keeps_its_nans() -> None:
    """This is still XGBoost: routing a NULL to a learned split direction is the point.

    Imputing here would change what Component 7's parameters were tuned for and make the
    comparison meaningless.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    window = training_frame(frame, fold).join(
        cats.select("target_inspection_id", "chain_key", "facility_type", "community_area", "zip"),
        on="target_inspection_id",
        how="left",
    )
    matrix = embed.augmented_matrix(window, BOOSTED, donor)
    base_width = len(tree_preprocess.matrix_columns(boosting_spec_for("xgboost")))
    assert np.isnan(matrix[:, :base_width]).any(), "the tree half was imputed"


def test_the_embedding_block_is_dense() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    window = training_frame(frame, fold).join(
        cats.select("target_inspection_id", "chain_key", "facility_type", "community_area", "zip"),
        on="target_inspection_id",
        how="left",
    )
    matrix = embed.augmented_matrix(window, BOOSTED, donor)
    base_width = len(tree_preprocess.matrix_columns(boosting_spec_for("xgboost")))
    block = matrix[:, base_width:]
    assert block.shape[1] == 16
    assert np.all(np.isfinite(block))


# --- 2. the block is labelled ------------------------------------------------


def test_embedding_columns_are_zero_padded_so_they_sort_numerically() -> None:
    """Without padding ``chain_emb_10`` would sort before ``chain_emb_2``."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    names = embed.embedding_columns(donor, "chain")
    assert names[0] == "chain_emb_00"
    assert names[-1] == "chain_emb_15"
    assert list(names) == sorted(names)


def test_the_column_count_matches_the_matrix_width() -> None:
    """Every importance would be mislabelled if these disagreed."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    fitted = embed.fit_fold(
        BOOSTED, training_frame(frame, fold), fold, donor=donor, categoricals=cats
    )
    assert len(fitted.matrix_columns) == len(fitted.importances)
    assert len(fitted.embedding_columns) == 16


# --- 3. vector lookup --------------------------------------------------------


def test_each_row_receives_its_own_categorys_vector() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)

    window = training_frame(frame, fold).join(
        cats.select("target_inspection_id", "chain_key", "facility_type", "community_area", "zip"),
        on="target_inspection_id",
        how="left",
    )
    vectors = embed.lookup_vectors(window, donor, "chain")
    table = donor.embedding_for("chain")
    assert vectors.shape == (window.height, table.dim)

    from sentinel.neural import encode

    resolved = encode.resolve_categories(window, donor.encoding.chains)
    first = str(resolved["chain"].to_list()[0])
    index = list(table.categories).index(first)
    assert np.allclose(vectors[0], np.asarray(table.vectors[index]))


def test_an_unseen_category_gets_the_networks_own_fallback_vector() -> None:
    """Zeros would hand XGBoost a value the network never assigned to anything."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)

    window = (
        training_frame(frame, fold)
        .head(5)
        .join(
            cats.select(
                "target_inspection_id", "chain_key", "facility_type", "community_area", "zip"
            ),
            on="target_inspection_id",
            how="left",
        )
    )
    unseen = window.with_columns(pl.lit("NEVER_SEEN_ANYWHERE").alias("chain_key"))
    vectors = embed.lookup_vectors(unseen, donor, "chain")

    table = donor.embedding_for("chain")
    from sentinel.neural.definitions import INDEPENDENT_CHAIN

    fallback = np.asarray(table.vectors[list(table.categories).index(INDEPENDENT_CHAIN)])
    assert np.allclose(vectors[0], fallback)
    assert not np.allclose(vectors[0], 0.0), "the fallback was zeros, not a learned row"


# --- 4. wiring ---------------------------------------------------------------


def test_the_registry_declares_this_experiments_donor() -> None:
    assert EMBEDDING_DONOR["xgboost_chain_embeddings"] == "neural_embeddings"


def test_the_fit_records_which_network_supplied_its_vectors() -> None:
    """The provenance a reader needs to check the temporal argument."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    fitted = embed.fit_fold(
        BOOSTED, training_frame(frame, fold), fold, donor=donor, categoricals=cats
    )
    assert fitted.donor_model == "neural_embeddings"
    assert fitted.donor_fold_id == fold.fold_id
    assert fitted.trained_through == fold.train_end


def test_scoring_produces_one_probability_per_test_row() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    fitted = embed.fit_fold(
        BOOSTED, training_frame(frame, fold), fold, donor=donor, categoricals=cats
    )
    window = folds_module.window_frame(frame, fold)
    ids, scores = embed.score_window(fitted, window, donor=donor, categoricals=cats)
    assert ids == [str(v) for v in window["target_inspection_id"].to_list()]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_an_empty_training_window_is_refused() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    with pytest.raises(embed.EmbedError, match="no training rows"):
        embed.fit_fold(
            BOOSTED,
            training_frame(frame, fold).head(0),
            fold,
            donor=donor,
            categoricals=cats,
        )


def test_embedding_rows_describe_every_vector() -> None:
    """The artifact the visualisation and any later inspection read."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    donor = _donor(frame, fold, cats)
    rows = embed.embedding_rows(donor)

    expected = sum(
        donor.encoding.vocabulary_for(c).size * donor.embedding_for(c).dim
        for c in DONOR.entity_columns
    )
    assert len(rows) == expected
    families = {r["family"] for r in rows}
    assert families == set(DONOR.entity_columns)
    for row in rows[:20]:
        assert isinstance(row["value"], float)
        assert row["category"]
