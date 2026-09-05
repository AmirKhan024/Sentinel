"""Vocabularies, chain membership and the two categorical representations.

``encode`` holds the fitted objects that a leakage test has to be able to reason about,
so these tests are about *what a vocabulary contains* and *where it came from* rather
than about index arithmetic. The temporal properties are driven in
``test_neural_leakage.py``; this file establishes that the mechanism they rely on
behaves as documented.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sentinel.neural import encode
from sentinel.neural.definitions import (
    INDEPENDENT_CHAIN,
    UNKNOWN_CATEGORY,
    CategoricalEncoding,
    spec_for,
)

PRIMARY = spec_for("neural_embeddings")
ONEHOT = spec_for("neural_onehot")
NUMERIC_ONLY = spec_for("neural_numeric_only")


def _frame(rows: list[tuple[str, str, str, str, str]]) -> pl.DataFrame:
    """(establishment_id, chain_key, facility_type, community_area, zip)."""
    return pl.DataFrame(
        {
            "establishment_id": [r[0] for r in rows],
            "chain_key": [r[1] for r in rows],
            "facility_type": [r[2] for r in rows],
            "community_area": [r[3] for r in rows],
            "zip": [r[4] for r in rows],
        }
    )


def _simple() -> pl.DataFrame:
    return _frame(
        [
            ("EST-1", "acme", "RESTAURANT", "8", "60601"),
            ("EST-2", "acme", "RESTAURANT", "9", "60602"),
            ("EST-3", "solo", "BAKERY", "8", "60601"),
            ("EST-4", "other", "SCHOOL", "10", "60603"),
        ]
    )


# --- 1. chain membership -----------------------------------------------------


def test_a_name_shared_by_two_establishments_is_a_chain() -> None:
    chains = encode.chain_membership(_simple())
    assert chains == frozenset({"acme"})


def test_a_name_held_by_one_establishment_is_not_a_chain() -> None:
    """ "Not part of a chain" is a fact, and it gets its own category rather than a null."""
    chains = encode.chain_membership(_simple())
    assert "solo" not in chains
    resolved = encode.resolve_categories(_simple(), chains)
    values = resolved["chain"].to_list()
    assert values == ["acme", "acme", INDEPENDENT_CHAIN, INDEPENDENT_CHAIN]


def test_a_name_repeated_by_one_establishment_is_still_not_a_chain() -> None:
    """Membership counts distinct establishments, not rows.

    Otherwise a frequently inspected single location would become a "chain" purely by
    being inspected often, which is a property of inspection cadence rather than of the
    business.
    """
    frame = _frame(
        [
            ("EST-1", "acme", "RESTAURANT", "8", "60601"),
            ("EST-1", "acme", "RESTAURANT", "8", "60601"),
            ("EST-1", "acme", "RESTAURANT", "8", "60601"),
        ]
    )
    assert encode.chain_membership(frame) == frozenset()


def test_unknown_is_never_counted_as_a_chain() -> None:
    """Two establishments with no history are not thereby in a chain together."""
    frame = _frame(
        [
            ("EST-1", UNKNOWN_CATEGORY, "RESTAURANT", "8", "60601"),
            ("EST-2", UNKNOWN_CATEGORY, "RESTAURANT", "8", "60601"),
        ]
    )
    assert encode.chain_membership(frame) == frozenset()
    resolved = encode.resolve_categories(frame, frozenset())
    assert resolved["chain"].to_list() == [UNKNOWN_CATEGORY, UNKNOWN_CATEGORY]


# --- 2. vocabularies ---------------------------------------------------------


def test_index_zero_is_always_unknown() -> None:
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    for vocab in encoding.vocabularies:
        assert vocab.categories[0] == UNKNOWN_CATEGORY
        assert vocab.index_of(UNKNOWN_CATEGORY) == encode.UNKNOWN_INDEX == 0


def test_the_rest_of_a_vocabulary_is_sorted() -> None:
    """Sorted, not insertion-ordered: insertion order is row order."""
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    for vocab in encoding.vocabularies:
        rest = list(vocab.categories[1:])
        assert rest == sorted(rest)


def test_a_vocabulary_covers_exactly_the_declared_families() -> None:
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    assert encoding.columns == PRIMARY.entity_columns
    ablation = spec_for("neural_no_community_area")
    ablated = encode.fit_encoding(_simple(), ablation)
    assert "community_area" not in ablated.columns
    assert set(ablated.columns) == set(ablation.entity_columns)


def test_an_unseen_value_maps_to_the_unknown_index() -> None:
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    unseen = _frame([("EST-9", "acme", "NEW TYPE", "99", "99999")])
    codes = encode.index_matrix(unseen, PRIMARY, encoding)
    for position, column in enumerate(PRIMARY.entity_columns):
        if column == "chain":
            continue
        assert codes[0, position] == encode.UNKNOWN_INDEX, f"{column} did not fall back"


def test_a_spec_with_no_categoricals_produces_a_zero_width_encoding() -> None:
    encoding = encode.fit_encoding(_simple(), NUMERIC_ONLY)
    assert encoding.vocabularies == ()
    assert encoding.chains == frozenset()
    codes = encode.index_matrix(_simple(), NUMERIC_ONLY, encoding)
    assert codes.shape == (4, 0)


def test_fitting_on_an_empty_window_is_refused() -> None:
    with pytest.raises(encode.EncodeError, match="empty training window"):
        encode.fit_encoding(_simple().head(0), PRIMARY)


def test_a_missing_column_is_refused() -> None:
    with pytest.raises(encode.EncodeError, match="no column"):
        encode.fit_encoding(_simple().drop("zip"), PRIMARY)


def test_vocabulary_sizes_are_reported() -> None:
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    sizes = encoding.sizes
    # UNKNOWN + acme + INDEPENDENT
    assert sizes["chain"] == 3
    # UNKNOWN + RESTAURANT + BAKERY + SCHOOL
    assert sizes["facility_type"] == 4
    assert sizes["community_area"] == 4
    assert sizes["zip"] == 4


# --- 3. the one-hot control --------------------------------------------------


def test_one_hot_is_over_exactly_the_same_categories_as_the_embedding() -> None:
    """Experiment B must isolate the representation, not the vocabulary."""
    embedding = encode.fit_encoding(_simple(), PRIMARY)
    onehot = encode.fit_encoding(_simple(), ONEHOT)
    assert embedding.sizes == onehot.sizes
    assert embedding.chains == onehot.chains


def test_one_hot_rows_sum_to_one_per_family() -> None:
    encoding = encode.fit_encoding(_simple(), ONEHOT)
    matrix = encode.one_hot_matrix(_simple(), ONEHOT, encoding)
    assert matrix.shape[0] == 4
    assert matrix.shape[1] == sum(encoding.sizes.values())
    # Exactly one indicator per family per row.
    assert np.allclose(matrix.sum(axis=1), len(ONEHOT.entity_columns))
    assert set(np.unique(matrix)) <= {0.0, 1.0}


def test_one_hot_keeps_the_unknown_level() -> None:
    """Dropping it would make "unseen" indistinguishable from "all indicators zero"."""
    encoding = encode.fit_encoding(_simple(), ONEHOT)
    names = encode.one_hot_columns(ONEHOT, encoding)
    for column in ONEHOT.entity_columns:
        assert f"{column}={UNKNOWN_CATEGORY}" in names


def test_one_hot_column_names_are_deterministic() -> None:
    first = encode.one_hot_columns(ONEHOT, encode.fit_encoding(_simple(), ONEHOT))
    second = encode.one_hot_columns(ONEHOT, encode.fit_encoding(_simple(), ONEHOT))
    assert first == second


def test_a_non_onehot_spec_produces_no_indicator_columns() -> None:
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    assert encode.one_hot_columns(PRIMARY, encoding) == ()
    assert encode.one_hot_matrix(_simple(), PRIMARY, encoding).shape == (4, 0)


# --- 4. diagnostics ----------------------------------------------------------


def test_unseen_rate_counts_rows_not_categories() -> None:
    """The diagnostic that explains a flat embedding result."""
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    later = _frame(
        [
            ("EST-9", "acme", "RESTAURANT", "8", "60601"),
            ("EST-9", "acme", "BRAND NEW", "8", "60601"),
        ]
    )
    rates = encode.unseen_rate(later, PRIMARY, encoding)
    assert rates["facility_type"] == 0.5
    assert rates["zip"] == 0.0


def test_unseen_rate_is_empty_for_a_numeric_only_spec() -> None:
    encoding = encode.fit_encoding(_simple(), NUMERIC_ONLY)
    assert encode.unseen_rate(_simple(), NUMERIC_ONLY, encoding) == {}


def test_index_matrix_column_order_follows_the_declared_family_order() -> None:
    encoding = encode.fit_encoding(_simple(), PRIMARY)
    codes = encode.index_matrix(_simple(), PRIMARY, encoding)
    assert codes.shape == (4, len(PRIMARY.entity_columns))
    chain_position = PRIMARY.entity_columns.index("chain")
    chain_vocab = encoding.vocabulary_for("chain")
    assert codes[0, chain_position] == chain_vocab.index_of("acme")


def test_vocabulary_for_rejects_an_unfitted_column() -> None:
    encoding = encode.fit_encoding(_simple(), spec_for("neural_no_zip"))
    with pytest.raises(encode.EncodeError, match="no vocabulary fitted"):
        encoding.vocabulary_for("zip")


def test_encoding_is_reproducible() -> None:
    """Two fits over the same rows must agree exactly, including on chain membership."""
    first = encode.fit_encoding(_simple(), PRIMARY)
    second = encode.fit_encoding(_simple().sort("establishment_id", descending=True), PRIMARY)
    assert first.chains == second.chains
    assert first.sizes == second.sizes
    for left, right in zip(first.vocabularies, second.vocabularies, strict=True):
        assert left.categories == right.categories, (
            f"{left.column}: vocabulary depends on input row order"
        )


def test_the_encoding_spec_types_are_what_the_registry_declares() -> None:
    assert PRIMARY.encoding is CategoricalEncoding.EMBEDDING
    assert ONEHOT.encoding is CategoricalEncoding.ONE_HOT
    assert NUMERIC_ONLY.encoding is CategoricalEncoding.NONE
