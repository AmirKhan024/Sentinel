"""The experimental as-of categorical join, driven against a hand-made raw frame.

This module is the one place Component 8 reaches outside Component 4's contract, so its
tests are about the boundary rather than about the values: does a row ever get its own
attributes, does a row with no history get a real token, and does the join stay
one-to-one.

The raw frame here is built by hand rather than sampled, because the property under test
is what happens at a date boundary and a sampled frame would only exercise it by luck.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.neural import categoricals as cats
from sentinel.neural.definitions import UNKNOWN_CATEGORY


def _raw(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "inspection_id": pl.Utf8,
            "inspection_date": pl.Utf8,
            "facility_type": pl.Utf8,
            "zip": pl.Utf8,
            cats.COMMUNITY_AREA_COLUMN: pl.Utf8,
        },
    )


def _assignments(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "inspection_id": pl.Utf8,
            "establishment_id": pl.Utf8,
            "name_key": pl.Utf8,
        },
    )


def _features(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "target_inspection_id": pl.Utf8,
            "establishment_id": pl.Utf8,
            "inspection_date": pl.Utf8,
        },
    )


def _history(raw: pl.DataFrame, assignments: pl.DataFrame) -> pl.DataFrame:
    """``load_history`` without touching the filesystem."""
    return (
        raw.with_columns(
            pl.col("inspection_date").str.slice(0, 10).str.to_date().alias("rd"),
            pl.col("facility_type")
            .str.strip_chars()
            .str.to_uppercase()
            .str.replace_all(r"\s+", " ")
            .replace("", None)
            .fill_null(UNKNOWN_CATEGORY)
            .alias("facility_type"),
            pl.col("zip").str.extract(r"^(\d{5})", 1).fill_null(UNKNOWN_CATEGORY).alias("zip"),
            pl.col(cats.COMMUNITY_AREA_COLUMN)
            .str.extract(r"^(\d+)", 1)
            .fill_null(UNKNOWN_CATEGORY)
            .alias("community_area"),
        )
        .join(assignments, on="inspection_id", how="inner")
        .with_columns(pl.col("name_key").fill_null(UNKNOWN_CATEGORY).alias("chain_key"))
        .select("inspection_id", "establishment_id", "rd", *cats.EMITTED_CATEGORICALS)
    )


# --- 1. the as-of boundary ---------------------------------------------------


def test_a_row_never_supplies_its_own_categoricals() -> None:
    """The single most important property in this module.

    An exact date match would mean the target inspection handed the model its own
    attributes, and every temporal check downstream would still pass.
    """
    raw = _raw(
        [
            {
                "inspection_id": "S1",
                "inspection_date": "2020-06-01T00:00:00.000",
                "facility_type": "Restaurant",
                "zip": "60601",
                cats.COMMUNITY_AREA_COLUMN: "8",
            }
        ]
    )
    assignments = _assignments(
        [{"inspection_id": "S1", "establishment_id": "EST-1", "name_key": "acme"}]
    )
    features = _features(
        [
            {
                "target_inspection_id": "S1",
                "establishment_id": "EST-1",
                "inspection_date": "2020-06-01",
            }
        ]
    )
    table = cats.build_categoricals(features, _history(raw, assignments))
    row = table.to_dicts()[0]
    assert row["facility_type"] == UNKNOWN_CATEGORY, (
        "the row's own inspection supplied its facility type"
    )
    assert row["source_inspection_id"] is None
    assert row["source_inspection_date"] is None


def test_the_most_recent_earlier_inspection_supplies_the_value() -> None:
    raw = _raw(
        [
            {
                "inspection_id": "S1",
                "inspection_date": "2019-01-01T00:00:00.000",
                "facility_type": "Bakery",
                "zip": "60601",
                cats.COMMUNITY_AREA_COLUMN: "8",
            },
            {
                "inspection_id": "S2",
                "inspection_date": "2020-01-01T00:00:00.000",
                "facility_type": "Restaurant",
                "zip": "60602",
                cats.COMMUNITY_AREA_COLUMN: "9",
            },
        ]
    )
    assignments = _assignments(
        [
            {"inspection_id": "S1", "establishment_id": "EST-1", "name_key": "acme"},
            {"inspection_id": "S2", "establishment_id": "EST-1", "name_key": "acme two"},
        ]
    )
    features = _features(
        [
            {
                "target_inspection_id": "S3",
                "establishment_id": "EST-1",
                "inspection_date": "2021-01-01",
            }
        ]
    )
    row = cats.build_categoricals(features, _history(raw, assignments)).to_dicts()[0]
    assert row["facility_type"] == "RESTAURANT", "the older inspection won"
    assert row["zip"] == "60602"
    assert row["community_area"] == "9"
    assert row["chain_key"] == "acme two"
    assert row["source_inspection_id"] == "S2"
    assert row["source_inspection_date"] == date(2020, 1, 1)
    assert row["days_since_source"] == 366


def test_a_later_inspection_never_supplies_a_value() -> None:
    """A future inspection must be invisible even when it is the only one."""
    raw = _raw(
        [
            {
                "inspection_id": "S9",
                "inspection_date": "2023-01-01T00:00:00.000",
                "facility_type": "Restaurant",
                "zip": "60601",
                cats.COMMUNITY_AREA_COLUMN: "8",
            }
        ]
    )
    assignments = _assignments(
        [{"inspection_id": "S9", "establishment_id": "EST-1", "name_key": "acme"}]
    )
    features = _features(
        [
            {
                "target_inspection_id": "S3",
                "establishment_id": "EST-1",
                "inspection_date": "2021-01-01",
            }
        ]
    )
    row = cats.build_categoricals(features, _history(raw, assignments)).to_dicts()[0]
    assert row["facility_type"] == UNKNOWN_CATEGORY
    assert row["source_inspection_id"] is None


def test_another_establishments_history_is_never_borrowed() -> None:
    """The join is per establishment; a neighbour's attributes must not leak across."""
    raw = _raw(
        [
            {
                "inspection_id": "S1",
                "inspection_date": "2019-01-01T00:00:00.000",
                "facility_type": "Bakery",
                "zip": "60601",
                cats.COMMUNITY_AREA_COLUMN: "8",
            }
        ]
    )
    assignments = _assignments(
        [{"inspection_id": "S1", "establishment_id": "EST-OTHER", "name_key": "other"}]
    )
    features = _features(
        [
            {
                "target_inspection_id": "S3",
                "establishment_id": "EST-1",
                "inspection_date": "2021-01-01",
            }
        ]
    )
    row = cats.build_categoricals(features, _history(raw, assignments)).to_dicts()[0]
    assert row["facility_type"] == UNKNOWN_CATEGORY
    assert row["chain_key"] == UNKNOWN_CATEGORY


# --- 2. normalisation --------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("Restaurant", "RESTAURANT"),
        ("  restaurant  ", "RESTAURANT"),
        ("GROCERY   STORE", "GROCERY STORE"),
        ("", UNKNOWN_CATEGORY),
    ],
)
def test_facility_type_is_normalised_conservatively(written: str, expected: str) -> None:
    """Case and whitespace are collapsed; synonyms are deliberately NOT merged."""
    raw = _raw(
        [
            {
                "inspection_id": "S1",
                "inspection_date": "2019-01-01T00:00:00.000",
                "facility_type": written,
                "zip": "60601",
                cats.COMMUNITY_AREA_COLUMN: "8",
            }
        ]
    )
    assignments = _assignments(
        [{"inspection_id": "S1", "establishment_id": "EST-1", "name_key": "acme"}]
    )
    features = _features(
        [
            {
                "target_inspection_id": "S3",
                "establishment_id": "EST-1",
                "inspection_date": "2021-01-01",
            }
        ]
    )
    row = cats.build_categoricals(features, _history(raw, assignments)).to_dicts()[0]
    assert row["facility_type"] == expected


def test_a_zip_plus_four_becomes_the_five_digit_prefix() -> None:
    raw = _raw(
        [
            {
                "inspection_id": "S1",
                "inspection_date": "2019-01-01T00:00:00.000",
                "facility_type": "Restaurant",
                "zip": "60601-1234",
                cats.COMMUNITY_AREA_COLUMN: "8",
            }
        ]
    )
    assignments = _assignments(
        [{"inspection_id": "S1", "establishment_id": "EST-1", "name_key": "acme"}]
    )
    features = _features(
        [
            {
                "target_inspection_id": "S3",
                "establishment_id": "EST-1",
                "inspection_date": "2021-01-01",
            }
        ]
    )
    row = cats.build_categoricals(features, _history(raw, assignments)).to_dicts()[0]
    assert row["zip"] == "60601"


def test_a_missing_community_area_becomes_unknown_not_null() -> None:
    """Socrata drops the computed region when a row has no coordinates."""
    raw = _raw(
        [
            {
                "inspection_id": "S1",
                "inspection_date": "2019-01-01T00:00:00.000",
                "facility_type": "Restaurant",
                "zip": "60601",
                cats.COMMUNITY_AREA_COLUMN: None,
            }
        ]
    )
    assignments = _assignments(
        [{"inspection_id": "S1", "establishment_id": "EST-1", "name_key": "acme"}]
    )
    features = _features(
        [
            {
                "target_inspection_id": "S3",
                "establishment_id": "EST-1",
                "inspection_date": "2021-01-01",
            }
        ]
    )
    row = cats.build_categoricals(features, _history(raw, assignments)).to_dicts()[0]
    assert row["community_area"] == UNKNOWN_CATEGORY
    assert row["community_area"] is not None


# --- 3. the join's shape -----------------------------------------------------


def test_the_join_is_one_to_one_on_the_feature_table() -> None:
    """Many-to-one would silently duplicate rows and keep a consistent matrix shape."""
    raw = _raw(
        [
            {
                "inspection_id": f"S{i}",
                "inspection_date": f"20{19 + i // 6}-0{1 + i % 6}-01T00:00:00.000",
                "facility_type": "Restaurant",
                "zip": "60601",
                cats.COMMUNITY_AREA_COLUMN: "8",
            }
            for i in range(12)
        ]
    )
    assignments = _assignments(
        [
            {"inspection_id": f"S{i}", "establishment_id": "EST-1", "name_key": "acme"}
            for i in range(12)
        ]
    )
    features = _features(
        [
            {
                "target_inspection_id": f"T{i}",
                "establishment_id": "EST-1",
                "inspection_date": "2023-01-01",
            }
            for i in range(5)
        ]
    )
    table = cats.build_categoricals(features, _history(raw, assignments))
    assert table.height == features.height
    assert table["target_inspection_id"].n_unique() == features.height


def test_a_feature_table_missing_a_key_column_is_refused() -> None:
    with pytest.raises(cats.CategoricalBuildError, match="missing column"):
        cats.build_categoricals(
            pl.DataFrame({"target_inspection_id": ["T1"]}),
            pl.DataFrame(
                {
                    "establishment_id": ["EST-1"],
                    "source_rd": [date(2020, 1, 1)],
                }
            ),
        )


def test_coverage_and_cardinality_describe_the_table() -> None:
    from tests.conftest import neural_categoricals_for, spanning_model_features

    features = spanning_model_features(days=400, per_day=2)
    table = neural_categoricals_for(features)
    coverage = cats.coverage(table)
    cardinality = cats.cardinality(table)
    for family in cats.EMITTED_CATEGORICALS:
        assert 0.0 <= coverage[family] <= 1.0
        assert cardinality[family] >= 1
    assert cardinality["facility_type"] == 4, "the fixture declares exactly four types"


def test_the_emitted_families_are_the_four_the_specification_names() -> None:
    assert set(cats.EMITTED_CATEGORICALS) == {
        "chain_key",
        "facility_type",
        "community_area",
        "zip",
    }


def test_the_module_declares_the_raw_columns_it_reads() -> None:
    """A schema change must surface as a missing-column error, not a silent absence."""
    assert "facility_type" in cats.RAW_COLUMNS
    assert "zip" in cats.RAW_COLUMNS
    assert cats.COMMUNITY_AREA_COLUMN in cats.RAW_COLUMNS
    assert "inspection_id" in cats.RAW_COLUMNS
    # Deliberately absent: nothing about the outcome may be read here.
    assert "results" not in cats.RAW_COLUMNS
    assert "violations" not in cats.RAW_COLUMNS
