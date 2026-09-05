"""The group frame: where the audit's rows get their identity, and how that is proved.

Everything downstream is arithmetic over this frame. If a row is labelled with the wrong
neighbourhood, every number in the artifact is wrong in a way nothing downstream can detect:
the metrics stay finite, the supports stay plausible, and no check fires. So the frame is
built defensively -- left joins with completeness *required*, rather than inner joins that
would quietly produce a smaller, plausible population.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from sentinel.fairness import groups
from sentinel.fairness.definitions import (
    AUDITED_GROUP_DEFINITIONS,
    UNKNOWN_GROUP,
    FairnessDefinitionError,
    group_definition_for,
)

COMMUNITY_AREA = group_definition_for("community_area")
ZIP = group_definition_for("zip")


def _source(rows: int = 8) -> pl.DataFrame:
    base = date(2024, 3, 1)
    return pl.DataFrame(
        {
            "target_inspection_id": [f"T{i:03d}" for i in range(rows)],
            "establishment_id": [f"EST-{i % 4:03d}" for i in range(rows)],
            "inspection_date": [(base + timedelta(days=i)).isoformat() for i in range(rows)],
            "source_inspection_id": [f"S{i:03d}" for i in range(rows)],
            "source_inspection_date": [base + timedelta(days=i - 30) for i in range(rows)],
            "days_since_source": [30] * rows,
            "community_area": [str(i % 3) for i in range(rows)],
            "zip": [f"6060{i % 2}" for i in range(rows)],
            "chain_key": ["X"] * rows,
            "facility_type": ["RESTAURANT"] * rows,
        }
    )


def _predictions(source: pl.DataFrame) -> pl.DataFrame:
    n = source.height
    return pl.DataFrame(
        {
            "target_inspection_id": source["target_inspection_id"],
            "model_name": ["logistic_regression_platt"] * n,
            "base_model_name": ["logistic_regression"] * n,
            "fold_set": ["quarterly"] * n,
            "fold_id": ["quarterly-2024Q2"] * n,
            "score": [0.4 + i / 100 for i in range(n)],
            "base_score": [0.3 + i / 100 for i in range(n)],
            "is_experimental": [False] * n,
            "method": ["platt"] * n,
        }
    )


def _labels(source: pl.DataFrame) -> pl.DataFrame:
    n = source.height
    return pl.DataFrame(
        {
            "target_inspection_id": source["target_inspection_id"],
            "target": [i % 2 for i in range(n)],
            "rd": [date(2024, 4, 1) + timedelta(days=i) for i in range(n)],
        }
    )


# --- 1. resolving which definitions to audit -----------------------------------


def test_no_names_means_every_audited_definition() -> None:
    specs = groups.resolve_definitions(None)
    assert tuple(spec.name for spec in specs) == AUDITED_GROUP_DEFINITIONS


def test_a_refused_definition_is_rejected_with_its_measurement() -> None:
    with pytest.raises(FairnessDefinitionError, match="98.3%"):
        groups.resolve_definitions(["ward"])


def test_asking_for_the_same_definition_twice_is_rejected() -> None:
    with pytest.raises(groups.GroupFrameError, match="requested twice"):
        groups.resolve_definitions(["zip", "zip"])


# --- 2. the source frame -------------------------------------------------------


def test_the_source_carries_the_provenance_columns_the_checks_need() -> None:
    source = groups.group_source(_source(), [COMMUNITY_AREA])
    for column in (groups.KEY, groups.ENTITY_COLUMN, *groups.SOURCE_COLUMNS):
        assert column in source.columns


def test_a_source_without_the_provenance_columns_is_rejected() -> None:
    """Provenance that cannot be checked is provenance being taken on trust."""
    bare = _source().drop("source_inspection_date")
    with pytest.raises(groups.GroupFrameError, match="as-of claim cannot be re-derived"):
        groups.group_source(bare, [COMMUNITY_AREA])


def test_a_source_missing_the_group_column_is_rejected() -> None:
    with pytest.raises(groups.GroupFrameError, match="missing column"):
        groups.group_source(_source().drop("zip"), [ZIP])


def test_a_duplicated_key_is_rejected_before_it_can_inflate_a_support_count() -> None:
    doubled = pl.concat([_source(3), _source(3)])
    with pytest.raises(groups.GroupFrameError, match="duplicate keys"):
        groups.group_source(doubled, [COMMUNITY_AREA])


def test_a_null_group_value_is_rejected_because_absence_is_a_token() -> None:
    """A null would be coerced somewhere on the way to a group-by, and where it landed
    would depend on which code path saw it first.
    """
    nulled = _source().with_columns(
        pl.when(pl.col("target_inspection_id") == "T000")
        .then(None)
        .otherwise(pl.col("community_area"))
        .alias("community_area")
    )
    with pytest.raises(groups.GroupFrameError, match="Absence is the token"):
        groups.group_source(nulled, [COMMUNITY_AREA])


def test_the_unknown_token_is_accepted_as_a_real_group_value() -> None:
    """It is a superset of the rows with no prior inspection -- the most interesting group
    in a missingness audit, and the one dropping nulls would delete.
    """
    with_unknown = _source().with_columns(
        pl.when(pl.col("target_inspection_id") == "T000")
        .then(pl.lit(UNKNOWN_GROUP))
        .otherwise(pl.col("community_area"))
        .alias("community_area")
    )
    source = groups.group_source(with_unknown, [COMMUNITY_AREA])
    assert UNKNOWN_GROUP in source["community_area"].to_list()


# --- 3. temporal validity --------------------------------------------------------


def test_the_measured_minimum_lag_is_returned_for_the_manifest() -> None:
    rows, minimum = groups.check_temporal_validity(_source())
    assert rows == 8
    assert minimum == 30


def test_rows_with_no_source_are_excluded_from_the_lag_rather_than_failing() -> None:
    """401 rows genuinely have no prior inspection; that is a fact, not a defect."""
    partial = _source().with_columns(
        pl.when(pl.col("target_inspection_id") == "T000")
        .then(None)
        .otherwise(pl.col("source_inspection_date"))
        .alias("source_inspection_date"),
        pl.when(pl.col("target_inspection_id") == "T000")
        .then(None)
        .otherwise(pl.col("days_since_source"))
        .alias("days_since_source"),
    )
    rows, minimum = groups.check_temporal_validity(partial)
    assert rows == 7
    assert minimum == 30


# --- 4. building the audited frame --------------------------------------------------


def test_the_frame_carries_both_stages_under_unambiguous_names() -> None:
    """Nothing downstream may read "score" and assume either one."""
    source = _source()
    frame = groups.audited_frame(
        groups.build_group_frame(_predictions(source), source, _labels(source), [COMMUNITY_AREA])
    )
    assert "calibrated_probability" in frame.columns
    assert "base_probability" in frame.columns
    assert "score" not in frame.columns
    assert "base_score" not in frame.columns


def test_the_frame_covers_every_scored_row() -> None:
    source = _source()
    result = groups.build_group_frame(
        _predictions(source), source, _labels(source), [COMMUNITY_AREA, ZIP]
    )
    assert groups.audited_frame(result).height == source.height


def test_the_observed_values_are_recorded_sorted_for_the_no_disappearance_check() -> None:
    source = _source()
    result = groups.build_group_frame(
        _predictions(source), source, _labels(source), [COMMUNITY_AREA]
    )
    assert result.observed_values["community_area"] == ("0", "1", "2")


def test_a_prediction_frame_without_the_base_score_is_rejected() -> None:
    """An artifact without both stages cannot answer whether calibration reached the groups."""
    source = _source()
    without_base = _predictions(source).drop("base_score")
    with pytest.raises(groups.GroupFrameError, match="base_score"):
        groups.build_group_frame(without_base, source, _labels(source), [COMMUNITY_AREA])


def test_a_label_frame_without_the_reference_date_is_rejected() -> None:
    """A within-group NDE is computed over that group's own slot calendar."""
    source = _source()
    labels = _labels(source).drop("rd")
    with pytest.raises(groups.GroupFrameError, match="'rd'"):
        groups.build_group_frame(_predictions(source), source, labels, [COMMUNITY_AREA])


def test_a_row_without_an_outcome_is_rejected_rather_than_dropped() -> None:
    source = _source()
    labels = _labels(source).head(4)
    with pytest.raises(groups.GroupFrameError, match="no outcome label"):
        groups.build_group_frame(_predictions(source), source, labels, [COMMUNITY_AREA])


def test_an_empty_prediction_frame_is_rejected() -> None:
    source = _source()
    empty = _predictions(source).head(0)
    with pytest.raises(groups.GroupFrameError, match="no predictions to audit"):
        groups.build_group_frame(empty, source, _labels(source), [COMMUNITY_AREA])


def test_the_frame_reports_the_provenance_it_verified() -> None:
    source = _source()
    result = groups.build_group_frame(
        _predictions(source), source, _labels(source), [COMMUNITY_AREA]
    )
    assert result.as_of_rows == source.height
    assert result.min_source_lag_days == 30
    assert tuple(spec.name for spec in result.definitions) == ("community_area",)
