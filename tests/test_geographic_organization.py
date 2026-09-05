"""Component 20: geographic proximity organization of a Component 19 selection.

Builds synthetic Component 19-shaped selection frames directly (from
``operational_selection.writer.OUTPUT_SCHEMA``), matching how the Component 18/19
suites build synthetic upstream-shaped frames rather than running the full pipeline.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.geographic_organization import grouping, metrics, validate
from sentinel.geographic_organization.definitions import (
    UNMAPPED_GROUP_ID,
    GeographicOrganizationDefinitionError,
    LocationStatus,
    validate_threshold,
)
from sentinel.geographic_organization.distance import haversine_km
from sentinel.operational_selection.writer import OUTPUT_SCHEMA

PLANNING_DATE = "2026-08-28"

# Real Chicago-area reference points (approximate), used for known-distance checks.
CITY_HALL = (41.8838, -87.6319)
WRIGLEY_FIELD = (41.9484, -87.6553)  # ~7.3 km from City Hall
O_HARE = (41.9742, -87.9073)  # ~27 km from City Hall


def _row(i: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "planning_date": PLANNING_DATE,
        "operational_selection_definition_version": "v1",
        "requested_capacity": 30,
        "policy_id": "pure_risk",
        "candidate_definition_version": "v1",
        "feature_definition_version": "v1",
        "operational_scoring_definition_version": "v1",
        "composite_model_name": "xgboost_platt",
        "base_model_name": "xgboost",
        "calibration_method": "platt",
        "establishment_id": f"EST-{i:06d}",
        "target_inspection_id": f"CANDIDATE::{PLANNING_DATE}::EST-{i:06d}",
        "canonical_name": f"NAME-{i}",
        "canonical_address": f"ADDR-{i}",
        "canonical_zip": "60601",
        "as_of_dba_name": f"NAME-{i}",
        "as_of_address": f"ADDR-{i}",
        "as_of_zip": "60601",
        "as_of_latitude": 41.88 + i * 0.001,
        "as_of_longitude": -87.63 - i * 0.001,
        "has_location": True,
        "n_prior_records": 5,
        "scoring_status": "scored",
        "base_score": round(1.0 - i * 0.01, 6),
        "calibrated_score": round(1.0 - i * 0.01, 6),
        "rank": i + 1,
        "coverage_eligible": False,
        "secondary_no_history": False,
        "selection_mechanism": "selected_by_risk_rank",
        "selection_reason": "selected_by_risk_rank",
        "policy_rank": i + 1,
        "is_selected": True,
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    ordered = [{name: row.get(name) for name in OUTPUT_SCHEMA} for row in rows]
    return pl.DataFrame(ordered, schema=OUTPUT_SCHEMA)


def _organize(rows: list[dict[str, object]], *, threshold_km: float = 1.5) -> pl.DataFrame:
    selected = _frame(rows).filter(pl.col("is_selected"))
    return grouping.assign_geographic_groups(selected, threshold_km=threshold_km)


# --- 1. selected-ID invariant (the most important test) -----------------------


def test_selected_ids_are_unchanged_by_geography() -> None:
    rows = [_row(i) for i in range(20)]
    selected = _frame(rows).filter(pl.col("is_selected"))
    plan = _organize(rows)
    check = validate.check_selected_ids_unchanged(selected, plan)
    assert check.passed
    assert set(selected["target_inspection_id"].to_list()) == set(
        plan["target_inspection_id"].to_list()
    )


# --- 2. location independence of selection -------------------------------------


def test_different_coordinates_can_change_groups_but_never_the_selected_ids() -> None:
    rows_a = [_row(i) for i in range(15)]
    rows_b = [
        _row(i, as_of_latitude=41.80 + i * 0.05, as_of_longitude=-87.90 - i * 0.05)
        for i in range(15)
    ]
    plan_a = _organize(rows_a)
    plan_b = _organize(rows_b)
    assert set(plan_a["target_inspection_id"].to_list()) == set(
        plan_b["target_inspection_id"].to_list()
    )
    # The two coordinate layouts are deliberately different (tight vs. spread out),
    # so at least the group *membership* is allowed to differ -- proving this test
    # is not accidentally checking two identical inputs.
    groups_a = plan_a.sort("target_inspection_id")["geographic_group_id"].to_list()
    groups_b = plan_b.sort("target_inspection_id")["geographic_group_id"].to_list()
    assert groups_a != groups_b


# --- 3. missing location ---------------------------------------------------------


def test_establishments_without_coordinates_are_preserved_and_marked() -> None:
    rows = [_row(i) for i in range(10)]
    rows[3]["as_of_latitude"] = None
    rows[3]["as_of_longitude"] = None
    rows[3]["has_location"] = False
    plan = _organize(rows)
    assert plan.height == 10  # nobody dropped
    row = plan.filter(pl.col("establishment_id") == "EST-000003").row(0, named=True)
    assert row["location_status"] == LocationStatus.LOCATION_UNAVAILABLE.value
    assert row["geographic_group_id"] == UNMAPPED_GROUP_ID
    assert row["as_of_latitude"] is None
    assert row["as_of_longitude"] is None  # never fabricated


def test_missing_location_check_catches_a_dropped_establishment() -> None:
    rows = [_row(i) for i in range(5)]
    rows[0]["as_of_latitude"] = None
    rows[0]["as_of_longitude"] = None
    selected = _frame(rows).filter(pl.col("is_selected"))
    plan = _organize(rows).filter(pl.col("establishment_id") != "EST-000000")  # simulate a drop
    check = validate.check_unmapped_establishments_are_preserved(selected, plan)
    assert not check.passed


# --- 4. location coverage -------------------------------------------------------


def test_location_coverage_counts_are_correct() -> None:
    rows = [_row(i) for i in range(10)]
    for i in (2, 5):
        rows[i]["as_of_latitude"] = None
        rows[i]["as_of_longitude"] = None
    plan = _organize(rows)
    check = validate.check_location_coverage_counts(
        plan, location_available_count=8, location_unavailable_count=2
    )
    assert check.passed
    wrong = validate.check_location_coverage_counts(
        plan, location_available_count=7, location_unavailable_count=3
    )
    assert not wrong.passed


# --- 5. deterministic clustering ------------------------------------------------


def test_same_input_produces_identical_groups_every_time() -> None:
    rows = [_row(i) for i in range(25)]
    first = _organize(rows)
    second = _organize(rows)
    assert first.sort("target_inspection_id").equals(second.sort("target_inspection_id"))


def test_group_ids_are_stable_regardless_of_input_row_order() -> None:
    rows = [_row(i) for i in range(20)]
    shuffled = list(reversed(rows))
    ordered_plan = _organize(rows).sort("establishment_id")
    shuffled_plan = _organize(shuffled).sort("establishment_id")
    assert ordered_plan["geographic_group_id"].to_list() == shuffled_plan[
        "geographic_group_id"
    ].to_list()


def test_group_numbering_follows_centroid_latitude_descending() -> None:
    rows = [_row(i) for i in range(20)]
    plan = _organize(rows)
    check = validate.check_group_ids_ordered_by_centroid_latitude(plan)
    assert check.passed


# --- 6. risk/policy immutability ------------------------------------------------


def test_risk_and_policy_fields_are_never_rewritten() -> None:
    rows = [_row(i) for i in range(15)]
    selected = _frame(rows).filter(pl.col("is_selected"))
    plan = _organize(rows)
    check = validate.check_risk_and_policy_fields_unchanged(selected, plan)
    assert check.passed
    for col in validate.IMMUTABLE_FIELDS:
        left = selected.sort("target_inspection_id")[col].to_list()
        right = plan.sort("target_inspection_id")[col].to_list()
        assert left == right


# --- 7. only the selected set is clustered --------------------------------------


def test_non_selected_rows_are_refused_not_silently_grouped() -> None:
    rows = [_row(i) for i in range(5)]
    rows.append(_row(5, is_selected=False))
    full_frame = _frame(rows)  # deliberately NOT filtered to is_selected
    with pytest.raises(ValueError, match="non-selected"):
        grouping.assign_geographic_groups(full_frame, threshold_km=1.5)


# --- 8. duplicate entity safety --------------------------------------------------


def test_duplicate_establishment_id_is_refused() -> None:
    rows = [_row(i) for i in range(5)]
    rows.append(_row(5, establishment_id="EST-000000"))
    selected = _frame(rows).filter(pl.col("is_selected"))
    with pytest.raises(ValueError, match="duplicate"):
        grouping.assign_geographic_groups(selected, threshold_km=1.5)


def test_no_establishment_appears_in_two_groups() -> None:
    rows = [_row(i) for i in range(20)]
    plan = _organize(rows)
    check = validate.check_no_duplicate_group_membership(plan)
    assert check.passed


# --- 9. distance correctness ------------------------------------------------------


def test_haversine_distance_between_known_chicago_points() -> None:
    d = haversine_km(*CITY_HALL, *WRIGLEY_FIELD)
    assert 6.5 < d < 8.0  # ~7.3 km, real-world great-circle distance

    d_ohare = haversine_km(*CITY_HALL, *O_HARE)
    assert 24.0 < d_ohare < 30.0  # ~27 km


def test_distance_to_self_is_zero() -> None:
    assert haversine_km(*CITY_HALL, *CITY_HALL) == pytest.approx(0.0, abs=1e-9)


def test_distance_is_symmetric() -> None:
    a = haversine_km(*CITY_HALL, *WRIGLEY_FIELD)
    b = haversine_km(*WRIGLEY_FIELD, *CITY_HALL)
    assert a == pytest.approx(b)


def test_points_within_threshold_are_grouped_points_beyond_are_not() -> None:
    close_a = _row(0, as_of_latitude=CITY_HALL[0], as_of_longitude=CITY_HALL[1])
    close_b = _row(
        1,
        as_of_latitude=CITY_HALL[0] + 0.001,
        as_of_longitude=CITY_HALL[1] + 0.001,
    )  # well under 1.5 km
    far = _row(2, as_of_latitude=O_HARE[0], as_of_longitude=O_HARE[1])  # ~27 km away
    plan = _organize([close_a, close_b, far], threshold_km=1.5)
    a_group = plan.filter(pl.col("establishment_id") == "EST-000000")["geographic_group_id"][0]
    b_group = plan.filter(pl.col("establishment_id") == "EST-000001")["geographic_group_id"][0]
    far_group = plan.filter(pl.col("establishment_id") == "EST-000002")["geographic_group_id"][0]
    assert a_group == b_group
    assert far_group != a_group


# --- 10. empty selection -----------------------------------------------------------


def test_empty_selection_is_a_valid_empty_grouping() -> None:
    empty = _frame([]).filter(pl.col("is_selected"))
    plan = grouping.assign_geographic_groups(empty, threshold_km=1.5)
    assert plan.height == 0
    metrics_list = metrics.compute_group_metrics(
        plan.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("location_status"),
            pl.lit(None, dtype=pl.Utf8).alias("geographic_group_id"),
            pl.lit(None, dtype=pl.Utf8).alias("geographic_group_label"),
        )
    )
    assert metrics_list == []


# --- 11. one establishment -----------------------------------------------------------


def test_a_single_establishment_forms_one_group() -> None:
    rows = [_row(0)]
    plan = _organize(rows)
    assert plan.height == 1
    assert plan["geographic_group_id"][0] == "area_1"
    group_metrics = metrics.compute_group_metrics(plan)
    assert len(group_metrics) == 1
    assert group_metrics[0].size == 1
    assert group_metrics[0].max_within_group_distance_km == 0.0
    assert group_metrics[0].avg_within_group_distance_km == 0.0


# --- threshold validation -----------------------------------------------------------


def test_threshold_bounds_are_enforced() -> None:
    validate_threshold(1.5)  # does not raise
    with pytest.raises(GeographicOrganizationDefinitionError):
        validate_threshold(0.0)
    with pytest.raises(GeographicOrganizationDefinitionError):
        validate_threshold(100.0)


# --- coordinates are never fabricated / altered -------------------------------------


def test_coordinates_are_never_fabricated_or_altered_check() -> None:
    rows = [_row(i) for i in range(10)]
    selected = _frame(rows).filter(pl.col("is_selected"))
    plan = _organize(rows)
    check = validate.check_no_fabricated_coordinates(selected, plan)
    assert check.passed

    tampered = plan.with_columns(pl.lit(99.9).alias("as_of_latitude"))
    tampered_check = validate.check_no_fabricated_coordinates(selected, tampered)
    assert not tampered_check.passed
