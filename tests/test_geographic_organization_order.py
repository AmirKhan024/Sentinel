"""Component 20 v2: work blocks and suggested work order.

Builds synthetic Component 19-shaped, already-grouped frames directly, matching
``test_geographic_organization.py``'s existing convention.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.geographic_organization import grouping, metrics, organization
from sentinel.geographic_organization.definitions import (
    GEO_THRESHOLD_PRESETS,
    UNMAPPED_GROUP_ID,
    GeographicOrganizationDefinitionError,
    OrganizationMode,
    resolve_threshold_km,
)
from sentinel.geographic_organization.distance import haversine_km
from sentinel.operational_selection.writer import OUTPUT_SCHEMA

PLANNING_DATE = "2026-08-28"
CITY_HALL = (41.8838, -87.6319)


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
        "as_of_latitude": CITY_HALL[0] + i * 0.001,
        "as_of_longitude": CITY_HALL[1] - i * 0.001,
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


def _grouped(rows: list[dict[str, object]], *, threshold_km: float = 1.5) -> pl.DataFrame:
    selected = _frame(rows).filter(pl.col("is_selected"))
    return grouping.assign_geographic_groups(selected, threshold_km=threshold_km)


# --- threshold resolution -----------------------------------------------------------


def test_threshold_preset_resolves_to_documented_km_value() -> None:
    for name, km in GEO_THRESHOLD_PRESETS.items():
        assert resolve_threshold_km(threshold_km=None, threshold_preset=name) == km


def test_threshold_km_and_preset_together_is_refused() -> None:
    with pytest.raises(GeographicOrganizationDefinitionError, match="both given"):
        resolve_threshold_km(threshold_km=2.0, threshold_preset="broad")


def test_neither_threshold_arg_uses_the_default() -> None:
    from sentinel.geographic_organization.definitions import DEFAULT_GEO_THRESHOLD_KM

    assert resolve_threshold_km(threshold_km=None, threshold_preset=None) == DEFAULT_GEO_THRESHOLD_KM


def test_unknown_threshold_preset_is_refused() -> None:
    with pytest.raises(GeographicOrganizationDefinitionError):
        resolve_threshold_km(threshold_km=None, threshold_preset="extreme")


# --- work block construction --------------------------------------------------------


def test_work_blocks_carry_highest_sentinel_rank_and_rank_range() -> None:
    # Three establishments close together with ranks 5, 1, 9 -- highest priority is rank 1.
    rows = [
        _row(0, policy_rank=5, rank=5),
        _row(1, policy_rank=1, rank=1),
        _row(2, policy_rank=9, rank=9),
    ]
    grouped = _grouped(rows, threshold_km=5.0)
    group_metrics = metrics.compute_group_metrics(grouped)
    blocks = organization.build_work_blocks(grouped, group_metrics)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.highest_sentinel_rank == 1
    assert block.rank_range == (1, 9)
    assert block.member_ranks == [1, 5, 9]
    assert not block.has_unmapped_member


def test_unmapped_block_has_no_rank_range_but_is_labelled() -> None:
    rows = [_row(0, as_of_latitude=None, as_of_longitude=None, has_location=False)]
    grouped = _grouped(rows)
    group_metrics = metrics.compute_group_metrics(grouped)
    blocks = organization.build_work_blocks(grouped, group_metrics)
    assert len(blocks) == 1
    assert blocks[0].block_id == UNMAPPED_GROUP_ID
    assert blocks[0].has_unmapped_member
    assert "coordinates are unavailable" in blocks[0].rationale.lower()


# --- suggested_work_order: risk_first -------------------------------------------------


def test_risk_first_order_is_exactly_policy_rank_ascending() -> None:
    rows = [
        _row(0, policy_rank=9),
        _row(1, policy_rank=2),
        _row(2, policy_rank=5),
    ]
    grouped = _grouped(rows, threshold_km=5.0)
    order = organization.suggested_work_order(grouped, mode=OrganizationMode.RISK_FIRST)
    ranks_in_order = [rank for _, _, rank in order]
    assert ranks_in_order == [2, 5, 9]


def test_risk_first_order_never_reorders_regardless_of_geography() -> None:
    # Deliberately scatter coordinates so geography alone would suggest a different order.
    rows = [
        _row(0, policy_rank=1, as_of_latitude=41.70, as_of_longitude=-87.90),
        _row(1, policy_rank=2, as_of_latitude=41.88, as_of_longitude=-87.63),
        _row(2, policy_rank=3, as_of_latitude=42.00, as_of_longitude=-87.70),
    ]
    grouped = _grouped(rows, threshold_km=50.0)  # force one block regardless of spread
    order = organization.suggested_work_order(grouped, mode=OrganizationMode.RISK_FIRST)
    assert [rank for _, _, rank in order] == [1, 2, 3]


# --- suggested_work_order: geography_assisted ------------------------------------------


def test_geography_assisted_order_is_a_valid_permutation() -> None:
    rows = [_row(i, policy_rank=i + 1) for i in range(6)]
    grouped = _grouped(rows, threshold_km=5.0)
    order = organization.suggested_work_order(grouped, mode=OrganizationMode.GEOGRAPHY_ASSISTED)
    ids_in_order = sorted(tid for _, tid, _ in order)
    assert ids_in_order == sorted(grouped["target_inspection_id"].to_list())
    assert [idx for idx, _, _ in order] == list(range(1, 7))


def test_geography_assisted_order_is_deterministic() -> None:
    rows = [_row(i, policy_rank=(i * 7) % 11 + 1) for i in range(8)]
    grouped = _grouped(rows, threshold_km=5.0)
    first = organization.suggested_work_order(grouped, mode=OrganizationMode.GEOGRAPHY_ASSISTED)
    second = organization.suggested_work_order(grouped, mode=OrganizationMode.GEOGRAPHY_ASSISTED)
    assert first == second


def test_geography_assisted_order_starts_at_highest_priority_member() -> None:
    rows = [
        _row(0, policy_rank=9, as_of_latitude=41.90, as_of_longitude=-87.60),
        _row(1, policy_rank=1, as_of_latitude=41.80, as_of_longitude=-87.90),
        _row(2, policy_rank=5, as_of_latitude=41.85, as_of_longitude=-87.75),
    ]
    grouped = _grouped(rows, threshold_km=50.0)
    order = organization.suggested_work_order(grouped, mode=OrganizationMode.GEOGRAPHY_ASSISTED)
    first_rank = order[0][2]
    assert first_rank == 1  # seeded from the highest-priority (lowest policy_rank) member


def test_geography_assisted_may_reorder_relative_to_strict_rank() -> None:
    # rank-1 establishment is far from the other two, which are close together.
    rows = [
        _row(0, policy_rank=1, as_of_latitude=41.70, as_of_longitude=-87.90),
        _row(1, policy_rank=2, as_of_latitude=41.90, as_of_longitude=-87.60),
        _row(2, policy_rank=3, as_of_latitude=41.9001, as_of_longitude=-87.6001),
    ]
    grouped = _grouped(rows, threshold_km=50.0)
    order = organization.suggested_work_order(grouped, mode=OrganizationMode.GEOGRAPHY_ASSISTED)
    ranks_in_order = [rank for _, _, rank in order]
    # Starts at rank 1 (highest priority), but the tail visits the two close establishments
    # nearest-first -- rank 3 is nearer to rank 1's *last visited* neighbour, not necessarily
    # strict rank order end to end. The key property under test is: it is a valid order that
    # starts at rank 1.
    assert ranks_in_order[0] == 1
    assert set(ranks_in_order) == {1, 2, 3}


def test_single_member_block_order_is_trivial_regardless_of_mode() -> None:
    rows = [_row(0, policy_rank=7)]
    grouped = _grouped(rows)
    for mode in (OrganizationMode.RISK_FIRST, OrganizationMode.GEOGRAPHY_ASSISTED):
        order = organization.suggested_work_order(grouped, mode=mode)
        assert order == [(1, grouped["target_inspection_id"][0], 7)]


# --- rationale strings ----------------------------------------------------------------


def test_every_organization_mode_has_a_rationale() -> None:
    for mode in OrganizationMode:
        rationale = organization.organization_mode_rationale(mode)
        assert isinstance(rationale, str)
        assert rationale
        assert "route" not in rationale.lower() or "not a driving route" in rationale.lower()
