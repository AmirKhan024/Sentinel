"""Node construction and blocking."""

from __future__ import annotations

import pytest

from sentinel.entity.blocking import candidate_pairs, license_blocks, spatial_blocks
from sentinel.entity.models import DEFAULT_THRESHOLDS, Thresholds
from sentinel.entity.nodes import IDENTITY_COLUMNS, blacklisted_coordinates, build_nodes
from tests.conftest import entity_scenario, make_entity_record

T = DEFAULT_THRESHOLDS


def test_identical_identity_fields_collapse_to_one_node() -> None:
    frame = entity_scenario(
        [
            make_entity_record(1, inspection_id="100001"),
            make_entity_record(2, inspection_id="100002", dba_name=None),
        ]
    )
    # Force the two rows to share every identity field.
    frame = entity_scenario(
        [
            make_entity_record(1, inspection_id="100001"),
            make_entity_record(1, inspection_id="100002"),
        ]
    )
    nodes, assignment = build_nodes(frame)
    assert len(nodes) == 1
    assert set(assignment.values()) == {nodes[0].node_id}
    assert nodes[0].inspection_ids == ("100001", "100002")


def test_differing_identity_fields_produce_separate_nodes() -> None:
    frame = entity_scenario(
        [
            make_entity_record(1, inspection_id="100001", dba_name="ONE"),
            make_entity_record(1, inspection_id="100002", dba_name="TWO"),
        ]
    )
    nodes, _ = build_nodes(frame)
    assert len(nodes) == 2


def test_case_and_whitespace_variants_collapse() -> None:
    frame = entity_scenario(
        [
            make_entity_record(1, inspection_id="100001", dba_name="Joe's Pizza", aka_name="X"),
            make_entity_record(1, inspection_id="100002", dba_name="JOE'S  PIZZA", aka_name="X"),
        ]
    )
    nodes, _ = build_nodes(frame)
    assert len(nodes) == 1


def test_every_inspection_maps_to_exactly_one_node() -> None:
    frame = entity_scenario(
        [make_entity_record(i, inspection_id=str(200000 + i)) for i in range(8)]
    )
    nodes, assignment = build_nodes(frame)
    assert len(assignment) == 8
    assert set(assignment.values()) <= {n.node_id for n in nodes}


def test_node_ids_are_stable_under_row_order() -> None:
    rows = [make_entity_record(i, inspection_id=str(300000 + i)) for i in range(6)]
    forward, _ = build_nodes(entity_scenario(rows))
    backward, _ = build_nodes(entity_scenario(list(reversed(rows))))
    assert [n.node_id for n in forward] == [n.node_id for n in backward]


def test_min_inspection_id_is_numeric_minimum() -> None:
    frame = entity_scenario(
        [
            make_entity_record(1, inspection_id="900"),
            make_entity_record(1, inspection_id="1000"),
        ]
    )
    nodes, _ = build_nodes(frame)
    # Lexicographically "1000" < "900"; numerically it is not.
    assert nodes[0].min_inspection_id == 900


def test_non_numeric_inspection_id_raises() -> None:
    frame = entity_scenario([make_entity_record(1, inspection_id="ABC123")])
    with pytest.raises(ValueError, match="Non-numeric inspection_id"):
        build_nodes(frame)


def test_missing_identity_column_raises() -> None:
    frame = entity_scenario([make_entity_record(1)]).drop("dba_name")
    with pytest.raises(ValueError, match="missing required identity columns"):
        build_nodes(frame)


def test_identity_columns_exclude_every_outcome_field() -> None:
    """The leakage boundary, asserted rather than assumed (findings §14)."""
    for outcome in ("results", "violations", "risk", "inspection_date"):
        assert outcome not in IDENTITY_COLUMNS


def test_sentinel_licence_is_dropped_from_the_node() -> None:
    frame = entity_scenario([make_entity_record(1, license_="0")])
    nodes, _ = build_nodes(frame)
    assert nodes[0].license_key is None


def test_blacklisted_coordinates_flags_overloaded_points() -> None:
    rows = [
        make_entity_record(
            i,
            inspection_id=str(400000 + i),
            address=f"{100 + i} N MAIN ST",
            latitude="41.8781",
            longitude="-87.6298",
        )
        for i in range(6)
    ]
    nodes, _ = build_nodes(entity_scenario(rows))
    assert blacklisted_coordinates(nodes, max_addresses=4) == frozenset({"41.8781,-87.6298"})


def test_blacklist_is_empty_when_a_coordinate_covers_few_addresses() -> None:
    rows = [
        make_entity_record(i, inspection_id=str(410000 + i), address=f"{100 + i} N MAIN ST")
        for i in range(2)
    ]
    nodes, _ = build_nodes(entity_scenario(rows))
    assert blacklisted_coordinates(nodes, max_addresses=4) == frozenset()


# --- blocking ------------------------------------------------------------


def test_spatial_blocks_key_on_zip_and_house_number() -> None:
    rows = [
        make_entity_record(1, inspection_id="1", address="123 N MAIN ST", dba_name="A"),
        make_entity_record(2, inspection_id="2", address="123 N MAIN AVE", dba_name="B"),
    ]
    nodes, _ = build_nodes(entity_scenario(rows))
    blocks = spatial_blocks(nodes)
    assert list(blocks) == ["60601|123"]
    assert len(blocks["60601|123"]) == 2


def test_nodes_without_a_house_number_are_absent_from_spatial_blocks() -> None:
    rows = [make_entity_record(1, inspection_id="1", address="MERCHANDISE MART PLAZA")]
    nodes, _ = build_nodes(entity_scenario(rows))
    assert spatial_blocks(nodes) == {}


def test_licence_blocks_exclude_sentinels() -> None:
    rows = [
        make_entity_record(1, inspection_id="1", license_="0", dba_name="A"),
        make_entity_record(2, inspection_id="2", license_="0", dba_name="B"),
    ]
    nodes, _ = build_nodes(entity_scenario(rows))
    assert license_blocks(nodes) == {}


def test_candidate_pairs_are_canonically_ordered_and_unique() -> None:
    rows = [make_entity_record(i, inspection_id=str(i), dba_name=f"NAME {i}") for i in range(5)]
    nodes, _ = build_nodes(entity_scenario(rows))
    pairs, _ = candidate_pairs(nodes, T)
    assert all(left < right for left, right in pairs)
    assert len(pairs) == len(set(pairs))


def test_no_self_pairs_are_generated() -> None:
    rows = [make_entity_record(i, inspection_id=str(i), dba_name=f"NAME {i}") for i in range(4)]
    nodes, _ = build_nodes(entity_scenario(rows))
    pairs, _ = candidate_pairs(nodes, T)
    assert all(left != right for left, right in pairs)


def test_oversized_blocks_are_skipped_and_reported() -> None:
    """A skipped block splits rather than guesses, and says so."""
    rows = [
        make_entity_record(i, inspection_id=str(500000 + i), dba_name=f"NAME {i}") for i in range(6)
    ]
    nodes, _ = build_nodes(entity_scenario(rows))
    tight = Thresholds(max_block_size=3)
    pairs, oversized = candidate_pairs(nodes, tight)
    assert pairs == []
    assert oversized
    assert all(block.size > 3 for block in oversized)


def test_pair_generation_is_stable_across_runs() -> None:
    rows = [make_entity_record(i, inspection_id=str(i), dba_name=f"NAME {i}") for i in range(6)]
    nodes, _ = build_nodes(entity_scenario(rows))
    first, _ = candidate_pairs(nodes, T)
    second, _ = candidate_pairs(list(reversed(nodes)), T)
    assert first == second
