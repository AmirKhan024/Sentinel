"""Clustering, establishment identity and the degradation ladder."""

from __future__ import annotations

from sentinel.entity.cluster import (
    build_clusters,
    check_invariants,
    cluster_content_sha256,
    establishment_id_for,
)
from sentinel.entity.models import (
    DEFAULT_THRESHOLDS,
    MatchTier,
    PairSignals,
    PairVerdict,
    Thresholds,
)
from tests.test_entity_evidence import node

T = DEFAULT_THRESHOLDS


def edge(left: str, right: str, tier: MatchTier = MatchTier.STRONG) -> PairVerdict:
    return PairVerdict(left, right, tier, "S1", PairSignals(same_license=True, same_addr_key=True))


# --- establishment identity ---------------------------------------------


def test_establishment_id_anchors_on_the_earliest_inspection() -> None:
    members = [node("N-1", inspection_id=500), node("N-2", inspection_id=100)]
    assert establishment_id_for(members) == "EST-00000000100"


def test_establishment_id_is_zero_padded_to_eleven_digits() -> None:
    assert establishment_id_for([node("N-1", inspection_id=67435)]) == "EST-00000067435"


def test_establishment_id_ignores_member_order() -> None:
    members = [node("N-1", inspection_id=500), node("N-2", inspection_id=100)]
    assert establishment_id_for(members) == establishment_id_for(list(reversed(members)))


def test_establishment_id_compares_ids_numerically() -> None:
    """Lexicographically "1000" sorts before "900"; numerically it does not."""
    members = [node("N-1", inspection_id=900), node("N-2", inspection_id=1000)]
    assert establishment_id_for(members) == "EST-00000000900"


def test_content_hash_ignores_member_order() -> None:
    members = [node("N-1"), node("N-2")]
    assert cluster_content_sha256(members) == cluster_content_sha256(list(reversed(members)))


def test_content_hash_changes_when_membership_changes() -> None:
    base = [node("N-1"), node("N-2")]
    grown = [*base, node("N-3")]
    assert cluster_content_sha256(base) != cluster_content_sha256(grown)


# --- invariants ----------------------------------------------------------


def test_a_clean_cluster_violates_nothing() -> None:
    assert check_invariants([node("N-1"), node("N-2")], T) == []


def test_too_many_addresses_is_a_violation() -> None:
    members = [
        node(f"N-{i}", address=f"{100 + i} N MAIN ST", inspection_id=100 + i) for i in range(6)
    ]
    assert "too_many_addresses" in check_invariants(members, T)


def test_multiple_zips_is_a_violation() -> None:
    members = [node("N-1", zip_code="60601"), node("N-2", zip_code="60602")]
    assert "too_many_zips" in check_invariants(members, T)


def test_conflicting_units_is_a_violation() -> None:
    members = [
        node("N-1", address="123 N MAIN ST STE 100"),
        node("N-2", address="123 N MAIN ST STE 200"),
    ]
    assert "conflicting_units" in check_invariants(members, T)


def test_conflicting_directionals_is_a_violation() -> None:
    members = [node("N-1", address="123 N MAIN ST"), node("N-2", address="123 S MAIN ST")]
    assert "conflicting_directionals" in check_invariants(members, T)


def test_geographic_spread_is_a_violation() -> None:
    members = [
        node("N-1", lat="41.8781", lon="-87.6298"),
        node("N-2", lat="42.0500", lon="-87.9000"),
    ]
    assert "geographic_spread" in check_invariants(members, T)


def test_nearby_coordinates_are_not_a_violation() -> None:
    members = [
        node("N-1", lat="41.8781", lon="-87.6298"),
        node("N-2", lat="41.8782", lon="-87.6299"),
    ]
    assert "geographic_spread" not in check_invariants(members, T)


# --- clustering ----------------------------------------------------------


def test_unconnected_nodes_become_singletons() -> None:
    nodes = [node("N-1", inspection_id=1), node("N-2", inspection_id=2)]
    clusters, splits = build_clusters(nodes, [], T)
    assert len(clusters) == 2
    assert splits == {}


def test_connected_nodes_form_one_establishment() -> None:
    nodes = [node("N-1", inspection_id=1), node("N-2", inspection_id=2)]
    clusters, _ = build_clusters(nodes, [edge("N-1", "N-2")], T)
    assert len(clusters) == 1
    assert clusters[0].node_ids == ("N-1", "N-2")
    assert clusters[0].confidence == "high"


def test_a_probable_edge_lowers_confidence_to_medium() -> None:
    nodes = [node("N-1", inspection_id=1), node("N-2", inspection_id=2)]
    clusters, _ = build_clusters(nodes, [edge("N-1", "N-2", MatchTier.PROBABLE)], T)
    assert clusters[0].confidence == "medium"


def test_ambiguous_edges_do_not_merge() -> None:
    nodes = [node("N-1", inspection_id=1), node("N-2", inspection_id=2)]
    clusters, _ = build_clusters(nodes, [edge("N-1", "N-2", MatchTier.AMBIGUOUS)], T)
    assert len(clusters) == 2


def test_no_match_edges_do_not_merge() -> None:
    nodes = [node("N-1", inspection_id=1), node("N-2", inspection_id=2)]
    clusters, _ = build_clusters(nodes, [edge("N-1", "N-2", MatchTier.NO_MATCH)], T)
    assert len(clusters) == 2


def test_dropping_probable_edges_repairs_an_invalid_cluster() -> None:
    """Rung 1 of the degradation ladder."""
    nodes = [
        node("N-1", zip_code="60601", inspection_id=1),
        node("N-2", zip_code="60602", inspection_id=2),
    ]
    verdicts = [edge("N-1", "N-2", MatchTier.PROBABLE)]
    clusters, splits = build_clusters(nodes, verdicts, T)
    assert len(clusters) == 2
    assert all(c.split_reason == "probable_edges_dropped" for c in clusters)
    assert all(c.confidence == "reduced" for c in clusters)
    assert any(reason.startswith("invariant:") for reason in splits)


def test_address_split_is_used_when_strong_edges_still_violate() -> None:
    """Rung 2: the strong edges themselves produce an invalid cluster."""
    nodes = [
        node("N-1", address="123 N MAIN ST", zip_code="60601", inspection_id=1),
        node("N-2", address="456 N OTHER ST", zip_code="60602", inspection_id=2),
    ]
    clusters, splits = build_clusters(nodes, [edge("N-1", "N-2")], T)
    assert len(clusters) == 2
    assert splits.get("address_split", 0) >= 1
    assert all(c.split_reason in {"address_split", "atomised"} for c in clusters)


def test_atomisation_is_the_last_resort() -> None:
    """Rung 3: even one address group is invalid, so every node stands alone."""
    nodes = [
        node("N-1", address="123 N MAIN ST STE 100", inspection_id=1),
        node("N-2", address="123 N MAIN ST STE 200", inspection_id=2),
    ]
    # Same address key (the unit is stripped from the key), conflicting units.
    clusters, splits = build_clusters(nodes, [edge("N-1", "N-2")], T)
    assert len(clusters) == 2
    assert splits.get("atomised", 0) >= 1
    assert all(c.split_reason == "atomised" for c in clusters)


def test_every_node_lands_in_exactly_one_cluster() -> None:
    nodes = [node(f"N-{i}", inspection_id=i + 1) for i in range(5)]
    verdicts = [edge("N-0", "N-1"), edge("N-2", "N-3")]
    clusters, _ = build_clusters(nodes, verdicts, T)
    placed = [n for c in clusters for n in c.node_ids]
    assert sorted(placed) == sorted(n.node_id for n in nodes)


def test_clusters_are_returned_in_a_stable_order() -> None:
    nodes = [node(f"N-{i}", inspection_id=i + 1) for i in range(5)]
    forward, _ = build_clusters(nodes, [], T)
    backward, _ = build_clusters(list(reversed(nodes)), [], T)
    assert [c.establishment_id for c in forward] == [c.establishment_id for c in backward]


def test_clustering_is_invariant_to_edge_order() -> None:
    nodes = [node(f"N-{i}", inspection_id=i + 1) for i in range(4)]
    verdicts = [edge("N-0", "N-1"), edge("N-1", "N-2")]
    forward, _ = build_clusters(nodes, verdicts, T)
    backward, _ = build_clusters(nodes, list(reversed(verdicts)), T)
    assert [c.node_ids for c in forward] == [c.node_ids for c in backward]


def test_relaxed_thresholds_allow_a_wider_cluster() -> None:
    nodes = [
        node("N-1", zip_code="60601", inspection_id=1),
        node("N-2", zip_code="60602", inspection_id=2),
    ]
    relaxed = Thresholds(max_zips_per_cluster=2)
    clusters, splits = build_clusters(nodes, [edge("N-1", "N-2")], relaxed)
    assert len(clusters) == 1
    assert splits == {}
