"""Pairwise evidence rules and vetoes.

Every rule and every veto gets a test that builds the minimal node pair which
triggers it. The veto tests deliberately pair a veto against otherwise-strong
agreement, because the whole point of a veto is that it outranks agreement.
"""

from __future__ import annotations

import pytest

from sentinel.entity.evidence import (
    aka_conflict,
    compute_signals,
    evaluate_pair,
    haversine_m,
    name_containment,
    names_match,
)
from sentinel.entity.models import DEFAULT_THRESHOLDS, Geo, MatchTier, Node
from sentinel.entity.normalize import (
    normalize_address,
    normalize_facility_type,
    normalize_geo,
    normalize_license,
    normalize_name,
)

T = DEFAULT_THRESHOLDS


def node(
    node_id: str,
    *,
    name: str = "JOES DINER",
    aka: str | None = None,
    address: str = "123 N MAIN ST",
    zip_code: str | None = "60601",
    license_: str | None = "1000001",
    facility: str | None = "Restaurant",
    lat: str | None = "41.8781",
    lon: str | None = "-87.6298",
    inspection_id: int = 100001,
) -> Node:
    """Build a node directly, so a test states only what it is about."""
    return Node(
        node_id=node_id,
        license_key=normalize_license(license_),
        name=normalize_name(name),
        aka=normalize_name(aka if aka is not None else name),
        address=normalize_address(address, zip_code),
        geo=normalize_geo(lat, lon),
        facility_type_key=normalize_facility_type(facility),
        inspection_ids=(str(inspection_id),),
        min_inspection_id=inspection_id,
        raw_name=name,
        raw_address=address,
        raw_zip=zip_code,
    )


def verdict(left: Node, right: Node) -> tuple[str, MatchTier]:
    result = evaluate_pair(left, right, T)
    return result.rule_id, result.tier


# --- haversine -----------------------------------------------------------


def test_haversine_zero_distance() -> None:
    assert haversine_m(41.8781, -87.6298, 41.8781, -87.6298) == pytest.approx(0.0, abs=1e-6)


def test_haversine_one_degree_of_latitude() -> None:
    # ~111.19 km per degree of latitude on a sphere.
    assert haversine_m(41.0, -87.0, 42.0, -87.0) == pytest.approx(111_195, rel=0.001)


def test_haversine_is_symmetric() -> None:
    a = haversine_m(41.8781, -87.6298, 41.9000, -87.6500)
    b = haversine_m(41.9000, -87.6500, 41.8781, -87.6298)
    assert a == pytest.approx(b)


# --- name helpers --------------------------------------------------------


def test_names_match_across_dba_and_aka() -> None:
    """Findings §5: the legal entity is renamed while the premises is not."""
    legal = node("N-1", name="1918 WINTER STREET ILLINOIS LLC", aka="MARIANOS")
    trade = node("N-2", name="MARIANOS", aka="MARIANOS")
    assert names_match(legal, trade)


def test_name_containment_detects_a_strict_subset() -> None:
    assert name_containment(node("N-1", name="TACO SHOP"), node("N-2", name="TONYS TACO SHOP"))


def test_name_containment_is_false_for_identical_names() -> None:
    assert not name_containment(node("N-1", name="TACO SHOP"), node("N-2", name="TACO SHOP"))


def test_name_containment_is_false_for_unrelated_names() -> None:
    assert not name_containment(node("N-1", name="TACO SHOP"), node("N-2", name="PIZZA PLACE"))


def test_aka_conflict_when_trade_names_disagree() -> None:
    """The O'Hare concessionaire case."""
    starbucks = node("N-1", name="HOST INTERNATIONAL INC", aka="STARBUCKS T3 H2")
    rockets = node("N-2", name="HOST INTERNATIONAL INC", aka="JOHNNY ROCKETS T2")
    assert aka_conflict(starbucks, rockets)


def test_no_aka_conflict_when_one_trade_name_matches() -> None:
    plain = node("N-1", name="SUBWAY", aka="SUBWAY")
    numbered = node("N-2", name="SUBWAY", aka="SUBWAY 1234")
    assert not aka_conflict(plain, numbered)


def test_no_aka_conflict_when_an_aka_is_missing() -> None:
    assert not aka_conflict(node("N-1", aka=""), node("N-2", name="OTHER NAME"))


# --- strong rules --------------------------------------------------------


def test_s1_same_licence_same_address() -> None:
    assert verdict(node("N-1"), node("N-2", name="OTHER PLACE", aka="OTHER PLACE")) == (
        "S1",
        MatchTier.STRONG,
    )


def test_s2_same_address_same_name_different_licences() -> None:
    """The false-split fix: ABC RESTAURANT and ABC RESTAURANT LLC are one place.

    Findings §3.2: 18.47% of establishments hold more than one licence, so
    requiring licence agreement here would fracture them.
    """
    left = node("N-1", name="ABC RESTAURANT", address="123 MAIN ST", license_="10001")
    right = node("N-2", name="ABC RESTAURANT LLC", address="123 MAIN STREET", license_="20002")
    assert verdict(left, right) == ("S2", MatchTier.STRONG)


def test_s2_bridges_a_geocoded_address_variant() -> None:
    """Findings §8: a shared coordinate stands in for address equality."""
    left = node("N-1", address="5700 S LAKE SHORE DR", license_="10001")
    right = node(
        "N-2",
        address="5700 S JEAN BAPTISTE POINTE DUSABLE LAKE SHORE DR",
        license_="20002",
    )
    rule, tier = verdict(left, right)
    assert tier is MatchTier.STRONG
    assert rule == "S2"


def test_s3_same_licence_and_name_one_house_number_apart() -> None:
    left = node("N-1", address="123 N MAIN ST")
    right = node("N-2", address="125 N MAIN ST", lat="41.8782", lon="-87.6299")
    assert verdict(left, right) == ("S3", MatchTier.STRONG)


# --- probable rules ------------------------------------------------------


def test_p1_same_licence_near_address_different_names() -> None:
    left = node("N-1", name="ONE NAME", aka="ONE NAME", address="123 N MAIN ST")
    right = node(
        "N-2",
        name="OTHER NAME",
        aka="OTHER NAME",
        address="125 N MAIN ST",
        lat="41.8782",
        lon="-87.6299",
    )
    assert verdict(left, right) == ("P1", MatchTier.PROBABLE)


def test_p2_name_containment_at_one_address() -> None:
    left = node("N-1", name="HOT WOK CHINESE KITCHEN", license_="10001")
    right = node("N-2", name="NEW HOT WOK CHINESE KITCHEN", license_="20002")
    assert verdict(left, right) == ("P2", MatchTier.PROBABLE)


# --- ambiguous -----------------------------------------------------------


def test_a1_one_licence_at_two_different_places_is_not_merged() -> None:
    """Findings §3.3: 86 licences span more than a kilometre. Splitting is the
    safe error."""
    left = node("N-1", address="123 N MAIN ST")
    right = node("N-2", address="999 N OTHER ST", lat="41.9000", lon="-87.7000")
    rule, tier = verdict(left, right)
    assert (rule, tier) == ("A1", MatchTier.AMBIGUOUS)


def test_a2_containment_with_conflicting_facility_types() -> None:
    left = node("N-1", name="TACO SHOP", license_="10001", facility="Restaurant")
    right = node("N-2", name="TONYS TACO SHOP", license_="20002", facility="Grocery Store")
    assert verdict(left, right) == ("A2", MatchTier.AMBIGUOUS)


# --- no match ------------------------------------------------------------


def test_n2_different_businesses_at_one_address_are_not_ambiguous() -> None:
    """KARMA MINI MART and MB & S MARKET share an address and nothing else.

    Classifying this as ambiguous would bury the real review queue under
    ~108k pairs of ordinary neighbours, so it is a decision, not a doubt.
    """
    left = node("N-1", name="KARMA MINI MART", aka="KARMA MINI MART", license_="10001")
    right = node("N-2", name="MB AND S MARKET", aka="MB AND S MARKET", license_="20002")
    assert verdict(left, right) == ("N2", MatchTier.NO_MATCH)


def test_n0_unrelated_nodes_do_not_match() -> None:
    left = node("N-1", name="A PLACE", aka="A PLACE", address="1 N FIRST ST", license_="10001")
    right = node(
        "N-2",
        name="B PLACE",
        aka="B PLACE",
        address="2 N SECOND ST",
        license_="20002",
        lat="41.9",
        lon="-87.7",
    )
    assert verdict(left, right) == ("N0", MatchTier.NO_MATCH)


# --- vetoes, each beating otherwise-strong agreement ---------------------


def test_v1_directional_conflict_beats_a_matching_name_and_licence() -> None:
    left = node("N-1", address="123 N MAIN ST")
    right = node("N-2", address="123 S MAIN ST", lat="41.8000", lon="-87.6298")
    rule, tier = verdict(left, right)
    assert rule == "V1"
    assert tier is MatchTier.NO_MATCH


def test_v2_unit_conflict_beats_a_matching_name() -> None:
    left = node("N-1", address="123 N MAIN ST STE 100", license_="10001")
    right = node("N-2", address="123 N MAIN ST STE 200", license_="20002")
    assert verdict(left, right) == ("V2", MatchTier.NO_MATCH)


def test_v2_is_waived_when_the_licence_agrees() -> None:
    """One licence covering two suites is one operator's premises."""
    left = node("N-1", address="123 N MAIN ST STE 100", license_="10001")
    right = node("N-2", address="123 N MAIN ST STE 200", license_="10001")
    _, tier = verdict(left, right)
    assert tier is MatchTier.STRONG


def test_v3_store_number_conflict_beats_a_matching_name_and_address() -> None:
    """SUBWAY 1234 and SUBWAY 5678 at one address are different franchises."""
    left = node("N-1", name="SUBWAY 1234", aka="SUBWAY 1234", license_="10001")
    right = node("N-2", name="SUBWAY 5678", aka="SUBWAY 5678", license_="20002")
    assert verdict(left, right) == ("V3", MatchTier.NO_MATCH)


def test_v4_conflicting_trade_names_beat_a_shared_operator_name() -> None:
    """The O'Hare over-merge this veto was added to fix."""
    left = node("N-1", name="HOST INTERNATIONAL INC", aka="STARBUCKS", license_="10001")
    right = node("N-2", name="HOST INTERNATIONAL INC", aka="CHICAGO WATER WORKS", license_="20002")
    assert verdict(left, right) == ("V4", MatchTier.NO_MATCH)


def test_v4_is_waived_when_the_licence_agrees() -> None:
    """Findings §11: successive businesses on one licence at one address are the
    same physical premises under Sentinel's definition."""
    left = node("N-1", name="OLD TOWN BURGER SALOON", aka="OLD TOWN BURGER SALOON")
    right = node("N-2", name="TAVERN ON WELLS", aka="TAVERN ON WELLS")
    _, tier = verdict(left, right)
    assert tier is MatchTier.STRONG


# --- signals and symmetry ------------------------------------------------


def test_blank_facility_type_is_unknown_not_a_mismatch() -> None:
    signals = compute_signals(node("N-1", facility=""), node("N-2", facility="Restaurant"), T)
    assert signals.facility_agree is None


def test_blacklisted_coordinate_does_not_provide_address_equality() -> None:
    left = node("N-1", address="100 N FIRST ST")
    right = node("N-2", address="200 N SECOND ST", license_="20002")
    blacklist = frozenset({"41.8781,-87.6298"})
    assert not compute_signals(left, right, T, blacklist).same_coord


def test_same_coord_requires_matching_zip() -> None:
    left = node("N-1", address="100 N FIRST ST", zip_code="60601")
    right = node("N-2", address="200 N SECOND ST", zip_code="60602")
    assert not compute_signals(left, right, T).same_coord


@pytest.mark.parametrize(
    ("left_kwargs", "right_kwargs"),
    [
        ({}, {"name": "OTHER"}),
        ({"license_": "1"}, {"license_": "2", "name": "SUBWAY 1"}),
        ({"address": "123 N MAIN ST"}, {"address": "123 S MAIN ST"}),
        ({"name": "TACO SHOP"}, {"name": "TONYS TACO SHOP", "license_": "9"}),
    ],
)
def test_evaluation_is_symmetric(left_kwargs: dict[str, str], right_kwargs: dict[str, str]) -> None:
    """Argument order must never change the outcome."""
    left = node("N-1", **left_kwargs)  # type: ignore[arg-type]
    right = node("N-2", **right_kwargs)  # type: ignore[arg-type]
    forward = evaluate_pair(left, right, T)
    backward = evaluate_pair(right, left, T)
    assert forward.tier is backward.tier
    assert forward.rule_id == backward.rule_id
    assert (forward.left_node_id, forward.right_node_id) == (
        backward.left_node_id,
        backward.right_node_id,
    )


def test_unusable_licence_provides_no_evidence() -> None:
    """Findings §2: the '0' sentinel must not join 323 unrelated businesses."""
    left = node("N-1", name="A PLACE", aka="A PLACE", address="1 N FIRST ST", license_="0")
    right = node(
        "N-2",
        name="B PLACE",
        aka="B PLACE",
        address="2 N SECOND ST",
        license_="0",
        lat="41.9",
        lon="-87.7",
    )
    _, tier = verdict(left, right)
    assert tier is MatchTier.NO_MATCH


def test_nodes_without_usable_geo_still_match_on_address() -> None:
    left = node("N-1", lat=None, lon=None, license_="10001")
    right = node("N-2", lat=None, lon=None, license_="20002")
    assert left.geo == Geo(key=None, reason="missing")
    _, tier = verdict(left, right)
    assert tier is MatchTier.STRONG
