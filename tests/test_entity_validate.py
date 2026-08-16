"""Validation checks.

Each error-severity check gets a passing case and a failing case: a check that
cannot fail is not a check.
"""

from __future__ import annotations

from sentinel.entity.cluster import cluster_content_sha256
from sentinel.entity.models import DEFAULT_THRESHOLDS, Cluster, MatchTier, PairSignals, PairVerdict
from sentinel.entity.validate import (
    SEVERITY_ERROR,
    format_report,
    has_failures,
    validate_output,
)
from tests.test_entity_evidence import node

T = DEFAULT_THRESHOLDS


def cluster_of(*nodes, confidence: str = "high", split_reason: str | None = None) -> Cluster:  # type: ignore[no-untyped-def]
    from sentinel.entity.cluster import establishment_id_for

    return Cluster(
        establishment_id=establishment_id_for(list(nodes)),
        node_ids=tuple(sorted(n.node_id for n in nodes)),
        confidence=confidence,
        split_reason=split_reason,
        content_sha256=cluster_content_sha256(list(nodes)),
    )


def named(checks, name):  # type: ignore[no-untyped-def]
    return next(c for c in checks if c.name == name)


def test_a_correct_run_has_no_failures() -> None:
    a, b = node("N-1", inspection_id=1), node("N-2", inspection_id=2)
    clusters = [cluster_of(a), cluster_of(b)]
    checks = validate_output([a, b], clusters, [], {"1": "N-1", "2": "N-2"}, T, source_row_count=2)
    assert not has_failures(checks)


def test_row_count_mismatch_is_detected() -> None:
    a = node("N-1", inspection_id=1)
    checks = validate_output([a], [cluster_of(a)], [], {"1": "N-1"}, T, source_row_count=99)
    assert not named(checks, "every_inspection_assigned").passed
    assert has_failures(checks)


def test_a_node_in_two_clusters_is_detected() -> None:
    a, b = node("N-1", inspection_id=1), node("N-2", inspection_id=2)
    # Deliberately malformed: N-1 appears in both clusters.
    bad = Cluster("EST-00000000002", ("N-1", "N-2"), "high", None, "x")
    checks = validate_output(
        [a, b], [cluster_of(a), bad], [], {"1": "N-1", "2": "N-2"}, T, source_row_count=2
    )
    assert not named(checks, "node_in_exactly_one_cluster").passed


def test_an_unclustered_node_is_detected() -> None:
    a, b = node("N-1", inspection_id=1), node("N-2", inspection_id=2)
    checks = validate_output(
        [a, b], [cluster_of(a)], [], {"1": "N-1", "2": "N-2"}, T, source_row_count=2
    )
    assert not named(checks, "every_node_clustered").passed


def test_a_malformed_establishment_id_is_detected() -> None:
    a = node("N-1", inspection_id=1)
    bad = Cluster("not-an-id", ("N-1",), "high", None, "x")
    checks = validate_output([a], [bad], [], {"1": "N-1"}, T, source_row_count=1)
    assert not named(checks, "establishment_id_format").passed


def test_a_duplicated_establishment_id_is_detected() -> None:
    a, b = node("N-1", inspection_id=1), node("N-2", inspection_id=2)
    duplicate = Cluster("EST-00000000001", ("N-2",), "high", None, "x")
    checks = validate_output(
        [a, b], [cluster_of(a), duplicate], [], {"1": "N-1", "2": "N-2"}, T, source_row_count=2
    )
    assert not named(checks, "establishment_id_unique").passed


def test_a_wrongly_anchored_establishment_is_detected() -> None:
    a = node("N-1", inspection_id=500)
    wrong = Cluster("EST-00000000001", ("N-1",), "high", None, "x")
    checks = validate_output([a], [wrong], [], {"500": "N-1"}, T, source_row_count=1)
    assert not named(checks, "anchor_is_a_member").passed


def test_a_cluster_spanning_two_zips_is_detected() -> None:
    a = node("N-1", zip_code="60601", inspection_id=1)
    b = node("N-2", zip_code="60602", inspection_id=2)
    forced = Cluster("EST-00000000001", ("N-1", "N-2"), "high", None, "x")
    checks = validate_output([a, b], [forced], [], {"1": "N-1", "2": "N-2"}, T, source_row_count=2)
    assert not named(checks, "cluster_within_one_zip").passed


def test_a_cluster_with_too_many_addresses_is_detected() -> None:
    members = [
        node(f"N-{i}", address=f"{100 + i} N MAIN ST", inspection_id=i + 1) for i in range(6)
    ]
    forced = Cluster(
        "EST-00000000001", tuple(sorted(n.node_id for n in members)), "high", None, "x"
    )
    checks = validate_output(
        members,
        [forced],
        [],
        {str(i + 1): f"N-{i}" for i in range(6)},
        T,
        source_row_count=6,
    )
    assert not named(checks, "cluster_address_count").passed


def test_a_cluster_with_conflicting_units_is_detected() -> None:
    a = node("N-1", address="123 N MAIN ST STE 100", inspection_id=1)
    b = node("N-2", address="123 N MAIN ST STE 200", inspection_id=2)
    forced = Cluster("EST-00000000001", ("N-1", "N-2"), "high", None, "x")
    checks = validate_output([a, b], [forced], [], {"1": "N-1", "2": "N-2"}, T, source_row_count=2)
    assert not named(checks, "cluster_units_consistent").passed


def test_distributional_checks_never_fail_the_run() -> None:
    """Findings §10: 219 business names at one O'Hare address is legitimate, so
    a check that failed on density would fail on correct data."""
    a = node("N-1", inspection_id=1)
    checks = validate_output([a], [cluster_of(a)], [], {"1": "N-1"}, T, source_row_count=1)
    warnings = [c for c in checks if c.severity != SEVERITY_ERROR]
    assert warnings
    assert all(c.passed for c in warnings)
    assert not has_failures(checks)


def test_ambiguous_pairs_are_counted_for_review() -> None:
    a, b = node("N-1", inspection_id=1), node("N-2", inspection_id=2)
    verdict = PairVerdict("N-1", "N-2", MatchTier.AMBIGUOUS, "A1", PairSignals())
    checks = validate_output(
        [a, b],
        [cluster_of(a), cluster_of(b)],
        [verdict],
        {"1": "N-1", "2": "N-2"},
        T,
        source_row_count=2,
    )
    assert "1 candidate pairs" in named(checks, "ambiguous_pairs").detail


def test_split_establishments_are_reported() -> None:
    a = node("N-1", inspection_id=1)
    checks = validate_output(
        [a],
        [cluster_of(a, confidence="reduced", split_reason="address_split")],
        [],
        {"1": "N-1"},
        T,
        source_row_count=1,
    )
    assert "1 establishments were split" in named(checks, "split_establishments").detail


def test_report_marks_failures_and_notes_distinctly() -> None:
    a = node("N-1", inspection_id=1)
    checks = validate_output([a], [cluster_of(a)], [], {"1": "N-1"}, T, source_row_count=42)
    report = format_report(checks)
    assert "[FAIL] every_inspection_assigned" in report
    assert "[PASS] node_in_exactly_one_cluster" in report
    assert "[note] singleton_rate" in report
