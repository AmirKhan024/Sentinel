"""Turn accepted edges into establishments, and refuse the impossible ones.

Union-find over accepted edges produces connected components. A component is
only accepted as an establishment if it satisfies structural invariants derived
from the data (findings §3, §8, §9). If it does not, the component is rebuilt
under progressively stricter rules rather than being emitted as-is.

The degradation ladder matters more than it looks. Transitive merging is the
classic way entity resolution goes wrong: A merges with B, B with C, and C turns
out to be nothing like A. Every step of the ladder is deterministic and every
split is recorded with the invariant that caused it, so an over-merge degrades
into an explainable split instead of a silent corruption.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence

from sentinel.entity.evidence import haversine_m
from sentinel.entity.models import Cluster, MatchTier, Node, PairVerdict, Thresholds
from sentinel.entity.unionfind import UnionFind

logger = logging.getLogger(__name__)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_REDUCED = "reduced"


def establishment_id_for(members: Sequence[Node]) -> str:
    """Anchor the establishment on its earliest inspection.

    Chosen over a content hash because it is stable under data growth and
    traceable by hand. Inspection ids are assigned monotonically, so appending a
    later snapshot cannot change which row is a cluster's earliest, and the id
    points at exactly one raw row whose name and address a human can read.

    Zero-padded to 11 digits, comfortably above the 7-digit ids in the data.
    """
    if not members:
        raise ValueError("Cannot derive an establishment id for an empty cluster")
    anchor = min(node.min_inspection_id for node in members)
    return f"EST-{anchor:011d}"


def cluster_content_sha256(members: Sequence[Node]) -> str:
    """Hash of the cluster's membership, used to detect change between runs.

    This is deliberately *not* the identifier. It changes whenever membership
    changes, which is precisely what makes it useful for diffing two snapshots:
    same id with a different hash means the cluster absorbed or lost nodes.
    """
    payload = "\n".join(sorted(node.node_id for node in members))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_invariants(members: Sequence[Node], thresholds: Thresholds) -> list[str]:
    """Return the names of every invariant this candidate cluster violates."""
    violations: list[str] = []

    addr_keys = {n.address.key for n in members if n.address.key is not None}
    if len(addr_keys) > thresholds.max_addresses_per_cluster:
        violations.append("too_many_addresses")

    zips = {n.address.zip_key for n in members if n.address.zip_key is not None}
    if len(zips) > thresholds.max_zips_per_cluster:
        violations.append("too_many_zips")

    units = {n.address.unit for n in members if n.address.unit is not None}
    if len(units) > 1:
        violations.append("conflicting_units")

    directionals = {n.address.directional for n in members if n.address.directional is not None}
    if len(directionals) > 1:
        violations.append("conflicting_directionals")

    coords = [(n.geo.lat, n.geo.lon) for n in members if n.geo.lat is not None]
    if len(coords) > 1:
        span = _max_span_m([(lat, lon) for lat, lon in coords if lon is not None])
        if span > thresholds.max_cluster_span_m:
            violations.append("geographic_spread")

    return violations


def _max_span_m(coords: Sequence[tuple[float, float]]) -> float:
    """Largest pairwise distance in a small coordinate set.

    Quadratic, but clusters are tiny; the invariants above cap the number of
    distinct addresses long before this becomes expensive.
    """
    span = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            span = max(span, haversine_m(*coords[i], *coords[j]))
    return span


def _components(node_ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    union = UnionFind(node_ids)
    for left, right in edges:
        if left in union and right in union:
            union.union(left, right)
    return union.components()


def build_clusters(
    nodes: Sequence[Node],
    verdicts: Sequence[PairVerdict],
    thresholds: Thresholds,
) -> tuple[list[Cluster], dict[str, int]]:
    """Group nodes into establishments, splitting any group that fails a check.

    Returns the clusters plus a count of splits by reason.
    """
    by_id: Mapping[str, Node] = {n.node_id: n for n in nodes}

    strong_edges = [
        (v.left_node_id, v.right_node_id) for v in verdicts if v.tier is MatchTier.STRONG
    ]
    merge_edges = strong_edges + [
        (v.left_node_id, v.right_node_id) for v in verdicts if v.tier is MatchTier.PROBABLE
    ]
    probable_ids = {
        node_id
        for v in verdicts
        if v.tier is MatchTier.PROBABLE
        for node_id in (v.left_node_id, v.right_node_id)
    }

    clusters: list[Cluster] = []
    splits: dict[str, int] = {}

    for members_ids in _components([n.node_id for n in nodes], merge_edges).values():
        members = [by_id[i] for i in members_ids]
        violations = check_invariants(members, thresholds)

        if not violations:
            uses_probable = any(i in probable_ids for i in members_ids)
            confidence = CONFIDENCE_MEDIUM if uses_probable else CONFIDENCE_HIGH
            clusters.append(_make_cluster(members, confidence, None))
            continue

        # Rung 1: drop the probable edges and rebuild from strong evidence only.
        reason = f"invariant:{'+'.join(violations)}"
        splits[reason] = splits.get(reason, 0) + 1
        logger.warning("Cluster of %d nodes violates %s; rebuilding", len(members), reason)

        for sub_ids in _components(members_ids, strong_edges).values():
            sub_members = [by_id[i] for i in sub_ids]
            if not check_invariants(sub_members, thresholds):
                clusters.append(
                    _make_cluster(sub_members, CONFIDENCE_REDUCED, "probable_edges_dropped")
                )
                continue

            # Rung 2: split by address key.
            splits["address_split"] = splits.get("address_split", 0) + 1
            for addr_group in _group_by_address(sub_members):
                if not check_invariants(addr_group, thresholds):
                    clusters.append(_make_cluster(addr_group, CONFIDENCE_REDUCED, "address_split"))
                    continue

                # Rung 3: give up and keep every node separate. Splitting is
                # always available as a last resort and never produces a wrong
                # merge.
                splits["atomised"] = splits.get("atomised", 0) + 1
                for node in addr_group:
                    clusters.append(_make_cluster([node], CONFIDENCE_REDUCED, "atomised"))

    clusters.sort(key=lambda c: c.establishment_id)
    return clusters, splits


def _group_by_address(members: Sequence[Node]) -> list[list[Node]]:
    """Partition nodes by address key, keeping the partition order stable."""
    groups: dict[str, list[Node]] = {}
    for node in members:
        groups.setdefault(node.address.key or f"\x00{node.node_id}", []).append(node)
    return [groups[k] for k in sorted(groups)]


def _make_cluster(members: Sequence[Node], confidence: str, split_reason: str | None) -> Cluster:
    return Cluster(
        establishment_id=establishment_id_for(members),
        node_ids=tuple(sorted(n.node_id for n in members)),
        confidence=confidence,
        split_reason=split_reason,
        content_sha256=cluster_content_sha256(members),
    )
