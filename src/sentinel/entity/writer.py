"""Build and write the three Component 2 output tables.

Unlike the raw layer, this is an interim layer and may carry real types. The
all-``Utf8`` rule from ADR 0002 is a property of *raw* storage, not of every
file in the project. Identifiers are the exception and stay ``Utf8``:
``inspection_id``, ``establishment_id`` and ``license_key`` are labels rather
than quantities, and casting them would invite exactly the silent-null failure
that ADR 0002 exists to prevent.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from sentinel.entity.models import Cluster, MatchTier, Node, PairVerdict

logger = logging.getLogger(__name__)

# Column order is part of the data contract; changing it is a contract change.
ASSIGNMENTS_SCHEMA: dict[str, pl.DataType] = {
    "inspection_id": pl.Utf8(),
    "establishment_id": pl.Utf8(),
    "node_id": pl.Utf8(),
    "resolution_tier": pl.Utf8(),
    "has_ambiguous_link": pl.Boolean(),
    "license_key": pl.Utf8(),
    "name_key": pl.Utf8(),
    "addr_key": pl.Utf8(),
    "unit": pl.Utf8(),
    "zip_key": pl.Utf8(),
    "geo_usable": pl.Boolean(),
}

ESTABLISHMENTS_SCHEMA: dict[str, pl.DataType] = {
    "establishment_id": pl.Utf8(),
    "anchor_inspection_id": pl.Utf8(),
    "cluster_content_sha256": pl.Utf8(),
    "resolution_confidence": pl.Utf8(),
    "split_reason": pl.Utf8(),
    "n_nodes": pl.Int64(),
    "n_inspections": pl.Int64(),
    "n_licenses": pl.Int64(),
    "n_addresses": pl.Int64(),
    "n_names": pl.Int64(),
    "canonical_name": pl.Utf8(),
    "canonical_address": pl.Utf8(),
    "canonical_zip": pl.Utf8(),
}

EDGES_SCHEMA: dict[str, pl.DataType] = {
    "left_node_id": pl.Utf8(),
    "right_node_id": pl.Utf8(),
    "left_establishment_id": pl.Utf8(),
    "right_establishment_id": pl.Utf8(),
    "tier": pl.Utf8(),
    "rule_id": pl.Utf8(),
    "same_license": pl.Boolean(),
    "same_addr_key": pl.Boolean(),
    "same_coord": pl.Boolean(),
    "near_addr": pl.Boolean(),
    "same_zip": pl.Boolean(),
    "name_exact": pl.Boolean(),
    "name_containment": pl.Boolean(),
    "digit_conflict": pl.Boolean(),
    "unit_conflict": pl.Boolean(),
    "dir_conflict": pl.Boolean(),
    "aka_conflict": pl.Boolean(),
    "facility_agree": pl.Boolean(),
    "left_name": pl.Utf8(),
    "right_name": pl.Utf8(),
    "left_address": pl.Utf8(),
    "right_address": pl.Utf8(),
}


def _empty(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def build_assignments(
    nodes: Sequence[Node],
    clusters: Sequence[Cluster],
    assignment: Mapping[str, str],
    verdicts: Sequence[PairVerdict],
) -> pl.DataFrame:
    """One row per inspection.

    Deliberately carries no dates, counts or outcomes. That is anti-leakage
    hygiene (findings §14): a downstream join against this table cannot pull a
    full-history aggregate into a training row, because there is none to pull.
    """
    by_node = {n.node_id: n for n in nodes}
    node_to_cluster: dict[str, Cluster] = {}
    for cluster in clusters:
        for node_id in cluster.node_ids:
            node_to_cluster[node_id] = cluster

    ambiguous_nodes = {
        node_id
        for v in verdicts
        if v.tier is MatchTier.AMBIGUOUS
        for node_id in (v.left_node_id, v.right_node_id)
    }

    rows = []
    for inspection_id, node_id in assignment.items():
        node = by_node[node_id]
        cluster = node_to_cluster[node_id]
        if cluster.split_reason is not None:
            tier = "reduced"
        elif len(cluster.node_ids) == 1:
            tier = "singleton"
        else:
            tier = cluster.confidence
        rows.append(
            {
                "inspection_id": inspection_id,
                "establishment_id": cluster.establishment_id,
                "node_id": node_id,
                "resolution_tier": tier,
                "has_ambiguous_link": node_id in ambiguous_nodes,
                "license_key": node.license_key,
                "name_key": node.name.key,
                "addr_key": node.address.key,
                "unit": node.address.unit,
                "zip_key": node.address.zip_key,
                "geo_usable": node.geo.usable,
            }
        )

    if not rows:
        return _empty(ASSIGNMENTS_SCHEMA)
    return pl.DataFrame(rows, schema=dict(ASSIGNMENTS_SCHEMA)).sort("inspection_id")


def build_establishments(nodes: Sequence[Node], clusters: Sequence[Cluster]) -> pl.DataFrame:
    """One row per establishment.

    ``canonical_name`` and ``canonical_address`` are taken from the anchor node
    and exist for human readability only. The data contract marks them as
    display fields: they are not join keys, and they are not model features.
    """
    by_node = {n.node_id: n for n in nodes}

    rows = []
    for cluster in clusters:
        members = [by_node[i] for i in cluster.node_ids]
        anchor = min(members, key=lambda n: n.min_inspection_id)
        rows.append(
            {
                "establishment_id": cluster.establishment_id,
                "anchor_inspection_id": str(anchor.min_inspection_id),
                "cluster_content_sha256": cluster.content_sha256,
                "resolution_confidence": cluster.confidence,
                "split_reason": cluster.split_reason,
                "n_nodes": len(members),
                "n_inspections": sum(len(m.inspection_ids) for m in members),
                "n_licenses": len({m.license_key for m in members if m.license_key}),
                "n_addresses": len({m.address.key for m in members if m.address.key}),
                "n_names": len({m.name.key for m in members if m.name.key}),
                "canonical_name": anchor.raw_name,
                "canonical_address": anchor.raw_address,
                "canonical_zip": anchor.address.zip_key,
            }
        )

    if not rows:
        return _empty(ESTABLISHMENTS_SCHEMA)
    return pl.DataFrame(rows, schema=dict(ESTABLISHMENTS_SCHEMA)).sort("establishment_id")


def build_edges(
    nodes: Sequence[Node],
    clusters: Sequence[Cluster],
    verdicts: Sequence[PairVerdict],
) -> pl.DataFrame:
    """The audit table: every pair whose outcome is worth being able to explain.

    Kept to pairs that either merged, were declined as ambiguous, or were
    vetoed despite sharing a licence or an address. Plain no-matches between
    unrelated neighbours are omitted; there are hundreds of thousands of them
    and they carry no information a reader would ever ask about.
    """
    by_node = {n.node_id: n for n in nodes}
    node_to_est = {
        node_id: cluster.establishment_id for cluster in clusters for node_id in cluster.node_ids
    }

    rows = []
    for verdict in verdicts:
        interesting = verdict.tier is not MatchTier.NO_MATCH or (
            verdict.rule_id.startswith("V")
            and (verdict.signals.same_license or verdict.signals.addr_equivalent)
        )
        if not interesting:
            continue
        left, right = by_node[verdict.left_node_id], by_node[verdict.right_node_id]
        signals = verdict.signals
        rows.append(
            {
                "left_node_id": verdict.left_node_id,
                "right_node_id": verdict.right_node_id,
                "left_establishment_id": node_to_est.get(verdict.left_node_id),
                "right_establishment_id": node_to_est.get(verdict.right_node_id),
                "tier": verdict.tier.value,
                "rule_id": verdict.rule_id,
                "same_license": signals.same_license,
                "same_addr_key": signals.same_addr_key,
                "same_coord": signals.same_coord,
                "near_addr": signals.near_addr,
                "same_zip": signals.same_zip,
                "name_exact": signals.name_exact,
                "name_containment": signals.name_containment,
                "digit_conflict": signals.digit_conflict,
                "unit_conflict": signals.unit_conflict,
                "dir_conflict": signals.dir_conflict,
                "aka_conflict": signals.aka_conflict,
                "facility_agree": signals.facility_agree,
                "left_name": left.raw_name,
                "right_name": right.raw_name,
                "left_address": left.raw_address,
                "right_address": right.raw_address,
            }
        )

    if not rows:
        return _empty(EDGES_SCHEMA)
    return pl.DataFrame(rows, schema=dict(EDGES_SCHEMA)).sort(["left_node_id", "right_node_id"])


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write one output table as zstd Parquet, matching Component 1's settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    logger.info("Wrote %s (%d rows)", path, frame.height)
    return path


def schema_of(frame: pl.DataFrame) -> dict[str, str]:
    """Column name to dtype string, for recording in the manifest."""
    return {name: str(dtype) for name, dtype in frame.schema.items()}
