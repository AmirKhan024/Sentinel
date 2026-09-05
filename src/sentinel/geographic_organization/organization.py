"""Geographic work blocks and suggested work order for Component 20.

Separate from ``grouping.py`` deliberately: ``grouping.py`` is the pure geographic
proximity algorithm (Haversine + Union-Find), untouched by this module and testable
alone. This module turns a computed group into an operational planning unit -- a
"geographic work block" -- and, within it, a suggested work order.

Neither function here ever reads or writes ``calibrated_score``, ``base_score``,
``rank``, ``policy_rank``, ``selection_reason``, or ``selection_mechanism`` -- it only
reads ``policy_rank`` to order and label, and never mutates it. See
``validate.IMMUTABLE_FIELDS``.

A geographic work block is NOT a workday. Capacity/staffing is a separate constraint
this component does not model (``definitions.NON_GOALS``). Whether the establishments
in one block fit into a single inspector's day is a decision left to a supervisor and
to a future capacity-aware layer, not asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from sentinel.geographic_organization.definitions import (
    SUGGESTED_ORDER_RATIONALE,
    UNMAPPED_GROUP_ID,
    OrganizationMode,
)
from sentinel.geographic_organization.distance import haversine_km
from sentinel.geographic_organization.metrics import GeographicGroupMetrics


@dataclass
class WorkBlock:
    """One geographic work block: a proximity group carrying planning fields.

    ``highest_sentinel_rank`` and ``rank_range`` are read directly from Component 19's
    ``policy_rank`` column and never recomputed or adjusted -- this is how the block
    stays honest about Sentinel's own priority even when geography groups establishments
    of very different risk together.
    """

    block_id: str
    label: str
    establishment_ids: list[str]
    size: int
    centroid_lat: float | None
    centroid_lon: float | None
    avg_within_block_distance_km: float | None
    max_within_block_distance_km: float | None
    highest_sentinel_rank: int | None
    rank_range: tuple[int, int] | None
    member_ranks: list[int] = field(default_factory=list)
    has_unmapped_member: bool = False
    rationale: str = ""


def build_work_blocks(
    grouped_frame: pl.DataFrame,
    group_metrics_list: list[GeographicGroupMetrics],
) -> list[WorkBlock]:
    """Attach planning fields to each computed geographic group.

    ``grouped_frame`` must carry ``geographic_group_id`` and ``policy_rank`` (Component
    19's rank, read-only). One ``WorkBlock`` per entry in ``group_metrics_list``, in the
    same order (mapped groups northernmost first, then "unmapped" last if present).
    """
    blocks: list[WorkBlock] = []
    for gm in group_metrics_list:
        is_unmapped = gm.group_id == UNMAPPED_GROUP_ID
        member_rows = grouped_frame.filter(pl.col("geographic_group_id") == gm.group_id)
        ranks = sorted(int(r) for r in member_rows["policy_rank"].drop_nulls().to_list())
        highest_rank = ranks[0] if ranks else None
        rank_range = (ranks[0], ranks[-1]) if ranks else None
        rationale = (
            "Coordinates are unavailable for this establishment, so Sentinel cannot "
            "place it into a geographic work block."
            if is_unmapped
            else (
                f"These {gm.size} selected establishment(s) are within the configured "
                "geographic proximity threshold and can be considered as one field-work "
                "area. Proximity is based on straight-line geographic distance; Sentinel "
                "does not currently estimate driving time or traffic."
                if gm.size > 1
                else "A single selected establishment with no other selected "
                "establishment within the configured geographic proximity threshold."
            )
        )
        blocks.append(
            WorkBlock(
                block_id=gm.group_id,
                label=gm.group_label,
                establishment_ids=gm.member_establishment_ids,
                size=gm.size,
                centroid_lat=gm.centroid_lat,
                centroid_lon=gm.centroid_lon,
                avg_within_block_distance_km=gm.avg_within_group_distance_km,
                max_within_block_distance_km=gm.max_within_group_distance_km,
                highest_sentinel_rank=highest_rank,
                rank_range=rank_range,
                member_ranks=ranks,
                has_unmapped_member=is_unmapped,
                rationale=rationale,
            )
        )
    return blocks


def suggested_work_order(
    block_members: pl.DataFrame, *, mode: OrganizationMode
) -> list[tuple[int, str, int | None]]:
    """Suggested visiting order within one work block.

    ``block_members`` must carry ``target_inspection_id``, ``policy_rank``,
    ``as_of_latitude``, ``as_of_longitude`` for exactly one block's rows.

    Returns a list of ``(order_index, target_inspection_id, policy_rank)`` starting at 1,
    covering every input row exactly once (a permutation, never a subset).

    RISK_FIRST: ``policy_rank`` ascending (nulls last), tie-broken by
    ``target_inspection_id`` for determinism. Geography changes nothing.

    GEOGRAPHY_ASSISTED: a deterministic greedy nearest-neighbour chain seeded at the
    lowest ``policy_rank`` member. This is a heuristic ordering over straight-line
    geographic distance -- not a driving route -- and may reorder relative to strict
    risk rank to reduce spatial back-and-forth within the block.
    """
    rows = list(
        block_members.select(
            "target_inspection_id", "policy_rank", "as_of_latitude", "as_of_longitude"
        ).iter_rows(named=True)
    )

    def _rank_key(r: dict[str, Any]) -> tuple[int, str]:
        rank = r["policy_rank"]
        return (rank if rank is not None else 2**62, r["target_inspection_id"])

    if mode == OrganizationMode.RISK_FIRST or len(rows) <= 1:
        ordered = sorted(rows, key=_rank_key)
        return [(i + 1, r["target_inspection_id"], r["policy_rank"]) for i, r in enumerate(ordered)]

    # GEOGRAPHY_ASSISTED: greedy nearest-neighbour, seeded at the highest priority member.
    remaining = sorted(rows, key=_rank_key)
    current = remaining.pop(0)
    ordered = [current]
    while remaining:
        cur_lat, cur_lon = current["as_of_latitude"], current["as_of_longitude"]

        def _dist_key(
            r: dict[str, Any], cur_lat: float | None = cur_lat, cur_lon: float | None = cur_lon
        ) -> tuple[float, int, str]:
            if cur_lat is None or cur_lon is None or r["as_of_latitude"] is None:
                dist = float("inf")
            else:
                dist = haversine_km(cur_lat, cur_lon, r["as_of_latitude"], r["as_of_longitude"])
            rank = r["policy_rank"]
            return (dist, rank if rank is not None else 2**62, r["target_inspection_id"])

        remaining.sort(key=_dist_key)
        current = remaining.pop(0)
        ordered.append(current)

    return [(i + 1, r["target_inspection_id"], r["policy_rank"]) for i, r in enumerate(ordered)]


def organization_mode_rationale(mode: OrganizationMode) -> str:
    return SUGGESTED_ORDER_RATIONALE[mode]


__all__ = [
    "WorkBlock",
    "build_work_blocks",
    "organization_mode_rationale",
    "suggested_work_order",
]
