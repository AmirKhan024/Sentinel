"""Per-group geographic metrics for Component 20.

All metrics are geographic (Haversine / great-circle).
None of them represent driving distance, travel time, or route length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from sentinel.geographic_organization.definitions import UNMAPPED_GROUP_ID
from sentinel.geographic_organization.distance import haversine_km


@dataclass
class GeographicGroupMetrics:
    """Per-group geographic summary metrics.

    All distance values are Haversine (great-circle) in kilometres.
    These are NOT driving distances, travel times, or route lengths.

    max_within_group_distance_km is particularly important for geographic
    proximity groups (connected-component style): two establishments can be
    in the same group while being farther apart than the grouping threshold,
    because the threshold governs pairwise edges, not the group's total span.
    """

    group_id: str
    group_label: str
    size: int
    member_establishment_ids: list[str] = field(default_factory=list)

    # Centroid — arithmetic mean of member coordinates.
    # Valid only for groups with LOCATION_AVAILABLE members (i.e., not "unmapped").
    centroid_lat: float | None = None
    centroid_lon: float | None = None

    # Geographic spread within the group (Haversine km, NOT driving distance).
    max_within_group_distance_km: float | None = None
    avg_within_group_distance_km: float | None = None


def _max_pairwise_km(coords: list[tuple[float, float]]) -> float:
    """Largest pairwise geographic distance in a set of (lat, lon) pairs.

    O(N²) but groups are small (daily selected set is at most ~50 rows).
    """
    span = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            span = max(span, haversine_km(*coords[i], *coords[j]))
    return span


def _avg_pairwise_km(coords: list[tuple[float, float]]) -> float:
    """Average pairwise geographic distance in a set of (lat, lon) pairs."""
    if len(coords) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            total += haversine_km(*coords[i], *coords[j])
            count += 1
    return total / count if count > 0 else 0.0


def compute_group_metrics(grouped_frame: pl.DataFrame) -> list[GeographicGroupMetrics]:
    """Compute geographic metrics for every group in a grouped selected frame.

    Parameters
    ----------
    grouped_frame : pl.DataFrame
        Output of ``grouping.assign_geographic_groups``.
        Required columns: ``establishment_id``, ``geographic_group_id``,
        ``geographic_group_label``, ``as_of_latitude``, ``as_of_longitude``,
        ``location_status``.

    Returns
    -------
    list[GeographicGroupMetrics]
        One entry per group, with mapped groups ordered northernmost first
        and the "unmapped" group last.
    """
    results: list[GeographicGroupMetrics] = []
    unmapped_entry: GeographicGroupMetrics | None = None

    # Collect unique group IDs in a deterministic order.
    all_group_ids = sorted(grouped_frame["geographic_group_id"].unique().to_list())

    for gid in all_group_ids:
        group_rows = grouped_frame.filter(pl.col("geographic_group_id") == gid)
        group_label = group_rows["geographic_group_label"][0]
        member_ids = sorted(group_rows["establishment_id"].to_list())
        size = len(member_ids)

        if gid == UNMAPPED_GROUP_ID:
            # Unmapped group: no coordinates, no distance metrics.
            unmapped_entry = GeographicGroupMetrics(
                group_id=gid,
                group_label=group_label,
                size=size,
                member_establishment_ids=member_ids,
                centroid_lat=None,
                centroid_lon=None,
                max_within_group_distance_km=None,
                avg_within_group_distance_km=None,
            )
            continue

        # Collect coordinates for this group (all should be available).
        lats = group_rows["as_of_latitude"].to_list()
        lons = group_rows["as_of_longitude"].to_list()
        coords = [
            (float(lat), float(lon))
            for lat, lon in zip(lats, lons, strict=True)
            if lat is not None and lon is not None
        ]

        if not coords:
            # Defensive: mapped group with no usable coordinates (should not happen).
            centroid_lat = centroid_lon = None
            max_dist = avg_dist = None
        else:
            centroid_lat = sum(c[0] for c in coords) / len(coords)
            centroid_lon = sum(c[1] for c in coords) / len(coords)
            if len(coords) == 1:
                # Single establishment: spread is zero by definition.
                max_dist = 0.0
                avg_dist = 0.0
            else:
                max_dist = _max_pairwise_km(coords)
                avg_dist = _avg_pairwise_km(coords)

        results.append(
            GeographicGroupMetrics(
                group_id=gid,
                group_label=group_label,
                size=size,
                member_establishment_ids=member_ids,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                max_within_group_distance_km=max_dist,
                avg_within_group_distance_km=avg_dist,
            )
        )

    # Sort mapped groups northernmost first (by centroid_lat descending).
    results.sort(key=lambda m: -(m.centroid_lat if m.centroid_lat is not None else -math.inf))

    # Append unmapped last.
    if unmapped_entry is not None:
        results.append(unmapped_entry)

    return results


__all__ = ["GeographicGroupMetrics", "compute_group_metrics"]
