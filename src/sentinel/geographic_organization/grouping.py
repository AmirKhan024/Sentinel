"""Geographic proximity grouping for Component 20.

Algorithm: distance-threshold connected components via Union-Find.

Two selected establishments are placed in the same geographic proximity group
if their Haversine (great-circle) geographic distance is at most `threshold_km`.
Connected components under this pairwise threshold define the groups.

IMPORTANT — what "geographic proximity group" means
----------------------------------------------------
This algorithm forms *connected-component* groups, NOT radius-bounded clusters.

Example: A --1 km-- B --1 km-- C with threshold 1.5 km
  → A, B, C are one group even though A and C are 2 km apart.

This is by design: it chains geographically adjacent establishments.
The ``max_within_group_distance_km`` metric (computed by metrics.py) quantifies
the actual spread of each group, which is the operationally useful number.

The documentation and UI must say "geographic proximity group" rather than
"every member is within threshold_km of every other member."

Why this algorithm
------------------
- Deterministic: identical input → identical output (no random initialization)
- No preset cluster count (unlike k-means)
- No hyperparameter search on operational data (unlike DBSCAN min_samples tuning)
- No additional dependencies: reuses entity.unionfind.UnionFind already in repo
- Interpretable: "these establishments are geographically adjacent"
- Robust to any selection size: 1 to N establishments
- Natural handling of unmapped establishments: excluded from the proximity graph,
  placed in a dedicated "unmapped" group

Determinism guarantee
---------------------
Group IDs are assigned by sorting mapped groups by centroid latitude descending
(northernmost first), then numbering them "Area 1", "Area 2", ...
The unmapped group always appears last.
This makes the labelling stable: the same input always produces the same labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from sentinel.entity.unionfind import UnionFind
from sentinel.geographic_organization.definitions import (
    DEFAULT_GEO_THRESHOLD_KM,
    GEO_GROUP_LABEL_PREFIX,
    UNMAPPED_GROUP_ID,
    UNMAPPED_GROUP_LABEL,
    LocationStatus,
)
from sentinel.geographic_organization.distance import haversine_km

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CoordRecord:
    """Lightweight coordinate record for an establishment in the proximity graph."""

    establishment_id: str
    lat: float
    lon: float


def _build_proximity_components(
    coords: list[_CoordRecord],
    threshold_km: float,
) -> dict[str, list[str]]:
    """Union-Find proximity grouping over a list of coordinate records.

    Returns a dict mapping each component's minimum establishment_id
    to the sorted list of establishment_ids in that component.

    O(N²) in the number of mapped establishments -- acceptable for the
    daily selected set (tens, not thousands of rows).
    """
    uf = UnionFind(c.establishment_id for c in coords)
    # Iterate all pairs; union those within the geographic distance threshold.
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            a, b = coords[i], coords[j]
            dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
            if dist <= threshold_km:
                uf.union(a.establishment_id, b.establishment_id)
    return uf.components()


def _stable_group_labels(
    components: dict[str, list[str]],
    coord_lookup: dict[str, _CoordRecord],
) -> list[tuple[str, str, list[str]]]:
    """Assign deterministic group IDs and labels ordered by centroid latitude descending.

    Returns a list of (group_id, group_label, [establishment_ids]) tuples,
    sorted northernmost first.

    Centroid latitude is the arithmetic mean of member latitudes -- correct for
    the small geographic extent of a Chicago inspection plan.
    """
    # Build (centroid_lat, component_members) pairs for stable sorting.
    sortable = []
    for members in components.values():
        lats = [coord_lookup[eid].lat for eid in members if eid in coord_lookup]
        centroid_lat = sum(lats) / len(lats) if lats else 0.0
        sortable.append((centroid_lat, sorted(members)))

    # Sort descending by centroid latitude (northernmost group first).
    # Tie-break by the minimum establishment_id in the group for full determinism.
    sortable.sort(key=lambda x: (-x[0], x[1][0]))

    result = []
    for rank, (_, members) in enumerate(sortable, start=1):
        group_id = f"{GEO_GROUP_LABEL_PREFIX.lower()}_{rank}"
        group_label = f"{GEO_GROUP_LABEL_PREFIX} {rank}"
        result.append((group_id, group_label, members))
    return result


def assign_geographic_groups(
    selected_frame: pl.DataFrame,
    *,
    threshold_km: float = DEFAULT_GEO_THRESHOLD_KM,
) -> pl.DataFrame:
    """Assign geographic proximity groups to the selected establishment frame.

    Parameters
    ----------
    selected_frame : pl.DataFrame
        Must contain only ``is_selected == True`` rows from Component 19.
        Required columns: ``establishment_id``, ``as_of_latitude``,
        ``as_of_longitude``, ``has_location``.
    threshold_km : float
        Geographic proximity threshold in kilometres.
        Default is ``DEFAULT_GEO_THRESHOLD_KM`` (1.5 km) -- a configurable
        operational heuristic, not a validated optimal distance.

    Returns
    -------
    pl.DataFrame
        The input frame with two new columns added:
        - ``location_status``: "location_available" or "location_unavailable"
        - ``geographic_group_id``: e.g. "area_1", "area_2", "unmapped"
        - ``geographic_group_label``: e.g. "Area 1", "Area 2",
          "Unmapped / Location unavailable"

    Notes
    -----
    Groups are *geographic proximity connected components*, not radius-bounded
    clusters. Two establishments that are each within threshold_km of a common
    neighbour will be in the same group even if they are more than threshold_km
    apart from each other. Use ``max_within_group_distance_km`` (computed by
    metrics.py) to understand the actual geographic spread of each group.

    The selected-ID set is never altered: this function only adds columns.
    """
    required = ("establishment_id", "as_of_latitude", "as_of_longitude")
    missing = [c for c in required if c not in selected_frame.columns]
    if missing:
        raise ValueError(
            f"assign_geographic_groups is missing required column(s): {', '.join(missing)}"
        )

    # Guard: must receive only selected rows.
    if "is_selected" in selected_frame.columns:
        non_selected = int(selected_frame.filter(~pl.col("is_selected")).height)
        if non_selected > 0:
            raise ValueError(
                f"assign_geographic_groups received {non_selected} non-selected rows. "
                "Pass only is_selected==True rows to avoid accidentally grouping "
                "non-selected establishments."
            )

    n_total = selected_frame.height
    n_unique = selected_frame["establishment_id"].n_unique()
    if n_unique != n_total:
        raise ValueError(
            f"assign_geographic_groups received {n_total - n_unique} duplicate "
            "establishment_id value(s). A canonical establishment must appear at "
            "most once in a selected set, or it could be placed in two groups at once."
        )

    # Partition into mapped and unmapped.
    mapped_mask = (
        selected_frame["as_of_latitude"].is_not_null()
        & selected_frame["as_of_longitude"].is_not_null()
    )

    establishment_ids = selected_frame["establishment_id"].to_list()
    lats = selected_frame["as_of_latitude"].to_list()
    lons = selected_frame["as_of_longitude"].to_list()
    mapped_mask_list = mapped_mask.to_list()

    # Build coordinate records for mapped establishments only.
    coords: list[_CoordRecord] = []
    coord_lookup: dict[str, _CoordRecord] = {}
    for eid, lat, lon, is_mapped in zip(
        establishment_ids, lats, lons, mapped_mask_list, strict=True
    ):
        if is_mapped:
            rec = _CoordRecord(establishment_id=eid, lat=float(lat), lon=float(lon))
            coords.append(rec)
            coord_lookup[eid] = rec

    n_mapped = len(coords)
    n_unmapped = n_total - n_mapped

    logger.info(
        "Geographic grouping: %d selected, %d mapped, %d unmapped, threshold=%.2f km",
        n_total,
        n_mapped,
        n_unmapped,
        threshold_km,
    )

    # Run Union-Find proximity grouping on the mapped set.
    components: dict[str, list[str]] = {}
    if coords:
        components = _build_proximity_components(coords, threshold_km)

    # Assign stable, deterministic group IDs/labels.
    labelled = _stable_group_labels(components, coord_lookup)

    # Build a lookup: establishment_id -> (group_id, group_label).
    eid_to_group: dict[str, tuple[str, str]] = {}
    for group_id, group_label, members in labelled:
        for eid in members:
            eid_to_group[eid] = (group_id, group_label)

    # Assign columns row-by-row in the original frame order.
    group_ids: list[str] = []
    group_labels: list[str] = []
    location_statuses: list[str] = []
    for eid, is_mapped in zip(establishment_ids, mapped_mask_list, strict=True):
        if is_mapped:
            gid, glabel = eid_to_group[eid]
            group_ids.append(gid)
            group_labels.append(glabel)
            location_statuses.append(LocationStatus.LOCATION_AVAILABLE.value)
        else:
            group_ids.append(UNMAPPED_GROUP_ID)
            group_labels.append(UNMAPPED_GROUP_LABEL)
            location_statuses.append(LocationStatus.LOCATION_UNAVAILABLE.value)

    return selected_frame.with_columns(
        pl.Series("location_status", location_statuses, dtype=pl.Utf8),
        pl.Series("geographic_group_id", group_ids, dtype=pl.Utf8),
        pl.Series("geographic_group_label", group_labels, dtype=pl.Utf8),
    )


__all__ = ["assign_geographic_groups"]
