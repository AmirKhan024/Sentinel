"""Frozen contracts for Component 20.

Terminology note
-----------------
This module uses "geographic distance", "geographic proximity", and "geographic work
block" exclusively. The terms "route", "travel time", "driving distance", "optimal
route", and "confirmed schedule" are deliberately absent -- this component has no
road-network, travel-time, staffing, or calendar source. A "geographic work block" is
the operational name for a geographic proximity group: the same connected component,
carrying the additional planning fields (suggested order, rank range, rationale) a
supervisor needs to treat it as a practical unit of field work. It is NOT a workday --
capacity/staffing is a separate constraint this component does not model (NON_GOALS).
"""

from __future__ import annotations

from enum import StrEnum

# --- Versioning -------------------------------------------------------------

#: Bumped whenever the grouping algorithm, distance definition, or output schema
#: changes in a way that makes two runs with different versions incomparable.
#: v2: added work-block planning fields (suggested_order_in_block, organization_mode,
#: highest_sentinel_rank_in_block) and the organization-mode/threshold-preset concepts.
#: The underlying grouping algorithm (Haversine + Union-Find connected components) and
#: every v1 column are unchanged.
GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION = "v2"

# --- Distance threshold -----------------------------------------------------

#: Default geographic proximity threshold used to form connected-component groups.
#:
#: IMPORTANT: 1.5 km is an *initial configurable operational heuristic*, not a
#: validated Chicago inspection travel threshold.
#: It was chosen as a reasonable starting point for a city-block scale (roughly
#: a 15-20 minute walk in an urban grid) and should be validated and adjusted by
#: operational supervisors based on real inspection workflow experience.
#:
#: This value can be overridden at every call site and on the CLI (--threshold-km).
#: No code treats it as scientifically optimal.
DEFAULT_GEO_THRESHOLD_KM: float = 1.5

#: Hard lower bound: a threshold of zero or less would collapse everything into
#: singletons in a meaningless way.
MIN_GEO_THRESHOLD_KM: float = 0.01

#: Hard upper bound: a threshold covering the diameter of greater Chicago
#: (~50 km) would collapse all establishments into one group, also meaningless.
MAX_GEO_THRESHOLD_KM: float = 50.0

#: Named convenience labels over the same configurable ``threshold_km`` -- not a
#: second algorithm, not a validated set of operational distances. "balanced" is
#: exactly ``DEFAULT_GEO_THRESHOLD_KM`` so the preset and the historical default agree.
#: Selectable on the CLI via ``--threshold-preset`` as an alternative to ``--threshold-km``.
GEO_THRESHOLD_PRESETS: dict[str, float] = {
    "tight": 1.0,
    "balanced": DEFAULT_GEO_THRESHOLD_KM,
    "broad": 5.0,
}

# --- Organization mode -------------------------------------------------------


class OrganizationMode(StrEnum):
    """How a suggested work order is produced within a geographic work block.

    RISK_FIRST -- the default. The suggested order within a block is exactly
                  ``policy_rank`` ascending: geography groups and labels establishments,
                  but never reorders them relative to Sentinel's own priority. This is
                  the conservative choice the brief requires as the default: geography
                  must not casually reorder the highest-risk cases.

    GEOGRAPHY_ASSISTED -- a deterministic nearest-neighbour heuristic seeded at the
                  block's highest-priority (lowest policy_rank) establishment, intended
                  to reduce unnecessary spatial back-and-forth. It still surfaces the
                  block's highest Sentinel priority prominently; it accepts a small
                  amount of risk-rank reordering within the block in exchange for
                  geographic coherence. It is a heuristic ordering, not a route: it
                  uses only straight-line geographic distance and does not know street
                  networks, one-way streets, or travel time.
    """

    RISK_FIRST = "risk_first"
    GEOGRAPHY_ASSISTED = "geography_assisted"


#: Human-readable rationale surfaced in the manifest and the API for each mode, so a
#: supervisor never has to infer the ordering rule from the data alone.
SUGGESTED_ORDER_RATIONALE: dict[str, str] = {
    OrganizationMode.RISK_FIRST: (
        "Suggested order preserves Sentinel's priority ordering exactly (policy_rank "
        "ascending). Geography is used only to group and label establishments, never "
        "to reorder them."
    ),
    OrganizationMode.GEOGRAPHY_ASSISTED: (
        "Suggested order balances Sentinel's priority with geographic proximity: it "
        "starts from the block's highest-priority establishment and then visits the "
        "nearest remaining establishment at each step. This is a deterministic "
        "heuristic based on straight-line geographic distance -- not a driving route -- "
        "and it may revisit establishments in a different order than strict risk rank."
    ),
}

# --- Location status --------------------------------------------------------


class LocationStatus(StrEnum):
    """Whether a selected establishment has usable geographic coordinates.

    LOCATION_AVAILABLE -- as_of_latitude and as_of_longitude are both non-null.
                          The establishment participates in the proximity graph.

    LOCATION_UNAVAILABLE -- at least one coordinate is null (has_location=False).
                            The establishment is preserved in the plan, placed in
                            the 'unmapped' group, and never fabricated or removed.
    """

    LOCATION_AVAILABLE = "location_available"
    LOCATION_UNAVAILABLE = "location_unavailable"


# --- Group labels -----------------------------------------------------------

#: The group ID assigned to every selected establishment without usable coordinates.
#: Placed last in any ordered output.
UNMAPPED_GROUP_ID = "unmapped"

#: Human-readable label for the unmapped group.
UNMAPPED_GROUP_LABEL = "Unmapped / Location unavailable"

#: Prefix for deterministically labelled geographic proximity groups.
#: Groups are labelled "Area 1", "Area 2", … ordered by centroid latitude descending
#: (northernmost group first).
GEO_GROUP_LABEL_PREFIX = "Area"

# --- Non-goals documentation ------------------------------------------------

#: Explicit list of capabilities NOT implemented in Component 20.
#: Referenced in docstrings and tests to make the boundary unambiguous.
NON_GOALS: tuple[str, ...] = (
    "route_optimization",
    "travel_time_estimation",
    "driving_directions",
    "inspector_assignment",
    "inspector_start_locations",
    "working_hour_scheduling",
    "inspection_duration_modeling",
    "risk_re_ranking_by_geography",
    "capacity_selection",
    "fake_capacity_claims",
)


class GeographicOrganizationDefinitionError(ValueError):
    """Raised when the geographic organization configuration is invalid."""


def validate_threshold(threshold_km: float) -> None:
    """Raise if the requested threshold is outside the acceptable range."""
    if not (MIN_GEO_THRESHOLD_KM <= threshold_km <= MAX_GEO_THRESHOLD_KM):
        raise GeographicOrganizationDefinitionError(
            f"threshold_km={threshold_km} is outside the acceptable range "
            f"[{MIN_GEO_THRESHOLD_KM}, {MAX_GEO_THRESHOLD_KM}]. "
            "This is a configurable operational heuristic -- adjust it to match "
            "your operational context, but keep it within plausible geographic bounds."
        )


def resolve_threshold_km(*, threshold_km: float | None, threshold_preset: str | None) -> float:
    """Resolve an explicit ``threshold_km`` or a named preset to one km value.

    Exactly one of the two may be given; passing both is refused rather than silently
    preferring one, because a caller who set both almost certainly means only one of
    them. Passing neither returns ``DEFAULT_GEO_THRESHOLD_KM``.
    """
    if threshold_km is not None and threshold_preset is not None:
        raise GeographicOrganizationDefinitionError(
            "threshold_km and threshold_preset were both given -- pass at most one. "
            f"threshold_preset={threshold_preset!r} would resolve to "
            f"{GEO_THRESHOLD_PRESETS.get(threshold_preset)} km, which may not be what "
            f"threshold_km={threshold_km} intended."
        )
    if threshold_preset is not None:
        if threshold_preset not in GEO_THRESHOLD_PRESETS:
            raise GeographicOrganizationDefinitionError(
                f"threshold_preset={threshold_preset!r} is not one of "
                f"{sorted(GEO_THRESHOLD_PRESETS)}"
            )
        return GEO_THRESHOLD_PRESETS[threshold_preset]
    if threshold_km is not None:
        return threshold_km
    return DEFAULT_GEO_THRESHOLD_KM


__all__ = [
    "DEFAULT_GEO_THRESHOLD_KM",
    "GEO_GROUP_LABEL_PREFIX",
    "GEO_THRESHOLD_PRESETS",
    "GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION",
    "MAX_GEO_THRESHOLD_KM",
    "MIN_GEO_THRESHOLD_KM",
    "NON_GOALS",
    "SUGGESTED_ORDER_RATIONALE",
    "UNMAPPED_GROUP_ID",
    "UNMAPPED_GROUP_LABEL",
    "GeographicOrganizationDefinitionError",
    "LocationStatus",
    "OrganizationMode",
    "resolve_threshold_km",
    "validate_threshold",
]
