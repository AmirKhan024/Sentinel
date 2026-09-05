"""Geographic distance calculation for Component 20.

Single responsibility: Haversine (great-circle) geographic distance in kilometres.

Terminology
-----------
This module computes *geographic distance* (straight-line, great-circle).
It does NOT compute:
  - driving distance
  - travel time
  - road-network distance
  - route length

Those would require a road-network or travel-time data source that does not exist
in this repository.

Implementation
--------------
Reuses ``entity.evidence.haversine_m`` -- the only spatial distance function in
the repository -- rather than re-implementing the Haversine formula.
This ensures the same calculation is used consistently throughout Sentinel and
cannot silently diverge between the entity-resolution cluster check and the
geographic planning layer.
"""

from __future__ import annotations

from sentinel.entity.evidence import haversine_m

# Earth radius in metres used by haversine_m, documented here for unit clarity.
_M_PER_KM: float = 1_000.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle geographic distance in kilometres between two coordinate pairs.

    Parameters
    ----------
    lat1, lon1 : float
        WGS-84 latitude/longitude of the first point, in decimal degrees.
    lat2, lon2 : float
        WGS-84 latitude/longitude of the second point, in decimal degrees.

    Returns
    -------
    float
        Great-circle geographic distance in kilometres.
        This is NOT driving distance, travel time, or road-network distance.

    Notes
    -----
    Delegates to ``entity.evidence.haversine_m`` (metres), which uses the
    standard Haversine formula with Earth radius 6,371,000 m.
    """
    return haversine_m(lat1, lon1, lat2, lon2) / _M_PER_KM


__all__ = ["haversine_km"]
