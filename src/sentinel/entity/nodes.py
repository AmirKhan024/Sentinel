"""Collapse raw inspection rows into distinct identity signatures (nodes).

Matching runs on nodes, not inspections. A node is one distinct way the city
recorded a place: a tuple of licence, names, address, unit, zip and coordinate.
Many inspections share a signature, so this shrinks the comparison space and,
more importantly, makes the audit table readable — an edge is a statement about
two ways of writing a place down, not about two individual visits.

Only identity fields are read here. ``results``, ``violations`` and ``risk`` are
never touched, and following findings §11 neither is ``inspection_date`` beyond
recording it as a display attribute. That is what keeps identity reconstruction
separate from as-of feature availability (findings §14).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from sentinel.entity.models import Geo, Node, NormalizedAddress, NormalizedName
from sentinel.entity.normalize import (
    normalize_address,
    normalize_facility_type,
    normalize_geo,
    normalize_license,
    normalize_name,
)

logger = logging.getLogger(__name__)

# The raw columns entity resolution is permitted to read. Listing them here
# rather than reading the frame wholesale makes the leakage boundary explicit
# and greppable: an outcome column cannot reach the matcher by accident.
IDENTITY_COLUMNS = (
    "inspection_id",
    "dba_name",
    "aka_name",
    "license_",
    "facility_type",
    "address",
    "zip",
    "latitude",
    "longitude",
)


@dataclass
class _Accumulator:
    """Mutable per-node scratch space used while scanning the frame."""

    license_key: str | None
    name: NormalizedName
    aka: NormalizedName
    address: NormalizedAddress
    geo: Geo
    facility_type_key: str | None
    inspection_ids: list[str]
    raw_name: str
    raw_address: str
    raw_zip: str | None


def _signature(
    license_key: str | None,
    name_key: str | None,
    aka_key: str | None,
    addr_key: str | None,
    unit: str | None,
    zip_key: str | None,
    geo_key: str | None,
) -> str:
    """Build the canonical string a node's identity hash is taken over."""
    parts = [license_key, name_key, aka_key, addr_key, unit, zip_key, geo_key]
    return "\x1f".join("" if p is None else p for p in parts)


def node_id_for(signature: str) -> str:
    """Deterministic node identifier: a truncated hash of the signature.

    Content-derived rather than sequential, so a node keeps its identifier
    regardless of the order rows were read in. Truncated to 16 hex characters,
    which is far more than enough to separate the tens of thousands of nodes a
    snapshot produces.
    """
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return f"N-{digest[:16]}"


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def build_nodes(frame: pl.DataFrame) -> tuple[list[Node], dict[str, str]]:
    """Turn a raw frame into nodes plus an inspection_id -> node_id mapping.

    Returns nodes sorted by ``node_id`` so downstream iteration order is fixed.
    """
    missing = [c for c in IDENTITY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Raw frame is missing required identity columns: {', '.join(missing)}")

    # Only the identity columns are pulled out of the frame at all.
    subset = frame.select(IDENTITY_COLUMNS)

    accumulated: dict[str, _Accumulator] = {}
    assignment: dict[str, str] = {}

    for row in subset.iter_rows(named=True):
        inspection_id = _as_str(row["inspection_id"])
        if inspection_id is None:
            raise ValueError("Encountered a row with no inspection_id; cannot resolve identity")

        license_key = normalize_license(_as_str(row["license_"]))
        name = normalize_name(_as_str(row["dba_name"]))
        aka = normalize_name(_as_str(row["aka_name"]))
        address = normalize_address(_as_str(row["address"]), _as_str(row["zip"]))
        geo = normalize_geo(_as_str(row["latitude"]), _as_str(row["longitude"]))
        facility = normalize_facility_type(_as_str(row["facility_type"]))

        signature = _signature(
            license_key,
            name.key,
            aka.key,
            address.key,
            address.unit,
            address.zip_key,
            geo.key,
        )
        node_id = node_id_for(signature)
        assignment[inspection_id] = node_id

        bucket = accumulated.get(node_id)
        if bucket is None:
            accumulated[node_id] = _Accumulator(
                license_key=license_key,
                name=name,
                aka=aka,
                address=address,
                geo=geo,
                facility_type_key=facility,
                inspection_ids=[inspection_id],
                raw_name=_as_str(row["dba_name"]) or "",
                raw_address=_as_str(row["address"]) or "",
                raw_zip=_as_str(row["zip"]),
            )
        else:
            bucket.inspection_ids.append(inspection_id)

    nodes: list[Node] = []
    for node_id, bucket in accumulated.items():
        ordered = tuple(sorted(bucket.inspection_ids, key=_inspection_sort_key))
        nodes.append(
            Node(
                node_id=node_id,
                license_key=bucket.license_key,
                name=bucket.name,
                aka=bucket.aka,
                address=bucket.address,
                geo=bucket.geo,
                facility_type_key=bucket.facility_type_key,
                inspection_ids=ordered,
                min_inspection_id=_inspection_sort_key(ordered[0]),
                raw_name=bucket.raw_name,
                raw_address=bucket.raw_address,
                raw_zip=bucket.raw_zip,
            )
        )

    nodes.sort(key=lambda n: n.node_id)
    logger.info("Built %d nodes from %d inspection rows", len(nodes), len(assignment))
    return nodes, assignment


def _inspection_sort_key(inspection_id: str) -> int:
    """Compare inspection ids numerically.

    Findings §1 confirmed every id in the snapshot is numeric. A non-numeric id
    would silently sort as a string and change which row anchors an
    establishment, so it raises instead.
    """
    if not inspection_id.isdigit():
        raise ValueError(
            f"Non-numeric inspection_id {inspection_id!r}; establishment ids assume numeric ids"
        )
    return int(inspection_id)


def blacklisted_coordinates(nodes: Sequence[Node], *, max_addresses: int) -> frozenset[str]:
    """Coordinates covering too many distinct addresses to mean one place.

    Findings §8: 95.4% of coordinates map to a single address key and the worst
    observed covers 4. Anything beyond that is a geocoder artefact and must not
    be allowed to stand in for address equality.
    """
    per_coord: dict[str, set[str]] = {}
    for node in nodes:
        if node.geo.key is None or node.address.key is None:
            continue
        per_coord.setdefault(node.geo.key, set()).add(node.address.key)
    return frozenset(k for k, addrs in per_coord.items() if len(addrs) > max_addresses)
