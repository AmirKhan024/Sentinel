"""Establishment display identity: the human-readable name and address behind an `establishment_id`.

Component 2 (entity resolution) already computes ``canonical_name`` and ``canonical_address`` for
every establishment, explicitly for this purpose -- see ``sentinel.entity.writer.build_establishments``,
whose docstring calls them "display fields... for human readability only." Nothing downstream ever
read them: every recommendation, schedule, backlog and review row exposed only ``establishment_id``,
so every page in the product showed an inspector a string like ``EST-00002282595`` instead of the
restaurant's actual name. This module is the missing join.
"""

from __future__ import annotations

import polars as pl

from sentinel.api.errors import ArtifactNotFound
from sentinel.api.services.artifacts import read_table, resolve_latest
from sentinel.config import Settings

_IDENTITY_COLUMNS = ("establishment_name", "establishment_address")


def load_establishment_identity(settings: Settings) -> pl.DataFrame | None:
    """The latest entity-resolution establishment table, reduced to display columns.

    Returns ``None`` (never raises) when Component 2 hasn't been run in this environment --
    identity is a display enhancement, not a dependency any other page should fail on.
    """
    try:
        path = resolve_latest(settings.entity_resolution_interim_dir, prefix="establishments")
    except ArtifactNotFound:
        return None
    frame = read_table(path)
    return frame.select(
        "establishment_id",
        pl.col("canonical_name").alias("establishment_name"),
        pl.col("canonical_address").alias("establishment_address"),
    )


def join_establishment_identity(frame: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """Left-join display name/address onto any frame keyed by ``establishment_id``.

    Always returns a frame carrying both identity columns, so callers can unpack a row straight
    into a schema without checking whether the join found a match or ran at all.
    """
    identity = load_establishment_identity(settings)
    if identity is None:
        return frame.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("establishment_name"),
            pl.lit(None, dtype=pl.Utf8).alias("establishment_address"),
        )
    return frame.join(identity, on="establishment_id", how="left")


def establishment_identity_row(
    settings: Settings, establishment_id: str
) -> tuple[str | None, str | None]:
    """The display name and address for one establishment, or ``(None, None)`` if unavailable."""
    identity = load_establishment_identity(settings)
    if identity is None:
        return None, None
    row = identity.filter(pl.col("establishment_id") == establishment_id)
    if row.height == 0:
        return None, None
    record = row.row(0, named=True)
    return record["establishment_name"], record["establishment_address"]


__all__ = [
    "establishment_identity_row",
    "join_establishment_identity",
    "load_establishment_identity",
]
