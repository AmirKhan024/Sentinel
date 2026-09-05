"""Component 4's feature engine, run against a planning date instead of a real target.

This module contains no feature logic of its own. Every SQL string it executes is
imported from ``sentinel.features.historical`` unmodified:

* ``HISTORY_SQL`` -- builds the ``history`` view from ``raw`` and ``assignments``.
  Identical for both modes, because history is a property of the establishment,
  not of why a reference date was chosen.
* ``aggregate_sql()`` -- the one range join, ``h.inspection_date < t.inspection_date``,
  that is the entire leakage boundary Component 4 exists to enforce.
* ``features_sql()`` -- the missing-value rules that turn the raw aggregates into
  Component 4's declared feature columns.

The only thing this module supplies that ``historical.compute_features()`` does not
is the ``targets`` view itself: Component 4 builds it from Component 3's real
target table (``historical.TARGETS_SQL``); this builds it from
``candidates.universe.build_candidate_universe``'s output. Both views expose the
same seven columns, so ``aggregate_sql()`` and ``features_sql()`` cannot tell them
apart -- which is the guarantee that historical and operational features are the
same feature engine, not two engines that happen to agree today.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

from sentinel.features import historical

logger = logging.getLogger(__name__)


def compute_operational_features(
    conn: duckdb.DuckDBPyConnection, *, targets: pl.DataFrame
) -> duckdb.DuckDBPyRelation:
    """Run Component 4's aggregation and derivation against a candidate ``targets`` frame.

    Expects ``raw``, ``assignments`` and ``flags`` already registered on ``conn`` --
    the same three ``historical.compute_features`` expects, minus ``target_rows``,
    which real target rows would supply and which operational mode has no analogue
    for. Registers ``targets`` directly instead of deriving it via
    ``historical.TARGETS_SQL``.
    """
    conn.register("targets", targets)
    conn.execute(historical.HISTORY_SQL)
    conn.execute(historical.aggregate_sql())
    conn.execute(historical.features_sql())
    logger.info("Computed operational as-of features (boundary: inspection_date < planning_date)")
    return conn.sql(
        "SELECT * FROM features ORDER BY establishment_id, inspection_date, target_inspection_id"
    )


__all__ = ["compute_operational_features"]
