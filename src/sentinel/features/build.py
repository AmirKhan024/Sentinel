"""Orchestration: raw + identities + targets in, as-of feature table out.

The only module in the package that touches the filesystem, so everything it
calls stays a pure function of its inputs and the build is deterministic.

Determinism rests on three properties, asserted by the test suite:

1. The priority classifier is a pure function of the violation text.
2. The aggregation is a ``GROUP BY`` over a range join, which is order-independent.
3. The output is sorted by ``(establishment_id, inspection_date, target_inspection_id)``.

Together these mean shuffling any input cannot change a single feature value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.features import historical, validate, writer
from sentinel.features.definitions import (
    FEATURE_COLUMNS,
    FEATURE_DEFINITION_VERSION,
    WINDOW_DAYS,
)
from sentinel.features.models import ArtifactRecord, FeatureManifest, ValidationCheck
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.target.models import CODE_ERA_START

logger = logging.getLogger(__name__)

DATASET_SLUG = "as_of_features"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Stated in the manifest so a consumer never has to infer it from the code.
TEMPORAL_BOUNDARY = "history.inspection_date < target.inspection_date (strictly before)"

REQUIRED_RAW_COLUMNS = (
    "inspection_id",
    "inspection_date",
    "inspection_type",
    "results",
    "violations",
    "dba_name",
)


class FeatureConstructionError(RuntimeError):
    """Raised when the feature table cannot be built at all."""


@dataclass
class FeatureResult:
    """Everything a caller needs after a build, written or not."""

    features: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: FeatureManifest
    features_path: Path | None = None
    manifest_path: Path | None = None


def build_features(
    settings: Settings,
    *,
    parquet_path: Path,
    assignments_path: Path,
    targets_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> FeatureResult:
    """Construct the as-of feature table from the three upstream artifacts."""
    for label, path in (
        ("Raw Parquet", parquet_path),
        ("Component 2 assignments", assignments_path),
        ("Component 3 targets", targets_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    started = datetime.now(UTC)
    logger.info(
        "Building as-of features from %s + %s + %s",
        parquet_path.name,
        assignments_path.name,
        targets_path.name,
    )

    raw = pl.read_parquet(parquet_path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise FeatureConstructionError(
            f"Raw snapshot is missing required columns: {', '.join(missing)}"
        )

    assignments = pl.read_parquet(assignments_path, columns=["inspection_id", "establishment_id"])
    targets = pl.read_parquet(targets_path)

    # Classify every inspection's violations with Component 3's own parser, so
    # "priority" cannot mean one thing in the label and another in the feature.
    flags = historical.priority_flags(
        raw.select(["inspection_id", "violations", "inspection_date"])
    )

    conn = duckdb.connect(database=":memory:")
    try:
        conn.register("raw", raw)
        conn.register("assignments", assignments)
        conn.register("target_rows", targets)
        conn.register("flags", flags)

        relation = historical.compute_features(conn)
        frame = writer.finalize(relation.pl())

        eligible = int(targets.filter(pl.col("target_status") == "eligible").height)
        conn.register("features_out", frame)
        checks = validate.validate_features(
            conn,
            columns=list(frame.columns),
            eligible_target_rows=eligible,
        )
        history_row = conn.execute("SELECT count(*) FROM history").fetchone()
        history_rows = int(history_row[0]) if history_row else 0
    finally:
        conn.close()

    destination = output_dir or settings.features_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    features_path: Path | None = None
    if not dry_run:
        features_path = destination / f"{DATASET_SLUG}_{stamp}.parquet"
        writer.write_table(frame, features_path)
        artifacts.append(
            ArtifactRecord(
                path=features_path.name,
                bytes=features_path.stat().st_size,
                sha256=compute_sha256(features_path),
                row_count=frame.height,
                schema=writer.schema_of(frame),
            )
        )

    manifest = FeatureManifest(
        code_version=__version__,
        feature_definition_version=FEATURE_DEFINITION_VERSION,
        built_at=started.isoformat(),
        source_path=parquet_path.name,
        source_sha256=compute_sha256(parquet_path),
        assignments_path=assignments_path.name,
        assignments_sha256=compute_sha256(assignments_path),
        targets_path=targets_path.name,
        targets_sha256=compute_sha256(targets_path),
        temporal_boundary=TEMPORAL_BOUNDARY,
        window_days=list(WINDOW_DAYS),
        code_era_start=CODE_ERA_START,
        eligible_target_rows=eligible,
        feature_rows=frame.height,
        feature_count=len(FEATURE_COLUMNS),
        rows_without_any_history=_count_zero(frame, "prior_inspection_count_any_type"),
        rows_without_prior_canvass=_count_zero(frame, "prior_canvass_count"),
        rows_without_code_era_canvass=_count_zero(frame, "prior_canvass_count_code_era"),
        rows_with_name_change=int(frame.filter(pl.col("name_changed_since_last_canvass")).height),
        null_rates=writer.null_rates(frame),
        artifacts=artifacts,
        checks=[
            {
                "name": c.name,
                "severity": c.severity,
                "passed": str(c.passed),
                "detail": c.detail,
            }
            for c in checks
        ],
    )

    manifest_path: Path | None = None
    if not dry_run and features_path is not None:
        manifest_path = manifest_path_for(features_path)
        write_manifest(manifest, manifest_path)

    logger.info("Built %d feature rows from %d history rows", frame.height, history_rows)
    return FeatureResult(
        features=frame,
        checks=checks,
        manifest=manifest,
        features_path=features_path,
        manifest_path=manifest_path,
    )


def _count_zero(frame: pl.DataFrame, column: str) -> int:
    if frame.height == 0:
        return 0
    return int(frame.filter(pl.col(column) == 0).height)


def summarize(result: FeatureResult) -> str:
    """One-screen summary of a build, printed by the CLI."""
    m = result.manifest
    lines = [
        f"target rows (eligible): {m.eligible_target_rows}",
        f"feature rows:           {m.feature_rows}",
        f"features:               {m.feature_count}",
        f"boundary:               {m.temporal_boundary}",
        f"windows:                {', '.join(f'{d}d' for d in m.window_days)}",
        f"definition:             {m.feature_definition_version}",
        f"  no history at all:    {m.rows_without_any_history}",
        f"  no prior canvass:     {m.rows_without_prior_canvass}",
        f"  no code-era canvass:  {m.rows_without_code_era_canvass}",
        f"  after a name change:  {m.rows_with_name_change}",
    ]
    if result.features_path is not None:
        lines.append(f"features:               {result.features_path}")
        lines.append(f"manifest:               {result.manifest_path}")
    return "\n".join(lines)
