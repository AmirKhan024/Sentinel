"""Orchestration: raw + identities + a planning date in, candidate table out.

The only module in the package that touches the filesystem or the clock, matching
every other component's convention. Mirrors ``features.build.build_features``
structurally -- same three-stage shape (load, compute, validate, write) -- because
this *is* Component 4's shape, run against a different source for one input.

Determinism rests on the same three properties ``features.build`` documents, plus
one more specific to this component: the candidate universe query and the
synthetic id scheme are both pure functions of (raw snapshot, assignments,
planning_date), so the same three inputs always yield the same candidate rows
with the same ids, in any row order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import polars as pl

from sentinel import __version__
from sentinel.candidates import universe, validate, writer
from sentinel.candidates.definitions import (
    CANDIDATE_DEFINITION_VERSION,
    CANDIDATE_ELIGIBILITY_RULE,
    CANDIDATE_TEMPORAL_BOUNDARY,
)
from sentinel.candidates.features import compute_operational_features
from sentinel.candidates.models import ArtifactRecord, CandidateManifest, ValidationCheck
from sentinel.candidates.universe import CandidateGenerationError
from sentinel.config import Settings
from sentinel.features import historical
from sentinel.features import validate as feature_validate
from sentinel.features.definitions import FEATURE_DEFINITION_VERSION
from sentinel.features.writer import finalize as finalize_features
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest

logger = logging.getLogger(__name__)

DATASET_SLUG = "operational_candidates"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

REQUIRED_RAW_COLUMNS = (
    "inspection_id",
    "inspection_date",
    "inspection_type",
    "results",
    "violations",
    "dba_name",
    "address",
    "zip",
    "latitude",
    "longitude",
)

REQUIRED_ESTABLISHMENT_COLUMNS = (
    "establishment_id",
    "canonical_name",
    "canonical_address",
    "canonical_zip",
)


@dataclass
class CandidateResult:
    """Everything a caller needs after a build, written or not."""

    candidates: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: CandidateManifest
    candidates_path: Path | None = None
    manifest_path: Path | None = None


def build_candidates(
    settings: Settings,
    *,
    planning_date: str,
    parquet_path: Path,
    assignments_path: Path,
    establishments_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> CandidateResult:
    """Construct the operational candidate table for one planning date.

    Raises :class:`~sentinel.candidates.universe.CandidateGenerationError` when the
    planning date cannot be supported at all (malformed, or on/before the earliest
    inspection record in the raw snapshot). A supportable-but-stale date (later
    than the most recent ingested record) does not raise -- it is recorded as a
    warning in the manifest instead, because no future information is ever read
    either way.
    """
    for label, path in (
        ("Raw Parquet", parquet_path),
        ("Component 2 assignments", assignments_path),
        ("Component 2 establishments", establishments_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    started = datetime.now(UTC)
    logger.info(
        "Building operational candidates for planning_date=%s from %s + %s + %s",
        planning_date,
        parquet_path.name,
        assignments_path.name,
        establishments_path.name,
    )

    raw = pl.read_parquet(parquet_path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise CandidateGenerationError(
            f"Raw snapshot is missing required columns: {', '.join(missing)}"
        )

    assignments = pl.read_parquet(assignments_path, columns=["inspection_id", "establishment_id"])
    establishments = pl.read_parquet(establishments_path)
    missing_est = [c for c in REQUIRED_ESTABLISHMENT_COLUMNS if c not in establishments.columns]
    if missing_est:
        raise CandidateGenerationError(
            f"Establishments table is missing required columns: {', '.join(missing_est)}"
        )

    conn = duckdb.connect(database=":memory:")
    try:
        conn.register("raw", raw)
        conn.register("assignments", assignments)

        # Validated before the expensive per-row violation parse below: a
        # malformed or out-of-range planning date should fail fast rather than
        # after paying for work whose result will be thrown away.
        candidate_universe = universe.build_candidate_universe(conn, planning_date=planning_date)

        flags = historical.priority_flags(
            raw.select(["inspection_id", "violations", "inspection_date"])
        )
        conn.register("flags", flags)

        relation = compute_operational_features(conn, targets=candidate_universe.targets)
        pure_frame = finalize_features(relation.pl())

        candidate_count = candidate_universe.targets.height
        conn.register("features_out", pure_frame)
        feature_checks = feature_validate.validate_features(
            conn,
            columns=list(pure_frame.columns),
            eligible_target_rows=candidate_count,
        )
        candidate_checks = validate.validate_candidate_universe(conn)
        checks = [*feature_checks, *candidate_checks]
    finally:
        conn.close()

    combined = writer.combine(
        pure_frame,
        candidate_universe.metadata,
        establishments,
        planning_date=planning_date,
    )

    destination = output_dir or settings.operational_candidates_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    candidates_path: Path | None = None
    if not dry_run:
        candidates_path = destination / f"{DATASET_SLUG}_{planning_date}_{stamp}.parquet"
        writer.write_table(combined, candidates_path)
        artifacts.append(
            ArtifactRecord(
                path=candidates_path.name,
                bytes=candidates_path.stat().st_size,
                sha256=compute_sha256(candidates_path),
                row_count=combined.height,
                schema=writer.schema_of(combined),
            )
        )

    cold_start = int(pure_frame.filter(pl.col("prior_canvass_count") == 0).height)
    missing_location = int(candidate_universe.metadata.filter(~pl.col("has_location")).height)

    manifest = CandidateManifest(
        code_version=__version__,
        candidate_definition_version=CANDIDATE_DEFINITION_VERSION,
        feature_definition_version=FEATURE_DEFINITION_VERSION,
        built_at=started.isoformat(),
        planning_date=planning_date,
        source_path=parquet_path.name,
        source_sha256=compute_sha256(parquet_path),
        assignments_path=assignments_path.name,
        assignments_sha256=compute_sha256(assignments_path),
        establishments_path=establishments_path.name,
        establishments_sha256=compute_sha256(establishments_path),
        temporal_boundary=CANDIDATE_TEMPORAL_BOUNDARY,
        candidate_eligibility_rule=CANDIDATE_ELIGIBILITY_RULE,
        min_supported_planning_date=candidate_universe.min_raw_inspection_date,
        max_ingested_inspection_date=candidate_universe.max_raw_inspection_date,
        days_beyond_ingested_data=max(
            0,
            (
                date.fromisoformat(planning_date)
                - date.fromisoformat(candidate_universe.max_raw_inspection_date)
            ).days,
        ),
        candidate_count=candidate_count,
        feature_count=pure_frame.width,
        cold_start_candidates=cold_start,
        candidates_missing_location=missing_location,
        null_rates=writer.null_rates(pure_frame),
        warnings=candidate_universe.warnings,
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
    if not dry_run and candidates_path is not None:
        manifest_path = manifest_path_for(candidates_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Built %d operational candidates (%d cold start, %d missing location)",
        candidate_count,
        cold_start,
        missing_location,
    )
    return CandidateResult(
        candidates=combined,
        checks=checks,
        manifest=manifest,
        candidates_path=candidates_path,
        manifest_path=manifest_path,
    )


def summarize(result: CandidateResult) -> str:
    """One-screen summary of a build, printed by the CLI."""
    m = result.manifest
    lines = [
        f"planning date:          {m.planning_date}",
        f"candidates:              {m.candidate_count}",
        f"features:                {m.feature_count}",
        f"boundary:                {m.temporal_boundary}",
        f"definition:              {m.candidate_definition_version} "
        f"(features {m.feature_definition_version})",
        f"data ingested through:   {m.max_ingested_inspection_date}",
        f"earliest supportable:    {m.min_supported_planning_date}",
        f"  cold start (no canvass): {m.cold_start_candidates}",
        f"  missing location:       {m.candidates_missing_location}",
    ]
    if m.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in m.warnings)
    if result.candidates_path is not None:
        lines.append(f"candidates:              {result.candidates_path}")
        lines.append(f"manifest:                {result.manifest_path}")
    return "\n".join(lines)


__all__ = ["CandidateResult", "build_candidates", "summarize"]
