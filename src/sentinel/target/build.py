"""Orchestration: raw snapshot + Component 2 assignments in, target table out.

The only module in the package that touches the filesystem, so everything it
calls stays a pure function and the build is deterministic by construction.

Determinism rests on three properties, each asserted by the test suite:

1. Classification is a pure function of the violation text.
2. Same-day collapse orders contributing inspections by numeric id.
3. Output rows are sorted by (establishment_id, inspection_date, inspection_id).

Together these mean shuffling the input rows cannot change a single label.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.target import construct, validate, writer
from sentinel.target.models import (
    CANVASS_TYPE,
    CODE_ERA_START,
    INSPECTED_RESULTS,
    TARGET_DEFINITION_VERSION,
    ArtifactRecord,
    InspectionOutcome,
    TargetManifest,
    TargetRow,
    TargetStatus,
    ValidationCheck,
)

logger = logging.getLogger(__name__)

DATASET_SLUG = "inspection_targets"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# The raw columns target construction is permitted to read. Listing them makes
# the boundary explicit: `risk`, `latitude`, `facility_type` and everything else
# are Component 4's business, not the target's.
REQUIRED_RAW_COLUMNS = (
    "inspection_id",
    "inspection_date",
    "inspection_type",
    "results",
    "violations",
)


class TargetConstructionError(RuntimeError):
    """Raised when the target cannot be built at all."""


@dataclass
class TargetResult:
    """Everything a caller needs after a build, written or not."""

    targets: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: TargetManifest
    targets_path: Path | None = None
    manifest_path: Path | None = None


def _as_str(value: object) -> str:
    return "" if value is None else str(value)


def build_targets(
    settings: Settings,
    *,
    parquet_path: Path,
    assignments_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> TargetResult:
    """Construct the prediction target from one snapshot and one assignment table."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Raw Parquet file not found: {parquet_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Component 2 assignments not found: {assignments_path}")

    started = datetime.now(UTC)
    logger.info("Building targets from %s + %s", parquet_path.name, assignments_path.name)

    raw = pl.read_parquet(parquet_path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise TargetConstructionError(
            f"Raw snapshot is missing required columns: {', '.join(missing)}"
        )

    assignments = pl.read_parquet(assignments_path, columns=["inspection_id", "establishment_id"])

    # Inner join: identity is Component 2's contract, and an inspection with no
    # assignment has no establishment to attribute an outcome to.
    joined = raw.select(REQUIRED_RAW_COLUMNS).join(assignments, on="inspection_id", how="inner")
    if joined.height != raw.height:
        logger.warning(
            "%d raw rows had no Component 2 assignment and were dropped",
            raw.height - joined.height,
        )

    outcomes: list[InspectionOutcome] = [
        construct.classify_inspection(
            inspection_id=_as_str(row["inspection_id"]),
            establishment_id=_as_str(row["establishment_id"]),
            inspection_date=_as_str(row["inspection_date"]),
            inspection_type=_as_str(row["inspection_type"]),
            results=_as_str(row["results"]),
            violations=row["violations"],
        )
        for row in joined.iter_rows(named=True)
    ]

    rows: list[TargetRow] = construct.build_target_rows(outcomes)
    frame = writer.build_targets(rows)

    checks = validate.validate_targets(
        rows,
        known_establishment_ids=frozenset(assignments["establishment_id"].to_list()),
        known_inspection_ids=frozenset(raw["inspection_id"].to_list()),
        source_row_count=joined.height,
    )

    eligible = [r for r in rows if r.target_status is TargetStatus.ELIGIBLE]
    positives = sum(1 for r in eligible if r.target == 1)
    by_year: Counter[str] = Counter()
    pos_by_year: Counter[str] = Counter()
    for row in eligible:
        year = row.inspection_date[:4]
        by_year[year] += 1
        if row.target == 1:
            pos_by_year[year] += 1

    destination = output_dir or settings.target_interim_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    targets_path: Path | None = None
    if not dry_run:
        targets_path = destination / f"{DATASET_SLUG}_{stamp}.parquet"
        writer.write_table(frame, targets_path)
        artifacts.append(
            ArtifactRecord(
                path=targets_path.name,
                bytes=targets_path.stat().st_size,
                sha256=compute_sha256(targets_path),
                row_count=frame.height,
                schema=writer.schema_of(frame),
            )
        )

    manifest = TargetManifest(
        code_version=__version__,
        target_definition_version=TARGET_DEFINITION_VERSION,
        built_at=started.isoformat(),
        source_path=parquet_path.name,
        source_sha256=compute_sha256(parquet_path),
        source_row_count=raw.height,
        assignments_path=assignments_path.name,
        assignments_sha256=compute_sha256(assignments_path),
        code_era_start=CODE_ERA_START,
        inspected_results=sorted(INSPECTED_RESULTS),
        canvass_type=CANVASS_TYPE,
        target_rows=len(rows),
        eligible_rows=len(eligible),
        positive_rows=positives,
        negative_rows=len(eligible) - positives,
        positive_rate=round(positives / len(eligible), 6) if eligible else 0.0,
        status_counts=dict(Counter(r.target_status.value for r in rows)),
        positive_rate_by_year={y: round(pos_by_year[y] / by_year[y], 6) for y in sorted(by_year)},
        collapsed_multi_canvass_days=sum(1 for r in eligible if r.n_contributing_inspections > 1),
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
    if not dry_run and targets_path is not None:
        manifest_path = manifest_path_for(targets_path)
        write_manifest(manifest, manifest_path)

    return TargetResult(
        targets=frame,
        checks=checks,
        manifest=manifest,
        targets_path=targets_path,
        manifest_path=manifest_path,
    )


def summarize(result: TargetResult) -> str:
    """One-screen summary of a build, printed by the CLI."""
    m = result.manifest
    lines = [
        f"source rows:      {m.source_row_count}",
        f"target rows:      {m.target_rows}",
        f"  eligible:       {m.eligible_rows}",
        f"  positive:       {m.positive_rows}",
        f"  negative:       {m.negative_rows}",
        f"positive rate:    {m.positive_rate:.4f}",
        f"definition:       {m.target_definition_version}",
    ]
    for status, count in sorted(m.status_counts.items()):
        if status != TargetStatus.ELIGIBLE.value:
            lines.append(f"  {status}: {count}")
    if result.targets_path is not None:
        lines.append(f"targets:          {result.targets_path}")
        lines.append(f"manifest:         {result.manifest_path}")
    return "\n".join(lines)
