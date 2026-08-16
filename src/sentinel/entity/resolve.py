"""Orchestration: raw Parquet in, establishment identities out.

The only module in the package that touches the filesystem. Everything it calls
is a pure function of its arguments, which is what makes the pipeline
deterministic and every other module trivially unit-testable.

Determinism comes from four properties, none of them accidental:

1. Node identifiers are content hashes, so they do not depend on row order.
2. Candidate pairs are canonically ordered and sorted before evaluation.
3. Connected components are re-labelled by their minimum member.
4. Establishment identifiers are derived from the cluster's earliest inspection.

Together these mean that shuffling the input rows cannot change the output
mapping, which is asserted directly in the test suite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.entity import blocking, cluster, evidence, nodes, validate, writer
from sentinel.entity.models import (
    DEFAULT_THRESHOLDS,
    NORMALIZATION_VERSION,
    ArtifactRecord,
    MatchTier,
    PairVerdict,
    ResolutionManifest,
    Thresholds,
    ValidationCheck,
)
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest

logger = logging.getLogger(__name__)

DATASET_SLUG = "establishment_assignments"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class EntityResolutionError(RuntimeError):
    """Raised when resolution cannot proceed at all."""


@dataclass
class ResolutionResult:
    """Everything a caller needs after a run, written or not."""

    assignments: pl.DataFrame
    establishments: pl.DataFrame
    edges: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: ResolutionManifest
    assignments_path: Path | None = None
    establishments_path: Path | None = None
    edges_path: Path | None = None
    manifest_path: Path | None = None


def resolve_establishments(
    settings: Settings,
    *,
    parquet_path: Path,
    output_dir: Path | None = None,
    thresholds: Thresholds | None = None,
    dry_run: bool = False,
) -> ResolutionResult:
    """Resolve one raw snapshot into stable establishment identities."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Raw Parquet file not found: {parquet_path}")

    active = thresholds or DEFAULT_THRESHOLDS
    started = datetime.now(UTC)
    logger.info("Resolving entities from %s", parquet_path)

    frame = pl.read_parquet(parquet_path)
    source_rows = frame.height
    if source_rows == 0:
        logger.warning("Source Parquet is empty; producing empty outputs")

    node_list, assignment = nodes.build_nodes(frame)
    blacklist = nodes.blacklisted_coordinates(
        node_list, max_addresses=active.max_addresses_per_coordinate
    )
    if blacklist:
        logger.info("Blacklisted %d geocoder-artefact coordinates", len(blacklist))

    pairs, oversized = blocking.candidate_pairs(node_list, active, blacklist)

    by_id = {n.node_id: n for n in node_list}
    verdicts: list[PairVerdict] = [
        evidence.evaluate_pair(by_id[left], by_id[right], active, blacklist)
        for left, right in pairs
    ]

    clusters, splits = cluster.build_clusters(node_list, verdicts, active)
    logger.info("Resolved %d nodes into %d establishments", len(node_list), len(clusters))

    checks = validate.validate_output(
        node_list, clusters, verdicts, assignment, active, source_row_count=source_rows
    )

    assignments_frame = writer.build_assignments(node_list, clusters, assignment, verdicts)
    establishments_frame = writer.build_establishments(node_list, clusters)
    edges_frame = writer.build_edges(node_list, clusters, verdicts)

    tier_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    for verdict in verdicts:
        tier_counts[verdict.tier.value] = tier_counts.get(verdict.tier.value, 0) + 1
        rule_counts[verdict.rule_id] = rule_counts.get(verdict.rule_id, 0) + 1

    destination = output_dir or settings.entity_resolution_interim_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    paths: dict[str, Path] = {}
    if not dry_run:
        for slug, table in (
            (DATASET_SLUG, assignments_frame),
            ("establishments", establishments_frame),
            ("entity_resolution_edges", edges_frame),
        ):
            path = destination / f"{slug}_{stamp}.parquet"
            writer.write_table(table, path)
            paths[slug] = path
            artifacts.append(
                ArtifactRecord(
                    path=path.name,
                    bytes=path.stat().st_size,
                    sha256=compute_sha256(path),
                    row_count=table.height,
                    schema=writer.schema_of(table),
                )
            )

    manifest = ResolutionManifest(
        code_version=__version__,
        normalization_version=NORMALIZATION_VERSION,
        resolved_at=started.isoformat(),
        source_path=parquet_path.name,
        source_sha256=compute_sha256(parquet_path),
        source_row_count=source_rows,
        thresholds={
            "max_addresses_per_cluster": float(active.max_addresses_per_cluster),
            "max_zips_per_cluster": float(active.max_zips_per_cluster),
            "max_addresses_per_coordinate": float(active.max_addresses_per_coordinate),
            "max_block_size": float(active.max_block_size),
            "near_house_number_delta": float(active.near_house_number_delta),
            "max_cluster_span_m": active.max_cluster_span_m,
        },
        node_count=len(node_list),
        establishment_count=len(clusters),
        singleton_establishment_count=sum(1 for c in clusters if len(c.node_ids) == 1),
        candidate_pair_count=len(pairs),
        edges_by_tier=tier_counts,
        edges_by_rule=rule_counts,
        splits_by_reason=splits,
        oversized_block_count=len(oversized),
        blacklisted_coordinate_count=len(blacklist),
        unusable_license_rows=sum(
            len(n.inspection_ids) for n in node_list if n.license_key is None
        ),
        unusable_address_rows=sum(
            len(n.inspection_ids) for n in node_list if n.address.key is None
        ),
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
    if not dry_run:
        manifest_path = manifest_path_for(paths[DATASET_SLUG])
        write_manifest(manifest, manifest_path)

    return ResolutionResult(
        assignments=assignments_frame,
        establishments=establishments_frame,
        edges=edges_frame,
        checks=checks,
        manifest=manifest,
        assignments_path=paths.get(DATASET_SLUG),
        establishments_path=paths.get("establishments"),
        edges_path=paths.get("entity_resolution_edges"),
        manifest_path=manifest_path,
    )


def summarize(result: ResolutionResult) -> str:
    """One-screen summary of a run, printed by the CLI."""
    manifest = result.manifest
    ambiguous = manifest.edges_by_tier.get(MatchTier.AMBIGUOUS.value, 0)
    lines = [
        f"source rows:      {manifest.source_row_count}",
        f"nodes:            {manifest.node_count}",
        f"establishments:   {manifest.establishment_count}",
        f"  singletons:     {manifest.singleton_establishment_count}",
        f"candidate pairs:  {manifest.candidate_pair_count}",
        f"  strong:         {manifest.edges_by_tier.get(MatchTier.STRONG.value, 0)}",
        f"  probable:       {manifest.edges_by_tier.get(MatchTier.PROBABLE.value, 0)}",
        f"  ambiguous:      {ambiguous}",
        f"oversized blocks: {manifest.oversized_block_count}",
    ]
    if result.assignments_path is not None:
        lines.append(f"assignments:      {result.assignments_path}")
        lines.append(f"establishments:   {result.establishments_path}")
        lines.append(f"edges:            {result.edges_path}")
        lines.append(f"manifest:         {result.manifest_path}")
    return "\n".join(lines)
