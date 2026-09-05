"""Orchestration: a Component 19 selection in, a geographic plan out.

The only module in the package that touches the filesystem or the clock. Component 20
never reads Component 18's output directly -- only Component 19's, so location can
never bypass capacity/policy selection (the dependency the task requires be
structurally enforced, not just documented).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.geographic_organization import grouping, metrics, organization, validate, writer
from sentinel.geographic_organization.definitions import (
    GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION,
    UNMAPPED_GROUP_ID,
    OrganizationMode,
    resolve_threshold_km,
    validate_threshold,
)
from sentinel.geographic_organization.metrics import GeographicGroupMetrics
from sentinel.geographic_organization.models import (
    ArtifactRecord,
    GeographicOrganizationManifest,
    GeographicOrganizationResult,
    GeographicPlanSummary,
)
from sentinel.manifest import compute_sha256, manifest_path_for, read_manifest_as, write_manifest
from sentinel.operational_selection.models import OperationalSelectionManifest

logger = logging.getLogger(__name__)

DATASET_SLUG = "geographic_inspection_plan"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
GEOGRAPHIC_ALGORITHM = "distance_threshold_connected_components"

REQUIRED_SELECTION_COLUMNS = ("target_inspection_id", "is_selected", "planning_date")


class GeographicOrganizationBuildError(RuntimeError):
    """Raised when a geographic plan cannot be built at all."""


def build_geographic_plan(
    settings: Settings,
    *,
    selection_path: Path,
    threshold_km: float | None = None,
    threshold_preset: str | None = None,
    organization_mode: OrganizationMode = OrganizationMode.RISK_FIRST,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> GeographicOrganizationResult:
    """Geographically organize exactly Component 19's selected establishments.

    Never fabricates a coordinate, never adds or removes a selected establishment,
    and never rewrites a risk or policy field -- enforced by ``validate.run_all_checks``
    below, not merely by convention.

    ``threshold_km`` and ``threshold_preset`` are mutually exclusive (see
    ``definitions.resolve_threshold_km``); passing neither uses
    ``DEFAULT_GEO_THRESHOLD_KM``. ``organization_mode`` controls only the suggested
    order within a work block -- it never changes which establishments are grouped
    together (that is still ``threshold_km`` alone).
    """
    threshold_km = resolve_threshold_km(
        threshold_km=threshold_km, threshold_preset=threshold_preset
    )
    validate_threshold(threshold_km)

    if not selection_path.exists():
        raise FileNotFoundError(f"Component 19 selection set not found: {selection_path}")
    selection_manifest_path = manifest_path_for(selection_path)
    if not selection_manifest_path.exists():
        raise FileNotFoundError(f"Component 19 manifest not found: {selection_manifest_path}")

    started = datetime.now(UTC)

    selection_frame = pl.read_parquet(selection_path)
    missing = [c for c in REQUIRED_SELECTION_COLUMNS if c not in selection_frame.columns]
    if missing:
        raise GeographicOrganizationBuildError(
            f"{selection_path.name}: not a Component 19 selection set, missing {', '.join(missing)}"
        )
    if selection_frame.is_empty():
        raise GeographicOrganizationBuildError(
            f"{selection_path.name}: the selection artifact itself has zero rows"
        )

    selection_manifest = read_manifest_as(OperationalSelectionManifest, selection_manifest_path)
    planning_date = selection_manifest.planning_date

    selected = selection_frame.filter(pl.col("is_selected"))
    selected_count = selected.height

    if selected_count == 0:
        logger.info(
            "Zero establishments selected for planning_date=%s (capacity=%d) -- "
            "producing a valid, empty geographic plan",
            planning_date,
            selection_manifest.requested_capacity,
        )
        grouped = selected.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("location_status"),
            pl.lit(None, dtype=pl.Utf8).alias("geographic_group_id"),
            pl.lit(None, dtype=pl.Utf8).alias("geographic_group_label"),
            pl.lit(None, dtype=pl.Utf8).alias("work_block_id"),
            pl.lit(None, dtype=pl.Utf8).alias("work_block_label"),
            pl.lit(None, dtype=pl.Int64).alias("suggested_order_in_block"),
            pl.lit(organization_mode.value).alias("organization_mode"),
            pl.lit(None, dtype=pl.Int64).alias("highest_sentinel_rank_in_block"),
        )
        group_metrics_list: list[GeographicGroupMetrics] = []
        work_blocks: list[organization.WorkBlock] = []
    else:
        grouped = grouping.assign_geographic_groups(selected, threshold_km=threshold_km)
        group_metrics_list = metrics.compute_group_metrics(grouped)
        work_blocks = organization.build_work_blocks(grouped, group_metrics_list)

        # Suggested order and highest-rank-in-block are keyed by target_inspection_id
        # (Component 19's stable identity column, not establishment_id) so a duplicate
        # establishment_id could never collide -- matches how the rest of Component 20
        # keys its own invariant checks.
        order_map: dict[str, int | None] = {}
        rank_map: dict[str, int | None] = {}
        for block in work_blocks:
            block_members = grouped.filter(pl.col("geographic_group_id") == block.block_id)
            if block.has_unmapped_member:
                # No suggested order within the unmapped pseudo-block: there is no
                # geography to order by, and pretending otherwise would fabricate one.
                for tid in block_members["target_inspection_id"].to_list():
                    order_map[tid] = None
                    rank_map[tid] = None
                continue
            for order_index, target_id, _rank in organization.suggested_work_order(
                block_members, mode=organization_mode
            ):
                order_map[target_id] = order_index
                rank_map[target_id] = block.highest_sentinel_rank

        grouped = grouped.with_columns(
            pl.col("geographic_group_id").alias("work_block_id"),
            pl.col("geographic_group_label").alias("work_block_label"),
            pl.lit(organization_mode.value).alias("organization_mode"),
            pl.col("target_inspection_id")
            .map_elements(lambda tid: order_map.get(tid), return_dtype=pl.Int64)
            .alias("suggested_order_in_block"),
            pl.col("target_inspection_id")
            .map_elements(lambda tid: rank_map.get(tid), return_dtype=pl.Int64)
            .alias("highest_sentinel_rank_in_block"),
        )

    plan_frame = writer.finalize(
        grouped,
        geographic_organization_definition_version=GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION,
        geographic_algorithm=GEOGRAPHIC_ALGORITHM,
        threshold_km=threshold_km,
    )

    # Re-derived once, from the finalized plan itself, and passed to the checks below
    # as the "claimed" counts they independently verify against the plan's own rows.
    location_available_count = int(
        plan_frame.filter(pl.col("location_status") == "location_available").height
    )
    location_unavailable_count = selected_count - location_available_count
    location_coverage_pct = (
        round(100.0 * location_available_count / selected_count, 1) if selected_count else 0.0
    )
    geographic_group_count = int(
        plan_frame.filter(pl.col("geographic_group_id") != UNMAPPED_GROUP_ID)[
            "geographic_group_id"
        ].n_unique()
    )

    checks = validate.run_all_checks(
        selected,
        plan_frame,
        location_available_count=location_available_count,
        location_unavailable_count=location_unavailable_count,
    )

    group_metrics_dicts: list[dict[str, object]] = [
        {
            "group_id": m.group_id,
            "group_label": m.group_label,
            "size": m.size,
            "member_establishment_ids": m.member_establishment_ids,
            "centroid_lat": m.centroid_lat,
            "centroid_lon": m.centroid_lon,
            "max_within_group_distance_km": m.max_within_group_distance_km,
            "avg_within_group_distance_km": m.avg_within_group_distance_km,
        }
        for m in group_metrics_list
    ]

    work_block_dicts: list[dict[str, object]] = [
        {
            "block_id": b.block_id,
            "label": b.label,
            "establishment_ids": b.establishment_ids,
            "size": b.size,
            "centroid_lat": b.centroid_lat,
            "centroid_lon": b.centroid_lon,
            "avg_within_block_distance_km": b.avg_within_block_distance_km,
            "max_within_block_distance_km": b.max_within_block_distance_km,
            "highest_sentinel_rank": b.highest_sentinel_rank,
            "rank_range": list(b.rank_range) if b.rank_range else None,
            "member_ranks": b.member_ranks,
            "has_unmapped_member": b.has_unmapped_member,
            "rationale": b.rationale,
        }
        for b in work_blocks
    ]

    mapped_blocks = [b for b in work_blocks if not b.has_unmapped_member]
    singleton_blocks = [b for b in mapped_blocks if b.size == 1]
    notes: list[str] = [organization.organization_mode_rationale(organization_mode)]
    if mapped_blocks and len(singleton_blocks) >= max(1, round(0.7 * len(mapped_blocks))):
        notes.append(
            f"{len(singleton_blocks)} of {len(mapped_blocks)} geographic work block(s) "
            f"contain a single establishment; the selected establishments are spatially "
            f"dispersed at the {threshold_km} km threshold. A broader threshold (see "
            "--threshold-preset broad, or a larger --threshold-km) will merge more of "
            "them into shared blocks, at the cost of grouping establishments farther apart."
        )

    destination = output_dir or settings.geographic_organization_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    plan_path: Path | None = None
    if not dry_run:
        plan_path = destination / f"{DATASET_SLUG}_{planning_date}_{stamp}.parquet"
        writer.write_table(plan_frame, plan_path)
        artifacts.append(
            ArtifactRecord(
                path=plan_path.name,
                bytes=plan_path.stat().st_size,
                sha256=compute_sha256(plan_path),
                row_count=plan_frame.height,
                schema=writer.schema_of(plan_frame),
            )
        )

    warnings: list[str] = []
    if location_unavailable_count > 0:
        warnings.append(
            f"{location_unavailable_count} of {selected_count} selected establishment(s) "
            "have no usable coordinates and are preserved in the 'unmapped' group rather "
            "than fabricated a location"
        )

    manifest = GeographicOrganizationManifest(
        code_version=__version__,
        geographic_organization_definition_version=GEOGRAPHIC_ORGANIZATION_DEFINITION_VERSION,
        built_at=started.isoformat(),
        planning_date=planning_date,
        selection_artifact_path=selection_path.name,
        selection_artifact_sha256=compute_sha256(selection_path),
        operational_selection_definition_version=selection_manifest.operational_selection_definition_version,
        composite_model_name=selection_manifest.composite_model_name,
        geographic_algorithm=GEOGRAPHIC_ALGORITHM,
        threshold_km=threshold_km,
        selected_count=selected_count,
        location_available_count=location_available_count,
        location_unavailable_count=location_unavailable_count,
        location_coverage_pct=location_coverage_pct,
        geographic_group_count=geographic_group_count,
        group_metrics=group_metrics_dicts,
        organization_mode=organization_mode.value,
        threshold_preset=threshold_preset,
        work_blocks=work_block_dicts,
        notes=notes,
        warnings=warnings,
        artifacts=artifacts,
        checks=[
            {"name": c.name, "severity": c.severity, "passed": str(c.passed), "detail": c.detail}
            for c in checks
        ],
    )

    manifest_path: Path | None = None
    if not dry_run and plan_path is not None:
        manifest_path = manifest_path_for(plan_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Organized %d selected establishments into %d geographic group(s) "
        "(%d unmapped) for planning_date=%s",
        selected_count,
        geographic_group_count,
        location_unavailable_count,
        planning_date,
    )

    summary = GeographicPlanSummary(
        planning_date=planning_date,
        selected_count=selected_count,
        location_available_count=location_available_count,
        location_unavailable_count=location_unavailable_count,
        location_coverage_pct=location_coverage_pct,
        geographic_group_count=geographic_group_count,
        threshold_km=threshold_km,
        group_metrics=group_metrics_list,
        organization_mode=organization_mode.value,
        threshold_preset=threshold_preset,
        notes=notes,
    )

    return GeographicOrganizationResult(
        plan_frame=plan_frame,
        summary=summary,
        checks=checks,
        manifest=manifest,
        plan_path=plan_path,
        manifest_path=manifest_path,
    )


def summarize(result: GeographicOrganizationResult) -> str:
    """One-screen summary of a build, printed by the CLI."""
    m = result.manifest
    lines = [
        f"planning date:           {m.planning_date}",
        f"model:                   {m.composite_model_name}",
        f"algorithm:               {m.geographic_algorithm} (threshold {m.threshold_km} km"
        f"{f', preset {m.threshold_preset}' if m.threshold_preset else ''})",
        f"organization mode:       {m.organization_mode}",
        f"selected inspection workload: {m.selected_count}",
        f"  with coordinates:      {m.location_available_count}",
        f"  without coordinates:   {m.location_unavailable_count}",
        f"  location coverage:     {m.location_coverage_pct}%",
        f"geographic work blocks:  {m.geographic_group_count}",
    ]
    for block in m.work_blocks:
        if block["has_unmapped_member"]:
            continue
        rank_range = cast("list[int] | None", block["rank_range"])
        rank_text = (
            f"ranks #{rank_range[0]}-#{rank_range[1]}" if rank_range else "no ranked members"
        )
        lines.append(
            f"  {block['label']}: {block['size']} establishment(s), "
            f"highest priority #{block['highest_sentinel_rank']}, {rank_text}, "
            f"max spread {block['max_within_block_distance_km']} km"
        )
    if m.notes:
        lines.append("notes:")
        lines.extend(f"  - {n}" for n in m.notes)
    if m.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in m.warnings)
    if result.plan_path is not None:
        lines.append(f"plan:                    {result.plan_path}")
        lines.append(f"manifest:                {result.manifest_path}")
    return "\n".join(lines)


__all__ = [
    "GEOGRAPHIC_ALGORITHM",
    "GeographicOrganizationBuildError",
    "build_geographic_plan",
    "summarize",
]
