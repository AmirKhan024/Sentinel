"""Orchestration: a Component 18 priority set + a capacity request in, a selection out.

The only module in the package that touches the filesystem or the clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.features.models import ValidationCheck
from sentinel.manifest import compute_sha256, manifest_path_for, read_manifest_as, write_manifest
from sentinel.operational_scoring.models import OperationalScoringManifest
from sentinel.operational_selection import select, validate, writer
from sentinel.operational_selection.definitions import (
    ALLOCATION_SOURCE,
    OPERATIONAL_SELECTION_DEFINITION_VERSION,
)
from sentinel.operational_selection.models import (
    ArtifactRecord,
    OperationalCapacityRequest,
    OperationalSelectionManifest,
)
from sentinel.operational_selection.select import SelectionError

logger = logging.getLogger(__name__)

DATASET_SLUG = "operational_selection"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

REQUIRED_PRIORITY_COLUMNS = ("target_inspection_id", "scoring_status", "planning_date")


class OperationalSelectionBuildError(RuntimeError):
    """Raised when a capacity-constrained selection cannot be built at all."""


@dataclass
class OperationalSelectionResult:
    """Everything a caller needs after a build, written or not."""

    selection: pl.DataFrame
    checks: list[ValidationCheck]
    manifest: OperationalSelectionManifest
    selection_path: Path | None = None
    manifest_path: Path | None = None


def build_operational_selection(
    settings: Settings,
    *,
    priority_path: Path,
    maximum_inspections: int,
    planning_date: str | None = None,
    policy_id: str = "",
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> OperationalSelectionResult:
    """Select up to ``maximum_inspections`` establishments from a Component 18 priority set.

    ``planning_date``, if given, must match the priority set's own date -- a mismatch
    is refused rather than silently trusted (Component 18 priority sets from different
    planning runs must never be interchangeable by accident).
    """
    if not priority_path.exists():
        raise FileNotFoundError(f"Component 18 priority set not found: {priority_path}")
    priority_manifest_path = manifest_path_for(priority_path)
    if not priority_manifest_path.exists():
        raise FileNotFoundError(
            f"Component 18 manifest not found: {priority_manifest_path}. Operational "
            "selection needs it to know which operational fold produced this priority set"
        )

    started = datetime.now(UTC)

    priority_frame = pl.read_parquet(priority_path)
    missing = [c for c in REQUIRED_PRIORITY_COLUMNS if c not in priority_frame.columns]
    if missing:
        raise OperationalSelectionBuildError(
            f"{priority_path.name}: not a Component 18 priority set, missing {', '.join(missing)}"
        )
    if priority_frame.is_empty():
        raise OperationalSelectionBuildError(
            f"{priority_path.name}: zero candidates in the priority set -- there is "
            "nothing to select from"
        )

    scoring_manifest = read_manifest_as(OperationalScoringManifest, priority_manifest_path)
    resolved_planning_date = planning_date or scoring_manifest.planning_date
    if resolved_planning_date != scoring_manifest.planning_date:
        raise OperationalSelectionBuildError(
            f"requested planning_date {resolved_planning_date!r} does not match the "
            f"priority set's own {scoring_manifest.planning_date!r}"
        )

    request = OperationalCapacityRequest(
        planning_date=resolved_planning_date,
        maximum_inspections=maximum_inspections,
        policy_id=policy_id,
    )

    try:
        result = select.select_candidates(
            priority_frame=priority_frame,
            capacity=request,
            operational_fold_set=scoring_manifest.operational_fold_set,
            operational_fold_id=scoring_manifest.operational_fold_id,
        )
    except SelectionError as exc:
        raise OperationalSelectionBuildError(str(exc)) from exc

    checks = validate.run_all_checks(priority_frame, result)

    finalized = writer.finalize(
        result.frame,
        operational_selection_definition_version=OPERATIONAL_SELECTION_DEFINITION_VERSION,
        requested_capacity=result.requested_capacity,
        policy_id=result.policy.policy_id,
    )

    destination = output_dir or settings.operational_selection_processed_dir
    stamp = started.strftime(TIMESTAMP_FORMAT)

    artifacts: list[ArtifactRecord] = []
    selection_path: Path | None = None
    if not dry_run:
        selection_path = (
            destination
            / f"{DATASET_SLUG}_{resolved_planning_date}_cap{maximum_inspections}_{stamp}.parquet"
        )
        writer.write_table(finalized, selection_path)
        artifacts.append(
            ArtifactRecord(
                path=selection_path.name,
                bytes=selection_path.stat().st_size,
                sha256=compute_sha256(selection_path),
                row_count=finalized.height,
                schema=writer.schema_of(finalized),
            )
        )

    capacity_utilization = (
        result.selected_count / result.requested_capacity if result.requested_capacity > 0 else None
    )
    warnings: list[str] = []
    if result.unfilled_capacity > 0:
        warnings.append(
            f"capacity_shortfall: requested {result.requested_capacity}, only "
            f"{result.selected_count} establishment(s) were selectable -- "
            f"{result.unfilled_capacity} slot(s) unfilled. No establishment was fabricated "
            "to fill them."
        )
    if result.unscorable_count > 0:
        warnings.append(
            f"{result.unscorable_count} of {result.ranked_candidate_count} ranked "
            "candidate(s) were never scored by Component 18 and were excluded from "
            "allocation entirely"
        )

    manifest = OperationalSelectionManifest(
        code_version=__version__,
        operational_selection_definition_version=OPERATIONAL_SELECTION_DEFINITION_VERSION,
        built_at=started.isoformat(),
        planning_date=resolved_planning_date,
        priority_set_path=priority_path.name,
        priority_set_sha256=compute_sha256(priority_path),
        operational_scoring_definition_version=scoring_manifest.operational_scoring_definition_version,
        composite_model_name=scoring_manifest.composite_model_name,
        requested_capacity=result.requested_capacity,
        policy_id=result.policy.policy_id,
        policy_mechanism=result.policy.mechanism.value,
        policy_reserve_share=result.policy.reserve_share,
        allocation_source=ALLOCATION_SOURCE,
        ranked_candidate_count=result.ranked_candidate_count,
        selectable_candidate_count=result.selectable_candidate_count,
        unscorable_count=result.unscorable_count,
        selected_count=result.selected_count,
        reserve_selected_count=result.reserve_selected_count,
        risk_selected_count=result.risk_selected_count,
        unfilled_capacity=result.unfilled_capacity,
        capacity_utilization=capacity_utilization,
        coverage_eligible_selected_count=result.coverage_eligible_selected_count,
        warnings=warnings,
        artifacts=artifacts,
        checks=[
            {"name": c.name, "severity": c.severity, "passed": str(c.passed), "detail": c.detail}
            for c in checks
        ],
    )

    manifest_path: Path | None = None
    if not dry_run and selection_path is not None:
        manifest_path = manifest_path_for(selection_path)
        write_manifest(manifest, manifest_path)

    logger.info(
        "Selected %d/%d requested (%d ranked, %d unscorable) for planning_date=%s",
        result.selected_count,
        result.requested_capacity,
        result.ranked_candidate_count,
        result.unscorable_count,
        resolved_planning_date,
    )
    return OperationalSelectionResult(
        selection=finalized,
        checks=checks,
        manifest=manifest,
        selection_path=selection_path,
        manifest_path=manifest_path,
    )


def summarize(result: OperationalSelectionResult) -> str:
    """One-screen summary of a build, printed by the CLI."""
    m = result.manifest
    lines = [
        f"planning date:          {m.planning_date}",
        f"model:                  {m.composite_model_name}",
        f"policy:                 {m.policy_id} ({m.policy_mechanism}, "
        f"reserve share {m.policy_reserve_share})",
        f"requested capacity:     {m.requested_capacity}",
        f"ranked candidates:      {m.ranked_candidate_count}",
        f"  unscorable:           {m.unscorable_count}",
        f"  selectable:           {m.selectable_candidate_count}",
        f"selected:               {m.selected_count}",
        f"  by risk rank:         {m.risk_selected_count}",
        f"  by coverage reserve:  {m.reserve_selected_count}",
        f"  coverage-eligible:    {m.coverage_eligible_selected_count}",
        f"unfilled capacity:      {m.unfilled_capacity}",
        f"capacity utilization:   {m.capacity_utilization}",
    ]
    if m.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in m.warnings)
    if result.selection_path is not None:
        lines.append(f"selection:              {result.selection_path}")
        lines.append(f"manifest:               {result.manifest_path}")
    return "\n".join(lines)


__all__ = [
    "OperationalSelectionBuildError",
    "OperationalSelectionResult",
    "build_operational_selection",
    "summarize",
]
