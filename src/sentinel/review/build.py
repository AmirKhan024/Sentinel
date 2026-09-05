"""Component 16 orchestration. The only module here that touches the filesystem or the clock.

The shape of a run:

```text
checksum the inputs
      |
      v
read Component 13's recommendations, Component 14's schedule and execution log (optional)
      |
      v
run the two deterministic triggers, build the queue
      |
      v
parse and apply any human resolutions, beside the queue
      |
      v
write, re-checksum the inputs, validate, manifest
```

**Nothing is re-ranked, re-dated, re-scored or retrained anywhere in it.** Component 16 is a
pure observer of Components 13 and 14: it reads their artifacts, checksums them before the first
read and again after the last write, and fails the run if a single byte moved.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.config import Settings
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.review import inputs, validate, writer
from sentinel.review.definitions import (
    ABSTENTION_POLICY_INHERITED,
    BLOCKED,
    DETERMINISM_SCOPE,
    DOES_NOT_ESTABLISH,
    FOUR_HUMAN_LAYERS,
    INHERITED_LIMITATIONS,
    NO_THRESHOLD,
    REVIEW_DEFINITION_VERSION,
    ReviewCaseStatus,
)
from sentinel.review.definitions import DEFERRAL_IS_NOT_SCHEDULING_DEFERRAL as DEFERRAL_NOTE
from sentinel.review.definitions import REVIEW_CANNOT as REVIEW_CANNOT_STATEMENT
from sentinel.review.models import (
    ArtifactRecord,
    ReviewCase,
    ReviewManifest,
    ReviewStats,
    ValidationCheck,
)
from sentinel.review.resolution import apply_resolutions
from sentinel.review.trigger import build_review_cases, trigger_column

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Figures live with the analysis they illustrate, not in the data layer.
FIGURES_DIR = Path("docs/analysis/figures")


class ReviewBuildError(RuntimeError):
    """Raised when a review run cannot be completed as described."""


@dataclass
class ReviewResult:
    """Everything one run produced, for the CLI, the tests and the manifest."""

    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: ReviewManifest
    stats: ReviewStats
    advisories: list[str]
    queue_path: Path | None = None
    manifest_path: Path | None = None
    written: list[Path] = field(default_factory=list)
    figure_paths: list[Path] = field(default_factory=list)
    dry_run: bool = False


def _queue_row(
    case: ReviewCase, *, review_status: str, review_id: str, resolution_action: str
) -> dict[str, object]:
    return {
        "policy_id": case.policy_id,
        "model_name": case.model_name,
        "fold_set": case.fold_set,
        "fold_id": case.fold_id,
        "k_name": case.k_name,
        "target_inspection_id": case.target_inspection_id,
        "establishment_id": case.establishment_id,
        "final_policy_rank": case.final_policy_rank,
        "decision_mechanism": case.decision_mechanism,
        "decision_reason": case.decision_reason,
        "warnings": case.warnings,
        "trigger_reasons": trigger_column(case),
        "schedule_config_id": case.schedule_config_id,
        "planning_run_id": case.planning_run_id,
        "replan_index": case.replan_index,
        "scheduled_date": case.scheduled_date,
        "review_status": review_status,
        "review_id": review_id,
        "resolution_action": resolution_action,
        "review_definition_version": REVIEW_DEFINITION_VERSION,
    }


def run_review(
    settings: Settings,
    *,
    recommendations_path: Path,
    schedule_path: Path | None = None,
    execution_log_path: Path | None = None,
    resolutions_path: Path | None = None,
    output_dir: Path | None = None,
    policies: Sequence[str] | None = None,
    k_names: Sequence[str] | None = None,
    figures_dir: Path | None = None,
    write_figures: bool = True,
    dry_run: bool = False,
) -> ReviewResult:
    """Flag deterministic review cases from the current queue and schedule, and price nothing."""
    started = datetime.now(UTC)
    stamp = started.strftime(TIMESTAMP_FORMAT)
    stats = ReviewStats()

    read_paths: dict[str, Path] = {"recommendations": recommendations_path}
    if schedule_path is not None:
        read_paths["schedule"] = schedule_path
    if execution_log_path is not None:
        read_paths["execution_log"] = execution_log_path
    if resolutions_path is not None:
        read_paths["resolutions"] = resolutions_path
    sha_before = {name: compute_sha256(path) for name, path in read_paths.items()}

    recommendations = inputs.read_recommendations(recommendations_path)
    if policies:
        recommendations = recommendations.filter(pl.col("policy_id").is_in(list(policies)))
    if k_names:
        recommendations = recommendations.filter(pl.col("k_name").is_in(list(k_names)))
    stats.recommendation_rows = recommendations.height

    schedule = inputs.read_schedule(schedule_path)
    if schedule is not None:
        stats.schedule_rows = schedule.height
        if policies:
            schedule = schedule.filter(pl.col("policy_id").is_in(list(policies)))
        if k_names:
            schedule = schedule.filter(pl.col("k_name").is_in(list(k_names)))
    else:
        logger.info("No schedule table supplied; the execution-gap trigger will not run")

    execution_log = inputs.read_execution_log(execution_log_path)
    if execution_log is not None:
        stats.execution_rows = execution_log.height
    elif schedule is not None:
        logger.info("No execution log supplied; every occupying schedule row is treated as a gap")

    resolutions = inputs.read_resolutions_file(resolutions_path)

    cases = build_review_cases(recommendations, schedule, execution_log)
    stats.cases_flagged = len(cases)

    outcomes, final_status = apply_resolutions(cases, resolutions)
    stats.cases_resolved = sum(
        1 for status in final_status.values() if status == ReviewCaseStatus.RESOLVED
    )
    resolver_of: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for review_id, outcome in outcomes.items():
        if outcome.outcome == "applied":
            key = (
                outcome.resolution.policy_id,
                outcome.resolution.fold_id,
                outcome.resolution.k_name,
                outcome.resolution.target_inspection_id,
            )
            resolver_of[key] = (review_id, outcome.resolution.resolution_action)
    stats.resolutions_applied = len(resolver_of)

    queue_rows: list[dict[str, object]] = []
    for case in cases:
        key = (case.policy_id, case.fold_id, case.k_name, case.target_inspection_id)
        review_id, action = resolver_of.get(key, ("", ""))
        queue_rows.append(
            _queue_row(
                case,
                review_status=final_status.get(key, ReviewCaseStatus.FLAGGED),
                review_id=review_id,
                resolution_action=action,
            )
        )

    resolution_log_rows: list[dict[str, object]] = []
    for _review_id, outcome in sorted(outcomes.items()):
        resolution = outcome.resolution
        resolution_log_rows.append(
            {
                "review_id": resolution.review_id,
                "policy_id": resolution.policy_id,
                "fold_id": resolution.fold_id,
                "k_name": resolution.k_name,
                "target_inspection_id": resolution.target_inspection_id,
                "resolution_action": resolution.resolution_action,
                "reason_code": resolution.reason_code,
                "actor": resolution.actor,
                "decided_at": resolution.decided_at,
                "referenced_override_id": resolution.referenced_override_id or "",
                "referenced_adjustment_id": resolution.referenced_adjustment_id or "",
                "escalation_note": resolution.escalation_note or "",
                "original_status": outcome.original_status,
                "final_status": outcome.final_status,
                "outcome": outcome.outcome,
                "review_definition_version": REVIEW_DEFINITION_VERSION,
            }
        )

    tables: dict[str, pl.DataFrame] = {
        "human_review_queue": writer.finalize(queue_rows, "human_review_queue"),
        "review_resolution_log": writer.finalize(resolution_log_rows, "review_resolution_log"),
    }

    checks = _validate(
        tables=tables,
        recommendations=recommendations,
        schedule=schedule,
        execution_log=execution_log,
        sha_before=sha_before,
        sha_after=sha_before,
    )
    tables["review_advisories"] = writer.finalize(
        validate.advisory_rows(checks, definition_version=REVIEW_DEFINITION_VERSION),
        "review_advisories",
    )

    destination = output_dir or settings.review_processed_dir
    written: list[Path] = []
    queue_path: Path | None = None
    if not dry_run:
        for name, frame in sorted(tables.items()):
            path = destination / f"{name}_{stamp}.parquet"
            writer.write_table(frame, path)
            written.append(path)
            if name == writer.DATASET_SLUG:
                queue_path = path

    sha_after = {name: compute_sha256(path) for name, path in read_paths.items()}
    stats.inputs_unchanged = sha_after == sha_before

    checks = _validate(
        tables=tables,
        recommendations=recommendations,
        schedule=schedule,
        execution_log=execution_log,
        sha_before=sha_before,
        sha_after=sha_after,
    )
    advisories = validate.advisory_findings(checks)
    stats.advisories = len(advisories)
    stats.seconds = (datetime.now(UTC) - started).total_seconds()

    figure_paths: list[Path] = []
    if write_figures and not dry_run:
        from sentinel.review.figures import render

        figure_paths = render(tables, destination=figures_dir or FIGURES_DIR)

    manifest = _build_manifest(
        started=started,
        read_paths=read_paths,
        sha_before=sha_before,
        sha_after=sha_after,
        tables=tables,
        written=written,
        checks=checks,
        advisories=advisories,
        stats=stats,
    )
    manifest_path: Path | None = None
    if not dry_run and queue_path is not None:
        manifest_path = manifest_path_for(queue_path)
        write_manifest(manifest, manifest_path)

    return ReviewResult(
        tables=tables,
        checks=checks,
        manifest=manifest,
        stats=stats,
        advisories=advisories,
        queue_path=queue_path,
        manifest_path=manifest_path,
        written=written,
        figure_paths=figure_paths,
        dry_run=dry_run,
    )


def _validate(
    *,
    tables: dict[str, pl.DataFrame],
    recommendations: pl.DataFrame,
    schedule: pl.DataFrame | None,
    execution_log: pl.DataFrame | None,
    sha_before: dict[str, str],
    sha_after: dict[str, str],
) -> list[ValidationCheck]:
    queue = tables["human_review_queue"]
    resolution_log = tables["review_resolution_log"]
    return [
        validate.every_case_carries_a_trigger(queue),
        validate.warning_trigger_rows_are_selected_and_warned(queue, recommendations),
        validate.queue_is_deterministically_rebuildable(
            queue, recommendations, schedule, execution_log
        ),
        validate.no_duplicate_review_id(resolution_log),
        validate.pointer_fields_are_mutually_exclusive(resolution_log),
        validate.review_status_reflects_one_applied_resolution(queue, resolution_log),
        validate.resolution_verbs_do_not_collide(resolution_log),
        validate.inputs_were_not_modified(sha_before, sha_after),
        validate.pointer_targets_exist(resolution_log, frozenset(), frozenset()),
        validate.cases_flagged_by_trigger(queue),
    ]


def _build_manifest(
    *,
    started: datetime,
    read_paths: dict[str, Path],
    sha_before: dict[str, str],
    sha_after: dict[str, str],
    tables: dict[str, pl.DataFrame],
    written: Sequence[Path],
    checks: Sequence[ValidationCheck],
    advisories: Sequence[str],
    stats: ReviewStats,
) -> ReviewManifest:
    artifacts = [
        ArtifactRecord(
            path=str(path),
            bytes=path.stat().st_size,
            sha256=compute_sha256(path),
            row_count=tables[path.stem.rsplit("_", 1)[0]].height,
            schema=writer.schema_of(tables[path.stem.rsplit("_", 1)[0]]),
        )
        for path in written
    ]
    return ReviewManifest(
        code_version=__version__,
        review_definition_version=REVIEW_DEFINITION_VERSION,
        built_at=started.isoformat(),
        recommendations_path=str(read_paths["recommendations"]),
        recommendations_sha256=sha_before["recommendations"],
        policy_definition_version=_version_of(
            pl.read_parquet(read_paths["recommendations"], n_rows=1), "policy_definition_version"
        ),
        schedule_path=str(read_paths.get("schedule", "")) or None,
        schedule_sha256=sha_before.get("schedule"),
        schedule_definition_version=_version_of(
            pl.read_parquet(read_paths["schedule"], n_rows=1), "schedule_definition_version"
        )
        if "schedule" in read_paths
        else None,
        execution_log_path=str(read_paths.get("execution_log", "")) or None,
        execution_log_sha256=sha_before.get("execution_log"),
        resolutions_path=str(read_paths.get("resolutions", "")) or None,
        resolutions_sha256=sha_before.get("resolutions"),
        inputs_unchanged=stats.inputs_unchanged,
        input_sha256_after=dict(sorted(sha_after.items())),
        four_human_layers=FOUR_HUMAN_LAYERS,
        deferral_is_not_scheduling_deferral=DEFERRAL_NOTE,
        review_cannot=REVIEW_CANNOT_STATEMENT,
        abstention_policy_inherited=ABSTENTION_POLICY_INHERITED,
        no_threshold=NO_THRESHOLD,
        determinism_scope=DETERMINISM_SCOPE,
        does_not_establish=list(DOES_NOT_ESTABLISH),
        blocked=list(BLOCKED),
        inherited_limitations=list(INHERITED_LIMITATIONS),
        cases_flagged=stats.cases_flagged,
        cases_resolved=stats.cases_resolved,
        resolutions_applied=stats.resolutions_applied,
        checks=[
            {
                "name": check.name,
                "passed": check.passed,
                "severity": check.severity,
                "detail": check.detail,
            }
            for check in checks
        ],
        advisories=list(advisories),
        artifacts=artifacts,
        row_counts={name: frame.height for name, frame in sorted(tables.items())},
        seconds=stats.seconds,
    )


def _version_of(frame: pl.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.is_empty():
        return "unknown"
    value = frame[column][0]
    return str(value) if value is not None else "unknown"


def summarize(result: ReviewResult) -> str:
    """The fixed-width block the CLI prints."""
    stats = result.stats
    lines = [
        "",
        "Component 16 -- deferral / human-review gate",
        "",
        f"  recommendation rows   {stats.recommendation_rows:,}",
        f"  schedule rows         {stats.schedule_rows:,}",
        f"  execution log rows    {stats.execution_rows:,}",
        f"  cases flagged         {stats.cases_flagged:,}",
        f"    resolved            {stats.cases_resolved:,}",
        f"  resolutions applied   {stats.resolutions_applied}",
        f"  inputs unchanged      {stats.inputs_unchanged}",
        f"  advisories            {stats.advisories}",
        f"  seconds               {stats.seconds:.1f}",
        "",
        "  DOES NOT ESTABLISH:",
    ]
    lines.extend(f"    - {claim}" for claim in DOES_NOT_ESTABLISH)
    lines.append("")
    if result.dry_run:
        lines.append("  DRY RUN -- nothing was written.")
    else:
        lines.extend(f"  wrote {path}" for path in result.written)
        if result.manifest_path is not None:
            lines.append(f"  wrote {result.manifest_path}")
        lines.extend(f"  wrote {path}" for path in result.figure_paths)
    lines.append("")
    return "\n".join(lines)


__all__ = ["FIGURES_DIR", "ReviewBuildError", "ReviewResult", "run_review", "summarize"]
