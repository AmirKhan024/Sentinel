"""Loading the authoritative artifacts, and assembling the queues a schedule is built from.

The one module here that touches Parquet on the way in. It reads Components 5 and 13's outputs
and produces nothing but typed structures, so every later module can be pure.

**This component fits nothing, scores nothing and re-derives nothing it can read.** Component
13's queue arrives as an artifact and is checked rather than recomputed; a second implementation
of the allocator here would be a second answer to a question that already has an authoritative
one, and the two would eventually disagree.

**The input contract is enforced on the way in, not assumed.** A scheduling layer is exactly
the place where a quietly incomplete queue would go unnoticed -- a policy artifact that dropped
rows would produce a shorter queue, a fuller horizon and a better utilisation number, all for a
reason that has nothing to do with scheduling. So the ranks are checked for uniqueness and
contiguity, the selected count is checked against ``k``, the mechanisms and reasons are checked
against Component 13's own frozen vocabulary, and any failure refuses the run.

**The fold cross-check is the most important one here.** Every horizon in this component
descends from ``test_median_daily_capacity``, and Component 13's ``k_1_day`` is that same
number. If the two disagree, the fold table and the policy artifact describe different
snapshots, and every horizon built from them would be silently wrong. That is checked before
anything is scheduled.

**The override log is read as evidence, never as an instruction.** Component 13 owns what an
override does; this module reads the log only to stamp ``recommendation_override_id`` onto the
schedule row, so the chain from a human decision to a planned date stays traceable. It never
changes a rank.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

from sentinel.policy.definitions import MECHANISM_REASONS, POLICY_BY_ID
from sentinel.scheduling.adjustments import parse_adjustments
from sentinel.scheduling.execution import parse_execution_events
from sentinel.scheduling.models import Adjustment, ExecutionEvent, QueueRow

#: Columns the schedule cannot be built without. A missing one refuses the run rather than
#: producing a plan with a blank provenance column.
REQUIRED_RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    "policy_id",
    "model_name",
    "fold_set",
    "fold_id",
    "k_name",
    "k",
    "target_inspection_id",
    "establishment_id",
    "inspection_date",
    "model_rank",
    "final_policy_rank",
    "is_selected",
    "decision_mechanism",
    "decision_reason",
    "coverage_eligible",
    "score",
    "base_score",
    "warnings",
    "policy_definition_version",
)

#: Columns that must never appear in a scheduling input. The outcome is the one thing this
#: component must not see: a scheduler that could read the label could order by it.
FORBIDDEN_COLUMNS: tuple[str, ...] = ("target", "target_status")

#: The cell key. One plan per combination of these.
CELL_COLUMNS: tuple[str, ...] = ("policy_id", "model_name", "fold_set", "fold_id", "k_name")


class ScheduleInputError(ValueError):
    """Raised when an input artifact cannot be trusted enough to build a schedule from."""


def read_recommendations(path: Path) -> pl.DataFrame:
    """Component 13's recommendation universe."""
    if not path.exists():
        raise FileNotFoundError(f"Recommendation artifact not found: {path}")
    frame = pl.read_parquet(path)
    missing = [c for c in REQUIRED_RECOMMENDATION_COLUMNS if c not in frame.columns]
    if missing:
        raise ScheduleInputError(
            f"{path.name}: missing {', '.join(missing)}. The scheduling layer needs Component "
            "13's full provenance, not only the selection flag"
        )
    present = [c for c in FORBIDDEN_COLUMNS if c in frame.columns]
    if present:
        raise ScheduleInputError(
            f"{path.name}: carries outcome column(s) {', '.join(present)}. No outcome may reach "
            "a scheduling artifact; a scheduler that could read the label could order by it"
        )
    return frame


def validate_recommendations(frame: pl.DataFrame, *, source: str) -> None:
    """The authoritative scheduling input contract. Refuses the whole run on any breach.

    Every one of these would produce a plausible schedule from a broken queue. A duplicate
    decision key would schedule one inspection twice; a gapped rank would make the placement
    order ambiguous; a selected count below ``k`` would leave slots idle and report a shortfall
    that never existed.
    """
    models = frame["model_name"].unique().to_list()
    if len(models) != 1:
        raise ScheduleInputError(
            f"{source}: carries {len(models)} models ({', '.join(sorted(map(str, models[:5])))}). "
            "Component 13 emits the selected production model, and a schedule over two models "
            "would silently be two schedules"
        )

    unknown_policies = sorted(set(frame["policy_id"].to_list()) - set(POLICY_BY_ID))
    if unknown_policies:
        raise ScheduleInputError(
            f"{source}: unknown policy_id(s) {', '.join(unknown_policies)}. The scheduling "
            "layer never invents a policy it has no definition for"
        )

    if frame["inspection_date"].null_count():
        raise ScheduleInputError(
            f"{source}: {frame['inspection_date'].null_count()} row(s) have no inspection_date. "
            "The operating calendar is derived from that column, so a null would drop a real "
            "operating day or invent a missing one"
        )

    duplicates = (
        frame.group_by([*CELL_COLUMNS, "target_inspection_id"]).len().filter(pl.col("len") > 1)
    )
    if duplicates.height:
        offenders = duplicates.head(3)["target_inspection_id"].to_list()
        raise ScheduleInputError(
            f"{source}: {duplicates.height} duplicate decision key(s), first "
            f"{', '.join(map(str, offenders))}. One row per scored inspection per cell"
        )

    selected = frame.filter(pl.col("is_selected"))
    if selected.is_empty():
        raise ScheduleInputError(f"{source}: no row is selected. There is no queue to schedule")

    if selected["final_policy_rank"].null_count():
        raise ScheduleInputError(
            f"{source}: a selected row carries no final_policy_rank. The rank is the only "
            "ordering key this component has"
        )
    stray = frame.filter(~pl.col("is_selected") & pl.col("final_policy_rank").is_not_null())
    if stray.height:
        raise ScheduleInputError(
            f"{source}: {stray.height} unselected row(s) carry a final_policy_rank. A rank on a "
            "row nobody selected is a queue position for an inspection nobody approved"
        )

    counts = selected.group_by(list(CELL_COLUMNS)).agg(
        pl.len().alias("n_selected"),
        pl.col("k").first().alias("k"),
        pl.col("final_policy_rank").min().alias("lo"),
        pl.col("final_policy_rank").max().alias("hi"),
        pl.col("final_policy_rank").n_unique().alias("n_ranks"),
    )
    wrong = counts.filter(pl.col("n_selected") != pl.col("k"))
    if wrong.height:
        row = wrong.row(0, named=True)
        raise ScheduleInputError(
            f"{source}: {wrong.height} cell(s) select a number of rows other than k, first "
            f"{row['policy_id']}/{row['fold_id']}/{row['k_name']}: {row['n_selected']} of "
            f"{row['k']}"
        )
    gapped = counts.filter(
        (pl.col("lo") != 1)
        | (pl.col("hi") != pl.col("n_selected"))
        | (pl.col("n_ranks") != pl.col("n_selected"))
    )
    if gapped.height:
        row = gapped.row(0, named=True)
        raise ScheduleInputError(
            f"{source}: {gapped.height} cell(s) have ranks that are not unique and contiguous "
            f"from 1, first {row['policy_id']}/{row['fold_id']}/{row['k_name']} "
            f"({row['lo']}..{row['hi']} over {row['n_ranks']} distinct)"
        )

    mechanisms = set(frame["decision_mechanism"].unique().to_list())
    unknown = sorted(mechanisms - set(MECHANISM_REASONS))
    if unknown:
        raise ScheduleInputError(f"{source}: unknown decision_mechanism(s) {', '.join(unknown)}")
    pairs = frame.select("decision_mechanism", "decision_reason").unique()
    for mechanism, reason in pairs.iter_rows():
        allowed = MECHANISM_REASONS.get(str(mechanism), frozenset())
        if str(reason) not in allowed:
            raise ScheduleInputError(
                f"{source}: mechanism {mechanism!r} carries reason {reason!r}, which Component "
                "13's own contract forbids"
            )


def read_folds(path: Path) -> pl.DataFrame:
    """Component 5's fold table, for ``test_median_daily_capacity``."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation fold table not found: {path}")
    frame = pl.read_parquet(path)
    required = (
        "fold_set",
        "fold_id",
        "test_median_daily_capacity",
        "evaluation_definition_version",
    )
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ScheduleInputError(f"{path.name}: missing {', '.join(missing)}")
    return frame


def median_daily_by_fold(folds: pl.DataFrame) -> dict[str, int]:
    """The measured median daily rate per fold, floored at one."""
    return {
        str(row["fold_id"]): max(1, int(row["test_median_daily_capacity"] or 1))
        for row in folds.iter_rows(named=True)
    }


def validate_folds_against_recommendations(
    recommendations: pl.DataFrame, medians: dict[str, int], *, source: str
) -> None:
    """Check that the fold table and the policy artifact describe the same snapshot.

    ``k_1_day`` **is** ``test_median_daily_capacity`` -- that is how
    ``evaluation.simulate.capacity_k_values`` produced it. If the two disagree, one of the two
    artifacts has moved, and every horizon derived from the pair would be built from a rate the
    queue was not cut at. This is the check that stops a stale fold table producing a plausible
    schedule for the wrong calendar.
    """
    day = (
        recommendations.filter(pl.col("k_name") == "k_1_day")
        .group_by("fold_id")
        .agg(pl.col("k").first())
    )
    mismatched: list[str] = []
    for row in day.iter_rows(named=True):
        fold_id = str(row["fold_id"])
        if fold_id not in medians:
            mismatched.append(f"{fold_id} (absent from the fold table)")
        elif int(row["k"]) != medians[fold_id]:
            mismatched.append(f"{fold_id} (k_1_day={row['k']}, median={medians[fold_id]})")
    if mismatched:
        raise ScheduleInputError(
            f"{source}: the fold table and the recommendation artifact disagree about daily "
            f"capacity in {len(mismatched)} fold(s): {', '.join(mismatched[:5])}. They describe "
            "different snapshots, and every horizon built from the pair would be wrong"
        )


def read_override_log(path: Path | None) -> dict[str, str]:
    """Component 13's override log, read as provenance evidence only.

    Returns a mapping from scored inspection to the override that touched it, so the schedule
    row can name it. Nothing here changes a rank, a mechanism or a selection: Component 13 owns
    what an override does, and this component only records that one happened.
    """
    if path is None or not path.exists():
        return {}
    frame = pl.read_parquet(path)
    if frame.is_empty() or "override_id" not in frame.columns:
        return {}
    applied = frame.filter(pl.col("outcome") == "applied")
    if applied.is_empty():
        return {}
    return dict(
        zip(
            applied["target_inspection_id"].to_list(),
            applied["override_id"].to_list(),
            strict=True,
        )
    )


def _read_json_list(path: Path, *, what: str) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScheduleInputError(f"{path.name}: not valid JSON -- {exc}") from exc
    if not isinstance(payload, list):
        raise ScheduleInputError(
            f"{path.name}: the {what} contract is a JSON list of objects, got "
            f"{type(payload).__name__}"
        )
    return payload


def read_adjustments(path: Path | None) -> list[Adjustment]:
    """Decode and validate a scheduling adjustment file, or return nothing."""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Adjustment file not found: {path}")
    return parse_adjustments(_read_json_list(path, what="scheduling adjustment"))


def read_execution_events(path: Path | None) -> list[ExecutionEvent]:
    """Decode and validate an execution event file, or return nothing."""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Execution file not found: {path}")
    return parse_execution_events(_read_json_list(path, what="execution event"))


def queue_rows(cell: pl.DataFrame) -> tuple[QueueRow, ...]:
    """One cell's approved queue, as typed rows carrying Component 13's provenance verbatim.

    Returned in ``final_policy_rank`` order. The allocator re-sorts anyway -- it must, because
    it cannot assume its caller sorted -- but handing it an ordered sequence keeps the two
    consistent for anyone reading a test.
    """
    ordered = cell.filter(pl.col("is_selected")).sort("final_policy_rank", "target_inspection_id")
    return tuple(
        QueueRow(
            target_inspection_id=str(row["target_inspection_id"]),
            establishment_id=str(row["establishment_id"]),
            final_policy_rank=int(row["final_policy_rank"]),
            model_rank=int(row["model_rank"]),
            decision_mechanism=str(row["decision_mechanism"]),
            decision_reason=str(row["decision_reason"]),
            score=float(row["score"]),
            base_score=float(row["base_score"]),
            coverage_eligible=bool(row["coverage_eligible"]),
            warnings=str(row["warnings"]),
            recommendation_date=row["inspection_date"],
        )
        for row in ordered.iter_rows(named=True)
    )


def observed_calendars(recommendations: pl.DataFrame) -> dict[str, tuple[tuple[date, int], ...]]:
    """Each fold's operating days and observed volumes, derived once.

    Derived once per fold rather than per cell, and this matters: the calendar is
    policy-independent and capacity-independent, so re-deriving it inside the 630-cell loop
    would recompute one measurement 35 times per fold for no reason.

    The universe is taken from a single (policy, capacity) slice because Component 13 writes the
    whole prediction universe into every cell -- so any one slice is the universe, and taking
    all of them would count every operating day 35 times over.
    """
    first_policy = sorted(recommendations["policy_id"].unique().to_list())[0]
    first_k = sorted(recommendations["k_name"].unique().to_list())[0]
    slice_ = recommendations.filter(
        (pl.col("policy_id") == first_policy) & (pl.col("k_name") == first_k)
    )
    grouped = slice_.group_by("fold_id", "inspection_date").len().sort("fold_id", "inspection_date")
    out: dict[str, list[tuple[date, int]]] = {}
    for row in grouped.iter_rows(named=True):
        out.setdefault(str(row["fold_id"]), []).append((row["inspection_date"], int(row["len"])))
    return {fold_id: tuple(days) for fold_id, days in out.items()}


def cell_frames(
    recommendations: pl.DataFrame,
    *,
    policies: Sequence[str] | None = None,
    k_names: Sequence[str] | None = None,
) -> list[tuple[dict[str, str], pl.DataFrame]]:
    """Every (policy, model, fold set, fold, capacity) cell, in a deterministic order."""
    frame = recommendations
    if policies:
        frame = frame.filter(pl.col("policy_id").is_in(list(policies)))
    if k_names:
        frame = frame.filter(pl.col("k_name").is_in(list(k_names)))
    if frame.is_empty():
        raise ScheduleInputError("no cell matches the requested policies and capacity levels")
    keys = frame.select(CELL_COLUMNS).unique().sort(list(CELL_COLUMNS))
    out: list[tuple[dict[str, str], pl.DataFrame]] = []
    for row in keys.iter_rows(named=True):
        mask = pl.lit(True)
        for column in CELL_COLUMNS:
            mask = mask & (pl.col(column) == row[column])
        out.append(({k: str(v) for k, v in row.items()}, frame.filter(mask)))
    return out


__all__ = [
    "CELL_COLUMNS",
    "FORBIDDEN_COLUMNS",
    "REQUIRED_RECOMMENDATION_COLUMNS",
    "ScheduleInputError",
    "cell_frames",
    "median_daily_by_fold",
    "observed_calendars",
    "queue_rows",
    "read_adjustments",
    "read_execution_events",
    "read_folds",
    "read_override_log",
    "read_recommendations",
    "validate_folds_against_recommendations",
    "validate_recommendations",
]
