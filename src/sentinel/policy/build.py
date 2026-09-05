"""Component 13 orchestration. The only module here that touches the filesystem or the clock.

The shape of a run:

```text
checksum the inputs
      |
      v
apply the frozen model-selection rule   (Components 5 and 9's artifacts)
      |
      v
enforce Component 5's prediction contract, fold by fold
      |
      v
for each policy x fold x capacity:  allocate -> decide -> measure
      |
      v
compare, difference against pure_risk, mark the frontier
      |
      v
annotate with warnings, apply human overrides beside the queue
      |
      v
write, re-checksum the inputs, validate, manifest
```

**Nothing is fitted, refitted, rescored or recalibrated anywhere in it.** Component 13 is a
pure observer of nine closed components: it reads their artifacts, checksums them before the
first read and again after the last write, and fails the run if a single byte moved.

**The queue is built for one model; the comparison is run for all four.** The recommendation
artifact needs a selected model, because a department cannot work four queues. But the
component's central finding -- what a coverage reserve costs -- should not depend on which
model the tie rule happened to pick, so ``policy_comparison`` is computed for every admissible
candidate and a reader can check that the conclusion survives the choice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

import sentinel.evaluation.folds as folds_module
from sentinel import __version__
from sentinel.config import Settings
from sentinel.evaluation.models import FoldSpec
from sentinel.evaluation.simulate import build_window as build_simulation_window
from sentinel.evaluation.simulate import capacity_k_values
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest
from sentinel.policy import evaluate, governance, inputs, validate, writer
from sentinel.policy.allocation import allocate, decide, model_ranks
from sentinel.policy.definitions import (
    ABSTENTION_POLICY,
    BASELINE_POLICY_ID,
    BLOCKED,
    CANDIDATE_MODELS,
    CAPACITY_SEMANTICS,
    DETERMINISM_SCOPE,
    DISCARDED_TIE_BAND,
    DOES_NOT_ESTABLISH,
    ELIGIBILITY_COLUMN,
    ELIGIBILITY_IS_NOT_GEOGRAPHY,
    ELIGIBILITY_RULE,
    ELIGIBLE_POPULATION_SHARE,
    INHERITED_LIMITATIONS,
    K_LEVELS,
    NO_WINNER_STATEMENT,
    OVERRIDE_CANNOT,
    POLICY_DEFINITION_VERSION,
    POLICY_GRID,
    POLICY_WINNER_RULE,
    PRIMARY_K_LEVEL,
    PRODUCTION_MODEL_CLAIM,
    REFUSED_MODELS,
    RESERVE_SHARES,
    SECONDARY_FLAG_COLUMN,
    SELECTION_AXES,
    SELECTION_FOLD_SET,
    SELECTION_TIE_RULE,
    PolicySpec,
    policy_for,
)
from sentinel.policy.eligibility import ELIGIBLE_FLAG
from sentinel.policy.models import (
    Allocation,
    ArtifactRecord,
    Override,
    PolicyManifest,
    PolicyStats,
    PolicyWindow,
    ValidationCheck,
)
from sentinel.policy.select import Selection, select

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Figures live with the analysis they illustrate, not in the data layer.
FIGURES_DIR = Path("docs/analysis/figures")


class PolicyBuildError(RuntimeError):
    """Raised when a policy run cannot be completed as described."""


@dataclass
class PolicyResult:
    """Everything one run produced, for the CLI, the tests and the manifest."""

    selection: Selection
    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: PolicyManifest
    stats: PolicyStats
    advisories: list[str]
    winner: str | None
    recommendations_path: Path | None = None
    manifest_path: Path | None = None
    written: list[Path] = field(default_factory=list)
    figure_paths: list[Path] = field(default_factory=list)
    dry_run: bool = False


# --- the per-model pass --------------------------------------------------------


def _capacities(features: pl.DataFrame, folds: Sequence[FoldSpec]) -> dict[str, dict[str, int]]:
    """Each fold's capacity cutoffs, from Component 5's derivation.

    Computed once from the feature table rather than per model: every calibrated model scores
    an identical id set, so the capacity of a window is a property of the window. Recomputing
    it per model would invite four answers to a question that has one.
    """
    out: dict[str, dict[str, int]] = {}
    for fold in folds:
        test = folds_module.window_frame(features, fold, date_column=inputs.DATE_COLUMN)
        if test.is_empty():
            continue
        window = build_simulation_window(
            ids=test["target_inspection_id"].to_list(),
            labels=test["target"].to_list(),
            dates=test[inputs.DATE_COLUMN].to_list(),
        )
        values = capacity_k_values(
            window, median_daily=inputs.median_daily_capacity(features, fold)
        )
        missing = [name for name in K_LEVELS if name not in values]
        if missing:
            raise PolicyBuildError(
                f"{fold.fold_id}: Component 5 derived no {', '.join(missing)} cutoff for this "
                "window. A policy cannot be reported at a capacity the window cannot support"
            )
        out[fold.fold_id] = {name: values[name] for name in K_LEVELS}
    return out


def _windows(
    features: pl.DataFrame,
    predictions: pl.DataFrame,
    folds: Sequence[FoldSpec],
    capacities: dict[str, dict[str, int]],
) -> dict[str, PolicyWindow]:
    return {
        fold.fold_id: inputs.build_window(
            features,
            predictions,
            fold,
            median_daily=inputs.median_daily_capacity(features, fold),
        )
        for fold in folds
        if fold.fold_id in capacities
    }


def _allocations(
    windows: dict[str, PolicyWindow],
    capacities: dict[str, dict[str, int]],
    specs: Sequence[PolicySpec],
) -> dict[tuple[str, str, str], Allocation]:
    """One allocation per (policy, fold, capacity)."""
    out: dict[tuple[str, str, str], Allocation] = {}
    for spec in specs:
        for fold_id, window in windows.items():
            for k_name in K_LEVELS:
                out[(spec.policy_id, fold_id, k_name)] = allocate(
                    window, spec, k_name=k_name, k=capacities[fold_id][k_name]
                )
    return out


def _comparison_rows(
    windows: dict[str, PolicyWindow],
    allocations: dict[tuple[str, str, str], Allocation],
    *,
    model_name: str,
) -> list[dict[str, object]]:
    """Metrics for every cell, each differenced against ``pure_risk`` at the same operating point.

    The baseline is looked up cell by cell rather than pooled. A difference of pooled means
    would let a large fold stand in for a small one, and the small recent folds are exactly
    where profile 7 found the coverage question actually bites.
    """
    measured = {
        key: evaluate.cell_metrics(
            windows[key[1]], allocation, definition_version=POLICY_DEFINITION_VERSION
        )
        for key, allocation in allocations.items()
    }
    rows: list[dict[str, object]] = []
    for key, cell in measured.items():
        _policy_id, fold_id, k_name = key
        baseline = measured.get((BASELINE_POLICY_ID, fold_id, k_name))
        if baseline is None:
            raise PolicyBuildError(
                f"{fold_id}/{k_name}: no {BASELINE_POLICY_ID} cell to difference against. Every "
                "opportunity cost in this component is relative to it"
            )
        row = dict(cell)
        row["model_name"] = model_name
        row.update(evaluate.opportunity_cost(cell, baseline))
        rows.append(row)
    return rows


def _allocation_rows(
    allocations: dict[tuple[str, str, str], Allocation], *, model_name: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (policy_id, _fold_id, _k_name), allocation in allocations.items():
        spec = policy_for(policy_id)
        rows.append(
            {
                "policy_id": policy_id,
                "model_name": model_name,
                "fold_set": allocation.fold_set,
                "fold_id": allocation.fold_id,
                "k_name": allocation.k_name,
                "k": allocation.k,
                "n_universe": allocation.n_universe,
                "reserve_mechanism": str(spec.mechanism),
                "reserve_share": spec.reserve_share,
                "reserve_target": allocation.reserve_target,
                "n_eligible_available": allocation.n_eligible_available,
                "n_eligible_in_risk_top_k": allocation.n_eligible_in_risk_top_k,
                "n_risk": allocation.n_risk,
                "n_reserve": allocation.n_reserve,
                "n_selected": allocation.n_selected,
                "reserve_inert": allocation.reserve_inert,
                "policy_definition_version": POLICY_DEFINITION_VERSION,
            }
        )
    return rows


def _recommendation_frame(
    window: PolicyWindow,
    allocation: Allocation,
    *,
    model_name: str,
    establishments: dict[str, str],
    group_labels: dict[str, str],
    group_support: dict[str, str],
) -> pl.DataFrame:
    """One cell's recommendations, built columnwise.

    Columnwise rather than row-by-row because the universe grain is 1.45 million rows across a
    full run, and a Python dict per row would cost more memory than the entire artifact.
    """
    mechanisms, reasons, ranks = decide(window, allocation)
    ranking = model_ranks(window)
    groups = [group_labels.get(row_id, "") for row_id in window.ids]
    statuses = [group_support.get(value, "") if value else "" for value in groups]
    warnings = [
        governance.warnings_for(
            eligible=window.eligible[i],
            secondary_no_history=window.secondary_no_history[i],
            group_value=groups[i] or None,
            group_status=statuses[i] or None,
        )
        for i in range(window.n)
    ]
    height = window.n
    return pl.DataFrame(
        {
            "policy_id": [allocation.policy_id] * height,
            "model_name": [model_name] * height,
            "fold_set": [allocation.fold_set] * height,
            "fold_id": [allocation.fold_id] * height,
            "k_name": [allocation.k_name] * height,
            "k": [allocation.k] * height,
            "target_inspection_id": list(window.ids),
            "establishment_id": [establishments.get(row_id, "") for row_id in window.ids],
            "inspection_date": list(window.dates),
            "base_score": list(window.base_scores),
            "score": list(window.scores),
            "model_rank": list(ranking),
            "final_policy_rank": list(ranks),
            "is_selected": [rank is not None for rank in ranks],
            "decision_mechanism": list(mechanisms),
            "decision_reason": list(reasons),
            "coverage_eligible": list(window.eligible),
            "secondary_no_history": list(window.secondary_no_history),
            "warnings": warnings,
            "group_value": groups,
            "group_status": statuses,
            "policy_definition_version": [POLICY_DEFINITION_VERSION] * height,
        },
        schema=writer.SCHEMAS["inspection_recommendations"],
    )


#: The columns ``warnings_do_not_change_the_queue`` compares. Everything that would differ if
#: a Component 12 signal had reached an allocation decision, and nothing that would not.
SIGNATURE_COLUMNS: tuple[str, ...] = (
    "policy_id",
    "model_name",
    "fold_set",
    "fold_id",
    "k_name",
    "target_inspection_id",
    "final_policy_rank",
    "decision_mechanism",
)


def _queue_signature(
    window: PolicyWindow, allocation: Allocation, *, model_name: str
) -> pl.DataFrame:
    """The queue rebuilt with every warning input withheld, reduced to the columns that matter.

    Built by re-running the same allocation and decision path with no group label and no
    support status in scope, then compared against the real artifact. That comparison is what
    turns "the audit informs governance but never scoring" from a claim in a docstring into a
    check that goes red.

    Only the identifying columns and the two outputs a leak would move are kept. Materialising
    a second full universe-grained artifact to compare two columns would cost more memory than
    the artifact being validated.
    """
    mechanisms, _reasons, ranks = decide(window, allocation)
    height = window.n
    return pl.DataFrame(
        {
            "policy_id": [allocation.policy_id] * height,
            "model_name": [model_name] * height,
            "fold_set": [allocation.fold_set] * height,
            "fold_id": [allocation.fold_id] * height,
            "k_name": [allocation.k_name] * height,
            "target_inspection_id": list(window.ids),
            "final_policy_rank": list(ranks),
            "decision_mechanism": list(mechanisms),
        },
        schema={
            "policy_id": pl.Utf8(),
            "model_name": pl.Utf8(),
            "fold_set": pl.Utf8(),
            "fold_id": pl.Utf8(),
            "k_name": pl.Utf8(),
            "target_inspection_id": pl.Utf8(),
            "final_policy_rank": pl.Int64(),
            "decision_mechanism": pl.Utf8(),
        },
    )


def _decision_reason_rows(recommendations: pl.DataFrame) -> list[dict[str, object]]:
    """The distribution of *why*, which is what a reviewer reads before any single row."""
    if recommendations.is_empty():
        return []
    grouped = (
        recommendations.group_by(
            "policy_id",
            "model_name",
            "fold_set",
            "fold_id",
            "k_name",
            "decision_mechanism",
            "decision_reason",
            "warnings",
        )
        .len()
        .rename({"len": "n_rows"})
    )
    return [
        {**row, "policy_definition_version": POLICY_DEFINITION_VERSION}
        for row in grouped.iter_rows(named=True)
    ]


# --- orchestration --------------------------------------------------------------


def run_policy(
    settings: Settings,
    *,
    features_path: Path,
    calibrated_path: Path,
    folds_path: Path,
    simulation_path: Path,
    metrics_path: Path,
    sensitivity_path: Path,
    categoricals_path: Path | None = None,
    fairness_support_path: Path | None = None,
    overrides_path: Path | None = None,
    output_dir: Path | None = None,
    policies: Sequence[str] | None = None,
    model: str | None = None,
    figures_dir: Path | None = None,
    write_figures: bool = True,
    dry_run: bool = False,
) -> PolicyResult:
    """Turn calibrated predictions into a capacity-constrained recommendation, and price it."""
    started = datetime.now(UTC)
    stamp = started.strftime(TIMESTAMP_FORMAT)
    stats = PolicyStats()

    read_paths: dict[str, Path] = {
        "features": features_path,
        "calibrated_predictions": calibrated_path,
        "evaluation_folds": folds_path,
        "simulation_summary": simulation_path,
        "evaluation_metrics": metrics_path,
        "sensitivity": sensitivity_path,
    }
    if categoricals_path is not None:
        read_paths["categoricals"] = categoricals_path
    if fairness_support_path is not None:
        read_paths["fairness_support"] = fairness_support_path
    if overrides_path is not None:
        read_paths["overrides"] = overrides_path
    sha_before = {name: compute_sha256(path) for name, path in read_paths.items()}

    specs = _resolve_policies(policies)
    features = inputs.load_features(features_path)
    folds = inputs.load_folds(features, folds_path)
    capacities = _capacities(features, folds)
    stats.folds = len(capacities)
    stats.fold_sets = sorted({fold.fold_set for fold in folds if fold.fold_id in capacities})
    stats.policies = len(specs)
    stats.eligible_rows = int(features[ELIGIBLE_FLAG].sum())

    # --- 1. which model does the policy carry? ---------------------------------
    selection = select(
        simulation=pl.read_parquet(simulation_path),
        metrics=pl.read_parquet(metrics_path),
        sensitivity=pl.read_parquet(sensitivity_path),
        definition_version=POLICY_DEFINITION_VERSION,
    )
    selected_model = model or selection.model_name
    if model is not None and model not in CANDIDATE_MODELS:
        raise PolicyBuildError(
            f"{model!r} is not an admissible candidate. Admissible: "
            f"{', '.join(CANDIDATE_MODELS)}. Refused: {', '.join(REFUSED_MODELS)}"
        )
    stats.selected_model = selected_model
    stats.selection_axis = selection.decided_on_axis
    logger.info(
        "Production model %s, decided on axis %s (%d of %d candidates tied on NDE)",
        selected_model,
        selection.decided_on_axis,
        selection.n_tied_on_nde,
        len(CANDIDATE_MODELS),
    )

    # --- 2. the comparison, for every candidate --------------------------------
    group_labels = inputs.load_group_labels(categoricals_path)
    establishments = inputs.establishment_ids(features)
    comparison_rows: list[dict[str, object]] = []
    frontier_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    recommendation_frames: list[pl.DataFrame] = []
    withheld_frames: list[pl.DataFrame] = []
    group_rows: list[dict[str, object]] = []
    override_rows: list[dict[str, object]] = []

    overrides = inputs.read_override_file(overrides_path)
    support_by_fold_set = {
        fold_set: inputs.load_group_support(fairness_support_path, fold_set=fold_set)
        for fold_set in stats.fold_sets
    }

    for candidate in CANDIDATE_MODELS:
        predictions = inputs.load_predictions(calibrated_path, model_name=candidate)
        stats.prediction_rows += predictions.height
        inputs.enforce_prediction_contract(calibrated_path, folds, features, model_name=candidate)
        windows = _windows(features, predictions, folds, capacities)
        allocations = _allocations(windows, capacities, specs)
        comparison_rows.extend(_comparison_rows(windows, allocations, model_name=candidate))
        allocation_rows.extend(_allocation_rows(allocations, model_name=candidate))

        if candidate != selected_model:
            continue

        stats.universe_rows = sum(window.n for window in windows.values())
        recommendation_frames, withheld_frames, group_rows, override_rows = _production_pass(
            windows=windows,
            allocations=allocations,
            model_name=candidate,
            establishments=establishments,
            group_labels=group_labels,
            support_by_fold_set=support_by_fold_set,
            overrides=overrides,
            stats=stats,
        )

    for candidate in CANDIDATE_MODELS:
        for fold_set in stats.fold_sets:
            frontier_rows.extend(
                {**row, "model_name": candidate}
                for row in evaluate.frontier(
                    [r for r in comparison_rows if r["model_name"] == candidate],
                    fold_set=fold_set,
                    definition_version=POLICY_DEFINITION_VERSION,
                )
            )

    # --- 3. eligibility summary ------------------------------------------------
    eligibility_rows = _eligibility_rows(features, folds, capacities)

    # --- 4. assemble -----------------------------------------------------------
    recommendations = (
        pl.concat(recommendation_frames)
        if recommendation_frames
        else writer.empty("inspection_recommendations")
    )
    withheld = (
        pl.concat(withheld_frames)
        if withheld_frames
        else recommendations.select(list(SIGNATURE_COLUMNS))
    )
    stats.queue_rows = (
        int(recommendations["is_selected"].sum()) if not recommendations.is_empty() else 0
    )
    stats.reserve_rows = (
        recommendations.filter(pl.col("decision_mechanism") == "coverage_reserve").height
        if not recommendations.is_empty()
        else 0
    )
    stats.overrides_applied = sum(
        1 for row in override_rows if row.get("outcome") == governance.OUTCOME_APPLIED
    )

    tables: dict[str, pl.DataFrame] = {
        "policy_configurations": writer.finalize(
            _configuration_rows(specs), "policy_configurations"
        ),
        "policy_model_selection": writer.finalize(
            [dict(row) for row in selection.rows], "policy_model_selection"
        ),
        "policy_coverage_eligibility": writer.finalize(
            eligibility_rows, "policy_coverage_eligibility"
        ),
        "inspection_recommendations": recommendations.sort(
            writer.SORT_KEYS["inspection_recommendations"]
        ),
        "policy_selection_allocation": writer.finalize(
            allocation_rows, "policy_selection_allocation"
        ),
        "policy_comparison": writer.finalize(comparison_rows, "policy_comparison"),
        "policy_frontier": writer.finalize(frontier_rows, "policy_frontier"),
        "policy_group_audit": writer.finalize(group_rows, "policy_group_audit"),
        "policy_decision_reasons": writer.finalize(
            _decision_reason_rows(recommendations), "policy_decision_reasons"
        ),
        "policy_override_log": writer.finalize(override_rows, "policy_override_log"),
    }
    stats.inert_cells = int(tables["policy_selection_allocation"]["reserve_inert"].sum())

    winner = evaluate.winner(
        tables["policy_frontier"]
        .filter(
            (pl.col("model_name") == selected_model) & (pl.col("fold_set") == SELECTION_FOLD_SET)
        )
        .to_dicts(),
        k_name=PRIMARY_K_LEVEL,
    )

    # Validated twice, deliberately. The advisory table is an artifact, so it has to exist
    # before anything is written; but `inputs_were_not_modified` can only be answered after
    # the last write. The first pass compares the input checksums to themselves -- which is
    # honest, because at that point nothing has been written -- and the second pass, below,
    # is the one whose result reaches the manifest and the exit code. No advisory check reads
    # a checksum, so the two passes always agree on the advisory set.
    checks = _validate(
        tables=tables,
        features=features,
        withheld=withheld,
        selected_model=selected_model,
        support=support_by_fold_set.get(SELECTION_FOLD_SET, {}),
        winner=winner,
        sha_before=sha_before,
        sha_after=sha_before,
    )
    tables["policy_advisories"] = writer.finalize(
        validate.advisory_rows(checks, definition_version=POLICY_DEFINITION_VERSION),
        "policy_advisories",
    )

    # --- 5. write --------------------------------------------------------------
    destination = output_dir or settings.policy_processed_dir
    written: list[Path] = []
    recommendations_path: Path | None = None
    if not dry_run:
        for name, frame in sorted(tables.items()):
            path = destination / f"{name}_{stamp}.parquet"
            writer.write_table(frame, path)
            written.append(path)
            if name == writer.DATASET_SLUG:
                recommendations_path = path

    sha_after = {name: compute_sha256(path) for name, path in read_paths.items()}
    stats.inputs_unchanged = sha_after == sha_before

    checks = _validate(
        tables=tables,
        features=features,
        withheld=withheld,
        selected_model=selected_model,
        support=support_by_fold_set.get(SELECTION_FOLD_SET, {}),
        winner=winner,
        sha_before=sha_before,
        sha_after=sha_after,
    )
    advisories = validate.advisory_findings(checks)
    stats.advisories = len(advisories)
    stats.seconds = (datetime.now(UTC) - started).total_seconds()

    figure_paths: list[Path] = []
    if write_figures and not dry_run:
        from sentinel.policy.figures import render

        figure_paths = render(tables, destination=figures_dir or FIGURES_DIR)

    manifest = _build_manifest(
        started=started,
        read_paths=read_paths,
        sha_before=sha_before,
        sha_after=sha_after,
        features=features,
        selection=selection,
        selected_model=selected_model,
        winner=winner,
        tables=tables,
        written=written,
        checks=checks,
        advisories=advisories,
        stats=stats,
    )
    manifest_path: Path | None = None
    if not dry_run and recommendations_path is not None:
        manifest_path = manifest_path_for(recommendations_path)
        write_manifest(manifest, manifest_path)

    return PolicyResult(
        selection=selection,
        tables=tables,
        checks=checks,
        manifest=manifest,
        stats=stats,
        advisories=advisories,
        winner=winner,
        recommendations_path=recommendations_path,
        manifest_path=manifest_path,
        written=written,
        figure_paths=figure_paths,
        dry_run=dry_run,
    )


def _production_pass(
    *,
    windows: dict[str, PolicyWindow],
    allocations: dict[tuple[str, str, str], Allocation],
    model_name: str,
    establishments: dict[str, str],
    group_labels: dict[str, str],
    support_by_fold_set: dict[str, dict[str, str]],
    overrides: Sequence[Override],
    stats: PolicyStats,
) -> tuple[
    list[pl.DataFrame], list[pl.DataFrame], list[dict[str, object]], list[dict[str, object]]
]:
    """Build the recommendation artifact for the selected model, twice.

    The second build withholds every warning input -- no group label, no support status -- and
    is never written. It exists so ``warnings_do_not_change_the_queue`` can compare the two and
    turn "the audit informs governance but not scoring" from a claim into a measurement.
    """
    recommendation_frames: list[pl.DataFrame] = []
    withheld_frames: list[pl.DataFrame] = []
    group_rows: list[dict[str, object]] = []
    override_rows: list[dict[str, object]] = []

    for key, allocation in allocations.items():
        window = windows[key[1]]
        support = support_by_fold_set.get(allocation.fold_set, {})
        recommendation_frames.append(
            _recommendation_frame(
                window,
                allocation,
                model_name=model_name,
                establishments=establishments,
                group_labels=group_labels,
                group_support=support,
            )
        )
        withheld_frames.append(_queue_signature(window, allocation, model_name=model_name))
        if group_labels:
            group_rows.extend(
                evaluate.group_audit(
                    window,
                    allocation,
                    groups=[
                        group_labels.get(row_id, governance.UNKNOWN_GROUP) for row_id in window.ids
                    ],
                    support=support,
                    model_name=model_name,
                    definition_version=POLICY_DEFINITION_VERSION,
                )
            )
        if overrides:
            mechanisms, reasons, ranks = decide(window, allocation)
            log, _final = governance.apply_overrides(
                window,
                allocation,
                overrides,
                mechanisms=mechanisms,
                reasons=reasons,
                ranks=ranks,
                definition_version=POLICY_DEFINITION_VERSION,
            )
            override_rows.extend(log)

    return recommendation_frames, withheld_frames, group_rows, override_rows


def _resolve_policies(names: Sequence[str] | None) -> list[PolicySpec]:
    """The requested policies, always including the baseline every cost is measured against."""
    if names is None:
        return list(POLICY_GRID)
    wanted = {policy_for(name).policy_id for name in names}
    wanted.add(BASELINE_POLICY_ID)
    return [spec for spec in POLICY_GRID if spec.policy_id in wanted]


def _configuration_rows(specs: Sequence[PolicySpec]) -> list[dict[str, object]]:
    return [
        {
            "policy_id": spec.policy_id,
            "reserve_mechanism": str(spec.mechanism),
            "reserve_share": spec.reserve_share,
            "is_baseline": spec.policy_id == BASELINE_POLICY_ID,
            "rationale": spec.rationale,
            "policy_definition_version": POLICY_DEFINITION_VERSION,
        }
        for spec in specs
    ]


def _eligibility_rows(
    features: pl.DataFrame, folds: Sequence[FoldSpec], capacities: dict[str, dict[str, int]]
) -> list[dict[str, object]]:
    """Eligibility at the fold grain and pooled per fold set."""
    from sentinel.policy import eligibility

    rows: list[dict[str, object]] = []
    per_set: dict[str, list[pl.DataFrame]] = {}
    for fold in folds:
        if fold.fold_id not in capacities:
            continue
        window = folds_module.window_frame(features, fold, date_column=inputs.DATE_COLUMN)
        rows.append(
            eligibility.summarize(
                window,
                grain="fold",
                fold_set=fold.fold_set,
                fold_id=fold.fold_id,
                definition_version=POLICY_DEFINITION_VERSION,
            )
        )
        per_set.setdefault(fold.fold_set, []).append(window)
    for fold_set, frames in sorted(per_set.items()):
        rows.append(
            eligibility.summarize(
                pl.concat(frames),
                grain="fold_set",
                fold_set=fold_set,
                fold_id="",
                definition_version=POLICY_DEFINITION_VERSION,
            )
        )
    return rows


def _validate(
    *,
    tables: dict[str, pl.DataFrame],
    features: pl.DataFrame,
    withheld: pl.DataFrame,
    selected_model: str,
    support: dict[str, str],
    winner: str | None,
    sha_before: dict[str, str],
    sha_after: dict[str, str],
) -> list[ValidationCheck]:
    """Every check, wired in one place so the list is readable as a list."""
    recommendations = tables["inspection_recommendations"]
    allocation = tables["policy_selection_allocation"].filter(
        pl.col("model_name") == selected_model
    )
    sortable = {name: frame for name, frame in tables.items() if name in writer.SORT_KEYS}
    return [
        validate.tables_are_deterministically_sorted(sortable, writer.SORT_KEYS),
        validate.recommendations_cover_the_universe(recommendations, allocation),
        validate.selected_counts_equal_capacity(recommendations, allocation),
        validate.allocations_are_internally_consistent(tables["policy_selection_allocation"]),
        validate.no_establishment_is_selected_twice(recommendations),
        validate.every_row_declares_a_valid_mechanism(recommendations),
        validate.reserve_rows_are_eligible(recommendations),
        validate.risk_rows_satisfy_the_risk_contract(recommendations, allocation),
        validate.policy_ranks_are_unique_and_contiguous(recommendations, allocation),
        validate.eligibility_matches_the_declared_rule(
            features, column=ELIGIBILITY_COLUMN, flag=ELIGIBLE_FLAG
        ),
        validate.no_outcome_column_reaches_the_policy(
            recommendations, tables["policy_selection_allocation"]
        ),
        validate.warnings_do_not_change_the_queue(recommendations, withheld),
        validate.configurations_match_the_frozen_grid(tables["policy_configurations"]),
        validate.comparison_covers_every_policy(
            tables["policy_comparison"], tables["policy_selection_allocation"]
        ),
        validate.unsupported_groups_are_preserved(tables["policy_group_audit"], support),
        validate.overrides_are_fully_attributed(tables["policy_override_log"]),
        validate.overrides_left_the_deterministic_queue_intact(
            recommendations, tables["policy_override_log"]
        ),
        validate.inputs_were_not_modified(sha_before, sha_after),
        validate.reserve_is_not_inert(tables["policy_selection_allocation"]),
        validate.coverage_is_not_free(tables["policy_comparison"]),
        validate.group_representation_is_stable(tables["policy_group_audit"]),
        validate.a_winner_was_determined(winner, NO_WINNER_STATEMENT),
    ]


def _build_manifest(
    *,
    started: datetime,
    read_paths: dict[str, Path],
    sha_before: dict[str, str],
    sha_after: dict[str, str],
    features: pl.DataFrame,
    selection: Selection,
    selected_model: str,
    winner: str | None,
    tables: dict[str, pl.DataFrame],
    written: Sequence[Path],
    checks: Sequence[ValidationCheck],
    advisories: Sequence[str],
    stats: PolicyStats,
) -> PolicyManifest:
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
    return PolicyManifest(
        code_version=__version__,
        policy_definition_version=POLICY_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=str(read_paths["features"]),
        features_sha256=sha_before["features"],
        feature_definition_version=_version_of(features, "feature_definition_version"),
        calibrated_predictions_path=str(read_paths["calibrated_predictions"]),
        calibrated_predictions_sha256=sha_before["calibrated_predictions"],
        calibration_definition_version=_version_of(
            pl.read_parquet(read_paths["calibrated_predictions"], n_rows=1),
            "calibration_definition_version",
        ),
        evaluation_folds_path=str(read_paths["evaluation_folds"]),
        evaluation_folds_sha256=sha_before["evaluation_folds"],
        evaluation_definition_version=_version_of(
            pl.read_parquet(read_paths["evaluation_folds"], n_rows=1),
            "evaluation_definition_version",
        ),
        simulation_summary_path=str(read_paths["simulation_summary"]),
        simulation_summary_sha256=sha_before["simulation_summary"],
        evaluation_metrics_path=str(read_paths["evaluation_metrics"]),
        evaluation_metrics_sha256=sha_before["evaluation_metrics"],
        sensitivity_path=str(read_paths["sensitivity"]),
        sensitivity_sha256=sha_before["sensitivity"],
        fairness_support_path=str(read_paths.get("fairness_support", "")) or None,
        fairness_support_sha256=sha_before.get("fairness_support"),
        categoricals_path=str(read_paths.get("categoricals", "")) or None,
        categoricals_sha256=sha_before.get("categoricals"),
        overrides_path=str(read_paths.get("overrides", "")) or None,
        overrides_sha256=sha_before.get("overrides"),
        inputs_unchanged=stats.inputs_unchanged,
        input_sha256_after=dict(sorted(sha_after.items())),
        eligibility_column=ELIGIBILITY_COLUMN,
        eligibility_rule=ELIGIBILITY_RULE,
        eligibility_is_not_geography=ELIGIBILITY_IS_NOT_GEOGRAPHY,
        secondary_flag_column=SECONDARY_FLAG_COLUMN,
        eligible_population_share=ELIGIBLE_POPULATION_SHARE,
        capacity_semantics=CAPACITY_SEMANTICS,
        k_levels=list(K_LEVELS),
        primary_k_level=PRIMARY_K_LEVEL,
        reserve_shares=list(RESERVE_SHARES),
        policy_grid=[
            {
                "policy_id": spec.policy_id,
                "reserve_mechanism": str(spec.mechanism),
                "reserve_share": spec.reserve_share,
            }
            for spec in POLICY_GRID
        ],
        candidate_models=list(CANDIDATE_MODELS),
        refused_models=list(REFUSED_MODELS),
        selection_axes=[list(axis) for axis in SELECTION_AXES],
        selection_tie_rule=SELECTION_TIE_RULE,
        selection_fold_set=SELECTION_FOLD_SET,
        discarded_tie_band=DISCARDED_TIE_BAND,
        selected_model=selected_model,
        selection_decided_on_axis=selection.decided_on_axis,
        selected_model_under_discarded_band=selection.under_discarded_band,
        production_model_claim=PRODUCTION_MODEL_CLAIM,
        policy_winner=winner,
        policy_winner_rule=POLICY_WINNER_RULE,
        no_winner_statement=None if winner else NO_WINNER_STATEMENT,
        abstention_policy=ABSTENTION_POLICY,
        override_cannot=OVERRIDE_CANNOT,
        determinism_scope=DETERMINISM_SCOPE,
        overrides_applied=stats.overrides_applied,
        does_not_establish=list(DOES_NOT_ESTABLISH),
        blocked=list(BLOCKED),
        inherited_limitations=list(INHERITED_LIMITATIONS),
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


def summarize(result: PolicyResult) -> str:
    """The fixed-width block the CLI prints. The boundary is printed with the counts."""
    stats = result.stats
    lines = [
        "",
        "Component 13 -- decision policy and deployment governance",
        "",
        f"  production model      {stats.selected_model}",
        f"    decided on axis     {result.selection.decided_on_axis} "
        f"({result.selection.n_tied_on_nde} of {len(CANDIDATE_MODELS)} tied on NDE)",
        f"    discarded-band pick {result.selection.under_discarded_band}",
        f"  policies compared     {stats.policies}",
        f"  folds                 {stats.folds} ({', '.join(stats.fold_sets)})",
        f"  prediction universe   {stats.universe_rows:,} scored rows",
        f"  coverage-eligible     {stats.eligible_rows:,} feature rows",
        f"  recommendations       {result.tables['inspection_recommendations'].height:,} rows",
        f"    selected            {stats.queue_rows:,}",
        f"    via reserve         {stats.reserve_rows:,}",
        f"  inert reserve cells   {stats.inert_cells:,}",
        f"  overrides applied     {stats.overrides_applied}",
        f"  inputs unchanged      {stats.inputs_unchanged}",
        f"  advisories            {stats.advisories}",
        f"  seconds               {stats.seconds:.1f}",
        "",
        f"  POLICY WINNER: {result.winner or NO_WINNER_STATEMENT}",
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


__all__ = ["FIGURES_DIR", "PolicyBuildError", "PolicyResult", "run_policy", "summarize"]
