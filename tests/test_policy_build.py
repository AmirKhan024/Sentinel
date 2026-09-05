"""The whole component, end to end, over a synthetic snapshot.

Builds a full set of Component 4/5/8/9/12-shaped inputs on disk, runs ``decide`` over them, and
asserts the properties that only appear once every module is wired together: the manifest, the
boundary, the checksum gate, the artifact set, and the fact that the queue does not move when
the advisory inputs are withheld.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.manifest import compute_sha256
from sentinel.policy import validate as policy_validate
from sentinel.policy import writer
from sentinel.policy.build import run_policy, summarize
from sentinel.policy.definitions import (
    BASELINE_POLICY_ID,
    CANDIDATE_MODELS,
    DOES_NOT_ESTABLISH,
    K_LEVELS,
)
from tests.conftest import (
    calibrated_predictions_for,
    make_override,
    neural_categoricals_for,
    spanning_model_features,
)

MODELS = CANDIDATE_MODELS


def _evaluation_artifacts(features: pl.DataFrame, directory: Path) -> dict[str, Path]:
    """Component 5-shaped folds, simulation, metrics and sensitivity for the model rule.

    Written rather than mocked, because ``select`` reads them by column name from Parquet and a
    mock would not catch a schema drift in the artifact it actually consumes.
    """
    from sentinel.evaluation import folds as folds_module

    dated = features.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
    start = folds_module.min_date(dated, "rd")
    end = folds_module.max_date(dated, "rd")
    assert start is not None and end is not None
    specs = [
        *folds_module.quarterly_folds(data_start=start, data_end=end),
        *folds_module.covid_shift_fold(data_end=end),
    ]
    folds = pl.DataFrame(
        [
            {
                "fold_set": spec.fold_set,
                "fold_id": spec.fold_id,
                "test_start": spec.test_start,
                "test_end": spec.test_end,
                "evaluation_definition_version": "v1",
            }
            for spec in specs
        ]
    )

    # A deliberate spread so the rule has something to do: the models tie on NDE (their bands
    # overlap) and separate on ECE, which reproduces the real run's shape.
    simulation = pl.DataFrame(
        [
            {
                "model_name": model,
                "schedule_name": "model",
                "fold_set": "quarterly",
                "normalized_discovery_efficiency": 0.25 - index * 0.002,
            }
            for index, model in enumerate(MODELS)
        ]
    )
    sensitivity = pl.DataFrame(
        [
            {
                "model_name": model,
                "fold_set": "quarterly",
                "p05": 0.22,
                "p95": 0.28,
            }
            for model in MODELS
        ]
    )
    metric_rows: list[dict[str, Any]] = []
    for index, model in enumerate(MODELS):
        metric_rows.append(
            {
                "model_name": model,
                "fold_set": "quarterly",
                "metric": "ece",
                "k_name": "",
                "value": 0.05 + index * 0.01,
            }
        )
        metric_rows.append(
            {
                "model_name": model,
                "fold_set": "quarterly",
                "metric": "precision_at_k",
                "k_name": "k_1_day",
                "value": 0.6,
            }
        )

    paths = {
        "folds": directory / "evaluation_folds.parquet",
        "simulation": directory / "simulation_summary.parquet",
        "metrics": directory / "evaluation_metrics.parquet",
        "sensitivity": directory / "sensitivity.parquet",
    }
    folds.write_parquet(paths["folds"])
    simulation.write_parquet(paths["simulation"])
    pl.DataFrame(metric_rows).write_parquet(paths["metrics"])
    sensitivity.write_parquet(paths["sensitivity"])
    return paths


def _support_table(categoricals: pl.DataFrame, directory: Path) -> Path:
    """A Component 12-shaped support table marking one community area unsupported."""
    values = sorted(categoricals["community_area"].unique().to_list())
    rows = [
        {
            "group_definition": "community_area",
            "group_value": value,
            "grain": "fold_set",
            "fold_set": fold_set,
            "ranking_status": "insufficient_support" if index == 0 else "supported",
            "fairness_definition_version": "v1",
        }
        for fold_set in ("quarterly", "covid_shift")
        for index, value in enumerate(values)
    ]
    path = directory / "fairness_group_support.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("policy-inputs")
    features = spanning_model_features(days=1600, per_day=3)
    predictions = calibrated_predictions_for(features, models=MODELS)
    categoricals = neural_categoricals_for(features)

    paths: dict[str, Path] = {
        "features": directory / "as_of_features.parquet",
        "calibrated": directory / "calibrated_predictions.parquet",
        "categoricals": directory / "neural_categoricals.parquet",
    }
    features.write_parquet(paths["features"])
    predictions.write_parquet(paths["calibrated"])
    categoricals.write_parquet(paths["categoricals"])
    paths.update(_evaluation_artifacts(features, directory))
    paths["support"] = _support_table(categoricals, directory)
    return paths


def _run(settings: Settings, inputs: dict[str, Path], **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "features_path": inputs["features"],
        "calibrated_path": inputs["calibrated"],
        "folds_path": inputs["folds"],
        "simulation_path": inputs["simulation"],
        "metrics_path": inputs["metrics"],
        "sensitivity_path": inputs["sensitivity"],
        "categoricals_path": inputs["categoricals"],
        "fairness_support_path": inputs["support"],
        "write_figures": False,
        "dry_run": True,
    }
    kwargs.update(overrides)
    return run_policy(settings, **kwargs)


# --- 1. the run completes and validates -----------------------------------------


def test_a_full_run_passes_every_error_severity_check(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    result = _run(settings, inputs)
    assert not policy_validate.has_failures(result.checks)


def test_every_declared_table_is_produced(settings: Settings, inputs: dict[str, Path]) -> None:
    result = _run(settings, inputs)
    assert set(result.tables) == set(writer.SCHEMAS)


def test_the_recommendation_universe_is_policies_times_capacities_times_rows(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """The grain is uniform: every policy sees every candidate at every capacity."""
    result = _run(settings, inputs)
    recommendations = result.tables["inspection_recommendations"]
    cells = recommendations.select("policy_id", "fold_id", "k_name").unique().height
    policies = result.tables["policy_configurations"].height
    folds = result.tables["policy_coverage_eligibility"].filter(pl.col("grain") == "fold").height
    assert cells == policies * folds * len(K_LEVELS)


def test_the_comparison_covers_every_candidate_model_not_only_the_selected_one(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """The conclusion about what coverage costs must not depend on which model the rule picked."""
    result = _run(settings, inputs)
    compared = set(result.tables["policy_comparison"]["model_name"].unique().to_list())
    assert compared == set(MODELS)
    queued = set(result.tables["inspection_recommendations"]["model_name"].unique().to_list())
    assert queued == {result.stats.selected_model}


# --- 2. the model selection --------------------------------------------------------


def test_the_rule_selects_one_model_and_records_which_axis_decided(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    result = _run(settings, inputs)
    assert result.stats.selected_model in MODELS
    assert result.selection.decided_on_axis in {"nde", "ece", "precision_at_k_1_day", "model_name"}
    selected = result.tables["policy_model_selection"].filter(pl.col("is_selected"))
    assert selected.height == 1


def test_the_model_override_flag_replaces_the_rules_answer(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """A diagnostic override, recorded in the artifact rather than hidden."""
    other = next(m for m in MODELS if m != _run(settings, inputs).stats.selected_model)
    result = _run(settings, inputs, model=other)
    assert result.stats.selected_model == other


def test_an_inadmissible_model_is_refused(settings: Settings, inputs: dict[str, Path]) -> None:
    with pytest.raises(Exception, match="not an admissible candidate"):
        _run(settings, inputs, model="xgboost_chain_embeddings_platt")


# --- 3. the policies actually differ ------------------------------------------------


def test_the_baseline_grants_no_reserve_anywhere(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    result = _run(settings, inputs)
    baseline = result.tables["policy_selection_allocation"].filter(
        pl.col("policy_id") == BASELINE_POLICY_ID
    )
    assert baseline["n_reserve"].sum() == 0
    assert baseline["reserve_target"].sum() == 0


def test_a_forced_reserve_grants_slots_somewhere(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """If no policy ever allocated anything, the whole comparison would be vacuous."""
    result = _run(settings, inputs)
    forced = result.tables["policy_selection_allocation"].filter(
        pl.col("reserve_mechanism") == "forced"
    )
    assert forced["n_reserve"].sum() > 0


def test_every_reserve_selection_is_coverage_eligible(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    result = _run(settings, inputs)
    reserve = result.tables["inspection_recommendations"].filter(
        pl.col("decision_mechanism") == "coverage_reserve"
    )
    assert reserve.height > 0
    assert reserve["coverage_eligible"].all()


def test_the_opportunity_cost_of_the_baseline_is_exactly_zero(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    result = _run(settings, inputs)
    baseline = result.tables["policy_comparison"].filter(pl.col("policy_id") == BASELINE_POLICY_ID)
    assert baseline["delta_positives"].abs().max() == 0.0


# --- 4. the advisory inputs never reach the queue --------------------------------------


def test_withholding_the_group_frame_leaves_the_queue_byte_identical(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """The strongest available statement that Component 12's output is advisory only.

    Not the in-run check -- that compares two paths inside one process. This runs the whole
    component twice, once with the group artifacts and once without, and compares the ranks.
    """
    columns = ["policy_id", "fold_id", "k_name", "target_inspection_id", "final_policy_rank"]
    with_groups = _run(settings, inputs).tables["inspection_recommendations"]
    without = _run(settings, inputs, categoricals_path=None, fairness_support_path=None).tables[
        "inspection_recommendations"
    ]
    assert with_groups.select(columns).equals(without.select(columns))


def test_withholding_the_group_frame_blanks_only_the_advisory_columns(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    without = _run(settings, inputs, categoricals_path=None, fairness_support_path=None).tables[
        "inspection_recommendations"
    ]
    assert set(without["group_value"].unique().to_list()) == {""}
    assert set(without["warnings"].unique().to_list()) <= {
        "none",
        "limited_history",
        "limited_history|no_prior_inspection",
    }


# --- 5. the manifest ------------------------------------------------------------------


def test_the_manifest_carries_the_frozen_grid_and_the_selection_rule(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    manifest = _run(settings, inputs).manifest
    assert len(manifest.policy_grid) == 7
    assert manifest.selection_tie_rule
    assert manifest.selected_model in MODELS
    assert manifest.selected_model_under_discarded_band in MODELS


def test_the_manifest_carries_the_boundary_verbatim(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """It travels with the artifact precisely so it cannot be lost in a screenshot."""
    manifest = _run(settings, inputs).manifest
    assert manifest.does_not_establish == list(DOES_NOT_ESTABLISH)
    assert manifest.blocked
    assert manifest.inherited_limitations


def test_the_manifest_scopes_the_determinism_claim_to_the_inputs(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """Claiming byte-identity for something a human typed would be the easiest lie here."""
    manifest = _run(settings, inputs).manifest
    assert "override" in manifest.determinism_scope.lower()


def test_the_manifest_records_that_no_winner_was_named_when_none_was(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    result = _run(settings, inputs)
    if result.winner is None:
        assert result.manifest.no_winner_statement
    else:
        assert result.manifest.no_winner_statement is None


def test_the_inputs_are_unchanged_after_the_run(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """A pure observer of nine closed components, proved by checksum rather than promised."""
    before = {name: compute_sha256(path) for name, path in inputs.items()}
    result = _run(settings, inputs)
    after = {name: compute_sha256(path) for name, path in inputs.items()}
    assert before == after
    assert result.stats.inputs_unchanged


# --- 6. writing ------------------------------------------------------------------------


def test_a_dry_run_writes_nothing(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    destination = tmp_path / "empty"
    result = _run(settings, inputs, output_dir=destination, dry_run=True)
    assert result.written == []
    assert result.manifest_path is None
    assert not destination.exists()


def test_a_real_run_writes_every_table_and_a_manifest(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    destination = tmp_path / "written"
    result = _run(settings, inputs, output_dir=destination, dry_run=False)
    assert len(result.written) == len(writer.SCHEMAS)
    assert result.manifest_path is not None
    assert result.manifest_path.exists()
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["component"] == "decision_policy"


# --- 7. overrides end to end -------------------------------------------------------------


def test_an_override_file_is_applied_logged_and_leaves_the_queue_intact(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """The layer separation, checked through the whole component rather than in one function."""
    baseline = _run(settings, inputs)
    queue = baseline.tables["inspection_recommendations"].filter(
        (pl.col("policy_id") == BASELINE_POLICY_ID)
        & (pl.col("k_name") == "k_1_day")
        & ~pl.col("is_selected")
    )
    victim = queue.row(0, named=True)
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            [
                make_override(
                    1,
                    policy_id=BASELINE_POLICY_ID,
                    fold_id=str(victim["fold_id"]),
                    k_name="k_1_day",
                    target_inspection_id=str(victim["target_inspection_id"]),
                )
            ]
        ),
        encoding="utf-8",
    )
    result = _run(settings, inputs, overrides_path=override_path)

    log = result.tables["policy_override_log"]
    assert log.height >= 1
    assert result.stats.overrides_applied >= 1
    # The deterministic artifact is written unchanged: the original recommendation survives.
    columns = ["policy_id", "fold_id", "k_name", "target_inspection_id", "final_policy_rank"]
    assert (
        result.tables["inspection_recommendations"]
        .select(columns)
        .equals(baseline.tables["inspection_recommendations"].select(columns))
    )
    assert not policy_validate.has_failures(result.checks)


# --- 8. the summary ---------------------------------------------------------------------


def test_the_summary_prints_the_boundary_beside_the_counts(
    settings: Settings, inputs: dict[str, Path]
) -> None:
    """It must not be possible to read a run's output without the boundary."""
    text = summarize(_run(settings, inputs))
    assert "DOES NOT ESTABLISH" in text
    assert "production model" in text
    assert "POLICY WINNER" in text
    assert "DRY RUN" in text
