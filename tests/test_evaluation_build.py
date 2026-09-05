"""End-to-end: one feature table in, six evaluation tables and a manifest out.

The orchestration tests. Where ``test_evaluation_simulate.py`` proves the
arithmetic and ``test_evaluation_leakage.py`` proves the split, this file proves
the run holds together: schemas are the declared ones, the manifest pins its
inputs and its seeds, a dry run writes nothing, and two runs over the same data
produce identical tables.

Determinism is the property worth insisting on. The only randomness in the
component is the seeded random reference schedule and the seeded label re-draw;
everything else is structural. If two runs ever differ, some ordering has become
dependent on Parquet row order.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.evaluation import validate as evaluation_validate
from sentinel.evaluation.build import (
    BLOCKED_EXPERIMENTS,
    DEFAULT_MODELS,
    ESTIMAND,
    EvaluationError,
    run_evaluation,
    summarize,
)
from sentinel.evaluation.writer import SCHEMAS
from tests.conftest import feature_scenario, spanning_features

SMALL_SEEDS = 3
SMALL_REPLICATIONS = 5


@pytest.fixture
def features(tmp_path: Path) -> Path:
    path = tmp_path / "as_of_features_20260816T150313Z.parquet"
    spanning_features(days=1800, per_day=2).write_parquet(path)
    return path


def _run(settings: Settings, features: Path, **overrides: object):
    kwargs: dict[str, object] = {
        "features_path": features,
        "dry_run": True,
        "random_seeds": SMALL_SEEDS,
        "sensitivity_replications": SMALL_REPLICATIONS,
    }
    kwargs.update(overrides)
    return run_evaluation(settings, **kwargs)  # type: ignore[arg-type]


# --- 1. the run produces what it says ---------------------------------------


def test_a_run_produces_every_declared_table(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    assert set(result.tables) == set(SCHEMAS)


def test_every_table_matches_its_declared_schema(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    for name, table in result.tables.items():
        assert list(table.columns) == list(SCHEMAS[name]), name
        for column, dtype in SCHEMAS[name].items():
            assert table.schema[column] == dtype, f"{name}.{column}"


def test_folds_are_built_from_the_data_and_the_covid_fold_is_separate(
    settings: Settings, features: Path
) -> None:
    result = _run(settings, features)
    sets = set(result.tables["evaluation_folds"]["fold_set"].to_list())
    assert "quarterly" in sets
    assert result.manifest.folds == len(result.folds)


def test_every_default_model_is_scored(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    scored = set(result.tables["evaluation_metrics"]["model_name"].to_list())
    assert scored == set(DEFAULT_MODELS)


def test_the_five_reference_schedules_all_appear(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    schedules = set(result.tables["simulation_summary"]["schedule_name"].to_list())
    assert schedules == {"optimal", "worst", "business_as_usual", "random", "model"}


def test_no_probability_metric_is_emitted_for_a_ranking_only_baseline(
    settings: Settings, features: Path
) -> None:
    """None of these producers claims to emit a probability, so a Brier score
    for any of them would be a fabrication. The rows must simply not exist."""
    result = _run(settings, features)
    kinds = set(result.tables["evaluation_metrics"]["metric_kind"].to_list())
    metrics = set(result.tables["evaluation_metrics"]["metric"].to_list())
    assert kinds == {"ranking"}
    assert metrics.isdisjoint({"brier", "log_loss", "ece", "mce"})


def test_k_values_are_derived_from_measured_capacity(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    at_k = result.tables["evaluation_metrics"].filter(pl.col("metric") == "precision_at_k")
    assert at_k.height > 0
    assert at_k["k"].min() >= 1  # type: ignore[operator]
    assert "k_1_day" in set(at_k["k_name"].to_list())


# --- 2. the analytic bounds survive the full pipeline ----------------------


def test_optimal_and_worst_bracket_every_other_schedule(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    sim = result.tables["simulation_summary"].drop_nulls("normalized_discovery_efficiency")
    optimal = sim.filter(pl.col("schedule_name") == "optimal")
    worst = sim.filter(pl.col("schedule_name") == "worst")
    assert optimal["normalized_discovery_efficiency"].min() == pytest.approx(1.0, abs=1e-9)
    assert worst["normalized_discovery_efficiency"].max() == pytest.approx(-1.0, abs=1e-9)
    assert sim["normalized_discovery_efficiency"].max() <= 1.0 + 1e-9  # type: ignore[operator]
    assert sim["normalized_discovery_efficiency"].min() >= -1.0 - 1e-9  # type: ignore[operator]


def test_business_as_usual_moves_nobody(settings: Settings, features: Path) -> None:
    """The identity, asserted end to end rather than on a toy window."""
    result = _run(settings, features)
    bau = result.tables["simulation_summary"].filter(pl.col("schedule_name") == "business_as_usual")
    assert bau.height > 0
    assert bau["mean_days_earlier"].drop_nulls().to_list() == pytest.approx(
        [0.0] * bau["mean_days_earlier"].drop_nulls().len()
    )


def test_the_discovery_curve_ends_at_every_positive(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    curves = result.tables["discovery_curves"]
    last = curves.group_by(["fold_id", "schedule_name", "model_name", "seed"]).agg(
        pl.col("cumulative_positive_fraction").max().alias("final")
    )
    assert last["final"].max() == pytest.approx(1.0)


# --- 3. determinism ---------------------------------------------------------


def test_two_runs_over_the_same_data_produce_identical_tables(
    settings: Settings, features: Path
) -> None:
    first = _run(settings, features)
    second = _run(settings, features)
    for name in SCHEMAS:
        assert first.tables[name].equals(second.tables[name]), name


def test_shuffling_the_input_rows_changes_nothing(settings: Settings, tmp_path: Path) -> None:
    """Parquet row order is not a contract, so no result may depend on it."""
    frame = spanning_features(days=1800, per_day=2)
    ordered = tmp_path / "ordered.parquet"
    shuffled = tmp_path / "shuffled.parquet"
    frame.write_parquet(ordered)
    frame.sample(fraction=1.0, shuffle=True, seed=20260816).write_parquet(shuffled)

    a = _run(settings, ordered)
    b = _run(settings, shuffled)
    for name in SCHEMAS:
        assert a.tables[name].equals(b.tables[name]), name


def test_every_random_seed_is_recorded_in_the_manifest(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    assert len(result.manifest.random_seeds) == SMALL_SEEDS
    assert result.manifest.sensitivity_replications == SMALL_REPLICATIONS
    assert isinstance(result.manifest.sensitivity_seed, int)


# --- 4. writing and the manifest --------------------------------------------


def test_a_dry_run_writes_nothing(settings: Settings, features: Path, tmp_path: Path) -> None:
    destination = tmp_path / "out"
    result = _run(settings, features, output_dir=destination, dry_run=True)
    assert result.folds_path is None
    assert result.manifest_path is None
    assert not destination.exists()


def test_a_real_run_writes_every_table_and_one_manifest(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "out"
    result = _run(settings, features, output_dir=destination, dry_run=False)

    written = sorted(p.name for p in destination.glob("*.parquet"))
    assert len(written) == len(SCHEMAS)
    assert result.folds_path is not None and result.folds_path.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert result.manifest_path.name.startswith("manifest_evaluation_folds_")
    assert len(list(destination.glob("manifest_*.json"))) == 1


def test_the_manifest_pins_its_input_and_states_its_estimand(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "out"
    result = _run(settings, features, output_dir=destination, dry_run=False)
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]

    assert payload["features_path"] == features.name
    assert len(payload["features_sha256"]) == 64
    assert payload["estimand"] == ESTIMAND
    assert "re-ordering" in payload["estimand"].lower()
    assert "not a causal estimate" in payload["estimand"]
    assert payload["component"] == "temporal_evaluation"


def test_the_manifest_records_every_artifact_with_a_checksum(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "out"
    result = _run(settings, features, output_dir=destination, dry_run=False)
    assert len(result.manifest.artifacts) == len(SCHEMAS)
    for artifact in result.manifest.artifacts:
        assert len(artifact.sha256) == 64
        assert artifact.bytes > 0
        assert (destination / artifact.path).exists()


def test_blocked_experiments_are_named_in_the_manifest(settings: Settings, features: Path) -> None:
    """A blocked experiment is reported, never quietly omitted."""
    result = _run(settings, features)
    assert result.manifest.blocked == list(BLOCKED_EXPERIMENTS)
    assert any("NOAA" in item for item in result.manifest.blocked)
    assert any("Component 6" in item for item in result.manifest.blocked)


def test_the_excluded_partial_window_is_named(settings: Settings, features: Path) -> None:
    result = _run(settings, features)
    assert result.manifest.excluded_partial_windows


# --- 5. folds-only mode -----------------------------------------------------


def test_folds_only_emits_the_split_and_nothing_else(settings: Settings, features: Path) -> None:
    result = _run(settings, features, folds_only=True)
    assert result.tables["evaluation_folds"].height > 0
    for name in ("evaluation_metrics", "discovery_curves", "simulation_summary"):
        assert result.tables[name].height == 0
        assert list(result.tables[name].columns) == list(SCHEMAS[name])


def test_folds_only_still_runs_the_leakage_checks(settings: Settings, features: Path) -> None:
    result = _run(settings, features, folds_only=True)
    names = {c.name for c in result.checks}
    assert "test_is_isolated" in names
    assert not evaluation_validate.has_failures(result.checks)


# --- 6. failure modes -------------------------------------------------------


def test_a_missing_feature_table_is_reported_clearly(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        run_evaluation(settings, features_path=tmp_path / "absent.parquet")


def test_a_feature_table_missing_a_required_column_is_rejected(
    settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "bad.parquet"
    spanning_features(days=100).drop("target").write_parquet(path)
    with pytest.raises(EvaluationError, match="missing required columns"):
        run_evaluation(settings, features_path=path, dry_run=True)


def test_too_little_data_is_refused_rather_than_producing_a_fabricated_fold(
    settings: Settings, tmp_path: Path
) -> None:
    """Folds are never invented to fill a report."""
    path = tmp_path / "short.parquet"
    spanning_features(days=200).write_parquet(path)
    with pytest.raises(EvaluationError, match="never fabricated"):
        run_evaluation(settings, features_path=path, dry_run=True)


def test_a_feature_table_with_no_rows_is_refused(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    feature_scenario([]).write_parquet(path)
    with pytest.raises(EvaluationError):
        run_evaluation(settings, features_path=path, dry_run=True)


def test_a_window_in_which_every_row_is_positive_yields_no_efficiency(
    settings: Settings, tmp_path: Path
) -> None:
    """Undefined rather than misleading: with nothing to separate, every
    schedule is identical."""
    path = tmp_path / "all_positive.parquet"
    frame = spanning_features(days=1800, per_day=2, positive_every=1)
    frame.write_parquet(path)
    result = _run(settings, path)
    sim = result.tables["simulation_summary"]
    assert sim["normalized_discovery_efficiency"].null_count() == sim.height


def test_a_window_with_no_positives_yields_no_efficiency(
    settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "none_positive.parquet"
    frame = spanning_features(days=1800, per_day=2).with_columns(
        pl.lit(0, dtype=pl.Int8).alias("target")
    )
    frame.write_parquet(path)
    result = _run(settings, path)
    sim = result.tables["simulation_summary"]
    assert sim["normalized_discovery_efficiency"].null_count() == sim.height


# --- 7. the summary ---------------------------------------------------------


def test_the_summary_states_the_estimand_and_the_blocked_work(
    settings: Settings, features: Path
) -> None:
    text = summarize(_run(settings, features))
    assert "estimand" in text
    assert "re-ordering only" in text
    assert "BLOCKED" in text
    assert "folds:" in text


def test_the_summary_names_the_written_paths_after_a_real_run(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    result = _run(settings, features, output_dir=tmp_path / "out", dry_run=False)
    text = summarize(result)
    assert "manifest:" in text
    assert str(result.folds_path) in text
