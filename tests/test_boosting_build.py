"""End-to-end runs, asserted on behaviour rather than existence.

``assert output.exists()`` proves only that something was written. These tests read what
was written and check it against the contract: schemas by name *and* dtype, declared sort
keys, byte-identical repeat runs, and a manifest whose provenance actually pins the
inputs.

The run is module-scoped because fitting three models over six folds twice is the
expensive part, and every assertion below is a read of an immutable result.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from sentinel.boosting import build, validate, writer
from sentinel.boosting.definitions import BOOSTING_DEFINITION_VERSION, BOOSTING_REGISTRY
from sentinel.config import Settings
from sentinel.manifest import compute_sha256
from tests.conftest import spanning_model_features


@pytest.fixture(scope="module")
def features(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("boosting_build_input")
    path = tmp / "as_of_features_20260101T000000Z.parquet"
    spanning_model_features(days=1900).write_parquet(path)
    return path


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory, features: Path) -> build.BoostingResult:
    tmp = tmp_path_factory.mktemp("boosting_build_output")
    return build.train_boosting(Settings(data_dir=tmp), features_path=features, output_dir=tmp)


# --- 1. what gets written ------------------------------------------------------


def test_the_run_writes_three_tables_and_one_manifest(run: build.BoostingResult) -> None:
    assert run.predictions_path is not None
    assert run.manifest_path is not None
    written = sorted(p.name for p in run.predictions_path.parent.glob("*"))
    assert sum(1 for n in written if n.endswith(".parquet")) == 3
    assert sum(1 for n in written if n.startswith("manifest_")) == 1


def test_the_manifest_sits_beside_the_predictions(run: build.BoostingResult) -> None:
    assert run.manifest_path is not None
    assert run.predictions_path is not None
    assert run.manifest_path.parent == run.predictions_path.parent
    assert run.predictions_path.stem in run.manifest_path.name


@pytest.mark.parametrize(
    "table", ["boosted_predictions", "boosted_importances", "boosted_training_log"]
)
def test_each_table_matches_its_declared_schema_by_name_and_dtype(
    run: build.BoostingResult, table: str
) -> None:
    frame = run.tables[table]
    assert list(frame.columns) == list(writer.SCHEMAS[table])
    for name, dtype in writer.SCHEMAS[table].items():
        assert frame.schema[name] == dtype, f"{table}.{name}"


@pytest.mark.parametrize(
    "table", ["boosted_predictions", "boosted_importances", "boosted_training_log"]
)
def test_each_table_is_sorted_by_its_declared_key(run: build.BoostingResult, table: str) -> None:
    frame = run.tables[table]
    keys = writer.SORT_KEYS[table]
    assert frame.equals(frame.sort(keys))


def test_no_table_carries_a_timestamp_or_a_duration(run: build.BoostingResult) -> None:
    """Wall clock in a Parquet file would contradict the determinism claim."""
    for table in ("boosted_predictions", "boosted_importances", "boosted_training_log"):
        for name in run.tables[table].columns:
            assert "seconds" not in name
            assert "built_at" not in name
            assert "timestamp" not in name


# --- 2. coverage ----------------------------------------------------------------


def test_every_registered_model_appears(run: build.BoostingResult) -> None:
    names = set(run.tables["boosted_predictions"]["model_name"].unique().to_list())
    assert names == {s.name for s in BOOSTING_REGISTRY}


def test_both_fold_sets_are_scored_and_kept_separate(run: build.BoostingResult) -> None:
    """They answer different questions and must never be averaged together."""
    sets = set(run.tables["boosted_predictions"]["fold_set"].unique().to_list())
    assert sets == {"quarterly", "covid_shift"}


def test_the_importance_table_has_one_row_per_matrix_column_per_fit(
    run: build.BoostingResult,
) -> None:
    importances = run.tables["boosted_importances"]
    fits = run.tables["boosted_training_log"].height
    assert importances.height == fits * 30


def test_the_training_log_has_one_row_per_model_per_fold(run: build.BoostingResult) -> None:
    log = run.tables["boosted_training_log"]
    assert log.height == len(BOOSTING_REGISTRY) * len(run.folds)
    assert log.select(["model_name", "fold_id"]).unique().height == log.height


def test_every_training_log_row_records_the_horizon_and_the_unused_window(
    run: build.BoostingResult,
) -> None:
    log = run.tables["boosted_training_log"]
    assert (log["trained_through"] == log["train_end"]).all()
    assert (log["calibration_end_unused"] > log["trained_through"]).all()
    assert not log["early_stopped"].any()


def test_every_fit_saw_nan_cells(run: build.BoostingResult) -> None:
    """Zero would mean the NULLs were filled somewhere upstream."""
    assert (run.tables["boosted_training_log"]["train_nan_cells"] > 0).all()


def test_only_the_ablation_carries_a_weight(run: build.BoostingResult) -> None:
    log = run.tables["boosted_training_log"]
    weighted = log.filter(pl.col("class_weighted"))
    unweighted = log.filter(~pl.col("class_weighted"))
    assert set(weighted["model_name"].unique().to_list()) == {"xgboost_class_weighted"}
    assert (unweighted["scale_pos_weight"] == 1.0).all()
    assert (weighted["scale_pos_weight"] != 1.0).all()


# --- 3. validation ---------------------------------------------------------------


def test_no_error_severity_check_failed(run: build.BoostingResult) -> None:
    failures = [c for c in run.checks if not c.passed and c.severity == validate.SEVERITY_ERROR]
    assert not failures, validate.format_report(failures)


def test_the_checks_that_matter_are_present_and_passing(run: build.BoostingResult) -> None:
    """Named explicitly, so silently dropping one shows up here."""
    by_name = {c.name: c for c in run.checks}
    for name in (
        "no_preprocessing_statistics_were_fitted",
        "nulls_reached_the_estimator",
        "trained_through_is_the_training_end",
        "final_fits_did_no_early_stopping",
        "class_weighting_is_not_the_default",
        "predictions_cover_every_fold_exactly",
        "scores_are_probabilities",
    ):
        assert name in by_name, f"{name} is missing from the check set"
        assert by_name[name].passed


# --- 4. determinism ---------------------------------------------------------------


def test_two_runs_produce_identical_tables(
    tmp_path: Path, features: Path, run: build.BoostingResult
) -> None:
    second = build.train_boosting(
        Settings(data_dir=tmp_path), features_path=features, output_dir=tmp_path
    )
    for table in run.tables:
        assert run.tables[table].equals(second.tables[table]), table


def test_shuffling_the_input_file_changes_no_output(
    tmp_path: Path, features: Path, run: build.BoostingResult
) -> None:
    """The canonical training sort, exercised through the whole command."""
    shuffled_path = tmp_path / "as_of_features_20260102T000000Z.parquet"
    pl.read_parquet(features).sample(fraction=1.0, shuffle=True, seed=5).write_parquet(
        shuffled_path
    )
    second = build.train_boosting(
        Settings(data_dir=tmp_path), features_path=shuffled_path, output_dir=tmp_path
    )
    assert run.tables["boosted_predictions"].equals(second.tables["boosted_predictions"])


# --- 5. the manifest ---------------------------------------------------------------


def test_the_manifest_pins_the_feature_table_by_checksum(
    run: build.BoostingResult, features: Path
) -> None:
    assert run.manifest.features_sha256 == compute_sha256(features)
    assert run.manifest.features_path == features.name


def test_the_manifest_records_every_library_whose_version_changes_the_result(
    run: build.BoostingResult,
) -> None:
    for value in (
        run.manifest.xgboost_version,
        run.manifest.lightgbm_version,
        run.manifest.numpy_version,
    ):
        assert value and value != "not installed"


def test_the_manifest_states_the_probability_semantics(run: build.BoostingResult) -> None:
    """The single most misreadable thing this component emits."""
    assert "RAW" in run.manifest.probability_semantics
    assert "NOT a calibrated probability" in run.manifest.probability_semantics
    assert "Component 9" in run.manifest.probability_semantics


def test_the_manifest_states_that_no_preprocessing_was_fitted(
    run: build.BoostingResult,
) -> None:
    assert "no imputation" in run.manifest.preprocessing
    assert "no scaling" in run.manifest.preprocessing


def test_the_manifest_records_the_blocked_experiments(run: build.BoostingResult) -> None:
    blocked = " ".join(run.manifest.blocked)
    assert "inspector" in blocked
    assert "calibration" in blocked
    assert "SHAP" in blocked


def test_the_manifest_records_the_parameter_provenance(run: build.BoostingResult) -> None:
    """A reader must be able to tell a frozen search result from a placeholder."""
    assert run.manifest.tuned_params_provenance


def test_the_manifest_records_each_models_parameters_per_fold_set(
    run: build.BoostingResult,
) -> None:
    keys = set(run.manifest.model_params)
    assert {"xgboost/quarterly", "xgboost/covid_shift"} <= keys
    assert {"lightgbm/quarterly", "lightgbm/covid_shift"} <= keys


def test_the_manifest_is_valid_json_on_disk(run: build.BoostingResult) -> None:
    assert run.manifest_path is not None
    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert payload["component"] == "boosted_models"
    assert payload["boosting_definition_version"] == BOOSTING_DEFINITION_VERSION
    assert len(payload["artifacts"]) == 3


def test_every_artifact_record_matches_the_file_on_disk(run: build.BoostingResult) -> None:
    assert run.predictions_path is not None
    directory = run.predictions_path.parent
    for record in run.manifest.artifacts:
        path = directory / record.path
        assert path.exists()
        assert record.sha256 == compute_sha256(path)
        assert record.bytes == path.stat().st_size


# --- 6. dry run and refusals ---------------------------------------------------


def test_a_dry_run_writes_nothing(tmp_path: Path, features: Path) -> None:
    result = build.train_boosting(
        Settings(data_dir=tmp_path), features_path=features, output_dir=tmp_path, dry_run=True
    )
    assert result.predictions_path is None
    assert result.manifest_path is None
    assert not list(tmp_path.glob("*.parquet"))
    assert result.tables["boosted_predictions"].height > 0


def test_a_dry_run_still_validates(tmp_path: Path, features: Path) -> None:
    result = build.train_boosting(
        Settings(data_dir=tmp_path), features_path=features, output_dir=tmp_path, dry_run=True
    )
    assert result.checks
    assert not validate.has_failures(result.checks)


def test_an_unknown_model_is_refused(tmp_path: Path, features: Path) -> None:
    with pytest.raises(build.BoostingBuildError, match="Unknown boosted model"):
        build.train_boosting(
            Settings(data_dir=tmp_path),
            features_path=features,
            output_dir=tmp_path,
            models=["catboost"],
        )


def test_an_empty_model_list_is_refused(tmp_path: Path, features: Path) -> None:
    with pytest.raises(build.BoostingBuildError, match="no models requested"):
        build.train_boosting(
            Settings(data_dir=tmp_path), features_path=features, output_dir=tmp_path, models=[]
        )


def test_a_missing_feature_table_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build.train_boosting(Settings(data_dir=tmp_path), features_path=tmp_path / "absent.parquet")


def test_a_feature_table_with_duplicate_ids_is_refused(tmp_path: Path, features: Path) -> None:
    """The prediction contract requires one score per id and could not be met."""
    frame = pl.read_parquet(features)
    path = tmp_path / "as_of_features_20260103T000000Z.parquet"
    pl.concat([frame, frame.head(5)]).write_parquet(path)
    with pytest.raises(build.BoostingBuildError, match="duplicated target_inspection_id"):
        build.train_boosting(Settings(data_dir=tmp_path), features_path=path, output_dir=tmp_path)


# --- 7. the summary reports no metric ------------------------------------------


def test_the_summary_reports_no_metric_component_5_owns(run: build.BoostingResult) -> None:
    """Component 7 predicts; Component 5 evaluates. Two answers would be one too many."""
    out = build.summarize(run)
    assert "evaluate --predictions" in out
    for metric in ("roc_auc", "ROC-AUC", "pr_auc", "PR-AUC", "NDE", "precision@k", "days earlier"):
        assert metric not in out


def test_the_summary_says_the_probabilities_are_raw(run: build.BoostingResult) -> None:
    out = build.summarize(run)
    assert "RAW, uncalibrated" in out
    assert "Component 9" in out


def test_the_summary_names_every_blocked_experiment(run: build.BoostingResult) -> None:
    out = build.summarize(run)
    assert out.count("BLOCKED:") == len(build.BLOCKED_EXPERIMENTS)
