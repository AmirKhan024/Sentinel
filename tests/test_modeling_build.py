"""End-to-end: a feature table in, prediction artifacts out.

These tests use a small synthetic table so the whole train -> predict -> validate ->
write path runs without the full 57,727-row dataset, and they assert behaviour rather
than existence. `assert output.exists()` proves only that something was written.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.evaluation.contract import read_predictions, validate_predictions
from sentinel.evaluation.folds import window_frame
from sentinel.modeling import validate as modeling_validate
from sentinel.modeling import writer
from sentinel.modeling.build import (
    BLOCKED_EXPERIMENTS,
    BaselineResult,
    BaselineTrainingError,
    summarize,
    train_baselines,
)
from sentinel.modeling.definitions import MODEL_DEFINITION_VERSION, MODEL_REGISTRY
from tests.conftest import model_feature_scenario, spanning_model_features


@pytest.fixture(scope="module")
def features_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A full-width feature table long enough to yield several quarterly folds."""
    path = tmp_path_factory.mktemp("features") / "as_of_features_20260817T000000Z.parquet"
    spanning_model_features(days=1900).write_parquet(path)
    return path


@pytest.fixture(scope="module")
def result(features_file: Path, tmp_path_factory: pytest.TempPathFactory) -> BaselineResult:
    settings = Settings(data_dir=tmp_path_factory.mktemp("data"))
    return train_baselines(settings, features_path=features_file, dry_run=True)


# --- what a run produces -----------------------------------------------------


def test_validation_passes(result: BaselineResult) -> None:
    failed = [
        c.name
        for c in result.checks
        if not c.passed and c.severity == modeling_validate.SEVERITY_ERROR
    ]
    assert failed == []


def test_every_registered_model_is_fitted_on_every_fold(result: BaselineResult) -> None:
    assert len(result.folds) >= 4
    assert len(result.fitted) == len(MODEL_REGISTRY) * len(result.folds)
    per_model = {spec.name: 0 for spec in MODEL_REGISTRY}
    for model in result.fitted:
        per_model[model.spec.name] += 1
    assert set(per_model.values()) == {len(result.folds)}


def test_one_model_per_fold_not_one_model_reused(result: BaselineResult) -> None:
    """Fold N's model is a fresh fit over the expanded window, not fold N-1's reused."""
    primary = [m for m in result.fitted if m.spec.name == "logistic_regression"]
    coefficients = {m.coefficients for m in primary}
    assert len(coefficients) == len(primary)
    row_counts = [m.train_rows for m in sorted(primary, key=lambda m: m.train_end)]
    assert row_counts == sorted(row_counts)
    assert len(set(row_counts)) == len(row_counts)


def test_the_training_window_expands_and_keeps_its_anchor(result: BaselineResult) -> None:
    primary = sorted(
        (m for m in result.fitted if m.spec.name == "logistic_regression"),
        key=lambda m: m.train_end,
    )
    assert all(m.train_start == primary[0].train_start for m in primary)
    assert [m.train_end for m in primary] == sorted(m.train_end for m in primary)


def test_predictions_cover_every_test_row_exactly_once(
    result: BaselineResult, features_file: Path
) -> None:
    frame = pl.read_parquet(features_file).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    predictions = result.tables["baseline_predictions"]
    for fold in result.folds:
        expected = set(window_frame(frame, fold)["target_inspection_id"].to_list())
        for spec in MODEL_REGISTRY:
            got = predictions.filter(
                (pl.col("fold_id") == fold.fold_id) & (pl.col("model_name") == spec.name)
            )["target_inspection_id"].to_list()
            assert set(got) == expected
            assert len(got) == len(expected)


def test_every_prediction_declares_the_training_end(result: BaselineResult) -> None:
    predictions = result.tables["baseline_predictions"]
    by_fold = {f.fold_id: f for f in result.folds}
    for row in predictions.select(["fold_id", "trained_through"]).unique().iter_rows(named=True):
        assert row["trained_through"] == by_fold[row["fold_id"]].train_end


def test_every_prediction_declares_a_probability(result: BaselineResult) -> None:
    predictions = result.tables["baseline_predictions"]
    assert predictions["is_probability"].all()
    assert predictions["is_probability"].null_count() == 0


# --- the coefficients artifact ----------------------------------------------


def test_coefficients_carry_one_row_per_term_plus_an_intercept(result: BaselineResult) -> None:
    coefficients = result.tables["baseline_coefficients"]
    for model in result.fitted:
        subset = coefficients.filter(
            (pl.col("model_name") == model.spec.name) & (pl.col("fold_id") == model.fold_id)
        )
        assert subset.height == len(model.matrix_columns) + 1
        terms = subset["term"].to_list()
        assert "__intercept__" in terms
        assert set(terms) - {"__intercept__"} == set(model.matrix_columns)


def test_coefficients_carry_the_scaler_statistics(result: BaselineResult) -> None:
    """Without these the numbers are on an unstated standardised scale."""
    coefficients = result.tables["baseline_coefficients"]
    terms = coefficients.filter(pl.col("term") != "__intercept__")
    assert terms["scaler_mean"].null_count() == 0
    assert terms["scaler_scale"].null_count() == 0
    intercept = coefficients.filter(pl.col("term") == "__intercept__")
    assert intercept["scaler_mean"].null_count() == intercept.height


def test_imputed_fill_value_is_recorded_for_nullable_terms_only(result: BaselineResult) -> None:
    coefficients = result.tables["baseline_coefficients"]
    filled = coefficients.filter(pl.col("imputed_fill_value").is_not_null())
    assert filled.height > 0
    never_null_terms = filled.filter(pl.col("term").str.starts_with("canvasses_last_"))
    assert never_null_terms.height == 0


# --- the training log --------------------------------------------------------


def test_training_log_has_one_row_per_model_per_fold(result: BaselineResult) -> None:
    log = result.tables["baseline_training_log"]
    assert log.height == len(MODEL_REGISTRY) * len(result.folds)
    assert log.select(["model_name", "fold_id"]).unique().height == log.height


def test_training_log_records_the_unused_calibration_window(result: BaselineResult) -> None:
    """Recorded to make visible that Component 6 did not use it. Component 9 will."""
    log = result.tables["baseline_training_log"]
    assert (log["calibration_end_unused"] > log["trained_through"]).all()
    assert (log["test_start"] > log["calibration_end_unused"]).all()


def test_training_log_flags_the_approximation(result: BaselineResult) -> None:
    log = result.tables["baseline_training_log"]
    flagged = set(log.filter(pl.col("is_approximation"))["model_name"].unique().to_list())
    assert flagged == {"cdph_2015_approximation"}


def test_training_log_records_convergence(result: BaselineResult) -> None:
    log = result.tables["baseline_training_log"]
    assert log["converged"].all()
    assert (log["n_iter"] < log["max_iter"]).all()


def test_no_table_carries_a_timestamp_or_duration(result: BaselineResult) -> None:
    """Wall clock in a Parquet means two identical runs produce different bytes."""
    for name, table in result.tables.items():
        for column in table.columns:
            assert "second" not in column, f"{name}.{column}"
            assert "elapsed" not in column, f"{name}.{column}"
            assert "built_at" not in column, f"{name}.{column}"


# --- the output contract -----------------------------------------------------


@pytest.mark.parametrize("table", sorted(writer.SCHEMAS))
def test_schema_matches_the_contract(result: BaselineResult, table: str) -> None:
    frame = result.tables[table]
    expected = writer.SCHEMAS[table]
    assert list(frame.columns) == list(expected)
    assert [str(d) for d in frame.dtypes] == [str(expected[c]) for c in frame.columns]


@pytest.mark.parametrize("table", sorted(writer.SCHEMAS))
def test_tables_are_sorted_by_their_declared_key(result: BaselineResult, table: str) -> None:
    frame = result.tables[table]
    assert frame.equals(frame.sort(writer.SORT_KEYS[table]))


def test_empty_returns_a_correctly_typed_frame() -> None:
    for table, schema in writer.SCHEMAS.items():
        frame = writer.empty(table)
        assert frame.height == 0
        assert list(frame.columns) == list(schema)


def test_finalize_rejects_an_unknown_table() -> None:
    with pytest.raises(KeyError, match="Unknown table"):
        writer.finalize([], "not_a_table")


def test_finalize_rejects_rows_missing_a_column() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        writer.finalize([{"model_name": "x"}], "baseline_predictions")


# --- the Component 5 seam ----------------------------------------------------


def test_the_artifact_is_accepted_by_the_component_5_contract(
    features_file: Path, tmp_path: Path
) -> None:
    """The point of the whole component: Component 5 must accept these predictions.

    Written, re-read through `read_predictions`, and offered to `validate_predictions`
    fold by fold -- exactly the path `sentinel evaluate --predictions` takes.
    """
    settings = Settings(data_dir=tmp_path / "data")
    out = tmp_path / "predictions"
    result = train_baselines(settings, features_path=features_file, output_dir=out)
    assert result.predictions_path is not None

    frame = pl.read_parquet(features_file).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    total = 0
    for fold in result.folds:
        expected = window_frame(frame, fold)["target_inspection_id"].to_list()
        sets = read_predictions(result.predictions_path, fold_id=fold.fold_id)
        assert len(sets) == len(MODEL_REGISTRY)
        for prediction_set in sets:
            validate_predictions(prediction_set, fold, expected)
            assert prediction_set.is_probability is True
            assert prediction_set.trained_through == fold.train_end
            total += 1
    assert total == len(MODEL_REGISTRY) * len(result.folds)


def test_declared_horizon_is_earlier_than_the_contract_ceiling(result: BaselineResult) -> None:
    """The contract permits calibration_end; declaring train_end is stricter. ADR 0014."""
    for model in result.fitted:
        fold = next(f for f in result.folds if f.fold_id == model.fold_id)
        assert model.trained_through == fold.train_end
        assert model.trained_through < fold.calibration_end


# --- writing and provenance --------------------------------------------------


def test_written_tables_reload_identically(features_file: Path, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    out = tmp_path / "out"
    result = train_baselines(settings, features_path=features_file, output_dir=out)
    for name, table in result.tables.items():
        written = sorted(out.glob(f"{name}_*.parquet"))
        assert len(written) == 1
        assert pl.read_parquet(written[0]).equals(table)


def test_manifest_pins_the_input_and_the_libraries(features_file: Path, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    out = tmp_path / "out"
    result = train_baselines(settings, features_path=features_file, output_dir=out)
    assert result.manifest_path is not None
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert payload["features_path"] == features_file.name
    assert len(payload["features_sha256"]) == 64
    assert payload["model_definition_version"] == MODEL_DEFINITION_VERSION
    assert payload["sklearn_version"]
    assert payload["numpy_version"]
    assert payload["blas_threads"]
    assert payload["component"] == "baseline_models"
    assert payload["fits"] == len(result.fitted)
    assert len(payload["artifacts"]) == 3
    for artifact in payload["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["row_count"] > 0


def test_manifest_records_the_approximation_note(features_file: Path, tmp_path: Path) -> None:
    """The caveat must travel with the artifact, not live only in a document."""
    settings = Settings(data_dir=tmp_path / "data")
    result = train_baselines(settings, features_path=features_file, output_dir=tmp_path / "out")
    assert result.manifest_path is not None
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    notes = payload["approximation_notes"]
    assert set(notes) == {"cdph_2015_approximation"}
    assert "APPROXIMATION" in notes["cdph_2015_approximation"]
    for topic in ("inspector", "311", "weather", "risk category"):
        assert topic in notes["cdph_2015_approximation"]


def test_manifest_records_the_blocked_experiments(result: BaselineResult) -> None:
    assert result.manifest.blocked == list(BLOCKED_EXPERIMENTS)
    joined = " ".join(result.manifest.blocked)
    assert "CDPH 2015 replication" in joined
    assert "days overdue" in joined
    assert "tuning" in joined


def test_manifest_states_the_horizon_and_score_semantics(result: BaselineResult) -> None:
    assert "train_end" in result.manifest.trained_through_semantics
    assert "higher score" in result.manifest.score_direction
    assert len(result.manifest.indicator_columns) == 4
    assert result.manifest.missing_value_rules


def test_manifest_is_written_beside_the_primary_artifact(
    features_file: Path, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    result = train_baselines(settings, features_path=features_file, output_dir=tmp_path / "out")
    assert result.predictions_path is not None
    assert result.manifest_path is not None
    assert result.manifest_path.name == f"manifest_{result.predictions_path.stem}.json"
    assert result.manifest_path.parent == result.predictions_path.parent


def test_default_destination_is_the_predictions_layer(features_file: Path, tmp_path: Path) -> None:
    """ADR 0014: a sibling of features/ and evaluation/, never inside either."""
    settings = Settings(data_dir=tmp_path / "data")
    result = train_baselines(settings, features_path=features_file)
    assert result.predictions_path is not None
    assert result.predictions_path.parent == settings.predictions_processed_dir
    assert result.predictions_path.parent.name == "predictions"


def test_dry_run_writes_nothing(features_file: Path, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    out = tmp_path / "out"
    result = train_baselines(settings, features_path=features_file, output_dir=out, dry_run=True)
    assert result.predictions_path is None
    assert result.manifest_path is None
    assert not out.exists()
    assert result.tables["baseline_predictions"].height > 0


# --- determinism -------------------------------------------------------------


def test_two_runs_produce_identical_tables(features_file: Path, tmp_path: Path) -> None:
    """Only the filename stamp may differ between runs over the same input."""
    settings = Settings(data_dir=tmp_path / "data")
    first = train_baselines(settings, features_path=features_file, dry_run=True)
    second = train_baselines(settings, features_path=features_file, dry_run=True)
    for name in writer.SCHEMAS:
        assert first.tables[name].equals(second.tables[name]), name


def test_shuffling_the_input_file_changes_no_output(tmp_path: Path) -> None:
    import random

    frame = spanning_model_features(days=1900)
    order = list(range(frame.height))
    random.Random(20260817).shuffle(order)

    ordered_path = tmp_path / "as_of_features_20260817T000000Z.parquet"
    shuffled_path = tmp_path / "as_of_features_20260817T000001Z.parquet"
    frame.write_parquet(ordered_path)
    frame[order].write_parquet(shuffled_path)

    settings = Settings(data_dir=tmp_path / "data")
    ordered = train_baselines(settings, features_path=ordered_path, dry_run=True)
    shuffled = train_baselines(settings, features_path=shuffled_path, dry_run=True)
    for name in writer.SCHEMAS:
        assert ordered.tables[name].equals(shuffled.tables[name]), name


# --- model selection ---------------------------------------------------------


def test_a_single_model_can_be_requested(features_file: Path, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    result = train_baselines(
        settings,
        features_path=features_file,
        models=["logistic_regression"],
        dry_run=True,
    )
    assert result.manifest.models == ["logistic_regression"]
    assert set(result.tables["baseline_predictions"]["model_name"].unique()) == {
        "logistic_regression"
    }


def test_an_unknown_model_fails_before_anything_is_fitted(
    features_file: Path, tmp_path: Path
) -> None:
    """A typo in --models must not quietly halve the portfolio."""
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(BaselineTrainingError, match="Unknown model"):
        train_baselines(
            settings, features_path=features_file, models=["logistic_regresion"], dry_run=True
        )


def test_an_empty_model_list_is_refused(features_file: Path, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(BaselineTrainingError, match="no models requested"):
        train_baselines(settings, features_path=features_file, models=[], dry_run=True)


# --- failure modes -----------------------------------------------------------


def test_a_missing_feature_file_raises(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(FileNotFoundError):
        train_baselines(settings, features_path=tmp_path / "absent.parquet", dry_run=True)


def test_a_table_missing_a_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "as_of_features_20260817T000000Z.parquet"
    spanning_model_features(days=1900).drop("target").write_parquet(path)
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(BaselineTrainingError, match="missing required columns"):
        train_baselines(settings, features_path=path, dry_run=True)


def test_an_empty_table_raises(tmp_path: Path) -> None:
    path = tmp_path / "as_of_features_20260817T000000Z.parquet"
    model_feature_scenario([]).write_parquet(path)
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(BaselineTrainingError, match="empty"):
        train_baselines(settings, features_path=path, dry_run=True)


def test_a_duplicated_target_id_raises(tmp_path: Path) -> None:
    """The contract requires one score per id; a duplicate could not satisfy it."""
    frame = spanning_model_features(days=1900)
    doubled = pl.concat([frame, frame.head(1)])
    path = tmp_path / "as_of_features_20260817T000000Z.parquet"
    doubled.write_parquet(path)
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(BaselineTrainingError, match="duplicated target_inspection_id"):
        train_baselines(settings, features_path=path, dry_run=True)


def test_a_span_too_short_for_a_fold_raises(tmp_path: Path) -> None:
    """Folds are never fabricated to make a run succeed."""
    path = tmp_path / "as_of_features_20260817T000000Z.parquet"
    spanning_model_features(days=200).write_parquet(path)
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(BaselineTrainingError, match="too short to build a single"):
        train_baselines(settings, features_path=path, dry_run=True)


# --- the summary -------------------------------------------------------------


def test_summary_reports_no_metric_and_points_at_the_evaluator(result: BaselineResult) -> None:
    """Component 6 must not report a number Component 5 owns."""
    text = summarize(result)
    assert "evaluate --predictions" in text
    for metric in ("roc_auc", "ROC-AUC", "precision@k", "NDE", "pr_auc"):
        assert metric not in text


def test_summary_flags_the_approximation(result: BaselineResult) -> None:
    text = summarize(result)
    assert "APPROXIMATION" in text
    assert "cdph_2015_approximation" in text
