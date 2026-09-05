"""The seam: does Component 5 read what Component 8 writes, unchanged?

Modelled on ``test_boosting_contract.py`` and inheriting its governing principle: *a
rejection that has never been observed is indistinguishable from one that cannot happen.*
Component 5 shipped exactly that defect once (``scores_respect_the_decision_point``,
declared and unreachable, fixed in ADR 0014), so the rejections here are driven rather
than described.

The fixture runs the **real** build command end to end into a temporary directory. A
hand-assembled frame would test the schema constant rather than the pipeline, and the
question this file exists to answer is whether the artifact a real run produces is one the
evaluator accepts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.evaluation import contract
from sentinel.evaluation import folds as folds_module
from sentinel.neural import build, writer
from tests.conftest import neural_categoricals_for, spanning_model_features


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One real ``train_neural`` run, shared by every test in this module.

    Two models rather than nine, and a tiny epoch budget: the question here is the
    artifact's shape and the evaluator's acceptance of it, not the models' quality.
    """
    directory = tmp_path_factory.mktemp("neural_contract")
    # per_day=3 rather than 2 because ``covid_shift``'s training window is short: at two
    # rows a day its inner validation side falls below MIN_INNER_SPLIT_ROWS and the fold
    # is (correctly) refused. The real covid_shift fold has 12,660 training rows, so this
    # is a property of the fixture's density and not of the fold structure.
    frame = spanning_model_features(days=1600, per_day=3)

    features_path = directory / "as_of_features_20260101T000000Z.parquet"
    frame.write_parquet(features_path)

    cats_path = directory / "neural_categoricals_20260101T000000Z.parquet"
    neural_categoricals_for(frame).write_parquet(cats_path)

    settings = Settings(data_dir=directory / "data")
    result = build.train_neural(
        settings,
        features_path=features_path,
        categoricals_path=cats_path,
        output_dir=directory / "out",
        models=["neural_embeddings", "neural_numeric_only"],
        seed_sweep=False,
        render_figures=False,
    )
    return {
        "result": result,
        "predictions_path": result.predictions_path,
        "frame": frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd")),
    }


# --- 1. the evaluator reads it -----------------------------------------------


def test_component_5_reads_the_artifact_without_translation(artifact: dict[str, Any]) -> None:
    """The whole point of the slug: ``sentinel evaluate --predictions`` just works."""
    path: Path = artifact["predictions_path"]
    assert path is not None and path.exists()
    sets = contract.read_predictions(path)
    assert sets, "the evaluator found no prediction sets in the artifact"
    for prediction_set in sets:
        assert prediction_set.frame.columns == list(contract.PREDICTION_COLUMNS)
        assert prediction_set.is_probability is True


def test_every_fold_validates_against_its_own_window(artifact: dict[str, Any]) -> None:
    """The evaluator's own acceptance check, driven per fold."""
    frame = artifact["frame"]
    result = artifact["result"]
    sets = contract.read_predictions(artifact["predictions_path"])
    by_fold = {f.fold_id: f for f in result.folds}

    checked = 0
    for prediction_set in sets:
        fold = by_fold[prediction_set.fold_id]
        window = folds_module.window_frame(frame, fold)
        expected = [str(v) for v in window["target_inspection_id"].to_list()]
        contract.validate_predictions(prediction_set, fold, expected)
        checked += 1
    assert checked >= 2, f"only {checked} prediction set(s) validated; the loop is weak"


def test_trained_through_never_exceeds_the_calibration_end(artifact: dict[str, Any]) -> None:
    """The evaluator's horizon rule, and Component 8's stricter own promise."""
    result = artifact["result"]
    by_fold = {f.fold_id: f for f in result.folds}
    for model in result.fitted:
        fold = by_fold[model.fold_id]
        assert model.trained_through <= fold.calibration_end
        assert model.trained_through == fold.train_end


# --- 2. the rejections, driven ------------------------------------------------


def test_a_missing_score_is_rejected_rather_than_imputed(artifact: dict[str, Any]) -> None:
    frame = artifact["frame"]
    result = artifact["result"]
    sets = contract.read_predictions(artifact["predictions_path"])
    prediction_set = sets[0]
    fold = {f.fold_id: f for f in result.folds}[prediction_set.fold_id]
    window = folds_module.window_frame(frame, fold)
    expected = [str(v) for v in window["target_inspection_id"].to_list()]

    short = type(prediction_set)(
        frame=prediction_set.frame.head(prediction_set.frame.height - 1),
        model_name=prediction_set.model_name,
        model_version=prediction_set.model_version,
        fold_id=prediction_set.fold_id,
        trained_through=prediction_set.trained_through,
        is_probability=prediction_set.is_probability,
    )
    with pytest.raises(contract.PredictionContractError):
        contract.validate_predictions(short, fold, expected)


def test_an_extra_score_is_rejected(artifact: dict[str, Any]) -> None:
    frame = artifact["frame"]
    result = artifact["result"]
    sets = contract.read_predictions(artifact["predictions_path"])
    prediction_set = sets[0]
    fold = {f.fold_id: f for f in result.folds}[prediction_set.fold_id]
    window = folds_module.window_frame(frame, fold)
    expected = [str(v) for v in window["target_inspection_id"].to_list()]

    extra = pl.concat(
        [
            prediction_set.frame,
            pl.DataFrame({"target_inspection_id": ["NOT_A_ROW"], "score": [0.5]}),
        ]
    )
    widened = type(prediction_set)(
        frame=extra,
        model_name=prediction_set.model_name,
        model_version=prediction_set.model_version,
        fold_id=prediction_set.fold_id,
        trained_through=prediction_set.trained_through,
        is_probability=prediction_set.is_probability,
    )
    with pytest.raises(contract.PredictionContractError):
        contract.validate_predictions(widened, fold, expected)


# --- 3. the artifact's own schema ---------------------------------------------


def test_the_neural_slug_differs_from_component_6s_and_7s() -> None:
    """C6's and C7's benchmarks must stay visible; a shared slug would overwrite them."""
    from sentinel.boosting.build import PREDICTIONS_SLUG as BOOSTED_SLUG
    from sentinel.modeling.build import DATASET_SLUG as BASELINE_SLUG

    assert build.PREDICTIONS_SLUG == "neural_predictions"
    assert build.PREDICTIONS_SLUG != BASELINE_SLUG
    assert build.PREDICTIONS_SLUG != BOOSTED_SLUG


def test_the_prediction_schema_matches_components_6_and_7_column_for_column() -> None:
    """Only the trailing definition-version column may differ."""
    from sentinel.boosting.writer import PREDICTIONS_SCHEMA as BOOSTED

    neural = list(writer.PREDICTIONS_SCHEMA)
    boosted = list(BOOSTED)
    assert neural[:-1] == boosted[:-1]
    assert neural[-1] == "neural_definition_version"
    assert boosted[-1] == "boosting_definition_version"


def test_the_written_table_matches_the_declared_schema(artifact: dict[str, Any]) -> None:
    table = pl.read_parquet(artifact["predictions_path"])
    assert list(table.columns) == list(writer.PREDICTIONS_SCHEMA)
    for name, dtype in writer.PREDICTIONS_SCHEMA.items():
        assert table.schema[name] == dtype, f"{name} has dtype {table.schema[name]}"


def test_the_categorical_layer_is_written_somewhere_else(artifact: dict[str, Any]) -> None:
    """Component 8's experimental inputs must not sit beside the feature table.

    Co-location with ``features/`` is exactly the invitation to join, and a categorical
    joined onto a feature table is the change Component 4 owns. ADR 0022.
    """
    settings = Settings(data_dir=Path("data"))
    assert settings.neural_processed_dir != settings.features_processed_dir
    assert settings.neural_processed_dir.name == "neural"


def test_every_table_has_a_declared_sort_key() -> None:
    """Row order is the last place non-determinism could hide."""
    assert set(writer.SORT_KEYS) == set(writer.SCHEMAS)
    for table, keys in writer.SORT_KEYS.items():
        for key in keys:
            assert key in writer.SCHEMAS[table], f"{table} sorts on undeclared column {key}"


def test_finalize_rejects_rows_missing_a_column() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        writer.finalize([{"target_inspection_id": "T1"}], "neural_predictions")


def test_empty_returns_a_correctly_typed_zero_row_frame() -> None:
    for table in writer.SCHEMAS:
        frame = writer.empty(table)
        assert frame.height == 0
        assert list(frame.columns) == list(writer.SCHEMAS[table])


def test_an_unknown_table_is_refused() -> None:
    with pytest.raises(KeyError):
        writer.finalize([], "not_a_table")
    with pytest.raises(KeyError):
        writer.empty("not_a_table")


def test_no_table_carries_a_timestamp_or_a_duration() -> None:
    """Timings live in the manifest, beside ``built_at``, where irreproducibility lives."""
    for table, schema in writer.SCHEMAS.items():
        for column in schema:
            assert "timestamp" not in column
            assert column not in {"built_at", "seconds", "elapsed", "duration"}, (
                f"{table}.{column} would make the file non-reproducible"
            )


# --- 4. the manifest -----------------------------------------------------------


def test_the_manifest_records_the_device_and_the_determinism_caveat(
    artifact: dict[str, Any],
) -> None:
    manifest = artifact["result"].manifest
    assert manifest.device == "cpu"
    assert manifest.torch_threads == 1
    assert manifest.deterministic_algorithms is True
    assert "bit-identical" in manifest.determinism_caveat.lower()
    assert "GPU" in manifest.determinism_caveat
    assert "deliberately unused" in manifest.determinism_caveat


def test_the_manifest_states_the_probabilities_are_uncalibrated(
    artifact: dict[str, Any],
) -> None:
    manifest = artifact["result"].manifest
    assert "NOT a calibrated probability" in manifest.probability_semantics
    assert "Component 9" in manifest.probability_semantics


def test_the_manifest_explains_the_early_stopping_horizon(artifact: dict[str, Any]) -> None:
    """The one place Component 8 does something Components 6 and 7 did not."""
    manifest = artifact["result"].manifest
    assert "END OF THE TRAINING DATA" in manifest.trained_through_semantics
    assert "never the fold's calibration or test window" in manifest.trained_through_semantics


def test_the_manifest_records_the_architecture_and_the_embedding_dimensions(
    artifact: dict[str, Any],
) -> None:
    manifest = artifact["result"].manifest
    assert manifest.hidden_sizes == [256, 128]
    assert manifest.dropout == 0.3
    assert manifest.batch_size == 512
    assert manifest.optimizer == "AdamW"
    assert manifest.loss == "BCEWithLogitsLoss"
    assert manifest.embedding_dims == {
        "chain": 16,
        "facility_type": 8,
        "community_area": 8,
        "zip": 8,
    }
    assert "BatchNorm1d" in manifest.architecture


def test_the_manifest_records_what_was_refused(artifact: dict[str, Any]) -> None:
    """Blocked experiments are reported, never faked."""
    blocked = " ".join(artifact["result"].manifest.blocked)
    assert "establishment_id embedding" in blocked
    assert "inspector" in blocked
    assert "demographic" in blocked


def test_the_manifest_pins_both_inputs_by_checksum(artifact: dict[str, Any]) -> None:
    manifest = artifact["result"].manifest
    assert len(manifest.features_sha256) == 64
    assert len(manifest.categoricals_sha256) == 64
    assert manifest.feature_definition_version == "v1"
