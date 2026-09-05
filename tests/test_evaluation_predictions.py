"""The Component 5 / Component 6 seam: evaluating an external prediction artifact.

`contract.read_predictions` and `PREDICTION_METADATA_COLUMNS` shipped with Component 5
and were unwired until Component 6 existed. This file covers the wiring.

The first test is the one that matters most. `--predictions` is strictly additive, so
omitting it must reproduce the previous behaviour *exactly* rather than approximately --
proved by comparing tables, not argued in a commit message.

The second most important is the id-alignment test. A persisted artifact carries its
producer's row order, not the evaluation window's, so zipping the two positionally would
attach every score to the wrong establishment while producing a perfectly plausible
number. That failure has no symptom, so it needs a test.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.evaluation import validate as evaluation_validate
from sentinel.evaluation.build import DEFAULT_MODELS, run_evaluation
from sentinel.evaluation.contract import PredictionContractError, PredictionHorizonError
from sentinel.evaluation.folds import (
    covid_shift_fold,
    max_date,
    min_date,
    quarterly_folds,
    window_frame,
)
from sentinel.evaluation.writer import SCHEMAS
from tests.conftest import spanning_features

SMALL_SEEDS = 3
SMALL_REPLICATIONS = 5

EXTERNAL_MODEL = "logistic_regression"


@pytest.fixture
def features(tmp_path: Path) -> Path:
    path = tmp_path / "as_of_features_20260816T150313Z.parquet"
    spanning_features(days=1800, per_day=2).write_parquet(path)
    return path


def _run(settings: Settings, features: Path, **overrides: object):  # type: ignore[no-untyped-def]
    kwargs: dict[str, object] = {
        "features_path": features,
        "dry_run": True,
        "random_seeds": SMALL_SEEDS,
        "sensitivity_replications": SMALL_REPLICATIONS,
    }
    kwargs.update(overrides)
    return run_evaluation(settings, **kwargs)  # type: ignore[arg-type]


def _artifact(
    path: Path,
    features: Path,
    *,
    model_name: str = EXTERNAL_MODEL,
    is_probability: bool = True,
    trained_through_offset: int = 0,
    folds_to_write: int | None = None,
    reverse_rows: bool = False,
    drop_rows: int = 0,
) -> Path:
    """A minimal Component 6-shaped prediction file over the fixture's folds.

    Scores are a deterministic function of the label plus an id-derived jitter, so the
    model has real signal without being perfectly separable.
    """
    frame = pl.read_parquet(features).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    start, end = min_date(frame, "rd"), max_date(frame, "rd")
    assert start is not None and end is not None
    folds = [*quarterly_folds(data_start=start, data_end=end), *covid_shift_fold(data_end=end)]
    if folds_to_write is not None:
        folds = folds[:folds_to_write]

    rows: list[dict[str, object]] = []
    for fold in folds:
        window = window_frame(frame, fold)
        ids = window["target_inspection_id"].to_list()
        labels = window["target"].to_list()
        for row_id, label in zip(ids, labels, strict=True):
            jitter = (int(row_id) % 97) / 1000.0
            score = 0.35 + 0.3 * float(label) + jitter
            rows.append(
                {
                    "target_inspection_id": row_id,
                    "score": min(max(score, 0.001), 0.999),
                    "model_name": model_name,
                    "model_version": "v1",
                    "fold_id": fold.fold_id,
                    "trained_through": fold.train_end + timedelta(days=trained_through_offset),
                    "is_probability": is_probability,
                }
            )
    table = pl.DataFrame(
        rows,
        schema={
            "target_inspection_id": pl.Utf8,
            "score": pl.Float64,
            "model_name": pl.Utf8,
            "model_version": pl.Utf8,
            "fold_id": pl.Utf8,
            "trained_through": pl.Date,
            "is_probability": pl.Boolean,
        },
    )
    if reverse_rows:
        table = table.reverse()
    if drop_rows:
        table = table.head(table.height - drop_rows)
    table.write_parquet(path)
    return path


# --- 1. the flag is additive -------------------------------------------------


def test_omitting_the_flag_reproduces_the_previous_behaviour(
    settings: Settings, features: Path
) -> None:
    without = _run(settings, features)
    explicit_none = _run(settings, features, predictions_path=None)
    for name in SCHEMAS:
        assert without.tables[name].equals(explicit_none.tables[name]), name
    assert without.manifest.models == explicit_none.manifest.models
    assert without.manifest.predictions_path is None
    assert without.manifest.predictions_sha256 is None


def test_the_built_in_baselines_are_unchanged_by_an_external_artifact(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Adding a model must not perturb the numbers reported for the heuristics."""
    baseline = _run(settings, features)
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    combined = _run(settings, features, predictions_path=artifact)

    # Both sides are filtered identically. `simulation_summary` also carries the
    # `optimal` and `worst` reference schedules under `model_name`, and those are not in
    # DEFAULT_MODELS -- filtering only one side would compare different row sets.
    for name in ("evaluation_metrics", "simulation_summary", "sensitivity"):
        keep = pl.col("model_name").is_in(list(DEFAULT_MODELS))
        before = baseline.tables[name].filter(keep)
        after = combined.tables[name].filter(keep)
        assert before.height > 0, name
        assert before.sort(before.columns).equals(after.sort(after.columns)), name


# --- 2. alignment by id, never by row order ---------------------------------


def test_scores_are_aligned_by_id_not_by_row_order(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Same predictions, opposite row order, identical metrics required."""
    forward = _artifact(tmp_path / "forward.parquet", features)
    backward = _artifact(tmp_path / "reversed.parquet", features, reverse_rows=True)
    first = _run(settings, features, predictions_path=forward)
    second = _run(settings, features, predictions_path=backward)
    for name in SCHEMAS:
        assert first.tables[name].equals(second.tables[name]), name


def test_a_model_with_real_signal_beats_random_on_roc_auc(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """A weak end-to-end sanity check on alignment: a score built from the label must
    land above 0.5. If alignment were broken it would land at roughly 0.5."""
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    auc = result.tables["evaluation_metrics"].filter(
        (pl.col("model_name") == EXTERNAL_MODEL) & (pl.col("metric") == "roc_auc")
    )
    assert auc.height > 0
    assert auc["value"].min() > 0.5


# --- 3. the external model is measured like any other -----------------------


def test_an_external_model_is_scored_alongside_the_baselines(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    scored = set(result.tables["evaluation_metrics"]["model_name"].unique().to_list())
    assert EXTERNAL_MODEL in scored
    assert set(DEFAULT_MODELS) <= scored
    assert EXTERNAL_MODEL in result.manifest.models


def test_an_external_model_is_measured_on_every_fold(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Including covid_shift, which read_predictions would skip silently if absent."""
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    metrics = result.tables["evaluation_metrics"].filter(pl.col("model_name") == EXTERNAL_MODEL)
    assert set(metrics["fold_id"].unique().to_list()) == {f.fold_id for f in result.folds}
    assert any(f.fold_set == "covid_shift" for f in result.folds)


def test_an_external_model_carries_its_own_version(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """`_append_metrics` hardcoded the ranker version until a second producer existed."""
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    metrics = result.tables["evaluation_metrics"]
    assert metrics.filter(pl.col("model_name") == EXTERNAL_MODEL)[
        "model_version"
    ].unique().to_list() == ["v1"]


def test_an_external_model_appears_in_the_simulation_and_the_curves(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    for table in ("simulation_summary", "discovery_curves", "sensitivity"):
        names = set(result.tables[table]["model_name"].unique().to_list())
        assert EXTERNAL_MODEL in names, table


# --- 4. probability metrics --------------------------------------------------


def test_a_probability_model_emits_exactly_four_probability_metrics(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    probability = result.tables["evaluation_metrics"].filter(pl.col("metric_kind") == "probability")
    assert set(probability["metric"].unique().to_list()) == {"brier", "log_loss", "ece", "mce"}
    assert set(probability["model_name"].unique().to_list()) == {EXTERNAL_MODEL}
    assert probability["value"].null_count() == 0


def test_threshold_metrics_are_never_emitted(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """They need a threshold and METRICS_SCHEMA has no column to record one in."""
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    emitted = set(result.tables["evaluation_metrics"]["metric"].unique().to_list())
    assert not emitted & {"precision", "recall", "f1"}


def test_a_ranking_only_external_model_emits_no_probability_metric(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Inventing a Brier score for a ranking would be a fabrication."""
    artifact = _artifact(tmp_path / "ranking.parquet", features, is_probability=False)
    result = _run(settings, features, predictions_path=artifact)
    assert (
        result.tables["evaluation_metrics"].filter(pl.col("metric_kind") == "probability").height
        == 0
    )


def test_the_built_in_baselines_never_emit_a_probability_metric(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    probability = result.tables["evaluation_metrics"].filter(pl.col("metric_kind") == "probability")
    assert not set(probability["model_name"].unique().to_list()) & set(DEFAULT_MODELS)


# --- 5. the horizon check can now fail --------------------------------------


def test_a_horizon_violation_fails_its_own_check_not_the_coverage_check(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """`horizon_rejections` was declared and never populated, so
    `scores_respect_the_decision_point` could not fail. A violation surfaced as a
    coverage failure instead -- the right outcome for the wrong reason.
    """
    artifact = _artifact(tmp_path / "cheating.parquet", features, trained_through_offset=400)
    result = _run(settings, features, predictions_path=artifact)

    horizon = next(c for c in result.checks if c.name == "scores_respect_the_decision_point")
    coverage = next(c for c in result.checks if c.name == "predictions_cover_test_exactly")
    assert not horizon.passed
    assert coverage.passed, "a horizon violation must not masquerade as a coverage failure"
    assert evaluation_validate.has_failures(result.checks)


def test_a_horizon_violation_keeps_the_model_out_of_the_results(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Rejected at the door rather than quietly producing an excellent, meaningless
    number."""
    artifact = _artifact(tmp_path / "cheating.parquet", features, trained_through_offset=400)
    result = _run(settings, features, predictions_path=artifact)
    metrics = result.tables["evaluation_metrics"]
    assert EXTERNAL_MODEL not in metrics["model_name"].unique().to_list()


def test_a_coverage_violation_fails_the_coverage_check(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """The other side of the discrimination: a real coverage defect still reports as one."""
    artifact = _artifact(tmp_path / "short.parquet", features, drop_rows=5)
    result = _run(settings, features, predictions_path=artifact)

    coverage = next(c for c in result.checks if c.name == "predictions_cover_test_exactly")
    horizon = next(c for c in result.checks if c.name == "scores_respect_the_decision_point")
    assert not coverage.passed
    assert horizon.passed
    assert evaluation_validate.has_failures(result.checks)


def test_a_horizon_error_is_a_subclass_of_the_contract_error() -> None:
    """So every existing `except PredictionContractError` keeps catching it unchanged."""
    assert issubclass(PredictionHorizonError, PredictionContractError)


def test_the_horizon_ceiling_is_still_the_calibration_end(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Component 6 declares train_end, but the *contract* still permits calibration_end
    -- Component 9 will need it. Declaring it must not be rejected.
    """
    frame = pl.read_parquet(features).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    start, end = min_date(frame, "rd"), max_date(frame, "rd")
    assert start is not None and end is not None
    first = quarterly_folds(data_start=start, data_end=end)[0]
    offset = (first.calibration_end - first.train_end).days

    artifact = _artifact(tmp_path / "calibrated.parquet", features, trained_through_offset=offset)
    result = _run(settings, features, predictions_path=artifact)
    # Later folds have a wider train->calibration gap, so a fixed offset overshoots for
    # some of them; the first fold is the one this asserts about.
    metrics = result.tables["evaluation_metrics"].filter(
        (pl.col("model_name") == EXTERNAL_MODEL) & (pl.col("fold_id") == first.fold_id)
    )
    assert metrics.height > 0


# --- 6. partial and missing artifacts ---------------------------------------


def test_a_fold_with_no_prediction_set_warns_rather_than_crashing(
    settings: Settings, features: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """read_predictions returns [] with no error, so the gap has to be logged."""
    artifact = _artifact(tmp_path / "partial.parquet", features, folds_to_write=1)
    with caplog.at_level(logging.WARNING, logger="sentinel.evaluation.build"):
        result = _run(settings, features, predictions_path=artifact)

    assert not evaluation_validate.has_failures(result.checks)
    assert "carries no prediction set" in caplog.text
    metrics = result.tables["evaluation_metrics"].filter(pl.col("model_name") == EXTERNAL_MODEL)
    assert metrics["fold_id"].n_unique() == 1


def test_a_missing_prediction_artifact_is_reported_clearly(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="Prediction artifact not found"):
        _run(settings, features, predictions_path=tmp_path / "absent.parquet")


def test_several_models_in_one_file_are_all_scored(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Component 6 writes three models into one artifact."""
    first = pl.read_parquet(_artifact(tmp_path / "a.parquet", features, model_name="model_a"))
    second = pl.read_parquet(_artifact(tmp_path / "b.parquet", features, model_name="model_b"))
    combined = tmp_path / "both.parquet"
    pl.concat([first, second]).write_parquet(combined)

    result = _run(settings, features, predictions_path=combined)
    scored = set(result.tables["evaluation_metrics"]["model_name"].unique().to_list())
    assert {"model_a", "model_b"} <= scored


# --- 7. provenance and determinism ------------------------------------------


def test_the_manifest_pins_the_prediction_artifact(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    """Without this, "what was the model allowed to know?" stops being answerable."""
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    assert result.manifest.predictions_path == artifact.name
    assert result.manifest.predictions_sha256 is not None
    assert len(result.manifest.predictions_sha256) == 64


def test_the_summary_names_the_artifact_it_read(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    from sentinel.evaluation.build import summarize

    artifact = _artifact(tmp_path / "predictions.parquet", features)
    result = _run(settings, features, predictions_path=artifact)
    assert artifact.name in summarize(result)


def test_two_runs_with_the_same_artifact_are_identical(
    settings: Settings, features: Path, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path / "predictions.parquet", features)
    first = _run(settings, features, predictions_path=artifact)
    second = _run(settings, features, predictions_path=artifact)
    for name in SCHEMAS:
        assert first.tables[name].equals(second.tables[name]), name
