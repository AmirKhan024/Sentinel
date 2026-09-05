"""Orchestration: what a run produces, and what it refuses to produce.

``build`` is the only module in the package that touches the filesystem, so these tests
are about the pipeline's shape -- artifact counts, the fold loop, the seed sweep and the
figures -- rather than about model quality, which is Component 5's to measure.

The end-to-end fixture is deliberately small: two models, a short span. The full nine-model
run over eighteen real folds takes hours, and repeating it in a unit suite would buy
nothing the contract tests do not already cover.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.neural import build
from tests.conftest import neural_categoricals_for, spanning_model_features


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("neural_build")
    frame = spanning_model_features(days=1600, per_day=3)
    features = tmp / "as_of_features_20260101T000000Z.parquet"
    frame.write_parquet(features)
    categoricals = tmp / "neural_categoricals_20260101T000000Z.parquet"
    neural_categoricals_for(frame).write_parquet(categoricals)
    return {"features": features, "categoricals": categoricals, "dir": tmp}


def _run(inputs: dict[str, Path], out: Path, **kwargs: Any) -> Any:
    settings = Settings(data_dir=inputs["dir"] / "data")
    options: dict[str, Any] = {
        "models": ["neural_numeric_only"],
        "seed_sweep": False,
        "render_figures": False,
    }
    options.update(kwargs)
    return build.train_neural(
        settings,
        features_path=inputs["features"],
        categoricals_path=inputs["categoricals"],
        output_dir=out,
        **options,
    )


# --- 1. the artifacts --------------------------------------------------------


def test_a_run_writes_every_declared_table(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path)
    for name in (
        "neural_predictions",
        "neural_training_log",
        "neural_epoch_log",
        "neural_embeddings",
        "neural_seed_variation",
    ):
        assert name in result.tables, f"{name} was not produced"
        assert list(tmp_path.glob(f"{name}_*.parquet")), f"{name} was not written"


def test_the_manifest_lands_beside_the_predictions(inputs: dict[str, Path], tmp_path: Path) -> None:
    """One manifest per run, beside the primary artifact -- the project convention."""
    result = _run(inputs, tmp_path)
    assert result.manifest_path is not None
    assert result.manifest_path.name.startswith("manifest_neural_predictions_")
    assert len(list(tmp_path.glob("manifest_*.json"))) == 1


def test_every_written_table_is_recorded_in_the_manifest(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(inputs, tmp_path)
    recorded = {a.path for a in result.manifest.artifacts}
    written = {p.name for p in tmp_path.glob("*.parquet")}
    assert recorded == written
    for artifact in result.manifest.artifacts:
        assert len(artifact.sha256) == 64
        assert artifact.bytes > 0


def test_one_training_log_row_per_model_per_fold(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path)
    log = result.tables["neural_training_log"]
    assert log.height == len(result.folds)
    assert log["fold_id"].n_unique() == len(result.folds)


def test_one_prediction_per_model_per_test_row(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path)
    predictions = result.tables["neural_predictions"]
    keys = predictions.select("model_name", "fold_id", "target_inspection_id")
    assert keys.unique().height == predictions.height


def test_the_epoch_log_records_the_best_epoch_flag(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path)
    epochs = result.tables["neural_epoch_log"]
    assert epochs.height > 0
    per_fold = epochs.group_by("model_name", "fold_id").agg(
        pl.col("is_best_epoch").sum().alias("flags")
    )
    assert (per_fold["flags"] == 1).all(), "a fold marked zero or many best epochs"


def test_embeddings_are_written_only_for_the_representative_model(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    """Every ablation learns vectors; only one model's are the ones under study."""
    result = _run(inputs, tmp_path, models=["neural_embeddings", "neural_no_chain"])
    embeddings = result.tables["neural_embeddings"]
    assert embeddings.height > 0
    assert set(embeddings["model_name"].to_list()) == {"neural_embeddings"}


def test_a_numeric_only_run_writes_no_embeddings(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path)
    assert result.tables["neural_embeddings"].height == 0


# --- 2. the fold loop --------------------------------------------------------


def test_both_fold_sets_are_built_and_neither_is_averaged_away(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    """``covid_shift`` is a separate fold set precisely so it cannot join the headline."""
    result = _run(inputs, tmp_path)
    sets = {f.fold_set for f in result.folds}
    assert sets == {"quarterly", "covid_shift"}
    assert result.manifest.fold_sets["covid_shift"] == 1
    assert result.manifest.fold_sets["quarterly"] >= 2


def test_each_fold_is_fitted_from_scratch(inputs: dict[str, Path], tmp_path: Path) -> None:
    """Fold N is not fold N-1 warm-started; a vocabulary would carry across if it were."""
    result = _run(inputs, tmp_path, models=["neural_embeddings"])
    seen = {id(m) for m in result.fitted}
    assert len(seen) == len(result.fitted) == len(result.folds)
    encodings = [m.encoding.sizes for m in result.fitted if m.fold_set == "quarterly"]
    assert len(encodings) >= 2
    # Expanding windows mean later folds see at least as many categories.
    chains = [e.get("chain", 0) for e in encodings]
    assert chains == sorted(chains), "a later fold saw fewer chains than an earlier one"


# --- 3. the seed sweep -------------------------------------------------------


def test_the_seed_sweep_records_every_declared_seed(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    from sentinel.neural.definitions import SEED_SWEEP

    result = _run(inputs, tmp_path, models=["neural_embeddings"], seed_sweep=True)
    variation = result.tables["neural_seed_variation"]
    assert variation.height == len(SEED_SWEEP) * len(result.folds)
    assert set(variation["seed"].to_list()) == set(SEED_SWEEP)
    assert variation["pr_auc"].null_count() == 0


def test_the_seed_sweep_is_skippable(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path, seed_sweep=False)
    assert result.tables["neural_seed_variation"].height == 0


# --- 4. refusals -------------------------------------------------------------


def test_a_missing_categorical_table_is_refused(inputs: dict[str, Path], tmp_path: Path) -> None:
    settings = Settings(data_dir=inputs["dir"] / "data")
    with pytest.raises(FileNotFoundError, match="build-neural-categoricals"):
        build.train_neural(
            settings,
            features_path=inputs["features"],
            categoricals_path=tmp_path / "absent.parquet",
            output_dir=tmp_path,
        )


def test_a_categorical_table_that_does_not_cover_the_features_is_refused(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    """A partial join would leave rows with no categoricals and no error."""
    short = tmp_path / "short_cats.parquet"
    pl.read_parquet(inputs["categoricals"]).head(10).write_parquet(short)
    settings = Settings(data_dir=inputs["dir"] / "data")
    with pytest.raises(build.NeuralBuildError, match="would have no categoricals"):
        build.train_neural(
            settings,
            features_path=inputs["features"],
            categoricals_path=short,
            output_dir=tmp_path,
        )


def test_the_embedding_fed_booster_needs_its_donor(inputs: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(build.NeuralBuildError, match="consumes embeddings from"):
        _run(inputs, tmp_path, models=["xgboost_chain_embeddings"])


def test_an_unknown_model_is_refused(inputs: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(build.NeuralBuildError, match="Unknown neural model"):
        _run(inputs, tmp_path, models=["neural_embedings"])


def test_an_empty_model_list_is_refused(inputs: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(build.NeuralBuildError, match="no models requested"):
        _run(inputs, tmp_path, models=[])


def test_a_duplicated_target_inspection_id_is_refused(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    frame = pl.read_parquet(inputs["features"])
    doubled = pl.concat([frame, frame.head(1)])
    path = tmp_path / "dup_features.parquet"
    doubled.write_parquet(path)
    settings = Settings(data_dir=inputs["dir"] / "data")
    with pytest.raises(build.NeuralBuildError, match="duplicated target_inspection_id"):
        build.train_neural(
            settings,
            features_path=path,
            categoricals_path=inputs["categoricals"],
            output_dir=tmp_path,
        )


def test_a_dry_run_writes_nothing_but_still_validates(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(inputs, tmp_path, dry_run=True)
    assert not list(tmp_path.glob("*.parquet"))
    assert result.predictions_path is None
    assert result.checks, "a dry run produced no validation checks"


# --- 5. figures --------------------------------------------------------------


def test_figures_are_rendered_for_the_representative_model(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    """The two the specification requires: learning curves and a chain projection."""
    figures = tmp_path / "figures"
    result = _run(
        inputs,
        tmp_path,
        models=["neural_embeddings"],
        render_figures=True,
        figures_dir=figures,
    )
    assert result.figure_paths, "no figure was produced"
    names = [p.name for p in result.figure_paths]
    assert any("learning_curve" in n for n in names)
    for path in result.figure_paths:
        assert path.exists() and path.stat().st_size > 0
        assert path.suffix == ".png"


def test_the_figure_filename_names_the_fold_it_describes(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    """So a reader is never guessing which fold a picture is of."""
    figures = tmp_path / "figures"
    result = _run(
        inputs,
        tmp_path,
        models=["neural_embeddings"],
        render_figures=True,
        figures_dir=figures,
    )
    quarterly = [f for f in result.folds if f.fold_set == "quarterly"]
    last = max(quarterly, key=lambda f: f.train_end)
    assert any(last.fold_id in p.name for p in result.figure_paths)


def test_figures_are_skippable(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path, render_figures=False)
    assert result.figure_paths == []


# --- 6. the summary ----------------------------------------------------------


def test_the_summary_points_at_the_next_command(inputs: dict[str, Path], tmp_path: Path) -> None:
    """Component 8 predicts; Component 5 evaluates. The handoff is an artifact path."""
    result = _run(inputs, tmp_path)
    text = build.summarize(result)
    assert "sentinel evaluate --predictions" in text
    assert str(result.predictions_path) in text


def test_the_summary_reports_no_metric(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(inputs, tmp_path)
    text = build.summarize(result).lower()
    for forbidden in ("roc", "pr-auc", "nde", "brier", "ece"):
        assert forbidden not in text
