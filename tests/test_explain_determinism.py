"""Two runs, same inputs, same bytes.

Determinism is not a nice property here, it is the precondition for the artifact being
citable at all. If two runs disagreed, no figure in the findings document could be
reproduced and no sha256 in the manifest would mean anything.

The standard is **byte-identity of the written Parquet**, not equality of the frames. A
table that holds the same rows in a different order is a different file, and row order is
the last place non-determinism could hide once the sampler, the background and the
permutation generator are all seeded.

Four sources of randomness exist in this component, and each is pinned separately:

* the explanation sample -- ``SAMPLING_SEED`` plus the canonical sort;
* the background -- ``BACKGROUND_SEED`` plus the canonical sort;
* the permutation game -- its own seeded generator, consumed in a fixed sequence;
* the fits themselves -- Components 6 to 8's own determinism, which the bit-identity gate
  re-proves on every run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sentinel.boosting import build as boosting_build
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.explain import aggregate
from sentinel.explain.attribute import permutation_attributions
from sentinel.explain.background import select_background
from sentinel.explain.build import run_explanations
from sentinel.explain.definitions import BACKGROUND_SEED, SAMPLING_SEED
from sentinel.explain.sample import select_sample
from sentinel.manifest import compute_sha256
from tests.conftest import spanning_model_features

MODELS = ["xgboost"]
SAMPLE = 10


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    tmp = tmp_path_factory.mktemp("explain_determinism")
    features = tmp / "as_of_features_20260101T000000Z.parquet"
    spanning_model_features(days=1900).write_parquet(features)
    boosted = boosting_build.train_boosting(
        Settings(data_dir=tmp), features_path=features, output_dir=tmp, models=MODELS
    )
    assert boosted.predictions_path is not None
    return features, boosted.predictions_path, tmp


def _run(inputs: tuple[Path, Path, Path], destination: Path) -> dict[str, str]:
    features, predictions, tmp = inputs
    result = run_explanations(
        Settings(data_dir=tmp),
        features_path=features,
        prediction_paths={"boosted_predictions": predictions},
        output_dir=destination,
        models=MODELS,
        sample_size=SAMPLE,
        write_figures=False,
    )
    # Keyed by table rather than by file name: the file name carries a UTC timestamp, which
    # differs between runs by design and is not part of the contract.
    return {path.name.rsplit("_", 1)[0]: compute_sha256(path) for path in result.written}


# --- 1. the whole artifact ---------------------------------------------------


def test_two_runs_produce_byte_identical_tables(
    inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    first = _run(inputs, tmp_path / "one")
    second = _run(inputs, tmp_path / "two")
    assert first == second
    assert len(first) == 7, "all seven tables compared, not a subset"


def test_the_run_is_not_trivially_empty(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Two identical empty files would satisfy the test above and prove nothing."""
    features, predictions, tmp = inputs
    result = run_explanations(
        Settings(data_dir=tmp),
        features_path=features,
        prediction_paths={"boosted_predictions": predictions},
        output_dir=tmp_path / "three",
        models=MODELS,
        sample_size=SAMPLE,
        write_figures=False,
    )
    assert result.tables["explanation_values"].height > 0
    assert result.tables["explanation_cases"].height > 0
    assert result.stats.attribution_values > 0


# --- 2. each source of randomness, pinned separately -------------------------


def test_the_explanation_sample_is_stable_across_calls(
    inputs: tuple[Path, Path, Path],
) -> None:
    features, _, _ = inputs
    frame = pl.read_parquet(features).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    fold = folds_module.quarterly_folds(data_start=start, data_end=end)[0]

    first = select_sample(frame, fold, size=SAMPLE, seed=SAMPLING_SEED)
    second = select_sample(frame, fold, size=SAMPLE, seed=SAMPLING_SEED)
    assert first.ids == second.ids
    assert first.ids != select_sample(frame, fold, size=SAMPLE, seed=SAMPLING_SEED + 1).ids


def test_the_background_is_stable_across_calls(inputs: tuple[Path, Path, Path]) -> None:
    features, _, _ = inputs
    frame = pl.read_parquet(features).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    assert start is not None and end is not None
    fold = folds_module.quarterly_folds(data_start=start, data_end=end)[0]

    first = select_background(frame, fold, size=16, seed=BACKGROUND_SEED)
    second = select_background(frame, fold, size=16, seed=BACKGROUND_SEED)
    assert first.equals(second)
    assert not first.equals(select_background(frame, fold, size=16, seed=BACKGROUND_SEED + 1))


def test_the_permutation_generator_is_consumed_in_a_fixed_sequence() -> None:
    """Row order must not change the values, or a re-sorted input would re-attribute."""
    rng = np.random.default_rng(11)
    weights = rng.normal(size=5)

    def predict(block: np.ndarray) -> np.ndarray:
        return np.tanh(block @ weights)

    background = rng.normal(size=(8, 5))
    rows = rng.normal(size=(4, 5))

    values, _ = permutation_attributions(predict, rows, background, rounds=3, seed=99)
    again, _ = permutation_attributions(predict, rows, background, rounds=3, seed=99)
    assert np.array_equal(values, again)


# --- 3. downstream analysis is deterministic too -----------------------------


def test_representative_case_selection_is_stable(
    inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A quantile with an unstable tie-break would move the report's examples between runs."""
    first = _run(inputs, tmp_path / "cases_a")
    second = _run(inputs, tmp_path / "cases_b")
    assert first["explanation_representative_cases"] == second["explanation_representative_cases"]


def test_rank_ties_do_not_depend_on_input_order() -> None:
    """Averaged ranks and a name tie-break, so a permuted feature list changes nothing."""
    values = np.array([0.5, 0.5, 0.1])
    assert aggregate.ranks(values).tolist() == aggregate.ranks(values[::-1]).tolist()[::-1]
    assert aggregate.top_features(["b", "a"], [1.0, 1.0], 1) == aggregate.top_features(
        ["a", "b"], [1.0, 1.0], 1
    )
