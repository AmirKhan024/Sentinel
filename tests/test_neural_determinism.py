"""Reproducibility, and the honest limits of the claim.

Component 7 established that reproducibility is a real issue in this project, and a
network has strictly more stochasticity than a booster: weight initialisation, batch
composition and dropout masks on top of the float-summation order a booster already has.

The claim these tests support is narrow and is the same one Components 6 and 7 make:
*identical predictions for a fixed input, a fixed row order, a fixed library set and a
single thread.* What they deliberately do **not** claim is that a different seed gives the
same answer -- that is measured by the multi-seed experiment and reported as a spread,
because quantifying the variation is honest and asserting it away is not.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from sentinel.neural import net, predict, train
from sentinel.neural.definitions import DEFAULT_SEED, SEED_SWEEP, spec_for
from tests.conftest import neural_categoricals_for, spanning_model_features

PRIMARY = spec_for("neural_embeddings")


def _base() -> pl.DataFrame:
    return spanning_model_features(days=1600, per_day=2).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def _fold(frame: pl.DataFrame) -> FoldSpec:
    built = folds_module.quarterly_folds(
        data_start=folds_module.min_date(frame, "rd"),
        data_end=folds_module.max_date(frame, "rd"),
    )
    assert built
    return built[0]


def _run(frame: pl.DataFrame, fold: FoldSpec, cats: pl.DataFrame, seed: int | None = None):  # type: ignore[no-untyped-def]
    fitted = train.fit_fold(
        PRIMARY,
        training_frame(frame, fold),
        fold,
        categoricals=cats,
        max_epochs=4,
        seed=seed,
    )
    _, scores = predict.score_window(
        fitted, folds_module.window_frame(frame, fold), categoricals=cats
    )
    return fitted, scores


# --- 1. the claim ------------------------------------------------------------


def test_the_same_seed_produces_bit_identical_predictions() -> None:
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    _, first = _run(frame, fold, cats)
    _, second = _run(frame, fold, cats)
    assert first == second, "two runs of an identical configuration disagreed"


def test_the_same_seed_produces_bit_identical_weights() -> None:
    """Not just the scores: the learned representation itself."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    first, _ = _run(frame, fold, cats)
    second, _ = _run(frame, fold, cats)
    for left, right in zip(first.embeddings, second.embeddings, strict=True):
        assert left.vectors == right.vectors, f"{left.column}: embedding table moved"
    assert first.scaler_mean == second.scaler_mean
    assert first.best_epoch == second.best_epoch
    assert [e.train_loss for e in first.epochs] == [e.train_loss for e in second.epochs]


def test_the_epoch_trajectory_is_reproducible() -> None:
    """Batch composition is drawn from a seeded generator, not global state."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    first, _ = _run(frame, fold, cats)
    second, _ = _run(frame, fold, cats)
    assert [(e.epoch, e.train_loss, e.validation_loss, e.learning_rate) for e in first.epochs] == [
        (e.epoch, e.train_loss, e.validation_loss, e.learning_rate) for e in second.epochs
    ]


# --- 2. the limits of the claim ----------------------------------------------


def test_a_different_seed_gives_a_different_answer() -> None:
    """A vacuity guard, and the reason the multi-seed experiment exists.

    If this passed by giving identical results, the reproducibility tests above would be
    measuring a model with no stochasticity rather than a pinned one -- and the
    seed-variation artifact would be reporting a spread of exactly zero for the wrong
    reason.
    """
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    _, first = _run(frame, fold, cats, seed=42)
    _, second = _run(frame, fold, cats, seed=43)
    assert first != second, (
        "two different seeds produced identical predictions, so seeding is inert and the "
        "reproducibility claim above is meaningless"
    )


def test_the_seed_is_recorded_on_the_fit() -> None:
    """A number that cannot be traced to a seed cannot be reproduced."""
    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, _ = _run(frame, fold, cats, seed=44)
    assert fitted.seed == 44
    default, _ = _run(frame, fold, cats)
    assert default.seed == DEFAULT_SEED


def test_the_seed_sweep_contains_the_headline_seed() -> None:
    """So the reported spread includes the run that produced the predictions."""
    assert DEFAULT_SEED in SEED_SWEEP
    assert len(set(SEED_SWEEP)) == len(SEED_SWEEP)


# --- 3. the environment ------------------------------------------------------


def test_training_pins_a_single_thread() -> None:
    """The direct analogue of Component 7's ``n_jobs=1``.

    A float reduction over threads depends on the order the threads finish in, so a
    multi-threaded fit is reproducible only approximately -- and this project's standard
    for "did not move" is bit-identity.
    """
    import torch

    frame = _base()
    fold = _fold(frame)
    _run(frame, fold, neural_categoricals_for(frame))
    assert torch.get_num_threads() == 1


def test_training_enables_deterministic_algorithms() -> None:
    """torch raises rather than silently choosing a nondeterministic kernel."""
    import torch

    frame = _base()
    fold = _fold(frame)
    _run(frame, fold, neural_categoricals_for(frame))
    assert torch.are_deterministic_algorithms_enabled()


def test_the_fit_runs_on_the_cpu() -> None:
    """A CUDA device may be present; it is deliberately unused. ADR 0020."""
    assert net.device_name() == "cpu"


def test_a_caller_cannot_reach_the_module_of_a_foreign_fit() -> None:
    """``scorer_for`` refuses a record it did not produce, rather than guessing."""
    from sentinel.neural.models import FittedNetwork

    frame = _base()
    fold = _fold(frame)
    cats = neural_categoricals_for(frame)
    fitted, _ = _run(frame, fold, cats)
    clone = FittedNetwork(**{f: getattr(fitted, f) for f in fitted.__slots__})
    with pytest.raises(train.NeuralTrainError, match="no live module"):
        train.scorer_for(clone)
