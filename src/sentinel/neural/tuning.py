"""The learning-rate sweep. Component 7's protocol, applied to Component 8's one knob.

There is deliberately no second tuning protocol in this repository. ``boosting.tuning``
already derives a search region from the fold definitions, refuses one that reaches a test
window, and carves inner folds that mirror the outer structure gap included. A
learning-rate search is a hyperparameter search, so it reuses all three unchanged --
``tuning_region``, ``first_test_start`` and ``build_inner_folds`` are imported, not
reimplemented. ADR 0017 is the contract.

What differs from Component 7 is only the *shape* of the search, and it is smaller on
purpose:

**A grid, not TPE.** Five rates spanning two decades around the specified baseline. The
specification asks whether the neural result is sensitive to the learning rate -- a
question a grid answers directly and legibly, and which a TPE sampler would answer worse
because it concentrates draws near the optimum and leaves the tails unmeasured. It also
asks not to tune endlessly, and a five-point grid cannot.

**One model, not all of them.** The rate is searched for ``neural_embeddings`` and applied
to every network. Searching per model would be eight studies to choose one number each,
and the differences between the models are ablations of the same architecture -- if the
optimal rate differed between them, the ablations would no longer be isolating what they
claim to.

**Two fold sets, two studies.** Same reason Component 7 runs two: the quarterly tuning
region contains the ``covid_shift`` *test* window, so a single shared study would select a
rate using the shift fold's own test labels and make the shift result optimistically
biased. That result -- the one where model orderings invert -- is the one this project
most needs to keep honest.

Selection is by mean PR-AUC across the inner validation windows, ties broken toward the
rate closest to the specified baseline. The tie-break matters more than it sounds: the
grid is coarse and the differences are small, so without a declared rule the winner would
be decided by float noise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

import polars as pl

from sentinel.boosting.tuning import (
    build_inner_folds,
    first_test_start,
    tuning_region,
)
from sentinel.evaluation import metrics
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from sentinel.neural import predict, train
from sentinel.neural.definitions import (
    BASELINE_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_GRID,
    MAX_EPOCHS,
    TUNING_SEED,
    NeuralSpec,
)
from sentinel.neural.models import SweepPoint, SweepResult

logger = logging.getLogger(__name__)

OBJECTIVE = (
    "mean PR-AUC across temporally ordered inner validation windows, each strictly later "
    "than its own inner training window and all strictly earlier than the fold set's "
    "first test window"
)

SEARCH = "exhaustive grid over LEARNING_RATE_GRID; no adaptive sampler"

SELECTION_RULE = (
    "highest mean inner-validation PR-AUC; ties within 1e-6 broken toward the rate "
    "closest to the specification's 1e-3 baseline"
)


class NeuralTuningError(RuntimeError):
    """Raised when a sweep cannot be run, or cannot be run honestly."""


def sweep_fold_set(
    spec: NeuralSpec,
    frame: pl.DataFrame,
    outer_folds: Sequence[FoldSpec],
    *,
    fold_set: str,
    categoricals: pl.DataFrame,
    grid: Sequence[float] = LEARNING_RATE_GRID,
    seed: int = TUNING_SEED,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> SweepResult:
    """Run the grid over one fold set's inner folds.

    The inner folds come from Component 7's builder, which re-derives the safety property
    -- every inner window ends before the fold set's first test window -- from dates
    computed off the data rather than trusting a recorded claim.
    """
    started = datetime.now(UTC)
    region_start, region_end = tuning_region(fold_set, outer_folds)
    horizon = first_test_start(fold_set, outer_folds)
    inner = build_inner_folds(fold_set, outer_folds)

    # ``build_inner_folds`` already refuses a region that reaches a test window. It is
    # re-derived here anyway, from the same fold definitions, because this is the one
    # property whose failure would be invisible in every number the sweep produces.
    latest = max(f.test_end for f in inner)
    if region_end >= horizon or latest >= horizon:
        raise NeuralTuningError(
            f"{fold_set}: the sweep would read to {max(region_end, latest)} but the first "
            f"test window starts {horizon}. A learning rate selected over this region "
            "would have been chosen using test labels."
        )

    if not grid:
        raise NeuralTuningError(f"{fold_set}: the learning-rate grid is empty")

    points: list[SweepPoint] = []
    for rate in grid:
        for fold in inner:
            inner_train = training_frame(frame, fold)
            # The inner fold's *test* window is this search's validation window. These are
            # genuine FoldSpecs, so ``window_frame`` sorts them exactly as an outer window
            # is sorted and no parallel implementation can disagree.
            from sentinel.evaluation import folds as folds_module

            validation = folds_module.window_frame(frame, fold)
            if inner_train.height == 0 or validation.height == 0:
                raise NeuralTuningError(
                    f"{fold.fold_id}: inner fold has an empty window; folds are never "
                    "fabricated and never silently skipped"
                )
            fitted = train.fit_fold(
                spec,
                inner_train,
                fold,
                categoricals=categoricals,
                learning_rate=rate,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
            )
            _, window_scores = predict.score_window(fitted, validation, categoricals=categoricals)
            labels = [int(v) for v in validation["target"].to_list()]
            score = metrics.pr_auc(labels, window_scores)
            if score is None:
                raise NeuralTuningError(
                    f"{fold.fold_id}: PR-AUC is undefined (single-class validation "
                    "window), so this rate cannot be scored"
                )
            points.append(
                SweepPoint(
                    fold_id=fold.fold_id,
                    learning_rate=rate,
                    train_rows=inner_train.height,
                    validation_rows=validation.height,
                    pr_auc=score,
                    best_epoch=fitted.best_epoch,
                )
            )
            logger.info(
                "%s lr=%.0e %s: PR-AUC %.4f (best epoch %d)",
                fold_set,
                rate,
                fold.fold_id,
                score,
                fitted.best_epoch,
            )

    mean_scores: tuple[tuple[float, float], ...] = tuple(
        (rate, _mean([p.pr_auc for p in points if p.learning_rate == rate])) for rate in grid
    )
    best_rate, reason = select_rate(mean_scores)

    elapsed = (datetime.now(UTC) - started).total_seconds()
    logger.info(
        "%s: selected lr=%.0e from %d rates over %d inner fold(s) in %.1fs",
        fold_set,
        best_rate,
        len(grid),
        len(inner),
        elapsed,
    )
    return SweepResult(
        study=f"{spec.name}-{fold_set}",
        model_name=spec.name,
        fold_set=fold_set,
        region_start=region_start,
        region_end=region_end,
        inner_folds=tuple(f.fold_id for f in inner),
        points=tuple(points),
        scores=mean_scores,
        best_learning_rate=best_rate,
        selection_reason=reason,
        seed=seed,
        seconds=elapsed,
    )


def select_rate(scores: Sequence[tuple[float, float]]) -> tuple[float, str]:
    """Pick the winning rate and say why, in one place a test can drive.

    The tie-break toward the baseline is declared rather than emergent. On a coarse grid
    over a dataset where three model classes agree within 0.005 NDE, two rates finishing
    within 1e-6 of each other is a real possibility, and letting float ordering decide
    would make the selected rate an artifact of summation order.
    """
    if not scores:
        raise NeuralTuningError("no rates were scored")
    best = max(value for _, value in scores)
    tied = [rate for rate, value in scores if best - value <= 1e-6]
    if len(tied) == 1:
        return tied[0], f"highest mean inner-validation PR-AUC ({best:.6f})"
    chosen = min(tied, key=lambda r: (abs(r - BASELINE_LEARNING_RATE), r))
    return chosen, (
        f"{len(tied)} rates tied within 1e-6 at PR-AUC {best:.6f}; broke toward the "
        f"specification's {BASELINE_LEARNING_RATE:g} baseline"
    )


def frozen_block(results: Sequence[SweepResult]) -> str:
    """The literal a human pastes into ``definitions.TUNED_HYPERPARAMS``.

    Printed rather than written to a source file, exactly as ``tune-boosting`` does. A
    parameter set loaded from disk at training time could change without a diff, and the
    entire value of freezing is that it cannot.
    """
    lines = ["TUNED_HYPERPARAMS: Mapping[str, Mapping[str, float]] = {"]
    for result in sorted(results, key=lambda r: r.fold_set):
        lines.append(f"    # {result.selection_reason}")
        lines.append(
            f"    # {len(result.inner_folds)} inner fold(s) over "
            f"{result.region_start}..{result.region_end}, all ending before the first test start."
        )
        lines.append(
            f'    "{result.fold_set}": {{"learning_rate": {result.best_learning_rate!r}}},'
        )
    lines.append("}")
    return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise NeuralTuningError("cannot average an empty set of scores")
    return float(sum(values) / len(values))


__all__ = [
    "OBJECTIVE",
    "SEARCH",
    "SELECTION_RULE",
    "NeuralTuningError",
    "frozen_block",
    "select_rate",
    "sweep_fold_set",
]
