"""Choosing which predictions get explained. Pure -- no I/O.

Two rules, and both are about what the selection is *not* allowed to see.

**No label participates.** Not the target, not a metric, not whether the model was right.
Selecting rows the model got right would produce a findings document describing a model
that does not exist; selecting rows it got wrong would produce a different fiction. The
sample is drawn uniformly from the fold's test window, and the ``target`` column is never
read in this module -- ``select_sample`` takes the frame and touches two columns of it, and
a test asserts that permuting the labels leaves the selection identical.

**No score participates either**, which is the less obvious half. Sampling the highest-risk
rows would be defensible-sounding and would bias every global importance number towards the
features that drive the top of the ranking. The *representative cases* in
``aggregate.representative_cases`` are chosen by predicted-score quantile, deliberately and
separately -- that is a reporting choice about which three explanations to print, made after
the fact, and it never affects which rows were explained or what the importance table says.

The same ids are used for every model. All four supported models score the identical test
window, so drawing one sample per fold and reusing it makes a cross-model importance
comparison a comparison of models. Drawing per model would leave any difference between two
importance tables part real and part sampling noise, with no way to separate them.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.explain.definitions import SAMPLE_STRATEGY, SAMPLING_POPULATION
from sentinel.explain.models import ExplanationSample

logger = logging.getLogger(__name__)

#: The columns ``select_sample`` is permitted to read. Enforced by a test that hands the
#: function a frame whose every other column has been corrupted and asserts the selection
#: is unchanged -- the executable form of "no label participates".
SELECTION_COLUMNS: tuple[str, ...] = ("rd", "target_inspection_id")


class SampleError(RuntimeError):
    """Raised when an explanation sample cannot be drawn for a fold."""


def select_sample(
    frame: pl.DataFrame, fold: FoldSpec, *, size: int, seed: int
) -> ExplanationSample:
    """``size`` ids drawn uniformly from ``fold``'s test window, deterministically.

    The window comes from ``evaluation.folds.window_frame``, which is the same function
    every component's ``score_window`` requires and which returns rows already sorted by
    ``(rd, target_inspection_id)``. Sorting first is what makes the draw reproducible: the
    generator indexes into positions, so a different row order is a different sample even
    at the same seed.
    """
    if size <= 0:
        raise SampleError(f"sample size must be positive, got {size}")

    window = folds_module.window_frame(frame, fold)
    if window.height == 0:
        raise SampleError(f"fold {fold.fold_id} has an empty test window, so nothing to explain")

    take = min(size, window.height)
    if take < size:
        logger.info(
            "Fold %s test window holds %d rows; sample reduced from %d to %d",
            fold.fold_id,
            window.height,
            size,
            take,
        )
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(window.height, size=take, replace=False))
    ids = [str(v) for v in window["target_inspection_id"].to_list()]
    chosen = tuple(ids[p] for p in positions)

    return ExplanationSample(
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        ids=chosen,
        population_rows=window.height,
        strategy=SAMPLE_STRATEGY,
        seed=seed,
        population=SAMPLING_POPULATION,
    )


def row_positions(sample: ExplanationSample, row_ids: tuple[str, ...]) -> list[int]:
    """Where each sampled id sits in a refit model's row order.

    Positional rather than a join, and checked rather than assumed: a mis-join here would
    attach one establishment's attributions to another's id, and every additivity check
    would still pass because the rows are internally consistent -- just about the wrong
    establishment.
    """
    index = {row_id: position for position, row_id in enumerate(row_ids)}
    missing = [row_id for row_id in sample.ids if row_id not in index]
    if missing:
        raise SampleError(
            f"fold {sample.fold_id}: {len(missing)} sampled id(s) are absent from the "
            f"scored window, first {missing[0]!r}. The sample and the model disagree about "
            "which rows the fold contains."
        )
    return [index[row_id] for row_id in sample.ids]


__all__ = ["SELECTION_COLUMNS", "SampleError", "row_positions", "select_sample"]
