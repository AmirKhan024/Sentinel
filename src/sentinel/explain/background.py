"""Temporally safe reference rows for the explainers that need them. Pure -- no I/O.

**A background dataset is part of the explanation, not a technicality behind it.** Every
SHAP value answers "how much did this feature move the output, relative to what?", and the
background *is* the "relative to what". Get the background wrong and the numbers are still
additive, still finite, still plausible -- and answering a question nobody asked.

Which makes it a leakage surface of exactly the kind this project keeps finding. Drawing
reference rows from the test window would be the natural, convenient thing to do: those
rows are already in hand, they are the same distribution as the rows being explained, and
every additivity check would still pass. It would also mean the explanation's reference
point encodes the period the model is being judged on -- information the model did not have
when it produced the score being explained.

So the background comes from the fold's **training** window and nowhere else, via
``modeling.train.training_frame`` -- the repository's one definition of "train", the same
function every fit calls, and the one Component 5's ``future_rows_never_enter_training``
check independently re-derives. Every returned row is dated on or before ``fold.train_end``
by construction, and ``validate`` re-derives that from the frame rather than trusting this
docstring, because a comment is not a check.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl

from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import TRAIN_SORT_KEYS, training_frame

logger = logging.getLogger(__name__)


class BackgroundError(RuntimeError):
    """Raised when a temporally safe background cannot be built for a fold."""


def select_background(
    frame: pl.DataFrame, fold: FoldSpec, *, size: int, seed: int, date_column: str = "rd"
) -> pl.DataFrame:
    """``size`` reference rows drawn from ``fold``'s training window, deterministically.

    Uniform without replacement. Not stratified, and not sampled to match the test window's
    composition: a background chosen to resemble the rows being explained would make the
    reference point depend on those rows, and the expected value would stop being a
    property of the model's training period.

    Determinism comes from two things together, and both are needed. The window is sorted by
    ``TRAIN_SORT_KEYS`` before anything is drawn, so the row *order* the generator indexes
    into is fixed; and the generator is seeded from an integer constant. Seeding alone would
    not be enough, because ``training_frame`` returns rows in whatever order the filter
    produced them from a frame whose own order a caller controls.
    """
    if size <= 0:
        raise BackgroundError(f"background size must be positive, got {size}")

    window = training_frame(frame, fold).sort(TRAIN_SORT_KEYS)
    if window.height == 0:
        raise BackgroundError(
            f"fold {fold.fold_id} has an empty training window, so no reference rows exist"
        )

    take = min(size, window.height)
    if take < size:
        logger.info(
            "Fold %s training window holds %d rows; background reduced from %d to %d",
            fold.fold_id,
            window.height,
            size,
            take,
        )
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(window.height, size=take, replace=False))
    # Narrowed explicitly: polars types row indexing as a wide union, and an Any
    # escaping here would defeat strict mode for every caller.
    selected: pl.DataFrame = window[positions.tolist()]

    # The claim this module exists to make, re-derived from the rows actually selected
    # rather than inferred from the function they came out of. Cheap, and it is the
    # difference between a guarantee and an intention.
    # Narrowed before comparison: polars types a column aggregate as a wide union, and
    # ``anything > date`` is not defined for most of it.
    latest = selected[date_column].max()
    if isinstance(latest, date) and latest > fold.train_end:
        raise BackgroundError(
            f"fold {fold.fold_id}: background contains a row dated {latest}, after the "
            f"training horizon {fold.train_end}. A reference set that has seen the future "
            "answers a counterfactual the model was never asked."
        )
    return selected


def background_is_safe(
    background: pl.DataFrame, fold: FoldSpec, *, date_column: str = "rd"
) -> tuple[bool, list[str]]:
    """Whether every background row sits at or before ``fold.train_end``.

    Returns the verdict and the offending dates rather than raising, so ``validate`` can
    report every fold's outcome in one pass instead of stopping at the first failure.
    """
    if background.height == 0:
        return True, []
    late = background.filter(pl.col(date_column) > fold.train_end)
    if late.height == 0:
        return True, []
    offenders = [
        f"{fold.fold_id}/{row}: dated {day} > train_end {fold.train_end}"
        for row, day in zip(
            late["target_inspection_id"].to_list()[:20],
            late[date_column].to_list()[:20],
            strict=True,
        )
    ]
    return False, offenders


def background_ids(background: pl.DataFrame) -> set[str]:
    """The ids in a background set, for the containment check in ``validate``."""
    if background.height == 0:
        return set()
    return {str(v) for v in background["target_inspection_id"].to_list()}


__all__ = ["BackgroundError", "background_ids", "background_is_safe", "select_background"]
