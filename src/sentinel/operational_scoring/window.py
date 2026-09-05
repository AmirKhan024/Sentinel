"""The synthetic training window: a real `FoldSpec`, never a real evaluation fold.

`FoldSpec` is Component 5's contract, and Components 6, 7 and 8's ``fit_fold()``
functions all accept exactly one -- reused here unmodified, rather than generalized or
forked, because the functions only ever read ``train_start``/``train_end`` (via
``modeling.train.training_frame``) plus the two identity fields. Nothing about that
contract needs to change for operational mode; what changes is *what fills it*.

The calibration/test windows a `FoldSpec` structurally requires are never read by
anything this component calls, and are filled with the smallest valid placeholder
rather than any real date range, so nobody downstream can mistake them for a real
measurement.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from sentinel.evaluation.folds import CODE_ERA_ANCHOR, max_date
from sentinel.evaluation.models import FoldError, FoldSpec
from sentinel.operational_scoring.definitions import (
    OPERATIONAL_FOLD_SET,
    PLACEHOLDER_WINDOW_SPAN,
)


class OperationalWindowError(ValueError):
    """Raised when no valid training window exists for a planning date."""


def build_operational_fold(
    *, planning_date: date, historical_features: pl.DataFrame, date_column: str = "rd"
) -> FoldSpec:
    """The training-only `FoldSpec` Components 6/7/8's ``fit_fold()`` require.

    ``historical_features`` must be Component 4's real, labelled feature table (with a
    parsed ``date_column``) -- never Component 17's candidate table, which carries no
    label and cannot train anything. Training reads only rows with a reference date
    strictly before ``planning_date``, via ``train_end`` below; ``modeling.train.
    training_frame`` (Components 6-8's own shared function) does the actual filtering,
    unmodified.
    """
    available_max = max_date(historical_features, date_column)
    if available_max is None:
        raise OperationalWindowError(
            "the historical feature table has no usable reference dates; there is "
            "nothing to train an operational model on"
        )

    train_end = min(planning_date - timedelta(days=1), available_max)
    if train_end < CODE_ERA_ANCHOR:
        raise OperationalWindowError(
            f"planning_date {planning_date.isoformat()} leaves no code-era training "
            f"data (anchor {CODE_ERA_ANCHOR.isoformat()}) strictly before it -- the "
            "earliest supportable planning date is the day after the anchor"
        )

    calibration_start = calibration_end = planning_date
    test_start = test_end = planning_date + PLACEHOLDER_WINDOW_SPAN

    try:
        return FoldSpec(
            fold_set=OPERATIONAL_FOLD_SET,
            fold_id=f"operational-{planning_date.isoformat()}",
            train_start=CODE_ERA_ANCHOR,
            train_end=train_end,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            test_start=test_start,
            test_end=test_end,
        )
    except FoldError as exc:
        raise OperationalWindowError(str(exc)) from exc


__all__ = ["OperationalWindowError", "build_operational_fold"]
