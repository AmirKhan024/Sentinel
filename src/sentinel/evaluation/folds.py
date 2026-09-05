"""Rolling-origin fold construction.

Why this exists at all: a random split of this table would let the model learn
from 2024 and be scored on 2022. Sentinel will never operate that way -- at any
real decision point the future does not exist yet -- so an evaluation that
allows it measures something the deployed system can never reproduce.

The structure is an **expanding** window. Training always starts at the code-era
anchor and grows forward one quarter at a time; calibration is the next quarter;
test is the quarter after that.

    fold 1   train 2018-07-01..2021-12-31 | cal 2022Q1 | test 2022Q2
    fold 2   train 2018-07-01..2022-03-31 | cal 2022Q2 | test 2022Q3
    fold 3   train 2018-07-01..2022-06-30 | cal 2022Q3 | test 2022Q4

Expanding rather than sliding because a regulator retrains on everything it has;
throwing away 2019 to keep a fixed-width window would model a system nobody
would build.

Calibration sits **between** train and test and is never merged into either.
Component 9 will fit Platt or isotonic on it, and a calibrator fitted on the
test period would make the reported probabilities self-fulfilling.

Partial windows are excluded, never fabricated. The snapshot ends 2026-08-14, so
2026Q3 exists in the data as 328 rows across 32 days; treating that as a quarter
would silently compare a two-thirds window against full ones.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, timedelta

import polars as pl

from sentinel.evaluation.models import FoldSpec, FoldStats

logger = logging.getLogger(__name__)

#: Component 3's eligibility boundary, and therefore the earliest date any fold
#: may train on. Restated rather than imported as a date so this module has one
#: obvious place to look for the anchor.
CODE_ERA_ANCHOR = date(2018, 7, 1)

#: Quarters of training before the first fold. 2018Q3..2021Q4 inclusive is 14,
#: which reproduces the project spec's Fold 1 exactly: train through Dec 2021,
#: calibrate on 2022Q1, test on 2022Q2.
MIN_TRAIN_QUARTERS = 14

QUARTERLY = "quarterly"
COVID_SHIFT = "covid_shift"


def quarter_start(day: date) -> date:
    """First day of the calendar quarter containing ``day``."""
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def add_quarters(start: date, count: int) -> date:
    """Shift a quarter-start date by ``count`` quarters, forwards or backwards."""
    index = start.year * 4 + (start.month - 1) // 3 + count
    return date(index // 4, 3 * (index % 4) + 1, 1)


def quarter_end(start: date) -> date:
    """Last day of the quarter beginning at ``start``."""
    return add_quarters(start, 1) - timedelta(days=1)


def quarter_key(day: date) -> str:
    """Human-readable quarter label, e.g. ``2022Q2``. Used as the fold id."""
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def quarterly_folds(
    *,
    data_start: date,
    data_end: date,
    anchor: date = CODE_ERA_ANCHOR,
    min_train_quarters: int = MIN_TRAIN_QUARTERS,
) -> list[FoldSpec]:
    """Build every quarterly fold the data can fully support.

    A fold is emitted only when its test quarter is **entirely** covered by the
    snapshot. The count is therefore derived from the data, never hardcoded --
    re-ingesting a later snapshot adds folds without a code change.
    """
    if data_start > data_end:
        raise ValueError(f"data_start {data_start} is after data_end {data_end}")

    first_calibration = add_quarters(quarter_start(anchor), min_train_quarters)
    folds: list[FoldSpec] = []

    calibration = first_calibration
    while True:
        test = add_quarters(calibration, 1)
        test_last = quarter_end(test)
        if test_last > data_end:
            break
        folds.append(
            FoldSpec(
                fold_set=QUARTERLY,
                fold_id=f"{QUARTERLY}-{quarter_key(test)}",
                train_start=anchor,
                train_end=calibration - timedelta(days=1),
                calibration_start=calibration,
                calibration_end=quarter_end(calibration),
                test_start=test,
                test_end=test_last,
            )
        )
        calibration = add_quarters(calibration, 1)

    logger.info(
        "Built %d quarterly folds; first test %s, last test %s",
        len(folds),
        folds[0].test_start if folds else "none",
        folds[-1].test_end if folds else "none",
    )
    return folds


def excluded_partial_windows(*, data_end: date, folds: Sequence[FoldSpec]) -> list[str]:
    """Quarters that exist in the data but were dropped as incomplete.

    Reported in the manifest so a silent truncation cannot be mistaken for full
    coverage. There is at most one: the quarter the snapshot stops inside.
    """
    if not folds:
        return []
    next_test = add_quarters(folds[-1].test_start, 1)
    if next_test <= data_end < quarter_end(next_test):
        return [f"{quarter_key(next_test)} (snapshot ends {data_end}, quarter is incomplete)"]
    return []


def covid_shift_fold(*, data_end: date, anchor: date = CODE_ERA_ANCHOR) -> list[FoldSpec]:
    """The distribution-shift experiment: train pre-COVID, test on the disruption.

    A single fold, deliberately not part of the rolling set, because it answers a
    different question -- not *is the ranking good on average* but *does it
    survive an operational regime change*. Chicago's inspection programme was
    suspended and restarted during 2020, and the base rate falls from 0.773 in
    the training window to 0.513 in the test window.
    """
    test_end = date(2021, 12, 31)
    if test_end > data_end:
        logger.warning("Snapshot ends %s; the covid_shift test window is incomplete", data_end)
        return []
    return [
        FoldSpec(
            fold_set=COVID_SHIFT,
            fold_id=f"{COVID_SHIFT}-2020H2-2021",
            train_start=anchor,
            train_end=date(2020, 2, 29),
            calibration_start=date(2020, 3, 1),
            calibration_end=date(2020, 5, 31),
            test_start=date(2020, 6, 1),
            test_end=test_end,
        )
    ]


def assign_split(frame: pl.DataFrame, fold: FoldSpec, *, date_column: str = "rd") -> pl.DataFrame:
    """Label every row with its split for one fold.

    Implemented as one chained ``when/then`` over date comparisons rather than a
    join, so a row can only ever receive one label and the three windows cannot
    silently overlap.
    """
    day = pl.col(date_column)
    return frame.with_columns(
        pl.when(day.is_between(fold.train_start, fold.train_end))
        .then(pl.lit("train"))
        .when(day.is_between(fold.calibration_start, fold.calibration_end))
        .then(pl.lit("calibration"))
        .when(day.is_between(fold.test_start, fold.test_end))
        .then(pl.lit("test"))
        .otherwise(pl.lit("outside"))
        .alias("split")
    )


def min_date(frame: pl.DataFrame, column: str) -> date | None:
    """Earliest value of a date column, narrowed to ``date``.

    Polars types a column aggregate as a wide union, so every date comparison in
    this component would otherwise need an ignore comment. Narrowing once here
    keeps the comparisons honest and the type checker satisfied.
    """
    value = frame[column].min()
    return value if isinstance(value, date) else None


def max_date(frame: pl.DataFrame, column: str) -> date | None:
    """Latest value of a date column, narrowed to ``date``."""
    value = frame[column].max()
    return value if isinstance(value, date) else None


def _number(value: object, default: float = 0.0) -> float:
    """Narrow a polars aggregate to a float."""
    return float(value) if isinstance(value, int | float) else default


def _rate(frame: pl.DataFrame) -> float | None:
    if frame.height == 0:
        return None
    return round(_number(frame["target"].mean()), 6)


def fold_stats(frame: pl.DataFrame, fold: FoldSpec, *, date_column: str = "rd") -> FoldStats:
    """Measure one fold against the data: row counts, base rates, real capacity.

    ``test_median_daily_capacity`` is the median number of inspections Chicago
    actually performed on a day inside the test window. It is the honest source
    of ``k`` for precision@k -- inventing a round number would make the headline
    operational metric an assumption rather than a measurement.
    """
    labelled = assign_split(frame, fold, date_column=date_column)
    train = labelled.filter(pl.col("split") == "train")
    calibration = labelled.filter(pl.col("split") == "calibration")
    test = labelled.filter(pl.col("split") == "test")

    if test.height:
        per_day = test.group_by(date_column).len()
        median_capacity = _number(per_day["len"].median())
        days = int(per_day.height)
        establishments = int(test["establishment_id"].n_unique())
    else:
        median_capacity, days, establishments = None, 0, 0

    return FoldStats(
        fold_id=fold.fold_id,
        train_rows=train.height,
        calibration_rows=calibration.height,
        test_rows=test.height,
        train_positive_rate=_rate(train),
        calibration_positive_rate=_rate(calibration),
        test_positive_rate=_rate(test),
        test_establishments=establishments,
        test_days=days,
        test_median_daily_capacity=median_capacity,
    )


def window_frame(frame: pl.DataFrame, fold: FoldSpec, *, date_column: str = "rd") -> pl.DataFrame:
    """The rows of one fold's test window, in deterministic order.

    Named ``window_frame`` rather than ``test_frame`` because pytest collects
    any importable callable whose name starts with ``test_``, and a helper that
    a test module imports would be mistaken for a test case.

    Sorted by ``(reference date, target_inspection_id)`` so that every consumer
    -- schedules, curves, metrics -- starts from the same sequence and no result
    can depend on Parquet row order.
    """
    labelled = assign_split(frame, fold, date_column=date_column)
    return labelled.filter(pl.col("split") == "test").sort([date_column, "target_inspection_id"])


__all__ = [
    "CODE_ERA_ANCHOR",
    "COVID_SHIFT",
    "MIN_TRAIN_QUARTERS",
    "QUARTERLY",
    "add_quarters",
    "assign_split",
    "covid_shift_fold",
    "excluded_partial_windows",
    "fold_stats",
    "max_date",
    "min_date",
    "quarter_end",
    "quarter_key",
    "quarter_start",
    "quarterly_folds",
    "window_frame",
]
