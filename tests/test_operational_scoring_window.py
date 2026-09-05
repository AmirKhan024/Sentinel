"""The synthetic operational training window: construction and its guardrails."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.evaluation.folds import CODE_ERA_ANCHOR, COVID_SHIFT, QUARTERLY
from sentinel.operational_scoring.definitions import OPERATIONAL_FOLD_SET
from sentinel.operational_scoring.window import OperationalWindowError, build_operational_fold


def _features(dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"rd": [date.fromisoformat(d) for d in dates]})


def test_fold_set_is_never_a_real_evaluation_fold_set() -> None:
    assert OPERATIONAL_FOLD_SET not in (QUARTERLY, COVID_SHIFT)


def test_train_end_is_the_day_before_planning_date() -> None:
    features = _features(["2020-01-01", "2026-08-14"])
    fold = build_operational_fold(
        planning_date=date(2026, 8, 28), historical_features=features
    )
    assert fold.train_end == date(2026, 8, 14)  # bounded by available data, not planning_date - 1
    assert fold.train_start == CODE_ERA_ANCHOR
    assert fold.fold_set == OPERATIONAL_FOLD_SET
    assert fold.fold_id == "operational-2026-08-28"


def test_train_end_never_exceeds_available_data() -> None:
    features = _features(["2020-01-01", "2022-06-30"])
    fold = build_operational_fold(
        planning_date=date(2026, 8, 28), historical_features=features
    )
    assert fold.train_end == date(2022, 6, 30)


def test_train_end_is_capped_at_the_day_before_planning_date_even_with_later_data() -> None:
    """A planning date in the *past* relative to the data must not train on its future."""
    features = _features(["2020-01-01", "2026-08-14"])
    fold = build_operational_fold(
        planning_date=date(2022, 1, 1), historical_features=features
    )
    assert fold.train_end == date(2021, 12, 31)


def test_planning_date_on_the_code_era_anchor_is_refused() -> None:
    features = _features(["2018-07-01", "2020-01-01"])
    with pytest.raises(OperationalWindowError, match="no code-era training data"):
        build_operational_fold(planning_date=CODE_ERA_ANCHOR, historical_features=features)


def test_empty_historical_features_is_refused() -> None:
    empty = pl.DataFrame(schema={"rd": pl.Date})
    with pytest.raises(OperationalWindowError, match="no usable reference dates"):
        build_operational_fold(planning_date=date(2026, 8, 28), historical_features=empty)


def test_calibration_and_test_windows_never_overlap_train() -> None:
    features = _features(["2020-01-01", "2026-08-14"])
    fold = build_operational_fold(
        planning_date=date(2026, 8, 28), historical_features=features
    )
    assert fold.train_end < fold.calibration_start
    assert fold.calibration_end < fold.test_start
