"""Component 18's own validation checks, tested directly and in isolation."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.evaluation.models import FoldSpec
from sentinel.operational_scoring import validate
from tests.conftest import make_model_feature_row, model_feature_scenario


def _candidates(count: int) -> pl.DataFrame:
    rows = [
        make_model_feature_row(
            i,
            establishment_id=f"EST-{i}",
            target=None,
            target_status="operational_candidate",
        )
        for i in range(count)
    ]
    return model_feature_scenario(rows)


def test_missing_declared_column_raises_a_whole_table_error() -> None:
    frame = _candidates(3).drop("prior_canvass_fail_rate")
    with pytest.raises(validate.FeatureContractError, match="prior_canvass_fail_rate"):
        validate.check_feature_contract(frame)


def test_never_null_violation_excludes_only_that_row() -> None:
    frame = _candidates(4)
    corrupted = frame.with_columns(
        pl.when(pl.col("establishment_id") == "EST-1")
        .then(None)
        .otherwise(pl.col("prior_inspection_count_any_type"))
        .alias("prior_inspection_count_any_type")
    )
    valid, excluded, checks = validate.check_feature_contract(corrupted)
    assert excluded.height == 1
    assert excluded["establishment_id"][0] == "EST-1"
    assert valid.height == 3
    assert not validate.has_failures(checks)  # a warn, not an error -- the run still proceeds


def test_empty_candidate_table_is_not_an_error() -> None:
    empty = _candidates(0)
    valid, excluded, checks = validate.check_feature_contract(empty)
    assert valid.height == 0
    assert excluded.height == 0
    assert not validate.has_failures(checks)


def test_identity_preservation_catches_a_dropped_or_substituted_id() -> None:
    check = validate.check_identity_preservation(["A", "B", "C"], ["A", "B"])
    assert not check.passed
    check_ok = validate.check_identity_preservation(["A", "B"], ["B", "A"])
    assert check_ok.passed


def test_scores_outside_zero_one_are_rejected() -> None:
    assert validate.check_scores_are_probabilities([0.0, 0.5, 1.0]).passed
    assert not validate.check_scores_are_probabilities([0.5, 1.2]).passed


def test_rank_permutation_check() -> None:
    assert validate.check_rank_is_a_permutation([1, 2, 3]).passed
    assert not validate.check_rank_is_a_permutation([1, 1, 3]).passed
    assert not validate.check_rank_is_a_permutation([1, 2, 4]).passed


def test_no_future_leakage_check_flags_a_row_on_or_after_train_end() -> None:
    fold = FoldSpec(
        fold_set="operational",
        fold_id="operational-2027-01-01",
        train_start=date(2018, 7, 1),
        train_end=date(2026, 12, 31),
        calibration_start=date(2027, 1, 1),
        calibration_end=date(2027, 1, 1),
        test_start=date(2027, 1, 2),
        test_end=date(2027, 1, 2),
    )
    clean = pl.DataFrame({"rd": [date(2020, 1, 1), date(2026, 12, 31)]})
    assert validate.check_no_future_leakage(fold, clean).passed

    dirty = pl.DataFrame({"rd": [date(2020, 1, 1), date(2027, 1, 1)]})
    assert not validate.check_no_future_leakage(fold, dirty).passed
