"""Disparity and drift: four measures instead of one score, and a trend only when earned.

The two modules tested here are where a fairness audit is most tempted to over-claim. A single
"fairness score" would hide a weighting of incompatible criteria; a trend line through two
points would turn a shortage of data into a finding. Both are refused, and these tests pin the
refusals rather than the conveniences.
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from sentinel.fairness import disparity, drift
from sentinel.fairness.definitions import (
    DRIFT_MIN_FOLDS,
    FAIRNESS_DEFINITION_VERSION,
    Grain,
    GroupStatus,
)
from sentinel.fairness.models import GroupSupport


def _comparable(rows: list[tuple[str, float | None, int, str]], **overrides: str) -> pl.DataFrame:
    """``(group, value, n_rows, status)`` in one comparable cell."""
    base: dict[str, str] = {
        "model_name": "xgboost_platt",
        "stage": "calibrated",
        "group_definition": "community_area",
        "grain": Grain.FOLD_SET.value,
        "fold_set": "quarterly",
        "fold_id": "",
        "metric": "ece",
        "k_name": "",
    }
    base.update(overrides)
    return pl.DataFrame(
        {
            **{key: [value] * len(rows) for key, value in base.items()},
            "group_value": [r[0] for r in rows],
            "value": [r[1] for r in rows],
            "n_rows": [r[2] for r in rows],
            "group_status": [r[3] for r in rows],
        }
    )


SUPPORTED = GroupStatus.SUPPORTED.value
UNSUPPORTED = GroupStatus.INSUFFICIENT_SUPPORT.value


# --- 1. the four measures --------------------------------------------------------


def test_four_measures_are_emitted_per_cell_and_none_is_a_summary_score() -> None:
    rows = disparity.summarise(
        _comparable([("A", 0.02, 400, SUPPORTED), ("B", 0.10, 600, SUPPORTED)]),
        {("xgboost_platt", "calibrated", "quarterly", "", "ece"): 0.05},
    )
    assert {r["measure"] for r in rows} == {"spread", "ratio", "max_deviation", "weighted_sd"}
    assert len(rows) == 4


def test_the_measures_match_arithmetic_done_by_hand() -> None:
    """A 0.02 on 400 rows and a 0.10 on 600, against a pooled reference of 0.05."""
    rows = {
        r["measure"]: r["value"]
        for r in disparity.summarise(
            _comparable([("A", 0.02, 400, SUPPORTED), ("B", 0.10, 600, SUPPORTED)]),
            {("xgboost_platt", "calibrated", "quarterly", "", "ece"): 0.05},
        )
    }
    assert rows["spread"] == pytest.approx(0.08)
    assert rows["ratio"] == pytest.approx(5.0)
    assert rows["max_deviation"] == pytest.approx(0.05)


def test_the_extremes_carry_their_row_counts() -> None:
    """A dramatic ratio must never be quotable without its support in the same record."""
    row = disparity.summarise(
        _comparable([("A", 0.02, 400, SUPPORTED), ("B", 0.10, 250, SUPPORTED)]), {}
    )[0]
    assert (row["max_group"], row["max_group_rows"]) == ("B", 250)
    assert (row["min_group"], row["min_group_rows"]) == ("A", 400)


# --- 2. unsupported groups are excluded from the maths and counted in the row ------


def test_unsupported_groups_are_excluded_and_counted() -> None:
    """A spread over 2 of 3 groups is a different claim from one over all 3."""
    rows = disparity.summarise(
        _comparable(
            [
                ("A", 0.02, 400, SUPPORTED),
                ("B", 0.10, 600, SUPPORTED),
                ("C", 0.90, 12, UNSUPPORTED),
            ]
        ),
        {},
    )
    spread = next(r for r in rows if r["measure"] == "spread")
    assert spread["n_groups_supported"] == 2
    assert spread["n_groups_unsupported"] == 1
    # C's 0.90 must not reach the arithmetic.
    assert spread["value"] == pytest.approx(0.08)


def test_a_null_value_on_a_supported_group_is_excluded_too() -> None:
    rows = disparity.summarise(
        _comparable([("A", 0.02, 400, SUPPORTED), ("B", None, 600, SUPPORTED)]), {}
    )
    spread = next(r for r in rows if r["measure"] == "spread")
    assert spread["n_groups_supported"] == 1
    assert spread["value"] is None


# --- 3. undefined measures say why -----------------------------------------------


def test_one_supported_group_yields_nulls_with_a_stated_reason() -> None:
    rows = disparity.summarise(_comparable([("A", 0.02, 400, SUPPORTED)]), {})
    for row in rows:
        assert row["value"] is None
        assert "a disparity needs at least two" in str(row["undefined_reason"])


def test_a_zero_minimum_makes_the_ratio_null_and_says_so() -> None:
    """A vanished denominator is not an infinite disparity."""
    rows = disparity.summarise(
        _comparable([("A", 0.0, 400, SUPPORTED), ("B", 0.10, 600, SUPPORTED)]), {}
    )
    ratio = next(r for r in rows if r["measure"] == "ratio")
    assert ratio["value"] is None
    assert "not an infinite disparity" in str(ratio["undefined_reason"])
    # The spread is still perfectly well defined on the same values.
    assert next(r for r in rows if r["measure"] == "spread")["value"] == pytest.approx(0.10)


def test_a_missing_reference_makes_only_the_deviation_null() -> None:
    rows = {
        r["measure"]: r
        for r in disparity.summarise(
            _comparable([("A", 0.02, 400, SUPPORTED), ("B", 0.10, 600, SUPPORTED)]), {}
        )
    }
    assert rows["max_deviation"]["value"] is None
    assert "no pooled reference" in str(rows["max_deviation"]["undefined_reason"])
    assert rows["spread"]["value"] is not None


def test_a_defined_measure_carries_no_undefined_reason() -> None:
    rows = disparity.summarise(
        _comparable([("A", 0.02, 400, SUPPORTED), ("B", 0.10, 600, SUPPORTED)]), {}
    )
    spread = next(r for r in rows if r["measure"] == "spread")
    assert spread["undefined_reason"] == ""


# --- 4. cells are never mixed ------------------------------------------------------


def test_two_metrics_are_summarised_as_two_cells_not_one() -> None:
    frame = pl.concat(
        [
            _comparable([("A", 0.02, 400, SUPPORTED), ("B", 0.10, 600, SUPPORTED)], metric="ece"),
            _comparable(
                [("A", 0.60, 400, SUPPORTED), ("B", 0.65, 600, SUPPORTED)], metric="roc_auc"
            ),
        ]
    )
    rows = disparity.summarise(frame, {})
    assert len({r["metric"] for r in rows}) == 2
    assert len(rows) == 8


def test_a_frame_missing_a_comparable_column_is_rejected() -> None:
    frame = _comparable([("A", 0.02, 400, SUPPORTED)]).drop("n_rows")
    with pytest.raises(disparity.DisparityError, match="missing n_rows"):
        disparity.summarise(frame, {})


def test_an_empty_frame_produces_no_rows() -> None:
    assert disparity.summarise(_comparable([]), {}) == []


# --- 5. drift: a trend only when there are folds to see one in ---------------------


def _fold_disparities(values: list[float | None]) -> pl.DataFrame:
    n = len(values)
    return pl.DataFrame(
        {
            "model_name": ["xgboost_platt"] * n,
            "stage": ["calibrated"] * n,
            "group_definition": ["community_area"] * n,
            "fold_set": ["quarterly"] * n,
            "metric": ["ece"] * n,
            "k_name": [""] * n,
            "measure": ["spread"] * n,
            "grain": [Grain.FOLD.value] * n,
            "fold_id": [f"quarterly-20{22 + i // 4}Q{i % 4 + 1}" for i in range(n)],
            "value": values,
        }
    )


def test_fewer_than_three_measured_folds_is_insufficient_rather_than_a_trend() -> None:
    """Two points are a line through any two numbers."""
    row = drift.series(_fold_disparities([0.02, 0.10]))[0]
    assert row["trend"] == drift.TREND_INSUFFICIENT
    assert row["relative_change"] is None
    assert row["folds_measured"] == 2


def test_a_series_that_grew_by_more_than_the_threshold_is_widening() -> None:
    row = drift.series(_fold_disparities([0.02, 0.03, 0.04, 0.10]))[0]
    assert row["trend"] == drift.TREND_WIDENING
    assert row["relative_change"] == pytest.approx((0.10 - 0.02) / 0.02)


def test_a_series_that_shrank_is_narrowing() -> None:
    row = drift.series(_fold_disparities([0.10, 0.08, 0.05, 0.02]))[0]
    assert row["trend"] == drift.TREND_NARROWING


def test_a_flat_series_is_stable() -> None:
    row = drift.series(_fold_disparities([0.05, 0.051, 0.049, 0.05]))[0]
    assert row["trend"] == drift.TREND_STABLE


def test_null_folds_are_excluded_from_the_statistics_and_still_counted() -> None:
    """So a record says both what was measured and what was available."""
    row = drift.series(_fold_disparities([0.02, None, 0.04, None, 0.10]))[0]
    assert row["folds_measured"] == 3
    assert row["folds_total"] == 5
    assert row["first_spread"] == pytest.approx(0.02)
    assert row["last_spread"] == pytest.approx(0.10)


def test_the_pooled_grain_contributes_no_series() -> None:
    """A pooled row is one number and has no series."""
    pooled = _fold_disparities([0.02, 0.05, 0.09]).with_columns(
        pl.lit(Grain.FOLD_SET.value).alias("grain")
    )
    assert drift.series(pooled) == []


def test_a_series_starting_at_exactly_zero_reports_a_null_change_not_infinity() -> None:
    row = drift.series(_fold_disparities([0.0, 0.05, 0.09]))[0]
    assert row["relative_change"] is None
    assert row["trend"] == drift.TREND_WIDENING


def test_the_fold_sd_is_a_spread_and_the_threshold_needs_three_folds() -> None:
    assert drift.sample_sd([0.4]) is None
    assert drift.sample_sd([0.2, 0.4]) == pytest.approx(0.1414213562, rel=1e-6)
    assert DRIFT_MIN_FOLDS >= 3


def test_every_drift_row_carries_the_definition_version() -> None:
    row = drift.series(_fold_disparities([0.02, 0.05, 0.09]))[0]
    assert row["fairness_definition_version"] == FAIRNESS_DEFINITION_VERSION


# --- 6. representation travel, the context a drift claim needs ---------------------


def _support_records(shares: list[float]) -> list[GroupSupport]:
    return [
        GroupSupport(
            group_definition="community_area",
            group_value="A",
            grain=Grain.FOLD.value,
            fold_set="quarterly",
            fold_id=f"quarterly-2022Q{i + 1}",
            n_rows=100,
            n_positive=50,
            n_negative=50,
            base_rate=0.5,
            representation_share=share,
            ranking_status=GroupStatus.SUPPORTED,
            calibration_status=GroupStatus.SUPPORTED,
            insufficient_reason="",
        )
        for i, share in enumerate(shares)
    ]


def test_representation_travel_reports_the_range_and_the_folds_present() -> None:
    travel = drift.representation_travel(_support_records([0.05, 0.15, 0.10]))
    low, high, span, folds = travel["A"]
    assert (low, high, folds) == (0.05, 0.15, 3)
    assert span == pytest.approx(0.10)


def test_pooled_support_records_are_excluded_from_the_travel() -> None:
    """A pooled share is one number and has no travel."""
    pooled = [
        dataclasses.replace(record, grain=Grain.FOLD_SET.value)
        for record in _support_records([0.05, 0.15])
    ]
    assert drift.representation_travel(pooled) == {}


def test_a_moving_population_becomes_an_advisory_line() -> None:
    notes = drift.advisory_lines(
        [],
        drift.representation_travel(_support_records([0.02, 0.20])),
        representation_threshold=0.05,
    )
    assert any("two candidate explanations" in note for note in notes)


def test_covid_is_recognised_as_the_fold_set_that_is_never_pooled() -> None:
    assert drift.covid_is_separate("covid_shift")
    assert not drift.covid_is_separate("quarterly")
