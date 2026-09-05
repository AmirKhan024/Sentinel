"""Loading a frozen calibrator from Component 9's persisted parameters. No fitting."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from sentinel.calibration.definitions import Method
from sentinel.calibration.predict import apply as apply_calibration
from sentinel.operational_scoring.calibrator import CalibratorLoadError, load_frozen_calibrator

COMMON = {
    "model_name": "xgboost",
    "fold_set": "quarterly",
    "fold_id": "quarterly-2022Q2",
    "fit_rows": 1000,
    "fit_positive_rate": 0.4,
    "fit_start": date(2021, 10, 1),
    "fit_end": date(2021, 12, 31),
    "was_selected": True,
    "calibration_definition_version": "v1",
}


def _parameters(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _breakpoints(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_loads_a_platt_calibrator_and_reproduces_its_mapping(tmp_path: Path) -> None:
    params = _parameters(
        [
            {
                **COMMON,
                "method": "platt",
                "input_transform": "logit(p)",
                "term": "coef",
                "value": 1.5,
            },
            {
                **COMMON,
                "method": "platt",
                "input_transform": "logit(p)",
                "term": "intercept",
                "value": -0.2,
            },
        ]
    )
    params_path = tmp_path / "params.parquet"
    breakpoints_path = tmp_path / "breakpoints.parquet"
    params.write_parquet(params_path)
    _breakpoints([]).write_parquet(breakpoints_path)

    calibrator = load_frozen_calibrator(
        base_model_name="xgboost",
        method="platt",
        fold_set="quarterly",
        fold_id="quarterly-2022Q2",
        parameters_path=params_path,
        breakpoints_path=breakpoints_path,
    )
    assert calibrator.method is Method.PLATT
    assert calibrator.coefficient == pytest.approx(1.5)
    assert calibrator.intercept == pytest.approx(-0.2)
    assert calibrator.estimator is None

    scores = apply_calibration(calibrator, [0.5])
    assert 0.0 <= scores[0] <= 1.0


def test_loads_an_isotonic_calibrator_and_reproduces_its_mapping(tmp_path: Path) -> None:
    params = _parameters(
        [
            {
                **COMMON,
                "method": "isotonic",
                "input_transform": "p",
                "term": "breakpoint_count",
                "value": 3.0,
            }
        ]
    )
    breakpoints = _breakpoints(
        [
            {
                "model_name": "xgboost",
                "fold_set": "quarterly",
                "fold_id": "quarterly-2022Q2",
                "breakpoint_index": i,
                "x_threshold": x,
                "y_threshold": y,
                "x_min": 0.0,
                "x_max": 1.0,
                "breakpoint_count": 3,
                "was_selected": True,
                "calibration_definition_version": "v1",
            }
            for i, (x, y) in enumerate([(0.0, 0.1), (0.5, 0.5), (1.0, 0.9)])
        ]
    )
    params_path = tmp_path / "params.parquet"
    breakpoints_path = tmp_path / "breakpoints.parquet"
    params.write_parquet(params_path)
    breakpoints.write_parquet(breakpoints_path)

    calibrator = load_frozen_calibrator(
        base_model_name="xgboost",
        method="isotonic",
        fold_set="quarterly",
        fold_id="quarterly-2022Q2",
        parameters_path=params_path,
        breakpoints_path=breakpoints_path,
    )
    assert calibrator.method is Method.ISOTONIC
    assert calibrator.x_thresholds == (0.0, 0.5, 1.0)
    assert calibrator.y_thresholds == (0.1, 0.5, 0.9)

    scores = apply_calibration(calibrator, [0.5])
    assert scores[0] == pytest.approx(0.5)


def test_missing_method_for_the_requested_fold_is_refused(tmp_path: Path) -> None:
    params = _parameters(
        [{**COMMON, "method": "platt", "input_transform": "logit(p)", "term": "coef", "value": 1.0}]
    )
    params_path = tmp_path / "params.parquet"
    breakpoints_path = tmp_path / "breakpoints.parquet"
    params.write_parquet(params_path)
    _breakpoints([]).write_parquet(breakpoints_path)

    with pytest.raises(CalibratorLoadError, match="no 'isotonic' calibrator"):
        load_frozen_calibrator(
            base_model_name="xgboost",
            method="isotonic",
            fold_set="quarterly",
            fold_id="quarterly-2022Q2",
            parameters_path=params_path,
            breakpoints_path=breakpoints_path,
        )
