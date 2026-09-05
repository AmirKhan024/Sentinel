"""Resolving Component 13's selected model to a base model, method and calibrator fold.

The tie-break test guards a real bug found while implementing this component: an
earlier version used ``DataFrame.unique()`` without ``maintain_order=True`` after a
``sort()``, which let two runs over byte-identical input pick two different folds'
calibrators. Caught by running the same operational scoring twice and comparing --
exactly the shape of test kept here.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.operational_scoring.selection import ModelSelectionError, resolve_production_model

# ``simulation``/``metrics``/``sensitivity`` are read from real artifacts whose
# ``model_name`` is already the composite (base + calibration method) name -- the same
# name ``calibrated_predictions.model_name`` carries. ``base_model_name`` is a separate
# column only ``calibrated_predictions`` has.
MODELS = ("model_a_platt", "model_b_platt")


def _simulation(nde: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_name": list(nde),
            "schedule_name": ["model"] * len(nde),
            "fold_set": ["quarterly"] * len(nde),
            "normalized_discovery_efficiency": list(nde.values()),
        }
    )


def _sensitivity(bands: dict[str, tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_name": list(bands),
            "fold_set": ["quarterly"] * len(bands),
            "p05": [low for low, _ in bands.values()],
            "p95": [high for _, high in bands.values()],
        }
    )


def _metrics(ece: dict[str, float], precision: dict[str, float]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for model, value in ece.items():
        rows.append(
            {
                "model_name": model,
                "fold_set": "quarterly",
                "metric": "ece",
                "k_name": "",
                "value": value,
            }
        )
    for model, value in precision.items():
        rows.append(
            {
                "model_name": model,
                "fold_set": "quarterly",
                "metric": "precision_at_k",
                "k_name": "k_1_day",
                "value": value,
            }
        )
    return pl.DataFrame(rows)


def _calibrated_predictions(rows: list[dict[str, object]]) -> pl.DataFrame:
    """A minimal, deliberately test-row-duplicated calibrated_predictions frame.

    Real Component 9 output has many physical rows per (model_name, fold_id) -- one per
    scored test row. Duplicated here on purpose, because the bug this test guards only
    manifests when there is more than one physical row to deduplicate.
    """
    out: list[dict[str, object]] = []
    for row in rows:
        for target_id in ("T1", "T2", "T3"):
            out.append({**row, "target_inspection_id": target_id})
    if not out:
        # A real, empty Component 9 artifact still carries its full schema (see
        # ``calibration.writer.empty``) -- never a columnless frame.
        return pl.DataFrame(
            schema={
                "model_name": pl.Utf8,
                "base_model_name": pl.Utf8,
                "method": pl.Utf8,
                "fold_set": pl.Utf8,
                "fold_id": pl.Utf8,
                "calibrator_fitted_through": pl.Date,
                "target_inspection_id": pl.Utf8,
            }
        )
    return pl.DataFrame(out)


def test_resolves_the_selected_models_most_recent_calibrator() -> None:
    simulation = _simulation({"model_a_platt": 0.30, "model_b_platt": 0.10})
    sensitivity = _sensitivity({"model_a_platt": (0.25, 0.35), "model_b_platt": (0.05, 0.15)})
    metrics = _metrics(
        ece={"model_a_platt": 0.05, "model_b_platt": 0.20},
        precision={"model_a_platt": 0.5, "model_b_platt": 0.3},
    )
    calibrated = _calibrated_predictions(
        [
            {
                "model_name": "model_a_platt",
                "base_model_name": "model_a",
                "method": "platt",
                "fold_set": "quarterly",
                "fold_id": "quarterly-2022Q2",
                "calibrator_fitted_through": date(2022, 3, 31),
            },
            {
                "model_name": "model_a_platt",
                "base_model_name": "model_a",
                "method": "platt",
                "fold_set": "quarterly",
                "fold_id": "quarterly-2022Q3",
                "calibrator_fitted_through": date(2022, 6, 30),
            },
        ]
    )
    choice = resolve_production_model(
        simulation=simulation,
        metrics=metrics,
        sensitivity=sensitivity,
        calibrated_predictions=calibrated,
        models=MODELS,
    )
    assert choice.composite_model_name == "model_a_platt"
    assert choice.base_model_name == "model_a"
    assert choice.method == "platt"
    # The later fold, not the first one encountered in the (duplicated) input.
    assert choice.calibration_fold_id == "quarterly-2022Q3"


def test_tied_calibrator_fitted_through_breaks_on_fold_id_deterministically() -> None:
    """Two folds tied on the freshest date; the tie-break must be the same every time."""
    simulation = _simulation({"model_a_platt": 0.30})
    sensitivity = _sensitivity({"model_a_platt": (0.25, 0.35)})
    metrics = _metrics(ece={"model_a_platt": 0.05}, precision={"model_a_platt": 0.5})
    tied_date = date(2022, 6, 30)
    calibrated = _calibrated_predictions(
        [
            {
                "model_name": "model_a_platt",
                "base_model_name": "model_a",
                "method": "platt",
                "fold_set": "quarterly",
                "fold_id": "quarterly-2022Q9",
                "calibrator_fitted_through": tied_date,
            },
            {
                "model_name": "model_a_platt",
                "base_model_name": "model_a",
                "method": "platt",
                "fold_set": "quarterly",
                "fold_id": "quarterly-2022Q3",
                "calibrator_fitted_through": tied_date,
            },
            {
                "model_name": "model_a_platt",
                "base_model_name": "model_a",
                "method": "platt",
                "fold_set": "quarterly",
                "fold_id": "quarterly-2022Q5",
                "calibrator_fitted_through": tied_date,
            },
        ]
    )
    results = [
        resolve_production_model(
            simulation=simulation,
            metrics=metrics,
            sensitivity=sensitivity,
            calibrated_predictions=calibrated,
            models=("model_a_platt",),
        )
        for _ in range(20)
    ]
    fold_ids = {r.calibration_fold_id for r in results}
    assert fold_ids == {"quarterly-2022Q3"}, (
        f"the tie-break must always pick the same fold; got {fold_ids} across 20 calls"
    )


def test_repeated_resolution_over_identical_input_is_always_identical() -> None:
    """The regression guard: same input, run many times, must never disagree."""
    simulation = _simulation({"model_a_platt": 0.30, "model_b_platt": 0.10})
    sensitivity = _sensitivity({"model_a_platt": (0.25, 0.35), "model_b_platt": (0.05, 0.15)})
    metrics = _metrics(
        ece={"model_a_platt": 0.05, "model_b_platt": 0.20},
        precision={"model_a_platt": 0.5, "model_b_platt": 0.3},
    )
    rows = [
        {
            "model_name": "model_a_platt",
            "base_model_name": "model_a",
            "method": "platt",
            "fold_set": "quarterly",
            "fold_id": f"quarterly-2022Q{i}",
            "calibrator_fitted_through": date(2022, i, 1),
        }
        for i in range(1, 5)
    ]
    calibrated = _calibrated_predictions(rows)

    results = [
        resolve_production_model(
            simulation=simulation,
            metrics=metrics,
            sensitivity=sensitivity,
            calibrated_predictions=calibrated,
            models=MODELS,
        )
        for _ in range(20)
    ]
    assert len({r.calibration_fold_id for r in results}) == 1
    assert results[0].calibration_fold_id == "quarterly-2022Q4"


def test_selected_model_absent_from_calibrated_predictions_is_refused() -> None:
    simulation = _simulation({"model_a_platt": 0.30})
    sensitivity = _sensitivity({"model_a_platt": (0.25, 0.35)})
    metrics = _metrics(ece={"model_a_platt": 0.05}, precision={"model_a_platt": 0.5})
    calibrated = _calibrated_predictions([])
    with pytest.raises(ModelSelectionError, match="no rows"):
        resolve_production_model(
            simulation=simulation,
            metrics=metrics,
            sensitivity=sensitivity,
            calibrated_predictions=calibrated,
            models=("model_a_platt",),
        )
