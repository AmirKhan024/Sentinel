"""Which model, and which frozen calibrator. Reuses Component 13's rule; adds nothing.

``policy.select.select()`` is Component 13's frozen, pre-registered model-selection
rule -- the same one ``sentinel decide`` applies to build the historical recommendation
queue. Calling it here means operational mode and the historical recommendation queue
can never disagree about which model is "the" production model; there is one rule, read
in two places.

The composite name the rule returns (e.g. ``"xgboost_platt"``) is Component 9's
calibrated-model name, ``"<base>_<method>"`` by convention -- but the convention is
never parsed here. ``calibrated_predictions`` already carries a ``base_model_name``
column recording the split explicitly, and reading it is strictly safer than assuming a
naming pattern holds.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.operational_scoring.models import ProductionModelChoice
from sentinel.policy.definitions import CANDIDATE_MODELS, POLICY_DEFINITION_VERSION
from sentinel.policy.select import SelectionError, select


class ModelSelectionError(ValueError):
    """Raised when the production model, or its calibrator, cannot be resolved."""


def resolve_production_model(
    *,
    simulation: pl.DataFrame,
    metrics: pl.DataFrame,
    sensitivity: pl.DataFrame,
    calibrated_predictions: pl.DataFrame,
    models: Sequence[str] = CANDIDATE_MODELS,
) -> ProductionModelChoice:
    """Apply Component 13's selection rule, then resolve it to a base model + calibrator.

    ``models`` defaults to Component 13's own admissible candidate list -- production
    runs should never override it. It exists as a parameter (mirroring
    ``policy.select.select``'s own ``models`` parameter) so a test can exercise this
    function's calibrator-resolution logic against a small synthetic registry without
    needing real Component 5/9 artifacts for all four production candidates.

    The calibrator chosen is the **most recently fitted** one on record for the selected
    model -- the fold whose ``calibrator_fitted_through`` is latest -- because that is
    the calibration state closest to representing "now" among what this project has
    actually measured. It is still a historical fold's calibrator, and the manifest
    records exactly which one.
    """
    try:
        selection = select(
            simulation=simulation,
            metrics=metrics,
            sensitivity=sensitivity,
            definition_version=POLICY_DEFINITION_VERSION,
            models=models,
        )
    except SelectionError as exc:
        raise ModelSelectionError(f"model selection failed: {exc}") from exc

    rows = calibrated_predictions.filter(pl.col("model_name") == selection.model_name)
    if rows.is_empty():
        raise ModelSelectionError(
            f"selected model {selection.model_name!r} has no rows in the calibrated "
            "predictions artifact; the selection rule and the calibration artifact "
            "describe different runs"
        )
    # Two determinism hazards, both closed explicitly rather than trusted to a default:
    # ``unique()`` defaults to ``maintain_order=False``, under which the surviving row per
    # (fold_set, fold_id) is not guaranteed to be any particular one of the many duplicate
    # rows a fold contributes (one per scored test row) -- caught by a determinism check
    # that ran the same input twice and got two different calibrators back. And a sort key
    # of ``calibrator_fitted_through`` alone admits a genuine tie between two folds, which
    # ``fold_id`` (unique per row) breaks the same way on every run.
    latest = (
        rows.select(
            "base_model_name",
            "method",
            "fold_set",
            "fold_id",
            "calibrator_fitted_through",
        )
        .unique(subset=["fold_set", "fold_id"], maintain_order=True)
        .sort(["calibrator_fitted_through", "fold_id"], descending=[True, False])
        .row(0, named=True)
    )

    return ProductionModelChoice(
        composite_model_name=selection.model_name,
        base_model_name=str(latest["base_model_name"]),
        method=str(latest["method"]),
        calibration_fold_set=str(latest["fold_set"]),
        calibration_fold_id=str(latest["fold_id"]),
        decided_on_axis=selection.decided_on_axis,
        n_tied_on_nde=selection.n_tied_on_nde,
    )


__all__ = ["ModelSelectionError", "resolve_production_model"]
