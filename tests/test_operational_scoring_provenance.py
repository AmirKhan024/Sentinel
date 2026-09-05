"""The Component 18 provenance patch: hyperparameter source is explicit, not implicit.

Before this patch, ``operational_scoring.fit.TUNING_FOLD_SET`` (the "quarterly" borrow)
was discoverable only by reading a module docstring. These tests prove the manifest-level
object (``ScoringResult.hyperparameter_provenance``, surfaced into
``OperationalScoringManifest`` by ``build.py``) states it explicitly, and that what it
states is exactly the configuration that was actually used -- not a description that
could silently drift from the real values.
"""

from __future__ import annotations

from datetime import date

import pytest

from sentinel.boosting.definitions import estimator_params as boosting_estimator_params
from sentinel.boosting.definitions import spec_for as boosting_spec_for
from sentinel.calibration.definitions import Family, Method
from sentinel.calibration.models import FittedCalibrator
from sentinel.modeling.definitions import spec_for as modeling_spec_for
from sentinel.operational_scoring.fit import TUNING_FOLD_SET
from sentinel.operational_scoring.models import ProductionModelChoice
from sentinel.operational_scoring.score import score_candidates
from tests.test_operational_scoring_score import (
    PLANNING_DATE,
    _candidates_from,
    _historical_features,
)

LOGISTIC_CHOICE = ProductionModelChoice(
    composite_model_name="logistic_regression_platt",
    base_model_name="logistic_regression",
    method="platt",
    calibration_fold_set="quarterly",
    calibration_fold_id="quarterly-2026Q4",
    decided_on_axis="nde",
    n_tied_on_nde=1,
)

BOOSTED_CHOICE = ProductionModelChoice(
    composite_model_name="xgboost_platt",
    base_model_name="xgboost",
    method="platt",
    calibration_fold_set="quarterly",
    calibration_fold_id="quarterly-2026Q4",
    decided_on_axis="ece",
    n_tied_on_nde=4,
)


def _calibrator(model_name: str) -> FittedCalibrator:
    return FittedCalibrator(
        model_name=model_name,
        fold_set="quarterly",
        fold_id="quarterly-2026Q4",
        method=Method.PLATT,
        estimator=None,
        input_transform="logit(p)",
        fit_rows=100,
        fit_positive_rate=0.5,
        fit_start=date(2026, 10, 1),
        fit_end=date(2026, 12, 31),
        coefficient=1.0,
        intercept=0.0,
    )


@pytest.fixture(scope="module")
def historical_features():
    return _historical_features()


# --- 1 & 2. provenance is present and matches the real configuration ------


def test_logistic_provenance_has_no_borrowed_fold_set(historical_features) -> None:
    """Component 6 has no tuning stage at all -- there is nothing to borrow."""
    candidates = _candidates_from(historical_features, count=10)
    result = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=LOGISTIC_CHOICE,
        calibrator=_calibrator("logistic_regression"),
    )
    assert result.model_family is Family.LOGISTIC
    prov = result.hyperparameter_provenance
    assert prov.fold_set is None
    assert "no hyperparameter search stage" in prov.source

    real_params = modeling_spec_for("logistic_regression").params
    assert prov.values == {k: str(v) for k, v in real_params.items()}


def test_boosted_provenance_names_the_borrowed_quarterly_fold_set(historical_features) -> None:
    """The exact case this patch exists for: an explicit, checkable borrow, not a guess."""
    candidates = _candidates_from(historical_features, count=10)
    result = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=BOOSTED_CHOICE,
        calibrator=_calibrator("xgboost"),
    )
    assert result.model_family is Family.BOOSTED
    prov = result.hyperparameter_provenance
    assert prov.fold_set == TUNING_FOLD_SET == "quarterly"
    assert "TUNED_PARAMS" in prov.source
    assert "no separate operational tuning study exists" in prov.source

    # The declared values are not a description -- they are read back from
    # boosting.definitions and must match exactly what fit_and_score actually passed
    # to the estimator constructor.
    real_params = boosting_estimator_params(boosting_spec_for("xgboost"), TUNING_FOLD_SET)
    assert prov.values == {k: str(v) for k, v in real_params.items()}
    assert prov.values["n_estimators"] == "103"  # the frozen study result, read verbatim


def test_empty_candidate_set_reports_provenance_as_not_applicable(historical_features) -> None:
    from tests.conftest import model_feature_scenario

    empty = model_feature_scenario([])
    result = score_candidates(
        candidates=empty,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=BOOSTED_CHOICE,
        calibrator=_calibrator("xgboost"),
    )
    assert result.hyperparameter_provenance.fold_set is None
    assert result.hyperparameter_provenance.values == {}
    # The family is still reported: it is a registry lookup, not a fit.
    assert result.model_family is Family.BOOSTED


# --- 3. deterministic scoring is unaffected by the patch ------------------


def test_hyperparameter_provenance_is_itself_deterministic(historical_features) -> None:
    candidates = _candidates_from(historical_features, count=8)
    first = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=BOOSTED_CHOICE,
        calibrator=_calibrator("xgboost"),
    )
    second = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=BOOSTED_CHOICE,
        calibrator=_calibrator("xgboost"),
    )
    assert first.hyperparameter_provenance == second.hyperparameter_provenance
    assert first.frame.equals(second.frame)
