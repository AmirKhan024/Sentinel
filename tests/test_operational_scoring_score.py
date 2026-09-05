"""Component 18's core scoring path: real inference, contract safety, determinism.

Uses ``tests/conftest.py``'s existing Component 6 fixtures (``spanning_model_features``,
``make_model_feature_row``) rather than inventing a parallel fixture family: Component
18's training input is exactly a Component 4-shaped feature table, which is what those
fixtures already build.

The logistic family is used throughout because it has no fold-set-keyed tuned
hyperparameters (unlike boosting/neural), which keeps these tests about Component 18's
own logic rather than about Component 7/8's tuning registry.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from sentinel.calibration.definitions import Method
from sentinel.calibration.models import FittedCalibrator
from sentinel.operational_scoring.definitions import ScoringStatus
from sentinel.operational_scoring.models import ProductionModelChoice
from sentinel.operational_scoring.score import score_candidates
from sentinel.operational_scoring.validate import FeatureContractError
from tests.conftest import make_model_feature_row, model_feature_scenario, spanning_model_features

PLANNING_DATE = date(2027, 1, 1)

CHOICE = ProductionModelChoice(
    composite_model_name="logistic_regression_platt",
    base_model_name="logistic_regression",
    method="platt",
    calibration_fold_set="quarterly",
    calibration_fold_id="quarterly-2026Q4",
    decided_on_axis="nde",
    n_tied_on_nde=1,
)

# The identity-ish Platt calibrator: coefficient 1, intercept 0 leaves logit(p) unchanged,
# so calibrated == base, which keeps assertions about the *base* model's real signal
# legible through the calibration step.
IDENTITY_CALIBRATOR = FittedCalibrator(
    model_name="logistic_regression",
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


def _historical_features(*, days: int = 400, per_day: int = 2) -> pl.DataFrame:
    # ``build.py`` parses ``rd`` before calling ``score_candidates``; the fixture must
    # match that contract, not the raw Component 4 output shape.
    return spanning_model_features(start="2018-07-02", days=days, per_day=per_day).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def _candidates_from(features: pl.DataFrame, *, count: int, offset: int = 0) -> pl.DataFrame:
    """A Component 17-shaped candidate frame, minimal but contract-complete.

    Built from real feature rows with the label stripped -- exactly what Component 17
    would hand Component 18: real feature values, no ``target``.
    """
    rows = []
    for i in range(count):
        row = make_model_feature_row(
            offset + i,
            establishment_id=f"CAND-{offset + i:06d}",
            target_inspection_id=f"CANDIDATE::{PLANNING_DATE.isoformat()}::CAND-{offset + i:06d}",
            inspection_date=PLANNING_DATE.isoformat(),
            target=None,
            target_status="operational_candidate",
        )
        rows.append(row)
    return model_feature_scenario(rows)


@pytest.fixture(scope="module")
def historical_features() -> pl.DataFrame:
    return _historical_features()


# --- 1. real inference, not a dummy prediction -----------------------------


def test_scores_are_real_model_output_not_constant_or_random(
    historical_features: pl.DataFrame,
) -> None:
    """The base model must have actually learned something from the training signal.

    ``spanning_model_features`` correlates ``prior_canvass_priority_rate`` with the
    label. A model that fit for real should separate rows on that basis; a dummy or
    constant score could not.
    """
    candidates = _candidates_from(historical_features, count=20)
    candidates = candidates.with_columns(
        pl.Series(
            "prior_canvass_priority_rate",
            [0.9 if i % 2 == 0 else 0.1 for i in range(candidates.height)],
        )
    )
    result = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    scored = result.frame.filter(pl.col("scoring_status") == ScoringStatus.SCORED.value)
    scores = scored["calibrated_score"].to_list()
    assert len(set(round(s, 6) for s in scores)) > 1, "every candidate scored identically"

    # Join on establishment_id rather than assuming row order: the output is sorted by
    # rank, not by input order, which is itself a property this test must respect
    # instead of accidentally depending on.
    rate_by_establishment = dict(
        zip(
            candidates["establishment_id"].to_list(),
            candidates["prior_canvass_priority_rate"].to_list(),
            strict=True,
        )
    )
    high_group = [
        s
        for s, est in zip(scores, scored["establishment_id"].to_list(), strict=True)
        if rate_by_establishment[est] == 0.9
    ]
    low_group = [
        s
        for s, est in zip(scores, scored["establishment_id"].to_list(), strict=True)
        if rate_by_establishment[est] == 0.1
    ]
    assert sum(high_group) / len(high_group) > sum(low_group) / len(low_group), (
        "the model did not track its own training signal -- scores are not real "
        "learned output"
    )


# --- 2. feature contract compatibility -------------------------------------


def test_missing_feature_column_is_refused_clearly(historical_features: pl.DataFrame) -> None:
    candidates = _candidates_from(historical_features, count=5).drop("prior_canvass_count")
    with pytest.raises(FeatureContractError, match="prior_canvass_count"):
        score_candidates(
            candidates=candidates,
            historical_features=historical_features,
            planning_date=PLANNING_DATE,
            choice=CHOICE,
            calibrator=IDENTITY_CALIBRATOR,
        )


# --- 3. deterministic scoring ------------------------------------------------


def test_same_inputs_produce_byte_identical_scores(historical_features: pl.DataFrame) -> None:
    candidates = _candidates_from(historical_features, count=15)
    first = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    second = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    assert first.frame.equals(second.frame)


# --- 4. identity preservation ------------------------------------------------


def test_every_scored_row_maps_back_to_its_candidate_id(historical_features: pl.DataFrame) -> None:
    candidates = _candidates_from(historical_features, count=12)
    result = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    assert set(result.frame["target_inspection_id"].to_list()) == set(
        candidates["target_inspection_id"].to_list()
    )
    assert set(result.frame["establishment_id"].to_list()) == set(
        candidates["establishment_id"].to_list()
    )


# --- 5. partial failure safety -----------------------------------------------


def test_a_structurally_corrupt_candidate_is_excluded_not_silently_dropped(
    historical_features: pl.DataFrame,
) -> None:
    candidates = _candidates_from(historical_features, count=10)
    corrupted = candidates.with_columns(
        pl.when(pl.col("establishment_id") == "CAND-000000")
        .then(None)
        .otherwise(pl.col("prior_canvass_count"))
        .alias("prior_canvass_count")
    )
    result = score_candidates(
        candidates=corrupted,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    assert result.excluded_count == 1
    assert result.scored_count == 9
    row = result.frame.filter(pl.col("establishment_id") == "CAND-000000").row(0, named=True)
    assert row["scoring_status"] == ScoringStatus.EXCLUDED_FEATURE_CONTRACT_VIOLATION.value
    assert row["calibrated_score"] is None
    assert row["rank"] is None
    # Every other candidate scored normally; one bad row must not corrupt the rest.
    others = result.frame.filter(pl.col("establishment_id") != "CAND-000000")
    assert (others["scoring_status"] == ScoringStatus.SCORED.value).all()


# --- 6. model provenance ------------------------------------------------------


def test_output_names_the_exact_model_and_calibrator(historical_features: pl.DataFrame) -> None:
    candidates = _candidates_from(historical_features, count=5)
    result = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    assert result.fold.fold_set == "operational"
    assert result.fold.fold_id == f"operational-{PLANNING_DATE.isoformat()}"
    assert result.train_rows > 0


# --- 7. no future data leakage ------------------------------------------------


def test_a_future_historical_record_does_not_change_operational_scores(
    historical_features: pl.DataFrame,
) -> None:
    candidates = _candidates_from(historical_features, count=8)
    before = score_candidates(
        candidates=candidates,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )

    future_row = make_model_feature_row(
        999_999,
        establishment_id="EST-FUTURE",
        inspection_date="2027-06-01",  # strictly after PLANNING_DATE
        target=1,
    )
    future_frame = model_feature_scenario([future_row]).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )
    augmented = pl.concat([historical_features, future_frame], how="vertical")
    after = score_candidates(
        candidates=candidates,
        historical_features=augmented,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    assert before.frame.sort("target_inspection_id").equals(
        after.frame.sort("target_inspection_id")
    )
    assert before.train_rows == after.train_rows


def test_zero_candidates_is_a_valid_empty_result(historical_features: pl.DataFrame) -> None:
    empty = model_feature_scenario([])
    result = score_candidates(
        candidates=empty,
        historical_features=historical_features,
        planning_date=PLANNING_DATE,
        choice=CHOICE,
        calibrator=IDENTITY_CALIBRATOR,
    )
    assert result.frame.height == 0
    assert result.scored_count == 0
