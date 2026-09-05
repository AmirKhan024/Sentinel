"""The pure orchestration: candidates + a chosen model + a frozen calibrator -> a ranked set.

No filesystem access, no clock -- every input is already loaded, matching the project's
consistent split between a pure orchestration module and the ``build.py`` that reads and
writes on its behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from sentinel.calibration.definitions import Family
from sentinel.calibration.models import FittedCalibrator
from sentinel.calibration.predict import CalibrationPredictError
from sentinel.calibration.predict import apply as apply_calibration
from sentinel.evaluation.models import TIE_BREAK_COLUMN, FoldSpec
from sentinel.features.models import ValidationCheck
from sentinel.operational_scoring import fit, validate, window
from sentinel.operational_scoring.definitions import ScoringStatus
from sentinel.operational_scoring.models import HyperparameterProvenance, ProductionModelChoice
from sentinel.policy.eligibility import annotate as annotate_eligibility


class OperationalScoringError(RuntimeError):
    """Raised when the candidate set cannot be scored at all."""


@dataclass
class ScoringResult:
    """Everything ``build.py`` needs to write the artifact and the manifest."""

    frame: pl.DataFrame
    fold: FoldSpec
    checks: list[ValidationCheck]
    train_rows: int
    train_positive_rate: float | None
    scored_count: int
    excluded_count: int
    coverage_eligible_count: int
    model_family: Family
    hyperparameter_provenance: HyperparameterProvenance


def score_candidates(
    *,
    candidates: pl.DataFrame,
    historical_features: pl.DataFrame,
    planning_date: date,
    choice: ProductionModelChoice,
    calibrator: FittedCalibrator,
) -> ScoringResult:
    """Score, calibrate and rank one Component 17 candidate table.

    ``candidates`` is Component 17's output, unmodified. ``historical_features`` is
    Component 4's real, labelled feature table -- the only thing in this call with a
    ``target`` column -- used solely to fit the operational model; it is never itself
    scored or written to the output.
    """
    checks: list[ValidationCheck] = []

    valid, excluded, contract_checks = validate.check_feature_contract(candidates)
    checks.extend(contract_checks)

    annotated_all = annotate_eligibility(candidates)
    eligible_by_id = dict(
        zip(
            annotated_all["target_inspection_id"].to_list(),
            annotated_all["coverage_eligible"].to_list(),
            strict=True,
        )
    )
    secondary_by_id = dict(
        zip(
            annotated_all["target_inspection_id"].to_list(),
            annotated_all["secondary_no_history"].to_list(),
            strict=True,
        )
    )

    fold = window.build_operational_fold(
        planning_date=planning_date, historical_features=historical_features
    )
    checks.append(validate.check_planning_date_matches_fold(planning_date, fold))
    checks.append(validate.check_no_future_leakage(fold, historical_features))

    train_rows = 0
    train_positive_rate: float | None = None
    scored_frame = pl.DataFrame(
        schema={
            "target_inspection_id": pl.Utf8,
            "base_score": pl.Float64,
            "calibrated_score": pl.Float64,
        }
    )
    # Resolved unconditionally: a registry lookup, not a fit, so the family is known
    # (and reportable) even when there is nothing to score.
    family = fit.family_of(choice.base_model_name)
    hyperparameter_provenance = HyperparameterProvenance(
        fold_set=None, source="not applicable -- no candidates were scored", values={}
    )
    if valid.height > 0:
        from sentinel.modeling.train import training_frame

        train = training_frame(historical_features, fold)
        train_rows = train.height
        if train_rows:
            positive = train["target"].sum()
            train_positive_rate = float(positive) / train_rows if train_rows else None

        try:
            ids, base_scores, hyperparameter_provenance = fit.fit_and_score(
                base_model_name=choice.base_model_name,
                family=family,
                historical_features=historical_features,
                fold=fold,
                candidates=valid,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised typed, never swallowed
            raise OperationalScoringError(
                f"{choice.base_model_name}: operational fit/score failed for planning_date "
                f"{planning_date.isoformat()}: {exc}"
            ) from exc

        checks.append(
            validate.check_identity_preservation(valid["target_inspection_id"].to_list(), ids)
        )

        try:
            calibrated = apply_calibration(calibrator, base_scores)
        except CalibrationPredictError as exc:
            raise OperationalScoringError(f"calibration failed: {exc}") from exc
        checks.append(validate.check_scores_are_probabilities(calibrated))

        scored_frame = pl.DataFrame(
            {
                "target_inspection_id": ids,
                "base_score": base_scores,
                "calibrated_score": calibrated,
            }
        )

    # Deterministic rank: SCORE_DIRECTION and TIE_BREAK_COLUMN are Component 5's own
    # declared semantics, reused verbatim rather than re-declared.
    if not scored_frame.is_empty():
        ranked = scored_frame.sort(
            ["calibrated_score", TIE_BREAK_COLUMN], descending=[True, False]
        ).with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
        checks.append(validate.check_rank_is_a_permutation(ranked["rank"].to_list()))
    else:
        ranked = scored_frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("rank"))

    # Everything Component 17 carried beyond the raw feature columns and the label
    # placeholders -- establishment id, as-of location/name, provenance -- joined back
    # onto the scored/excluded rows. The Component 4 feature values themselves are not
    # repeated in the output: they were the model's input, not part of the priority
    # signal a later component needs to read.
    from sentinel.features.definitions import FEATURE_COLUMNS

    display_columns = [
        c
        for c in candidates.columns
        if c not in FEATURE_COLUMNS and c not in ("target", "target_status")
    ]
    display = candidates.select(display_columns)

    out = display.join(ranked, on="target_inspection_id", how="left").with_columns(
        pl.when(pl.col("calibrated_score").is_not_null())
        .then(pl.lit(ScoringStatus.SCORED.value))
        .otherwise(pl.lit(ScoringStatus.EXCLUDED_FEATURE_CONTRACT_VIOLATION.value))
        .alias("scoring_status"),
        pl.col("target_inspection_id")
        .replace_strict(eligible_by_id, default=None, return_dtype=pl.Boolean)
        .alias("coverage_eligible"),
        pl.col("target_inspection_id")
        .replace_strict(secondary_by_id, default=None, return_dtype=pl.Boolean)
        .alias("secondary_no_history"),
    )

    coverage_eligible_count = int(out.filter(pl.col("coverage_eligible")).height)

    out = out.sort([pl.col("rank").is_null(), "rank", "target_inspection_id"])

    return ScoringResult(
        frame=out,
        fold=fold,
        checks=checks,
        train_rows=train_rows,
        train_positive_rate=train_positive_rate,
        scored_count=ranked.height if not scored_frame.is_empty() else 0,
        excluded_count=excluded.height,
        coverage_eligible_count=coverage_eligible_count,
        model_family=family,
        hyperparameter_provenance=hyperparameter_provenance,
    )


__all__ = ["OperationalScoringError", "ScoringResult", "score_candidates"]
