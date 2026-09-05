"""Pre-scoring and post-scoring checks. Every one independently re-derived, not trusted.

Mirrors ``features.validate``'s posture: a check here does not assume the module that
should have guaranteed a property actually did. Component 17 validates its own output,
but this component re-checks the feature contract anyway, because a row that reaches
here having already been mis-produced upstream is exactly the case a second look exists
to catch.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from sentinel.evaluation.models import FoldSpec
from sentinel.features.definitions import FEATURE_COLUMNS, FEATURE_SPECS, NullRule
from sentinel.features.models import ValidationCheck

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

_NEVER_NULL_COLUMNS: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if spec.null_rule is NullRule.NEVER
)


class FeatureContractError(ValueError):
    """Raised when the candidate table is structurally incompatible, not just row-dirty."""


def _check(name: str, passed: bool, severity: str, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=passed, severity=severity, detail=detail)


def check_feature_contract(
    candidates: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, list[ValidationCheck]]:
    """Split candidates into scorable and excluded, and report both counts.

    A missing declared column is a whole-run defect -- refused outright, because no
    subset of rows could be scored without it. A ``NullRule.NEVER`` violation on a
    single row is row-local corruption: that row is excluded and reported, and every
    other row is unaffected. This is the one place Component 18 may drop a candidate,
    and it never does so silently.
    """
    checks: list[ValidationCheck] = []
    required = ("target_inspection_id", *FEATURE_COLUMNS)
    missing = [c for c in required if c not in candidates.columns]
    if missing:
        raise FeatureContractError(
            f"candidate table is missing required column(s): {', '.join(missing)}. This "
            "is a whole-table defect, not a per-row one -- no candidate could be scored"
        )
    checks.append(
        _check(
            "feature_contract_columns_present",
            True,
            SEVERITY_ERROR,
            f"all {len(FEATURE_COLUMNS)} declared feature columns present",
        )
    )

    if candidates.height == 0:
        checks.append(
            _check("no_never_null_violations", True, SEVERITY_WARN, "0 candidates to check")
        )
        return candidates, candidates.clear(), checks

    bad_mask = pl.any_horizontal([pl.col(c).is_null() for c in _NEVER_NULL_COLUMNS])
    excluded = candidates.filter(bad_mask)
    valid = candidates.filter(~bad_mask)
    checks.append(
        _check(
            "no_never_null_violations",
            excluded.height == 0,
            SEVERITY_WARN,
            f"{excluded.height} of {candidates.height} candidate(s) violate a "
            "never-null feature rule and were excluded from scoring rather than passed "
            "to the model",
        )
    )
    return valid, excluded, checks


def check_no_future_leakage(
    fold: FoldSpec, historical_features: pl.DataFrame, *, date_column: str = "rd"
) -> ValidationCheck:
    """Independently re-derive: no training row is dated on or after the planning date.

    Re-derived directly from the training window's own bound rather than by trusting
    ``window.py``, the same posture ``features.validate.temporal_boundary_holds`` takes
    toward ``historical.py``.
    """
    offenders = historical_features.filter(
        (pl.col(date_column) >= fold.train_start) & (pl.col(date_column) > fold.train_end)
    ).height
    return _check(
        "no_training_row_on_or_after_planning_date",
        offenders == 0,
        SEVERITY_ERROR,
        f"{offenders} training-window row(s) dated after the operational train_end "
        f"({fold.train_end.isoformat()})",
    )


def check_identity_preservation(
    candidate_ids: Sequence[str], scored_ids: Sequence[str]
) -> ValidationCheck:
    """Every scored id must be a candidate id, and vice versa -- no substitution, no drift."""
    left, right = set(candidate_ids), set(scored_ids)
    passed = left == right
    return _check(
        "scored_ids_equal_candidate_ids",
        passed,
        SEVERITY_ERROR,
        "scored candidate identifiers match the input set exactly"
        if passed
        else f"{len(left - right)} candidate(s) not scored, {len(right - left)} scored "
        "id(s) not in the candidate set",
    )


def check_scores_are_probabilities(scores: Sequence[float]) -> ValidationCheck:
    bad = [s for s in scores if not (0.0 <= s <= 1.0)]
    return _check(
        "calibrated_scores_within_zero_and_one",
        not bad,
        SEVERITY_ERROR,
        f"{len(bad)} calibrated score(s) outside [0, 1]",
    )


def check_rank_is_a_permutation(ranks: Sequence[int]) -> ValidationCheck:
    n = len(ranks)
    passed = sorted(ranks) == list(range(1, n + 1))
    return _check(
        "rank_is_a_dense_permutation",
        passed,
        SEVERITY_ERROR,
        f"ranks 1..{n} assigned exactly once each"
        if passed
        else "rank column is not a permutation",
    )


def check_planning_date_matches_fold(planning_date: date, fold: FoldSpec) -> ValidationCheck:
    passed = fold.train_end < planning_date <= fold.calibration_start
    return _check(
        "operational_fold_matches_planning_date",
        passed,
        SEVERITY_ERROR,
        f"train_end {fold.train_end.isoformat()} < planning_date {planning_date.isoformat()}",
    )


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


def format_report(checks: Sequence[ValidationCheck]) -> str:
    lines = ["", "Operational scoring validation report", "--------------------------------------"]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


__all__ = [
    "FeatureContractError",
    "check_feature_contract",
    "check_identity_preservation",
    "check_no_future_leakage",
    "check_planning_date_matches_fold",
    "check_rank_is_a_permutation",
    "check_scores_are_probabilities",
    "format_report",
    "has_failures",
]
