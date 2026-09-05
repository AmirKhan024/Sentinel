"""The check machinery: severities, reporting, and what fails a run.

``test_evaluation_leakage.py`` asks whether the checks find leakage. This file
asks the narrower questions around them: which failures are fatal, which are
advisory context, and whether the report a human reads actually says what
happened.

The distinction matters operationally. An error-severity failure exits the CLI
non-zero, because it means the evaluation could see the future and every number
it produced is untrustworthy. A warn-severity note -- base-rate drift, measured
capacity, blocked experiments -- is context a reader needs in order to interpret
the numbers, not a reason to reject them.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from sentinel.evaluation.folds import covid_shift_fold, fold_stats, quarterly_folds
from sentinel.evaluation.models import ValidationCheck
from sentinel.evaluation.validate import (
    SEVERITY_ERROR,
    SEVERITY_WARN,
    Observations,
    format_report,
    has_failures,
    validate_evaluation,
)
from tests.conftest import spanning_features

DATA_START = date(2018, 7, 2)
DATA_END = date(2026, 6, 30)


def _frame() -> pl.DataFrame:
    return spanning_features(days=1800, per_day=2).with_columns(
        pl.col("inspection_date").str.to_date().alias("rd")
    )


def _run(observations: Observations | None = None) -> list[ValidationCheck]:
    frame = _frame()
    folds = quarterly_folds(data_start=DATA_START, data_end=frame["rd"].max())  # type: ignore[arg-type]
    stats = [fold_stats(frame, fold) for fold in folds]
    return validate_evaluation(frame, folds, stats, observations or Observations())


def _named(checks: list[ValidationCheck], name: str) -> ValidationCheck:
    (check,) = [c for c in checks if c.name == name]
    return check


# --- 1. severities ----------------------------------------------------------


def test_a_clean_run_has_no_error_severity_failures() -> None:
    assert not has_failures(_run())


def test_every_leakage_check_is_error_severity() -> None:
    """Advisory context must never be able to mask a leak."""
    checks = _run()
    for name in (
        "fold_boundaries_are_strict",
        "calibration_sits_between",
        "folds_advance_monotonically",
        "training_window_expands",
        "future_rows_never_enter_training",
        "test_is_isolated",
        "no_split_overlap",
        "capacity_is_conserved",
        "business_as_usual_is_real",
        "score_direction_is_descending",
    ):
        assert _named(checks, name).severity == SEVERITY_ERROR, name


def test_context_notes_are_warn_severity_and_do_not_fail_a_run() -> None:
    checks = _run()
    for name in ("fold_sets", "base_rate_drift", "measured_capacity", "models_scored"):
        note = _named(checks, name)
        assert note.severity == SEVERITY_WARN, name
        assert note.passed, name
    assert not has_failures(checks)


def test_a_warn_severity_failure_alone_does_not_fail_the_run() -> None:
    checks = [ValidationCheck("advisory", False, SEVERITY_WARN, "just context")]
    assert not has_failures(checks)


def test_an_error_severity_failure_fails_the_run() -> None:
    checks = [ValidationCheck("leak", False, SEVERITY_ERROR, "the split saw the future")]
    assert has_failures(checks)


# --- 2. the simulation guards ----------------------------------------------


def test_capacity_conservation_is_checked_by_re_deriving_it() -> None:
    assert _named(_run(), "capacity_is_conserved").passed


def test_the_business_as_usual_identity_is_checked_on_the_real_folds() -> None:
    assert _named(_run(), "business_as_usual_is_real").passed


def test_score_direction_is_probed_rather_than_assumed() -> None:
    """A convention that gets inverted in a refactor would silently reverse
    every conclusion in the project, so it is asserted on every run."""
    check = _named(_run(), "score_direction_is_descending")
    assert check.passed
    assert "0.90" in check.detail


# --- 3. what the orchestrator reports --------------------------------------


def test_a_contract_rejection_fails_the_run() -> None:
    observations = Observations(contract_rejections=["model_a dropped 12 rows"])
    checks = _run(observations)
    assert not _named(checks, "predictions_cover_test_exactly").passed
    assert has_failures(checks)


def test_a_horizon_rejection_fails_the_run() -> None:
    observations = Observations(horizon_rejections=["model_b trained through 2025"])
    checks = _run(observations)
    assert not _named(checks, "scores_respect_the_decision_point").passed
    assert has_failures(checks)


def test_blocked_experiments_appear_as_a_note_with_their_reasons() -> None:
    observations = Observations(blocked=["weather is not ingested"])
    note = _named(_run(observations), "blocked_experiments")
    assert note.severity == SEVERITY_WARN
    assert note.offenders == ("weather is not ingested",)


def test_no_blocked_note_is_emitted_when_nothing_is_blocked() -> None:
    assert not [c for c in _run() if c.name == "blocked_experiments"]


# --- 4. the advisory context a reader needs --------------------------------


def test_base_rate_drift_is_reported_with_its_range() -> None:
    """Every base-rate-dependent metric must be read beside its prevalence, so
    the range is stated rather than left for the reader to discover."""
    detail = _named(_run(), "base_rate_drift").detail
    assert "base rate ranges" in detail
    assert "prevalence" in detail


def test_measured_capacity_is_reported_as_measured_not_chosen() -> None:
    detail = _named(_run(), "measured_capacity").detail
    assert "derived from this, not chosen" in detail


def test_the_fold_sets_are_named_with_their_counts() -> None:
    frame = _frame()
    folds = [
        *quarterly_folds(data_start=DATA_START, data_end=frame["rd"].max()),  # type: ignore[arg-type]
        *covid_shift_fold(data_end=frame["rd"].max()),  # type: ignore[arg-type]
    ]
    stats = [fold_stats(frame, fold) for fold in folds]
    checks = validate_evaluation(frame, folds, stats, Observations())
    detail = _named(checks, "fold_sets").detail
    assert "quarterly" in detail
    assert "covid_shift" in detail


# --- 5. the report a human reads -------------------------------------------


def test_the_report_marks_passes_notes_and_failures_distinctly() -> None:
    checks = [
        ValidationCheck("good", True, SEVERITY_ERROR, "held"),
        ValidationCheck("bad", False, SEVERITY_ERROR, "leaked", ("fold-1",)),
        ValidationCheck("context", True, SEVERITY_WARN, "for information"),
    ]
    report = format_report(checks)
    assert "[PASS] good" in report
    assert "[FAIL] bad" in report
    assert "[note] context" in report


def test_the_report_lists_offenders_for_a_failure() -> None:
    checks = [ValidationCheck("bad", False, SEVERITY_ERROR, "leaked", ("fold-1", "fold-2"))]
    report = format_report(checks)
    assert "- fold-1" in report
    assert "- fold-2" in report


def test_the_report_does_not_list_offenders_for_a_passing_error_check() -> None:
    """A passing check's offender list is empty by construction; noise here
    would bury the failures that matter."""
    checks = [ValidationCheck("good", True, SEVERITY_ERROR, "held", ("noise",))]
    assert "- noise" not in format_report(checks)


def test_the_report_has_a_heading() -> None:
    assert "Evaluation validation report" in format_report(_run())


def test_offender_lists_are_capped() -> None:
    from sentinel.evaluation.validate import MAX_OFFENDERS

    observations = Observations(contract_rejections=[f"model_{i}" for i in range(100)])
    check = _named(_run(observations), "predictions_cover_test_exactly")
    assert len(check.offenders) == MAX_OFFENDERS


# --- 6. re-derivation ------------------------------------------------------


def test_row_counts_are_re_counted_rather_than_trusted() -> None:
    """Component 3 owns the target and Component 4 owns the rows. If Component 5
    ever produced a second interpretation of either, this check would catch it.
    """
    frame = _frame()
    folds = quarterly_folds(data_start=DATA_START, data_end=frame["rd"].max())  # type: ignore[arg-type]
    stats = [fold_stats(frame, fold) for fold in folds]
    assert _named(
        validate_evaluation(frame, folds, stats, Observations()),
        "labels_are_read_not_redefined",
    ).passed


def test_a_disagreeing_row_count_is_caught() -> None:
    from dataclasses import replace

    frame = _frame()
    folds = quarterly_folds(data_start=DATA_START, data_end=frame["rd"].max())  # type: ignore[arg-type]
    stats = [fold_stats(frame, fold) for fold in folds]
    tampered = [replace(stats[0], test_rows=stats[0].test_rows + 1), *stats[1:]]
    checks = validate_evaluation(frame, folds, tampered, Observations())
    assert not _named(checks, "labels_are_read_not_redefined").passed
