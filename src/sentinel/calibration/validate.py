"""Post-calibration checks over the calibrators and their output.

Component 4 has a safety wall against features that can see the future, Component 5 against
an evaluation that can, and Component 6 against a model that can. This is the fourth: a wall
against a **calibrator** that can.

Following the precedent all three set, the important checks **re-derive** their answer from
the data rather than reading back what the orchestrator reported. Re-deriving each
calibrator's fitting rows from ``assign_split`` and intersecting them with the fold's test
window is the mechanical proof that no test row reached a fit; asking the orchestrator
whether it did the right thing would only prove the code agrees with itself.

The checks that matter most, and what each would catch:

1.  ``base_scores_reproduce_the_committed_artifact``  -- a re-executed model that is not the
    committed one, which would make every calibrator a correction to nothing
2.  ``no_test_row_enters_any_calibrator_fit``         -- the leak this component exists to avoid
3.  ``calibrator_fit_rows_lie_in_the_calibration_window`` -- a fit that reached outside its window
4.  ``inner_select_is_strictly_later_than_inner_fit`` -- a selection that read its own future
5.  ``method_selection_reads_no_future_fold``         -- the expanding-prefix guarantee
6.  ``trained_through_is_the_calibration_end``        -- an artifact understating what it knows
7.  ``the_calibrator_is_monotone``                    -- a calibrator that reorders risk
8.  ``platt_does_not_change_the_ranking``             -- strict monotonicity, exactly
9.  ``calibrated_predictions_cover_every_fold_exactly`` -- a dropped or duplicated test row
10. ``the_selection_rule_matches_the_frozen_literals`` -- a run reporting a rule it did not use
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date

import polars as pl

from sentinel.calibration.definitions import (
    MARGIN_TOLERANCE,
    SELECTION_METRIC,
    TIE_PREFERENCE,
    TIE_THRESHOLD,
    CandidateSpec,
    Method,
)
from sentinel.calibration.models import (
    FittedCalibrator,
    InnerSplit,
    SelectionOutcome,
    ValidationCheck,
)
from sentinel.calibration.predict import creates_ties, is_monotone
from sentinel.calibration.preprocess import logit
from sentinel.evaluation.folds import assign_split, window_frame
from sentinel.evaluation.models import FoldSpec

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

MAX_OFFENDERS = 20


def _as_int(value: object) -> int:
    """Narrow a value read out of an untyped row mapping. 0 for anything unexpected."""
    return int(value) if isinstance(value, int | float) else 0


def _check(
    name: str,
    passed: bool,
    severity: str,
    detail: str,
    offenders: Sequence[str] = (),
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail,
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- 1. the base models are the committed ones (ADR 0026) --------------------


def base_scores_reproduce_the_committed_artifact(
    mismatches: Mapping[str, int], offenders: Sequence[str], rows_compared: int
) -> ValidationCheck:
    """Every regenerated test score equals the committed one, bit for bit.

    The precondition for everything else in this component. A calibrator fitted on scores
    from a model that moved is a correction to nothing, so ``build.py`` refuses to fit one
    if this fails -- a validation report alone would let the artifact be written.

    Compared with ``==``, never ``math.isclose``. A tolerance here would convert the one
    check that makes ADR 0026 safe into a check that passes when the models differ.
    """
    total = sum(mismatches.values())
    per_model = ", ".join(f"{name} {count}" for name, count in sorted(mismatches.items()) if count)
    return _check(
        "base_scores_reproduce_the_committed_artifact",
        total == 0,
        SEVERITY_ERROR,
        (
            f"all {rows_compared} regenerated test-window score(s) are bit-identical to the "
            "committed Component 6/7/8 artifacts, so the re-executed fits are the fits those "
            "components published"
            if total == 0
            else (
                f"{total} regenerated score(s) differ from the committed artifact "
                f"({per_model}). Either a library version moved or the fit is not the one "
                "that was published; a tolerance is not the remedy. Note that a BLAS thread "
                "count differing from 'unset (library default)' is enough to cause this."
            )
        ),
        offenders,
    )


def recovered_logit_matches_the_native_margin(
    max_error: Mapping[str, float], unavailable: Sequence[str]
) -> ValidationCheck:
    """``logit(p)`` agrees with the base model's own decision margin (ADR 0027).

    Warn severity, and the tolerance is 1e-4 rather than a float64 epsilon **by
    measurement**: xgboost and the network compute in float32, so the sigmoid round trip
    cannot recover more than float32 carried. 2.6e-5 is correct behaviour; a double sigmoid
    or a sign flip would be O(1).
    """
    offenders = [
        f"{name}: max |logit(p) - margin| = {value:.3e} exceeds {MARGIN_TOLERANCE:.0e}"
        for name, value in sorted(max_error.items())
        if value > MARGIN_TOLERANCE
    ]
    observed = ", ".join(f"{name} {value:.2e}" for name, value in sorted(max_error.items()))
    note = f"; not reachable for {', '.join(unavailable)}" if unavailable else ""
    return _check(
        "recovered_logit_matches_the_native_margin",
        not offenders,
        SEVERITY_WARN,
        f"max |logit(p) - margin| by model: {observed}{note}. float32 models round-trip to "
        f"~1e-5, float64 models to ~1e-13; tolerance {MARGIN_TOLERANCE:.0e}",
        offenders,
    )


def no_probability_was_clamped(clamped: int) -> ValidationCheck:
    """No score sat at exactly 0 or 1, so the logit guard never fired."""
    return _check(
        "no_probability_was_clamped",
        clamped == 0,
        SEVERITY_WARN,
        (
            "no base score required clamping before the logit; nothing saturated"
            if clamped == 0
            else f"{clamped} base score(s) were clamped before the logit, meaning the base "
            "model has started saturating. Not an error, but the calibrator is extrapolating."
        ),
    )


# --- 2. no test row reached a calibrator -------------------------------------


def no_test_row_enters_any_calibrator_fit(
    fits: Mapping[tuple[str, str], Sequence[str]],
    folds: Mapping[str, FoldSpec],
    frame: pl.DataFrame,
) -> ValidationCheck:
    """The rows a calibrator was fitted on never intersect that fold's test window.

    Re-derived: the test ids come from ``evaluation.folds.window_frame`` on the feature
    table, not from anything this component recorded.
    """
    offenders: list[str] = []
    for (model_name, fold_id), fit_ids in sorted(fits.items()):
        fold = folds[fold_id]
        test_ids = {str(v) for v in window_frame(frame, fold)["target_inspection_id"].to_list()}
        overlap = test_ids & set(fit_ids)
        if overlap:
            offenders.append(
                f"{model_name}/{fold_id}: {len(overlap)} test row(s) in the calibrator fit, "
                f"e.g. {', '.join(sorted(overlap)[:3])}"
            )
    return _check(
        "no_test_row_enters_any_calibrator_fit",
        not offenders,
        SEVERITY_ERROR,
        f"no test row reached any of the {len(fits)} calibrator fits",
        offenders,
    )


def calibrator_fit_rows_lie_in_the_calibration_window(
    fits: Mapping[tuple[str, str], Sequence[str]],
    folds: Mapping[str, FoldSpec],
    frame: pl.DataFrame,
) -> ValidationCheck:
    """Every row a calibrator saw re-derives to ``split == "calibration"``.

    The positive form of the previous check: not merely "no test row" but "only calibration
    rows", which also catches a training row leaking in through a bad join.
    """
    offenders: list[str] = []
    for (model_name, fold_id), fit_ids in sorted(fits.items()):
        labelled = assign_split(frame, folds[fold_id])
        splits = dict(
            zip(
                (str(v) for v in labelled["target_inspection_id"].to_list()),
                (str(v) for v in labelled["split"].to_list()),
                strict=True,
            )
        )
        wrong = sorted({splits.get(row_id, "unknown") for row_id in fit_ids} - {"calibration"})
        if wrong:
            offenders.append(
                f"{model_name}/{fold_id}: fit rows also in split(s) {', '.join(wrong)}"
            )
    return _check(
        "calibrator_fit_rows_lie_in_the_calibration_window",
        not offenders,
        SEVERITY_ERROR,
        "every row every calibrator was fitted on re-derives to the calibration split",
        offenders,
    )


def inner_select_is_strictly_later_than_inner_fit(
    splits: Mapping[tuple[str, str], InnerSplit], dates: Mapping[tuple[str, str], Sequence[object]]
) -> ValidationCheck:
    """The selection window sits wholly after the window the candidates were fitted on.

    Also asserts the split never divides a single day: two inspections of the same
    establishment days apart share almost all of their as-of history, so a same-day split
    would put correlated rows on both sides and flatter whichever method overfits.
    """
    offenders: list[str] = []
    for key, split in sorted(splits.items()):
        day = list(dates[key])
        fit_days = {d for d in (day[i] for i in split.fit_index) if isinstance(d, date)}
        select_days = {d for d in (day[i] for i in split.select_index) if isinstance(d, date)}
        shared = fit_days & select_days
        if shared:
            offenders.append(f"{key[0]}/{key[1]}: {len(shared)} day(s) on both sides of the split")
            continue
        if fit_days and select_days and max(fit_days) >= min(select_days):
            offenders.append(f"{key[0]}/{key[1]}: inner-select starts before inner-fit ends")
    return _check(
        "inner_select_is_strictly_later_than_inner_fit",
        not offenders,
        SEVERITY_ERROR,
        f"all {len(splits)} inner splits are chronological and cut on a whole-day boundary",
        offenders,
    )


def method_selection_reads_no_future_fold(
    outcomes: Sequence[SelectionOutcome], folds: Mapping[str, FoldSpec], ordering: Mapping[str, int]
) -> ValidationCheck:
    """Fold k's method was chosen from folds whose calibration ends at or before its own.

    The expanding-prefix guarantee, re-derived from the fold dates. This is the check that
    would catch the design ADR 0025 rejected: pooling every fold's inner-select result to
    choose one method per model reads fold N-1's **test** window, because fold N's
    calibration window is exactly that.
    """
    offenders: list[str] = []
    for outcome in outcomes:
        fold = folds[outcome.fold_id]
        contributors = [
            other
            for other in folds.values()
            if other.fold_set == outcome.fold_set and ordering[other.fold_id] <= outcome.fold_index
        ]
        late = [o.fold_id for o in contributors if o.calibration_end > fold.calibration_end]
        if late:
            offenders.append(
                f"{outcome.model_name}/{outcome.fold_id}: prefix includes {', '.join(late)}, "
                "whose calibration window ends later"
            )
    return _check(
        "method_selection_reads_no_future_fold",
        not offenders,
        SEVERITY_ERROR,
        f"every one of the {len(outcomes)} selections used only folds ending at or before its "
        "own calibration end, so no selection read a later window",
        offenders,
    )


def folds_never_share_a_calibrator(calibrators: Sequence[FittedCalibrator]) -> ValidationCheck:
    """Each (model, fold) has its own calibrator, and covid_shift never borrows a quarterly one.

    A calibrator from another fold would carry that fold's window into this one -- the same
    guarantee ``neural.embed.fit_fold`` enforces on its donor network.
    """
    seen: dict[tuple[str, str], int] = {}
    for calibrator in calibrators:
        seen[(calibrator.model_name, calibrator.fold_id)] = (
            seen.get((calibrator.model_name, calibrator.fold_id), 0) + 1
        )
    duplicates = [f"{m}/{f}: {n} calibrators" for (m, f), n in sorted(seen.items()) if n != 1]
    crossed = [
        f"{c.model_name}/{c.fold_id}: fitted on {c.fit_start}..{c.fit_end}"
        for c in calibrators
        if c.fold_set == "covid_shift" and c.fit_start.year > 2020
    ]
    return _check(
        "folds_never_share_a_calibrator",
        not duplicates and not crossed,
        SEVERITY_ERROR,
        f"each of the {len(seen)} (model, fold) pairs has exactly one calibrator, fitted on "
        "its own window",
        [*duplicates, *crossed],
    )


# --- 3. provenance -----------------------------------------------------------


def horizons_are_declared_correctly(
    predictions: pl.DataFrame, folds: Mapping[str, FoldSpec]
) -> list[ValidationCheck]:
    """The three horizon columns say what they mean (ADR 0024).

    ``trained_through`` must be the calibration end -- the contract's ceiling, and the
    maximum of the two component horizons. Writing ``train_end`` there would be false: the
    calibrator really did read the calibration window. Writing anything later would be
    rejected by ``evaluation.contract.validate_predictions``.
    """
    checks: list[ValidationCheck] = []
    expectations = (
        ("trained_through", "calibration_end", "trained_through_is_the_calibration_end"),
        (
            "base_model_trained_through",
            "train_end",
            "base_model_trained_through_is_the_training_end",
        ),
        (
            "calibrator_fitted_through",
            "calibration_end",
            "calibrator_fitted_through_is_the_calibration_end",
        ),
        ("calibrated_prediction_available_from", "test_start", "available_from_is_the_test_start"),
    )
    for column, attribute, name in expectations:
        offenders: list[str] = []
        for fold_id, group in predictions.group_by("fold_id"):
            fold = folds[str(fold_id[0])]
            expected = getattr(fold, attribute)
            wrong = [v for v in group[column].unique().to_list() if v != expected]
            if wrong:
                offenders.append(f"{fold_id[0]}: {column} = {wrong}, expected {expected}")
        checks.append(
            _check(
                name,
                not offenders,
                SEVERITY_ERROR,
                f"every calibrated row declares {column} = fold.{attribute}",
                offenders,
            )
        )
    return checks


def calibrated_predictions_cover_every_fold_exactly(
    predictions: pl.DataFrame,
    folds: Mapping[str, FoldSpec],
    frame: pl.DataFrame,
    candidates: Sequence[CandidateSpec],
) -> ValidationCheck:
    """Every calibrated model scored every test row of every fold exactly once.

    Re-derived from ``window_frame`` rather than from a row count, so a dropped establishment
    and a duplicated one are both caught. A model that quietly drops the rows it finds hard
    posts a better precision@k for a reason unrelated to being better.
    """
    offenders: list[str] = []
    expected_models = {c.name for c in candidates}
    for fold_id, fold in sorted(folds.items()):
        expected = {str(v) for v in window_frame(frame, fold)["target_inspection_id"].to_list()}
        subset = predictions.filter(pl.col("fold_id") == fold_id)
        for base_name in sorted(expected_models):
            rows = subset.filter(pl.col("base_model_name") == base_name)
            got = [str(v) for v in rows["target_inspection_id"].to_list()]
            if len(got) != len(set(got)):
                offenders.append(
                    f"{base_name}/{fold_id}: {len(got) - len(set(got))} duplicate row(s)"
                )
            if set(got) != expected:
                offenders.append(
                    f"{base_name}/{fold_id}: {len(expected - set(got))} unscored, "
                    f"{len(set(got) - expected)} outside the test window"
                )
    return _check(
        "calibrated_predictions_cover_every_fold_exactly",
        not offenders,
        SEVERITY_ERROR,
        f"every candidate scored every test row of all {len(folds)} folds exactly once",
        offenders,
    )


def calibrated_scores_are_probabilities(predictions: pl.DataFrame) -> ValidationCheck:
    """Finite, in [0, 1], never null."""
    scores = predictions["score"].to_list()
    nulls = sum(1 for v in scores if v is None)
    bad = [float(v) for v in scores if v is not None and not math.isfinite(float(v))]
    finite = [float(v) for v in scores if v is not None and math.isfinite(float(v))]
    outside = [v for v in finite if not 0.0 <= v <= 1.0]
    offenders = []
    if nulls:
        offenders.append(f"{nulls} null score(s)")
    if bad:
        offenders.append(f"{len(bad)} non-finite score(s)")
    if outside:
        offenders.append(f"{len(outside)} score(s) outside [0, 1], e.g. {outside[:3]}")
    return _check(
        "calibrated_scores_are_probabilities",
        not offenders,
        SEVERITY_ERROR,
        f"all {len(scores)} calibrated scores are finite probabilities in [0, 1]",
        offenders,
    )


# --- 4. the calibrator did not re-rank ---------------------------------------


def the_calibrator_is_monotone(calibrators: Sequence[FittedCalibrator]) -> ValidationCheck:
    """Every frozen calibrator is non-decreasing over a grid spanning (0, 1).

    Probed on the *applied* mapping rather than inferred from parameters, because it is the
    applied mapping that must not reorder risk. Weak monotonicity: isotonic passes while
    producing plateaus, whose ties are counted separately.
    """
    offenders = [
        f"{c.model_name}/{c.fold_id}/{c.method.value}: mapping is not non-decreasing"
        for c in calibrators
        if not is_monotone(c)
    ]
    return _check(
        "the_calibrator_is_monotone",
        not offenders,
        SEVERITY_ERROR,
        f"all {len(calibrators)} calibrators are monotone over 200 probes spanning (0, 1)",
        offenders,
    )


def platt_does_not_change_the_ranking(
    ranking_rows: Sequence[Mapping[str, object]],
) -> ValidationCheck:
    """Platt is strictly monotone: zero inversions, zero new ties, rho exactly 1.

    Stronger than the monotonicity check and deliberately exact. Platt is a strictly
    increasing function, so ROC-AUC, PR-AUC, NDE and precision@k are unchanged by
    construction -- and if any of them moves, the calibrator is not what it claims to be.
    """
    offenders: list[str] = []
    checked = 0
    for row in ranking_rows:
        if row["stage"] != Method.PLATT.value:
            continue
        checked += 1
        rho = row["spearman_rho"]
        if row["inversions"] != 0 or row["new_ties_created"] != 0:
            offenders.append(
                f"{row['model_name']}/{row['fold_id']}: {row['inversions']} inversion(s), "
                f"{row['new_ties_created']} new tie(s)"
            )
        elif isinstance(rho, float) and abs(rho - 1.0) > 1e-12:
            offenders.append(f"{row['model_name']}/{row['fold_id']}: Spearman rho = {rho!r}")
    return _check(
        "platt_does_not_change_the_ranking",
        not offenders,
        SEVERITY_ERROR,
        f"all {checked} Platt fold(s) preserved the ordering exactly: no inversion, no new "
        "tie, Spearman rho 1.0",
        offenders,
    )


def isotonic_ties_are_counted_not_hidden(
    ranking_rows: Sequence[Mapping[str, object]], calibrators: Sequence[FittedCalibrator]
) -> ValidationCheck:
    """Isotonic's plateaus are reported as ties, and never as ranking inversions.

    Warn severity: creating ties is what isotonic *does*, not a defect. What would be a
    defect is an inversion, and this check asserts there are none while surfacing the tie
    count and the top-k movement it causes.
    """
    offenders: list[str] = []
    ties = movement = 0
    for row in ranking_rows:
        if row["stage"] != Method.ISOTONIC.value:
            continue
        ties += _as_int(row["new_ties_created"])
        movement += _as_int(row["top_k_membership_changed"])
        if row["inversions"] != 0:
            offenders.append(
                f"{row['model_name']}/{row['fold_id']}: {row['inversions']} inversion(s) -- a "
                "monotone map cannot invert, so this is a defect rather than a tie"
            )
    tying = sum(1 for c in calibrators if c.method is Method.ISOTONIC and creates_ties(c))
    return _check(
        "isotonic_ties_are_counted_not_hidden",
        not offenders,
        SEVERITY_WARN,
        f"isotonic created {ties} new tie(s) across all folds and moved {movement} top-k "
        f"membership(s); {tying} isotonic calibrator(s) produce plateaus. Ties are not "
        "ranking inversions, and there were no inversions.",
        offenders,
    )


# --- 5. the protocol was the frozen one --------------------------------------


def the_selection_rule_matches_the_frozen_literals(
    manifest_values: Mapping[str, object],
) -> ValidationCheck:
    """The manifest reports the rule that is actually in ``definitions.py``.

    A run cannot describe a threshold it did not apply. This is what makes ADR 0025's
    pre-registration checkable by machine rather than only by reading a commit date.
    """
    expected: dict[str, object] = {
        "selection_metric": SELECTION_METRIC,
        "tie_threshold": TIE_THRESHOLD,
        "tie_preference": TIE_PREFERENCE.value,
    }
    offenders = [
        f"{key}: manifest says {manifest_values.get(key)!r}, definitions.py says {value!r}"
        for key, value in expected.items()
        if manifest_values.get(key) != value
    ]
    return _check(
        "the_selection_rule_matches_the_frozen_literals",
        not offenders,
        SEVERITY_ERROR,
        f"the manifest reports the frozen rule: {SELECTION_METRIC}, tie threshold "
        f"{TIE_THRESHOLD}, preference {TIE_PREFERENCE.value}",
        offenders,
    )


def the_persisted_calibrator_reproduces_the_mapping(
    calibrators: Sequence[FittedCalibrator], probes: int = 50
) -> ValidationCheck:
    """The extracted parameters give the same answer as the fitted scikit-learn object.

    If these disagreed, ``calibrator_parameters_*.parquet`` would be a description of
    something other than what ran, and a consumer reading the artifact would get different
    probabilities from the ones this component published.
    """
    import numpy as np

    from sentinel.calibration.predict import apply

    grid = [(i + 0.5) / probes for i in range(probes)]
    offenders: list[str] = []
    for calibrator in calibrators:
        from_parameters = apply(calibrator, grid)
        if calibrator.method is Method.PLATT:
            x = np.asarray([logit(p) for p in grid], dtype=np.float64).reshape(-1, 1)
            native = [float(v) for v in calibrator.estimator.predict_proba(x)[:, 1]]
        else:
            native = [
                float(v)
                for v in calibrator.estimator.predict(np.asarray(grid, dtype=np.float64))
            ]
        worst = max(abs(a - b) for a, b in zip(from_parameters, native, strict=True))
        if worst > 1e-9:
            offenders.append(
                f"{calibrator.model_name}/{calibrator.fold_id}/{calibrator.method.value}: "
                f"persisted mapping differs from the fitted estimator by {worst:.3e}"
            )
    return _check(
        "the_persisted_calibrator_reproduces_the_mapping",
        not offenders,
        SEVERITY_ERROR,
        f"all {len(calibrators)} persisted calibrators reproduce their fitted estimator to "
        "within 1e-9, so the artifact is the calibrator rather than a description of it",
        offenders,
    )


def calibration_is_reported_honestly(drift_rows: Sequence[Mapping[str, object]]) -> ValidationCheck:
    """Reports where calibration made a metric worse. Never fails on a worse number.

    Deliberately advisory. A check that failed when a calibrator did not help would create
    pressure to tune until it did, which is the loop this whole component is built to avoid.
    """
    worse: list[str] = []
    for row in drift_rows:
        if row["stage"] != "selected":
            continue
        baseline = next(
            (
                r
                for r in drift_rows
                if r["model_name"] == row["model_name"]
                and r["fold_id"] == row["fold_id"]
                and r["stage"] == "uncalibrated"
            ),
            None,
        )
        if baseline is None:
            continue
        before, after = baseline["ece"], row["ece"]
        if isinstance(before, float) and isinstance(after, float) and after > before:
            worse.append(f"{row['model_name']}/{row['fold_id']}: ECE {before:.4f} -> {after:.4f}")
    return _check(
        "calibration_is_reported_honestly",
        True,
        SEVERITY_WARN,
        (
            f"calibration raised test ECE on {len(worse)} of the (model, fold) cells. This is "
            "reported, not treated as a failure: a check that went red on a worse number "
            "would create pressure to tune until it went green, which is the loop the "
            "temporal protocol exists to prevent."
            if worse
            else "calibration did not raise test ECE on any (model, fold) cell"
        ),
        worse,
    )


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """Render the checks as a plain text block for the CLI."""
    lines = ["", "Calibration validation report", "-----------------------------"]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
        if check.offenders and not (check.passed and check.severity == SEVERITY_ERROR):
            lines.extend(f"           - {offender}" for offender in check.offenders)
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """Whether any error-severity check failed."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


__all__ = [
    "MAX_OFFENDERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "base_scores_reproduce_the_committed_artifact",
    "calibrated_predictions_cover_every_fold_exactly",
    "calibrated_scores_are_probabilities",
    "calibration_is_reported_honestly",
    "calibrator_fit_rows_lie_in_the_calibration_window",
    "folds_never_share_a_calibrator",
    "format_report",
    "has_failures",
    "horizons_are_declared_correctly",
    "inner_select_is_strictly_later_than_inner_fit",
    "isotonic_ties_are_counted_not_hidden",
    "method_selection_reads_no_future_fold",
    "no_probability_was_clamped",
    "no_test_row_enters_any_calibrator_fit",
    "platt_does_not_change_the_ranking",
    "recovered_logit_matches_the_native_margin",
    "the_calibrator_is_monotone",
    "the_persisted_calibrator_reproduces_the_mapping",
    "the_selection_rule_matches_the_frozen_literals",
]
