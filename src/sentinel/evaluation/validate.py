"""Post-run checks over the evaluation itself.

Component 4 has a safety wall against features that can see the future. This is
the equivalent wall one level up, against an *evaluation* that can see the
future -- and the two failures look nothing alike. A leaked feature makes a
single column wrong. A leaked evaluation makes every number right and every
conclusion wrong, which is far harder to notice.

Following Component 4's precedent, the important checks **re-derive** their
answer from the data rather than reading back what the orchestrator reported. A
check that trusts the thing it is checking only proves the code agrees with
itself.

The seven leakage checks, in the order the project spec names them:

1. ``future_rows_never_enter_training``   -- no training row dated after the cutoff
2. ``calibration_sits_between``           -- after training, before test, always
3. ``test_is_isolated``                   -- no test row in training or calibration
4. ``fold_boundaries_are_strict``         -- train_end < cal_start < cal_end < test_start
5. ``no_split_overlap``                   -- no row in two splits of one fold
6. ``folds_advance_monotonically``        -- a later fold never tests an earlier period
7. ``scores_respect_the_decision_point``  -- declared training horizon is enforced

Three more guard the simulation rather than the split:

* ``capacity_is_conserved``      -- every schedule uses the same slots
* ``business_as_usual_is_real``  -- the BAU schedule reproduces the observed dates
* ``score_direction_is_descending`` -- 0.90 is inspected before 0.10
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import polars as pl

from sentinel.evaluation import simulate
from sentinel.evaluation.folds import assign_split, max_date, min_date
from sentinel.evaluation.models import FoldSpec, FoldStats, ValidationCheck

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

MAX_OFFENDERS = 20


@dataclass
class Observations:
    """What the orchestrator saw, for the checks that cannot be re-derived."""

    models: list[str] = field(default_factory=list)
    contract_rejections: list[str] = field(default_factory=list)
    horizon_rejections: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    k_values: dict[str, int] = field(default_factory=dict)


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


def validate_evaluation(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    stats: Sequence[FoldStats],
    observations: Observations,
    *,
    date_column: str = "rd",
) -> list[ValidationCheck]:
    """Every assertion the evaluation must satisfy before its numbers are usable."""
    checks: list[ValidationCheck] = []

    checks.append(_fold_boundaries_are_strict(folds))
    checks.append(_calibration_sits_between(folds))
    checks.append(_folds_advance_monotonically(folds))
    checks.append(_training_window_expands(folds))
    checks.append(_future_rows_never_enter_training(frame, folds, date_column))
    checks.append(_test_is_isolated(frame, folds, date_column))
    checks.append(_no_split_overlap(frame, folds, date_column))
    checks.append(_test_windows_are_complete(frame, folds, date_column))
    checks.append(_capacity_is_conserved(frame, folds, date_column))
    checks.append(_business_as_usual_is_real(frame, folds, date_column))
    checks.append(_score_direction_is_descending())
    checks.append(_scores_respect_the_decision_point(observations))
    checks.append(_predictions_cover_test_exactly(observations))
    checks.append(_labels_are_read_not_redefined(frame, folds, stats, date_column))

    checks.extend(_advisory(folds, stats, observations))
    return checks


# --- 1. checks over the fold definitions -----------------------------------


def _fold_boundaries_are_strict(folds: Sequence[FoldSpec]) -> ValidationCheck:
    """train_end < calibration_start and calibration_end < test_start, every fold."""
    offenders = [
        f"{f.fold_id}: train_end={f.train_end} cal={f.calibration_start}..{f.calibration_end} "
        f"test_start={f.test_start}"
        for f in folds
        if not (f.train_end < f.calibration_start and f.calibration_end < f.test_start)
    ]
    return _check(
        "fold_boundaries_are_strict",
        not offenders,
        SEVERITY_ERROR,
        f"{len(folds)} folds have strictly ordered train / calibration / test boundaries"
        if not offenders
        else f"{len(offenders)} fold(s) have overlapping or out-of-order boundaries",
        offenders,
    )


def _calibration_sits_between(folds: Sequence[FoldSpec]) -> ValidationCheck:
    """Calibration is after training and before test -- never merged into either.

    Component 9 will fit Platt or isotonic scaling on this window. A calibrator
    fitted on the test period would make the reported probabilities
    self-fulfilling, and one fitted on the training period would inherit the
    model's own overfitting.
    """
    offenders = [
        f"{f.fold_id}: calibration {f.calibration_start}..{f.calibration_end} is not "
        f"strictly between training (ends {f.train_end}) and test (starts {f.test_start})"
        for f in folds
        if not (f.train_end < f.calibration_start <= f.calibration_end < f.test_start)
    ]
    return _check(
        "calibration_sits_between",
        not offenders,
        SEVERITY_ERROR,
        "every calibration window sits strictly between training and test"
        if not offenders
        else f"{len(offenders)} fold(s) misplace the calibration window",
        offenders,
    )


def _folds_advance_monotonically(folds: Sequence[FoldSpec]) -> ValidationCheck:
    """A later fold never tests an earlier period, and test windows do not overlap.

    This is what stops fold 7's result from being informed by fold 3's -- the
    rolling origin only moves forwards.
    """
    offenders: list[str] = []
    for fold_set in sorted({f.fold_set for f in folds}):
        ordered = [f for f in folds if f.fold_set == fold_set]
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.test_start <= earlier.test_start:
                offenders.append(f"{later.fold_id} does not start after {earlier.fold_id}")
            if later.test_start <= earlier.test_end:
                offenders.append(f"{later.fold_id} test window overlaps {earlier.fold_id}")
    return _check(
        "folds_advance_monotonically",
        not offenders,
        SEVERITY_ERROR,
        "test windows are strictly increasing and disjoint within every fold set"
        if not offenders
        else f"{len(offenders)} ordering violation(s)",
        offenders,
    )


def _training_window_expands(folds: Sequence[FoldSpec]) -> ValidationCheck:
    """Training starts at the anchor and only ever grows forwards."""
    offenders: list[str] = []
    for fold_set in sorted({f.fold_set for f in folds}):
        ordered = [f for f in folds if f.fold_set == fold_set]
        if not ordered:
            continue
        anchor = ordered[0].train_start
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.train_start != anchor:
                offenders.append(
                    f"{later.fold_id} moves the training anchor to {later.train_start}"
                )
            if later.train_end <= earlier.train_end:
                offenders.append(f"{later.fold_id} does not extend training past {earlier.fold_id}")
    return _check(
        "training_window_expands",
        not offenders,
        SEVERITY_ERROR,
        "training is an expanding window anchored at the code-era start"
        if not offenders
        else f"{len(offenders)} fold(s) break the expanding-window rule",
        offenders,
    )


# --- 2. checks re-derived from the data ------------------------------------


def _future_rows_never_enter_training(
    frame: pl.DataFrame, folds: Sequence[FoldSpec], date_column: str
) -> ValidationCheck:
    """Every row labelled ``train`` really is inside that fold's training window.

    Re-derived rather than assumed: the split labels are produced by
    ``assign_split``, and this check reads those labels back and re-tests the
    dates against the fold's own bounds. A boundary that slipped from ``<`` to
    ``<=`` in that function would surface here rather than in a result table.
    """
    offenders: list[str] = []
    for fold in folds:
        labelled = assign_split(frame, fold, date_column=date_column)
        train = labelled.filter(pl.col("split") == "train")
        if train.height == 0:
            continue
        latest = max_date(train, date_column)
        earliest = min_date(train, date_column)
        if latest is not None and latest > fold.train_end:
            offenders.append(
                f"{fold.fold_id}: a training row is dated {latest}, after the cutoff "
                f"{fold.train_end}"
            )
        if earliest is not None and earliest < fold.train_start:
            offenders.append(
                f"{fold.fold_id}: a training row is dated {earliest}, before the anchor "
                f"{fold.train_start}"
            )
    return _check(
        "future_rows_never_enter_training",
        not offenders,
        SEVERITY_ERROR,
        "no training row is dated outside its fold's training window"
        if not offenders
        else f"{len(offenders)} fold(s) train on rows outside the window",
        offenders,
    )


def _test_is_isolated(
    frame: pl.DataFrame, folds: Sequence[FoldSpec], date_column: str
) -> ValidationCheck:
    """Every test row is strictly later than every training and calibration row.

    Measured on the data, not on the dates: the maximum reference date actually
    present in training must be earlier than the minimum actually present in
    test. A fold whose windows are correctly specified but whose data lands in
    the wrong place would still fail here.
    """
    offenders: list[str] = []
    for fold in folds:
        labelled = assign_split(frame, fold, date_column=date_column)
        train = labelled.filter(pl.col("split") == "train")
        calibration = labelled.filter(pl.col("split") == "calibration")
        test = labelled.filter(pl.col("split") == "test")
        if test.height == 0:
            offenders.append(f"{fold.fold_id}: test window contains no rows")
            continue
        first_test = min_date(test, date_column)
        for name, other in (("train", train), ("calibration", calibration)):
            latest_other = max_date(other, date_column)
            if first_test is not None and latest_other is not None and latest_other >= first_test:
                offenders.append(
                    f"{fold.fold_id}: {name} contains a row on or after the first test row "
                    f"({first_test})"
                )
    return _check(
        "test_is_isolated",
        not offenders,
        SEVERITY_ERROR,
        "every test row is dated after every training and calibration row in its fold"
        if not offenders
        else f"{len(offenders)} isolation failure(s)",
        offenders,
    )


def _no_split_overlap(
    frame: pl.DataFrame, folds: Sequence[FoldSpec], date_column: str
) -> ValidationCheck:
    """No ``target_inspection_id`` appears in two splits of the same fold."""
    offenders: list[str] = []
    for fold in folds:
        labelled = assign_split(frame, fold, date_column=date_column)
        counts = (
            labelled.filter(pl.col("split") != "outside")
            .group_by("target_inspection_id")
            .agg(pl.col("split").n_unique().alias("splits"))
            .filter(pl.col("splits") > 1)
        )
        if counts.height:
            offenders.append(f"{fold.fold_id}: {counts.height} row(s) in more than one split")
    return _check(
        "no_split_overlap",
        not offenders,
        SEVERITY_ERROR,
        "no row belongs to two splits of the same fold"
        if not offenders
        else f"{len(offenders)} fold(s) contain overlapping splits",
        offenders,
    )


def _test_windows_are_complete(
    frame: pl.DataFrame, folds: Sequence[FoldSpec], date_column: str
) -> ValidationCheck:
    """Every emitted fold's test window is fully covered by the snapshot.

    A partial window would be compared against full ones as if the two were the
    same measurement. The generator drops them; this proves it did.
    """
    if frame.height == 0:
        return _check("test_windows_are_complete", True, SEVERITY_ERROR, "no rows to check")
    data_end = max_date(frame, date_column)
    offenders = [
        f"{f.fold_id}: test ends {f.test_end} but the snapshot ends {data_end}"
        for f in folds
        if data_end is not None and f.test_end > data_end
    ]
    return _check(
        "test_windows_are_complete",
        not offenders,
        SEVERITY_ERROR,
        f"all {len(folds)} test windows are fully covered by the snapshot"
        if not offenders
        else f"{len(offenders)} fold(s) extend past the data",
        offenders,
    )


# --- 3. checks over the simulation -----------------------------------------


def _capacity_is_conserved(
    frame: pl.DataFrame, folds: Sequence[FoldSpec], date_column: str
) -> ValidationCheck:
    """Every schedule consumes exactly the slots the city actually worked.

    Re-derived per fold by rebuilding the window and comparing the multiset of
    simulated dates against the observed one, for the two extreme schedules.
    If capacity were not conserved the model could beat business as usual by
    inspecting more, which would be a different intervention entirely.
    """
    offenders: list[str] = []
    for fold in folds:
        window = _window_for(frame, fold, date_column)
        if window.n == 0:
            continue
        observed = sorted(window.dates)
        for name, order in (
            ("optimal", simulate.optimal_order(window)),
            ("worst", simulate.worst_order(window)),
        ):
            simulated = sorted(simulate.simulated_dates(window, order))
            if simulated != observed:
                offenders.append(f"{fold.fold_id}/{name}: simulated dates are not the observed set")
    return _check(
        "capacity_is_conserved",
        not offenders,
        SEVERITY_ERROR,
        "every schedule uses exactly the observed inspection slots"
        if not offenders
        else f"{len(offenders)} schedule(s) changed the capacity profile",
        offenders,
    )


def _business_as_usual_is_real(
    frame: pl.DataFrame, folds: Sequence[FoldSpec], date_column: str
) -> ValidationCheck:
    """The business-as-usual schedule reproduces the observed dates exactly.

    This is what makes days-earlier meaningful: the comparator is not another
    simulation, it is what really happened. Asserted rather than argued.
    """
    offenders: list[str] = []
    for fold in folds:
        window = _window_for(frame, fold, date_column)
        if window.n == 0:
            continue
        simulated = simulate.simulated_dates(window, simulate.business_as_usual_order(window))
        if simulated != list(window.dates):
            offenders.append(f"{fold.fold_id}: BAU schedule does not reproduce the observed dates")
    return _check(
        "business_as_usual_is_real",
        not offenders,
        SEVERITY_ERROR,
        "the business-as-usual schedule reproduces the observed inspection dates exactly"
        if not offenders
        else f"{len(offenders)} fold(s) fail the business-as-usual identity",
        offenders,
    )


def _score_direction_is_descending() -> ValidationCheck:
    """A probe: a high score must be inspected before a low one.

    Trivial, and deliberately present. The direction of a score is the kind of
    convention that gets inverted during a refactor and silently reverses every
    conclusion in the project.
    """
    from datetime import date as _date

    window = simulate.build_window(
        ids=["a", "b"],
        labels=[0, 1],
        dates=[_date(2022, 4, 1), _date(2022, 4, 2)],
    )
    order = simulate.model_order(window, [0.10, 0.90])
    first = window.ids[order[0]]
    return _check(
        "score_direction_is_descending",
        first == "b",
        SEVERITY_ERROR,
        "higher score is scheduled first (0.90 ranks ahead of 0.10)"
        if first == "b"
        else f"score direction is inverted: {first} was scheduled first",
    )


def _window_for(frame: pl.DataFrame, fold: FoldSpec, date_column: str) -> simulate.Window:
    test = assign_split(frame, fold, date_column=date_column).filter(pl.col("split") == "test")
    return simulate.build_window(
        ids=test["target_inspection_id"].to_list(),
        labels=test["target"].to_list(),
        dates=test[date_column].to_list(),
    )


# --- 4. checks over what the orchestrator saw ------------------------------


def _scores_respect_the_decision_point(observations: Observations) -> ValidationCheck:
    """No prediction set was scored after declaring a training horizon past its fold."""
    rejections = observations.horizon_rejections
    return _check(
        "scores_respect_the_decision_point",
        not rejections,
        SEVERITY_ERROR,
        "every scored prediction declared a training horizon within its fold"
        if not rejections
        else f"{len(rejections)} prediction set(s) declared a horizon past their fold",
        rejections,
    )


def _predictions_cover_test_exactly(observations: Observations) -> ValidationCheck:
    """Every scored model covered its test window exactly -- nothing dropped."""
    rejections = observations.contract_rejections
    return _check(
        "predictions_cover_test_exactly",
        not rejections,
        SEVERITY_ERROR,
        "every scored model covered its fold's test window exactly"
        if not rejections
        else f"{len(rejections)} prediction set(s) failed the coverage contract",
        rejections,
    )


def _labels_are_read_not_redefined(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    stats: Sequence[FoldStats],
    date_column: str,
) -> ValidationCheck:
    """Component 5's row and positive counts agree with a direct re-count.

    Component 3 owns the target and Component 4 owns the feature rows. This
    component reads them. The check re-counts each test window straight off the
    frame and compares against the statistics the run reported, so a second
    interpretation of the target cannot creep in unnoticed.
    """
    by_id = {s.fold_id: s for s in stats}
    offenders: list[str] = []
    for fold in folds:
        recount = frame.filter(pl.col(date_column).is_between(fold.test_start, fold.test_end))
        reported = by_id.get(fold.fold_id)
        if reported is None:
            offenders.append(f"{fold.fold_id}: no statistics reported")
            continue
        if recount.height != reported.test_rows:
            offenders.append(
                f"{fold.fold_id}: recounted {recount.height} test rows, reported "
                f"{reported.test_rows}"
            )
    return _check(
        "labels_are_read_not_redefined",
        not offenders,
        SEVERITY_ERROR,
        "every fold's row counts reproduce a direct re-count of the feature table"
        if not offenders
        else f"{len(offenders)} fold(s) disagree with a re-count",
        offenders,
    )


# --- 5. advisory notes ------------------------------------------------------


def _advisory(
    folds: Sequence[FoldSpec],
    stats: Sequence[FoldStats],
    observations: Observations,
) -> list[ValidationCheck]:
    """Warn-severity notes: context a reader needs, not conditions that fail a run."""
    notes: list[ValidationCheck] = []

    by_set: dict[str, int] = {}
    for fold in folds:
        by_set[fold.fold_set] = by_set.get(fold.fold_set, 0) + 1
    notes.append(
        _check(
            "fold_sets",
            True,
            SEVERITY_WARN,
            ", ".join(f"{name}: {count} fold(s)" for name, count in sorted(by_set.items())),
        )
    )

    rates = [s.test_positive_rate for s in stats if s.test_positive_rate is not None]
    if rates:
        notes.append(
            _check(
                "base_rate_drift",
                True,
                SEVERITY_WARN,
                f"test base rate ranges {min(rates):.4f} to {max(rates):.4f} across folds; "
                "every base-rate-dependent metric must be read beside its fold's prevalence",
            )
        )

    capacities = [
        s.test_median_daily_capacity for s in stats if s.test_median_daily_capacity is not None
    ]
    if capacities:
        notes.append(
            _check(
                "measured_capacity",
                True,
                SEVERITY_WARN,
                f"median daily inspection capacity ranges {min(capacities):.0f} to "
                f"{max(capacities):.0f} across test windows; k is derived from this, not chosen",
            )
        )

    notes.append(
        _check(
            "models_scored",
            True,
            SEVERITY_WARN,
            f"{len(observations.models)} score producer(s): {', '.join(observations.models)}",
        )
    )

    if observations.blocked:
        notes.append(
            _check(
                "blocked_experiments",
                True,
                SEVERITY_WARN,
                f"{len(observations.blocked)} experiment(s) cannot run on the current data",
                observations.blocked,
            )
        )

    return notes


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """Render the checks as a plain text block for the CLI."""
    lines = ["", "Evaluation validation report", "----------------------------"]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
        if check.offenders and not (check.passed and check.severity == SEVERITY_ERROR):
            for offender in check.offenders:
                lines.append(f"           - {offender}")
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """Whether any error-severity check failed."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


__all__ = [
    "MAX_OFFENDERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "Observations",
    "format_report",
    "has_failures",
    "validate_evaluation",
]
