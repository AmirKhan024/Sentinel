"""Runtime checks over a completed review run. Pure -- every check re-derives from the data.

The severity split is inherited unchanged from ADR 0034, restated by every component since: a
defect in the computation is an error and fails the run; a finding about the run (how many cases
were flagged, whether a pointer resolves yet) is an advisory and never does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.review.definitions import POINTER_FIELD_FOR_ACTION, ReviewResolutionAction
from sentinel.review.models import MAX_OFFENDERS, SEVERITY_ERROR, SEVERITY_WARN, ValidationCheck
from sentinel.review.trigger import build_review_cases, trigger_column


def _check(
    name: str,
    passed: bool,
    detail: str,
    *,
    severity: str = SEVERITY_ERROR,
    offenders: Sequence[str] = (),
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail,
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- error-severity checks: was the queue built correctly? ---------------------


def every_case_carries_a_trigger(queue: pl.DataFrame) -> ValidationCheck:
    """No row in the queue has a blank or none trigger_reasons column."""
    if queue.is_empty():
        return _check("every_case_carries_a_trigger", True, "no queue rows")
    bad = queue.filter((pl.col("trigger_reasons") == "") | (pl.col("trigger_reasons") == "none"))
    offenders = [
        str(r["target_inspection_id"]) for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "every_case_carries_a_trigger",
        bad.is_empty(),
        f"{bad.height} queue row(s) carry no trigger reason",
        offenders=offenders,
    )


def warning_trigger_rows_are_selected_and_warned(
    queue: pl.DataFrame, recommendations: pl.DataFrame
) -> ValidationCheck:
    """Every row reached through the warning trigger was selected and carried a warning.

    Re-derived from the recommendation table rather than trusted, matching the discipline every
    upstream component applies to its own claims.
    """
    if queue.is_empty():
        return _check("warning_trigger_rows_are_selected_and_warned", True, "no queue rows")
    warned = queue.filter(
        pl.col("trigger_reasons").str.contains("policy_warning_present", literal=True)
    )
    if warned.is_empty():
        return _check(
            "warning_trigger_rows_are_selected_and_warned", True, "no warning-triggered rows"
        )
    cell_keys = ["policy_id", "model_name", "fold_set", "fold_id", "k_name", "target_inspection_id"]
    joined = warned.select(cell_keys).join(
        recommendations.select(*cell_keys, "is_selected", "warnings"),
        on=cell_keys,
        how="left",
    )
    bad = joined.filter(
        pl.col("is_selected").is_null()
        | ~pl.col("is_selected")
        | (pl.col("warnings") == "none")
        | pl.col("warnings").is_null()
    )
    offenders = [
        str(r["target_inspection_id"]) for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "warning_trigger_rows_are_selected_and_warned",
        bad.is_empty(),
        f"{bad.height} warning-triggered row(s) were not selected or carried no warning",
        offenders=offenders,
    )


def queue_is_deterministically_rebuildable(
    queue: pl.DataFrame,
    recommendations: pl.DataFrame,
    schedule: pl.DataFrame | None,
    execution_log: pl.DataFrame | None,
) -> ValidationCheck:
    """Rebuilding the queue from the two triggers reproduces it exactly.

    The check that proves the queue reads only ``warnings`` and the schedule/execution
    anti-join -- never ``score``, ``base_score`` or ``final_policy_rank`` -- to decide
    membership.
    """
    rebuilt_cases = build_review_cases(recommendations, schedule, execution_log)
    rebuilt = sorted(
        (
            case.policy_id,
            case.model_name,
            case.fold_set,
            case.fold_id,
            case.k_name,
            case.target_inspection_id,
            trigger_column(case),
        )
        for case in rebuilt_cases
    )
    actual = sorted(
        (
            r["policy_id"],
            r["model_name"],
            r["fold_set"],
            r["fold_id"],
            r["k_name"],
            r["target_inspection_id"],
            r["trigger_reasons"],
        )
        for r in queue.iter_rows(named=True)
    )
    same = rebuilt == actual
    return _check(
        "queue_is_deterministically_rebuildable",
        same,
        "the queue is byte-identical to a fresh rebuild from the two triggers"
        if same
        else "rebuilding the queue from the two triggers produced a different set of cases",
    )


def no_duplicate_review_id(resolution_log: pl.DataFrame) -> ValidationCheck:
    """Every applied review_id in the log is unique."""
    if resolution_log.is_empty():
        return _check("no_duplicate_review_id", True, "no resolutions were applied")
    duplicated = resolution_log.select("review_id").is_duplicated()
    offenders = [
        str(r["review_id"])
        for r in resolution_log.filter(duplicated).head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "no_duplicate_review_id",
        not offenders,
        f"{int(duplicated.sum())} duplicated review_id row(s)",
        offenders=offenders,
    )


def pointer_fields_are_mutually_exclusive(resolution_log: pl.DataFrame) -> ValidationCheck:
    """Each resolution carries exactly the pointer field its action requires, and no other."""
    if resolution_log.is_empty():
        return _check("pointer_fields_are_mutually_exclusive", True, "no resolutions were applied")
    offenders: list[str] = []
    for row in resolution_log.iter_rows(named=True):
        action = row["resolution_action"]
        required = POINTER_FIELD_FOR_ACTION.get(action)
        override_present = bool((row.get("referenced_override_id") or "").strip())
        adjustment_present = bool((row.get("referenced_adjustment_id") or "").strip())
        if required == "referenced_override_id" and not override_present:
            offenders.append(f"{row['review_id']}: missing referenced_override_id")
        if required == "referenced_adjustment_id" and not adjustment_present:
            offenders.append(f"{row['review_id']}: missing referenced_adjustment_id")
        if required != "referenced_override_id" and override_present:
            offenders.append(f"{row['review_id']}: unexpected referenced_override_id")
        if required != "referenced_adjustment_id" and adjustment_present:
            offenders.append(f"{row['review_id']}: unexpected referenced_adjustment_id")
    return _check(
        "pointer_fields_are_mutually_exclusive",
        not offenders,
        f"{len(offenders)} resolution(s) carry the wrong pointer field for their action",
        offenders=offenders,
    )


def review_status_reflects_one_applied_resolution(
    queue: pl.DataFrame, resolution_log: pl.DataFrame
) -> ValidationCheck:
    """A case is RESOLVED in the queue iff exactly one applied resolution names its scope."""
    if queue.is_empty():
        return _check("review_status_reflects_one_applied_resolution", True, "no queue rows")
    applied = (
        resolution_log.filter(pl.col("outcome") == "applied")
        if not resolution_log.is_empty()
        else resolution_log
    )
    resolved_keys: set[tuple[str, str, str, str]] = set()
    if not applied.is_empty():
        resolved_keys = {
            (r["policy_id"], r["fold_id"], r["k_name"], r["target_inspection_id"])
            for r in applied.iter_rows(named=True)
        }
    offenders = []
    for row in queue.iter_rows(named=True):
        key = (row["policy_id"], row["fold_id"], row["k_name"], row["target_inspection_id"])
        expected = "resolved" if key in resolved_keys else "flagged"
        if row["review_status"] != expected:
            offenders.append(
                f"{row['target_inspection_id']}: status {row['review_status']!r}, expected "
                f"{expected!r}"
            )
    return _check(
        "review_status_reflects_one_applied_resolution",
        not offenders,
        f"{len(offenders)} queue row(s) disagree with the applied resolution log",
        offenders=offenders[:MAX_OFFENDERS],
    )


def resolution_verbs_do_not_collide(resolution_log: pl.DataFrame) -> ValidationCheck:
    """Re-check, from the written artifact, that resolution actions are Component 16's own.

    Matches Component 14's ``adjustments_are_not_overrides``-style re-check: the import-time
    guard in ``definitions.py`` proves the vocabulary is disjoint, and this proves the artifact
    the run actually wrote only ever used it.
    """
    known = {a.value for a in ReviewResolutionAction}
    if resolution_log.is_empty():
        return _check("resolution_verbs_do_not_collide", True, "no resolutions were applied")
    seen = set(resolution_log["resolution_action"].unique().to_list())
    unknown = sorted(seen - known)
    return _check(
        "resolution_verbs_do_not_collide",
        not unknown,
        f"{len(unknown)} unknown resolution action(s): {', '.join(unknown)}",
        offenders=unknown,
    )


def inputs_were_not_modified(
    before: Mapping[str, str], after: Mapping[str, str]
) -> ValidationCheck:
    """Every input file is byte-identical after the run.

    This component is a pure observer of Components 13 and 14. It fits nothing and edits
    nothing, and the way that stops being a promise and starts being a fact is a checksum taken
    before the first read and again after the last write.
    """
    offenders = [
        f"{name}: {before[name][:12]} -> {after.get(name, 'absent')[:12]}"
        for name in sorted(before)
        if after.get(name) != before[name]
    ]
    return _check(
        "inputs_were_not_modified",
        not offenders,
        f"{len(offenders)} input artifact(s) changed during the run",
        offenders=offenders,
    )


# --- advisory checks: what did this run find? -----------------------------------


def pointer_targets_exist(
    resolution_log: pl.DataFrame, override_ids: frozenset[str], adjustment_ids: frozenset[str]
) -> ValidationCheck:
    """ADVISORY. Does every pointer already resolve to a committed override or adjustment?

    A failure here is expected and not a defect: a human may record intent before submitting the
    override or adjustment itself. See ``POINTER_MAY_FORWARD_REFERENCE``.
    """
    if resolution_log.is_empty():
        return _check(
            "pointer_targets_exist", True, "no resolutions were applied", severity=SEVERITY_WARN
        )
    offenders = []
    for row in resolution_log.iter_rows(named=True):
        override_id = (row.get("referenced_override_id") or "").strip()
        adjustment_id = (row.get("referenced_adjustment_id") or "").strip()
        if override_id and override_id not in override_ids:
            offenders.append(f"{row['review_id']}: override_id {override_id} not yet committed")
        if adjustment_id and adjustment_id not in adjustment_ids:
            offenders.append(f"{row['review_id']}: adjustment_id {adjustment_id} not yet committed")
    return _check(
        "pointer_targets_exist",
        not offenders,
        f"{len(offenders)} pointer(s) reference an override or adjustment not yet committed. "
        "A forward reference to intent stated before submission, not an error.",
        severity=SEVERITY_WARN,
        offenders=offenders,
    )


def cases_flagged_by_trigger(queue: pl.DataFrame) -> ValidationCheck:
    """ADVISORY. Cases flagged this run, by trigger. Always emitted, never thresholded."""
    if queue.is_empty():
        return _check("cases_flagged_by_trigger", True, "no cases flagged", severity=SEVERITY_WARN)
    offenders = []
    for reason in ("policy_warning_present", "no_execution_record_on_scheduled_row"):
        count = queue.filter(pl.col("trigger_reasons").str.contains(reason, literal=True)).height
        if count:
            offenders.append(f"{reason}: {count}")
    return _check(
        "cases_flagged_by_trigger",
        False,
        f"{queue.height} case(s) flagged this run",
        severity=SEVERITY_WARN,
        offenders=offenders,
    )


# --- terminators -----------------------------------------------------------------


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """The full check list, errors first, with the boundary printed underneath."""
    lines = ["", "Component 16 -- deferral / human-review gate validation", ""]
    order = sorted(checks, key=lambda c: (c.passed, c.severity != SEVERITY_ERROR, c.name))
    for check in order:
        mark = "PASS" if check.passed else ("FAIL" if check.severity == SEVERITY_ERROR else "NOTE")
        lines.append(f"  [{mark}] {check.name} ({check.severity})")
        lines.append(f"         {check.detail}")
        lines.extend(f"           - {offender}" for offender in check.offenders)
    errors = sum(1 for c in checks if not c.passed and c.severity == SEVERITY_ERROR)
    warns = sum(1 for c in checks if not c.passed and c.severity == SEVERITY_WARN)
    lines.extend(
        [
            "",
            f"  {len(checks)} check(s): {errors} error(s), {warns} advisory finding(s)",
            "",
            "  A green run means the queue was built correctly and every resolution was applied",
            "  as recorded. It does not mean any flagged case is wrong, or that no unflagged",
            "  case needed review.",
            "",
        ]
    )
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """True when any error-severity check failed. Advisories never fail a run."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


def advisory_findings(checks: Sequence[ValidationCheck]) -> list[str]:
    """The advisory checks that fired, for the manifest."""
    return [f"{c.name}: {c.detail}" for c in checks if not c.passed and c.severity == SEVERITY_WARN]


def advisory_rows(
    checks: Sequence[ValidationCheck], *, definition_version: str
) -> list[dict[str, object]]:
    """Advisories as table rows."""
    return [
        {
            "code": check.name,
            "severity": check.severity,
            "scope": check.offenders[0] if check.offenders else "run",
            "n_cases": len(check.offenders),
            "detail": check.detail,
            "review_definition_version": definition_version,
        }
        for check in sorted(checks, key=lambda c: c.name)
        if not check.passed and check.severity == SEVERITY_WARN
    ]


__all__ = [
    "advisory_findings",
    "advisory_rows",
    "cases_flagged_by_trigger",
    "every_case_carries_a_trigger",
    "format_report",
    "has_failures",
    "inputs_were_not_modified",
    "no_duplicate_review_id",
    "pointer_fields_are_mutually_exclusive",
    "pointer_targets_exist",
    "queue_is_deterministically_rebuildable",
    "resolution_verbs_do_not_collide",
    "review_status_reflects_one_applied_resolution",
    "warning_trigger_rows_are_selected_and_warned",
]
