"""Runtime checks over a completed audit. Pure -- every check re-derives from the data.

**The severity split is this module's whole design, and ADR 0034 fixes it.**

```text
ERROR      the audit is wrong        -> fails the run, exit 1
ADVISORY   the world is uneven       -> recorded, printed, exit 0
```

A defect in the measurement fails the build. A disparity the measurement found does not, and
there is deliberately no flag to make one. The reason is not that inequality does not matter
-- this component exists because it does -- but that a red build is a demand for action, and
the only actions available to whoever faces a red fairness check are to change the model, to
change the metric, or to move the threshold. Two of those three are worse than the disparity,
and a test suite that creates pressure to take them is a test suite that will eventually get
what it asks for.

Every check re-derives its claim from the frames rather than reading a manifest. A manifest
records what a previous step believed; a check that reads one is asking the run to confirm
its own account of itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.fairness.definitions import (
    ADVISORY_CAPTURE_SPREAD,
    ADVISORY_ECE_SPREAD,
    ADVISORY_SELECTION_RATIO,
    CALIBRATION_MIN_ROWS,
    SUPPORT_MIN_NEGATIVE,
    SUPPORT_MIN_POSITIVE,
    SUPPORT_MIN_ROWS,
    GroupStatus,
)
from sentinel.fairness.models import MAX_OFFENDERS, SEVERITY_ERROR, SEVERITY_WARN, ValidationCheck
from sentinel.fairness.support import classify


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


def _cell_label(record: Mapping[str, object]) -> str:
    """Identify a cell fully enough that two rows are never confused for one.

    The grain and the fold belong in the label. Without them a pooled row and seventeen
    per-fold rows for the same (model, group, cutoff) render as identical strings, and an
    advisory list showing the same line four times reads as a bug in the audit rather than as
    four different quarters -- which is the opposite of what an advisory is for.
    """
    grain = str(record.get("grain", ""))
    fold_id = str(record.get("fold_id", ""))
    where = f"{grain}/{fold_id}" if fold_id else grain
    return (
        f"{record.get('model_name', '')}[{record.get('stage', '')}]/"
        f"{record.get('group_definition', '')}/{where}"
    )


# --- error-severity checks: is the audit sound? ------------------------------


def every_audited_row_has_a_prediction(
    audited: pl.DataFrame, predictions: pl.DataFrame
) -> ValidationCheck:
    """Every row measured corresponds to a row of the committed prediction artifact.

    Set equality, not containment. Containment would pass on a frame that quietly lost rows
    to a join, and a metric over a silently reduced population is exactly the defect no
    downstream number could reveal.
    """
    audited_keys = set(
        zip(
            audited.get_column("model_name").to_list(),
            audited.get_column("target_inspection_id").to_list(),
            strict=True,
        )
    )
    committed = set(
        zip(
            predictions.get_column("model_name").to_list(),
            predictions.get_column("target_inspection_id").to_list(),
            strict=True,
        )
    )
    missing = audited_keys - committed
    dropped = committed - audited_keys
    return _check(
        "every_audited_row_has_a_prediction",
        not missing and not dropped,
        (
            f"{len(audited_keys)} audited (model, id) pairs against {len(committed)} "
            f"committed; {len(missing)} fabricated, {len(dropped)} dropped"
        ),
        offenders=[f"{m}/{i}" for m, i in sorted(missing | dropped)],
    )


def every_group_value_comes_from_the_source(
    audited: pl.DataFrame,
    source: pl.DataFrame,
    columns: Sequence[str],
) -> ValidationCheck:
    """No group value was invented, normalised or remapped on its way into the audit.

    The audit reads a group column straight through, so any value it reports must appear in
    the group source. A value that does not would mean a transformation happened somewhere
    the artifact does not record -- and a neighbourhood id nobody can trace back is worse than
    no neighbourhood id.
    """
    offenders: list[str] = []
    for column in columns:
        if column not in audited.columns or column not in source.columns:
            offenders.append(f"{column}: column absent")
            continue
        seen = set(audited.get_column(column).unique().to_list())
        declared = set(source.get_column(column).unique().to_list())
        for value in sorted(seen - declared):
            offenders.append(f"{column}={value}")
    return _check(
        "every_group_value_comes_from_the_source",
        not offenders,
        f"{len(offenders)} audited group value(s) absent from the declared source column",
        offenders=offenders,
    )


def group_mapping_predates_every_row(source: pl.DataFrame) -> ValidationCheck:
    """No group value came from an inspection dated on or after the row it labels.

    Re-derived per row from the dates rather than read from Component 8's manifest. This is
    the leak this component is most exposed to and the least able to notice afterwards: a
    group value taken from the present would leave every metric finite, additive and
    plausible, and change only the question the numbers answer.
    """
    dated = source.with_columns(pl.col("inspection_date").str.to_date().alias("_rd"))
    with_source = dated.filter(pl.col("source_inspection_date").is_not_null())
    offenders = with_source.filter(pl.col("source_inspection_date") >= pl.col("_rd"))
    lags = with_source.get_column("days_since_source").drop_nulls()
    minimum = int(lags.min()) if lags.len() else None  # type: ignore[arg-type]
    return _check(
        "group_mapping_predates_every_row",
        offenders.is_empty() and (minimum is None or minimum >= 1),
        (
            f"{with_source.height} row(s) carry a source inspection; minimum lag "
            f"{minimum} day(s); {offenders.height} dated on or after their row"
        ),
        offenders=offenders.get_column("target_inspection_id").to_list(),
    )


def group_mapping_is_unambiguous(source: pl.DataFrame, columns: Sequence[str]) -> ValidationCheck:
    """One group value per key, per definition.

    A key mapping to two values would multiply audited rows on the join and inflate every
    support count -- making groups look better supported than they are, which is the one
    direction of error this component must not make.
    """
    offenders: list[str] = []
    keys = source.get_column("target_inspection_id")
    if keys.len() != keys.n_unique():
        offenders.append(f"{keys.len() - keys.n_unique()} duplicate key(s)")
    for column in columns:
        counts = source.group_by("target_inspection_id").agg(
            pl.col(column).n_unique().alias("values")
        )
        bad = counts.filter(pl.col("values") > 1)
        offenders.extend(f"{column}/{k}" for k in bad.get_column("target_inspection_id").to_list())
    return _check(
        "group_mapping_is_unambiguous",
        not offenders,
        f"{len(offenders)} ambiguous or duplicated group mapping(s)",
        offenders=offenders,
    )


def every_row_is_in_its_declared_fold(
    audited: pl.DataFrame, assignments: Mapping[str, str]
) -> ValidationCheck:
    """The fold on each audited row matches the split re-derived from the feature dates.

    ``assignments`` maps ``target_inspection_id`` to the fold whose *test* window contains its
    reference date, computed by the caller from ``evaluation.folds``. A row measured under the
    wrong fold would be scored by a model trained past its own date, which is the leak
    Component 5 built its whole contract to prevent.
    """
    offenders: list[str] = []
    for record in audited.select("target_inspection_id", "fold_id", "fold_set").unique().to_dicts():
        key = str(record["target_inspection_id"])
        expected = assignments.get(key)
        if expected is None:
            offenders.append(f"{key}: no re-derived fold")
        elif expected != str(record["fold_id"]):
            offenders.append(f"{key}: declared {record['fold_id']}, re-derived {expected}")
    return _check(
        "every_row_is_in_its_declared_fold",
        not offenders,
        f"{len(offenders)} row(s) whose declared fold does not match the re-derived split",
        offenders=offenders,
    )


def stages_are_not_confused(audited: pl.DataFrame, predictions: pl.DataFrame) -> ValidationCheck:
    """The base and calibrated columns hold what their names say.

    Re-joined against the committed artifact and compared with ``==``, not a tolerance. Platt
    moved every probability -- Component 9 measured ``score != base_score`` on all 207,680
    rows -- so a swap is detectable, and it is worth detecting: a calibrated ECE reported as
    an uncalibrated one would invert the component's central finding while every number stayed
    in range. MEMORY invariant 71.
    """
    joined = audited.join(
        predictions.select("target_inspection_id", "model_name", "score", "base_score"),
        on=["target_inspection_id", "model_name"],
        how="inner",
    )
    if joined.is_empty():
        return _check("stages_are_not_confused", False, "no rows to compare")
    calibrated_ok = joined.filter(pl.col("calibrated_probability") != pl.col("score")).height
    base_ok = joined.filter(pl.col("base_probability") != pl.col("base_score")).height
    distinct = joined.filter(pl.col("score") != pl.col("base_score")).height
    return _check(
        "stages_are_not_confused",
        calibrated_ok == 0 and base_ok == 0,
        (
            f"{joined.height} rows compared; {calibrated_ok} calibrated and {base_ok} base "
            f"mismatches; {distinct} rows where the two stages genuinely differ"
        ),
    )


def no_group_disappeared(
    support: pl.DataFrame, observed: Mapping[str, Sequence[str]]
) -> ValidationCheck:
    """Every group present in the data has a support row, whether or not it qualified.

    The check that makes the small-group policy real. Without it, a group could be filtered
    out anywhere upstream and the artifact would report "no disparity" in a font
    indistinguishable from "no disparity looked for". Absence of evidence has to stay visible.
    """
    offenders: list[str] = []
    for definition, values in observed.items():
        recorded = set(
            support.filter(pl.col("group_definition") == definition)
            .get_column("group_value")
            .unique()
            .to_list()
        )
        for value in sorted(set(values) - recorded):
            offenders.append(f"{definition}={value}")
    return _check(
        "no_group_disappeared",
        not offenders,
        f"{len(offenders)} observed group(s) missing from the support table",
        offenders=offenders,
    )


def every_metric_carries_support(metrics: pl.DataFrame) -> ValidationCheck:
    """No metric row is missing its counts, and no populated value is unsupported.

    Two failures in one check because they are the same defect from either side: a number
    without its support cannot be read, and a number published despite failing its floor is a
    number that should not have been read.
    """
    if metrics.is_empty():
        return _check("every_metric_carries_support", True, "no metric rows")
    missing_counts = metrics.filter(
        pl.col("n_rows").is_null() | pl.col("n_positive").is_null() | pl.col("n_negative").is_null()
    )
    published_unsupported = metrics.filter(
        (pl.col("group_status") == GroupStatus.INSUFFICIENT_SUPPORT.value)
        & pl.col("value").is_not_null()
    )
    return _check(
        "every_metric_carries_support",
        missing_counts.is_empty() and published_unsupported.is_empty(),
        (
            f"{missing_counts.height} row(s) without support counts; "
            f"{published_unsupported.height} unsupported row(s) carrying a value"
        ),
        offenders=published_unsupported.get_column("group_value").head(MAX_OFFENDERS).to_list(),
    )


def support_decisions_are_reproducible(support: pl.DataFrame) -> ValidationCheck:
    """Re-derive every support status from its own counts and the frozen floors.

    A status column that disagreed with the counts beside it would be the most quietly
    misleading thing this artifact could contain, because a reader checking the arithmetic
    would find the counts fine.
    """
    offenders: list[str] = []
    for record in support.to_dicts():
        ranking, calibration, _ = classify(
            int(record["n_rows"]), int(record["n_positive"]), int(record["n_negative"])
        )
        if (
            ranking.value != record["ranking_status"]
            or calibration.value != record["calibration_status"]
        ):
            offenders.append(f"{record['group_definition']}={record['group_value']}")
    return _check(
        "support_decisions_are_reproducible",
        not offenders,
        (
            f"{support.height} support row(s) re-derived against floors "
            f"{SUPPORT_MIN_ROWS}/{SUPPORT_MIN_POSITIVE}/{SUPPORT_MIN_NEGATIVE} rows and "
            f"{CALIBRATION_MIN_ROWS} for calibration; {len(offenders)} disagreed"
        ),
        offenders=offenders,
    )


def no_outcome_or_feature_column_leaked(tables: Mapping[str, pl.DataFrame]) -> ValidationCheck:
    """No label, score or feature column was smuggled into the artifact.

    The audit reads outcomes and probabilities and must publish neither. ``fairness_*`` tables
    are keyed by group rather than by row precisely so they cannot be joined back onto a
    feature table -- and a stray ``target`` or ``score`` column would undo that, turning the
    artifact into a per-row table one join from becoming a feature. ADR 0032.
    """
    forbidden = {"target", "score", "base_score", "calibrated_probability", "base_probability"}
    offenders = [
        f"{name}.{column}"
        for name, frame in sorted(tables.items())
        for column in frame.columns
        if column in forbidden
    ]
    return _check(
        "no_outcome_or_feature_column_leaked",
        not offenders,
        f"{len(offenders)} forbidden column(s) in the artifact",
        offenders=offenders,
    )


def tables_are_deterministically_sorted(
    tables: Mapping[str, pl.DataFrame], sort_keys: Mapping[str, Sequence[str]]
) -> ValidationCheck:
    """Every table is in its declared total order, with no duplicate key.

    Both halves matter. An unsorted table breaks byte-comparison between two runs; a duplicate
    key means the same measurement was written twice and one of them will be read.
    """
    offenders: list[str] = []
    for name, frame in sorted(tables.items()):
        keys = list(sort_keys[name])
        if frame.is_empty():
            continue
        if not frame.equals(frame.sort(keys)):
            offenders.append(f"{name}: not in sort order")
        if frame.select(keys).is_duplicated().any():
            offenders.append(f"{name}: duplicate sort key")
    return _check(
        "tables_are_deterministically_sorted",
        not offenders,
        f"{len(offenders)} table(s) unsorted or carrying a duplicate key",
        offenders=offenders,
    )


def inputs_were_not_modified(
    before: Mapping[str, str], after: Mapping[str, str]
) -> ValidationCheck:
    """Every input artifact's sha256 is unchanged.

    This component's entire value rests on it being an observer. An observer that quietly
    rewrote what it observed would invalidate every number in its own artifact and every
    earlier component's as well, so the claim is checked rather than asserted.
    """
    offenders = [
        f"{name}: {before[name][:12]} -> {digest[:12]}"
        for name, digest in sorted(after.items())
        if before.get(name) != digest
    ]
    return _check(
        "inputs_were_not_modified",
        not offenders,
        f"{len(after)} input artifact(s) re-checksummed; {len(offenders)} changed",
        offenders=offenders,
    )


def covid_was_not_pooled(tables: Mapping[str, pl.DataFrame]) -> ValidationCheck:
    """No aggregate row mixes `covid_shift` with the quarterly folds.

    An invariant since Component 5 and violated by accident in exactly the way a bare string
    comparison invites. Every aggregate row names exactly one fold set, so a row whose
    ``fold_set`` is neither of the two -- or a pooled 'all' -- would be a mean across a regime
    change that five components have now measured inverting their results.
    """
    allowed = {"quarterly", "covid_shift"}
    offenders = [
        f"{name}: fold_set={value}"
        for name, frame in sorted(tables.items())
        if "fold_set" in frame.columns
        for value in sorted(set(frame.get_column("fold_set").unique().to_list()) - allowed)
    ]
    return _check(
        "covid_was_not_pooled",
        not offenders,
        f"{len(offenders)} row group(s) naming a fold set outside {sorted(allowed)}",
        offenders=offenders,
    )


# --- advisory checks: what did the audit find? -------------------------------


def group_calibration_spread_is_modest(disparity: pl.DataFrame) -> ValidationCheck:
    """ADVISORY. How far apart are the groups' calibration errors?

    **A failure here is a finding, not a defect.** It says the model's probabilities matched
    outcomes less well in some neighbourhoods than others. It does not say the model
    discriminates, it does not say why, and it must never fail a build -- ADR 0034.
    """
    if disparity.is_empty():
        return _check(
            "group_calibration_spread_is_modest", True, "no disparity rows", severity=SEVERITY_WARN
        )
    ece = disparity.filter(
        (pl.col("metric") == "ece") & (pl.col("measure") == "spread")
    ).drop_nulls("value")
    worst = ece.sort("value", descending=True).head(MAX_OFFENDERS)
    exceeded = ece.filter(pl.col("value") > ADVISORY_ECE_SPREAD)
    return _check(
        "group_calibration_spread_is_modest",
        exceeded.is_empty(),
        (
            f"{exceeded.height} of {ece.height} cell(s) exceed an ECE spread of "
            f"{ADVISORY_ECE_SPREAD}. Evidence for Component 13, not an implementation error."
        ),
        severity=SEVERITY_WARN,
        offenders=[
            f"{r['model_name']}/{r['stage']}/{r['group_definition']}/{r['grain']}: "
            f"{r['value']:.4f} (worst {r['max_group']} on {r['max_group_rows']} rows)"
            for r in worst.to_dicts()
        ],
    )


def selection_rates_are_proportionate(priority: pl.DataFrame) -> ValidationCheck:
    """ADVISORY. How far from proportionate is each group's share of the top k?

    **Parity is not automatically desirable and this is not a target.** Outcome rates differ
    from 0.220 to 0.566 across supported community areas, so a working risk model is expected
    to select at different rates; equal selection would require ignoring a measured difference
    in outcomes. The advisory exists to make the trade-off visible, not to drive it to one.
    """
    if priority.is_empty():
        return _check(
            "selection_rates_are_proportionate", True, "no priority rows", severity=SEVERITY_WARN
        )
    supported = priority.filter(
        (pl.col("group_status") == GroupStatus.SUPPORTED.value)
        & pl.col("selection_rate_ratio").is_not_null()
    )
    extreme = supported.filter(
        (pl.col("selection_rate_ratio") > ADVISORY_SELECTION_RATIO)
        | (pl.col("selection_rate_ratio") < 1.0 / ADVISORY_SELECTION_RATIO)
    ).sort("selection_rate_ratio", descending=True)
    return _check(
        "selection_rates_are_proportionate",
        extreme.is_empty(),
        (
            f"{extreme.height} of {supported.height} supported (group, k) cell(s) sit outside "
            f"[{1 / ADVISORY_SELECTION_RATIO:.2f}, {ADVISORY_SELECTION_RATIO:.2f}]x the "
            "overall selection rate"
        ),
        severity=SEVERITY_WARN,
        offenders=[
            f"{_cell_label(r)} {r['group_value']}@{r['k_name']}: "
            f"{r['selection_rate_ratio']:.2f}x on {r['n_rows']} rows"
            for r in extreme.head(MAX_OFFENDERS).to_dicts()
        ],
    )


def capture_is_even_across_groups(disparity: pl.DataFrame) -> ValidationCheck:
    """ADVISORY. Does the priority set find the same share of every group's violations?

    The opportunity question, and the one most likely to matter operationally: a group whose
    violations are consistently found later is receiving less effective prioritisation, even
    if it is selected at an ordinary rate. Advisory, for the same reason as the others.
    """
    if disparity.is_empty():
        return _check(
            "capture_is_even_across_groups", True, "no disparity rows", severity=SEVERITY_WARN
        )
    capture = disparity.filter(
        (pl.col("metric") == "capture_rate") & (pl.col("measure") == "spread")
    ).drop_nulls("value")
    exceeded = capture.filter(pl.col("value") > ADVISORY_CAPTURE_SPREAD)
    return _check(
        "capture_is_even_across_groups",
        exceeded.is_empty(),
        (
            f"{exceeded.height} of {capture.height} cell(s) exceed a capture spread of "
            f"{ADVISORY_CAPTURE_SPREAD}"
        ),
        severity=SEVERITY_WARN,
        offenders=[
            f"{_cell_label(r)}@{r['k_name']}: {r['value']:.4f} "
            f"(worst {r['min_group']} at {r['min_value']:.4f} on {r['min_group_rows']} rows)"
            for r in capture.sort("value", descending=True).head(MAX_OFFENDERS).to_dicts()
        ],
    )


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """The full check list, errors first, with the boundary printed underneath.

    The boundary line is not decoration. A run that prints only green checks beside the word
    "fairness" invites exactly the reading ADR 0035 exists to prevent, so the summary says
    what a green run does and does not mean, every time.
    """
    lines = ["", "Component 12 -- fairness audit validation", ""]
    order = sorted(checks, key=lambda c: (c.passed, c.severity != SEVERITY_ERROR, c.name))
    for check in order:
        mark = "PASS" if check.passed else ("FAIL" if check.severity == SEVERITY_ERROR else "NOTE")
        lines.append(f"  [{mark}] {check.name} ({check.severity})")
        lines.append(f"         {check.detail}")
        for offender in check.offenders:
            lines.append(f"           - {offender}")
    errors = sum(1 for c in checks if not c.passed and c.severity == SEVERITY_ERROR)
    warns = sum(1 for c in checks if not c.passed and c.severity == SEVERITY_WARN)
    lines.extend(
        [
            "",
            f"  {len(checks)} check(s): {errors} error(s), {warns} advisory finding(s)",
            "",
            "  A green run means the audit is internally sound. It does NOT mean Sentinel is",
            "  fair. An advisory finding is a measured disparity -- evidence for a policy",
            "  component, never an implementation error. See ADR 0034 and ADR 0035.",
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


__all__ = [
    "advisory_findings",
    "capture_is_even_across_groups",
    "covid_was_not_pooled",
    "every_audited_row_has_a_prediction",
    "every_group_value_comes_from_the_source",
    "every_metric_carries_support",
    "every_row_is_in_its_declared_fold",
    "format_report",
    "group_calibration_spread_is_modest",
    "group_mapping_is_unambiguous",
    "group_mapping_predates_every_row",
    "has_failures",
    "inputs_were_not_modified",
    "no_group_disappeared",
    "no_outcome_or_feature_column_leaked",
    "selection_rates_are_proportionate",
    "stages_are_not_confused",
    "support_decisions_are_reproducible",
    "tables_are_deterministically_sorted",
]
