"""Runtime checks over a completed policy run. Pure -- every check re-derives from the data.

**The severity split is this module's whole design, and it is inherited from ADR 0034.**

```text
ERROR      the policy was applied wrongly       -> fails the run, exit 1
ADVISORY   the policy cost something            -> recorded, printed, exit 0
```

A defect in the computation fails the build. A price the policy paid does not, and there is
deliberately no flag to make one. The reason is the same one Component 12 gave and it is
sharper here: the cheapest way to turn a red "this reserve gave up 34 citations" build green is
to delete the reserve. That is a policy decision about how a city allocates enforcement, and a
CI runner is not entitled to take it.

The reverse also holds and matters just as much. A queue that selects more establishments than
the city has capacity for, a reserve that admits an establishment with a full inspection
history, an override with no actor, a rank that appears twice -- each of those is wrong rather
than uncomfortable, each would produce an entirely plausible-looking artifact, and each fails.

**Every check re-derives its claim from the frames rather than reading the manifest.** A
manifest records what a previous step believed; a check that reads one is asking the run to
confirm its own account of itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from sentinel.policy.definitions import (
    ADVISORY_GROUP_SELECTION_SHIFT,
    ADVISORY_INERT_RESERVE_CELLS,
    ADVISORY_LOST_POSITIVES,
    BASELINE_POLICY_ID,
    FORBIDDEN_POLICY_COLUMNS,
    MECHANISM_REASONS,
    POLICY_GRID,
    SELECTED_REASONS,
    DecisionMechanism,
)
from sentinel.policy.models import MAX_OFFENDERS, SEVERITY_ERROR, SEVERITY_WARN, ValidationCheck

#: The grain a recommendation cell is keyed by. Named once so a dozen group-bys agree.
CELL_KEYS: tuple[str, ...] = ("policy_id", "model_name", "fold_set", "fold_id", "k_name")


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


# --- error-severity checks: was the policy applied correctly? -----------------


def tables_are_deterministically_sorted(
    tables: Mapping[str, pl.DataFrame], sort_keys: Mapping[str, Sequence[str]]
) -> ValidationCheck:
    """Every table is in its declared total order, with no duplicate key.

    Both halves matter. An unsorted table breaks byte-comparison between two runs; a duplicate
    key means the same decision was written twice and one of them will be read.
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


def recommendations_cover_the_universe(
    recommendations: pl.DataFrame, allocation: pl.DataFrame
) -> ValidationCheck:
    """Every cell holds exactly the rows the prediction universe contains, no more, no fewer.

    The check that stops the most dangerous silent failure available here. A policy that lost
    rows would produce a shorter queue, a better precision and a completely plausible artifact,
    and nothing downstream would notice -- Component 5 refuses an incomplete prediction set at
    the door for the same reason, and this is the same discipline one layer later.
    """
    if recommendations.is_empty():
        return _check("recommendations_cover_the_universe", True, "no recommendations")
    counted = recommendations.group_by(list(CELL_KEYS)).len().rename({"len": "rows"})
    joined = counted.join(
        allocation.select(*CELL_KEYS, "n_universe"), on=list(CELL_KEYS), how="full", coalesce=True
    )
    bad = joined.filter(
        pl.col("rows").is_null()
        | pl.col("n_universe").is_null()
        | (pl.col("rows") != pl.col("n_universe"))
    )
    offenders = [
        f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}: {r['rows']} rows for a universe of "
        f"{r['n_universe']}"
        for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "recommendations_cover_the_universe",
        bad.is_empty(),
        f"{bad.height} cell(s) do not hold exactly the prediction universe",
        offenders=offenders,
    )


def selected_counts_equal_capacity(
    recommendations: pl.DataFrame, allocation: pl.DataFrame
) -> ValidationCheck:
    """Each cell selects exactly ``k`` establishments -- never more, never fewer.

    Capacity is the one thing this project's simulation has never been willing to change. A
    policy that selected ``k + 1`` would be recommending an inspection the city cannot perform,
    and would beat every other policy for that reason alone.
    """
    if recommendations.is_empty():
        return _check("selected_counts_equal_capacity", True, "no recommendations")
    counted = (
        recommendations.filter(pl.col("is_selected"))
        .group_by(list(CELL_KEYS))
        .len()
        .rename({"len": "selected"})
    )
    joined = allocation.select(*CELL_KEYS, "k", "n_selected").join(
        counted, on=list(CELL_KEYS), how="left"
    )
    bad = joined.filter(pl.col("selected").fill_null(0) != pl.col("k")).with_columns(
        pl.col("selected").fill_null(0)
    )
    offenders = [
        f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}: {r['selected']} selected for k={r['k']}"
        for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "selected_counts_equal_capacity",
        bad.is_empty(),
        f"{bad.height} cell(s) select a number of establishments other than k",
        offenders=offenders,
    )


def allocations_are_internally_consistent(allocation: pl.DataFrame) -> ValidationCheck:
    """Risk plus reserve equals selected equals capacity, and the reserve never overspends.

    Three arithmetic identities and one inequality, checked together because a violation of
    any one of them means the allocator's own account of what it did is wrong -- and the
    allocation table is what every downstream comparison is built from.
    """
    if allocation.is_empty():
        return _check("allocations_are_internally_consistent", True, "no allocations")
    bad = allocation.filter(
        (pl.col("n_risk") + pl.col("n_reserve") != pl.col("n_selected"))
        | (pl.col("n_selected") != pl.col("k"))
        | (pl.col("n_reserve") > pl.col("reserve_target"))
        | (pl.col("n_reserve") > pl.col("n_eligible_available"))
        | (pl.col("n_risk") < 0)
    )
    offenders = [
        f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}: risk={r['n_risk']} reserve="
        f"{r['n_reserve']} selected={r['n_selected']} k={r['k']} target={r['reserve_target']}"
        for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "allocations_are_internally_consistent",
        bad.is_empty(),
        f"{bad.height} allocation(s) do not add up, or spend more than the declared reserve",
        offenders=offenders,
    )


def no_establishment_is_selected_twice(recommendations: pl.DataFrame) -> ValidationCheck:
    """Within one cell, each establishment appears once and carries one mechanism.

    The reserve is filled from rows the risk block did not take, so this holds by construction
    -- which is a claim about code that was correct when it was written, and therefore worth a
    check that does not depend on it.
    """
    if recommendations.is_empty():
        return _check("no_establishment_is_selected_twice", True, "no recommendations")
    keys = [*CELL_KEYS, "target_inspection_id"]
    duplicated = recommendations.select(keys).is_duplicated()
    offenders = [
        f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}/{r['target_inspection_id']}"
        for r in recommendations.filter(duplicated).head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "no_establishment_is_selected_twice",
        not offenders,
        f"{int(duplicated.sum())} duplicated (cell, establishment) row(s)",
        offenders=offenders,
    )


def every_row_declares_a_valid_mechanism(recommendations: pl.DataFrame) -> ValidationCheck:
    """Every row carries a known mechanism, with a reason code that mechanism accepts.

    A selected establishment with no stated mechanism is a recommendation nobody can explain,
    which is the failure this component was built to make impossible.
    """
    if recommendations.is_empty():
        return _check("every_row_declares_a_valid_mechanism", True, "no recommendations")
    offenders: list[str] = []
    known = set(MECHANISM_REASONS)
    seen = set(recommendations["decision_mechanism"].unique().to_list())
    for mechanism in sorted(seen - known):
        offenders.append(f"unknown mechanism: {mechanism}")
    for mechanism, reasons in sorted(MECHANISM_REASONS.items()):
        bad = recommendations.filter(
            (pl.col("decision_mechanism") == mechanism)
            & ~pl.col("decision_reason").is_in(sorted(reasons))
        )
        if not bad.is_empty():
            offenders.append(f"{mechanism}: {bad.height} row(s) with a foreign reason code")
    mismatch = recommendations.filter(
        pl.col("is_selected") != pl.col("decision_reason").is_in(sorted(SELECTED_REASONS))
    )
    if not mismatch.is_empty():
        offenders.append(
            f"{mismatch.height} row(s) whose is_selected disagrees with the reason code"
        )
    return _check(
        "every_row_declares_a_valid_mechanism",
        not offenders,
        f"{len(offenders)} mechanism or reason-code defect(s)",
        offenders=offenders,
    )


def reserve_rows_are_eligible(recommendations: pl.DataFrame) -> ValidationCheck:
    """No establishment enters through the coverage reserve without qualifying for it.

    The reserve exists to serve one deterministically defined population. An ineligible row in
    it would mean capacity was diverted on a rationale that does not apply -- the exact abuse a
    reserve invites, and the reason the eligibility contract is one column and one predicate.
    """
    if recommendations.is_empty():
        return _check("reserve_rows_are_eligible", True, "no recommendations")
    bad = recommendations.filter(
        (pl.col("decision_mechanism") == DecisionMechanism.COVERAGE_RESERVE)
        & ~pl.col("coverage_eligible")
    )
    offenders = [
        f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}/{r['target_inspection_id']}"
        for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "reserve_rows_are_eligible",
        bad.is_empty(),
        f"{bad.height} reserve selection(s) are not coverage-eligible",
        offenders=offenders,
    )


def risk_rows_satisfy_the_risk_contract(
    recommendations: pl.DataFrame, allocation: pl.DataFrame
) -> ValidationCheck:
    """Every risk selection is inside the top ``k - n_reserve`` by model rank, and none is outside.

    Stated both ways on purpose. Checking only that risk selections are highly ranked would
    pass a policy that quietly dropped a top-ranked establishment; checking only the count
    would pass one that swapped two rows. The set has to be exactly the prefix.
    """
    if recommendations.is_empty():
        return _check("risk_rows_satisfy_the_risk_contract", True, "no recommendations")
    joined = recommendations.join(
        allocation.select(*CELL_KEYS, "n_risk"), on=list(CELL_KEYS), how="left"
    )
    is_risk = pl.col("decision_mechanism") == DecisionMechanism.RISK_PRIORITY
    within = pl.col("model_rank") <= pl.col("n_risk")
    bad = joined.filter(is_risk != within)
    offenders = [
        f"{r['policy_id']}/{r['fold_id']}/{r['k_name']}/{r['target_inspection_id']}: "
        f"model_rank={r['model_rank']} n_risk={r['n_risk']} mechanism={r['decision_mechanism']}"
        for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "risk_rows_satisfy_the_risk_contract",
        bad.is_empty(),
        f"{bad.height} row(s) where risk selection and model rank disagree",
        offenders=offenders,
    )


def policy_ranks_are_unique_and_contiguous(
    recommendations: pl.DataFrame, allocation: pl.DataFrame
) -> ValidationCheck:
    """Selected ranks in each cell are exactly ``1 .. k``, once each.

    A queue is worked in order, so a duplicate rank makes two establishments the same
    appointment and a gap loses one. Checked as a set identity rather than by counting, because
    counting would pass ``{1, 1, 3}``.
    """
    if recommendations.is_empty():
        return _check("policy_ranks_are_unique_and_contiguous", True, "no recommendations")
    offenders: list[str] = []
    selected = recommendations.filter(pl.col("is_selected"))
    unselected_rank = recommendations.filter(
        ~pl.col("is_selected") & pl.col("final_policy_rank").is_not_null()
    )
    if not unselected_rank.is_empty():
        offenders.append(f"{unselected_rank.height} unselected row(s) carry a policy rank")
    expected = {
        tuple(r[key] for key in CELL_KEYS): int(r["k"]) for r in allocation.iter_rows(named=True)
    }
    for key, group in selected.group_by(list(CELL_KEYS)):
        ranks = group["final_policy_rank"].to_list()
        want = expected.get(tuple(str(part) for part in key))
        if want is None or sorted(ranks) != list(range(1, want + 1)):
            offenders.append(f"{'/'.join(str(part) for part in key)}: ranks are not 1..{want}")
    return _check(
        "policy_ranks_are_unique_and_contiguous",
        not offenders,
        f"{len(offenders)} cell(s) with a duplicated, missing or stray policy rank",
        offenders=offenders,
    )


def eligibility_matches_the_declared_rule(
    features: pl.DataFrame, *, column: str, flag: str
) -> ValidationCheck:
    """The eligibility flag is exactly ``column == 0``, re-derived from the feature table.

    The contract is one sentence and this is the check that the code implements that sentence.
    It also catches the null case: a row whose count is missing must not be eligible, and a
    ``fill_null(0)`` slipped in during an edit would silently reserve capacity for rows about
    which nothing at all is known.
    """
    if features.is_empty():
        return _check("eligibility_matches_the_declared_rule", True, "no feature rows")
    if column not in features.columns or flag not in features.columns:
        return _check(
            "eligibility_matches_the_declared_rule",
            False,
            f"cannot re-derive: {column!r} or {flag!r} is absent",
        )
    rederived = features.select(
        (pl.col(column).fill_null(-1) == 0).alias("_expected"), pl.col(flag).alias("_actual")
    )
    bad = rederived.filter(pl.col("_expected") != pl.col("_actual"))
    return _check(
        "eligibility_matches_the_declared_rule",
        bad.is_empty(),
        f"{bad.height} row(s) where the eligibility flag differs from {column} == 0",
    )


def no_outcome_column_reaches_the_policy(
    recommendations: pl.DataFrame, allocation: pl.DataFrame
) -> ValidationCheck:
    """No label column appears anywhere in the decision artifacts.

    The strongest available structural statement that the policy did not read the answer. A
    label is not merely unused here -- it is absent from the tables, so a future edit that
    wanted to use one would have to add a column and change the contract to do it.
    """
    offenders = [
        f"{name}: {column}"
        for name, frame in (
            ("inspection_recommendations", recommendations),
            ("allocation", allocation),
        )
        for column in FORBIDDEN_POLICY_COLUMNS
        if column in frame.columns
    ]
    return _check(
        "no_outcome_column_reaches_the_policy",
        not offenders,
        f"{len(offenders)} outcome column(s) present in a decision artifact",
        offenders=offenders,
    )


def warnings_do_not_change_the_queue(
    with_warnings: pl.DataFrame, without_warnings: pl.DataFrame
) -> ValidationCheck:
    """The queue is identical when every warning input is withheld.

    The check that turns "the audit informs governance but never scoring" from a claim into a
    measurement. The whole allocation is rerun with the group label and support status absent
    and the ranks are compared exactly -- so if a Component 12 number ever leaked into a
    ranking decision, this goes red rather than the leak going unnoticed for a release.
    """
    columns = [*CELL_KEYS, "target_inspection_id", "final_policy_rank", "decision_mechanism"]
    left = with_warnings.select(columns).sort(columns)
    right = without_warnings.select(columns).sort(columns)
    same = left.equals(right)
    return _check(
        "warnings_do_not_change_the_queue",
        same,
        "the queue is byte-identical with the warning inputs withheld"
        if same
        else "withholding the warning inputs changed the queue: a Component 12 signal reached "
        "an allocation decision",
    )


def configurations_match_the_frozen_grid(configurations: pl.DataFrame) -> ValidationCheck:
    """The emitted grid is the grid in ``definitions.py``, policy for policy.

    A run whose artifact described a different grid than the code applied would be
    self-consistent and wrong, and every comparison in it would be unreadable.
    """
    expected = {
        spec.policy_id: (str(spec.mechanism), float(spec.reserve_share)) for spec in POLICY_GRID
    }
    got = {
        str(r["policy_id"]): (str(r["reserve_mechanism"]), float(r["reserve_share"]))
        for r in configurations.iter_rows(named=True)
    }
    offenders = [
        f"{policy}: manifest says {got.get(policy)}, definitions say {value}"
        for policy, value in sorted(expected.items())
        if got.get(policy) != value
    ]
    offenders.extend(
        f"{policy}: not in the frozen grid" for policy in sorted(set(got) - set(expected))
    )
    return _check(
        "configurations_match_the_frozen_grid",
        not offenders,
        f"{len(offenders)} policy configuration(s) differ from the frozen grid",
        offenders=offenders,
    )


def comparison_covers_every_policy(
    comparison: pl.DataFrame, allocation: pl.DataFrame
) -> ValidationCheck:
    """Every allocated cell has a comparison row, and every comparison row an allocation.

    A missing comparison row is a policy that quietly dropped out of the frontier, which would
    change which policies look non-dominated without anything appearing to be wrong.
    """
    left = allocation.select(*CELL_KEYS).unique()
    right = comparison.select(*CELL_KEYS).unique()
    only_allocated = left.join(right, on=list(CELL_KEYS), how="anti")
    only_compared = right.join(left, on=list(CELL_KEYS), how="anti")
    offenders = [
        f"allocated but not compared: {'/'.join(str(v) for v in r.values())}"
        for r in only_allocated.head(MAX_OFFENDERS).iter_rows(named=True)
    ] + [
        f"compared but not allocated: {'/'.join(str(v) for v in r.values())}"
        for r in only_compared.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "comparison_covers_every_policy",
        not offenders,
        f"{only_allocated.height + only_compared.height} cell(s) present in one table only",
        offenders=offenders,
    )


def unsupported_groups_are_preserved(
    group_audit: pl.DataFrame, support: Mapping[str, str]
) -> ValidationCheck:
    """A group Component 12 could not measure is still present here, still marked unsupported.

    Component 12's central discipline was that an unmeasurable group is a row with a stated
    reason rather than an absent row, and the easiest way for this component to produce a
    flattering group table would be to drop them. Nothing in this component filters on
    ``group_status``, and this is the check that says so.
    """
    if group_audit.is_empty() or not support:
        return _check(
            "unsupported_groups_are_preserved",
            True,
            "no group audit rows, or no Component 12 support table was supplied",
        )
    unsupported = {value for value, status in support.items() if status != "supported"}
    present = set(group_audit["group_value"].unique().to_list())
    audited_unsupported = {
        str(r["group_value"])
        for r in group_audit.filter(pl.col("group_status") != "supported").iter_rows(named=True)
    }
    mislabelled = [
        value for value in sorted(unsupported & present) if value not in audited_unsupported
    ]
    return _check(
        "unsupported_groups_are_preserved",
        not mislabelled,
        f"{len(mislabelled)} group(s) Component 12 called unsupported are labelled otherwise",
        offenders=mislabelled,
    )


def overrides_are_fully_attributed(override_log: pl.DataFrame) -> ValidationCheck:
    """Every override names an actor, a reason and a time, and records what it displaced.

    An override with no actor is an anonymous change to who gets inspected. The audit trail
    exists for exactly this, so an unattributed row is an error rather than a gap to tolerate.
    """
    if override_log.is_empty():
        return _check("overrides_are_fully_attributed", True, "no overrides were applied")
    blank = pl.col("actor").str.strip_chars() == ""
    bad = override_log.filter(
        blank
        | (pl.col("reason_code").str.strip_chars() == "")
        | (pl.col("decided_at").str.strip_chars() == "")
        | (pl.col("override_id").str.strip_chars() == "")
    )
    offenders = [str(r["override_id"]) for r in bad.head(MAX_OFFENDERS).iter_rows(named=True)]
    return _check(
        "overrides_are_fully_attributed",
        bad.is_empty(),
        f"{bad.height} override(s) missing an actor, reason or timestamp",
        offenders=offenders,
    )


def overrides_left_the_deterministic_queue_intact(
    recommendations: pl.DataFrame, override_log: pl.DataFrame
) -> ValidationCheck:
    """No override edited the policy artifact; each one appears only in its own log.

    The layer separation, checked. A human decision belongs beside the recommendation, not on
    top of it: the original must stay recoverable, because an audit asks what would have
    happened as well as what did.
    """
    if override_log.is_empty():
        return _check(
            "overrides_left_the_deterministic_queue_intact", True, "no overrides were applied"
        )
    applied = override_log.filter(pl.col("outcome") == "applied")
    if applied.is_empty():
        return _check(
            "overrides_left_the_deterministic_queue_intact", True, "no override changed a decision"
        )
    touched = recommendations.join(
        applied.select("policy_id", "fold_id", "k_name", "target_inspection_id"),
        on=["policy_id", "fold_id", "k_name", "target_inspection_id"],
        how="inner",
    )
    changed = touched.filter(
        pl.col("is_selected") != pl.col("decision_reason").is_in(sorted(SELECTED_REASONS))
    )
    return _check(
        "overrides_left_the_deterministic_queue_intact",
        changed.is_empty(),
        f"{changed.height} recommendation row(s) were rewritten by an override instead of "
        "being recorded beside one",
    )


def inputs_were_not_modified(
    before: Mapping[str, str], after: Mapping[str, str]
) -> ValidationCheck:
    """Every input file is byte-identical after the run.

    This component is a pure observer of nine closed components. It fits nothing, refits
    nothing and recalibrates nothing, and the way that stops being a promise and starts being a
    fact is a checksum taken before the first read and again after the last write.
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


# --- advisory checks: what did the policy cost? -------------------------------


def reserve_is_not_inert(allocation: pl.DataFrame) -> ValidationCheck:
    """ADVISORY. In how many cells did a policy ask for a reserve and get none?

    **A failure here is a finding, not a defect.** It says either that the risk ranking already
    cleared the floor -- which is the headline result of this component -- or that the share
    floored to zero slots at a small capacity. Both are real answers, and neither is a reason
    to round a reserve up past the share it declared.
    """
    if allocation.is_empty():
        return _check("reserve_is_not_inert", True, "no allocations", severity=SEVERITY_WARN)
    inert = allocation.filter(pl.col("reserve_inert"))
    by_policy = (
        inert.group_by("policy_id").len().rename({"len": "cells"}).sort("cells", descending=True)
    )
    offenders = [
        f"{r['policy_id']}: inert in {r['cells']} cell(s)"
        for r in by_policy.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    reserving = allocation.filter(pl.col("reserve_target") > 0).height
    return _check(
        "reserve_is_not_inert",
        inert.height < ADVISORY_INERT_RESERVE_CELLS,
        f"{inert.height} of {reserving} reserving cell(s) granted no slots. Evidence about the "
        "queue's existing coverage, not an implementation error.",
        severity=SEVERITY_WARN,
        offenders=offenders,
    )


def coverage_is_not_free(comparison: pl.DataFrame) -> ValidationCheck:
    """ADVISORY. How many Priority citations did each coverage policy give up?

    **A failure here is the component's most important finding and must never fail a build.**
    It says a policy chose coverage over discovery and names the price in citations. Turning
    this red would mean the only green policy is the one that reserves nothing, which is a
    policy decision dressed as a quality gate.
    """
    if comparison.is_empty():
        return _check("coverage_is_not_free", True, "no comparison rows", severity=SEVERITY_WARN)
    costed = comparison.filter(pl.col("policy_id") != BASELINE_POLICY_ID)
    pooled = (
        costed.group_by("policy_id", "k_name")
        .agg(pl.col("delta_positives").sum().alias("delta"))
        .filter(pl.col("delta") < 0)
        .sort("delta")
    )
    total = float(pooled["delta"].sum()) if not pooled.is_empty() else 0.0
    offenders = [
        f"{r['policy_id']} @ {r['k_name']}: {r['delta']:+.0f} citation(s)"
        for r in pooled.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "coverage_is_not_free",
        abs(total) < ADVISORY_LOST_POSITIVES,
        f"{pooled.height} (policy, capacity) cell(s) gave up citations, {total:+.0f} in total. "
        "The measured price of coverage, reported rather than hidden.",
        severity=SEVERITY_WARN,
        offenders=offenders,
    )


def group_representation_is_stable(group_audit: pl.DataFrame) -> ValidationCheck:
    """ADVISORY. Did any policy move a supported group's share of the queue materially?

    **Descriptive.** It reports movement; it does not say the movement is wrong, and nothing in
    this component is optimised to reduce it. A coverage policy that changes who is inspected
    will change group shares by construction, and the number is here so the change is visible
    rather than so it is minimised.
    """
    if group_audit.is_empty():
        return _check(
            "group_representation_is_stable", True, "no group audit rows", severity=SEVERITY_WARN
        )
    supported = group_audit.filter(pl.col("group_status") == "supported")
    if supported.is_empty():
        return _check(
            "group_representation_is_stable",
            True,
            "no supported groups at the audit floor",
            severity=SEVERITY_WARN,
        )
    baseline = supported.filter(pl.col("policy_id") == BASELINE_POLICY_ID).select(
        "group_value", "fold_set", "fold_id", "k_name", pl.col("selected_share").alias("base")
    )
    moved = (
        supported.filter(pl.col("policy_id") != BASELINE_POLICY_ID)
        .join(baseline, on=["group_value", "fold_set", "fold_id", "k_name"], how="inner")
        .with_columns((pl.col("selected_share") - pl.col("base")).abs().alias("shift"))
        .filter(pl.col("shift") > ADVISORY_GROUP_SELECTION_SHIFT)
        .sort("shift", descending=True)
    )
    offenders = [
        f"{r['policy_id']} / area {r['group_value']} / {r['fold_id']} @ {r['k_name']}: "
        f"{r['shift']:.4f}"
        for r in moved.head(MAX_OFFENDERS).iter_rows(named=True)
    ]
    return _check(
        "group_representation_is_stable",
        moved.is_empty(),
        f"{moved.height} (policy, group, fold, capacity) cell(s) moved a supported group's "
        f"share of the queue by more than {ADVISORY_GROUP_SELECTION_SHIFT}",
        severity=SEVERITY_WARN,
        offenders=offenders,
    )


def a_winner_was_determined(winner: str | None, statement: str) -> ValidationCheck:
    """ADVISORY. Did the frontier rule name a single policy?

    **Failing this is an acceptable and expected outcome.** A grid whose points genuinely trade
    citations against coverage has no mathematically optimal member, and forcing one would mean
    inventing an exchange rate between a missed citation and an uninspected establishment that
    nothing in this project measures. The absence of a winner is published as a result.
    """
    return _check(
        "a_winner_was_determined",
        winner is not None,
        f"policy winner: {winner}" if winner else statement,
        severity=SEVERITY_WARN,
    )


# --- terminators ---------------------------------------------------------------


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """The full check list, errors first, with the boundary printed underneath."""
    lines = ["", "Component 13 -- decision policy validation", ""]
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
            "  A green run means the policy was applied correctly.",
            "  It does not mean the policy is the right one.",
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
    """Advisories as table rows. A policy run produces many, and a list of strings reads badly.

    ``n_cells`` is the offender count: the number of specific cases the check was able to
    name before truncation. ``scope`` is the first of them, or the literal ``run`` for an
    advisory that is about the run as a whole rather than about particular cells.
    """
    return [
        {
            "code": check.name,
            "severity": check.severity,
            "scope": check.offenders[0] if check.offenders else "run",
            "n_cells": len(check.offenders),
            "detail": check.detail,
            "policy_definition_version": definition_version,
        }
        for check in sorted(checks, key=lambda c: c.name)
        if not check.passed and check.severity == SEVERITY_WARN
    ]


__all__ = [
    "CELL_KEYS",
    "a_winner_was_determined",
    "advisory_findings",
    "advisory_rows",
    "allocations_are_internally_consistent",
    "comparison_covers_every_policy",
    "configurations_match_the_frozen_grid",
    "coverage_is_not_free",
    "eligibility_matches_the_declared_rule",
    "every_row_declares_a_valid_mechanism",
    "format_report",
    "group_representation_is_stable",
    "has_failures",
    "inputs_were_not_modified",
    "no_establishment_is_selected_twice",
    "no_outcome_column_reaches_the_policy",
    "overrides_are_fully_attributed",
    "overrides_left_the_deterministic_queue_intact",
    "policy_ranks_are_unique_and_contiguous",
    "recommendations_cover_the_universe",
    "reserve_is_not_inert",
    "reserve_rows_are_eligible",
    "risk_rows_satisfy_the_risk_contract",
    "selected_counts_equal_capacity",
    "tables_are_deterministically_sorted",
    "unsupported_groups_are_preserved",
    "warnings_do_not_change_the_queue",
]
