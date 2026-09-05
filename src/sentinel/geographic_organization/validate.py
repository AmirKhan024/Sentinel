"""Post-organization checks. Every one independently re-derived, not trusted.

The first check in this module is the most important one in Component 20: the
selected-establishment-ID set must be byte-identical before and after geographic
organization. Everything else here is secondary to that invariant.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.features.models import ValidationCheck
from sentinel.geographic_organization.definitions import UNMAPPED_GROUP_ID

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

#: The Component 18/19 fields Component 20 must carry through unchanged. Reused
#: directly by ``check_risk_and_policy_fields_unchanged`` rather than restated.
IMMUTABLE_FIELDS: tuple[str, ...] = (
    "calibrated_score",
    "base_score",
    "rank",
    "policy_rank",
    "selection_reason",
    "selection_mechanism",
)


def _check(name: str, passed: bool, severity: str, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=passed, severity=severity, detail=detail)


def check_selected_ids_unchanged(
    selection_frame: pl.DataFrame, plan_frame: pl.DataFrame
) -> ValidationCheck:
    """The one invariant this component exists to protect.

    ``selection_frame`` is Component 19's ``is_selected == True`` subset;
    ``plan_frame`` is Component 20's output. Geography must never add, drop, or
    substitute an establishment.
    """
    selected_ids = set(selection_frame["target_inspection_id"].to_list())
    plan_ids = set(plan_frame["target_inspection_id"].to_list())
    passed = selected_ids == plan_ids
    return _check(
        "selected_ids_unchanged_by_geography",
        passed,
        SEVERITY_ERROR,
        "Component 19's and Component 20's selected-id sets are identical"
        if passed
        else f"{len(selected_ids - plan_ids)} selected id(s) missing from the geographic "
        f"plan, {len(plan_ids - selected_ids)} extra id(s) present -- geography altered "
        "who was selected, which must never happen",
    )


def check_risk_and_policy_fields_unchanged(
    selection_frame: pl.DataFrame, plan_frame: pl.DataFrame
) -> ValidationCheck:
    """Component 20 may add geographic columns; it may never rewrite a risk/policy one."""
    left = selection_frame.select(["target_inspection_id", *IMMUTABLE_FIELDS]).sort(
        "target_inspection_id"
    )
    right = plan_frame.select(["target_inspection_id", *IMMUTABLE_FIELDS]).sort(
        "target_inspection_id"
    )
    passed = left.equals(right)
    return _check(
        "risk_and_policy_fields_unchanged",
        passed,
        SEVERITY_ERROR,
        f"{', '.join(IMMUTABLE_FIELDS)} are byte-identical to Component 19's output"
        if passed
        else "Component 20 altered a risk or policy field -- this must never happen",
    )


def check_no_fabricated_coordinates(
    selection_frame: pl.DataFrame, plan_frame: pl.DataFrame
) -> ValidationCheck:
    """Coordinates in the plan must be exactly Component 19's, never invented or altered."""
    left = selection_frame.select(
        ["target_inspection_id", "as_of_latitude", "as_of_longitude"]
    ).sort("target_inspection_id")
    right = plan_frame.select(["target_inspection_id", "as_of_latitude", "as_of_longitude"]).sort(
        "target_inspection_id"
    )
    passed = left.equals(right)
    return _check(
        "coordinates_are_never_fabricated_or_altered",
        passed,
        SEVERITY_ERROR,
        "as_of_latitude/as_of_longitude are byte-identical to Component 19's output"
        if passed
        else "a coordinate value changed between Component 19 and Component 20",
    )


def check_no_duplicate_group_membership(plan_frame: pl.DataFrame) -> ValidationCheck:
    """No canonical establishment may appear in more than one geographic group."""
    duplicates = plan_frame.height - plan_frame["target_inspection_id"].n_unique()
    return _check(
        "no_duplicate_group_membership",
        duplicates == 0,
        SEVERITY_ERROR,
        f"{duplicates} establishment(s) appear more than once in the geographic plan",
    )


def check_location_coverage_counts(
    plan_frame: pl.DataFrame,
    *,
    location_available_count: int,
    location_unavailable_count: int,
) -> ValidationCheck:
    """Independently re-derive the coverage counts rather than trust the summary."""
    from sentinel.geographic_organization.definitions import LocationStatus

    real_available = int(
        plan_frame.filter(
            pl.col("location_status") == LocationStatus.LOCATION_AVAILABLE.value
        ).height
    )
    real_unavailable = int(
        plan_frame.filter(
            pl.col("location_status") == LocationStatus.LOCATION_UNAVAILABLE.value
        ).height
    )
    passed = (
        real_available == location_available_count
        and real_unavailable == location_unavailable_count
    )
    return _check(
        "location_coverage_counts_match_the_plan",
        passed,
        SEVERITY_ERROR,
        f"available={real_available}, unavailable={real_unavailable}"
        if passed
        else f"summary claims available={location_available_count}, "
        f"unavailable={location_unavailable_count}, but the plan itself has "
        f"available={real_available}, unavailable={real_unavailable}",
    )


def check_unmapped_establishments_are_preserved(
    selection_frame: pl.DataFrame, plan_frame: pl.DataFrame
) -> ValidationCheck:
    """Every establishment without coordinates is still present, in the unmapped group."""
    unmapped_in_selection = set(
        selection_frame.filter(
            pl.col("as_of_latitude").is_null() | pl.col("as_of_longitude").is_null()
        )["target_inspection_id"].to_list()
    )
    unmapped_in_plan = set(
        plan_frame.filter(pl.col("geographic_group_id") == UNMAPPED_GROUP_ID)[
            "target_inspection_id"
        ].to_list()
    )
    passed = unmapped_in_selection == unmapped_in_plan
    return _check(
        "unmapped_establishments_preserved_not_dropped",
        passed,
        SEVERITY_ERROR,
        f"{len(unmapped_in_selection)} location-unavailable establishment(s) are all "
        "present in the unmapped group"
        if passed
        else "an establishment without coordinates was dropped, or a mapped "
        "establishment was placed in the unmapped group",
    )


def check_group_ids_ordered_by_centroid_latitude(plan_frame: pl.DataFrame) -> ValidationCheck:
    """Determinism sanity check: mapped groups are numbered northernmost first."""
    mapped = plan_frame.filter(pl.col("geographic_group_id") != UNMAPPED_GROUP_ID)
    if mapped.is_empty():
        return _check(
            "group_ids_ordered_by_centroid_latitude", True, SEVERITY_WARN, "no mapped groups"
        )
    centroids = (
        mapped.group_by("geographic_group_id")
        .agg(pl.col("as_of_latitude").mean().alias("centroid_lat"))
        .sort("geographic_group_id")
    )
    ids_in_declared_order = centroids["geographic_group_id"].to_list()
    lats_in_declared_order = centroids["centroid_lat"].to_list()
    expected_order = sorted(
        range(len(ids_in_declared_order)), key=lambda i: -lats_in_declared_order[i]
    )
    actual_order = sorted(
        range(len(ids_in_declared_order)),
        key=lambda i: int(ids_in_declared_order[i].rsplit("_", 1)[-1]),
    )
    passed = expected_order == actual_order
    return _check(
        "group_ids_ordered_by_centroid_latitude",
        passed,
        SEVERITY_ERROR,
        "group numbering runs northernmost (highest centroid latitude) first"
        if passed
        else "group numbering does not follow the declared centroid-latitude-descending rule",
    )


def check_suggested_order_is_permutation_of_block(plan_frame: pl.DataFrame) -> ValidationCheck:
    """Within every mapped work block, the suggested order covers each member exactly once.

    The "unmapped" pseudo-block has no geography to order by, so it carries no suggested
    order by design (never a fabricated one) and is excluded from this check.
    """
    mapped = plan_frame.filter(pl.col("work_block_id") != UNMAPPED_GROUP_ID)
    bad_blocks: list[str] = []
    for block_id, group in mapped.group_by("work_block_id"):
        bid = block_id[0] if isinstance(block_id, tuple) else block_id
        orders = sorted(group["suggested_order_in_block"].to_list())
        expected = list(range(1, group.height + 1))
        if orders != expected:
            bad_blocks.append(str(bid))
    passed = not bad_blocks
    return _check(
        "suggested_order_is_permutation_of_block",
        passed,
        SEVERITY_ERROR,
        "every work block's suggested order is a 1..N permutation of its members"
        if passed
        else f"work block(s) with an invalid suggested order: {', '.join(bad_blocks)}",
    )


def check_risk_first_never_reorders(plan_frame: pl.DataFrame) -> ValidationCheck:
    """When organization_mode is risk_first, suggested order must equal policy_rank order.

    The mechanical proof of the brief's requirement that geography must not casually
    reorder the highest-risk cases in the conservative default mode.
    """
    risk_first = plan_frame.filter(pl.col("organization_mode") == "risk_first")
    if risk_first.is_empty():
        return _check(
            "risk_first_never_reorders", True, SEVERITY_WARN, "no risk_first rows in this plan"
        )
    bad_blocks: list[str] = []
    for block_id, group in risk_first.group_by("work_block_id"):
        bid = block_id[0] if isinstance(block_id, tuple) else block_id
        ordered = group.sort("suggested_order_in_block")
        ranks = ordered["policy_rank"].to_list()
        non_null = [r for r in ranks if r is not None]
        if non_null != sorted(non_null):
            bad_blocks.append(str(bid))
    passed = not bad_blocks
    return _check(
        "risk_first_never_reorders",
        passed,
        SEVERITY_ERROR,
        "risk_first suggested order matches policy_rank ascending in every block"
        if passed
        else f"risk_first reordered a block relative to policy_rank: {', '.join(bad_blocks)}",
    )


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


def format_report(checks: Sequence[ValidationCheck]) -> str:
    lines = [
        "",
        "Geographic organization validation report",
        "------------------------------------------",
    ]
    for check in checks:
        status = "note" if check.severity == SEVERITY_WARN else ("PASS" if check.passed else "FAIL")
        lines.append(f"  [{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def run_all_checks(
    selection_frame: pl.DataFrame,
    plan_frame: pl.DataFrame,
    *,
    location_available_count: int,
    location_unavailable_count: int,
) -> list[ValidationCheck]:
    return [
        check_selected_ids_unchanged(selection_frame, plan_frame),
        check_risk_and_policy_fields_unchanged(selection_frame, plan_frame),
        check_no_fabricated_coordinates(selection_frame, plan_frame),
        check_no_duplicate_group_membership(plan_frame),
        check_location_coverage_counts(
            plan_frame,
            location_available_count=location_available_count,
            location_unavailable_count=location_unavailable_count,
        ),
        check_unmapped_establishments_are_preserved(selection_frame, plan_frame),
        check_group_ids_ordered_by_centroid_latitude(plan_frame),
        check_suggested_order_is_permutation_of_block(plan_frame),
        check_risk_first_never_reorders(plan_frame),
    ]


__all__ = [
    "IMMUTABLE_FIELDS",
    "check_group_ids_ordered_by_centroid_latitude",
    "check_location_coverage_counts",
    "check_no_duplicate_group_membership",
    "check_no_fabricated_coordinates",
    "check_risk_and_policy_fields_unchanged",
    "check_risk_first_never_reorders",
    "check_selected_ids_unchanged",
    "check_suggested_order_is_permutation_of_block",
    "check_unmapped_establishments_are_preserved",
    "format_report",
    "has_failures",
    "run_all_checks",
]
