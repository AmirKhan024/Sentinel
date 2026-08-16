"""Post-resolution checks over the finished output.

Two severities, and the distinction is deliberate.

``error`` checks assert things that cannot be true of a correct run: an
inspection that lost its establishment, a cluster spanning two postal codes, an
anchor row that is not a member of its own cluster. These fail the run.

``warn`` checks surface things that are suspicious but legitimate. Findings §10
measured 219 distinct business names at one O'Hare address and 194 licences at
one shared kitchen; a check that failed on those would fail on correct data.
They are reported so a human looks at them, not so the pipeline stops.

The goal is that entity resolution is loud when it is wrong, rather than quietly
producing plausible-looking identities.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence

from sentinel.entity.models import (
    Cluster,
    MatchTier,
    Node,
    PairVerdict,
    Thresholds,
    ValidationCheck,
)

logger = logging.getLogger(__name__)

ESTABLISHMENT_ID_PATTERN = re.compile(r"^EST-\d{11}$")

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

# How many offending identifiers to keep on a failed check. Enough to start
# debugging, few enough that the manifest stays readable.
MAX_OFFENDERS = 20


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


def validate_output(
    nodes: Sequence[Node],
    clusters: Sequence[Cluster],
    verdicts: Sequence[PairVerdict],
    assignment: dict[str, str],
    thresholds: Thresholds,
    *,
    source_row_count: int,
) -> list[ValidationCheck]:
    """Run every structural and distributional check over a completed run."""
    checks: list[ValidationCheck] = []
    by_node = {n.node_id: n for n in nodes}
    node_to_cluster: dict[str, str] = {}
    duplicated: list[str] = []
    for cluster in clusters:
        for node_id in cluster.node_ids:
            if node_id in node_to_cluster:
                duplicated.append(node_id)
            node_to_cluster[node_id] = cluster.establishment_id

    # -- structural ---------------------------------------------------------
    checks.append(
        _check(
            "every_inspection_assigned",
            len(assignment) == source_row_count,
            SEVERITY_ERROR,
            f"{len(assignment)} assignments for {source_row_count} raw rows",
        )
    )

    checks.append(
        _check(
            "node_in_exactly_one_cluster",
            not duplicated,
            SEVERITY_ERROR,
            f"{len(duplicated)} nodes appear in more than one establishment",
            duplicated,
        )
    )

    unassigned = [n.node_id for n in nodes if n.node_id not in node_to_cluster]
    checks.append(
        _check(
            "every_node_clustered",
            not unassigned,
            SEVERITY_ERROR,
            f"{len(unassigned)} nodes were never placed in an establishment",
            unassigned,
        )
    )

    bad_ids = [
        c.establishment_id
        for c in clusters
        if not ESTABLISHMENT_ID_PATTERN.match(c.establishment_id)
    ]
    checks.append(
        _check(
            "establishment_id_format",
            not bad_ids,
            SEVERITY_ERROR,
            f"{len(bad_ids)} establishment ids do not match EST-<11 digits>",
            bad_ids,
        )
    )

    counts = Counter(c.establishment_id for c in clusters)
    repeated = sorted(est for est, n in counts.items() if n > 1)
    checks.append(
        _check(
            "establishment_id_unique",
            not repeated,
            SEVERITY_ERROR,
            f"{len(repeated)} establishment ids are used by more than one cluster",
            repeated,
        )
    )

    anchor_problems = [
        c.establishment_id
        for c in clusters
        if f"EST-{min(by_node[n].min_inspection_id for n in c.node_ids):011d}" != c.establishment_id
    ]
    checks.append(
        _check(
            "anchor_is_a_member",
            not anchor_problems,
            SEVERITY_ERROR,
            f"{len(anchor_problems)} establishments are not anchored on their earliest inspection",
            anchor_problems,
        )
    )

    # -- semantic -----------------------------------------------------------
    zip_violations: list[str] = []
    address_violations: list[str] = []
    unit_violations: list[str] = []
    for cluster in clusters:
        members = [by_node[n] for n in cluster.node_ids]
        if len({m.address.zip_key for m in members if m.address.zip_key}) > (
            thresholds.max_zips_per_cluster
        ):
            zip_violations.append(cluster.establishment_id)
        if len({m.address.key for m in members if m.address.key}) > (
            thresholds.max_addresses_per_cluster
        ):
            address_violations.append(cluster.establishment_id)
        if len({m.address.unit for m in members if m.address.unit}) > 1:
            unit_violations.append(cluster.establishment_id)

    checks.append(
        _check(
            "cluster_within_one_zip",
            not zip_violations,
            SEVERITY_ERROR,
            f"{len(zip_violations)} establishments span more than "
            f"{thresholds.max_zips_per_cluster} zip code(s)",
            zip_violations,
        )
    )
    checks.append(
        _check(
            "cluster_address_count",
            not address_violations,
            SEVERITY_ERROR,
            f"{len(address_violations)} establishments span more than "
            f"{thresholds.max_addresses_per_cluster} addresses",
            address_violations,
        )
    )
    checks.append(
        _check(
            "cluster_units_consistent",
            not unit_violations,
            SEVERITY_ERROR,
            f"{len(unit_violations)} establishments contain conflicting unit designators",
            unit_violations,
        )
    )

    # -- distributional (reported, never fatal) -----------------------------
    sizes = sorted(((len(c.node_ids), c.establishment_id) for c in clusters), reverse=True)
    largest = [f"{est}({n} nodes)" for n, est in sizes[:MAX_OFFENDERS]]
    checks.append(
        _check(
            "largest_establishments",
            True,
            SEVERITY_WARN,
            f"largest establishment holds {sizes[0][0] if sizes else 0} nodes",
            largest,
        )
    )

    singletons = sum(1 for c in clusters if len(c.node_ids) == 1)
    pct = 100.0 * singletons / len(clusters) if clusters else 0.0
    checks.append(
        _check(
            "singleton_rate",
            True,
            SEVERITY_WARN,
            f"{singletons} of {len(clusters)} establishments hold a single node ({pct:.1f}%)",
        )
    )

    ambiguous = [v for v in verdicts if v.tier is MatchTier.AMBIGUOUS]
    checks.append(
        _check(
            "ambiguous_pairs",
            True,
            SEVERITY_WARN,
            f"{len(ambiguous)} candidate pairs were declined as ambiguous and need review",
        )
    )

    reduced = [c.establishment_id for c in clusters if c.split_reason is not None]
    checks.append(
        _check(
            "split_establishments",
            True,
            SEVERITY_WARN,
            f"{len(reduced)} establishments were split after failing a cluster invariant",
            reduced,
        )
    )

    return checks


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """Render the checks as a plain text block for the CLI."""
    lines = ["", "Validation report", "-----------------"]
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
