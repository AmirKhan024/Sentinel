"""Post-build checks specific to Component 17.

Every check that already exists for a real feature table -- the temporal
boundary, the missing-value rules, the window-consistency rules, the schema
separation from Component 3's outcome columns -- is reused verbatim from
``sentinel.features.validate.validate_features``. Nothing about those checks
changes when the ``targets`` view is synthetic instead of real: they read
``targets``, ``history``, ``assignments`` and ``features`` by name, and do not
care how ``targets`` was produced.

This module adds only the two invariants that are genuinely new to Component 17:
that a synthetic candidate id can never collide with a real inspection id (the
leakage risk unique to inventing an identifier), and that every candidate really
does have the prior history its own eligibility rule requires (the eligibility
rule, re-derived independently rather than trusted).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import duckdb

from sentinel.candidates.definitions import SYNTHETIC_ID_PREFIX
from sentinel.candidates.models import ValidationCheck

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

MAX_OFFENDERS = 20


def _check(
    name: str, passed: bool, severity: str, detail: str, offenders: Sequence[str] = ()
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail,
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


def validate_candidate_universe(conn: duckdb.DuckDBPyConnection) -> list[ValidationCheck]:
    """The two invariants unique to a synthetic ``targets`` view.

    Expects ``raw``, ``assignments``, ``targets`` and ``history`` registered on
    ``conn``, exactly as ``candidates.features.compute_operational_features``
    leaves them.
    """
    checks: list[ValidationCheck] = []

    collisions = conn.execute(
        """
        SELECT t.target_inspection_id
        FROM targets t
        JOIN raw r ON r.inspection_id = t.target_inspection_id
        LIMIT 20
        """
    ).fetchall()
    checks.append(
        _check(
            "candidate_ids_never_collide_with_real_inspection_ids",
            not collisions,
            SEVERITY_ERROR,
            f"{len(collisions)} synthetic candidate id(s) match a real inspection_id. A "
            "collision here would let a candidate row be joined against real historical "
            "data as if it were that inspection.",
            [str(c[0]) for c in collisions],
        )
    )

    non_prefixed = conn.execute(
        f"""
        SELECT count(*) FROM targets
        WHERE target_inspection_id NOT LIKE '{SYNTHETIC_ID_PREFIX}%'
        """
    ).fetchone()
    non_prefixed_count = int(non_prefixed[0]) if non_prefixed else 0
    checks.append(
        _check(
            "candidate_ids_carry_the_synthetic_prefix",
            non_prefixed_count == 0,
            SEVERITY_ERROR,
            f"{non_prefixed_count} candidate id(s) lack the synthetic prefix and would be "
            "indistinguishable from a real inspection_id by inspection alone.",
        )
    )

    no_history = conn.execute(
        """
        SELECT count(*) FROM targets t
        WHERE NOT EXISTS (
            SELECT 1 FROM history h
            WHERE h.establishment_id = t.establishment_id
              AND h.inspection_date < t.inspection_date
        )
        """
    ).fetchone()
    no_history_count = int(no_history[0]) if no_history else 0
    checks.append(
        _check(
            "every_candidate_has_at_least_one_prior_record",
            no_history_count == 0,
            SEVERITY_ERROR,
            f"{no_history_count} candidate(s) have no record strictly before the planning "
            "date, violating this component's own eligibility rule (re-derived "
            "independently of the universe query that is supposed to enforce it).",
        )
    )

    return checks


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """Render the checks as a plain text block for the CLI."""
    lines = ["", "Candidate validation report", "----------------------------"]
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


__all__ = ["format_report", "has_failures", "validate_candidate_universe"]
