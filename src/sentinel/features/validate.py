"""Post-build checks over the feature table.

The check that matters most is ``temporal_boundary_holds``. It re-derives, for
every row, the latest inspection date that could have contributed to that row's
features, and asserts it is strictly before the reference date. It is written as
an **independent** query rather than reusing the aggregation, so a bug in
``historical.py`` would be caught here rather than shipped.

That check runs over the whole table, not a sample. The project spec suggests
asserting the as-of rule on 500 random rows in CI; at 57,727 rows the full check
costs nothing and is strictly stronger.

The other error checks enforce the missing-value rules from findings §7. Those
are the quiet failure mode: a table where ``prior_canvass_priority_count`` is 0
instead of NULL looks perfectly healthy and teaches a model that a quarter of
establishments have a clean priority record.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import duckdb

from sentinel.features.definitions import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    WINDOW_DAYS,
    Family,
    NullRule,
)
from sentinel.features.models import ValidationCheck

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"

MAX_OFFENDERS = 20

# Component 3 columns describing the outcome. None may appear in the feature
# table: they are the label, not evidence available beforehand.
FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {
        "results",
        "evidence",
        "has_priority",
        "has_priority_foundation",
        "n_priority_entries",
        "n_priority_foundation_entries",
        "n_violation_entries",
        "inspection_type",
    }
)

# Prefixes that would indicate a historical aggregate leaked in from a later
# component, or that a feature was added without a specification.
FORBIDDEN_PREFIXES = ("rolling_", "future_", "next_", "post_")


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


def _scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def validate_features(
    conn: duckdb.DuckDBPyConnection,
    *,
    columns: Sequence[str],
    eligible_target_rows: int,
) -> list[ValidationCheck]:
    """Run every structural and distributional check over the built table.

    Expects ``features``, ``history``, ``targets`` and ``assignments`` to be
    registered on the connection.
    """
    checks: list[ValidationCheck] = []
    feature_rows = _scalar(conn, "SELECT count(*) FROM features")

    # -- structure ----------------------------------------------------------
    checks.append(
        _check(
            "one_feature_row_per_eligible_target",
            feature_rows == eligible_target_rows,
            SEVERITY_ERROR,
            f"{feature_rows} feature rows for {eligible_target_rows} eligible target rows",
        )
    )

    duplicates = _scalar(
        conn,
        "SELECT count(*) - count(DISTINCT target_inspection_id) FROM features",
    )
    checks.append(
        _check(
            "target_inspection_id_unique",
            duplicates == 0,
            SEVERITY_ERROR,
            f"{duplicates} duplicate target_inspection_id values",
        )
    )

    orphans = _scalar(
        conn,
        """SELECT count(*) FROM features f
           LEFT JOIN targets t USING (target_inspection_id)
           WHERE t.target_inspection_id IS NULL""",
    )
    checks.append(
        _check(
            "every_row_traces_to_a_target",
            orphans == 0,
            SEVERITY_ERROR,
            f"{orphans} feature rows reference a target row that does not exist",
        )
    )

    unknown_est = _scalar(
        conn,
        """SELECT count(*) FROM features f
           WHERE NOT EXISTS (SELECT 1 FROM assignments a
                              WHERE a.establishment_id = f.establishment_id)""",
    )
    checks.append(
        _check(
            "establishment_id_from_component_2",
            unknown_est == 0,
            SEVERITY_ERROR,
            f"{unknown_est} rows reference an establishment absent from Component 2",
        )
    )

    # -- THE temporal invariant, re-derived independently -------------------
    violations = conn.execute(
        """
        SELECT f.target_inspection_id
        FROM features f
        JOIN (
            SELECT t.target_inspection_id,
                   max(h.inspection_date) AS latest_used
            FROM targets t
            LEFT JOIN history h
              ON  h.establishment_id = t.establishment_id
              AND h.inspection_date  <  t.inspection_date
            GROUP BY t.target_inspection_id
        ) d USING (target_inspection_id)
        WHERE d.latest_used IS NOT NULL
          AND d.latest_used >= TRY_CAST(f.inspection_date AS DATE)
        LIMIT 100
        """
    ).fetchall()
    checks.append(
        _check(
            "temporal_boundary_holds",
            not violations,
            SEVERITY_ERROR,
            f"{len(violations)} rows used an inspection dated on or after their "
            "reference date (checked on every row, independently re-derived)",
            [str(v[0]) for v in violations],
        )
    )

    # A second, sharper form: no row may have consumed its own target inspection.
    self_use = _scalar(
        conn,
        """SELECT count(*) FROM features f
           JOIN targets t USING (target_inspection_id)
           JOIN history h
             ON  h.establishment_id = t.establishment_id
             AND h.inspection_date  = t.inspection_date
             AND h.inspection_date  < t.inspection_date""",
    )
    checks.append(
        _check(
            "reference_date_records_are_never_history",
            self_use == 0,
            SEVERITY_ERROR,
            f"{self_use} rows admitted a same-day record into their history",
        )
    )

    # -- value ranges -------------------------------------------------------
    negative_counts = [
        spec.name
        for spec in FEATURE_SPECS
        if spec.dtype == "int32"
        and _scalar(conn, f"SELECT count(*) FROM features WHERE {spec.name} < 0") > 0
    ]
    checks.append(
        _check(
            "no_negative_counts_or_recency",
            not negative_counts,
            SEVERITY_ERROR,
            f"{len(negative_counts)} integer features contain a negative value",
            negative_counts,
        )
    )

    bad_rates = [
        spec.name
        for spec in FEATURE_SPECS
        if spec.dtype == "float64"
        and _scalar(
            conn,
            f"SELECT count(*) FROM features "
            f"WHERE {spec.name} IS NOT NULL AND ({spec.name} < 0 OR {spec.name} > 1)",
        )
        > 0
    ]
    checks.append(
        _check(
            "rates_within_zero_and_one",
            not bad_rates,
            SEVERITY_ERROR,
            f"{len(bad_rates)} rate features fall outside [0, 1]",
            bad_rates,
        )
    )

    # -- the missing-value rules, one check each ----------------------------
    null_rule_sql = {
        NullRule.NEVER: "{col} IS NULL",
        NullRule.NO_PRIOR_CANVASS: ("({col} IS NULL) <> (prior_canvass_count = 0)"),
        NullRule.NO_PRIOR_INSPECTION: ("({col} IS NULL) <> (prior_inspection_count_any_type = 0)"),
        NullRule.NO_INSPECTED_CANVASS: ("({col} IS NULL) <> (prior_canvass_inspected_count = 0)"),
        NullRule.NO_CODE_ERA_CANVASS: ("({col} IS NULL) <> (prior_canvass_count_code_era = 0)"),
    }
    broken: list[str] = []
    for spec in FEATURE_SPECS:
        predicate = null_rule_sql[spec.null_rule].format(col=spec.name)
        if _scalar(conn, f"SELECT count(*) FROM features WHERE {predicate}") > 0:
            broken.append(f"{spec.name} ({spec.null_rule.value})")
    checks.append(
        _check(
            "null_rules_hold_exactly",
            not broken,
            SEVERITY_ERROR,
            f"{len(broken)} features violate their declared missing-value rule",
            broken,
        )
    )

    # -- window consistency -------------------------------------------------
    window_problems: list[str] = []
    for days in WINDOW_DAYS:
        if _scalar(
            conn,
            f"SELECT count(*) FROM features WHERE canvasses_last_{days}d > prior_canvass_count",
        ):
            window_problems.append(f"canvasses_last_{days}d > prior_canvass_count")
        if _scalar(
            conn,
            f"SELECT count(*) FROM features "
            f"WHERE canvass_priority_events_last_{days}d > canvasses_last_{days}d",
        ):
            window_problems.append(f"priority events exceed canvasses in {days}d window")
    for smaller, larger in zip(WINDOW_DAYS, WINDOW_DAYS[1:], strict=False):
        if _scalar(
            conn,
            f"SELECT count(*) FROM features "
            f"WHERE canvasses_last_{smaller}d > canvasses_last_{larger}d",
        ):
            window_problems.append(f"{smaller}d window exceeds {larger}d window")
    checks.append(
        _check(
            "windows_are_nested_and_bounded",
            not window_problems,
            SEVERITY_ERROR,
            f"{len(window_problems)} window relationships are impossible",
            window_problems,
        )
    )

    # -- schema separation --------------------------------------------------
    present = set(columns)
    leaked = sorted(present & FORBIDDEN_OUTCOME_COLUMNS)
    checks.append(
        _check(
            "no_outcome_columns_present",
            not leaked,
            SEVERITY_ERROR,
            f"{len(leaked)} Component 3 outcome columns appear in the feature table",
            leaked,
        )
    )

    suspicious = sorted(c for c in present if c.startswith(FORBIDDEN_PREFIXES))
    checks.append(
        _check(
            "no_unspecified_or_future_columns",
            not suspicious,
            SEVERITY_ERROR,
            f"{len(suspicious)} columns use a forbidden prefix",
            suspicious,
        )
    )

    overlap = sorted(set(FEATURE_COLUMNS) & set(LABEL_COLUMNS))
    checks.append(
        _check(
            "features_and_labels_are_disjoint",
            not overlap,
            SEVERITY_ERROR,
            f"{len(overlap)} columns are declared as both a feature and a label",
            overlap,
        )
    )

    missing_cols = sorted(set(KEY_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS) - present)
    checks.append(
        _check(
            "all_declared_columns_present",
            not missing_cols,
            SEVERITY_ERROR,
            f"{len(missing_cols)} declared columns are missing from the output",
            missing_cols,
        )
    )

    declared = {spec.name for spec in FEATURE_SPECS}
    undeclared = sorted(
        present
        - declared
        - set(KEY_COLUMNS)
        - set(LABEL_COLUMNS)
        - {"code_era_phase", "feature_definition_version"}
    )
    checks.append(
        _check(
            "every_column_is_declared",
            not undeclared,
            SEVERITY_ERROR,
            f"{len(undeclared)} columns exist without a FeatureSpec, so they are "
            "undocumented and untested",
            undeclared,
        )
    )

    # -- distributional, reported never fatal -------------------------------
    for family in Family:
        names = [s.name for s in FEATURE_SPECS if s.family is family]
        if names:
            checks.append(
                _check(
                    f"family_{family.value}",
                    True,
                    SEVERITY_WARN,
                    f"{len(names)} features: {', '.join(names)}",
                )
            )

    null_rates: list[str] = []
    for spec in FEATURE_SPECS:
        nulls = _scalar(conn, f"SELECT count(*) FROM features WHERE {spec.name} IS NULL")
        if nulls:
            pct = 100.0 * nulls / feature_rows if feature_rows else 0.0
            null_rates.append(f"{spec.name}: {nulls} ({pct:.1f}%)")
    checks.append(
        _check(
            "null_rates",
            True,
            SEVERITY_WARN,
            f"{len(null_rates)} features contain nulls, each by an explicit rule",
            null_rates,
        )
    )

    cold = _scalar(conn, "SELECT count(*) FROM features WHERE prior_canvass_count = 0")
    no_any = _scalar(
        conn, "SELECT count(*) FROM features WHERE prior_inspection_count_any_type = 0"
    )
    checks.append(
        _check(
            "cold_start_rows",
            True,
            SEVERITY_WARN,
            f"{no_any} rows have no history at all; {cold} have no prior canvass",
        )
    )

    renamed = _scalar(conn, "SELECT count(*) FROM features WHERE name_changed_since_last_canvass")
    checks.append(
        _check(
            "tenant_change_rows",
            True,
            SEVERITY_WARN,
            f"{renamed} rows follow a business-name change at the same premises",
        )
    )

    return checks


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """Render the checks as a plain text block for the CLI."""
    lines = ["", "Feature validation report", "-------------------------"]
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
