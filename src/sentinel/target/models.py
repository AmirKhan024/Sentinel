"""Data structures for target construction.

Everything is frozen: the target is a pure function of one raw snapshot plus one
Component 2 assignment table, and immutability keeps that guarantee cheap.

The constants here are the target *definition*. Changing any of them changes
what the labels mean, which is why `TARGET_DEFINITION_VERSION` sits alongside
them and is stamped onto every output row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

# Bumped whenever the target definition changes: eligibility, the positive
# class, the parser's classification rules, or the same-day collapse. Two runs
# that disagree on this value did not build the same target, and the manifest
# records it so a model run can identify exactly which definition produced its
# labels.
TARGET_DEFINITION_VERSION = "v1"

# Chicago replaced Critical/Serious with Priority / Priority Foundation / Core on
# this date. Findings §5 measured the cutover as clean: June 2018 has 0 rows
# using the new terminology and 415 using the old; July 2018 has 761 and 0.
# Before this date the target is not merely sparse, it is undefined.
CODE_ERA_START = "2018-07-01"

# The end of the enforcement ramp-up window. Findings §15: the positive rate for
# 2018 H2 is 87.4% against 38.9% in 2026, and the boilerplate in the violation
# text shows a 90-day grace period was being applied. Flagged, not excluded.
ADOPTION_PHASE_END = "2019-01-01"

# Results that mean an inspection actually happened. The other four observed
# values (Out of Business, No Entry, Not Ready, Business Not Located) mean the
# inspector could not or did not inspect: 97.5-100% of them have null
# `violations` (findings §9).
INSPECTED_RESULTS = frozenset({"Pass", "Pass w/ Conditions", "Fail"})

# Results where a *passing* outcome makes empty violation text meaningful.
# `Pass` with no violations recorded is a true zero; `Fail` or
# `Pass w/ Conditions` with no violations recorded is self-contradictory
# (findings §10).
CLEAN_PASS_RESULT = "Pass"

# The eligible inspection type, after uppercasing and whitespace collapse.
# Findings §4: in the code era this matches 70,518 rows and is identical to the
# literal 'Canvass', but normalizing first guards against the casing variants
# that exist elsewhere in the column.
CANVASS_TYPE = "CANVASS"


class Severity(StrEnum):
    """How a single violation entry is classified by the inspector.

    Deliberately has no ``CORE`` member. Findings §7.2: 72% of entries carry no
    severity label at all and the word ``CORE`` is written only 7,943 times, so
    an unlabelled entry is absence of evidence, not evidence of Core. Calling it
    ``UNCLASSIFIED`` keeps that honest.
    """

    PRIORITY_FOUNDATION = "priority_foundation"
    PRIORITY = "priority"
    UNCLASSIFIED = "unclassified"

    @property
    def is_priority(self) -> bool:
        """Whether this severity counts toward a positive target."""
        return self in (Severity.PRIORITY, Severity.PRIORITY_FOUNDATION)


class TargetStatus(StrEnum):
    """Why a row does or does not carry a label.

    Every inspection in the snapshot receives one of these, so an exclusion is
    always countable rather than a silently missing row.
    """

    ELIGIBLE = "eligible"
    INELIGIBLE_ERA = "ineligible_era"
    INELIGIBLE_TYPE = "ineligible_type"
    INELIGIBLE_RESULT = "ineligible_result"
    UNKNOWN_VIOLATIONS = "unknown_violations"


class CodeEraPhase(StrEnum):
    """Which regulatory phase a target row falls in."""

    PRE_CODE = "pre_code"
    ADOPTION = "adoption"
    STABLE = "stable"


@dataclass(frozen=True, slots=True)
class ViolationEntry:
    """One ``|``-separated violation record, parsed.

    ``evidence`` is the exact text span that drove the classification, kept so a
    positive label can be traced to the words that produced it rather than to
    "the parser said so".
    """

    number: int | None
    title: str
    comments: str
    severity: Severity
    evidence: str | None = None

    @property
    def is_priority(self) -> bool:
        return self.severity.is_priority


@dataclass(frozen=True, slots=True)
class InspectionOutcome:
    """The parsed outcome of a single inspection record."""

    inspection_id: str
    establishment_id: str
    inspection_date: str
    inspection_type: str
    results: str
    status: TargetStatus
    label: bool | None
    entries: tuple[ViolationEntry, ...] = ()
    evidence: str | None = None

    @property
    def n_priority(self) -> int:
        return sum(1 for e in self.entries if e.severity is Severity.PRIORITY)

    @property
    def n_priority_foundation(self) -> int:
        return sum(1 for e in self.entries if e.severity is Severity.PRIORITY_FOUNDATION)


@dataclass(frozen=True, slots=True)
class TargetRow:
    """One prediction opportunity: an establishment on a date it was canvassed.

    Findings §11 and §13: the unit is (establishment, date) rather than a single
    inspection, because "inspect establishment E on date D" is one scheduling
    decision. 582 establishment-dates carry more than one eligible canvass and
    160 of those disagree, so the collapse rule is load-bearing rather than
    cosmetic.
    """

    establishment_id: str
    inspection_date: str
    target_inspection_id: str
    inspection_type: str
    results: str
    target: int | None
    target_status: TargetStatus
    has_priority: bool
    has_priority_foundation: bool
    n_priority_entries: int
    n_priority_foundation_entries: int
    n_violation_entries: int
    evidence: str | None
    n_contributing_inspections: int
    contributing_inspection_ids: str
    code_era_phase: CodeEraPhase
    target_definition_version: str = TARGET_DEFINITION_VERSION


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-construction assertion about the target table."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass
class TargetStats:
    """Counts accumulated during a build, surfaced in the manifest and CLI."""

    source_rows: int = 0
    target_rows: int = 0
    eligible_rows: int = 0
    positive_rows: int = 0
    negative_rows: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    positive_rate_by_year: dict[str, float] = field(default_factory=dict)
    collapsed_multi_canvass_days: int = 0
    narrative_exclusions: int = 0


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class TargetManifest(BaseModel):
    """Self-contained provenance and QA record for one target build.

    Pins *both* inputs by checksum. The labels are a function of the raw snapshot
    and the Component 2 assignments together, so recording only one would leave
    the output unreproducible.
    """

    component: str = "target_construction"
    code_version: str
    target_definition_version: str
    built_at: str

    source_path: str
    source_sha256: str
    source_row_count: int
    assignments_path: str
    assignments_sha256: str

    code_era_start: str
    inspected_results: list[str]
    canvass_type: str

    target_rows: int
    eligible_rows: int
    positive_rows: int
    negative_rows: int
    positive_rate: float
    status_counts: dict[str, int]
    positive_rate_by_year: dict[str, float]
    collapsed_multi_canvass_days: int

    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
