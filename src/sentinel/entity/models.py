"""Data structures for entity resolution.

Every structure here is frozen. Resolution is a pure function of the input
snapshot, and immutable values make that guarantee cheap to keep: nothing can
mutate a node halfway through clustering and change the outcome of a comparison
that already happened.

The threshold constants live in one place, `DEFAULT_THRESHOLDS`, so the numbers
that govern merge behaviour are greppable and citable. Each is justified by a
measurement in ``docs/analysis/entity_resolution_findings.md``; the section
reference is on the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

# Bumped whenever a normalization rule changes. Two runs that disagree on this
# value are not comparable, and the manifest records it so that a diff between
# two outputs can be attributed to code rather than to data.
NORMALIZATION_VERSION = "1"


class MatchTier(StrEnum):
    """How much evidence supported (or defeated) a candidate pair.

    ``AMBIGUOUS`` is a first-class outcome, not a failure. It means the pair was
    evaluated, the evidence was real but insufficient, and the resolver declined
    to merge. Those pairs are written to the audit table so a human can review
    them; forcing them into a match is how years of inspection history get
    silently fused onto the wrong establishment.
    """

    STRONG = "strong"
    PROBABLE = "probable"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


@dataclass(frozen=True, slots=True)
class NormalizedName:
    """A business name reduced to comparable form.

    ``digits`` is separated from ``tokens`` deliberately. Store numbers are the
    single most useful anti-merge signal in this dataset (``SUBWAY 1234`` and
    ``SUBWAY 5678`` are different franchise locations), so a digit disagreement
    has to be detectable independently of overall token overlap.
    """

    key: str | None
    tokens: frozenset[str] = frozenset()
    digits: frozenset[str] = frozenset()

    @property
    def usable(self) -> bool:
        return self.key is not None


@dataclass(frozen=True, slots=True)
class NormalizedAddress:
    """A street address split into the parts that matter for identity.

    ``key`` deliberately excludes the street suffix. Findings §7.1: the dataset
    has no long-form suffixes to canonicalize, but it does have addresses whose
    suffix is missing or wrong (``1901 W MADISON`` / ``AVE`` / ``ST`` are all the
    United Center). Excluding the suffix merges 133 such addresses with no
    observed false merges, because house number, directional, street name and
    zip together already pin the location in Chicago's grid.

    ``unit`` is kept rather than discarded: a suite disagreement is a veto.
    """

    key: str | None
    house_number: int | None = None
    street_body: str | None = None
    directional: str | None = None
    suffix: str | None = None
    unit: str | None = None
    zip_key: str | None = None
    flags: frozenset[str] = frozenset()

    @property
    def usable(self) -> bool:
        return self.key is not None


@dataclass(frozen=True, slots=True)
class Geo:
    """A coordinate pair, plus why it is or is not usable as evidence.

    ``key`` is the *raw* coordinate strings joined, not the parsed floats.
    Findings §8: coordinates are a deterministic function of the address string
    with exactly zero variance within an address, so exact string equality is
    both the right comparison and immune to float formatting drift.
    """

    key: str | None
    lat: float | None = None
    lon: float | None = None
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.key is not None


@dataclass(frozen=True, slots=True)
class Node:
    """A distinct identity signature observed in the snapshot.

    Matching operates on nodes rather than inspections. Many inspections share
    one signature, so this collapses the comparison space and makes the audit
    table readable: an edge between two nodes is a statement about two distinct
    ways the city recorded a place, not about two individual visits.
    """

    node_id: str
    license_key: str | None
    name: NormalizedName
    aka: NormalizedName
    address: NormalizedAddress
    geo: Geo
    facility_type_key: str | None
    inspection_ids: tuple[str, ...]
    min_inspection_id: int
    raw_name: str
    raw_address: str
    raw_zip: str | None

    @property
    def name_keys(self) -> frozenset[str]:
        """Every name this node can legitimately be matched on.

        Findings §5: ``dba_name`` is often the legal entity and ``aka_name`` the
        trade name (``1918 WINTER STREET ILLINOIS LLC`` / ``MARIANO'S``). Both
        are admitted, because a holding-company rename must not split a premises.
        """
        keys = {k for k in (self.name.key, self.aka.key) if k is not None}
        return frozenset(keys)

    @property
    def all_digits(self) -> frozenset[str]:
        return self.name.digits | self.aka.digits


@dataclass(frozen=True, slots=True)
class PairSignals:
    """Every comparison made between two nodes, recorded whatever the outcome.

    This is what makes a merge decision explainable after the fact: the audit
    table stores the whole signal vector, so "why were these two joined?" and
    "why were these two *not* joined?" are both answerable from the output alone.
    """

    same_license: bool = False
    same_addr_key: bool = False
    same_coord: bool = False
    near_addr: bool = False
    same_zip: bool = False
    name_exact: bool = False
    name_containment: bool = False
    digit_conflict: bool = False
    unit_conflict: bool = False
    dir_conflict: bool = False
    aka_conflict: bool = False
    facility_agree: bool | None = None

    @property
    def addr_equivalent(self) -> bool:
        """Whether the two nodes describe the same place.

        Two independent routes (findings §7.1, §8): an identical normalized
        address key, or an identical geocoded coordinate within one zip. The
        second catches variants no string rule can, such as Chicago's 2021
        rename of Lake Shore Drive.
        """
        return self.same_addr_key or self.same_coord


@dataclass(frozen=True, slots=True)
class PairVerdict:
    """The tier a pair reached and the single rule that decided it."""

    left_node_id: str
    right_node_id: str
    tier: MatchTier
    rule_id: str
    signals: PairSignals

    @property
    def merges(self) -> bool:
        return self.tier in (MatchTier.STRONG, MatchTier.PROBABLE)


@dataclass(frozen=True, slots=True)
class Cluster:
    """A resolved establishment: the set of nodes accepted as one premises."""

    establishment_id: str
    node_ids: tuple[str, ...]
    confidence: str
    split_reason: str | None
    content_sha256: str


@dataclass(frozen=True, slots=True)
class OversizedBlock:
    """A block too large to pair-expand, recorded rather than silently dropped."""

    block_kind: str
    block_key: str
    size: int


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Structural limits on resolution. Every value is measured, not chosen.

    There are no similarity thresholds here. Findings §4 measured that full name
    normalization resolves 0.21% of licences, which does not justify a fuzzy
    matching tier against the risk from 247 Subways (§4.2) and 219 distinct
    business names at one O'Hare address (§10). Matching is exact-after-
    normalization, so there is nothing to calibrate.
    """

    # §3.1: max distinct addresses per licence is 5, and most of that is
    # trailing-whitespace duplication that normalization removes.
    max_addresses_per_cluster: int = 4
    # §9: zip is single-valued for 99.75% of licences.
    max_zips_per_cluster: int = 1
    # §8: a coordinate covering more than 4 distinct address keys is a geocoder
    # artefact, not a place. The worst observed is exactly 4.
    max_addresses_per_coordinate: int = 4
    # §10: the largest legitimate block is O'Hare at 417 members. This bound
    # exists to stop a pathological future block, not to make identity
    # decisions, so it sits far above anything real.
    max_block_size: int = 5_000
    # §7: house-number typo tolerance, used only when a licence already agrees.
    near_house_number_delta: int = 2
    # §8: within-address coordinate spread is exactly 0 m, so any cluster
    # spanning real distance indicates a bug rather than a geographic fact.
    max_cluster_span_m: float = 2_000.0


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-resolution assertion about the output."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass
class ResolutionStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI."""

    rows: int = 0
    nodes: int = 0
    establishments: int = 0
    singleton_establishments: int = 0
    candidate_pairs: int = 0
    edges_by_tier: dict[str, int] = field(default_factory=dict)
    edges_by_rule: dict[str, int] = field(default_factory=dict)
    splits_by_reason: dict[str, int] = field(default_factory=dict)
    oversized_blocks: int = 0
    blacklisted_coordinates: int = 0
    unusable_license_rows: int = 0
    unusable_address_rows: int = 0


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class ResolutionManifest(BaseModel):
    """Self-contained provenance and QA record for one resolution run.

    Pinning ``source_sha256`` is what makes the output reproducible: establishment
    IDs are a deterministic function of a specific input file, and this records
    which one.
    """

    component: str = "entity_resolution"
    code_version: str
    normalization_version: str
    resolved_at: str

    source_path: str
    source_sha256: str
    source_row_count: int

    thresholds: dict[str, float]

    node_count: int
    establishment_count: int
    singleton_establishment_count: int
    candidate_pair_count: int
    edges_by_tier: dict[str, int]
    edges_by_rule: dict[str, int]
    splits_by_reason: dict[str, int]
    oversized_block_count: int
    blacklisted_coordinate_count: int
    unusable_license_rows: int
    unusable_address_rows: int

    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
