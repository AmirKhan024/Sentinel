"""Frozen contracts for Component 17.

Two definitions live here, and both are new -- neither has an analogue in
Components 1-16 -- so both are stated explicitly rather than left implicit in code.

**The candidate eligibility rule** is genuinely new. Component 3's eligibility
(``TargetStatus``) decides whether a real inspection that happened can be labelled.
Component 13's coverage eligibility (``policy.definitions.ELIGIBILITY_COLUMN``)
decides whether an establishment *already admitted* to a queue gets a coverage
reserve. Neither question is "does this establishment exist, as far as the data
can tell, on this hypothetical date" -- and that is the question Component 17 has
to answer before it can build a single feature, because Component 4's feature
engine has no notion of "this establishment" independent of a reference row.

**The synthetic identifier prefix** exists so a candidate's ``target_inspection_id``
can never collide with a real ``inspection_id``, checked directly in ``validate.py``
rather than assumed from the prefix alone.
"""

from __future__ import annotations

from enum import StrEnum

#: Bumped whenever the candidate eligibility rule, the synthetic id scheme, or the
#: as-of location convention changes. Two runs that disagree on this value did not
#: build the same candidate universe.
CANDIDATE_DEFINITION_VERSION = "v1"

#: Stated in the manifest so a consumer never has to infer it from the code. Deliberately
#: the same shape as ``features.build.TEMPORAL_BOUNDARY``: the boundary is the same rule,
#: applied against a planning date instead of a real inspection's own date.
CANDIDATE_TEMPORAL_BOUNDARY = "history.inspection_date < planning_date (strictly before)"

#: The eligibility rule, stated once so nobody has to reconstruct it from the query. An
#: establishment with zero prior records has not been observed to exist as of the
#: planning date -- admitting it as a candidate would mean assuming information (that it
#: exists at all) which is not actually available as of the boundary this component
#: exists to enforce. This is the smallest defensible rule: it excludes nothing else, and
#: in particular it does *not* reuse Component 13's coverage-eligibility predicate, which
#: answers a different question about establishments already admitted as candidates.
CANDIDATE_ELIGIBILITY_RULE = (
    "an establishment is an operational candidate for planning_date P if Component 2 "
    "resolved its identity and it has at least one real inspection record with "
    "inspection_date < P. Zero prior records means the establishment has not been "
    "observed to exist as of P, and is excluded -- not because it is judged safe or "
    "unsafe, but because nothing about it is knowable as of the boundary this "
    "component enforces."
)


class CandidateStatus(StrEnum):
    """The only status Component 17 emits into the ``target_status`` column.

    A distinct enum from Component 3's ``TargetStatus`` on purpose: a candidate row is
    never eligible, ineligible-era, ineligible-type or any of Component 3's real
    outcomes, because nothing has happened yet for it to be classified. Reusing that
    enum's ``ELIGIBLE`` member would make a candidate row indistinguishable from a real
    labelled one to any code that switches on ``target_status`` alone.
    """

    OPERATIONAL_CANDIDATE = "operational_candidate"


class CoverageWarning(StrEnum):
    """Machine-readable data-coverage warnings, distinct from validation checks.

    A validation check answers "is this candidate table internally correct". A coverage
    warning answers "does the requested planning date sit inside what the ingested data
    can actually speak to". Both matter, and conflating them would let a coverage
    problem hide inside a checks list nobody reads unless something already failed.
    """

    #: planning_date is later than the most recent ingested inspection record. Not a
    #: failure -- there is no future information involved, since only records strictly
    #: before planning_date are ever read -- but a candidate built this way reflects the
    #: state of the last ingest, not a live feed, and the caller must be told so
    #: explicitly rather than left to assume otherwise.
    PLANNING_DATE_BEYOND_INGESTED_DATA = "planning_date_beyond_ingested_data"

    #: The candidate universe is empty. Reported rather than silently returning a
    #: zero-row table with no explanation.
    NO_CANDIDATES_FOUND = "no_candidates_found"

    #: At least one candidate has no resolvable as-of location. Reported once at the
    #: table level so a caller does not have to scan every row to notice.
    CANDIDATES_MISSING_LOCATION = "candidates_missing_location"


#: Every synthetic candidate id starts with this token, which contains a character
#: (``:``) that never appears in a real Socrata ``inspection_id`` (those are decimal
#: digit strings). ``validate.py`` checks the stronger property directly -- no
#: candidate id equals a real inspection id -- rather than trusting the prefix alone.
SYNTHETIC_ID_PREFIX = "CANDIDATE::"


def synthetic_candidate_id(*, planning_date: str, establishment_id: str) -> str:
    """The deterministic id for one candidate.

    A pure function of its two inputs, which is what makes candidate generation
    reproducible: the same (planning_date, establishment_id) always yields the same
    id, on any machine, on any run, in any row order.
    """
    return f"{SYNTHETIC_ID_PREFIX}{planning_date}::{establishment_id}"


__all__ = [
    "CANDIDATE_DEFINITION_VERSION",
    "CANDIDATE_ELIGIBILITY_RULE",
    "CANDIDATE_TEMPORAL_BOUNDARY",
    "SYNTHETIC_ID_PREFIX",
    "CandidateStatus",
    "CoverageWarning",
    "synthetic_candidate_id",
]
