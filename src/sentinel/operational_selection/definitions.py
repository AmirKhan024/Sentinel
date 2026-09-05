"""Frozen contracts for Component 19."""

from __future__ import annotations

from sentinel.policy.definitions import BASELINE_POLICY_ID

#: Bumped whenever the capacity contract, the window-construction rule, or the
#: selection-reason vocabulary changes in a way that makes two runs incomparable.
OPERATIONAL_SELECTION_DEFINITION_VERSION = "v1"

#: The policy a request gets when it does not name one: the grid's own baseline,
#: plain top-k by calibrated risk with no coverage reserve. Never silently defaulted to
#: a reserve-bearing policy -- a caller who wants a reserve must ask for one by id.
DEFAULT_POLICY_ID = BASELINE_POLICY_ID

#: Stated once, reused in every manifest's ``allocation_source`` field.
ALLOCATION_SOURCE = (
    "sentinel.policy.allocation.allocate() / .decide() -- Component 13's frozen "
    "risk-block-plus-coverage-reserve engine, applied unmodified to a "
    "policy.models.PolicyWindow built from Component 18's scored output"
)

#: Two ``PolicyWindow`` fields (``labels``, ``median_daily_capacity``) exist only for
#: Component 13's historical use and are read by neither ``allocate()`` nor
#: ``decide()`` -- verified directly against ``policy/allocation.py``. Operational
#: candidates have no real label (nothing has happened yet) and no measured daily
#: rate (capacity here is a request, not an observation), so both are filled with this
#: explicit, self-describing sentinel rather than a fabricated 0 that could be misread
#: as real data.
UNKNOWN_LABEL = -1
NOT_APPLICABLE_MEDIAN_DAILY_CAPACITY = 0

#: Why a candidate never reached allocation at all -- distinct from a policy/capacity
#: outcome, and named so a reader never has to guess whether "not selected" meant
#: "outranked" or "was never scoreable in the first place".
EXCLUDED_UNSCORABLE_REASON = "excluded_unscorable_by_component_18"


class CapacityError(ValueError):
    """Raised when a requested capacity is not a valid, honest planning input."""


__all__ = [
    "ALLOCATION_SOURCE",
    "DEFAULT_POLICY_ID",
    "EXCLUDED_UNSCORABLE_REASON",
    "NOT_APPLICABLE_MEDIAN_DAILY_CAPACITY",
    "OPERATIONAL_SELECTION_DEFINITION_VERSION",
    "UNKNOWN_LABEL",
    "CapacityError",
]
