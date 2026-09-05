"""Component 19: operational capacity and inspection plan selection.

Component 18 produces a complete, deterministic priority ranking over every
operational candidate. Nothing in it is bounded -- it answers "in what order would we
inspect these" for an unlimited inspection program, which no real program has.

This package answers the next question: given an explicit, honest inspection capacity
(never an invented inspector count -- see ``operational_selection.models
.OperationalCapacityRequest``), which establishments does this planning run actually
select?

**Nothing here is a new allocation algorithm.** ``sentinel.policy.allocation.allocate()``
and ``.decide()`` -- Component 13's frozen risk-block-plus-coverage-reserve engine --
are imported and called unmodified. Component 19 supplies them with a
``policy.models.PolicyWindow`` built from Component 18's real, scored output instead of
a historical fold's test window; the allocation and decision logic itself, and the
``DecisionMechanism``/``DecisionReason`` vocabulary it emits, are identical to what
``sentinel decide`` already uses.

The full Component 18 priority queue is never discarded: this package's output
preserves every scored row, selected or not, each carrying why.
"""

from __future__ import annotations
