"""Component 17: operational candidate generation and as-of feature construction.

Components 1-16 answer questions about the past: "for this real inspection that
already happened, what did we know beforehand?" Every one of them is built around
a real inspection row -- Component 3's target table has one row per canvass that
occurred, and Component 4's as-of features are computed relative to that row's own
``inspection_date``.

Operational planning asks a different question: "for a planning date a supervisor
chose, which establishments should be considered, and what do we know about each of
them as of that date?" There is no future inspection row to hang that computation
off -- the whole point of a planning date is that nothing has happened yet.

This package is the bridge. It does not change what a feature means, does not
change the temporal boundary, and does not touch Components 1-16. It supplies a
second, synthetic source for the one thing Component 4's feature engine actually
needs from Component 3: a per-candidate reference date to join history against.
Everything downstream of that -- the join, the aggregation, the missing-value
rules, the validation -- is Component 4's code, imported and run unmodified.

See ``docs/analysis/`` for the architecture blueprint this component implements
(Component 17 of the operational-planning roadmap) and ``sentinel.features`` for
the feature engine this package reuses rather than duplicates.
"""

from __future__ import annotations
