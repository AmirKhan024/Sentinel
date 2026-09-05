"""The Sentinel API: a validated read/write HTTP boundary over Components 1-16's artifacts.

Not a numbered component. It sits beside the 1-21 roadmap as cross-cutting infrastructure: a
product/backend boundary that lets a frontend retrieve and act on Sentinel's deterministic
outputs without reading internal Parquet files or internal Python modules directly.

**Nothing here computes.** Every non-trivial number this package returns was already written to
``data/processed/`` by a batch CLI command; this package resolves the right file, reads it,
validates the caller's decision scope, and serialises the result. See ADR 0048.

**Writes are staged, never applied.** ``POST`` endpoints validate a human-input request against
the same pydantic contracts ``sentinel.policy``/``sentinel.scheduling``/``sentinel.review``
already define, then append it to an append-only staging file this package owns. Turning a
staged request into a new artifact remains a manual step through the existing
``sentinel decide`` / ``sentinel schedule`` / ``sentinel review`` CLI commands. See ADR 0049.

**No routing.** The dataset has no inspector, no travel time and no road network (ADR 0019, ADR
0043), so there are no routing endpoints here, by omission.
"""

from __future__ import annotations
