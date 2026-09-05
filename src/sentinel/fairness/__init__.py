"""Component 12 -- fairness and geographic equity audit.

Measures whether Sentinel's held-out predictions, calibrated probabilities, ranking behaviour
and top-k prioritisation differ across the geographic groups this data can define.

It is an audit. It retrains nothing, recalibrates nothing, selects no model, modifies no
prediction and introduces no correction. It is the first component in this project that
re-executes nothing at all -- every input already exists on disk -- so there is no ADR 0026
bit-identity gate here, only a checksum gate proving the inputs were not touched.

**A green run means the audit is internally sound. It does not mean Sentinel is fair.**

See ADR 0032 (where the artifacts live), ADR 0033 (which groups are admissible and why ward
is refused), ADR 0034 (the support policy, and why a measured disparity is advisory rather
than an error) and ADR 0035 (what this component does not claim).
"""

from __future__ import annotations

__all__: list[str] = []
