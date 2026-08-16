"""Component 3: target construction.

Turns inspection history into a precise, leakage-safe prediction target: for each
establishment-date on which a routine canvass occurred, did that canvass find at
least one Priority or Priority Foundation violation?

See ``docs/analysis/target_construction_findings.md`` for the measurements the
definition rests on, and ``docs/data_contracts/inspection_targets.md`` for the
output contract and the leakage rules Component 4 must respect.
"""

from __future__ import annotations
