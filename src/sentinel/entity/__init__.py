"""Component 2: entity resolution.

Maps each inspection to a stable ``establishment_id`` representing a physical
food-service premises, together with the evidence that justifies every merge and
every declined merge.

See ``docs/analysis/entity_resolution_findings.md`` for the measurements the
design rests on, and ``docs/data_contracts/establishment_assignments.md`` for
the output contract.
"""

from __future__ import annotations
