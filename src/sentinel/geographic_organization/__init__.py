"""Component 20: geographically aware inspection plan organization.

Component 19 answers: "Given a limited inspection capacity, which establishments are
selected?"

This package answers the next question: "Now that we know which establishments are
selected, how are they geographically distributed, and how can they be organized into
coherent geographic proximity groups for a supervisor?"

ARCHITECTURAL BOUNDARY
-----------------------
This package is strictly *downstream* of selection.

    COMPONENT 18 — ML scores + calibrated risk ranking
            ↓
    COMPONENT 19 — Capacity + policy selection  (WHO gets inspected)
            ↓
    COMPONENT 20 — Real location data + geographic proximity grouping
                   (HOW the selected work can be spatially organized)
            ↓
    COMPONENT 21 — Human review / approval

NON-NEGOTIABLE INVARIANT
------------------------
SELECTED_IDS_COMPONENT_19 == SELECTED_IDS_COMPONENT_20

Location organizes the inspection plan.
Location never changes it.

A geographically isolated high-priority establishment must remain in the plan.
It must not be removed because a nearby lower-priority group is more convenient.

NON-GOALS
----------
This package does NOT implement:
- Route optimization
- Travel-time estimation
- Driving directions
- Inspector assignment
- Inspector start locations
- Working-hour scheduling
- Inspection duration modeling

Those belong to future components.
"""

from __future__ import annotations
