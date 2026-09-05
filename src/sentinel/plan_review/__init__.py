"""Component 21: supervisor plan review.

Reads Component 20's geographic work-block plan and, optionally, a batch of human plan
decisions, and produces the supervisor-facing summary a review screen renders. Never
edits a geographic, risk, or policy field; never creates a Component 13 override or a
Component 14 adjustment itself. See ``definitions.py`` for the full frozen contract.
"""

from __future__ import annotations
