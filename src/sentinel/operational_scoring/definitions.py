"""Frozen contracts for Component 18.

Two things live here, and both exist to make an implicit choice explicit and checkable.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum

from sentinel.evaluation.folds import COVID_SHIFT, QUARTERLY

#: Bumped whenever the operational training-window rule, the calibrator-selection rule,
#: or the ranking/tie-break rule changes in a way that makes two runs incomparable.
OPERATIONAL_SCORING_DEFINITION_VERSION = "v1"

#: The ``fold_set`` every operational ``FoldSpec`` carries. Never a real evaluation fold
#: set -- checked below, at import time, rather than trusted -- so a downstream reader
#: (or a future contributor) can distinguish "Component 5 evaluated this" from "Component
#: 18 built this to reuse Components 6-8's fit functions" purely from the fold_set column.
OPERATIONAL_FOLD_SET = "operational"

if OPERATIONAL_FOLD_SET in (QUARTERLY, COVID_SHIFT):
    raise RuntimeError(
        "OPERATIONAL_FOLD_SET collides with a real evaluation fold set name; every "
        "downstream check that distinguishes historical from operational scoring by "
        "fold_set would silently stop working"
    )

#: The training-window rule, stated once. Mirrors the expanding-window convention
#: ``evaluation.folds`` already uses for real folds (train from the code-era anchor
#: forward), generalized to a live planning date instead of a quarter boundary.
OPERATIONAL_TRAINING_WINDOW_RULE = (
    "train_start = the code-era anchor (2018-07-01, evaluation.folds.CODE_ERA_ANCHOR); "
    "train_end = the day before planning_date, or the latest date the committed feature "
    "table actually has data through, whichever is earlier. No row dated on or after "
    "planning_date ever enters training."
)

#: Where the production model choice comes from, stated once so the manifest never has
#: to make a reader infer it. Reused verbatim in every manifest's ``model_selection_source``.
MODEL_SELECTION_SOURCE = (
    "sentinel.policy.select.select() -- Component 13's frozen, pre-registered "
    "model-selection rule, applied unmodified to the same Component 5/9 artifacts "
    "sentinel decide reads"
)

#: Where the applied calibrator comes from, stated once for the same reason.
CALIBRATION_SOURCE = (
    "sentinel.calibration persisted parameters (calibrator_parameters_*.parquet / "
    "calibrator_isotonic_breakpoints_*.parquet), loaded by "
    "operational_scoring.calibrator.load_frozen_calibrator -- the calibrator is never "
    "refit, only reloaded from Component 9's own frozen artifact"
)

#: Placeholder span for the synthetic calibration/test windows a `FoldSpec` requires
#: structurally but this component never reads. One day is the minimum span that keeps
#: `FoldSpec.__post_init__`'s ordering invariant satisfied without claiming any real
#: window.
PLACEHOLDER_WINDOW_SPAN = timedelta(days=1)


class ScoringStatus(StrEnum):
    """Why a candidate does, or does not, carry a score.

    Sentinel's models never abstain (mirrors ``policy.definitions.ABSTENTION_POLICY``),
    so in the ordinary case every candidate is ``SCORED``. The other members exist for
    the one failure mode that is structurally possible here: a candidate row that fails
    Component 18's own re-derived feature-contract check before the matrix is ever built.
    A candidate is excluded from scoring rather than the whole run failing, and the
    exclusion is named rather than silent.
    """

    SCORED = "scored"
    EXCLUDED_FEATURE_CONTRACT_VIOLATION = "excluded_feature_contract_violation"


__all__ = [
    "CALIBRATION_SOURCE",
    "MODEL_SELECTION_SOURCE",
    "OPERATIONAL_FOLD_SET",
    "OPERATIONAL_SCORING_DEFINITION_VERSION",
    "OPERATIONAL_TRAINING_WINDOW_RULE",
    "PLACEHOLDER_WINDOW_SPAN",
    "ScoringStatus",
]
