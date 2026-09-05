"""Component 5: temporal evaluation.

Component 4 answered *can my features see the future?*. This package answers the
different question *can my evaluation see the future?* -- and they are not the
same. A feature table with a perfect temporal boundary still produces a
dishonest result if it is split randomly, if calibration touches the test
period, or if the number reported is a hit rate rather than a measure of whether
the *schedule* improved.

The single rule this package is designed around:

    **A model may be scored on a window only with information that existed
    before that window opened, and the calibration period must sit strictly
    between training and test.**

Sentinel is a ranking and scheduling problem, not a classification problem.
Every establishment is inspected on a statutory cycle; capacity is fixed; only
the *order* is decidable. So the harness measures two things side by side:
statistical prediction quality (PR-AUC, ROC-AUC, Brier, ECE) and operational
ranking quality (discovery curves, days-earlier detection, precision@k under
real measured capacity).

The estimand, stated before any result: this simulation re-orders the
establishments that were **actually inspected** in a historical window. There
are no labels for establishments nobody visited, so counterfactual coverage
cannot be evaluated and is never claimed.

The evaluator is model-agnostic by design. It consumes scores through
``contract.py`` and does not know or care whether they came from a heuristic,
a logistic regression, gradient boosting or an ensemble.
"""

from __future__ import annotations

from sentinel.evaluation.models import (
    EVALUATION_DEFINITION_VERSION,
    FoldSpec,
    PredictionSet,
    ScheduleName,
)

__all__ = [
    "EVALUATION_DEFINITION_VERSION",
    "FoldSpec",
    "PredictionSet",
    "ScheduleName",
]
