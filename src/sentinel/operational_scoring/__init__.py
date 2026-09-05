"""Component 18: operational scoring, priority ranking, and the allocation bridge.

Component 17 answered "which establishments, and what do we know about them as of a
planning date." This package answers the next question: "what does Sentinel's own
validated model say about them, using exactly the model a deployment would actually
carry."

**The central finding this package is built around.** No fitted model is persisted
anywhere in this repository (ADR 0026) -- every ``train-*`` command refits from a
``FoldSpec`` each run, and Component 9's calibrators are the one exception: they are
persisted as *extracted parameters* (``calibrator_parameters_*.parquet``,
``calibrator_isotonic_breakpoints_*.parquet``), reproducible with plain arithmetic and
no estimator object. Component 17's own review reached this same conclusion.

Rather than introduce a foreign persistence mechanism (pickling an estimator, which
nothing else in the project does and which ADR 0026 deliberately avoids), this
component follows the repository's own established alternative: **re-execution from a
frozen, checkable specification**, the same pattern ``calibration.basescores`` and
``explain.refit`` already use to reproduce Components 6-8's fits bit-for-bit. The
"artifact" Component 18 introduces is not a blob -- it is a deterministic training
window (a synthetic, clearly-labelled ``FoldSpec`` that is never mistaken for an
evaluation fold) plus the frozen model-selection rule Component 13 already applies, so
that re-running the same planning date against the same committed data reproduces the
same scores exactly.

Calibration is reused as a true artifact, not re-fitted: Component 9's persisted
parameters are loaded directly and applied with ``calibration.predict.apply()``,
unmodified.
"""

from __future__ import annotations
