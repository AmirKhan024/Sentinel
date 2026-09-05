"""Component 6: baseline risk models.

The first component that fits anything. It consumes Component 4's as-of feature table
and Component 5's rolling-origin folds, trains one model per fold, and emits scores in
Component 5's prediction contract. It does not evaluate them -- Component 5 does.

The single rule the whole package is designed around:

    a model scoring fold F may be fitted only on rows whose reference date lies in
    F's training window, and every preprocessing statistic must come from those
    same rows

That second clause is the half most easily lost. An imputation median or a scaler
mean computed over the whole table before splitting is leakage that no fold boundary
catches, because the boundary is respected by the *fit* while the *transform* already
knows the future.

Two consequences worth stating here, because both were measured rather than assumed:

* Training rows are sorted canonically before fitting. Float summation order in the
  scaler and in the lbfgs gradient reaches the coefficients (7.049e-09 on fold 1), so
  the sort is what makes a re-run reproducible, not the seed -- ``lbfgs`` has no
  stochastic component at all.
* Missingness is encoded as four *family* indicators, not ten per-column ones. The
  null masks within a Component 4 null-rule family are byte-identical, so per-column
  indicators would be exact duplicates and the coefficients would be unreadable.

See ``docs/analysis/baseline_models_findings.md`` for the measurements behind both,
``docs/data_contracts/baseline_predictions.md`` for the output contract, and ADR 0014
for where predictions live and why ``trained_through`` is the training end.
"""

from sentinel.modeling.definitions import (
    MODEL_DEFINITION_VERSION,
    MODEL_REGISTRY,
    MODELS_BY_NAME,
    ModelSpec,
)

__all__ = [
    "MODELS_BY_NAME",
    "MODEL_DEFINITION_VERSION",
    "MODEL_REGISTRY",
    "ModelSpec",
]
