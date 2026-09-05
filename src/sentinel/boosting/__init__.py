"""Component 7: gradient-boosted risk models (XGBoost and LightGBM).

Component 6 established that Component 4's feature representation carries linear
signal. This component asks one question and no others: **can a nonlinear model learn a
better risk ranking from the same 26 features, under the same Component 5 evaluation?**

Three invariants hold everything together.

**The estimator changes; nothing else does.** Folds come from ``evaluation.folds``,
"training window" means what ``modeling.train.training_frame`` says it means, the feature
list is Component 4's contract, and the metrics are Component 5's. This package fits
models and writes scores. It computes no metric, defines no fold and adds no feature.

**Hyperparameters are selected only from data earlier than any test window.** Component 6
had no hyperparameters, so it needed no protocol. A booster has a dozen, and choosing
them is a place where a human can leak the future without any artifact recording it: fit,
read a test number, adjust, refit. The protocol in ``tuning`` makes that structurally
impossible rather than merely discouraged -- each fold set gets its own study confined to
a region strictly earlier than its own first test start, and the resulting parameters are
frozen as constants before any test window is opened. See ADR 0017.

**A raw probability is not a calibrated one.** These models emit ``predict_proba``
output, and boosted probabilities are typically *worse* calibrated than a penalised
GLM's. Nothing here corrects that. Component 9 owns calibration; this package reports
ECE and MCE through Component 5 and leaves them uncorrected.
"""

from sentinel.boosting.definitions import (
    BOOSTING_DEFINITION_VERSION,
    BOOSTING_MODELS_BY_NAME,
    BOOSTING_REGISTRY,
    BoostingSpec,
    Estimator,
)

__all__ = [
    "BOOSTING_DEFINITION_VERSION",
    "BOOSTING_MODELS_BY_NAME",
    "BOOSTING_REGISTRY",
    "BoostingSpec",
    "Estimator",
]
