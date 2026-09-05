"""Component 8: a neural network with entity embeddings, as an experiment.

Three invariants, and they are the reason this package exists in the shape it does.

**The estimator changes; the problem does not.** Same target, same folds, same
evaluation contract, same ``target_inspection_id`` set. The model that carries the
comparison against Components 6 and 7 -- ``neural_numeric_only`` -- sees exactly the 30
matrix columns they see, so a difference between it and XGBoost is a difference in
estimator and nothing else.

**The categoricals are experimental, and quarantined.** Component 4's table has no
categorical column. The four embedded families come from Component 8's own as-of join,
written to its own layer, and ``feature_definition_version`` stays ``v1``. See ADR 0022.

**Nothing later than ``fold.train_end`` is read by any final fit.** Vocabularies, scaling
statistics, chain membership, the early-stopping signal and the learning rate are all
derived from training rows only. The early-stopping validation split is carved from the
*end of the training window*, not from the fold's calibration window, which is what keeps
``trained_through = fold.train_end`` literally true. See ADR 0021.
"""

from sentinel.neural.definitions import (
    NEURAL_DEFINITION_VERSION,
    NEURAL_REGISTRY,
    SPECS_BY_NAME,
    NeuralSpec,
)

__all__ = [
    "NEURAL_DEFINITION_VERSION",
    "NEURAL_REGISTRY",
    "SPECS_BY_NAME",
    "NeuralSpec",
]
