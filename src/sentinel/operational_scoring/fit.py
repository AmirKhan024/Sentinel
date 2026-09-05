"""Re-executing the frozen training pipeline for one base model, on the operational window.

Every function this module calls is imported unmodified from Components 6, 7 or 8:
same registry spec, same seed, same frozen hyperparameters, same
``modeling.train.training_frame`` -- the repository's one definition of "train", shared
by all three families already. Nothing is tuned here and no feature is added; this is a
re-execution exactly in the sense ``explain.refit`` uses the word, aimed at a different
window.

Family dispatch reuses ``calibration.definitions.CANDIDATE_REGISTRY`` -- the same
registry Component 9 itself dispatches on -- rather than inferring a model's family from
its name.
"""

from __future__ import annotations

import polars as pl

from sentinel.calibration.definitions import CANDIDATE_REGISTRY, Family
from sentinel.evaluation.folds import QUARTERLY
from sentinel.evaluation.models import FoldSpec
from sentinel.modeling.train import training_frame
from sentinel.operational_scoring.models import HyperparameterProvenance

#: Which fold set's frozen tuned hyperparameters an operational fit borrows. Component
#: 18's own ``FoldSpec`` uses ``fold_set="operational"``, which no tuning study was ever
#: run for and never will be -- there is no "operational test window" a search could be
#: confined ahead of (ADR 0017). ``quarterly`` is not arbitrary: it is Sentinel's
#: production fold cadence and the one ``policy.definitions.SELECTION_FOLD_SET`` itself
#: reads to choose the production model, so operational scoring borrows hyperparameters
#: tuned for exactly the cadence its model choice is already anchored to.
TUNING_FOLD_SET = QUARTERLY


class OperationalFitError(RuntimeError):
    """Raised when a base model cannot be re-executed or cannot score the candidates."""


def family_of(base_model_name: str) -> Family:
    """The family Component 9 already declared for this base model."""
    for spec in CANDIDATE_REGISTRY:
        if spec.name == base_model_name:
            return spec.family
    raise OperationalFitError(
        f"{base_model_name!r} is not in Component 9's candidate registry, so its family "
        "-- and therefore how to fit and score it -- is unknown"
    )


def fit_and_score(
    *,
    base_model_name: str,
    family: Family,
    historical_features: pl.DataFrame,
    fold: FoldSpec,
    candidates: pl.DataFrame,
) -> tuple[list[str], list[float], HyperparameterProvenance]:
    """Fit the named base model on ``fold``'s training window, then score ``candidates``.

    ``candidates`` must carry ``target_inspection_id`` and every declared Component 4
    feature column -- exactly what Component 17 produces and what
    ``modeling/boosting/neural``'s ``score_window`` already require; no adapter is
    written because none is needed.

    Returns the scored ids, the scores, and exactly where the hyperparameters used to
    produce them came from -- not left implicit in this module's docstring, per the
    Component 18 provenance patch.
    """
    train = training_frame(historical_features, fold)
    if train.height == 0:
        raise OperationalFitError(
            f"{base_model_name}: operational fold {fold.fold_id} has no training rows"
        )

    if family is Family.LOGISTIC:
        from sentinel.modeling.definitions import spec_for as modeling_spec_for
        from sentinel.modeling.predict import score_window as modeling_score_window
        from sentinel.modeling.train import fit_fold as modeling_fit_fold

        modeling_spec = modeling_spec_for(base_model_name)
        modeling_fitted = modeling_fit_fold(modeling_spec, train, fold)
        ids, scores = modeling_score_window(modeling_fitted, candidates)
        provenance = HyperparameterProvenance(
            fold_set=None,
            source=(
                "sentinel.modeling.definitions.spec_for"
                f"({base_model_name!r}).params -- fixed, not tuned per fold set: "
                "Component 6 (logistic regression) has no hyperparameter search stage"
            ),
            values={k: str(v) for k, v in modeling_spec.params.items()},
        )
        return ids, scores, provenance

    if family is Family.BOOSTED:
        from sentinel.boosting.definitions import estimator_params
        from sentinel.boosting.definitions import spec_for as boosting_spec_for
        from sentinel.boosting.predict import score_window as boosting_score_window
        from sentinel.boosting.train import fit_fold as boosting_fit_fold

        boosting_spec = boosting_spec_for(base_model_name)
        boosting_fitted = boosting_fit_fold(
            boosting_spec, train, fold, params_fold_set=TUNING_FOLD_SET
        )
        ids, scores = boosting_score_window(boosting_fitted, candidates)
        provenance = HyperparameterProvenance(
            fold_set=TUNING_FOLD_SET,
            source=(
                "sentinel.boosting.definitions.TUNED_PARAMS"
                f"[{boosting_spec.estimator.value!r}][{TUNING_FOLD_SET!r}] -- "
                f"Component 7's frozen Optuna search result for fold set {TUNING_FOLD_SET!r} "
                "(ADR 0017), reused because no separate operational tuning study exists"
            ),
            values={k: str(v) for k, v in estimator_params(boosting_spec, TUNING_FOLD_SET).items()},
        )
        return ids, scores, provenance

    if family is Family.NEURAL_MLP:
        from sentinel.neural.definitions import learning_rate_for
        from sentinel.neural.definitions import spec_for as neural_spec_for
        from sentinel.neural.predict import score_window as neural_score_window
        from sentinel.neural.train import fit_fold as neural_fit_fold

        neural_spec = neural_spec_for(base_model_name)
        rate = learning_rate_for(TUNING_FOLD_SET)
        neural_fitted = neural_fit_fold(neural_spec, train, fold, learning_rate=rate)
        ids, scores = neural_score_window(neural_fitted, candidates)
        provenance = HyperparameterProvenance(
            fold_set=TUNING_FOLD_SET,
            source=(
                f"sentinel.neural.definitions.learning_rate_for({TUNING_FOLD_SET!r}) -- "
                f"Component 8's frozen learning-rate search result for fold set "
                f"{TUNING_FOLD_SET!r} (ADR 0017), reused because no separate operational "
                "tuning study exists"
            ),
            values={"learning_rate": str(rate)},
        )
        return ids, scores, provenance

    raise OperationalFitError(
        f"{base_model_name}: family {family} has no operational scoring path. "
        "NEURAL_EMBEDDING_BOOSTER in particular is Component 8's experimental, "
        "explanation-unsupported derivative (ADR 0022, ADR 0031) and is excluded from "
        "Component 13's admissible candidates, so this component should never be asked "
        "to fit one"
    )


__all__ = ["OperationalFitError", "family_of", "fit_and_score"]
