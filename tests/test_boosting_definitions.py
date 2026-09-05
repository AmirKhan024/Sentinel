"""The registry and the search space: what Component 7 is allowed to fit, and to search.

Two properties matter here and neither is checkable by reading the code.

**The registry cannot name something dangerous.** ``_guard_registry`` runs at import, so
a defective spec cannot be constructed and then used. A guard that has never been seen to
raise is indistinguishable from one that cannot, so each defect it rejects gets its own
test that drives it directly.

**The two searches are comparable.** The claim "XGBoost and LightGBM explored the same
space" is the load-bearing assumption behind every comparison between them. It is
checkable through ``SearchDimension.concept``, and the tests below check it rather than
trusting the prose in ``PARAM_MAPPING``.
"""

from __future__ import annotations

import pytest

from sentinel.boosting import definitions
from sentinel.boosting.definitions import (
    BOOSTING_DEFINITION_VERSION,
    BOOSTING_MODELS_BY_NAME,
    BOOSTING_REGISTRY,
    FIXED_PARAMS,
    PARAM_MAPPING,
    PARAMETER_DONOR,
    SEARCH_SPACE,
    TUNABLE_MODELS,
    BoostingSpec,
    Estimator,
    SearchDimension,
    estimator_params,
    n_estimators_of,
    spec_for,
    tuned_params,
)
from sentinel.features.definitions import FEATURE_COLUMNS
from sentinel.modeling.definitions import FORBIDDEN_COLUMNS

PRIMARY = BOOSTING_MODELS_BY_NAME["xgboost"]
SECONDARY = BOOSTING_MODELS_BY_NAME["lightgbm"]


# --- 1. the registry ---------------------------------------------------------


def test_the_registry_holds_the_three_declared_models() -> None:
    assert [s.name for s in BOOSTING_REGISTRY] == [
        "xgboost",
        "lightgbm",
        "xgboost_class_weighted",
    ]


def test_the_definition_version_is_pinned() -> None:
    """Bumping this is a contract change, so it should require editing a test."""
    assert BOOSTING_DEFINITION_VERSION == "v1"


def test_both_primary_models_use_the_full_component_4_contract() -> None:
    assert PRIMARY.feature_columns == FEATURE_COLUMNS
    assert SECONDARY.feature_columns == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 26


def test_the_two_primary_models_see_exactly_the_same_features() -> None:
    """Otherwise a difference between them would be a difference in their inputs."""
    assert PRIMARY.feature_columns == SECONDARY.feature_columns


def test_no_model_names_a_forbidden_column() -> None:
    for spec in BOOSTING_REGISTRY:
        assert not set(spec.feature_columns) & FORBIDDEN_COLUMNS


def test_only_the_ablation_declares_class_weighting() -> None:
    """Prevalence is 52.52%; weighting is an experiment, never a default."""
    weighted = [s.name for s in BOOSTING_REGISTRY if s.class_weighted]
    assert weighted == ["xgboost_class_weighted"]


def test_the_ablation_is_not_tunable_on_its_own() -> None:
    """It borrows its donor's parameters, so a separate search would vary two things."""
    assert "xgboost_class_weighted" not in TUNABLE_MODELS
    assert set(TUNABLE_MODELS) == {"xgboost", "lightgbm"}


def test_the_ablation_borrows_its_donors_parameters_exactly() -> None:
    ablation = spec_for("xgboost_class_weighted")
    for fold_set in ("quarterly", "covid_shift"):
        assert tuned_params(ablation, fold_set) == tuned_params(PRIMARY, fold_set)


def test_every_model_declares_a_probability() -> None:
    assert all(s.is_probability for s in BOOSTING_REGISTRY)


def test_spec_for_rejects_an_unknown_name_with_the_registered_ones_listed() -> None:
    with pytest.raises(KeyError, match="Unknown boosted model"):
        spec_for("catboost")


# --- 2. the import-time guard, driven directly ------------------------------


def _guard_with(registry: tuple[BoostingSpec, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(definitions, "BOOSTING_REGISTRY", registry)
    definitions._guard_registry()


def test_guard_raises_rather_than_asserts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``assert`` is stripped under ``python -O``; a guard that vanishes is not a guard."""
    from dataclasses import replace

    broken = (replace(PRIMARY, feature_columns=("target",)),)
    with pytest.raises(ValueError, match="non-feature column"):
        _guard_with(broken, monkeypatch)


def test_guard_rejects_a_duplicate_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="duplicate model name"):
        _guard_with((PRIMARY, PRIMARY), monkeypatch)


def test_guard_rejects_an_empty_feature_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="feature_columns is empty"):
        _guard_with((replace(PRIMARY, feature_columns=()),), monkeypatch)


def test_guard_rejects_an_unknown_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    broken = (replace(PRIMARY, feature_columns=(*FEATURE_COLUMNS, "inspector_id")),)
    with pytest.raises(ValueError, match="absent from Component"):
        _guard_with(broken, monkeypatch)


def test_guard_rejects_a_duplicated_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    doubled = (*FEATURE_COLUMNS, FEATURE_COLUMNS[0])
    with pytest.raises(ValueError, match="contains a duplicate"):
        _guard_with((replace(PRIMARY, feature_columns=doubled),), monkeypatch)


def test_guard_rejects_a_donor_that_is_not_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(definitions, "PARAMETER_DONOR", {"xgboost": "nonexistent"})
    with pytest.raises(ValueError, match="not a registered model"):
        definitions._guard_registry()


def test_guard_rejects_a_search_space_that_reaches_a_fixed_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tuned value must never overwrite a determinism or objective setting."""
    space = dict(SEARCH_SPACE)
    space[Estimator.XGBOOST] = (
        *SEARCH_SPACE[Estimator.XGBOOST],
        SearchDimension("tree_depth", "n_jobs", "int", 1, 8),
    )
    monkeypatch.setattr(definitions, "SEARCH_SPACE", space)
    with pytest.raises(ValueError, match="overwrite fixed parameter"):
        definitions._guard_registry()


def test_guard_rejects_searching_the_boosting_round_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``n_estimators`` comes from early stopping inside the objective, then is frozen."""
    space = dict(SEARCH_SPACE)
    space[Estimator.LIGHTGBM] = (
        *SEARCH_SPACE[Estimator.LIGHTGBM],
        SearchDimension("boosting_rounds", "n_estimators", "int", 10, 500),
    )
    monkeypatch.setattr(definitions, "SEARCH_SPACE", space)
    with pytest.raises(ValueError, match="must not be a searched dimension"):
        definitions._guard_registry()


def test_guard_rejects_an_undocumented_search_concept(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dimension absent from PARAM_MAPPING makes the two searches unprovable equals."""
    space = dict(SEARCH_SPACE)
    space[Estimator.XGBOOST] = (
        *SEARCH_SPACE[Estimator.XGBOOST],
        SearchDimension("undocumented_thing", "gamma", "float", 0.0, 1.0),
    )
    monkeypatch.setattr(definitions, "SEARCH_SPACE", space)
    with pytest.raises(ValueError, match="absent from PARAM_MAPPING"):
        definitions._guard_registry()


def test_a_search_dimension_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="exceeds high"):
        SearchDimension("tree_depth", "max_depth", "int", 10, 3)


def test_a_search_dimension_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="must be 'int' or 'float'"):
        SearchDimension("tree_depth", "max_depth", "categorical", 3, 10)


# --- 3. the two searches are comparable --------------------------------------


def test_every_shared_concept_is_tuned_in_both_libraries() -> None:
    """The claim behind every XGBoost-versus-LightGBM comparison, made checkable."""
    xgb = {d.concept for d in SEARCH_SPACE[Estimator.XGBOOST]}
    lgb = {d.concept for d in SEARCH_SPACE[Estimator.LIGHTGBM]}
    # ``leaf_count`` is LightGBM's alone: XGBoost's depth-wise growth implies it.
    assert lgb - xgb == {"leaf_count"}
    assert xgb - lgb == set()


def test_the_two_libraries_never_share_a_parameter_name_by_accident() -> None:
    """Where the names match, the semantics must match too, per PARAM_MAPPING."""
    xgb = {d.name for d in SEARCH_SPACE[Estimator.XGBOOST]}
    lgb = {d.name for d in SEARCH_SPACE[Estimator.LIGHTGBM]}
    shared = xgb & lgb
    assert shared == {"max_depth", "learning_rate"}
    for concept, xgb_name, lgb_name, _ in PARAM_MAPPING:
        if xgb_name is not None and xgb_name == lgb_name and xgb_name in shared:
            assert concept in {"tree_depth", "learning_rate"}


def test_the_shared_ranges_are_identical_where_the_concept_is_identical() -> None:
    """Different ranges would make a comparison a fact about the space, not the model."""
    by_concept = {
        estimator: {d.concept: d for d in dimensions}
        for estimator, dimensions in SEARCH_SPACE.items()
    }
    for concept in ("tree_depth", "learning_rate", "row_subsample", "column_subsample"):
        left = by_concept[Estimator.XGBOOST][concept]
        right = by_concept[Estimator.LIGHTGBM][concept]
        assert (left.low, left.high, left.log) == (right.low, right.high, right.log)


def test_param_mapping_documents_every_searched_dimension() -> None:
    documented = {row[0] for row in PARAM_MAPPING}
    for dimensions in SEARCH_SPACE.values():
        assert {d.concept for d in dimensions} <= documented


def test_param_mapping_names_match_the_search_space_names() -> None:
    """A mapping table that drifted from the space would document a search nobody ran."""
    index = {row[0]: row for row in PARAM_MAPPING}
    for estimator, dimensions in SEARCH_SPACE.items():
        position = 1 if estimator is Estimator.XGBOOST else 2
        for dimension in dimensions:
            assert index[dimension.concept][position] == dimension.name


def test_the_declared_ranges_match_the_specification() -> None:
    """The project specification's intended space, pinned so a silent widening fails."""
    by_name = {
        estimator: {d.name: d for d in dimensions} for estimator, dimensions in SEARCH_SPACE.items()
    }
    xgb = by_name[Estimator.XGBOOST]
    lgb = by_name[Estimator.LIGHTGBM]
    assert (xgb["max_depth"].low, xgb["max_depth"].high) == (3, 10)
    assert (xgb["learning_rate"].low, xgb["learning_rate"].high) == (0.01, 0.3)
    assert xgb["learning_rate"].log is True
    assert (xgb["subsample"].low, xgb["subsample"].high) == (0.6, 1.0)
    assert (xgb["colsample_bytree"].low, xgb["colsample_bytree"].high) == (0.6, 1.0)
    assert (lgb["bagging_fraction"].low, lgb["bagging_fraction"].high) == (0.6, 1.0)
    assert (lgb["feature_fraction"].low, lgb["feature_fraction"].high) == (0.6, 1.0)


# --- 4. fixed parameters and determinism -------------------------------------


def test_both_estimators_are_pinned_to_one_thread() -> None:
    """The determinism guarantee. A multi-threaded histogram reduction is not bit-stable."""
    assert FIXED_PARAMS[Estimator.XGBOOST]["n_jobs"] == 1
    assert FIXED_PARAMS[Estimator.LIGHTGBM]["n_jobs"] == 1
    assert FIXED_PARAMS[Estimator.LIGHTGBM]["num_threads"] == 1
    assert FIXED_PARAMS[Estimator.LIGHTGBM]["deterministic"] is True
    assert FIXED_PARAMS[Estimator.LIGHTGBM]["force_row_wise"] is True


def test_no_fixed_parameter_weights_the_classes() -> None:
    for params in FIXED_PARAMS.values():
        assert "scale_pos_weight" not in params
        assert "is_unbalance" not in params
        assert "class_weight" not in params


def test_xgboost_routes_nan_rather_than_imputing_it() -> None:
    import math

    missing = FIXED_PARAMS[Estimator.XGBOOST]["missing"]
    assert isinstance(missing, float) and math.isnan(missing)


def test_estimator_params_carry_every_seed_lightgbm_reads() -> None:
    """``random_state`` alone leaves bagging and feature sampling irreproducible."""
    params = estimator_params(SECONDARY, "quarterly")
    for key in ("random_state", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        assert params[key] == SECONDARY.seed


def test_estimator_params_merge_fixed_then_tuned_then_seed() -> None:
    params = estimator_params(PRIMARY, "quarterly")
    for key, value in FIXED_PARAMS[Estimator.XGBOOST].items():
        if key == "missing":
            continue
        assert params[key] == value
    for key, value in tuned_params(PRIMARY, "quarterly").items():
        assert params[key] == value


def test_each_fold_set_has_its_own_frozen_parameters() -> None:
    """Two studies means two entries; borrowing across fold sets is refused."""
    for name in TUNABLE_MODELS:
        spec = spec_for(name)
        assert tuned_params(spec, "quarterly")
        assert tuned_params(spec, "covid_shift")


def test_parameters_are_never_borrowed_across_fold_sets() -> None:
    with pytest.raises(KeyError, match="never borrowed across fold sets"):
        tuned_params(PRIMARY, "some_future_fold_set")


def test_every_frozen_parameter_set_declares_a_round_count() -> None:
    for name in BOOSTING_MODELS_BY_NAME:
        spec = spec_for(name)
        for fold_set in ("quarterly", "covid_shift"):
            assert n_estimators_of(spec, fold_set) >= 1


def test_the_donor_table_only_names_registered_models() -> None:
    registered = set(BOOSTING_MODELS_BY_NAME)
    for ablation, donor in PARAMETER_DONOR.items():
        assert ablation in registered
        assert donor in registered
