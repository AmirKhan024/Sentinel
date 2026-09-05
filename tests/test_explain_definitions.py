"""Component 11's frozen declarations, and the guard that refuses a dishonest one.

The registry is where this component states what it can and cannot do. Every test here
drives a *rejection*: a spec that claims support without a method, an approximation labelled
exact, an unsupported model with no reason. A guard whose failure path has never been
observed is indistinguishable from one that cannot fire, and Component 5 shipped exactly
that defect once (``scores_respect_the_decision_point``, declared and unreachable, fixed in
ADR 0014).
"""

from __future__ import annotations

import dataclasses

import pytest

from sentinel.calibration.definitions import CANDIDATE_REGISTRY, Family
from sentinel.explain import definitions as d
from sentinel.features.definitions import FEATURE_COLUMNS
from sentinel.modeling.definitions import FORBIDDEN_COLUMNS, indicator_columns

# --- 1. the support matrix ---------------------------------------------------


def test_every_calibration_candidate_has_an_explain_spec() -> None:
    """The two components speak about the same five models, so their tables line up."""
    assert {c.name for c in CANDIDATE_REGISTRY} == {s.name for s in d.EXPLAIN_REGISTRY}


def test_four_models_are_supported_and_the_embedding_booster_is_not() -> None:
    assert d.SUPPORTED_MODELS == (
        "logistic_regression",
        "xgboost",
        "lightgbm",
        "neural_numeric_only",
    )
    spec = d.spec_for("xgboost_chain_embeddings")
    assert spec.status is d.ExplanationStatus.UNSUPPORTED
    assert spec.method is None
    assert spec.output_space is None


def test_the_unsupported_reason_names_the_private_interface_and_the_proposed_fix() -> None:
    """The reason must be specific enough to act on, not "not supported"."""
    reason = d.spec_for("xgboost_chain_embeddings").unsupported_reason
    assert "_scorer_for" in reason
    assert "booster_for" in reason
    assert "ADR 0031" in reason


def test_each_family_gets_the_method_its_architecture_earns() -> None:
    methods = {s.name: s.method for s in d.EXPLAIN_REGISTRY if s.method}
    assert methods["logistic_regression"] is d.ExplanationMethod.LINEAR_SHAP
    assert methods["xgboost"] is d.ExplanationMethod.TREE_SHAP
    assert methods["lightgbm"] is d.ExplanationMethod.TREE_SHAP
    assert methods["neural_numeric_only"] is d.ExplanationMethod.PERMUTATION_SHAP


def test_only_the_permutation_method_is_labelled_approximate() -> None:
    """The one claim this component must not get wrong in the optimistic direction."""
    for spec in d.EXPLAIN_REGISTRY:
        if spec.method is d.ExplanationMethod.PERMUTATION_SHAP:
            assert spec.is_exact is False
        elif spec.method is not None:
            assert spec.is_exact is True


def test_the_name_source_differs_between_component_6_and_component_7() -> None:
    """The 19-of-30 permutation trap, recorded per model rather than inferred."""
    sources = {s.name: s.name_source for s in d.EXPLAIN_REGISTRY}
    assert sources["logistic_regression"].endswith("ordered_matrix_columns")
    assert sources["xgboost"].endswith("matrix_columns")
    assert sources["logistic_regression"] != sources["xgboost"]


# --- 2. every output space is declared ---------------------------------------


def test_log_odds_is_the_only_declared_output_space() -> None:
    """Probability space is rejected in prose, not declared and left unreachable."""
    assert [space.value for space in d.OutputSpace] == ["log_odds"]


def test_every_supported_model_shares_one_output_space() -> None:
    """What makes a cross-model importance comparison a comparison rather than a units error."""
    spaces = {s.output_space for s in d.EXPLAIN_REGISTRY if s.output_space}
    assert spaces == {d.OutputSpace.LOG_ODDS}


# --- 3. feature representation integrity -------------------------------------


def test_every_matrix_column_is_named_and_traceable() -> None:
    assert len(d.KNOWN_FEATURE_NAMES) == len(FEATURE_COLUMNS) + len(indicator_columns())
    for column in d.KNOWN_FEATURE_NAMES:
        original, derived = d.origin_of(column)
        assert original and derived


def test_a_component_4_feature_maps_to_itself() -> None:
    """Anything else would be an undeclared aggregation."""
    for column in FEATURE_COLUMNS:
        original, derived = d.origin_of(column)
        assert original == column
        assert derived == column
        assert d.kind_of(column) is d.FeatureKind.FEATURE


def test_a_family_indicator_lists_every_column_it_summarises() -> None:
    """An indicator belongs to no single column, and the artifact must not pretend it does."""
    for indicator in indicator_columns():
        original, derived = d.origin_of(indicator)
        assert original != indicator, "the family name, not the indicator's own name"
        members = derived.split(",")
        assert len(members) >= 1
        assert all(member in FEATURE_COLUMNS for member in members)
        assert d.kind_of(indicator) is d.FeatureKind.FAMILY_INDICATOR


def test_an_unknown_column_is_refused_rather_than_guessed() -> None:
    with pytest.raises(KeyError, match="not a known feature representation"):
        d.origin_of("feature_127")
    with pytest.raises(KeyError):
        d.kind_of("feature_127")


def test_no_forbidden_column_can_be_attributed() -> None:
    """Identity, the label and provenance are never model inputs, so never attributions."""
    assert not (d.KNOWN_FEATURE_NAMES & FORBIDDEN_COLUMNS)


# --- 4. the frozen budget ----------------------------------------------------


def test_every_method_has_a_positive_additivity_tolerance() -> None:
    for method in d.ExplanationMethod:
        assert d.tolerance_for(method) > 0


def test_the_tree_tolerance_is_looser_than_the_linear_one() -> None:
    """xgboost computes in float32 and the linear model in float64; measured, not assumed."""
    assert d.tolerance_for(d.ExplanationMethod.TREE_SHAP) > d.tolerance_for(
        d.ExplanationMethod.LINEAR_SHAP
    )


def test_the_sampling_seed_is_an_integer_constant_not_a_string_hash() -> None:
    """Python salts str hashing per process; Component 9 lost reproducibility to that once."""
    assert isinstance(d.SAMPLING_SEED, int)
    assert isinstance(d.BACKGROUND_SEED, int)


def test_the_background_strategy_names_the_training_window() -> None:
    assert "training window" in d.BACKGROUND_STRATEGY
    assert "train_end" in d.BACKGROUND_STRATEGY


def test_the_sample_strategy_says_no_label_participates() -> None:
    assert "no label" in d.SAMPLE_STRATEGY


def test_representative_quantiles_are_strictly_inside_the_unit_interval() -> None:
    assert set(d.REPRESENTATIVE_QUANTILES) == {"high", "medium", "low"}
    assert all(0.0 < q < 1.0 for q in d.REPRESENTATIVE_QUANTILES.values())


def test_model_selection_is_blocked_in_writing() -> None:
    """Component 11 is diagnostic evidence and must not be readable as a verdict."""
    assert "model selection" in d.BLOCKED_EXPERIMENTS
    assert "causal interpretation" in d.BLOCKED_EXPERIMENTS


# --- 5. the guard, driven ----------------------------------------------------
#
# Each of these builds a registry the guard must refuse. Without them the guard is a
# comment with an if-statement attached.


def _spec(**overrides: object) -> d.ExplanationSpec:
    base = dict(
        name="logistic_regression",
        family=Family.LOGISTIC,
        component=6,
        source_slug="baseline_predictions",
        status=d.ExplanationStatus.SUPPORTED,
        method=d.ExplanationMethod.LINEAR_SHAP,
        output_space=d.OutputSpace.LOG_ODDS,
        is_exact=True,
        name_source="modeling.preprocess.ordered_matrix_columns",
        rationale="fixture",
    )
    base.update(overrides)
    return d.ExplanationSpec(**base)  # type: ignore[arg-type]


def _guard_with(registry: tuple[d.ExplanationSpec, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d, "EXPLAIN_REGISTRY", registry)
    monkeypatch.setattr(
        d,
        "SUPPORTED_MODELS",
        tuple(s.name for s in registry if s.status is d.ExplanationStatus.SUPPORTED),
    )
    d._guard_registry()


def test_the_real_registry_passes_its_own_guard() -> None:
    """The positive control. Without it every rejection below could pass vacuously."""
    d._guard_registry()


def test_a_supported_spec_with_no_method_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="declares no explanation method"):
        _guard_with((_spec(method=None),), monkeypatch)


def test_a_supported_spec_with_no_output_space_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="declares no output space"):
        _guard_with((_spec(output_space=None),), monkeypatch)


def test_an_unsupported_spec_that_still_advertises_a_method_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported model must carry nothing resembling an explanation, including a promise."""
    with pytest.raises(ValueError, match="declares a method or an output space"):
        _guard_with(
            (_spec(status=d.ExplanationStatus.UNSUPPORTED, unsupported_reason="x"),), monkeypatch
        )


def test_an_unsupported_spec_with_no_reason_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unsupported with no stated reason"):
        _guard_with(
            (_spec(status=d.ExplanationStatus.UNSUPPORTED, method=None, output_space=None),),
            monkeypatch,
        )


def test_a_permutation_spec_labelled_exact_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single most damaging false claim available to this component."""
    with pytest.raises(ValueError, match="permutation sampling is never exact"):
        _guard_with(
            (_spec(method=d.ExplanationMethod.PERMUTATION_SHAP, is_exact=True),), monkeypatch
        )


def test_a_duplicate_model_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="duplicate model name"):
        _guard_with((_spec(), _spec()), monkeypatch)


def test_a_model_absent_from_component_9s_candidate_set_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="absent from Component 9's candidate set"):
        _guard_with((_spec(name="cdph_2015_approximation"),), monkeypatch)


def test_dropping_a_candidate_entirely_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every candidate is accounted for: supported, or unsupported with a reason."""
    with pytest.raises(ValueError, match="have no explain spec"):
        _guard_with((_spec(),), monkeypatch)


def test_a_registry_with_nothing_supported_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tuple(
        dataclasses.replace(
            spec,
            status=d.ExplanationStatus.UNSUPPORTED,
            method=None,
            output_space=None,
            unsupported_reason="fixture",
        )
        for spec in d.EXPLAIN_REGISTRY
    )
    with pytest.raises(ValueError, match="no model is supported"):
        _guard_with(registry, monkeypatch)
