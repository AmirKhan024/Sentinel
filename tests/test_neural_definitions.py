"""The registry, the guards, and the architecture constants.

``definitions`` is the single source of truth for what Component 8 fits, so these tests
mostly assert that the *guards* work rather than that the values are what they are. A
test that restated ``EMBEDDING_DIMS`` would fail whenever the table changed and would
catch nothing; a test that proves a spec naming ``establishment_id`` cannot be
constructed catches the thing that matters.
"""

from __future__ import annotations

import pytest

from sentinel.features.definitions import FEATURE_COLUMNS
from sentinel.modeling.definitions import FORBIDDEN_COLUMNS
from sentinel.neural import definitions as d

# --- 1. the registry ---------------------------------------------------------


def test_the_registry_covers_every_required_experiment() -> None:
    """Each experiment the specification names has a registered model."""
    names = {s.name for s in d.NEURAL_REGISTRY}
    required = {
        "neural_embeddings",  # A
        "neural_numeric_only",  # A, fair-comparison control
        "neural_onehot",  # B
        "neural_no_chain",  # C
        "neural_no_facility_type",  # C
        "neural_no_community_area",  # C and D
        "neural_no_zip",  # C
        "xgboost_chain_embeddings",  # embeddings -> XGBoost
    }
    missing = required - names
    assert not missing, f"the registry is missing {', '.join(sorted(missing))}"


def test_model_names_are_unique() -> None:
    names = [s.name for s in d.NEURAL_REGISTRY]
    assert len(names) == len(set(names))


def test_every_spec_carries_an_experiment_label() -> None:
    """The label is what ties an artifact row back to a question that was asked."""
    for spec in d.NEURAL_REGISTRY:
        assert spec.experiment, f"{spec.name} has no experiment label"


def test_the_ablations_differ_from_the_primary_in_exactly_one_family() -> None:
    """An ablation that changed two things would not isolate either."""
    primary = d.spec_for("neural_embeddings")
    for name, dropped in (
        ("neural_no_chain", "chain"),
        ("neural_no_facility_type", "facility_type"),
        ("neural_no_community_area", "community_area"),
        ("neural_no_zip", "zip"),
    ):
        spec = d.spec_for(name)
        assert set(primary.entity_columns) - set(spec.entity_columns) == {dropped}
        assert spec.feature_columns == primary.feature_columns
        assert spec.encoding == primary.encoding
        assert spec.pos_weighted == primary.pos_weighted
        assert spec.seed == primary.seed


def test_the_onehot_control_differs_only_in_encoding() -> None:
    """Experiment B must vary the representation and nothing else."""
    primary = d.spec_for("neural_embeddings")
    control = d.spec_for("neural_onehot")
    assert control.entity_columns == primary.entity_columns
    assert control.feature_columns == primary.feature_columns
    assert control.seed == primary.seed
    assert control.pos_weighted == primary.pos_weighted
    assert control.encoding is d.CategoricalEncoding.ONE_HOT
    assert primary.encoding is d.CategoricalEncoding.EMBEDDING


def test_the_fair_comparison_control_sees_the_component_7_feature_set() -> None:
    """``neural_numeric_only`` is what makes the C6/C7/C8 table honest."""
    spec = d.spec_for("neural_numeric_only")
    assert spec.feature_columns == FEATURE_COLUMNS
    assert spec.entity_columns == ()
    assert spec.encoding is d.CategoricalEncoding.NONE


def test_the_weighting_ablation_is_the_only_weighted_model() -> None:
    weighted = [s.name for s in d.NEURAL_REGISTRY if s.pos_weighted]
    assert weighted == ["neural_pos_weighted"]
    assert d.POS_WEIGHT_DEFAULT is None


# --- 2. the guards -----------------------------------------------------------


def _spec(**overrides: object) -> d.NeuralSpec:
    base: dict[str, object] = {
        "name": "probe",
        "version": "v1",
        "description": "constructed by a test",
        "learner": d.Learner.MLP,
        "encoding": d.CategoricalEncoding.EMBEDDING,
        "feature_columns": FEATURE_COLUMNS,
        "entity_columns": ("chain",),
        "is_probability": True,
        "seed": 42,
    }
    base.update(overrides)
    return d.NeuralSpec(**base)  # type: ignore[arg-type]


def _guard(spec: d.NeuralSpec) -> None:
    """Run the import-time guard against one hand-made spec.

    The registry is a module constant, so it is swapped for the duration rather than
    appended to -- a leaked probe spec would then be fitted by every later test.
    """
    original = d.NEURAL_REGISTRY
    original_by_name = d.SPECS_BY_NAME
    try:
        d.NEURAL_REGISTRY = (*original, spec)  # type: ignore[misc]
        d.SPECS_BY_NAME = {s.name: s for s in d.NEURAL_REGISTRY}  # type: ignore[misc]
        d._guard_registry()
    finally:
        d.NEURAL_REGISTRY = original  # type: ignore[misc]
        d.SPECS_BY_NAME = original_by_name  # type: ignore[misc]


def test_an_establishment_id_embedding_is_refused() -> None:
    """The single most important guard in Component 8.

    An embedding of establishment identity is the largest leakage surface in the project
    and Component 4 excludes identity by design. It must be impossible to declare, not
    merely discouraged.
    """
    with pytest.raises(ValueError, match="establishment_id in particular is refused"):
        _guard(_spec(entity_columns=("establishment_id",)))


def test_an_undeclared_entity_family_is_refused() -> None:
    with pytest.raises(ValueError, match="not a declared EntityFamily"):
        _guard(_spec(entity_columns=("inspector_name",)))


def test_a_forbidden_feature_column_is_refused() -> None:
    with pytest.raises(ValueError, match="non-feature column"):
        _guard(_spec(feature_columns=(*FEATURE_COLUMNS, "target")))


def test_a_feature_column_component_4_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match="absent from Component"):
        _guard(_spec(feature_columns=(*FEATURE_COLUMNS, "weather_temperature")))


def test_declaring_categoricals_with_encoding_none_is_refused() -> None:
    """Otherwise a spec would declare columns and silently ignore them."""
    with pytest.raises(ValueError, match="would be declared and then silently ignored"):
        _guard(_spec(encoding=d.CategoricalEncoding.NONE, entity_columns=("chain",)))


def test_declaring_an_encoding_with_no_categoricals_is_refused() -> None:
    with pytest.raises(ValueError, match="Use CategoricalEncoding.NONE"):
        _guard(_spec(encoding=d.CategoricalEncoding.EMBEDDING, entity_columns=()))


def test_a_duplicate_model_name_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate model name"):
        _guard(_spec(name="neural_embeddings"))


def test_the_shipped_registry_passes_its_own_guard() -> None:
    """Belt on top of the import-time brace."""
    d._guard_registry()


def test_no_spec_leaks_a_forbidden_column() -> None:
    for spec in d.NEURAL_REGISTRY:
        assert not set(spec.feature_columns) & FORBIDDEN_COLUMNS
        assert not set(spec.entity_columns) & FORBIDDEN_COLUMNS


# --- 3. the embedding donor --------------------------------------------------


def test_the_embedding_fed_booster_has_a_registered_donor() -> None:
    for spec in d.NEURAL_REGISTRY:
        if spec.learner is d.Learner.XGBOOST_EMBEDDING:
            assert spec.name in d.EMBEDDING_DONOR
            donor = d.EMBEDDING_DONOR[spec.name]
            assert donor in d.SPECS_BY_NAME
            assert d.spec_for(donor).learner is d.Learner.MLP


def test_the_donor_learns_every_embedding_its_dependent_needs() -> None:
    for dependent, donor in d.EMBEDDING_DONOR.items():
        needed = set(d.spec_for(dependent).entity_columns)
        available = set(d.spec_for(donor).entity_columns)
        assert needed <= available, f"{dependent} needs {needed - available} from {donor}"


# --- 4. the architecture is the specification's ------------------------------


def test_the_architecture_matches_the_specification() -> None:
    """These are named by the project specification, so they are pinned."""
    assert d.HIDDEN_SIZES == (256, 128)
    assert d.DROPOUT == 0.3
    assert d.OUTPUT_SIZE == 1
    assert d.BATCH_SIZE == 512
    assert d.MAX_EPOCHS == 200
    assert d.EARLY_STOPPING_PATIENCE == 15
    assert d.GRADIENT_CLIP_NORM == 1.0
    assert d.OPTIMIZER == "AdamW"
    assert d.SCHEDULER == "ReduceLROnPlateau"
    assert d.LOSS == "BCEWithLogitsLoss"
    assert d.BASELINE_LEARNING_RATE == 1e-3


def test_the_embedding_dimensions_match_the_specification() -> None:
    assert d.EMBEDDING_DIMS[d.EntityFamily.CHAIN] == 16
    assert d.EMBEDDING_DIMS[d.EntityFamily.FACILITY_TYPE] == 8
    assert d.EMBEDDING_DIMS[d.EntityFamily.COMMUNITY_AREA] == 8
    assert d.EMBEDDING_DIMS[d.EntityFamily.ZIP] == 8


def test_every_family_has_a_declared_dimension() -> None:
    for name in d.ENTITY_COLUMNS:
        assert d.embedding_dim(name) > 0


def test_entity_column_order_is_the_declared_family_order() -> None:
    """Concatenation order is a property of the enum, not of how a spec is written."""
    assert tuple(f.value for f in d.EntityFamily) == d.ENTITY_COLUMNS
    for spec in d.NEURAL_REGISTRY:
        ordered = [c for c in d.ENTITY_COLUMNS if c in set(spec.entity_columns)]
        assert list(spec.entity_columns) == ordered, f"{spec.name} permutes the family order"


def test_embedding_width_is_zero_for_non_embedding_specs() -> None:
    assert d.embedding_width(d.spec_for("neural_numeric_only")) == 0
    assert d.embedding_width(d.spec_for("neural_onehot")) == 0
    assert d.embedding_width(d.spec_for("neural_embeddings")) == 16 + 8 + 8 + 8


# --- 5. the frozen learning rates --------------------------------------------


def test_the_learning_rate_is_frozen_per_fold_set() -> None:
    """Two fold sets, two rates, because the quarterly region contains the shift test."""
    assert set(d.TUNED_HYPERPARAMS) == {"quarterly", "covid_shift"}
    for fold_set in d.TUNED_HYPERPARAMS:
        assert d.learning_rate_for(fold_set) in d.LEARNING_RATE_GRID


def test_the_provenance_is_not_a_placeholder() -> None:
    """A frozen value with no provenance is indistinguishable from a guess."""
    assert "PLACEHOLDER" not in d.TUNED_HYPERPARAMS_PROVENANCE
    assert "tune-neural" in d.TUNED_HYPERPARAMS_PROVENANCE
    assert "sha256" in d.TUNED_HYPERPARAMS_PROVENANCE


def test_an_unknown_fold_set_falls_back_to_the_specified_baseline() -> None:
    assert d.learning_rate_for("not_a_fold_set") == d.BASELINE_LEARNING_RATE


def test_the_grid_brackets_the_specified_baseline() -> None:
    """The sweep must be able to say the baseline was neither best nor worst."""
    assert d.BASELINE_LEARNING_RATE in d.LEARNING_RATE_GRID
    assert min(d.LEARNING_RATE_GRID) < d.BASELINE_LEARNING_RATE < max(d.LEARNING_RATE_GRID)


def test_the_seed_sweep_includes_the_default_seed() -> None:
    """So the reported spread contains the seed that produced the headline predictions."""
    assert d.DEFAULT_SEED in d.SEED_SWEEP
    assert len(d.SEED_SWEEP) >= 3


def test_reserved_tokens_cannot_collide_with_a_real_category() -> None:
    """Both are dunder-style precisely so a normalised name cannot equal them."""
    assert d.UNKNOWN_CATEGORY.startswith("__") and d.UNKNOWN_CATEGORY.endswith("__")
    assert d.INDEPENDENT_CHAIN.startswith("__") and d.INDEPENDENT_CHAIN.endswith("__")
    assert d.UNKNOWN_CATEGORY != d.INDEPENDENT_CHAIN


def test_spec_for_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown neural model"):
        d.spec_for("not_a_model")
