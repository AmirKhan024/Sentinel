"""The frozen specifications, and the guards that keep them honest.

Most of this file is about one property: **a run cannot report a rule it did not apply.**
ADR 0025's pre-registration is only worth something if the threshold in the manifest is the
threshold in the source, and if the candidate set is a decision rather than whatever
happened to be convenient.
"""

from __future__ import annotations

import pytest

from sentinel.calibration import definitions
from sentinel.calibration.definitions import (
    CANDIDATE_REGISTRY,
    CANDIDATES_BY_NAME,
    EMBEDDING_DONOR,
    EXCLUDED_MODELS,
    INNER_SELECT_FRACTION,
    INPUT_TRANSFORM,
    ISOTONIC_PARAMS,
    MARGIN_TOLERANCE,
    MIN_INNER_FIT_ROWS,
    MIN_INNER_SELECT_ROWS,
    PLATT_PARAMS,
    SELECTION_METRIC,
    STAGES,
    TIE_PREFERENCE,
    TIE_THRESHOLD,
    CalibrationDefinitionError,
    Method,
    spec_for,
)


def test_the_registry_guard_runs_at_import_and_passes() -> None:
    definitions._guard_registry()


def test_every_candidate_records_why_it_is_in_the_set() -> None:
    """A candidate set without reasons is an artifact dump."""
    for candidate in CANDIDATE_REGISTRY:
        assert candidate.rationale
        assert candidate.source_slug
        assert candidate.component in (6, 7, 8)


def test_the_candidate_set_is_the_five_that_were_agreed() -> None:
    assert [c.name for c in CANDIDATE_REGISTRY] == [
        "logistic_regression",
        "xgboost",
        "lightgbm",
        "neural_numeric_only",
        "xgboost_chain_embeddings",
    ]


def test_the_experimental_candidate_is_labelled_as_one() -> None:
    """ADR 0022's labelling regime: calibrating it well must not make it the headline."""
    experimental = [c.name for c in CANDIDATE_REGISTRY if c.is_experimental]
    assert experimental == ["xgboost_chain_embeddings"]


def test_every_excluded_model_records_why_it_was_excluded() -> None:
    assert EXCLUDED_MODELS
    for name, reason in EXCLUDED_MODELS.items():
        assert reason, name
        assert name not in CANDIDATES_BY_NAME


def test_the_embedding_donor_is_fitted_but_never_calibrated() -> None:
    """It is an input to a candidate, not a candidate. HANDOFF is explicit about that."""
    assert EMBEDDING_DONOR not in CANDIDATES_BY_NAME
    assert EMBEDDING_DONOR in EXCLUDED_MODELS


def test_a_calibrated_name_can_never_collide_with_a_base_model_name() -> None:
    """The structural guard against applying a calibrator twice."""
    for candidate in CANDIDATE_REGISTRY:
        for method in Method:
            calibrated = candidate.calibrated_name(method)
            assert calibrated != candidate.name
            assert calibrated not in CANDIDATES_BY_NAME
            with pytest.raises(CalibrationDefinitionError):
                spec_for(calibrated)


def test_spec_for_names_the_alternatives_when_it_fails() -> None:
    with pytest.raises(CalibrationDefinitionError, match="Unknown calibration candidate"):
        spec_for("logistic_regresion")


# --- the pre-registered rule -------------------------------------------------


def test_the_tie_rule_is_the_value_adr_0025_declared() -> None:
    """If this changes, ADR 0025 is no longer describing the code.

    0.005 is one median paired-gap SD, measured in ``calibration_findings.md`` section 6
    *before* the first production run. The plan proposed 0.002, which sat below the smallest
    observed SD; that correction is the reason this test names the number.
    """
    assert TIE_THRESHOLD == 0.005
    assert TIE_PREFERENCE is Method.PLATT
    assert SELECTION_METRIC == "inner_select_log_loss"


def test_the_selection_metric_is_not_ece() -> None:
    """ECE is not a proper scoring rule and its bin count is a free parameter.

    A rule that can be tuned by changing a bin count is not a rule.
    """
    assert "ece" not in SELECTION_METRIC


def test_the_inner_split_constants_are_the_measured_ones() -> None:
    assert INNER_SELECT_FRACTION == 0.30
    assert MIN_INNER_FIT_ROWS == 400
    assert MIN_INNER_SELECT_ROWS == 250


def test_the_margin_tolerance_admits_float32_base_models() -> None:
    """Set from measurement: the network's round trip is 2.6e-5, not float64 epsilon.

    The plan proposed 1e-9, which would have failed on 33,898 correct rows.
    """
    assert MARGIN_TOLERANCE == 1e-4
    assert MARGIN_TOLERANCE > 2.615e-05


def test_the_tie_threshold_must_be_positive() -> None:
    """A zero threshold would make the declared preference unreachable."""
    assert TIE_THRESHOLD > 0.0


# --- the calibrators ---------------------------------------------------------


def test_platt_is_effectively_unpenalised() -> None:
    """A default C=1.0 would shrink the slope and CAUSE the miscalibration Platt fixes."""
    assert float(PLATT_PARAMS["C"]) >= 1e9  # type: ignore[arg-type]
    assert PLATT_PARAMS["fit_intercept"] is True


def test_isotonic_clips_out_of_range_inputs() -> None:
    """Without the clip a test score outside the calibration range becomes NaN.

    The prediction contract rejects a null score, so this is a correctness requirement
    rather than a preference.
    """
    assert ISOTONIC_PARAMS["out_of_bounds"] == "clip"
    assert ISOTONIC_PARAMS["increasing"] is True
    assert (ISOTONIC_PARAMS["y_min"], ISOTONIC_PARAMS["y_max"]) == (0.0, 1.0)


def test_each_method_declares_its_input_transform() -> None:
    """Platt takes the logit, isotonic takes the probability. ADR 0027."""
    assert INPUT_TRANSFORM[Method.PLATT] == "logit"
    assert INPUT_TRANSFORM[Method.ISOTONIC] == "identity"
    assert set(INPUT_TRANSFORM) == set(Method)


def test_both_methods_are_implemented_not_only_the_better_one() -> None:
    """The brief requires both, because they carry different assumptions."""
    assert {m.value for m in Method} == {"platt", "isotonic"}


def test_the_stages_cover_before_both_methods_and_the_frozen_choice() -> None:
    assert STAGES == ("uncalibrated", "platt", "isotonic", "selected")


def test_the_blocked_list_names_what_component_9_does_not_do() -> None:
    """Reported rather than faked -- the convention every component since Component 7 keeps."""
    blocked = " ".join(definitions.BLOCKED_EXPERIMENTS)
    for topic in ("threshold", "seed averaging", "temperature", "test quarter", "ensembl"):
        assert topic in blocked


def test_seed_averaging_is_recorded_as_a_deliberate_deferral() -> None:
    """Not an oversight. Neural seed noise exceeds that family's whole advantage."""
    text = " ".join(definitions.BLOCKED_EXPERIMENTS)
    assert "deferred by decision, not oversight" in text


def test_the_bootstrap_seed_component_is_stable_across_processes() -> None:
    """``candidate_index`` must not depend on Python's salted string hashing.

    This is a regression test for a real defect. The bootstrap seed key originally used
    ``abs(hash(candidate.name))``, and Python salts ``str`` hashing per process — so two runs
    over identical inputs drew different resamples and ``calibration_bootstrap_*.parquet`` was
    not byte-reproducible. A byte-for-byte comparison of two runs caught it.

    The registry position is stable, and it is checked here against a hardcoded expectation so
    that reordering the registry is a visible change rather than a silent reseeding.
    """
    from sentinel.calibration.definitions import candidate_index

    assert [candidate_index(spec.name) for spec in CANDIDATE_REGISTRY] == [0, 1, 2, 3, 4]
    assert candidate_index("xgboost") == 1
    with pytest.raises(CalibrationDefinitionError):
        candidate_index("not_a_model")


def test_candidate_index_does_not_shift_when_the_run_subsets_models() -> None:
    """``--models xgboost`` alone must give xgboost the same seed it gets in a full run."""
    from sentinel.calibration.definitions import candidate_index

    assert candidate_index("xgboost") == 1
    assert candidate_index("neural_numeric_only") == 3
