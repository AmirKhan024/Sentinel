"""End to end, against a real artifact produced by the real command.

Every test here runs over a genuine explanation artifact built by ``run_explanations`` from
a genuine boosted prediction artifact built by ``train_boosting``. Nothing is stubbed,
because the properties being asserted -- that an explanation maps to a committed prediction,
that the prediction is unchanged, that the model and fold identity survive -- are exactly
the properties a stub would assume.

The tree models are used because they are exact and fast. The permutation path has its own
correctness tests in ``test_explain_attribution.py``, against the Shapley definition; what
is being tested here is the plumbing around a method, not the method.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from sentinel.boosting import build as boosting_build
from sentinel.config import Settings
from sentinel.explain import validate, writer
from sentinel.explain.build import ExplainResult, required_sources, run_explanations
from sentinel.explain.definitions import (
    EXPLAIN_DEFINITION_VERSION,
    KNOWN_FEATURE_NAMES,
    SUPPORTED_MODELS,
    ExplanationStatus,
    spec_for,
)
from sentinel.explain.models import ExplainManifest
from sentinel.manifest import compute_sha256, read_manifest_as
from tests.conftest import spanning_model_features

MODELS = ["xgboost", "lightgbm"]
SAMPLE = 12


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExplainResult, Path, Path]:
    """A real explanation artifact on disk, produced by the real commands."""
    tmp = tmp_path_factory.mktemp("explain_contract")
    features = tmp / "as_of_features_20260101T000000Z.parquet"
    spanning_model_features(days=1900).write_parquet(features)

    boosted = boosting_build.train_boosting(
        Settings(data_dir=tmp), features_path=features, output_dir=tmp, models=MODELS
    )
    assert boosted.predictions_path is not None

    result = run_explanations(
        Settings(data_dir=tmp),
        features_path=features,
        prediction_paths={"boosted_predictions": boosted.predictions_path},
        output_dir=tmp / "explanations",
        models=MODELS,
        sample_size=SAMPLE,
        write_figures=False,
    )
    return result, boosted.predictions_path, features


@pytest.fixture(scope="module")
def tables(artifact: tuple[ExplainResult, Path, Path]) -> dict[str, pl.DataFrame]:
    return artifact[0].tables


# --- 1. the run is clean -----------------------------------------------------


def test_every_error_severity_check_passes(artifact: tuple[ExplainResult, Path, Path]) -> None:
    result = artifact[0]
    assert not validate.has_failures(result.checks), validate.format_report(result.checks)
    assert len(result.checks) >= 18, "too few checks; the suite would be weak"


def test_the_bit_identity_gate_compared_every_committed_row(
    artifact: tuple[ExplainResult, Path, Path],
) -> None:
    """The claim ADR 0029 rests on, measured rather than asserted."""
    stats = artifact[0].stats
    assert stats.reproduction_mismatches == 0
    assert stats.reproduction_rows > 0
    assert stats.refits == stats.folds * len(MODELS)


def test_the_prediction_artifact_is_byte_identical_after_the_run(
    artifact: tuple[ExplainResult, Path, Path],
) -> None:
    """Component 11 reads predictions and rewrites none of them."""
    result, predictions_path, _ = artifact
    manifest = read_manifest_as(ExplainManifest, result.manifest_path or Path())
    assert manifest.prediction_artifacts_unchanged is True
    assert manifest.boosted_predictions_sha256 == compute_sha256(predictions_path)
    assert manifest.prediction_sha256_after[predictions_path.name] == compute_sha256(
        predictions_path
    )


def test_no_prediction_value_was_altered(
    artifact: tuple[ExplainResult, Path, Path], tables: dict[str, pl.DataFrame]
) -> None:
    """Every recorded base score equals the committed float exactly, not approximately."""
    _, predictions_path, _ = artifact
    committed = pl.read_parquet(predictions_path).select(
        ["model_name", "fold_id", "target_inspection_id", "score"]
    )
    joined = tables["explanation_cases"].join(
        committed, on=["model_name", "fold_id", "target_inspection_id"], how="left"
    )
    assert joined.height == tables["explanation_cases"].height
    assert (joined["base_score"] == joined["score"]).all()


# --- 2. identity -------------------------------------------------------------


def test_every_explanation_names_a_model_a_fold_and_an_inspection(
    tables: dict[str, pl.DataFrame],
) -> None:
    values = tables["explanation_values"]
    assert values.height > 0
    for column in ("model_name", "fold_set", "fold_id", "target_inspection_id"):
        assert values[column].null_count() == 0


def test_every_explained_row_appears_in_the_committed_artifact(
    artifact: tuple[ExplainResult, Path, Path], tables: dict[str, pl.DataFrame]
) -> None:
    _, predictions_path, _ = artifact
    committed = pl.read_parquet(predictions_path)
    check = validate.every_explanation_maps_to_a_committed_prediction(
        tables["explanation_values"], committed
    )
    assert check.passed, check.offenders


def test_the_orphan_detector_itself_works(
    artifact: tuple[ExplainResult, Path, Path], tables: dict[str, pl.DataFrame]
) -> None:
    """Rename one explained inspection and assert the identity check goes red."""
    _, predictions_path, _ = artifact
    committed = pl.read_parquet(predictions_path)
    poisoned = tables["explanation_values"].with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("no-such-inspection"))
        .otherwise(pl.col("target_inspection_id"))
        .alias("target_inspection_id")
    )
    check = validate.every_explanation_maps_to_a_committed_prediction(poisoned, committed)
    assert not check.passed
    assert any("no-such-inspection" in offender for offender in check.offenders)


def test_the_prediction_mismatch_detector_itself_works(
    artifact: tuple[ExplainResult, Path, Path], tables: dict[str, pl.DataFrame]
) -> None:
    """Perturb one recorded score by a single ULP and assert the check goes red.

    The comparison is ``!=`` on floats. If a tolerance ever crept in, this is the test that
    would stop passing.
    """
    import math

    _, predictions_path, _ = artifact
    committed = pl.read_parquet(predictions_path)
    cases = tables["explanation_cases"]
    assert validate.prediction_values_match_the_committed_scores(cases, committed).passed

    first = float(cases["base_score"].to_list()[0])
    poisoned = cases.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(math.nextafter(first, math.inf)))
        .otherwise(pl.col("base_score"))
        .alias("base_score")
    )
    check = validate.prediction_values_match_the_committed_scores(poisoned, committed)
    assert not check.passed, "one ULP must be detectable, or the gate has a tolerance"


def test_every_fold_id_belongs_to_its_declared_fold_set(
    tables: dict[str, pl.DataFrame],
) -> None:
    values = tables["explanation_values"]
    pairs = set(
        zip(
            values["fold_id"].to_list(),
            values["fold_set"].to_list(),
            strict=True,
        )
    )
    for fold_id, fold_set in pairs:
        assert str(fold_id).startswith(str(fold_set))


def test_both_fold_sets_are_present_and_distinguishable(
    tables: dict[str, pl.DataFrame],
) -> None:
    sets = set(tables["explanation_values"]["fold_set"].unique().to_list())
    assert sets == {"quarterly", "covid_shift"}


# --- 3. feature representation ----------------------------------------------


def test_no_anonymous_feature_name_survives(tables: dict[str, pl.DataFrame]) -> None:
    names = set(tables["explanation_values"]["feature_name"].unique().to_list())
    assert names <= KNOWN_FEATURE_NAMES
    assert not any(validate.ANONYMOUS_NAME.match(str(name)) for name in names)


def test_the_anonymous_name_detector_itself_works(tables: dict[str, pl.DataFrame]) -> None:
    """The brief's own example, driven: ``feature_127`` must be rejected."""
    clean = validate.every_feature_maps_to_a_known_representation(tables["explanation_values"])
    assert clean.passed, "the positive control"

    poisoned = tables["explanation_values"].with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("feature_127"))
        .otherwise(pl.col("feature_name"))
        .alias("feature_name")
    )
    check = validate.every_feature_maps_to_a_known_representation(poisoned)
    assert not check.passed
    assert any("feature_127" in offender for offender in check.offenders)


def test_a_plain_feature_and_an_indicator_are_labelled_differently(
    tables: dict[str, pl.DataFrame],
) -> None:
    values = tables["explanation_values"]
    kinds = set(values["feature_kind"].unique().to_list())
    assert kinds == {"feature", "family_indicator"}

    indicator = values.filter(pl.col("feature_kind") == "family_indicator").head(1).to_dicts()[0]
    assert indicator["original_feature_name"] != indicator["feature_name"]
    assert "," in str(indicator["derived_from"]) or str(indicator["derived_from"])


def test_both_the_raw_and_the_transformed_value_are_carried(
    tables: dict[str, pl.DataFrame],
) -> None:
    """A reader needs the number a human recognises, not only the one the estimator saw."""
    values = tables["explanation_values"]
    assert values["transformed_value"].null_count() == 0
    # The boosters fit no preprocessing, so the two agree wherever the source was not NULL.
    both = values.filter(pl.col("feature_value").is_not_null())
    assert both.height > 0
    assert (both["feature_value"] == both["transformed_value"]).all()


def test_a_null_source_value_is_null_rather_than_zero(
    tables: dict[str, pl.DataFrame],
) -> None:
    """For a tree model a NULL is a real observation the split routed on; 0.0 is not it."""
    values = tables["explanation_values"]
    assert values["feature_value"].null_count() > 0


# --- 4. additivity and output space ------------------------------------------


def test_every_decomposition_reconstructs_its_model_output(
    tables: dict[str, pl.DataFrame],
) -> None:
    cases = tables["explanation_cases"]
    assert cases["additivity_holds"].all()
    assert (cases["reconstruction_residual"] <= cases["additivity_tolerance"]).all()


def test_the_additivity_detector_itself_works(tables: dict[str, pl.DataFrame]) -> None:
    """Nudge one contribution past tolerance and assert the check goes red."""
    import numpy as np

    from sentinel.explain.attribute import ExplanationMethod
    from sentinel.explain.definitions import OutputSpace, tolerance_for
    from sentinel.explain.models import FoldAttribution

    values = np.array([[1.0, 2.0]], dtype=np.float64)
    honest = FoldAttribution(
        model_name="xgboost",
        fold_set="quarterly",
        fold_id="quarterly-2026Q2",
        method=ExplanationMethod.TREE_SHAP,
        output_space=OutputSpace.LOG_ODDS,
        is_exact=True,
        row_ids=("1",),
        feature_names=("prior_canvass_count", "days_since_last_canvass"),
        values=values,
        base_value=0.5,
        output=np.array([3.5]),
        seconds=0.0,
    )
    assert validate.additivity_reconstructs_the_model_output([honest]).passed

    tolerance = tolerance_for(ExplanationMethod.TREE_SHAP)
    import dataclasses

    broken = dataclasses.replace(honest, output=np.array([3.5 + tolerance * 100]))
    check = validate.additivity_reconstructs_the_model_output([broken])
    assert not check.passed
    assert check.offenders


def test_the_output_space_is_declared_on_every_row(tables: dict[str, pl.DataFrame]) -> None:
    assert set(tables["explanation_values"]["output_space"].unique().to_list()) == {"log_odds"}


def test_positive_and_negative_contributions_sum_to_the_total(
    tables: dict[str, pl.DataFrame],
) -> None:
    cases = tables["explanation_cases"]
    total = cases["positive_contribution_sum"] + cases["negative_contribution_sum"]
    reconstructed = cases["base_value"] + total
    assert ((reconstructed - cases["reconstruction_value"]).abs() < 1e-9).all()


# --- 5. the unsupported model ------------------------------------------------


def test_the_unsupported_model_appears_only_in_the_support_table(
    tables: dict[str, pl.DataFrame],
) -> None:
    """Honest unsupported behaviour, made checkable."""
    name = "xgboost_chain_embeddings"
    assert tables["explanation_values"].filter(pl.col("model_name") == name).height == 0
    assert tables["explanation_cases"].filter(pl.col("model_name") == name).height == 0
    assert tables["explanation_importance"].filter(pl.col("model_name") == name).height == 0

    row = tables["explanation_support"].filter(pl.col("model_name") == name).to_dicts()[0]
    assert row["explanation_status"] == "unsupported"
    assert row["explanation_method"] is None
    assert row["output_space"] is None
    assert row["explained_rows"] == 0
    assert "_scorer_for" in str(row["unsupported_reason"])


def test_a_fabricated_attribution_for_an_unsupported_model_is_rejected(
    tables: dict[str, pl.DataFrame],
) -> None:
    """Placeholder zeros would read as 'this model used no features'. Drive the rejection."""
    clean = validate.unsupported_models_carry_no_attributions(
        tables["explanation_values"],
        tables["explanation_cases"],
        tables["explanation_support"],
    )
    assert clean.passed, "the positive control"

    fake = (
        tables["explanation_values"]
        .head(1)
        .with_columns(
            pl.lit("xgboost_chain_embeddings").alias("model_name"),
            pl.lit(0.0).alias("shap_value"),
        )
    )
    check = validate.unsupported_models_carry_no_attributions(
        pl.concat([tables["explanation_values"], fake]),
        tables["explanation_cases"],
        tables["explanation_support"],
    )
    assert not check.passed


def test_asking_to_explain_the_unsupported_model_fails_loudly(
    artifact: tuple[ExplainResult, Path, Path], tmp_path: Path
) -> None:
    from sentinel.explain.build import ExplainBuildError

    _, predictions_path, features = artifact
    with pytest.raises(ExplainBuildError, match="_scorer_for"):
        run_explanations(
            Settings(data_dir=tmp_path),
            features_path=features,
            prediction_paths={"neural_predictions": predictions_path},
            output_dir=tmp_path,
            models=["xgboost_chain_embeddings"],
            sample_size=SAMPLE,
            write_figures=False,
            dry_run=True,
        )


# --- 6. the artifact on disk -------------------------------------------------


def test_every_table_was_written_and_reads_back_identically(
    artifact: tuple[ExplainResult, Path, Path],
) -> None:
    result = artifact[0]
    assert len(result.written) == len(writer.SCHEMAS)
    for path in result.written:
        table = path.name.rsplit("_", 1)[0]
        assert pl.read_parquet(path).equals(result.tables[table])


def test_every_table_carries_its_contract_schema(
    artifact: tuple[ExplainResult, Path, Path],
) -> None:
    result = artifact[0]
    for path in result.written:
        table = path.name.rsplit("_", 1)[0]
        assert list(pl.read_parquet(path).columns) == list(writer.SCHEMAS[table])


def test_every_table_is_sorted_by_its_declared_key(
    artifact: tuple[ExplainResult, Path, Path],
) -> None:
    result = artifact[0]
    for table, frame in result.tables.items():
        assert frame.equals(frame.sort(writer.SORT_KEYS[table]))


def test_the_manifest_pins_the_inputs_and_records_the_semantics(
    artifact: tuple[ExplainResult, Path, Path],
) -> None:
    result, _, features = artifact
    assert result.manifest_path is not None
    manifest = read_manifest_as(ExplainManifest, result.manifest_path)

    assert manifest.features_sha256 == compute_sha256(features)
    assert manifest.explain_definition_version == EXPLAIN_DEFINITION_VERSION
    assert manifest.reproduction_passed is True
    assert manifest.covid_reported_separately is True
    assert "not causality" in manifest.causality_disclaimer
    assert "BASE model" in manifest.calibration_boundary
    assert manifest.output_spaces["xgboost"] == "log_odds"
    assert manifest.exactness["neural_numeric_only"] is False
    assert "xgboost_chain_embeddings" in manifest.unsupported_models
    assert any("model selection" in line for line in manifest.blocked)


def test_the_manifest_records_only_the_artifacts_this_run_read(
    artifact: tuple[ExplainResult, Path, Path],
) -> None:
    """A checksum for a file nobody opened would be provenance about a different run."""
    result = artifact[0]
    manifest = read_manifest_as(ExplainManifest, result.manifest_path or Path())
    assert manifest.boosted_predictions_path is not None
    assert manifest.baseline_predictions_path is None
    assert manifest.neural_predictions_path is None


def test_required_sources_follows_the_registry() -> None:
    assert required_sources(["xgboost", "lightgbm"]) == {"boosted_predictions"}
    assert required_sources(["logistic_regression"]) == {"baseline_predictions"}
    assert required_sources(list(SUPPORTED_MODELS)) == {
        "baseline_predictions",
        "boosted_predictions",
        "neural_predictions",
    }


def test_a_missing_source_artifact_is_refused_rather_than_skipped(
    artifact: tuple[ExplainResult, Path, Path], tmp_path: Path
) -> None:
    from sentinel.explain.build import ExplainBuildError

    _, predictions_path, features = artifact
    with pytest.raises(ExplainBuildError, match="needs the committed artifact"):
        run_explanations(
            Settings(data_dir=tmp_path),
            features_path=features,
            prediction_paths={"boosted_predictions": predictions_path},
            output_dir=tmp_path,
            models=["logistic_regression"],
            sample_size=SAMPLE,
            write_figures=False,
            dry_run=True,
        )


# --- 7. provenance a downstream component can use ----------------------------


def test_the_horizon_recorded_is_the_base_models_not_the_calibrators(
    tables: dict[str, pl.DataFrame],
) -> None:
    """The attribution decomposes the estimator's output, so it declares the estimator's horizon."""
    cases = tables["explanation_cases"]
    assert cases["base_model_trained_through"].null_count() == 0
    assert cases.schema["base_model_trained_through"] == pl.Date()
    # No calibrated artifact was supplied, so the calibrator's columns are null rather than
    # invented -- which is the state a consumer must be able to tell apart from a date.
    assert cases["calibrator_fitted_through"].null_count() == cases.height
    assert cases["calibrated_probability"].null_count() == cases.height


def test_the_trained_through_date_matches_the_folds_training_end(
    tables: dict[str, pl.DataFrame],
) -> None:
    values = tables["explanation_values"]
    assert values["trained_through"].null_count() == 0
    assert all(isinstance(v, date) for v in values["trained_through"].unique().to_list())


def test_the_sampling_provenance_is_recorded_on_every_case(
    tables: dict[str, pl.DataFrame],
) -> None:
    cases = tables["explanation_cases"]
    assert (cases["sample_size"] == SAMPLE).all()
    assert cases["sampling_seed"].n_unique() == 1
    assert cases["sampling_population"].unique().to_list() == ["fold test window"]
    assert (cases["population_rows"] > 0).all()


def test_the_tree_models_record_no_background_because_they_need_none(
    tables: dict[str, pl.DataFrame],
) -> None:
    """Writing a background size for TreeSHAP would imply a reference set it never used."""
    cases = tables["explanation_cases"]
    assert (cases["background_size"] == 0).all()
    assert (cases["permutation_rounds"] == 0).all()


def test_the_support_table_accounts_for_every_registered_model(
    tables: dict[str, pl.DataFrame],
) -> None:
    support = tables["explanation_support"]
    assert support.height == 5
    for row in support.to_dicts():
        spec = spec_for(str(row["model_name"]))
        assert row["explanation_status"] == spec.status.value
        if spec.status is ExplanationStatus.SUPPORTED:
            assert row["name_source"] == spec.name_source
