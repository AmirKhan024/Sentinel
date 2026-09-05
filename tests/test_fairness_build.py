"""End to end: the audit runs, writes ten tables, changes nothing, and repeats exactly.

These tests drive the real orchestration over a synthetic snapshot big enough to build real
quarterly folds from. What they are checking is not that the numbers are right -- the metric
tests do that -- but that the *run* is what it claims to be: a complete pass over every
audited row, an artifact whose every group is accounted for, and an observer that left its
inputs untouched.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sentinel.config import Settings
from sentinel.fairness import validate, writer
from sentinel.fairness.build import FairnessBuildError, run_fairness_audit, summarize
from sentinel.fairness.definitions import (
    DOES_NOT_ESTABLISH,
    GroupStatus,
    Stage,
)
from sentinel.manifest import compute_sha256
from tests.conftest import (
    calibrated_predictions_for,
    explanation_values_for,
    neural_categoricals_for,
    spanning_model_features,
)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Write a full synthetic input set once; every test in this file reads it.

    Module-scoped because building the folds and the prediction frame costs a second or two
    and no test mutates the files -- which is itself the property the no-mutation check is
    asserting, so a shared fixture is a mild extra proof of it.
    """
    directory = tmp_path_factory.mktemp("fairness_inputs")
    features = spanning_model_features(days=1900, per_day=3)
    categoricals = neural_categoricals_for(features)
    predictions = calibrated_predictions_for(features)
    explanations = explanation_values_for(predictions)

    paths = {
        "features": directory / "as_of_features_20260825T000000Z.parquet",
        "categoricals": directory / "neural_categoricals_20260825T000000Z.parquet",
        "predictions": directory / "calibrated_predictions_20260825T000000Z.parquet",
        "explanations": directory / "explanation_values_20260825T000000Z.parquet",
    }
    features.write_parquet(paths["features"])
    categoricals.write_parquet(paths["categoricals"])
    predictions.write_parquet(paths["predictions"])
    explanations.write_parquet(paths["explanations"])
    return paths


def _run(settings: Settings, inputs: dict[str, Path], tmp_path: Path, **kwargs: object) -> object:
    return run_fairness_audit(
        settings,
        features_path=inputs["features"],
        calibrated_path=inputs["predictions"],
        categoricals_path=inputs["categoricals"],
        explanations_path=inputs.get("explanations"),
        output_dir=tmp_path / "out",
        write_figures=False,
        **kwargs,  # type: ignore[arg-type]
    )


# --- 1. the run produces the artifact it declares ------------------------------


def test_the_run_writes_every_declared_table_plus_a_manifest(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path)
    written = {path.stem.rsplit("_", 1)[0] for path in result.written}  # type: ignore[attr-defined]
    assert written == set(writer.SCHEMAS)
    assert result.manifest_path is not None  # type: ignore[attr-defined]
    assert result.manifest_path.exists()  # type: ignore[attr-defined]


def test_a_dry_run_writes_nothing_but_still_validates(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    assert result.written == []  # type: ignore[attr-defined]
    assert result.manifest_path is None  # type: ignore[attr-defined]
    assert result.checks  # type: ignore[attr-defined]
    assert not (tmp_path / "out").exists()


def test_every_error_severity_check_passes_on_an_honest_input(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    assert not validate.has_failures(result.checks)  # type: ignore[attr-defined]


def test_the_audit_covers_every_scored_row(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    expected = pl.read_parquet(inputs["predictions"]).height
    assert result.stats.audited_rows == expected  # type: ignore[attr-defined]


# --- 2. no group disappears -----------------------------------------------------


def test_every_observed_group_appears_in_the_support_table(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """The property that makes the small-group policy real."""
    result = _run(settings, inputs, tmp_path, dry_run=True)
    support = result.tables["fairness_group_support"]  # type: ignore[attr-defined]
    categoricals = pl.read_parquet(inputs["categoricals"])
    observed = set(categoricals["community_area"].unique().to_list())
    recorded = set(
        support.filter(pl.col("group_definition") == "community_area")["group_value"]
        .unique()
        .to_list()
    )
    assert observed <= recorded


def test_unsupported_groups_are_rows_with_nulls_and_a_reason(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    metrics = result.tables["fairness_group_metrics"]  # type: ignore[attr-defined]
    unsupported = metrics.filter(pl.col("group_status") == GroupStatus.INSUFFICIENT_SUPPORT.value)
    if unsupported.is_empty():
        pytest.skip("this synthetic snapshot supported every group")
    assert unsupported["value"].null_count() == unsupported.height
    assert (unsupported["insufficient_reason"].str.len_chars() > 0).all()
    # Counts are still real: support gates the reading, not the arithmetic.
    assert unsupported["n_rows"].null_count() == 0


# --- 3. both stages are measured and never confused -------------------------------


def test_both_prediction_stages_are_measured(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    metrics = result.tables["fairness_group_metrics"]  # type: ignore[attr-defined]
    assert set(metrics["stage"].unique().to_list()) == {Stage.BASE.value, Stage.CALIBRATED.value}


def test_the_calibration_table_answers_did_platt_reach_this_group(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    calibration = result.tables["fairness_group_calibration"]  # type: ignore[attr-defined]
    assert "improved" in calibration.columns
    supported = calibration.filter(pl.col("base_value").is_not_null())
    if not supported.is_empty():
        assert supported["improved"].null_count() < supported.height


# --- 4. covid is never pooled -------------------------------------------------------


def test_the_two_fold_sets_stay_separate(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    for name, frame in result.tables.items():  # type: ignore[attr-defined]
        if "fold_set" in frame.columns and not frame.is_empty():
            assert set(frame["fold_set"].unique().to_list()) <= {"quarterly", "covid_shift"}, name


# --- 5. the run is an observer -------------------------------------------------------


def test_the_inputs_are_byte_identical_after_the_run(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """This component's whole value rests on it being an observer."""
    before = {name: compute_sha256(path) for name, path in inputs.items()}
    result = _run(settings, inputs, tmp_path)
    after = {name: compute_sha256(path) for name, path in inputs.items()}
    assert before == after
    assert result.stats.inputs_unchanged  # type: ignore[attr-defined]
    assert result.manifest.inputs_unchanged  # type: ignore[attr-defined]


# --- 6. determinism -------------------------------------------------------------------


def test_two_runs_produce_byte_identical_tables(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """Same inputs, same configuration, same bytes. The standard every component holds to."""
    first = _run(settings, inputs, tmp_path / "a", dry_run=True)
    second = _run(settings, inputs, tmp_path / "b", dry_run=True)
    for name in writer.SCHEMAS:
        assert first.tables[name].equals(second.tables[name]), name  # type: ignore[attr-defined]


def test_shuffling_the_prediction_rows_does_not_change_the_artifact(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """Parquet row order is not a contract, so a shuffled input must audit identically."""
    shuffled_dir = tmp_path / "shuffled"
    shuffled_dir.mkdir()
    shuffled_path = shuffled_dir / "calibrated_predictions_20260825T000000Z.parquet"
    pl.read_parquet(inputs["predictions"]).sample(fraction=1.0, shuffle=True, seed=7).write_parquet(
        shuffled_path
    )

    baseline = _run(settings, inputs, tmp_path / "base", dry_run=True)
    shuffled = run_fairness_audit(
        settings,
        features_path=inputs["features"],
        calibrated_path=shuffled_path,
        categoricals_path=inputs["categoricals"],
        explanations_path=inputs["explanations"],
        output_dir=tmp_path / "out2",
        write_figures=False,
        dry_run=True,
    )
    for name in ("fairness_group_support", "fairness_group_metrics", "fairness_disparity"):
        assert baseline.tables[name].equals(shuffled.tables[name]), name  # type: ignore[attr-defined]


# --- 7. the boundary travels with the artifact -----------------------------------------


def test_the_manifest_carries_the_claims_the_component_cannot_make(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """ADR 0035: a document can be lost, and a manifest sits beside the data."""
    result = _run(settings, inputs, tmp_path, dry_run=True)
    assert result.manifest.does_not_establish == list(DOES_NOT_ESTABLISH)  # type: ignore[attr-defined]
    assert result.manifest.blocked  # type: ignore[attr-defined]
    assert result.manifest.inherited_limitations  # type: ignore[attr-defined]
    assert any("ADR 0019" in line for line in result.manifest.inherited_limitations)  # type: ignore[attr-defined]


def test_the_manifest_restates_the_refused_definitions(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    refused = result.manifest.refused_group_definitions  # type: ignore[attr-defined]
    assert "ward" in refused
    assert "boundary version" in refused["ward"]


def test_the_definitions_table_carries_the_refusals_as_data(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """A refusal that lives only in prose stops travelling the moment someone reads the
    Parquet instead of the ADR.
    """
    result = _run(settings, inputs, tmp_path, dry_run=True)
    table = result.tables["fairness_group_definitions"]  # type: ignore[attr-defined]
    ward = table.filter(pl.col("group_definition") == "ward")
    assert ward.height == 1
    assert ward["status"].item() == "refused"
    assert "98.3%" in ward["refusal_reason"].item()


def test_the_summary_prints_the_boundary_on_every_run(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(settings, inputs, tmp_path, dry_run=True)
    text = summarize(result)  # type: ignore[arg-type]
    assert "DOES NOT ESTABLISH" in text
    assert "insufficient support" in text
    assert "DRY RUN" in text


# --- 8. rejection paths ------------------------------------------------------------------


def test_an_unknown_model_is_rejected_with_the_available_ones(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(FairnessBuildError, match="available"):
        _run(settings, inputs, tmp_path, dry_run=True, models=["random_forest_platt"])


def test_a_refused_group_definition_is_rejected(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    from sentinel.fairness.definitions import FairnessDefinitionError

    with pytest.raises(FairnessDefinitionError, match="refused"):
        _run(settings, inputs, tmp_path, dry_run=True, group_definitions=["ward"])


def test_a_frame_that_is_not_a_feature_table_is_rejected(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    bad = tmp_path / "not_features.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(bad)
    with pytest.raises(FairnessBuildError, match="not a Component 4 feature table"):
        run_fairness_audit(
            settings,
            features_path=bad,
            calibrated_path=inputs["predictions"],
            categoricals_path=inputs["categoricals"],
            output_dir=tmp_path / "out3",
            write_figures=False,
            dry_run=True,
        )


def test_the_attribution_table_is_empty_without_component_elevens_artifact(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """Absent is a supported state: the run is complete and one table is empty."""
    result = run_fairness_audit(
        settings,
        features_path=inputs["features"],
        calibrated_path=inputs["predictions"],
        categoricals_path=inputs["categoricals"],
        explanations_path=None,
        output_dir=tmp_path / "out4",
        write_figures=False,
        dry_run=True,
    )
    assert result.tables["fairness_attribution_profiles"].height == 0
    assert not validate.has_failures(result.checks)


# --- 9. the aggregation itself must not depend on summation order ------------------------


def test_attribution_means_do_not_depend_on_row_order(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """The specific defect two full production runs found, pinned so it cannot come back.

    Polars aggregates a group in parallel and adds the rows in whatever order they reach a
    thread, so two runs produced ``mean_abs_shap`` values differing at 1.8e-15. Sorting the
    frame first was not enough -- it fixes which rows are in which group, not the order they
    are added in. ``attribution._profile_means`` uses ``math.fsum``, which is exactly rounded
    and therefore order-independent on every machine rather than on this one.

    The difference was far below anything a reader would act on: every rank, every Spearman
    correlation and every count was identical. It still mattered, because a table that is only
    *nearly* reproducible is a table whose two-run checksum comparison has stopped being a
    detector.
    """
    from sentinel.fairness import attribution
    from sentinel.fairness.definitions import group_definition_for
    from sentinel.fairness.groups import group_source

    explanations = pl.read_parquet(inputs["explanations"])
    spec = group_definition_for("community_area")
    lookup = group_source(pl.read_parquet(inputs["categoricals"]), [spec]).select(
        "target_inspection_id", spec.source_column
    )

    straight = attribution.profiles(explanations, lookup, spec, fold_set="quarterly", min_rows=1)
    shuffled = attribution.profiles(
        explanations.sample(fraction=1.0, shuffle=True, seed=11),
        lookup,
        spec,
        fold_set="quarterly",
        min_rows=1,
    )

    assert straight, "the fixture produced no profiles, so this test would pass vacuously"
    assert len(straight) == len(shuffled)
    for a, b in zip(straight, shuffled, strict=True):
        # Exact equality, not approximate. Approximate is the state this test exists to reject.
        assert a.mean_abs_shap == b.mean_abs_shap
        assert a.mean_shap == b.mean_shap
        assert (a.model_name, a.group_value, a.feature_name, a.rank) == (
            b.model_name,
            b.group_value,
            b.feature_name,
            b.rank,
        )


def test_the_whole_attribution_table_is_stable_under_a_shuffle(
    settings: Settings, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """The same property at the artifact level, where a checksum comparison would see it."""
    shuffled_dir = tmp_path / "shuffled_explanations"
    shuffled_dir.mkdir()
    shuffled_path = shuffled_dir / "explanation_values_20260825T000000Z.parquet"
    pl.read_parquet(inputs["explanations"]).sample(
        fraction=1.0, shuffle=True, seed=3
    ).write_parquet(shuffled_path)

    baseline = _run(settings, inputs, tmp_path / "base_attr", dry_run=True)
    shuffled = run_fairness_audit(
        settings,
        features_path=inputs["features"],
        calibrated_path=inputs["predictions"],
        categoricals_path=inputs["categoricals"],
        explanations_path=shuffled_path,
        output_dir=tmp_path / "out_attr",
        write_figures=False,
        dry_run=True,
    )
    assert baseline.tables["fairness_attribution_profiles"].equals(  # type: ignore[attr-defined]
        shuffled.tables["fairness_attribution_profiles"]
    )
