"""The safety wall: every check driven to red on purpose.

A check whose failure path has never been observed is indistinguishable from one that cannot
fire. Component 5 shipped exactly that defect once, and Components 9 and 11 each answered it
with a test that breaks the thing the check exists to catch. This file is Component 12's.

The failure modes here are quieter than any earlier component's. A leaked feature makes one
column wrong. A leaked *group mapping* leaves every number finite, additive and plausible, and
changes only the question they answer -- so nothing raises, no metric moves out of range, and
the audit reports a disparity across neighbourhoods it has mislabelled.

The last test in this file is the most important one and it asserts the opposite of the
others: **a large measured disparity must NOT turn the build red.**
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from sentinel.fairness import groups, validate
from sentinel.fairness.definitions import GroupStatus, group_definition_for
from sentinel.fairness.models import SEVERITY_ERROR, SEVERITY_WARN

COMMUNITY_AREA = group_definition_for("community_area")


def _source(rows: int = 6, **overrides: object) -> pl.DataFrame:
    """A minimal group source in Component 8's shape, valid unless deliberately broken."""
    base = date(2024, 3, 1)
    frame = pl.DataFrame(
        {
            "target_inspection_id": [f"T{i:03d}" for i in range(rows)],
            "establishment_id": [f"EST-{i % 3:03d}" for i in range(rows)],
            "inspection_date": [(base + timedelta(days=i)).isoformat() for i in range(rows)],
            "source_inspection_id": [f"S{i:03d}" for i in range(rows)],
            "source_inspection_date": [base + timedelta(days=i - 30) for i in range(rows)],
            "days_since_source": [30] * rows,
            "community_area": [str(i % 2) for i in range(rows)],
            "zip": [f"6060{i % 2}" for i in range(rows)],
        }
    )
    return frame.with_columns(**overrides) if overrides else frame


def _audited(source: pl.DataFrame) -> pl.DataFrame:
    """The joined audit frame the checks read, derived from a source frame."""
    n = source.height
    return source.select("target_inspection_id", "community_area", "zip").with_columns(
        pl.lit("logistic_regression_platt").alias("model_name"),
        pl.lit("quarterly").alias("fold_set"),
        pl.lit("quarterly-2024Q2").alias("fold_id"),
        pl.Series("calibrated_probability", [0.4 + i / 100 for i in range(n)]),
        pl.Series("base_probability", [0.3 + i / 100 for i in range(n)]),
        pl.Series("target", [i % 2 for i in range(n)]),
    )


def _predictions(audited: pl.DataFrame) -> pl.DataFrame:
    return audited.select(
        "target_inspection_id",
        "model_name",
        pl.col("calibrated_probability").alias("score"),
        pl.col("base_probability").alias("base_score"),
    )


# --- 1. temporal safety: the group mapping may not see the present -------------


def test_a_group_mapping_dated_after_its_row_turns_the_check_red() -> None:
    """The leak this component is most exposed to and least able to notice afterwards."""
    bad = _source().with_columns(
        pl.col("source_inspection_date").dt.offset_by("400d").alias("source_inspection_date")
    )
    check = validate.group_mapping_predates_every_row(bad)
    assert not check.passed
    assert check.severity == SEVERITY_ERROR


def test_a_same_day_group_mapping_turns_the_check_red() -> None:
    """A zero-day lag means a row supplied its own attributes.

    Component 8's join uses `allow_exact_matches=False` precisely to make this
    unconstructable, so observing it means something upstream changed.
    """
    bad = _source().with_columns(
        pl.col("inspection_date").str.to_date().alias("source_inspection_date"),
        pl.lit(0).cast(pl.Int32).alias("days_since_source"),
    )
    assert not validate.group_mapping_predates_every_row(bad).passed


def test_the_temporal_detector_passes_on_an_honest_frame() -> None:
    """Otherwise the two tests above would pass against a check that always fails."""
    check = validate.group_mapping_predates_every_row(_source())
    assert check.passed
    assert "minimum lag 30 day(s)" in check.detail


def test_building_a_frame_from_a_future_mapping_raises_rather_than_measuring_it() -> None:
    """The build refuses before any metric is computed, not after."""
    bad = _source().with_columns(
        pl.col("source_inspection_date").dt.offset_by("400d").alias("source_inspection_date")
    )
    with pytest.raises(groups.GroupFrameError, match="future information"):
        groups.check_temporal_validity(bad)


# --- 2. join integrity ----------------------------------------------------------


def test_a_key_mapping_to_two_group_values_turns_the_check_red() -> None:
    """An ambiguous mapping multiplies audited rows and inflates every support count."""
    doubled = pl.concat([_source(2), _source(2).with_columns(pl.lit("99").alias("community_area"))])
    check = validate.group_mapping_is_unambiguous(doubled, ["community_area"])
    assert not check.passed
    assert check.severity == SEVERITY_ERROR


def test_a_group_value_absent_from_the_source_turns_the_check_red() -> None:
    """A neighbourhood id nobody can trace back is worse than no neighbourhood id."""
    source = _source()
    audited = _audited(source).with_columns(pl.lit("77").alias("community_area"))
    check = validate.every_group_value_comes_from_the_source(audited, source, ["community_area"])
    assert not check.passed
    assert "community_area=77" in check.offenders


def test_a_scored_row_missing_from_the_group_frame_raises_rather_than_being_dropped() -> None:
    """An inner join would produce a smaller, plausible frame and report over it."""
    source = _source(6)
    predictions = _predictions(_audited(source)).with_columns(
        pl.lit("quarterly").alias("fold_set"),
        pl.lit("quarterly-2024Q2").alias("fold_id"),
        pl.lit("logistic_regression").alias("base_model_name"),
        pl.lit(False).alias("is_experimental"),
        pl.lit("platt").alias("method"),
    )
    labels = source.select("target_inspection_id").with_columns(
        pl.lit(1).alias("target"), pl.lit(date(2024, 4, 1)).alias("rd")
    )
    with pytest.raises(groups.GroupFrameError, match="have no community_area value"):
        groups.build_group_frame(predictions, source.head(3), labels, [COMMUNITY_AREA])


def test_a_dropped_audited_row_turns_the_prediction_check_red() -> None:
    """Set equality, not containment: containment passes on a frame that lost rows."""
    source = _source(6)
    audited = _audited(source)
    check = validate.every_audited_row_has_a_prediction(audited.head(4), _predictions(audited))
    assert not check.passed
    assert "2 dropped" in check.detail


# --- 3. stage integrity: base and calibrated may not be swapped ------------------


def test_swapping_base_and_calibrated_turns_the_check_red() -> None:
    """A calibrated ECE reported as an uncalibrated one would invert the headline finding.

    And every number would stay in range while it happened, which is why this is checked
    against the committed artifact with `==` rather than inspected by eye.
    """
    source = _source()
    audited = _audited(source)
    predictions = _predictions(audited)
    swapped = audited.with_columns(
        pl.col("base_probability").alias("calibrated_probability"),
        pl.col("calibrated_probability").alias("base_probability"),
    )
    check = validate.stages_are_not_confused(swapped, predictions)
    assert not check.passed
    assert check.severity == SEVERITY_ERROR


def test_the_stage_detector_passes_on_an_honest_frame() -> None:
    source = _source()
    audited = _audited(source)
    check = validate.stages_are_not_confused(audited, _predictions(audited))
    assert check.passed
    assert "0 calibrated and 0 base mismatches" in check.detail


def test_the_stage_column_lookup_refuses_an_unknown_stage() -> None:
    assert groups.stage_column("base") == "base_probability"
    assert groups.stage_column("calibrated") == "calibrated_probability"
    with pytest.raises(groups.GroupFrameError, match="unknown prediction stage"):
        groups.stage_column("raw")


# --- 4. support may not disappear -------------------------------------------------


def test_removing_a_small_group_from_the_support_table_turns_the_check_red() -> None:
    """The check that makes the small-group policy real rather than aspirational."""
    support = pl.DataFrame(
        {
            "group_definition": ["community_area"] * 2,
            "group_value": ["0", "1"],
        }
    )
    check = validate.no_group_disappeared(support, {"community_area": ["0", "1", "2"]})
    assert not check.passed
    assert "community_area=2" in check.offenders


def test_the_support_detector_passes_when_every_group_is_present() -> None:
    support = pl.DataFrame(
        {"group_definition": ["community_area"] * 3, "group_value": ["0", "1", "2"]}
    )
    assert validate.no_group_disappeared(support, {"community_area": ["0", "1", "2"]}).passed


def test_publishing_a_value_for_an_unsupported_group_turns_the_check_red() -> None:
    """A number that failed its floor is a number that should not have been read."""
    metrics = pl.DataFrame(
        {
            "n_rows": [12],
            "n_positive": [3],
            "n_negative": [9],
            "group_status": [GroupStatus.INSUFFICIENT_SUPPORT.value],
            "value": [0.83],
            "group_value": ["2"],
        }
    )
    check = validate.every_metric_carries_support(metrics)
    assert not check.passed
    assert "1 unsupported row(s) carrying a value" in check.detail


def test_a_support_status_disagreeing_with_its_own_counts_turns_the_check_red() -> None:
    """A status column that contradicts the counts beside it is the quietest lie available."""
    support = pl.DataFrame(
        {
            "group_definition": ["community_area"],
            "group_value": ["2"],
            "n_rows": [12],
            "n_positive": [3],
            "n_negative": [9],
            "ranking_status": [GroupStatus.SUPPORTED.value],
            "calibration_status": [GroupStatus.SUPPORTED.value],
        }
    )
    assert not validate.support_decisions_are_reproducible(support).passed


# --- 5. the artifact may not become joinable --------------------------------------


def test_an_outcome_column_in_the_artifact_turns_the_check_red() -> None:
    """These tables are keyed by group so they cannot be joined onto a feature table.

    A stray `target` or `score` column would undo that, and a per-group number joined back
    onto training rows would make a model's measured behaviour on a neighbourhood an input
    to how it treats that neighbourhood next time. ADR 0032.
    """
    tables = {"fairness_group_metrics": pl.DataFrame({"value": [0.5], "target": [1]})}
    check = validate.no_outcome_or_feature_column_leaked(tables)
    assert not check.passed
    assert "fairness_group_metrics.target" in check.offenders


def test_a_score_column_in_the_artifact_turns_the_check_red() -> None:
    tables = {"fairness_group_support": pl.DataFrame({"n_rows": [10], "score": [0.5]})}
    assert not validate.no_outcome_or_feature_column_leaked(tables).passed


# --- 6. covid may not be pooled ------------------------------------------------------


def test_a_pooled_fold_set_turns_the_check_red() -> None:
    """Averaging covid_shift into the quarterly mean has been forbidden since Component 5."""
    tables = {"fairness_disparity": pl.DataFrame({"fold_set": ["all"]})}
    check = validate.covid_was_not_pooled(tables)
    assert not check.passed
    assert "fairness_disparity: fold_set=all" in check.offenders


def test_the_two_real_fold_sets_pass() -> None:
    tables = {"fairness_disparity": pl.DataFrame({"fold_set": ["quarterly", "covid_shift"]})}
    assert validate.covid_was_not_pooled(tables).passed


# --- 7. the inputs may not move -------------------------------------------------------


def test_a_changed_input_checksum_turns_the_check_red() -> None:
    """This component's whole value rests on it being an observer."""
    check = validate.inputs_were_not_modified({"features": "a" * 64}, {"features": "b" * 64})
    assert not check.passed
    assert check.severity == SEVERITY_ERROR


def test_unchanged_checksums_pass() -> None:
    digests = {"features": "a" * 64, "categoricals": "c" * 64}
    assert validate.inputs_were_not_modified(digests, digests).passed


# --- 8. determinism of the written tables ---------------------------------------------


def test_an_unsorted_table_turns_the_check_red() -> None:
    frame = pl.DataFrame({"group_value": ["2", "1", "0"]})
    check = validate.tables_are_deterministically_sorted({"t": frame}, {"t": ["group_value"]})
    assert not check.passed
    assert "t: not in sort order" in check.offenders


def test_a_duplicate_sort_key_turns_the_check_red() -> None:
    frame = pl.DataFrame({"group_value": ["0", "0", "1"]})
    check = validate.tables_are_deterministically_sorted({"t": frame}, {"t": ["group_value"]})
    assert not check.passed
    assert "t: duplicate sort key" in check.offenders


# --- 9. THE MOST IMPORTANT TEST IN THIS FILE ------------------------------------------


def test_an_enormous_disparity_is_advisory_and_never_an_error() -> None:
    """A measured inequality is evidence. It is not a defect in this code.

    This is the test that keeps the component honest. If a large disparity failed the build,
    every future change to this repository would be made under pressure to move that number,
    and the cheapest ways to move it are to change the metric or the threshold. ADR 0034.
    """
    catastrophic = pl.DataFrame(
        {
            "model_name": ["xgboost_platt"],
            "stage": ["calibrated"],
            "group_definition": ["community_area"],
            "grain": ["fold_set"],
            "fold_id": [""],
            "metric": ["ece"],
            "measure": ["spread"],
            "value": [0.95],
            "max_group": ["8"],
            "max_value": [0.99],
            "max_group_rows": [4000],
            "min_group": ["24"],
            "min_value": [0.04],
            "min_group_rows": [3800],
            "n_groups_supported": [51],
        }
    )
    check = validate.group_calibration_spread_is_modest(catastrophic)

    assert not check.passed, "a 0.95 ECE spread should be reported"
    assert check.severity == SEVERITY_WARN, "and it must be advisory, not an error"
    assert not validate.has_failures([check]), "an advisory must never fail the run"
    assert "not an implementation error" in check.detail


def test_an_extreme_selection_rate_ratio_is_advisory_too() -> None:
    """Base rates differ 0.220 to 0.566, so unequal selection is expected behaviour."""
    priority = pl.DataFrame(
        {
            "model_name": ["xgboost_platt"],
            "stage": ["calibrated"],
            "group_definition": ["community_area"],
            "grain": ["fold_set"],
            "fold_id": [""],
            "group_value": ["8"],
            "group_status": [GroupStatus.SUPPORTED.value],
            "selection_rate_ratio": [9.5],
            "k_name": ["k_pct_05"],
            "n_rows": [409],
        }
    )
    check = validate.selection_rates_are_proportionate(priority)
    assert not check.passed
    assert check.severity == SEVERITY_WARN
    assert not validate.has_failures([check])


def test_a_run_with_only_advisory_findings_reports_zero_errors() -> None:
    """The whole point, stated as an assertion rather than as a docstring."""
    checks = [
        validate.inputs_were_not_modified({"f": "a" * 64}, {"f": "a" * 64}),
        validate.group_calibration_spread_is_modest(
            pl.DataFrame(
                {
                    "model_name": ["m"],
                    "stage": ["calibrated"],
                    "group_definition": ["community_area"],
                    "grain": ["fold_set"],
                    "fold_id": [""],
                    "metric": ["ece"],
                    "measure": ["spread"],
                    "value": [0.9],
                    "max_group": ["a"],
                    "max_value": [0.95],
                    "max_group_rows": [500],
                    "min_group": ["b"],
                    "min_value": [0.05],
                    "min_group_rows": [500],
                    "n_groups_supported": [2],
                }
            )
        ),
    ]
    assert not validate.has_failures(checks)
    assert len(validate.advisory_findings(checks)) == 1
    report = validate.format_report(checks)
    assert "0 error(s), 1 advisory finding(s)" in report
    assert "does NOT mean Sentinel is" in report
