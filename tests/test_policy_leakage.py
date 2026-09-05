"""The safety wall: every check driven to red on purpose.

A check whose failure path has never been observed is indistinguishable from one that cannot
fire. Component 5 shipped exactly that defect once; Components 9, 11 and 12 each answered it
with a file that breaks the thing the check exists to catch. This is Component 13's.

The defects injected here are the ones that would produce a **plausible-looking but
operationally wrong** recommendation: a queue longer than the city's capacity, a reserve
holding an establishment with a full inspection history, an override with no actor, a rank that
appears twice, a policy that silently lost rows. Every one of those would pass a casual read of
the artifact.

**The last section asserts the opposite of the others.** A reserve that gave up citations, a
reserve that did nothing, a group whose share of the queue moved, and a frontier with no winner
must all stay advisory and keep the run green. The cheapest way to turn a red "this reserve
cost 34 citations" build green is to delete the reserve -- which is a policy decision about how
a city allocates enforcement, and not one a CI runner is entitled to take.
"""

from __future__ import annotations

import polars as pl
import pytest

from sentinel.policy import validate, writer
from sentinel.policy.definitions import (
    BASELINE_POLICY_ID,
    ELIGIBILITY_COLUMN,
    DecisionMechanism,
    DecisionReason,
)
from sentinel.policy.eligibility import ELIGIBLE_FLAG
from sentinel.policy.models import SEVERITY_ERROR, SEVERITY_WARN

CELLS = ["policy_id", "model_name", "fold_set", "fold_id", "k_name"]


# --- minimal-but-valid frames -------------------------------------------------------


def _recommendations() -> pl.DataFrame:
    """Four rows, two selected, one of them through the reserve. Valid in every respect."""
    rows = [
        {
            "policy_id": "coverage_forced_population_share",
            "model_name": "xgboost_platt",
            "fold_set": "quarterly",
            "fold_id": "quarterly-2022Q2",
            "k_name": "k_1_day",
            "k": 2,
            "target_inspection_id": row_id,
            "establishment_id": f"EST-{index}",
            "inspection_date": None,
            "base_score": 0.4,
            "score": score,
            "model_rank": rank,
            "final_policy_rank": policy_rank,
            "is_selected": policy_rank is not None,
            "decision_mechanism": mechanism,
            "decision_reason": reason,
            "coverage_eligible": eligible,
            "secondary_no_history": False,
            "warnings": "none",
            "group_value": "12",
            "group_status": "supported",
            "policy_definition_version": "v1",
        }
        for index, (row_id, score, rank, policy_rank, mechanism, reason, eligible) in enumerate(
            [
                (
                    "T1",
                    0.9,
                    1,
                    1,
                    DecisionMechanism.RISK_PRIORITY,
                    DecisionReason.SELECTED_BY_RISK_RANK,
                    False,
                ),
                (
                    "T4",
                    0.2,
                    4,
                    2,
                    DecisionMechanism.COVERAGE_RESERVE,
                    DecisionReason.SELECTED_BY_COVERAGE_RESERVE,
                    True,
                ),
                (
                    "T2",
                    0.8,
                    2,
                    None,
                    DecisionMechanism.NOT_SELECTED,
                    DecisionReason.NOT_SELECTED_CAPACITY_EXHAUSTED,
                    False,
                ),
                (
                    "T3",
                    0.7,
                    3,
                    None,
                    DecisionMechanism.NOT_SELECTED,
                    DecisionReason.NOT_SELECTED_CAPACITY_EXHAUSTED,
                    False,
                ),
            ]
        )
    ]
    frame = pl.DataFrame(rows, schema=writer.SCHEMAS["inspection_recommendations"])
    return frame.sort(writer.SORT_KEYS["inspection_recommendations"])


def _allocation(**overrides: object) -> pl.DataFrame:
    row: dict[str, object] = {
        "policy_id": "coverage_forced_population_share",
        "model_name": "xgboost_platt",
        "fold_set": "quarterly",
        "fold_id": "quarterly-2022Q2",
        "k_name": "k_1_day",
        "k": 2,
        "n_universe": 4,
        "reserve_mechanism": "forced",
        "reserve_share": 0.10,
        "reserve_target": 1,
        "n_eligible_available": 1,
        "n_eligible_in_risk_top_k": 0,
        "n_risk": 1,
        "n_reserve": 1,
        "n_selected": 2,
        "reserve_inert": False,
        "policy_definition_version": "v1",
    }
    row.update(overrides)
    return pl.DataFrame([row], schema=writer.SCHEMAS["policy_selection_allocation"])


def _features(eligible: bool = True) -> pl.DataFrame:
    return pl.DataFrame(
        {
            ELIGIBILITY_COLUMN: [0 if eligible else 4],
            ELIGIBLE_FLAG: [eligible],
        }
    )


def _assert_red(check: object, severity: str = SEVERITY_ERROR) -> None:
    assert not check.passed  # type: ignore[attr-defined]
    assert check.severity == severity  # type: ignore[attr-defined]


# --- 1. capacity ---------------------------------------------------------------------


def test_the_honest_frames_are_green() -> None:
    """The paired green test. Without it a red test proves only that the check always fires."""
    recommendations, allocation = _recommendations(), _allocation()
    assert validate.selected_counts_equal_capacity(recommendations, allocation).passed
    assert validate.recommendations_cover_the_universe(recommendations, allocation).passed
    assert validate.allocations_are_internally_consistent(allocation).passed
    assert validate.policy_ranks_are_unique_and_contiguous(recommendations, allocation).passed
    assert validate.risk_rows_satisfy_the_risk_contract(recommendations, allocation).passed
    assert validate.reserve_rows_are_eligible(recommendations).passed
    assert validate.every_row_declares_a_valid_mechanism(recommendations).passed
    assert validate.no_establishment_is_selected_twice(recommendations).passed


def test_selecting_more_establishments_than_capacity_turns_the_check_red() -> None:
    """A queue longer than the day would beat every other policy for that reason alone."""
    bad = _recommendations().with_columns(pl.lit(True).alias("is_selected"))
    _assert_red(validate.selected_counts_equal_capacity(bad, _allocation()))


def test_selecting_fewer_establishments_than_capacity_turns_the_check_red() -> None:
    """Under-filling wastes an inspector's day and flatters precision at the same time."""
    bad = _recommendations().with_columns(pl.lit(False).alias("is_selected"))
    _assert_red(validate.selected_counts_equal_capacity(bad, _allocation()))


def test_a_reserve_larger_than_its_declared_allocation_turns_the_check_red() -> None:
    """A policy that quietly overspends its own budget is not the policy it says it is."""
    _assert_red(validate.allocations_are_internally_consistent(_allocation(n_reserve=5)))


def test_an_allocation_whose_parts_do_not_sum_turns_the_check_red() -> None:
    _assert_red(validate.allocations_are_internally_consistent(_allocation(n_risk=7)))


def test_a_reserve_exceeding_the_eligible_rows_that_exist_turns_the_check_red() -> None:
    """Selecting more eligible establishments than the window holds means one was invented."""
    _assert_red(validate.allocations_are_internally_consistent(_allocation(n_eligible_available=0)))


# --- 2. the mechanisms -----------------------------------------------------------------


def test_an_ineligible_establishment_in_the_reserve_turns_the_check_red() -> None:
    """Capacity diverted on a rationale that does not apply -- the abuse a reserve invites."""
    bad = _recommendations().with_columns(
        pl.when(pl.col("decision_mechanism") == DecisionMechanism.COVERAGE_RESERVE)
        .then(pl.lit(False))
        .otherwise(pl.col("coverage_eligible"))
        .alias("coverage_eligible")
    )
    _assert_red(validate.reserve_rows_are_eligible(bad))


def test_a_selected_row_with_no_mechanism_turns_the_check_red() -> None:
    """A recommendation nobody can explain is the failure this component exists to prevent."""
    bad = _recommendations().with_columns(
        pl.when(pl.col("is_selected"))
        .then(pl.lit(DecisionMechanism.NOT_SELECTED.value))
        .otherwise(pl.col("decision_mechanism"))
        .alias("decision_mechanism")
    )
    _assert_red(validate.every_row_declares_a_valid_mechanism(bad))


def test_a_mechanism_carrying_another_mechanisms_reason_turns_the_check_red() -> None:
    """A row that claims the reserve while citing a risk rank is a mislabelled decision."""
    bad = _recommendations().with_columns(
        pl.when(pl.col("decision_mechanism") == DecisionMechanism.COVERAGE_RESERVE)
        .then(pl.lit(DecisionReason.SELECTED_BY_RISK_RANK.value))
        .otherwise(pl.col("decision_reason"))
        .alias("decision_reason")
    )
    _assert_red(validate.every_row_declares_a_valid_mechanism(bad))


def test_an_unknown_mechanism_turns_the_check_red() -> None:
    bad = _recommendations().with_columns(pl.lit("invented_mechanism").alias("decision_mechanism"))
    _assert_red(validate.every_row_declares_a_valid_mechanism(bad))


def test_the_same_establishment_selected_by_both_mechanisms_turns_the_check_red() -> None:
    """The reserve is filled from what risk did not take, so a duplicate means it was not."""
    frame = _recommendations()
    duplicate = frame.filter(pl.col("target_inspection_id") == "T4").with_columns(
        pl.lit(DecisionMechanism.RISK_PRIORITY.value).alias("decision_mechanism"),
        pl.lit(DecisionReason.SELECTED_BY_RISK_RANK.value).alias("decision_reason"),
    )
    bad = pl.concat([frame, duplicate]).sort(writer.SORT_KEYS["inspection_recommendations"])
    _assert_red(validate.no_establishment_is_selected_twice(bad))


def test_a_risk_selection_from_outside_the_risk_prefix_turns_the_check_red() -> None:
    """Swapping a top-ranked establishment for a lower-ranked one, with the count unchanged."""
    bad = _recommendations().with_columns(
        pl.when(pl.col("target_inspection_id") == "T1")
        .then(pl.lit(DecisionMechanism.NOT_SELECTED.value))
        .when(pl.col("target_inspection_id") == "T2")
        .then(pl.lit(DecisionMechanism.RISK_PRIORITY.value))
        .otherwise(pl.col("decision_mechanism"))
        .alias("decision_mechanism")
    )
    _assert_red(validate.risk_rows_satisfy_the_risk_contract(bad, _allocation()))


# --- 3. the ranks ------------------------------------------------------------------------


def test_a_duplicated_policy_rank_turns_the_check_red() -> None:
    """Two establishments cannot be the same appointment."""
    bad = _recommendations().with_columns(
        pl.when(pl.col("is_selected")).then(pl.lit(1)).otherwise(None).alias("final_policy_rank")
    )
    _assert_red(validate.policy_ranks_are_unique_and_contiguous(bad, _allocation()))


def test_a_gap_in_the_policy_ranks_turns_the_check_red() -> None:
    bad = _recommendations().with_columns(
        pl.when(pl.col("final_policy_rank") == 2)
        .then(pl.lit(9))
        .otherwise(pl.col("final_policy_rank"))
        .alias("final_policy_rank")
    )
    _assert_red(validate.policy_ranks_are_unique_and_contiguous(bad, _allocation()))


def test_a_rank_on_an_unselected_row_turns_the_check_red() -> None:
    """A rank is a position in a queue. A row not in the queue does not have one."""
    bad = _recommendations().with_columns(pl.lit(1).alias("final_policy_rank"))
    _assert_red(validate.policy_ranks_are_unique_and_contiguous(bad, _allocation()))


# --- 4. the universe ----------------------------------------------------------------------


def test_a_disappeared_prediction_row_turns_the_check_red() -> None:
    """The most dangerous silent failure here: a shorter queue and a better precision."""
    bad = _recommendations().filter(pl.col("target_inspection_id") != "T3")
    _assert_red(validate.recommendations_cover_the_universe(bad, _allocation()))


def test_an_extra_row_that_was_never_predicted_turns_the_check_red() -> None:
    frame = _recommendations()
    extra = frame.head(1).with_columns(pl.lit("T999").alias("target_inspection_id"))
    bad = pl.concat([frame, extra]).sort(writer.SORT_KEYS["inspection_recommendations"])
    _assert_red(validate.recommendations_cover_the_universe(bad, _allocation()))


def test_a_policy_missing_from_the_comparison_turns_the_check_red() -> None:
    """A policy that quietly drops out changes which policies look non-dominated."""
    comparison = pl.DataFrame(
        {
            "policy_id": ["pure_risk"],
            "model_name": ["xgboost_platt"],
            "fold_set": ["quarterly"],
            "fold_id": ["quarterly-2022Q2"],
            "k_name": ["k_1_day"],
        }
    )
    _assert_red(validate.comparison_covers_every_policy(comparison, _allocation()))


# --- 5. temporal and outcome safety ---------------------------------------------------------


def test_an_eligibility_flag_that_disagrees_with_its_column_turns_the_check_red() -> None:
    """The check that catches a hand-edited flag, or a predicate that drifted from the rule."""
    bad = _features(eligible=True).with_columns(pl.lit(False).alias(ELIGIBLE_FLAG))
    _assert_red(
        validate.eligibility_matches_the_declared_rule(
            bad, column=ELIGIBILITY_COLUMN, flag=ELIGIBLE_FLAG
        )
    )


def test_treating_a_null_history_count_as_eligible_turns_the_check_red() -> None:
    """The ``fill_null(0)`` an edit might reach for, and the reason it must not.

    A null count means the measurement is missing. Reserving capacity for it would reserve
    capacity for rows whose history nobody looked up -- including, one snapshot later, rows a
    join quietly failed to match.
    """
    bad = pl.DataFrame(
        {ELIGIBILITY_COLUMN: [None], ELIGIBLE_FLAG: [True]},
        schema={ELIGIBILITY_COLUMN: pl.Int32, ELIGIBLE_FLAG: pl.Boolean},
    )
    _assert_red(
        validate.eligibility_matches_the_declared_rule(
            bad, column=ELIGIBILITY_COLUMN, flag=ELIGIBLE_FLAG
        )
    )


def test_an_outcome_column_in_a_decision_artifact_turns_the_check_red() -> None:
    """A label in the recommendation table is a policy that could have read the answer."""
    bad = _recommendations().with_columns(pl.lit(1).alias("target"))
    _assert_red(validate.no_outcome_column_reaches_the_policy(bad, _allocation()))


def test_a_warning_input_that_changes_the_queue_turns_the_check_red() -> None:
    """The leak this component is most exposed to and least able to notice afterwards.

    If a Component 12 group number ever reached a ranking decision, the artifact would look
    entirely normal. The only way to see it is to rebuild the queue without the group inputs
    and compare, which is what this check does.
    """
    real = _recommendations()
    signature = [*CELLS, "target_inspection_id", "final_policy_rank", "decision_mechanism"]
    tampered = real.select(signature).with_columns(
        pl.when(pl.col("final_policy_rank") == 1)
        .then(pl.lit(2))
        .otherwise(pl.col("final_policy_rank"))
        .alias("final_policy_rank")
    )
    _assert_red(validate.warnings_do_not_change_the_queue(real, tampered))


def test_the_queue_check_is_green_when_the_two_builds_agree() -> None:
    real = _recommendations()
    signature = [*CELLS, "target_inspection_id", "final_policy_rank", "decision_mechanism"]
    assert validate.warnings_do_not_change_the_queue(real, real.select(signature)).passed


# --- 6. the audit interaction ------------------------------------------------------------


def test_an_unsupported_group_relabelled_as_supported_turns_the_check_red() -> None:
    """Component 12's discipline: an unmeasurable group is a row with a reason, not an absence.

    The easiest way for this component to produce a flattering group table would be to promote
    the groups nobody could measure.
    """
    audit = pl.DataFrame(
        {"group_value": ["7"], "group_status": ["supported"]},
    )
    _assert_red(validate.unsupported_groups_are_preserved(audit, {"7": "insufficient_support"}))


def test_an_unsupported_group_carried_through_honestly_is_green() -> None:
    audit = pl.DataFrame({"group_value": ["7"], "group_status": ["insufficient_support"]})
    assert validate.unsupported_groups_are_preserved(audit, {"7": "insufficient_support"}).passed


# --- 7. the human layer ------------------------------------------------------------------


def _override_log(**overrides: object) -> pl.DataFrame:
    row: dict[str, object] = {
        "override_id": "OV-0001",
        "policy_id": BASELINE_POLICY_ID,
        "fold_set": "quarterly",
        "fold_id": "quarterly-2022Q2",
        "k_name": "k_1_day",
        "target_inspection_id": "T4",
        "action": "force_include",
        "reason_code": "outbreak",
        "actor": "inspector.smith",
        "decided_at": "2026-08-26T09:00:00Z",
        "original_is_selected": False,
        "original_mechanism": "not_selected",
        "original_reason": "not_selected_capacity_exhausted",
        "original_policy_rank": None,
        "final_is_selected": True,
        "displaced_target_inspection_id": "T1",
        "outcome": "applied",
        "policy_definition_version": "v1",
    }
    row.update(overrides)
    return pl.DataFrame([row], schema=writer.SCHEMAS["policy_override_log"])


@pytest.mark.parametrize("field", ["actor", "reason_code", "decided_at", "override_id"])
def test_an_unattributed_override_turns_the_check_red(field: str) -> None:
    """An override with no actor is an anonymous change to who gets inspected."""
    _assert_red(validate.overrides_are_fully_attributed(_override_log(**{field: "  "})))


def test_a_fully_attributed_override_is_green() -> None:
    assert validate.overrides_are_fully_attributed(_override_log()).passed


def test_an_override_that_rewrote_the_recommendation_turns_the_check_red() -> None:
    """The layer separation. The original recommendation must stay recoverable."""
    rewritten = _recommendations().with_columns(
        pl.when(pl.col("target_inspection_id") == "T4")
        .then(pl.lit(False))
        .otherwise(pl.col("is_selected"))
        .alias("is_selected")
    )
    _assert_red(
        validate.overrides_left_the_deterministic_queue_intact(
            rewritten,
            _override_log(policy_id="coverage_forced_population_share", target_inspection_id="T4"),
        )
    )


def test_an_override_recorded_beside_an_untouched_recommendation_is_green() -> None:
    """The correct behaviour: the queue is written unchanged and the decision sits next to it."""
    assert validate.overrides_left_the_deterministic_queue_intact(
        _recommendations(),
        _override_log(policy_id="coverage_forced_population_share", target_inspection_id="T4"),
    ).passed


# --- 8. the observer guarantee ---------------------------------------------------------


def test_a_changed_input_checksum_turns_the_check_red() -> None:
    """The run reads nine closed components and must not have touched any of them."""
    _assert_red(validate.inputs_were_not_modified({"features": "aaa"}, {"features": "bbb"}))


def test_a_vanished_input_turns_the_check_red() -> None:
    _assert_red(validate.inputs_were_not_modified({"features": "aaa"}, {}))


def test_unchanged_inputs_are_green() -> None:
    assert validate.inputs_were_not_modified({"features": "aaa"}, {"features": "aaa"}).passed


# --- 9. the sort contract ----------------------------------------------------------------


def test_an_unsorted_table_turns_the_check_red() -> None:
    """Byte-comparison between two runs is the whole reproducibility claim."""
    frame = _recommendations().reverse()
    _assert_red(
        validate.tables_are_deterministically_sorted(
            {"inspection_recommendations": frame}, writer.SORT_KEYS
        )
    )


def test_a_duplicate_sort_key_turns_the_check_red() -> None:
    frame = _recommendations()
    bad = pl.concat([frame, frame.head(1)]).sort(writer.SORT_KEYS["inspection_recommendations"])
    _assert_red(
        validate.tables_are_deterministically_sorted(
            {"inspection_recommendations": bad}, writer.SORT_KEYS
        )
    )


def test_a_grid_that_disagrees_with_the_frozen_definitions_turns_the_check_red() -> None:
    """A run whose artifact described a different grid than the code applied would be wrong."""
    bad = pl.DataFrame(
        {
            "policy_id": ["pure_risk"],
            "reserve_mechanism": ["forced"],
            "reserve_share": [0.5],
        }
    )
    _assert_red(validate.configurations_match_the_frozen_grid(bad))


# --- 10. THE OPPOSITE ASSERTION: a measured cost must never fail a build ------------------


def test_a_reserve_that_gave_up_citations_is_advisory_and_never_an_error() -> None:
    """The most important test in this file.

    A reserve that costs citations is the finding, not the bug. If this were an error the only
    green policy would be the one that reserves nothing -- which is a decision about how a city
    allocates enforcement, taken by a CI runner.
    """
    comparison = pl.DataFrame(
        {
            "policy_id": ["coverage_forced_double_share"],
            "k_name": ["k_1_week"],
            "delta_positives": [-34.0],
        }
    )
    check = validate.coverage_is_not_free(comparison)
    _assert_red(check, SEVERITY_WARN)
    assert not validate.has_failures([check])


def test_an_inert_reserve_is_advisory_and_never_an_error() -> None:
    """Inertness is this component's headline result: the risk queue already over-covers."""
    check = validate.reserve_is_not_inert(_allocation(reserve_inert=True, n_reserve=0))
    _assert_red(check, SEVERITY_WARN)
    assert not validate.has_failures([check])


def test_a_large_group_representation_shift_is_advisory_and_never_an_error() -> None:
    """A coverage policy changes who is inspected by construction. That is not a defect."""
    audit = pl.DataFrame(
        {
            "policy_id": [BASELINE_POLICY_ID, "coverage_forced_double_share"],
            "group_value": ["12", "12"],
            "fold_set": ["quarterly", "quarterly"],
            "fold_id": ["quarterly-2022Q2", "quarterly-2022Q2"],
            "k_name": ["k_1_day", "k_1_day"],
            "selected_share": [0.10, 0.90],
            "group_status": ["supported", "supported"],
        }
    )
    check = validate.group_representation_is_stable(audit)
    _assert_red(check, SEVERITY_WARN)
    assert not validate.has_failures([check])


def test_naming_no_policy_winner_is_advisory_and_never_an_error() -> None:
    """ "The data does not determine the correct policy" is a result, not a failure."""
    check = validate.a_winner_was_determined(None, "the data does not determine the policy")
    _assert_red(check, SEVERITY_WARN)
    assert not validate.has_failures([check])


def test_a_report_of_only_advisories_still_exits_green() -> None:
    checks = [
        validate.coverage_is_not_free(
            pl.DataFrame({"policy_id": ["p"], "k_name": ["k_1_day"], "delta_positives": [-99.0]})
        ),
        validate.a_winner_was_determined(None, "no winner"),
    ]
    assert not validate.has_failures(checks)
    assert len(validate.advisory_findings(checks)) == 2
    rows = validate.advisory_rows(checks, definition_version="v1")
    assert all(row["severity"] == SEVERITY_WARN for row in rows)


def test_the_report_says_a_green_run_does_not_mean_a_good_policy() -> None:
    """The sentence has to travel with the output, in these words."""
    text = validate.format_report([validate.a_winner_was_determined("p", "n/a")])
    assert "the policy was applied correctly" in text
    assert "It does not mean the policy is the right one." in text
