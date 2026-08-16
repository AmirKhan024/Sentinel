"""Real target cases harvested from the Chicago snapshot.

Every value was copied verbatim from
``food_inspections_20260816T070911Z.parquet`` (sha256 ``7d3c4069...ad38``) while
inspecting the first full target build. Real ``inspection_id`` values are kept so
a case can be traced back to the source row.

Literal Python rather than a data file for the same reason as
``real_cases.py``: ``*.csv`` and ``*.jsonl`` are gitignored project-wide, so a
fixture file would silently fail to commit.

Each case carries a ``why`` that is surfaced in the assertion message, so a
future failure explains which real regulatory situation just regressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetCase:
    """One real inspection and the label it must receive."""

    case_id: str
    why: str
    inspection_id: str
    inspection_date: str
    inspection_type: str
    results: str
    violations: str | None
    expected_status: str
    expected_target: int | None
    tags: tuple[str, ...] = field(default=())


TARGET_CASES: list[TargetCase] = [
    # --- positives --------------------------------------------------------
    TargetCase(
        case_id="priority_foundation_without_a_citation",
        why=(
            "A Priority Foundation violation that was found but not cited. 21,281 code-era "
            "entries carry a marker with no 7-38 municipal code; requiring one would turn all "
            "of them into false negatives. Note the result is Pass -- the label comes from the "
            "violation text, never from the result summary."
        ),
        inspection_id="2184575",
        inspection_date="2018-07-13T00:00:00.000",
        inspection_type="Canvass",
        results="Pass",
        violations=(
            "10. ADEQUATE HANDWASHING SINKS PROPERLY SUPPLIED AND ACCESSIBLE - Comments: "
            "OBSERVED NO EXPOSED HANDWASHING SINK IMEDIATLY ADJACENT TO WAREWASHING AREA. "
            "INSTALL AN EXPOSED HAND SINK WITH HOT AND COLD RUNNING WATER SOAP AND PAPER "
            "TOWELS. PRIORITY FOUNDATION 7-38-030 (C) NO CITATION ISSUED."
        ),
        expected_status="eligible",
        expected_target=1,
        tags=("priority_foundation", "no_citation", "pass_with_priority"),
    ),
    TargetCase(
        case_id="plain_priority_rodent_infestation",
        why=(
            "A plain Priority violation on item 38. The same numbered item is Core in other "
            "inspections, which is why the violation number is never used to classify "
            "severity -- item 38 is 45.5% Priority Foundation, 2.2% Priority and 52.3% "
            "unlabelled across the code era."
        ),
        inspection_id="2293499",
        inspection_date="2019-06-13T00:00:00.000",
        inspection_type="Canvass",
        results="Fail",
        violations=(
            "38. INSECTS, RODENTS, & ANIMALS NOT PRESENT - Comments: OBSERVED EVIDENCE OF "
            "RODENT INFESTATION THROUGHOUT THE PREMISES. OBSERVED OVER 500 MOUSE DROPPINGS. "
            "PRIORITY 7-38-020(A), CITATION ISSUED"
        ),
        expected_status="eligible",
        expected_target=1,
        tags=("priority", "citation"),
    ),
    TargetCase(
        case_id="pass_with_conditions_is_not_a_pass",
        why=(
            "Pass w/ Conditions behaves like Fail, not like Pass: priority violations are "
            "present in 97.6% of them among canvasses against 0.5% of plain Passes. A "
            "results=='Fail' target would have labelled 16,387 of these negative."
        ),
        inspection_id="2015563",
        inspection_date="2019-03-08T00:00:00.000",
        inspection_type="Canvass",
        results="Pass w/ Conditions",
        violations=(
            "3. MANAGEMENT, FOOD EMPLOYEE AND CONDITIONAL EMPLOYEE; KNOWLEDGE, "
            "RESPONSIBILITIES AND REPORTING - Comments: NO EMPLOYEE HEALTH POLICY "
            "ON-PREMISES. PRIORITY FOUNDATION 7-38-010."
        ),
        expected_status="eligible",
        expected_target=1,
        tags=("pass_with_conditions",),
    ),
    # --- negatives --------------------------------------------------------
    TargetCase(
        case_id="grace_period_boilerplate_is_not_a_priority_violation",
        why=(
            "The only occurrence of 'priority' is the standard 90-day grace-period notice "
            "appended to an unrelated allergen-training finding. Naive substring matching "
            "calls this positive; it is boilerplate policy text, not a classification of "
            "this violation. One of 10 inspection labels the narrative exclusion changes."
        ),
        inspection_id="2182182",
        inspection_date="2018-07-03T00:00:00.000",
        inspection_type="Canvass",
        results="Pass",
        violations=(
            "58. ALLERGEN TRAINING AS REQUIRED - Comments: Observed food allergen "
            "requirements not met. Instructed manager to provided. A 90 day grace period "
            "was given for all new priority and priority foundation violations. Citations "
            "will be issued on the next inspection after the 90 day grace period."
        ),
        expected_status="eligible",
        expected_target=0,
        tags=("narrative_exclusion", "grace_period", "false_positive"),
    ),
    TargetCase(
        case_id="core_only_violations_are_negative",
        why=(
            "Dirty floors are a real violation but not a Priority one. A target keyed on "
            "'any violation' would call this positive and predict paperwork rather than "
            "food-safety risk; item 55 alone accounts for 78,983 code-era entries, none "
            "priority."
        ),
        inspection_id="2015568",
        inspection_date="2019-03-08T00:00:00.000",
        inspection_type="Canvass",
        results="Pass",
        violations=(
            "55. PHYSICAL FACILITIES INSTALLED, MAINTAINED & CLEAN - Comments: FLOORS IN "
            "THE CORNERS, UNDER/AROUND SINKS, COOLERS/FREEZERS AND COOKING EQUIPMENT, IN "
            "THE STORAGE AREA AND BOILER ROOM WITH DIRT AND DEBRIS. INSTRUCTED TO CLEAN."
        ),
        expected_status="eligible",
        expected_target=0,
        tags=("core_only",),
    ),
    TargetCase(
        case_id="clean_pass_with_no_violation_text_is_a_true_zero",
        why=(
            "18.8% of passing canvasses record no violation text at all. For a passing "
            "inspection that is the expected encoding of 'nothing found', so it is a "
            "genuine negative rather than missing data."
        ),
        inspection_id="2015573",
        inspection_date="2019-03-11T00:00:00.000",
        inspection_type="Canvass",
        results="Pass",
        violations=None,
        expected_status="eligible",
        expected_target=0,
        tags=("true_zero", "missing_violations"),
    ),
    # --- exclusions -------------------------------------------------------
    TargetCase(
        case_id="fail_with_no_violation_text_is_unknown",
        why=(
            "Self-contradictory: the result asserts violations were found, the text records "
            "none. 79 rows in the snapshot. Guessing negative would inject false negatives "
            "into exactly the high-risk stratum; guessing positive would invent evidence."
        ),
        inspection_id="2233158",
        inspection_date="2018-11-14T00:00:00.000",
        inspection_type="Canvass",
        results="Fail",
        violations=None,
        expected_status="unknown_violations",
        expected_target=None,
        tags=("unknown", "missing_violations"),
    ),
    TargetCase(
        case_id="pre_code_era_canvass_is_ineligible",
        why=(
            "Chicago used Critical/Serious before 2018-07-01; the words Priority and "
            "Priority Foundation do not occur anywhere in that era. This inspection found a "
            "serious rodent violation, but the target being predicted did not exist yet, so "
            "the row is ineligible rather than negative."
        ),
        inspection_id="1072219",
        inspection_date="2012-10-29T00:00:00.000",
        inspection_type="Canvass",
        results="Fail",
        violations=(
            "18. NO EVIDENCE OF RODENT OR INSECT OUTER OPENINGS PROTECTED/RODENT PROOFED - "
            "Comments: NOTED EVIDENCE OF RODENT ACTIVITY, 15 MICE DROPPINGS IN KITCHEN "
            "AREA. SERIOUS VIOLATION 7-38-020, CITATION ISSUED."
        ),
        expected_status="ineligible_era",
        expected_target=None,
        tags=("era", "old_terminology"),
    ),
    TargetCase(
        case_id="out_of_business_is_ineligible_not_negative",
        why=(
            "No inspection took place; 99.9% of Out of Business records have null "
            "violations. Labelling it negative would teach that a closed establishment is a "
            "clean one. It is also not terminal: 24.9% of OOB records are followed by "
            "another inspection at the same premises, median 273 days later, because "
            "Component 2 tracks places rather than businesses."
        ),
        inspection_id="2237893",
        inspection_date="2019-01-15T00:00:00.000",
        inspection_type="Canvass",
        results="Out of Business",
        violations=None,
        expected_status="ineligible_result",
        expected_target=None,
        tags=("out_of_business", "exclusion"),
    ),
    TargetCase(
        case_id="complaint_inspection_is_ineligible",
        why=(
            "A complaint inspection happens because somebody complained, so its outcome is "
            "conditioned on a prior signal that is unavailable at scheduling time. Only "
            "routine canvasses are eligible."
        ),
        inspection_id="2293500",
        inspection_date="2019-06-13T00:00:00.000",
        inspection_type="Complaint",
        results="Fail",
        violations=(
            "3. MANAGEMENT - Comments: NO EMPLOYEE HEALTH POLICY. PRIORITY FOUNDATION 7-38-010."
        ),
        expected_status="ineligible_type",
        expected_target=None,
        tags=("inspection_type", "exclusion"),
    ),
    TargetCase(
        case_id="canvass_reinspection_is_ineligible",
        why=(
            "A canvass re-inspection exists only because an earlier inspection failed. "
            "Including these 16,998 code-era rows would condition the target on past failure "
            "and inflate the base rate."
        ),
        inspection_id="2293501",
        inspection_date="2019-06-20T00:00:00.000",
        inspection_type="Canvass Re-Inspection",
        results="Pass",
        violations="55. PHYSICAL FACILITIES - Comments: FLOORS CLEANED.",
        expected_status="ineligible_type",
        expected_target=None,
        tags=("inspection_type", "reinspection", "exclusion"),
    ),
    TargetCase(
        case_id="era_boundary_is_inclusive_of_2018_07_01",
        why=(
            "The cutover is clean: June 2018 has 0 rows using the new terminology and 415 "
            "using the old; July 2018 has 761 and 0. The first eligible day is 2018-07-01 "
            "itself, so an off-by-one here would silently drop or admit a month."
        ),
        inspection_id="2181999",
        inspection_date="2018-07-01T00:00:00.000",
        inspection_type="Canvass",
        results="Fail",
        violations="3. MANAGEMENT - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010.",
        expected_status="eligible",
        expected_target=1,
        tags=("era", "boundary"),
    ),
]
