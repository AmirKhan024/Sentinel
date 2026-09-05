"""The frozen scheduling contracts, and the import-time guard that keeps them consistent.

The guard is the point of this file. Every constant in ``scheduling/definitions.py`` has a way
of drifting during an edit and every one of them would fail silently and plausibly -- a grid
with two defaults, a status with no reason, an adjustment verb that collides with an override
verb. Each produces a run that finishes green and schedules the wrong establishments, or an
audit trail in which two different human decisions are indistinguishable.
"""

from __future__ import annotations

import pytest

from sentinel.policy.definitions import K_LEVELS as POLICY_K_LEVELS
from sentinel.policy.definitions import OverrideAction
from sentinel.scheduling import definitions as d


class TestHorizonRule:
    """``ceil(k / median)`` -- the capacity rule read backwards, not a new constant."""

    @pytest.mark.parametrize(
        ("k", "median", "expected"),
        [
            (28, 28, 1),
            (140, 28, 5),
            (82, 28, 3),
            (164, 28, 6),
            (16, 28, 1),
            (884, 22, 41),
            (22, 22, 1),
            (110, 22, 5),
            (1, 1, 1),
        ],
    )
    def test_reproduces_the_measured_horizons(self, k: int, median: int, expected: int) -> None:
        assert d.horizon_days(k, median) == expected

    def test_a_day_denominated_cutoff_spans_its_own_denomination(self) -> None:
        """``k_1_day`` is one day and ``k_1_week`` is five, for every plausible rate.

        This is why the rule is one line rather than a table of special cases: the two cutoffs
        that already carry a duration in their names fall out of the arithmetic.
        """
        for median in range(1, 60):
            assert d.horizon_days(median, median) == 1
            assert d.horizon_days(median * 5, median) == 5

    def test_the_exact_multiple_boundary_does_not_round_up(self) -> None:
        """Integer arithmetic, so a float representation cannot move the boundary."""
        assert d.horizon_days(100, 10) == 10
        assert d.horizon_days(101, 10) == 11
        assert d.horizon_days(99, 10) == 10

    @pytest.mark.parametrize(("k", "median"), [(0, 5), (-1, 5), (5, 0), (5, -3)])
    def test_refuses_a_capacity_or_rate_below_one(self, k: int, median: int) -> None:
        with pytest.raises(d.SchedulingDefinitionError):
            d.horizon_days(k, median)


class TestTheConfigurationGrid:
    def test_exactly_one_configuration_is_the_default(self) -> None:
        assert sum(1 for spec in d.CONFIG_GRID if spec.is_default) == 1

    def test_the_default_is_measured_and_not_a_scenario(self) -> None:
        """The observed calendar is the default because it describes days that happened.

        Defaulting to the scenario would make every headline number describe an assumption --
        and at two of five cutoffs that assumption is a tautology.
        """
        default = next(spec for spec in d.CONFIG_GRID if spec.is_default)
        assert default.capacity_mode is d.CapacityMode.OBSERVED_CALENDAR
        assert not default.is_scenario

    def test_the_flat_median_configuration_is_labelled_a_scenario(self) -> None:
        flat = next(
            spec for spec in d.CONFIG_GRID if spec.capacity_mode is d.CapacityMode.FLAT_MEDIAN
        )
        assert flat.is_scenario

    def test_every_capacity_mode_has_a_configuration(self) -> None:
        assert {spec.capacity_mode for spec in d.CONFIG_GRID} == set(d.CapacityMode)

    def test_every_configuration_states_why_it_exists(self) -> None:
        assert all(spec.rationale for spec in d.CONFIG_GRID)

    def test_config_for_returns_the_frozen_spec(self) -> None:
        spec = d.config_for("strict_priority__observed_calendar")
        assert spec.capacity_mode is d.CapacityMode.OBSERVED_CALENDAR

    def test_config_for_names_the_known_ids_when_asked_for_a_stranger(self) -> None:
        with pytest.raises(d.SchedulingDefinitionError, match="Known:"):
            d.config_for("strict_priority__wishful_thinking")


class TestTheStrategyGrid:
    def test_there_is_exactly_one_strategy_and_it_preserves_priority(self) -> None:
        """One strategy is a finding, not an omission.

        A constraint-aware strategy needs a constraint, and profile 7 is the inventory showing
        this dataset has none -- no closure calendar, no deadline, no availability window.
        """
        assert len(d.STRATEGY_GRID) == 1
        assert d.STRATEGY_GRID[0].preserves_priority_exactly

    def test_the_absence_of_a_second_strategy_is_recorded(self) -> None:
        assert "no constraint-aware strategy" in d.NO_CONSTRAINT_AWARE_STRATEGY

    def test_no_solver_is_recorded_with_its_argument(self) -> None:
        """The pyproject promise was that a dependency arrives with the component needing one.

        It is kept by checking rather than assuming, so the reason travels in a constant that
        goes into every manifest.
        """
        assert "no solver" in d.NO_SOLVER.lower()
        assert "closed form" in d.NO_SOLVER

    def test_the_allocation_is_described_as_what_it_is(self) -> None:
        assert "greedy" in d.ALLOCATION_CLAIM
        assert "Not optimal" in d.ALLOCATION_CLAIM


class TestTheControlledVocabularies:
    def test_every_reason_belongs_to_a_status(self) -> None:
        declared = {reason for reasons in d.STATUS_REASONS.values() for reason in reasons}
        assert set(d.ScheduleReason) == declared

    def test_every_status_declares_at_least_one_reason(self) -> None:
        assert set(d.STATUS_REASONS) == set(d.ScheduleStatus)
        assert all(reasons for reasons in d.STATUS_REASONS.values())

    def test_a_deferred_row_still_occupies_a_slot(self) -> None:
        """A deferral moves an inspection; it does not remove it.

        A capacity check that ignored deferrals would let a day be overbooked by exactly the
        rows somebody moved onto it.
        """
        assert d.ScheduleStatus.DEFERRED in d.OCCUPYING_STATUSES
        assert d.ScheduleStatus.SCHEDULED in d.OCCUPYING_STATUSES
        assert d.ScheduleStatus.BACKLOG not in d.OCCUPYING_STATUSES
        assert d.ScheduleStatus.CANCELLED not in d.OCCUPYING_STATUSES

    def test_recommended_is_not_a_schedule_status(self) -> None:
        """It is Component 13's ``is_selected``; a second home would give one fact two owners."""
        assert "recommended" not in {str(s) for s in d.ScheduleStatus}

    def test_completed_is_not_a_schedule_status(self) -> None:
        """It is an execution fact.

        A plan column that execution writes into is precisely the retroactive edit the temporal
        boundary exists to prevent.
        """
        assert "completed" not in {str(s) for s in d.ScheduleStatus}

    def test_constraint_adjusted_is_deliberately_absent(self) -> None:
        """No real operational constraint exists, so no run could emit it.

        A reason code no code path reaches is indistinguishable from one that is broken.
        """
        assert "constraint_adjusted" not in {str(r) for r in d.ScheduleReason}

    def test_the_inversion_vocabulary_carries_an_explicit_none(self) -> None:
        """A token, not a null.

        An empty cell is ambiguous between "no inversion" and "inversions were not computed",
        and only one of those is a statement about the schedule.
        """
        assert d.InversionReason.NONE == "none"

    def test_adjustment_verbs_are_disjoint_from_override_verbs(self) -> None:
        """The mechanical form of "an override and an adjustment are different things"."""
        assert not {str(a) for a in d.AdjustmentAction} & {str(o) for o in OverrideAction}

    def test_the_two_external_contracts_have_distinct_id_fields(self) -> None:
        assert d.ADJUSTMENT_REQUIRED_FIELDS[0] == "adjustment_id"
        assert d.EXECUTION_REQUIRED_FIELDS[0] == "execution_id"

    @pytest.mark.parametrize("field", ["actor", "reason_code"])
    def test_both_contracts_require_attribution(self, field: str) -> None:
        """An external change with no actor is an anonymous change to who gets inspected when."""
        assert field in d.ADJUSTMENT_REQUIRED_FIELDS
        assert field in d.EXECUTION_REQUIRED_FIELDS

    def test_only_a_not_performed_report_triggers_a_replan(self) -> None:
        assert frozenset({d.ExecutionStatus.NOT_PERFORMED}) == d.REPLAN_TRIGGERING_STATUSES

    def test_no_execution_record_is_derived_and_not_suppliable(self) -> None:
        """ "We do not know" is a summary category, never something a person can file."""
        assert d.NO_EXECUTION_RECORD not in {str(s) for s in d.ExecutionStatus}

    def test_every_capacity_mode_declares_its_source(self) -> None:
        assert set(d.CAPACITY_SOURCES) == set(d.CapacityMode)

    def test_an_adjustment_is_a_planning_run_trigger(self) -> None:
        assert "scheduling_adjustment" in d.REPLAN_TRIGGERS
        assert "original_plan" in d.REPLAN_TRIGGERS


class TestTheInheritedContracts:
    def test_the_capacity_levels_are_component_13s_by_identity(self) -> None:
        """Restating them would create a second list that could silently disagree."""
        assert d.K_LEVELS is POLICY_K_LEVELS

    def test_the_primary_capacity_level_is_in_the_grid(self) -> None:
        assert d.PRIMARY_K_LEVEL in d.K_LEVELS


class TestTheBoundaryLists:
    @pytest.mark.parametrize("name", ["DOES_NOT_ESTABLISH", "BLOCKED", "INHERITED_LIMITATIONS"])
    def test_the_boundary_travels_with_the_artifact(self, name: str) -> None:
        """Each goes into every manifest; an empty list silently drops the boundary."""
        assert getattr(d, name)

    def test_routing_is_refused_in_the_blocked_list(self) -> None:
        assert any("route optimisation" in line for line in d.BLOCKED)

    def test_re_ranking_is_refused_in_the_blocked_list(self) -> None:
        assert any("re-ranking" in line for line in d.BLOCKED)

    def test_raising_capacity_is_refused_in_the_blocked_list(self) -> None:
        assert any("raising capacity" in line for line in d.BLOCKED)

    def test_the_semantics_name_what_is_not_done(self) -> None:
        assert "not geographic route optimisation" in d.SCHEDULING_SEMANTICS

    def test_the_scenario_claim_says_the_scenario_is_tautological(self) -> None:
        assert "k_1_day" in d.CAPACITY_MODE_SCENARIO_CLAIM

    def test_a_green_run_is_scoped(self) -> None:
        assert "does not mean the city has enough capacity" in d.GREEN_RUN_MEANS

    def test_the_determinism_claim_excludes_human_input(self) -> None:
        assert "external human" in d.DETERMINISM_SCOPE

    def test_the_three_human_layers_are_named_separately(self) -> None:
        for phrase in ("recommendation override", "scheduling adjustment", "execution deviation"):
            assert phrase in d.THREE_HUMAN_LAYERS


class TestTheGuard:
    """The guard must reject each way the constants can drift. Each case is restored after."""

    def test_a_second_default_configuration_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doubled = tuple(
            d.ScheduleConfigSpec(
                schedule_config_id=spec.schedule_config_id,
                strategy_id=spec.strategy_id,
                capacity_mode=spec.capacity_mode,
                is_scenario=spec.is_scenario,
                is_default=True,
                rationale=spec.rationale,
            )
            for spec in d.CONFIG_GRID
        )
        monkeypatch.setattr(d, "CONFIG_GRID", doubled)
        with pytest.raises(d.SchedulingDefinitionError, match="exactly one"):
            d._guard_registry()

    def test_a_scenario_default_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flipped = tuple(
            d.ScheduleConfigSpec(
                schedule_config_id=spec.schedule_config_id,
                strategy_id=spec.strategy_id,
                capacity_mode=spec.capacity_mode,
                is_scenario=spec.is_scenario,
                is_default=spec.capacity_mode is d.CapacityMode.FLAT_MEDIAN,
                rationale=spec.rationale,
            )
            for spec in d.CONFIG_GRID
        )
        monkeypatch.setattr(d, "CONFIG_GRID", flipped)
        with pytest.raises(d.SchedulingDefinitionError):
            d._guard_registry()

    def test_an_orphaned_reason_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        trimmed = {
            status: frozenset(r for r in reasons if r != d.ScheduleReason.CANCELLED_IN_FIELD)
            for status, reasons in d.STATUS_REASONS.items()
        }
        monkeypatch.setattr(d, "STATUS_REASONS", trimmed)
        with pytest.raises(d.SchedulingDefinitionError, match="belong to no status"):
            d._guard_registry()

    def test_a_status_with_no_reason_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emptied = dict(d.STATUS_REASONS)
        emptied[d.ScheduleStatus.CANCELLED] = frozenset()
        monkeypatch.setattr(d, "STATUS_REASONS", emptied)
        with pytest.raises(d.SchedulingDefinitionError):
            d._guard_registry()

    def test_a_verb_colliding_with_an_override_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scheduling verb that is also an override verb makes the two layers confusable.

        Stood in for with a bare iterable rather than a real enum: the guard only ever iterates
        the vocabulary, and an enum cannot be subclassed once it has members.
        """
        colliding = type("Verbs", (), {"__iter__": lambda _: iter(["force_include"])})()
        monkeypatch.setattr(d, "AdjustmentAction", colliding)
        with pytest.raises(d.SchedulingDefinitionError, match="collide"):
            d._guard_registry()

    def test_an_empty_boundary_list_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "BLOCKED", ())
        with pytest.raises(d.SchedulingDefinitionError, match="empty"):
            d._guard_registry()

    def test_a_zero_advisory_threshold_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A threshold of zero fires on every run and carries no information."""
        monkeypatch.setattr(d, "ADVISORY_BACKLOG_ROWS", 0)
        with pytest.raises(d.SchedulingDefinitionError, match="at least 1"):
            d._guard_registry()

    def test_an_out_of_range_opening_day_threshold_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(d, "ADVISORY_LOW_VOLUME_OPENING_DAY", 1.5)
        with pytest.raises(d.SchedulingDefinitionError, match="fraction"):
            d._guard_registry()

    def test_the_shipped_constants_pass_their_own_guard(self) -> None:
        d._guard_registry()
