"""Frozen contracts for Component 13. Every policy constant lives here and nowhere else.

**The separation this module exists to hold.** Components 6 to 9 estimate the probability that
an establishment will be cited. This component decides what to do about it. A probability is
not an action, and the step between them is not arithmetic -- it is a choice about capacity,
about who gets served when there is not enough of it, and about which of those choices a
department is willing to defend. Sorting by score and taking the first *k* rows is that choice
made silently. Everything below is the same choice made out loud.

**Nothing here is a model.** No score is produced, adjusted, reweighted or recalibrated. The
policy layer reorders and allocates; it never edits an estimate. A reader who wants to know
why an establishment scored 0.62 must look at Components 6 to 9 and 11. A reader who wants to
know why it was inspected on Tuesday looks here. That the two questions have two different
answers, in two different places, is the point of the component.

**Every number in this file came from a measurement.** ``scripts/profile_policy.py`` ran
first, over the frozen artifacts, and ``docs/analysis/policy_findings.md`` holds its output.
The eligibility column, the reserve shares, the model-selection axes and the advisory
thresholds are all set from that output. Component 9 set three thresholds from expectation and
had to correct all three; a policy constant chosen the same way would be worse, because a
policy constant decides who gets inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sentinel.fairness.definitions import K_LEVELS as AUDIT_K_LEVELS

#: Bumped whenever the eligibility contract, the policy grid, the allocation semantics or the
#: selection rule change in a way that makes two runs incomparable.
POLICY_DEFINITION_VERSION = "v1"


class PolicyDefinitionError(ValueError):
    """Raised when the frozen policy contracts contradict each other."""


# --- 1. the eligibility contract ---------------------------------------------

#: The single column that decides coverage eligibility. Zero means the establishment has no
#: canvass on or after the 2018-07-01 code-era boundary (ADR 0009).
#:
#: Chosen from profile 1 over four candidates, and chosen because it is the *cause* of the
#: ranking difficulty rather than a correlate of it. This is the exact condition under which
#: ``prior_canvass_priority_count``, ``prior_canvass_priority_foundation_count``,
#: ``prior_canvass_priority_rate`` and ``priority_at_last_canvass`` are NULL -- the four
#: features that encode the outcome being predicted. When it is zero the model is not making a
#: weak judgement about the establishment; it is making no judgement, because the evidence it
#: would use does not exist.
ELIGIBILITY_COLUMN = "prior_canvass_count_code_era"

#: Stated as a sentence so no reader has to reconstruct it from an expression. A null is never
#: eligible: every candidate column carries ``NullRule.NEVER`` in Component 4, so a zero is a
#: real observation of no history, and a null would mean the count itself is missing. Treating
#: those two as the same thing would admit rows about which nothing at all is known.
ELIGIBILITY_RULE = (
    "coverage_eligible <=> prior_canvass_count_code_era == 0; a null is never eligible"
)

#: Reported on every row, and deliberately **not** a gate. Zero here means no inspection of any
#: type was ever recorded -- the strictest and most intuitive reading of "no history". Profile
#: 1 measured 401 of 57,727 rows, 0.69%, which at a day of real capacity is a reserve of zero
#: or one slot. A mechanism that cannot be measured cannot be defended, so this names a
#: population rather than allocating to one.
SECONDARY_FLAG_COLUMN = "prior_inspection_count_any_type"

#: The measured share of the quarterly test windows that is coverage-eligible: 3,410 of 32,696
#: rows (profile 2). Every reserve share in the grid is defined against this number, which is
#: the only reason a value near 0.10 appears anywhere in this component. Recorded rather than
#: recomputed at run time so a run over a different snapshot reports the drift instead of
#: silently moving the grid under the reader.
ELIGIBLE_POPULATION_SHARE = 0.1043

#: Columns that may never touch eligibility, ranking or allocation, checked by the validator
#: rather than trusted. ``target`` is the label; ``target_status`` is its provenance. An
#: eligibility rule that reads either would be deciding who to inspect using the answer.
FORBIDDEN_POLICY_COLUMNS: tuple[str, ...] = ("target", "target_status")

#: Why eligibility is a per-row fact from Component 4 and never a group label from Component
#: 12. Profile 8 measured the overlap -- 456 of 14,162 eligible rows sit in the ``__UNKNOWN__``
#: community area, and 66% of that group is eligible against 24% overall -- and the overlap is
#: exactly why the distinction has to be stated. See ADR 0038.
ELIGIBILITY_IS_NOT_GEOGRAPHY = (
    "coverage eligibility is missing inspection history, measured per row by Component 4. It "
    "is never a geographic group, never Component 12's __UNKNOWN__ token, and never a "
    "group-conditional number joined back onto a row"
)


# --- 2. capacity --------------------------------------------------------------

#: The cutoffs every policy is reported at. Imported from Component 12 rather than restated,
#: because a policy number and an audit number that describe different operating points cannot
#: be read against each other -- and comparing them is the whole reason Component 12's priority
#: audit was addressed to this component. Component 12 in turn derives them from
#: ``evaluation.simulate.capacity_k_values``, which derives them from each window's own
#: measured median daily inspection rate. Nothing in this chain is a round number.
K_LEVELS: tuple[str, ...] = AUDIT_K_LEVELS

#: The operating point the summary and the figures lead with. One day of real inspection
#: capacity is the unit a scheduler actually works in: a district supervisor asks "who do we
#: send people to tomorrow", not "who is in the top 5% of the quarter".
PRIMARY_K_LEVEL = "k_1_day"

#: Capacity is a rank position, never a probability. Restated here because Component 12
#: refused a probability threshold in prose and this is the component that would have been
#: tempted to add one. A cutoff at p = 0.5 is a number this project has never derived from
#: anything; a cutoff at k = 28 is the number of inspections Chicago worked that day.
CAPACITY_SEMANTICS = (
    "capacity is a rank position derived from the window's measured median daily inspection "
    "rate. No policy in this component uses a probability threshold, and there is no flag to "
    "add one"
)


# --- 3. the allocation mechanisms ---------------------------------------------


class ReserveMechanism(StrEnum):
    """How a policy converts a reserve share into slots.

    The two mechanisms are genuinely different policies, not two spellings of one, and profile
    7 is the reason both exist. ``FLOOR`` guarantees an outcome; ``FORCED`` guarantees a
    spend. When the risk ranking already clears the bar they diverge completely: the floor
    does nothing and the forced reserve buys additional coverage at a measured price.
    """

    #: No reserve. The risk ranking is the queue.
    NONE = "none"

    #: Guarantee that **at least** ``share * k`` of the queue is coverage-eligible, and do
    #: nothing when the risk ranking already delivers that. This is what a coverage reserve
    #: means operationally -- a floor under a population's access to inspection, not a
    #: quantity of capacity spent on it. Profile 7 measured that it is inert in 84 of 85
    #: quarterly cells at the population share and binds in 13 of 85 at twice it.
    FLOOR = "floor"

    #: Spend ``share * k`` slots on coverage-eligible rows the risk ranking did **not**
    #: already select. Retained because it is the mechanism most people mean by "reserve some
    #: capacity", and because it is the only way to add coverage above what risk delivers --
    #: so refusing to implement it would leave the trade-off unmeasured. Profile 6 prices it,
    #: and the price is not zero.
    FORCED = "forced"


class DecisionMechanism(StrEnum):
    """How an establishment came to be in, or out of, the recommended queue.

    The controlled vocabulary that makes "did the model decide this, or did the policy?"
    answerable per row. Every selected row carries exactly one of the first two, and the
    validator refuses a selected row that carries neither.
    """

    RISK_PRIORITY = "risk_priority"
    COVERAGE_RESERVE = "coverage_reserve"
    NOT_SELECTED = "not_selected"


class DecisionReason(StrEnum):
    """Why the mechanism landed where it did. Deterministic codes, never generated prose.

    No language model writes any value in this enum. A reason code that varied between runs
    would make the audit trail unreproducible, and a reason code a reader cannot enumerate is
    a reason code nobody can check.
    """

    #: In the top ``k - n_reserve`` by calibrated risk.
    SELECTED_BY_RISK_RANK = "selected_by_risk_rank"

    #: Coverage-eligible, outside the risk cutoff, and inside the reserve allocation.
    SELECTED_BY_COVERAGE_RESERVE = "selected_by_coverage_reserve"

    #: Not coverage-eligible and ranked below the risk cutoff. The ordinary outcome.
    NOT_SELECTED_CAPACITY_EXHAUSTED = "not_selected_capacity_exhausted"

    #: Coverage-eligible and ranked below the risk cutoff, but the reserve was absent, already
    #: filled, or floored to zero. Separated from the line above because "the reserve did not
    #: reach you" and "you were outranked" are different facts about the same non-selection,
    #: and only the first is a statement about this component's policy.
    NOT_SELECTED_RESERVE_EXHAUSTED = "not_selected_reserve_exhausted"


#: Reasons that mean the establishment is in the queue.
SELECTED_REASONS: frozenset[str] = frozenset(
    {DecisionReason.SELECTED_BY_RISK_RANK, DecisionReason.SELECTED_BY_COVERAGE_RESERVE}
)

#: The reason each mechanism is allowed to carry. Checked by the validator, so a row that
#: claims to have entered through the reserve while citing a risk rank is an error rather than
#: a curiosity.
MECHANISM_REASONS: dict[str, frozenset[str]] = {
    DecisionMechanism.RISK_PRIORITY: frozenset({DecisionReason.SELECTED_BY_RISK_RANK}),
    DecisionMechanism.COVERAGE_RESERVE: frozenset({DecisionReason.SELECTED_BY_COVERAGE_RESERVE}),
    DecisionMechanism.NOT_SELECTED: frozenset(
        {
            DecisionReason.NOT_SELECTED_CAPACITY_EXHAUSTED,
            DecisionReason.NOT_SELECTED_RESERVE_EXHAUSTED,
        }
    ),
}


# --- 4. the policy grid --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One candidate decision policy. Frozen before any comparison was run."""

    policy_id: str
    mechanism: ReserveMechanism
    reserve_share: float
    rationale: str


#: The reserve shares. Three points around the measured eligible population share: half of it,
#: it, and twice it. Not a hyperparameter search -- there is no optimiser here, and the
#: component publishes the trade-off at each point rather than selecting one.
RESERVE_SHARES: tuple[float, ...] = (0.05, 0.10, 0.20)

#: The seven candidate policies: the null policy, and each mechanism at each share.
#:
#: ``pure_risk`` is listed first and is a genuine candidate rather than a control. It is what
#: the system does today if nobody writes this component, and naming it as a policy is what
#: makes it comparable to the alternatives instead of being the invisible default they are
#: measured against.
POLICY_GRID: tuple[PolicySpec, ...] = (
    PolicySpec(
        policy_id="pure_risk",
        mechanism=ReserveMechanism.NONE,
        reserve_share=0.0,
        rationale=(
            "the implicit policy, made explicit. Take the top k by calibrated risk. Included "
            "as a candidate rather than as a baseline: sorting by score is a policy, and it "
            "should have to defend itself against the others on the same terms"
        ),
    ),
    PolicySpec(
        policy_id="coverage_floor_half_share",
        mechanism=ReserveMechanism.FLOOR,
        reserve_share=0.05,
        rationale=(
            "guarantee coverage-eligible establishments half their population share of "
            "capacity. Deliberate under-provision: the weakest guarantee that is still a "
            "guarantee"
        ),
    ),
    PolicySpec(
        policy_id="coverage_floor_population_share",
        mechanism=ReserveMechanism.FLOOR,
        reserve_share=0.10,
        rationale=(
            "guarantee coverage-eligible establishments their measured population share of "
            "capacity, 0.1043 rounded to the grid. Proportional access: the queue may not "
            "serve them less often than they occur"
        ),
    ),
    PolicySpec(
        policy_id="coverage_floor_double_share",
        mechanism=ReserveMechanism.FLOOR,
        reserve_share=0.20,
        rationale=(
            "guarantee twice the population share. Deliberate over-provision, on the argument "
            "that a population the model ranks at chance deserves more than proportional "
            "sampling because each inspection buys information as well as enforcement"
        ),
    ),
    PolicySpec(
        policy_id="coverage_forced_half_share",
        mechanism=ReserveMechanism.FORCED,
        reserve_share=0.05,
        rationale=(
            "spend half the population share of capacity on eligible establishments the risk "
            "ranking passed over. The smallest measurable version of the mechanism most "
            "people mean by 'reserve some capacity'"
        ),
    ),
    PolicySpec(
        policy_id="coverage_forced_population_share",
        mechanism=ReserveMechanism.FORCED,
        reserve_share=0.10,
        rationale=(
            "spend the population share of capacity on eligible establishments the risk "
            "ranking passed over, whether or not it already selected others"
        ),
    ),
    PolicySpec(
        policy_id="coverage_forced_double_share",
        mechanism=ReserveMechanism.FORCED,
        reserve_share=0.20,
        rationale=(
            "spend twice the population share the same way. The stress case, included so the "
            "cost curve has a point far enough out to have a shape"
        ),
    ),
)

POLICY_BY_ID: dict[str, PolicySpec] = {spec.policy_id: spec for spec in POLICY_GRID}

#: The policy every other policy's opportunity cost is measured against. Not a normative
#: choice -- it is the arithmetic baseline, because "what did the reserve cost" is only
#: defined relative to not having one.
BASELINE_POLICY_ID = "pure_risk"


def policy_for(policy_id: str) -> PolicySpec:
    """Look up a policy, or say which ones exist."""
    try:
        return POLICY_BY_ID[policy_id]
    except KeyError:
        known = ", ".join(sorted(POLICY_BY_ID))
        raise PolicyDefinitionError(f"unknown policy {policy_id!r}; known: {known}") from None


# --- 5. the model-selection rule ------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """One calibrated model, and whether a policy may carry it."""

    model_name: str
    admissible: bool
    reason: str


#: Every calibrated model Component 9 produced, with its admissibility decided **before** any
#: metric was read. The refusal is data rather than prose so a reader who opens
#: ``policy_model_selection`` instead of ADR 0039 still finds out why there are four rows and
#: not five.
MODEL_CANDIDATES: tuple[ModelCandidate, ...] = (
    ModelCandidate("lightgbm_platt", True, "supported by Component 11; not experimental"),
    ModelCandidate(
        "logistic_regression_platt", True, "supported by Component 11; not experimental"
    ),
    ModelCandidate(
        "neural_numeric_only_platt", True, "supported by Component 11; not experimental"
    ),
    ModelCandidate("xgboost_platt", True, "supported by Component 11; not experimental"),
    ModelCandidate(
        "xgboost_chain_embeddings_platt",
        False,
        (
            "experimental under ADR 0022 and reported unsupported by Component 11 under "
            "ADR 0031. A model whose recommendations cannot be explained to the inspector "
            "acting on them is not a deployment candidate, whatever it scores"
        ),
    ),
)

CANDIDATE_MODELS: tuple[str, ...] = tuple(c.model_name for c in MODEL_CANDIDATES if c.admissible)
REFUSED_MODELS: tuple[str, ...] = tuple(c.model_name for c in MODEL_CANDIDATES if not c.admissible)

#: The selection axes, in order, each with the direction that wins. Lexicographic: an axis is
#: consulted only when every earlier axis declared a tie.
#:
#: The order encodes what a decision layer needs, in order. Discovery efficiency first,
#: because ranking under capacity is the operational problem. Calibration second, because a
#: policy layer that publishes probabilities to human reviewers needs them to mean what they
#: say. Precision at one day of capacity third, because it is the most concrete operational
#: number. The model name last, so the rule always terminates.
SELECTION_AXES: tuple[tuple[str, str], ...] = (
    ("nde", "higher"),
    ("ece", "lower"),
    ("precision_at_k_1_day", "higher"),
    ("model_name", "lower"),
)

#: How axis 1 decides a tie, and the most consequential sentence in this file.
#:
#: Two models are tied on NDE when their sensitivity intervals overlap. Component 5's
#: ``sensitivity`` artifact perturbs the labels 1,000 times per fold and publishes each model's
#: NDE p05-p95 band; interval overlap is also how ``baseline_models_findings.md`` already
#: decided whether two NDE numbers differ, so this is the repository's existing precedent
#: rather than a new threshold.
#:
#: **This rule was fixed after the NDE column was first read, and that is recorded rather than
#: hidden.** The plan carried a placeholder -- Component 8's five-seed *ROC-AUC* spread of
#: 0.0058 -- and thresholding an NDE difference with a ROC-AUC spread is a unit error. The two
#: rules select different models, ``policy_model_selection`` reports both, and ADR 0039 states
#: the sequence. A rule chosen after seeing what it decides is defensible only if the choosing
#: is visible.
SELECTION_TIE_RULE = (
    "two models are tied on NDE when their Component 5 sensitivity intervals (p05-p95, 1,000 "
    "label-flip replications, quarterly mean) overlap"
)

#: Kept so the discarded alternative leaves a trace. A rejected rule that is never written
#: down is indistinguishable from one that was never considered.
DISCARDED_TIE_BAND = 0.0058

#: What the selected model is, and is not. Written into every manifest.
PRODUCTION_MODEL_CLAIM = (
    "an operating choice of the policy layer, applied from a rule fixed before it was run and "
    "recorded with its inputs. Revisable. NOT a finding that this model is the best one, and "
    "NOT a resolution of whether the ordering survives distribution shift -- the covid_shift "
    "fold orders these models differently and one shift episode cannot carry a selection rule"
)

#: Fold set the selection rule reads. The shift fold is reported beside the rule and never
#: averaged into it: pooling a single 18-month episode with 17 quarters would let one unusual
#: period outvote four years of ordinary ones.
SELECTION_FOLD_SET = "quarterly"


# --- 6. the frontier, and why there may be no winner ----------------------------

#: The axes a policy is judged on. Deliberately several and deliberately not combined into one
#: score: a single number would be a hidden weighting of operational effectiveness against
#: coverage, which is exactly the judgement this component refuses to make on a city's behalf.
FRONTIER_AXES: tuple[tuple[str, str], ...] = (
    ("positives_selected", "higher"),
    ("eligible_selected", "higher"),
)

#: When a winner may be declared, fixed before the frontier was computed.
#:
#: A policy wins only if it is non-dominated on every frontier axis **and** no other policy is
#: non-dominated -- that is, only if the trade-off turns out not to be one. The expected result
#: is that several policies survive, no winner is named, and the manifest says so. That is not
#: a failure to conclude. ADR 0034 established that this repository records a trade-off rather
#: than resolving one on the reader's behalf, and a policy grid whose points genuinely trade
#: positives against coverage has no mathematically optimal member.
POLICY_WINNER_RULE = (
    "a policy is declared the winner only when it is the unique non-dominated policy on every "
    "frontier axis. Otherwise no winner is named, the surviving frontier is published, and "
    "the choice is recorded as a governance decision this component does not make"
)

#: Printed and written into the manifest when the rule declines to name a winner.
NO_WINNER_STATEMENT = "the data does not determine the correct policy"


# --- 7. warnings, and why none of them is an abstention -------------------------


class PolicyWarning(StrEnum):
    """Advisory annotations on a recommendation. Never inputs to a rank.

    Each is a deterministic fact about what is known, not an estimate of how uncertain the
    score is. The distinction is the whole of ADR 0040: this project has never built a
    predictive interval, so a warning that implied one would be manufacturing a statistic.
    """

    #: Coverage-eligible: no code-era canvass history. The same predicate the reserve uses.
    LIMITED_HISTORY = "limited_history"

    #: No inspection of any type on record. A strict subset of the above, and reporting only.
    NO_PRIOR_INSPECTION = "no_prior_inspection"

    #: The row's as-of community area was below Component 12's support floor, so the audit
    #: could not say how the model behaves there. **Not** a statement that behaviour is bad --
    #: a statement that it is unmeasured, which is a different thing and the one a reviewer
    #: needs to know.
    INSUFFICIENT_GROUP_AUDIT_SUPPORT = "insufficient_group_audit_support"

    #: The row's as-of community area is Component 12's ``__UNKNOWN__`` token: no geography
    #: could be recovered. Carried so the population Component 12 flagged stays visible in the
    #: operational artifact, and carried as a label rather than as an adjustment.
    UNKNOWN_GEOGRAPHY = "unknown_geography"


#: The value written when a row carries no warning. A token rather than a null, for the same
#: reason Component 12 made ``__UNKNOWN__`` a group: an empty cell is ambiguous between "no
#: warning" and "warnings were not computed".
NO_WARNING = "none"

#: Multiple warnings are joined with this, sorted, so the column is a deterministic set rather
#: than a precedence the reader has to learn. Choosing one warning to display would mean
#: choosing which fact about an establishment a reviewer is allowed to see.
WARNING_SEPARATOR = "|"

#: The line between a warning and an abstention, stated so it cannot be blurred later.
ABSTENTION_POLICY = (
    "Sentinel never abstains. Every row in the prediction universe receives a recommendation "
    "and a rank; a warning annotates that recommendation rather than withholding it. An "
    "abstention would require a per-row confidence estimate, and this project has not built "
    "one -- no predictive interval, no conformal set, no ensemble spread. Emitting an "
    "abstention category anyway would be manufacturing the statistic that justifies it"
)


# --- 8. advisory thresholds ----------------------------------------------------

#: A reserve that floors to zero slots at some capacity. Not an error: profile 7 measured that
#: a small share at a small cutoff genuinely buys nothing, and rounding up to force a slot
#: would let the reserve exceed the share it declared.
ADVISORY_INERT_RESERVE_CELLS = 1

#: Positives given up, pooled over a fold set at one cutoff, before the cost is called out in
#: the summary rather than only tabulated. One citation is a real inspection that did not
#: happen, so the bar is deliberately at the smallest number that is not zero.
ADVISORY_LOST_POSITIVES = 1

#: How far a policy may move a group's selection share before the run says so out loud. Set to
#: Component 12's own representation-travel advisory so the two components flag the same size
#: of movement, rather than this one inventing a second sensitivity.
ADVISORY_GROUP_SELECTION_SHIFT = 0.05


# --- 9. the human layer ---------------------------------------------------------


class OverrideAction(StrEnum):
    """What a human reviewer may do to a queue. Two verbs, both auditable."""

    #: Put an establishment in the queue that the policy did not select. Displaces the
    #: lowest-ranked risk selection, and the displacement is recorded rather than absorbed.
    FORCE_INCLUDE = "force_include"

    #: Take an establishment out of the queue that the policy selected. The freed slot is
    #: **not** backfilled: filling it would be the policy making a second decision on the back
    #: of a human one, and the reviewer who removed a row did not ask for a replacement.
    FORCE_EXCLUDE = "force_exclude"


#: Every field an override must carry. Absent or blank is a validation error, not a default:
#: an override without an actor is an anonymous change to who gets inspected, which is the
#: precise thing an audit trail exists to prevent.
OVERRIDE_REQUIRED_FIELDS: tuple[str, ...] = (
    "override_id",
    "policy_id",
    "fold_id",
    "k_name",
    "target_inspection_id",
    "action",
    "reason_code",
    "actor",
    "decided_at",
)

#: What an override may never do. A reviewer changes a decision; they do not change the
#: evidence the decision was made from.
OVERRIDE_CANNOT = (
    "an override changes the final decision only. It never edits a score, a rank, a "
    "mechanism, a reason code or the deterministic queue artifact, all of which are written "
    "unchanged beside the override log so the original recommendation stays recoverable"
)

#: The scope of the reproducibility claim, stated precisely because overstating it would be
#: the easiest lie in this component.
DETERMINISM_SCOPE = (
    "the policy computation is deterministic: identical inputs produce byte-identical tables, "
    "and shuffling the input rows changes nothing. Overrides are external human decisions, so "
    "a run is byte-identical only given the identical override file, and the manifest pins "
    "that file by checksum rather than claiming it is reproducible"
)


# --- 10. the boundary ------------------------------------------------------------

#: Written into every manifest and printed on every run. The list Component 12 established the
#: convention for, adapted to what a *decision* artifact will be mistaken for.
DOES_NOT_ESTABLISH: tuple[str, ...] = (
    "that the recommended queue is the correct queue. It is the queue one stated policy "
    "produces from one selected model under one capacity assumption",
    "that the selected model is the best model. Four candidates were statistically "
    "indistinguishable on the headline metric and the rule broke the tie on a secondary axis",
    "that a coverage reserve is fair, or that its absence is unfair. Neither word is defined "
    "here and no fairness criterion was optimised",
    "that establishments with no history are treated correctly. The component measures what "
    "each policy does to them and prices it; which price is acceptable is not measured",
    "that the measured opportunity cost would be the realised cost. Every number is a "
    "re-ordering of inspections that already happened, and inherits Component 5's limitation "
    "whole",
    "any legal, regulatory or ethical position. No protected characteristic is observed "
    "anywhere in this project, and ADR 0035's boundary is inherited unchanged",
)

#: Experiments this component is not permitted to run, each with the ADR that blocked it.
BLOCKED: tuple[str, ...] = (
    "adjusting any score by geography, or applying a group-specific threshold or calibrator "
    "(ADR 0034)",
    "geographic quotas, or any allocation keyed to community area or ZIP (ADR 0038)",
    "re-fitting, re-calibrating or re-scoring any model (ADR 0026, ADR 0029)",
    "introducing a probability threshold (Component 12's THRESHOLD_POLICY)",
    "joining any Component 12 group-conditional number onto a feature table (ADR 0032)",
    "inspector-effect adjustment: the dataset has no inspector (ADR 0019)",
)

#: Limitations that arrive with the inputs and cannot be worked around here.
INHERITED_LIMITATIONS: tuple[str, ...] = (
    "Component 5: this is a re-ordering study over inspections that actually happened. No "
    "establishment nobody inspected has a label, so nothing here speaks to coverage of the "
    "uninspected",
    "Component 3: the target is that a Priority violation was *cited*, not that an "
    "establishment was unsafe",
    "Component 9: probabilities are calibrated in the quarterly mean, not in any single "
    "quarter, and calibration made ECE worse on 16 of 85 quarterly cells",
    "Component 12: geographic differences are confounded with inspection practice by "
    "construction, because Chicago assigns inspectors by district and the dataset names none",
)


def _guard_registry() -> None:
    """Check the frozen constants against each other, at import time.

    Every one of these has a way of drifting during an edit, and every one of them would fail
    silently and plausibly. A grid whose shares no longer match its ids, a mechanism with no
    policy, a reason code no mechanism accepts -- each produces a run that finishes green and
    recommends the wrong establishments.
    """
    ids = [spec.policy_id for spec in POLICY_GRID]
    if len(set(ids)) != len(ids):
        raise PolicyDefinitionError("duplicate policy_id in POLICY_GRID")

    baseline = [s for s in POLICY_GRID if s.policy_id == BASELINE_POLICY_ID]
    if len(baseline) != 1:
        raise PolicyDefinitionError(
            f"the baseline policy {BASELINE_POLICY_ID!r} must appear exactly once in the grid; "
            "every opportunity cost is measured against it"
        )
    if baseline[0].mechanism is not ReserveMechanism.NONE or baseline[0].reserve_share != 0.0:
        raise PolicyDefinitionError("the baseline policy must reserve nothing")

    for spec in POLICY_GRID:
        if not 0.0 <= spec.reserve_share < 1.0:
            raise PolicyDefinitionError(
                f"{spec.policy_id}: reserve share {spec.reserve_share} is not a share. A "
                "reserve of the entire window is not a policy, it is the absence of one"
            )
        if (spec.mechanism is ReserveMechanism.NONE) != (spec.reserve_share == 0.0):
            raise PolicyDefinitionError(
                f"{spec.policy_id}: mechanism {spec.mechanism} and share {spec.reserve_share} "
                "disagree about whether this policy reserves anything"
            )
        if not spec.rationale:
            raise PolicyDefinitionError(f"{spec.policy_id}: every policy states why it exists")

    reserving = {s.mechanism for s in POLICY_GRID if s.mechanism is not ReserveMechanism.NONE}
    missing = {ReserveMechanism.FLOOR, ReserveMechanism.FORCED} - reserving
    if missing:
        raise PolicyDefinitionError(
            f"no policy exercises {', '.join(sorted(missing))}. A mechanism with no policy is "
            "a code path no run ever takes, which is indistinguishable from one that is broken"
        )
    for mechanism in (ReserveMechanism.FLOOR, ReserveMechanism.FORCED):
        shares = tuple(sorted(s.reserve_share for s in POLICY_GRID if s.mechanism is mechanism))
        if shares != tuple(sorted(RESERVE_SHARES)):
            raise PolicyDefinitionError(
                f"{mechanism} covers shares {shares}, not the declared grid {RESERVE_SHARES}. "
                "The two mechanisms must be measured at identical shares or the comparison "
                "between them is not a comparison"
            )

    if ELIGIBILITY_COLUMN in FORBIDDEN_POLICY_COLUMNS:
        raise PolicyDefinitionError("the eligibility column is a label column")
    if SECONDARY_FLAG_COLUMN in FORBIDDEN_POLICY_COLUMNS:
        raise PolicyDefinitionError("the secondary flag column is a label column")
    if not FORBIDDEN_POLICY_COLUMNS:
        raise PolicyDefinitionError(
            "the forbidden-column list is empty. The validator checks eligibility against it, "
            "so an empty list turns that check into a formality that always passes"
        )

    if PRIMARY_K_LEVEL not in K_LEVELS:
        raise PolicyDefinitionError(
            f"the primary operating point {PRIMARY_K_LEVEL!r} is not one of the reported "
            f"cutoffs {K_LEVELS}"
        )
    if not K_LEVELS:
        raise PolicyDefinitionError("no capacity levels are declared")

    if not CANDIDATE_MODELS:
        raise PolicyDefinitionError(
            "every model was refused, so no policy can be built. A component that cannot "
            "produce a queue should say so rather than emit an empty one"
        )
    names = [c.model_name for c in MODEL_CANDIDATES]
    if len(set(names)) != len(names):
        raise PolicyDefinitionError("duplicate model in MODEL_CANDIDATES")
    for candidate in MODEL_CANDIDATES:
        if not candidate.reason:
            raise PolicyDefinitionError(
                f"{candidate.model_name}: admissibility is stated without a reason. A refusal "
                "with no measurement behind it is an opinion in a data structure"
            )
    if SELECTION_AXES[-1][0] != "model_name":
        raise PolicyDefinitionError(
            "the last selection axis must be the model name, or the rule can fail to terminate"
        )

    for decision, reasons in MECHANISM_REASONS.items():
        if not reasons:
            raise PolicyDefinitionError(f"{decision} accepts no reason code")
    covered = set().union(*MECHANISM_REASONS.values())
    if covered != set(DecisionReason):
        raise PolicyDefinitionError(
            f"reason codes {sorted(set(DecisionReason) - covered)} belong to no mechanism"
        )
    if not covered >= SELECTED_REASONS:
        raise PolicyDefinitionError("a selected reason belongs to no mechanism")

    if WARNING_SEPARATOR in NO_WARNING:
        raise PolicyDefinitionError("the no-warning token is not separable from a warning list")
    for warning in PolicyWarning:
        if WARNING_SEPARATOR in warning:
            raise PolicyDefinitionError(f"warning {warning!r} contains the separator")

    if not DOES_NOT_ESTABLISH:
        raise PolicyDefinitionError(
            "the boundary list is empty. It is written into every manifest and printed on "
            "every run precisely so it cannot be dropped"
        )
    if not BLOCKED:
        raise PolicyDefinitionError("the blocked-experiment list is empty")
    if not INHERITED_LIMITATIONS:
        raise PolicyDefinitionError("the inherited-limitation list is empty")


_guard_registry()


__all__ = [
    "ABSTENTION_POLICY",
    "ADVISORY_GROUP_SELECTION_SHIFT",
    "ADVISORY_INERT_RESERVE_CELLS",
    "ADVISORY_LOST_POSITIVES",
    "BASELINE_POLICY_ID",
    "BLOCKED",
    "CANDIDATE_MODELS",
    "CAPACITY_SEMANTICS",
    "DETERMINISM_SCOPE",
    "DISCARDED_TIE_BAND",
    "DOES_NOT_ESTABLISH",
    "ELIGIBILITY_COLUMN",
    "ELIGIBILITY_IS_NOT_GEOGRAPHY",
    "ELIGIBILITY_RULE",
    "ELIGIBLE_POPULATION_SHARE",
    "FORBIDDEN_POLICY_COLUMNS",
    "FRONTIER_AXES",
    "INHERITED_LIMITATIONS",
    "K_LEVELS",
    "MECHANISM_REASONS",
    "MODEL_CANDIDATES",
    "NO_WARNING",
    "NO_WINNER_STATEMENT",
    "OVERRIDE_CANNOT",
    "OVERRIDE_REQUIRED_FIELDS",
    "POLICY_BY_ID",
    "POLICY_DEFINITION_VERSION",
    "POLICY_GRID",
    "POLICY_WINNER_RULE",
    "PRIMARY_K_LEVEL",
    "PRODUCTION_MODEL_CLAIM",
    "REFUSED_MODELS",
    "RESERVE_SHARES",
    "SELECTED_REASONS",
    "SELECTION_AXES",
    "SELECTION_FOLD_SET",
    "SELECTION_TIE_RULE",
    "SECONDARY_FLAG_COLUMN",
    "DecisionMechanism",
    "DecisionReason",
    "ModelCandidate",
    "OverrideAction",
    "PolicyDefinitionError",
    "PolicySpec",
    "PolicyWarning",
    "ReserveMechanism",
    "policy_for",
]
