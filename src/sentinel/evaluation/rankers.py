"""Deterministic score producers -- the baselines Component 6 must beat.

Nothing here is fitted. There is no training, no coefficient, no hyperparameter.
Every ranker is a stated rule over columns Component 4 already published, which
is what keeps Component 5 inside its roadmap slot: the evaluation harness is
built and proven on real data *before* the first model exists, so the model is
measured by a yardstick it did not get to shape.

Three of these represent plausible operational alternatives, and that is the
point. If a gradient-boosted model cannot beat "inspect whoever is most overdue"
then it has not earned its complexity.

**Missing values are handled by a rule stated in advance**, and each rule is
recorded here with the consequence measured afterwards. Saying "I picked the
rule for this reason, then measured that it helped / cost me" is a different
thing from picking the rule because it helped.

The CDPH replication was **deferred to Component 6**, deliberately, and Component 6
has since shipped it as an explicitly labelled *approximation* -- the table below is
why it can only ever be one. The project
spec's §6.1 asks for a logistic regression reproducing Chicago's deployed 2015
model; a logistic regression is a fitted model and fitting one here would cross
the component boundary. What can be said now is which of its inputs Sentinel
could even reconstruct:

===========================  ============================================
2015 model input family      status in this repository
===========================  ============================================
prior violation history      available (Component 4, 12 features)
time since last inspection   available (``days_since_last_canvass``)
business age / tenure        available (``days_since_first_inspection``)
inspector identity           deliberately excluded -- audit Finding 1
nearby 311 complaints        NOT INGESTED -- Component 1 extension
burglary intensity           NOT INGESTED -- Component 1 extension
alcohol / tobacco licence    NOT INGESTED -- Component 1 extension
weather / temperature        NOT INGESTED -- Component 1 extension
facility type, risk category present in raw, not in the feature table
===========================  ============================================

So a faithful replication is not reachable from the current data contract, and
Component 6 must label whatever it builds an *approximation* with that table
attached. Component 5 does not fake it.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)

#: Bumped when a ranker's rule changes, so two runs are never silently compared.
RANKER_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class RankerSpec:
    """One baseline: what it is called, what it needs, and what it emits."""

    name: str
    description: str
    columns: tuple[str, ...]
    is_probability: bool
    null_rule: str


def random_scores(n: int, *, seed: int) -> list[float]:
    """Uniform noise. The floor: any ranking that cannot beat this is worthless.

    Seeded, and reported over many seeds rather than one, so the model is never
    flattered by an unusually poor draw.
    """
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]


def days_since_last_canvass_scores(frame: pl.DataFrame) -> list[float]:
    """Rank by how long it has been since the last routine canvass.

    The project spec's first heuristic is "rank by days overdue". A true
    statutory overdue figure needs the CDPH risk category (Risk 1 twice a year,
    Risk 2 annually, Risk 3 every two years) to know what the deadline *was*,
    and that column is in the raw snapshot but not in Component 4's feature
    table. Component 5 is forbidden from adding features, so this is the
    closest honest reconstruction: elapsed time, not deficit against a deadline.
    Recorded as a limitation rather than passed off as the spec's heuristic.

    **Null rule:** a row with no prior canvass has never been canvassed at all,
    which is the limiting case of "a long time since the last one", so it sorts
    first. Chosen on that argument; the measured consequence is that the
    never-canvassed group has a base rate of 0.5907 against 0.5252 overall, so
    the choice happens to help. Stated because it was measured, not tuned.
    """
    column = frame["days_since_last_canvass"]
    observed = [v for v in column.to_list() if v is not None]
    ceiling = float(max(observed) + 1) if observed else 1.0
    return [ceiling if v is None else float(v) for v in column.to_list()]


def priority_at_last_canvass_scores(frame: pl.DataFrame) -> list[float]:
    """Rank by whether the previous canvass cited a Priority violation.

    The project spec's second heuristic, and reconstructible exactly. This is
    the single most obvious thing a human scheduler would do.

    **Null rule:** null means no prior *code-era* canvass, so the establishment
    is neither known-clean nor known-cited. It scores 0.5 -- between the two --
    because collapsing "we know nothing" into "we know it was clean" would
    assert evidence that does not exist. The measured consequence is a cost: the
    null group's base rate is 0.7284, the highest of the three, so placing it in
    the middle ranks it lower than it deserves. Kept anyway, because choosing
    the tier by its outcome would be fitting to the label.
    """
    return [0.5 if v is None else (1.0 if v else 0.0) for v in frame["priority_at_last_canvass"]]


def prior_canvass_priority_rate_scores(frame: pl.DataFrame) -> list[float]:
    """Rank by the establishment's historical Priority citation rate.

    A slightly richer history heuristic than the binary one: not "did it happen
    last time" but "how often has it happened". Included because the difference
    between the two is a cheap, interpretable measure of how much memory beyond
    one inspection is worth.

    **Null rule:** null when there is no prior code-era canvass -- the rate has
    no denominator. Scores 0.5 for the same reason as above.
    """
    return [0.5 if v is None else float(v) for v in frame["prior_canvass_priority_rate"]]


def constant_scores(n: int) -> list[float]:
    """Every establishment identical. A diagnostic, not a baseline.

    Exists to exercise the tie-breaking path on real data: with every score
    equal, the entire ordering is decided by ``target_inspection_id``, and the
    result must still be perfectly reproducible across runs and row orders.
    """
    return [0.0] * n


RANKERS: tuple[RankerSpec, ...] = (
    RankerSpec(
        name="random",
        description="Uniform random scores, reported across many seeds.",
        columns=(),
        is_probability=False,
        null_rule="not applicable",
    ),
    RankerSpec(
        name="days_since_last_canvass",
        description="Longest since the last routine canvass goes first.",
        columns=("days_since_last_canvass",),
        is_probability=False,
        null_rule="never canvassed sorts first, as the limiting case of overdue",
    ),
    RankerSpec(
        name="priority_at_last_canvass",
        description="Cited a Priority violation at the previous canvass.",
        columns=("priority_at_last_canvass",),
        is_probability=False,
        null_rule="unknown scores 0.5, between known-clean and known-cited",
    ),
    RankerSpec(
        name="prior_canvass_priority_rate",
        description="Historical share of canvasses that cited a Priority violation.",
        columns=("prior_canvass_priority_rate",),
        is_probability=False,
        null_rule="unknown scores 0.5, between known-clean and known-cited",
    ),
    RankerSpec(
        name="constant",
        description="All scores equal; exercises deterministic tie-breaking.",
        columns=(),
        is_probability=False,
        null_rule="not applicable",
    ),
)

RANKERS_BY_NAME: dict[str, RankerSpec] = {spec.name: spec for spec in RANKERS}


def score(name: str, frame: pl.DataFrame, *, seed: int = 0) -> list[float]:
    """Produce scores for one named ranker over one window's rows.

    ``frame`` must already be in the window's canonical order; the returned list
    is aligned to it positionally.
    """
    if name not in RANKERS_BY_NAME:
        raise KeyError(f"unknown ranker: {name}")
    if name == "random":
        return random_scores(frame.height, seed=seed)
    if name == "days_since_last_canvass":
        return days_since_last_canvass_scores(frame)
    if name == "priority_at_last_canvass":
        return priority_at_last_canvass_scores(frame)
    if name == "prior_canvass_priority_rate":
        return prior_canvass_priority_rate_scores(frame)
    if name == "constant":
        return constant_scores(frame.height)
    raise KeyError(f"ranker {name} is declared but not implemented")


def required_columns(names: Sequence[str]) -> tuple[str, ...]:
    """Every feature column the named rankers need, deduplicated."""
    columns: list[str] = []
    for name in names:
        for column in RANKERS_BY_NAME[name].columns:
            if column not in columns:
                columns.append(column)
    return tuple(columns)


__all__ = [
    "RANKERS",
    "RANKERS_BY_NAME",
    "RANKER_VERSION",
    "RankerSpec",
    "constant_scores",
    "days_since_last_canvass_scores",
    "prior_canvass_priority_rate_scores",
    "priority_at_last_canvass_scores",
    "random_scores",
    "required_columns",
    "score",
]
