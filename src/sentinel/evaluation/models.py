"""Data structures for temporal evaluation.

Two ideas carry most of the weight here.

``FoldSpec`` is the unit of honesty: it names three date ranges and asserts, at
construction, that they are strictly ordered. Nothing downstream can quietly
overlap training with test, because the only way to obtain a fold is to build
one of these.

``PredictionSet`` is the model-agnostic contract. A future component hands over
scores plus a declaration of what it was allowed to know (``trained_through``),
and the harness checks that declaration against the fold before scoring
anything. ``is_probability`` is the other half: a ranking-only baseline is
never given fake probabilities so that it fits a calibration API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import polars as pl
from pydantic import BaseModel, Field

#: Bumped whenever fold construction, metric definitions or simulation
#: semantics change in a way that makes two runs incomparable.
EVALUATION_DEFINITION_VERSION = "v1"

#: Higher score means higher predicted risk of a Priority / Priority Foundation
#: citation. Stated as a constant so the direction is never inferred from code.
SCORE_DIRECTION = "descending: higher score = higher predicted violation risk"

#: Ties are broken on this column, ascending, as a string. The column is stable,
#: unique per row, and independent of frame order -- which is the whole
#: requirement. Whether the ordering is lexicographic or numeric is arbitrary
#: for a tie-break; being deterministic is not.
TIE_BREAK_COLUMN = "target_inspection_id"


class ScheduleName(StrEnum):
    """The five reference schedules every test window is compared against."""

    OPTIMAL = "optimal"
    MODEL = "model"
    BUSINESS_AS_USUAL = "business_as_usual"
    RANDOM = "random"
    WORST = "worst"


class Split(StrEnum):
    """Which period of a fold a row belongs to."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"
    OUTSIDE = "outside"


class FoldError(ValueError):
    """Raised when a fold's date ranges are not strictly ordered."""


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One rolling-origin fold. All bounds are inclusive dates.

    The ordering invariant is asserted in ``__post_init__`` rather than left to
    a test, because a fold that violates it is not a fold -- it is leakage with
    a name.
    """

    fold_set: str
    fold_id: str
    train_start: date
    train_end: date
    calibration_start: date
    calibration_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        if self.train_start > self.train_end:
            raise FoldError(f"{self.fold_id}: train_start {self.train_start} after train_end")
        if self.calibration_start > self.calibration_end:
            raise FoldError(f"{self.fold_id}: calibration_start after calibration_end")
        if self.test_start > self.test_end:
            raise FoldError(f"{self.fold_id}: test_start {self.test_start} after test_end")
        if self.train_end >= self.calibration_start:
            raise FoldError(
                f"{self.fold_id}: train_end {self.train_end} must be strictly before "
                f"calibration_start {self.calibration_start}"
            )
        if self.calibration_end >= self.test_start:
            raise FoldError(
                f"{self.fold_id}: calibration_end {self.calibration_end} must be strictly "
                f"before test_start {self.test_start}"
            )

    def split_of(self, day: date) -> Split:
        """Which split a reference date falls in. Outside is a legitimate answer."""
        if self.train_start <= day <= self.train_end:
            return Split.TRAIN
        if self.calibration_start <= day <= self.calibration_end:
            return Split.CALIBRATION
        if self.test_start <= day <= self.test_end:
            return Split.TEST
        return Split.OUTSIDE


@dataclass(frozen=True, slots=True)
class FoldStats:
    """Row counts and base rates for one fold, measured from the data."""

    fold_id: str
    train_rows: int
    calibration_rows: int
    test_rows: int
    train_positive_rate: float | None
    calibration_positive_rate: float | None
    test_positive_rate: float | None
    test_establishments: int
    test_days: int
    test_median_daily_capacity: float | None


@dataclass(frozen=True, slots=True)
class PredictionSet:
    """Scores for one fold's test window, from one model, plus its provenance.

    ``frame`` carries exactly two columns: ``target_inspection_id`` and
    ``score``. Nothing else is accepted, so a model cannot smuggle a label
    through.

    ``trained_through`` is the last reference date the producer was allowed to
    learn from. The harness rejects the set if it is later than the fold's
    calibration end -- this is the check that makes retrospective cheating hard
    to do by accident.
    """

    model_name: str
    model_version: str
    fold_id: str
    frame: pl.DataFrame
    is_probability: bool = False
    trained_through: date | None = None


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One post-run assertion about the evaluation."""

    name: str
    passed: bool
    severity: str
    detail: str
    offenders: tuple[str, ...] = ()


@dataclass
class EvaluationStats:
    """Counts accumulated during a run, surfaced in the manifest and the CLI."""

    feature_rows: int = 0
    fold_sets: list[str] = field(default_factory=list)
    folds: int = 0
    models: list[str] = field(default_factory=list)
    metric_rows: int = 0
    curve_rows: int = 0


class ArtifactRecord(BaseModel):
    """Provenance for one written file."""

    path: str
    bytes: int
    sha256: str
    row_count: int
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class EvaluationManifest(BaseModel):
    """Self-contained provenance and QA record for one evaluation run.

    Pins the feature table by checksum and records every seed, so the question
    *"exactly what was the model allowed to know when this score was produced?"*
    is answerable from the manifest alone, without reading source code.
    """

    component: str = "temporal_evaluation"
    code_version: str
    evaluation_definition_version: str
    built_at: str

    features_path: str
    features_sha256: str
    feature_definition_version: str

    #: The prediction artifact scored alongside the built-in baselines, if any. Pinned
    #: by checksum so a run is tied to the exact bytes it read -- without it, the
    #: manifest's promise ("what was the model allowed to know?") stops being answerable
    #: as soon as a fitted model is involved. None when only the heuristics were run.
    predictions_path: str | None = None
    predictions_sha256: str | None = None

    estimand: str
    score_direction: str
    tie_break_column: str
    capacity_semantics: str

    fold_cadence: str
    fold_sets: dict[str, int]
    folds: int
    first_test_start: str | None
    last_test_end: str | None
    excluded_partial_windows: list[str]

    models: list[str]
    random_seeds: list[int]
    sensitivity_replications: int
    sensitivity_seed: int

    feature_rows: int
    metric_rows: int
    curve_rows: int
    simulation_rows: int

    blocked: list[str]
    artifacts: list[ArtifactRecord]
    checks: list[dict[str, str]]
