"""Building the group frame, and proving it is admissible. Pure -- no filesystem, no clock.

The group frame is the audit's foundation and its leakage surface. Everything downstream is
arithmetic over it, so if a row is labelled with the wrong neighbourhood, or with a
neighbourhood it only acquired later, every number in the artifact is wrong in a way that
nothing further downstream can detect: the metrics stay finite, the supports stay plausible,
and no check fires. This module is where that is prevented.

Three properties are established here rather than assumed.

**Temporal validity.** Each group value comes from the establishment's most recent inspection
of any type *strictly before* the row's own date. Component 8 built the frame that way and
validates it; this module re-derives the strict inequality rather than trusting a manifest,
because a date comparison someone else ran is a claim and a date comparison run here is a
measurement.

**Join integrity.** One group value per audited row, from a key that maps to exactly one
value, with no scored row left unlabelled. A fairness audit that silently dropped rows on a
join would report metrics over a population nobody chose.

**Stage separation.** The audited frame carries both probabilities Component 9 wrote --
``base_score`` and ``score`` -- side by side under explicit names, so no downstream caller
has to remember which column is calibrated. MEMORY invariant 71.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sentinel.fairness.definitions import (
    AUDITED_GROUP_DEFINITIONS,
    UNKNOWN_GROUP,
    GroupDefinitionSpec,
    group_definition_for,
)
from sentinel.fairness.models import GroupFrame

#: The join key everywhere in this project: Component 3's identifier, Component 4's primary
#: key and the raw Socrata ``inspection_id`` at the same time. That coincidence is why this
#: audit invents no key and why its joins are exact rather than approximate.
KEY = "target_inspection_id"

#: Columns the audit reads off Component 9's artifact. ``score`` and ``base_score`` are
#: renamed on arrival so that nothing downstream can read "score" and assume either one.
PREDICTION_COLUMNS: tuple[str, ...] = (
    KEY,
    "model_name",
    "base_model_name",
    "fold_set",
    "fold_id",
    "score",
    "base_score",
    "is_experimental",
    "method",
)

#: The order every audited frame is put into before any metric touches it.
#:
#: Component 5's canonical order -- ``(inspection_date, target_inspection_id)`` -- extended by
#: the model and fold each row belongs to. **Load-bearing, not decorative**, and the project
#: has measured why three times: Component 6 found row order moving coefficients by 7e-09,
#: Component 7 found it moving a *prediction* by 1.1e-01, and Component 12 found it moving a
#: pooled ECE. The mechanism here is subtler than either: ``ece`` uses equal-mass bins, so
#: rows tied at a bin boundary are assigned by the order they arrive in, and a shuffled input
#: therefore produces a slightly different reference value for every disparity.
#:
#: Sorting once, here, is what makes ``two runs over shuffled inputs are byte-identical`` true
#: rather than nearly true.
CANONICAL_SORT: tuple[str, ...] = ("model_name", "fold_set", "fold_id", "rd", KEY)

#: Provenance columns carried from Component 8's as-of table. They are what make the
#: temporal claim checkable per row instead of asserted once.
SOURCE_COLUMNS: tuple[str, ...] = (
    "source_inspection_id",
    "source_inspection_date",
    "days_since_source",
)

#: Component 2's identity, carried for one purpose only: it is the block a group-level
#: bootstrap resamples. Establishments recur inside a group on a 358-day median canvass
#: cycle and their rows share an as-of history, so an i.i.d. row bootstrap understates the
#: standard error -- the measurement Component 9 made when it ran both schemes. It is never
#: an input to anything and never appears in an output table.
ENTITY_COLUMN = "establishment_id"


class GroupFrameError(ValueError):
    """The group frame could not be built without misrepresenting something."""


def resolve_definitions(names: Sequence[str] | None) -> tuple[GroupDefinitionSpec, ...]:
    """The group definitions to audit, defaulting to every audited one.

    Goes through :func:`group_definition_for`, so asking for a refused definition raises with
    the measurement that refused it rather than returning an empty result. A caller who typed
    ``ward`` deserves to be told why not.
    """
    requested = tuple(names) if names else AUDITED_GROUP_DEFINITIONS
    if not requested:
        raise GroupFrameError("no group definition requested")
    seen: set[str] = set()
    specs: list[GroupDefinitionSpec] = []
    for name in requested:
        if name in seen:
            raise GroupFrameError(f"group definition requested twice: {name}")
        seen.add(name)
        specs.append(group_definition_for(name))
    return tuple(specs)


def group_source(
    categoricals: pl.DataFrame,
    definitions: Sequence[GroupDefinitionSpec],
) -> pl.DataFrame:
    """The group columns and their provenance, one row per key.

    Reads Component 8's as-of table and nothing else. The alternative -- re-deriving the
    as-of join against raw -- would put a second implementation of "the same place" in this
    repository, which is exactly what ADR 0022 declined to do when it refused to derive
    ``chain`` from ``dba_name`` a second time.
    """
    missing = [
        spec.source_column for spec in definitions if spec.source_column not in categoricals.columns
    ]
    if missing:
        raise GroupFrameError(
            f"group source is missing column(s) {', '.join(missing)}. The audited "
            "definitions live in Component 8's as-of categorical table; a frame without "
            "them is not that table."
        )
    for column in (KEY, ENTITY_COLUMN, *SOURCE_COLUMNS, "inspection_date"):
        if column not in categoricals.columns:
            raise GroupFrameError(
                f"group source is missing {column!r}, so the as-of claim cannot be "
                "re-derived. Provenance that cannot be checked is provenance that is being "
                "taken on trust."
            )

    keys = categoricals.get_column(KEY)
    if keys.len() != keys.n_unique():
        raise GroupFrameError(
            f"group source has {keys.len() - keys.n_unique()} duplicate keys. An ambiguous "
            "mapping would silently multiply audited rows and inflate every support count."
        )

    columns = [spec.source_column for spec in definitions]
    nulls = {
        column: int(categoricals.get_column(column).null_count())
        for column in columns
        if categoricals.get_column(column).null_count()
    }
    if nulls:
        raise GroupFrameError(
            f"group source has nulls in {nulls}. Absence is the token {UNKNOWN_GROUP!r} in "
            "this project, never a null: a null would be coerced somewhere on the way to a "
            "group-by, and where it landed would depend on which code path saw it first."
        )

    return categoricals.select(KEY, ENTITY_COLUMN, "inspection_date", *SOURCE_COLUMNS, *columns)


def check_temporal_validity(source: pl.DataFrame) -> tuple[int, int | None]:
    """Re-derive, per row, that every group value predates the row it labels.

    Returns ``(rows_with_a_source, minimum_lag_days)``. Raises when any row's group value
    came from an inspection dated on or after the row's own date.

    A zero-day lag is the observable that matters. Component 8's join uses
    ``allow_exact_matches=False``, so a zero would mean a row had supplied its own
    attributes -- and every metric downstream would still look completely normal, because
    what changes is the question the numbers answer rather than their magnitude.
    """
    dated = source.with_columns(pl.col("inspection_date").str.to_date().alias("_rd"))
    with_source = dated.filter(pl.col("source_inspection_date").is_not_null())
    offenders = with_source.filter(pl.col("source_inspection_date") >= pl.col("_rd"))
    if not offenders.is_empty():
        sample = offenders.get_column(KEY).head(5).to_list()
        raise GroupFrameError(
            f"{offenders.height} group value(s) came from an inspection dated on or after "
            f"the row they label, e.g. {sample}. That is future information entering through "
            "the group mapping, and it is undetectable downstream because it changes what "
            "the numbers mean rather than whether they are finite."
        )
    lags = with_source.get_column("days_since_source").drop_nulls()
    minimum = int(lags.min()) if lags.len() else None  # type: ignore[arg-type]
    if minimum is not None and minimum < 1:
        raise GroupFrameError(
            f"minimum source lag is {minimum} days. A lag below one day means a row supplied "
            "its own group attributes."
        )
    return with_source.height, minimum


def build_group_frame(
    predictions: pl.DataFrame,
    categoricals: pl.DataFrame,
    labels: pl.DataFrame,
    definitions: Sequence[GroupDefinitionSpec],
) -> GroupFrame:
    """Join predictions, group values and outcomes into the frame the audit measures.

    An inner join would hide a defect by producing a smaller, plausible frame. This uses a
    left join from the predictions and then *requires* completeness, so a scored row without
    a group value or without a label stops the run instead of quietly leaving the population.
    """
    if predictions.is_empty():
        raise GroupFrameError("no predictions to audit")
    for column in PREDICTION_COLUMNS:
        if column not in predictions.columns:
            raise GroupFrameError(
                f"prediction artifact is missing {column!r}. Component 12 reads Component "
                "9's calibrated artifact, which carries the base score alongside the "
                "calibrated one; an artifact without both cannot answer whether calibration "
                "reached the groups."
            )
    for column in (KEY, "target", "rd"):
        if column not in labels.columns:
            raise GroupFrameError(
                f"label frame must carry {KEY!r}, 'target' and 'rd'. The reference date is "
                "needed because a within-group NDE is computed over that group's own slot "
                "calendar, and Component 5's simulation is defined on dates rather than on "
                "row order."
            )

    source = group_source(categoricals, definitions)
    as_of_rows, min_lag = check_temporal_validity(source)

    frame = (
        predictions.select(PREDICTION_COLUMNS)
        .rename({"score": "calibrated_probability", "base_score": "base_probability"})
        .join(
            source.select(KEY, ENTITY_COLUMN, *[spec.source_column for spec in definitions]),
            on=KEY,
            how="left",
        )
        .join(labels.select(KEY, "target", "rd"), on=KEY, how="left")
    )

    for spec in definitions:
        unlabelled = frame.filter(pl.col(spec.source_column).is_null()).height
        if unlabelled:
            raise GroupFrameError(
                f"{unlabelled} scored row(s) have no {spec.name} value. Dropping them would "
                "report metrics over a population nobody chose; the group source must cover "
                "every audited row."
            )
    unlabelled = frame.filter(pl.col("target").is_null()).height
    if unlabelled:
        raise GroupFrameError(f"{unlabelled} scored row(s) have no outcome label")

    frame = frame.with_columns(pl.col("target").cast(pl.Int64)).sort(CANONICAL_SORT)
    observed = {
        spec.name: tuple(sorted(frame.get_column(spec.source_column).unique().to_list()))
        for spec in definitions
    }
    return GroupFrame(
        frame=frame,
        definitions=tuple(definitions),
        as_of_rows=as_of_rows,
        min_source_lag_days=min_lag,
        observed_values=observed,
    )


def audited_frame(group_frame: GroupFrame) -> pl.DataFrame:
    """The joined frame, typed for callers that only have the container."""
    frame = group_frame.frame
    if not isinstance(frame, pl.DataFrame):  # pragma: no cover - defensive
        raise GroupFrameError("group frame does not hold a DataFrame")
    return frame


def stage_column(stage: str) -> str:
    """The frame column holding one prediction stage's probability.

    A function rather than a dict literal at each call site, because the single most damaging
    silent defect available in this component is measuring the calibrated probability and
    labelling it base. There is one place that mapping is written down.
    """
    if stage == "base":
        return "base_probability"
    if stage == "calibrated":
        return "calibrated_probability"
    raise GroupFrameError(f"unknown prediction stage {stage!r}")


def observed_group_values(frame: pl.DataFrame, spec: GroupDefinitionSpec) -> tuple[str, ...]:
    """Every group value present, sorted, ``__UNKNOWN__`` included.

    Sorted rather than insertion-ordered, for the reason Component 8 sorted its vocabularies:
    insertion order is row order, and row order is not a contract.
    """
    return tuple(sorted(frame.get_column(spec.source_column).unique().to_list()))


__all__ = [
    "CANONICAL_SORT",
    "ENTITY_COLUMN",
    "KEY",
    "PREDICTION_COLUMNS",
    "SOURCE_COLUMNS",
    "GroupFrameError",
    "audited_frame",
    "build_group_frame",
    "check_temporal_validity",
    "group_source",
    "observed_group_values",
    "resolve_definitions",
    "stage_column",
]
