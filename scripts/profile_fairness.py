"""Read-only profiling of the group-audit surface, before any fairness code is written.

Analysis tooling, not library code: it answers one-off questions about a snapshot, nothing
imports it, and it should not ship in the wheel. Output is markdown on stdout, pasted into
``docs/analysis/fairness_findings.md``.

⚠ **This script fits nothing, scores nothing and changes nothing.** Every model in this
project is frozen and every prediction already exists on disk. Component 12 is the first
component that re-executes none of them, so there is no bit-identity gate here -- only a
checksum gate at run time proving the inputs were not touched.

⚠ **This script is run before the support thresholds, the capacity levels and the display
policy are frozen, and is what fixes them.** Profiles 4, 7 and 9 measure how much data each
group actually has; the constants are then set from those measurements and written into ADR
0034. Component 9 set three thresholds from expectation and had to correct all three -- a
threshold set from expectation rather than measurement is a guess wearing a decimal point.

⚠ **It reads labels, and that is legitimate here.** Component 11's profiler could not,
because a profiler that reported test accuracy is the first step towards selecting a model
by hand. Component 12 is forbidden from selecting a model too, but its *subject matter* is
the joint distribution of outcome and group -- a base rate by neighbourhood is the thing
being audited, not a score being peeked at. No profile below ranks a model, and profile 5
reports outcome rates without reference to any prediction.

Questions this script answers
-----------------------------
1.  ``group_source_inventory``   -- which geographic columns exist at all, at what
                                    cardinality and missingness, and where does each one
                                    survive to? **The inventory ADR 0033 rests on.**
2.  ``group_temporal_stability`` -- does the as-of geography differ from the value recorded
                                    on the row itself, and are the two published ward layers
                                    mutually consistent? **Decides which definitions are
                                    admissible and which are refused.**
3.  ``group_join_integrity``     -- is the predictions-to-group join one-to-one, total, and
                                    free of ambiguous or duplicate mappings?
4.  ``group_support_population`` -- how many rows, positives and negatives does each
                                    (fold, group) cell hold, and each pooled group?
                                    **Fixes SUPPORT_MIN_ROWS, MIN_POSITIVE, MIN_NEGATIVE
                                    and CALIBRATION_MIN_ROWS.**
5.  ``group_outcome_rates``      -- how far apart are the groups' base rates, before any
                                    model is evaluated?
6.  ``representation_drift``     -- does each group's share of the evaluated population move
                                    across the 17 quarterly folds?
7.  ``capacity_and_k``           -- what capacity did the city actually work, and what top-k
                                    levels are therefore meaningful? **Fixes K_LEVELS.**
8.  ``missingness_by_group``     -- is null-rule family missingness distributed evenly across
                                    groups? **The Component 11 link.**
9.  ``attribution_support``      -- how many explained rows does each group have in Component
                                    11's bounded sample? **Decides whether the attribution
                                    profile comparison is supportable at all.**
10. ``covid_support``            -- the same support question for the covid_shift fold,
                                    reported separately and never pooled with the quarterly
                                    answer.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sentinel.evaluation.folds as folds_module  # noqa: E402
import sentinel.evaluation.simulate as simulate  # noqa: E402
from sentinel.config import load_settings  # noqa: E402
from sentinel.evaluation.models import FoldSpec  # noqa: E402
from sentinel.modeling.definitions import (  # noqa: E402
    family_indicator_name,
    indicator_source_column,
    null_families,
)
from sentinel.query import duckdb_queries  # noqa: E402

#: The two group definitions this project can actually audit, and the column each lives in
#: on Component 8's as-of table. Everything else is inventoried in profile 1 and refused.
GROUP_COLUMNS: tuple[str, ...] = ("community_area", "zip")

#: The absence token Component 8 writes when nothing could be carried forward. It is a real
#: group here, not a null to be dropped: "we have never seen this place before" is a fact
#: about the establishment, and profile 8 shows it is the same fact the null-rule family
#: indicators encode.
UNKNOWN = "__UNKNOWN__"

#: Raw geographic columns, with the concept each names. The two ``ward`` entries are the
#: point of profile 2: the dataset publishes a historical ward layer *and* a current one,
#: which is itself evidence that a ward identifier is not stable across vintages.
RAW_GEOGRAPHY: tuple[tuple[str, str], ...] = (
    ("zip", "postal code, as typed on the inspection record"),
    (":@computed_region_vrxf_vc4k", "community area (Socrata spatial join)"),
    (":@computed_region_6mkv_f3dw", "ZIP-code region (Socrata spatial join)"),
    (":@computed_region_43wa_7qmu", "ward, current boundaries"),
    (":@computed_region_awaf_s7ux", "ward, 2003-2015 boundaries"),
    (":@computed_region_bdys_3d7i", "census tract"),
    ("latitude", "point geography"),
    ("longitude", "point geography"),
    ("address", "street address"),
    ("city", "municipality, as typed"),
    ("state", "state, as typed"),
)

#: One model carries the whole support question: every calibrated model scores an identical
#: id set, so support is a property of the fold and the group rather than of the estimator.
#: Profile 3 asserts that rather than assuming it.
PROBE_MODEL = "xgboost_platt"

#: Candidate row floors swept in profile 4. The chosen values go into ADR 0034.
SUPPORT_GRID: tuple[int, ...] = (30, 50, 100, 200, 300, 500)

#: Candidate class floors swept in profile 4.
CLASS_GRID: tuple[int, ...] = (5, 10, 20, 30)

#: Equal-mass bins used by ``evaluation.metrics.ece``. Not a free parameter here: profile 4
#: asks what row count that bin count implies, rather than reducing it, so a group ECE stays
#: comparable with Component 9's global one.
CALIBRATION_BINS = 15

#: Rows shown per table where a full listing would be 78 lines of noise.
DISPLAY_ROWS = 12

BREAK = "\n\n"


# --- shared helpers ----------------------------------------------------------


def _folds(frame: pl.DataFrame) -> list[FoldSpec]:
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise SystemExit("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    return [*quarterly, *folds_module.covid_shift_fold(data_end=end)]


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def _as_float(value: object) -> float:
    """Coerce a polars aggregate to a float.

    ``Series.min()`` and friends are typed as a union of every dtype polars can hold, so a
    bare ``float(...)`` does not type-check under strict mode. Narrowing here keeps the
    conversion in one place rather than scattering ignores through the profiles.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected a numeric aggregate, got {type(value).__name__}")


def _both_classes(pooled: pl.DataFrame, floor: int) -> int:
    """Groups holding at least ``floor`` positives *and* ``floor`` negatives.

    Both, because a metric can be undefined for either reason: ROC-AUC needs one of each,
    and a group of 400 rows with three negatives supports nothing useful about ranking.
    """
    return pooled.filter((pl.col("positive") >= floor) & (pl.col("negative") >= floor)).height


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return dict.fromkeys(("min", "p25", "median", "p75", "p90", "max"), 0.0)
    return {
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _audited(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One probe model's scored rows, with the group frame and the label attached.

    This is the frame the whole audit rests on, assembled the way the implementation will
    assemble it: predictions joined to the as-of group frame and to the label, all on
    ``target_inspection_id``, which is Component 3's key and the raw ``inspection_id`` at
    the same time.
    """
    predictions = sources["predictions"].filter(pl.col("model_name") == PROBE_MODEL)
    if predictions.is_empty():
        raise SystemExit(f"probe model {PROBE_MODEL} absent from the calibrated artifact")
    groups = sources["categoricals"].select("target_inspection_id", *GROUP_COLUMNS)
    labels = sources["features"].select("target_inspection_id", "target")
    return predictions.join(groups, on="target_inspection_id", how="left").join(
        labels, on="target_inspection_id", how="left"
    )


# --- profiles ----------------------------------------------------------------


def group_source_inventory(sources: dict[str, pl.DataFrame]) -> str:
    """Every geographic column in the raw snapshot, and where it survives to.

    The point is that geography does *not* survive into the model frame. Component 4's table
    is 26 numeric history features and carries no location of any kind, so a disparity found
    here cannot be a case of the model reading a group label off the row. That is a fact
    worth measuring rather than asserting, and this profile measures it.
    """
    raw = sources["raw"]
    feature_columns = set(sources["features"].columns)
    categorical_columns = set(sources["categoricals"].columns)

    rows: list[list[str]] = []
    for column, concept in RAW_GEOGRAPHY:
        if column not in raw.columns:
            rows.append([f"`{column}`", concept, "absent", "-", "-", "-"])
            continue
        series = raw.get_column(column)
        rows.append(
            [
                f"`{column}`",
                concept,
                f"{series.n_unique():,}",
                f"{series.null_count():,}",
                "yes" if column in feature_columns else "**no**",
                "yes" if column in categorical_columns else "no",
            ]
        )

    body = _table(
        ["raw column", "concept", "distinct", "nulls", "a model feature?", "in the C8 layer?"],
        rows,
    )
    note = (
        f"Raw snapshot: {raw.height:,} rows, {raw.width} columns. Component 4's feature table "
        f"has {len(feature_columns - {'rd'})} columns and **not one of them is geographic** -- "
        "so no model in this project can read a group label off the row it is scoring. "
        "Component 8's as-of layer carries `community_area` and `zip` under ADR 0022, and "
        "that layer is the only place either survives past raw."
        + BREAK
        + "**This is safety by absence, not by design, and it proves nothing about "
        "fairness.** A model with no geographic input can still behave differently across "
        "geography, because the 26 features it does see -- inspection history, recency, "
        "citation rates -- are themselves distributed unevenly across the city. Fairness "
        "through unawareness is not fairness; it is the reason this component measures "
        "behaviour rather than inspecting the feature list."
    )
    return body + BREAK + note


def group_temporal_stability(sources: dict[str, pl.DataFrame]) -> str:
    """Is the as-of group value the same as the value recorded on the row itself?

    The as-of value comes from the establishment's most recent *earlier* inspection, so it
    is temporally safe by construction. The row's own value is contemporaneous and would be
    defensible for an evaluation-only field. This profile measures whether the safe choice
    costs anything -- and separately, whether a ward identifier is stable enough to use.
    """
    raw = sources["raw"]
    categoricals = sources["categoricals"]

    own = raw.select(
        pl.col("inspection_id").alias("target_inspection_id"),
        pl.col(":@computed_region_vrxf_vc4k").alias("own_community_area"),
        pl.col("zip").str.slice(0, 5).alias("own_zip"),
        pl.col(":@computed_region_43wa_7qmu").alias("ward_current"),
        pl.col(":@computed_region_awaf_s7ux").alias("ward_historical"),
    )
    joined = categoricals.join(own, on="target_inspection_id", how="left")

    rows: list[list[str]] = []
    for column, own_column in (("community_area", "own_community_area"), ("zip", "own_zip")):
        comparable = joined.filter(pl.col(column).ne(UNKNOWN) & pl.col(own_column).is_not_null())
        disagree = comparable.filter(pl.col(column) != pl.col(own_column)).height
        rows.append(
            [
                f"`{column}`",
                f"{comparable.height:,}",
                f"**{disagree:,}**",
                _fmt(disagree / comparable.height if comparable.height else 0.0, 6),
            ]
        )
    agreement = _table(["group definition", "rows where both exist", "disagreements", "rate"], rows)

    ward = joined.filter(
        pl.col("ward_current").is_not_null() & pl.col("ward_historical").is_not_null()
    )
    ward_disagree = ward.filter(pl.col("ward_current") != pl.col("ward_historical")).height
    ward_rate = ward_disagree / ward.height if ward.height else 0.0

    lags = categoricals.get_column("days_since_source").drop_nulls()
    lag_stats = _quantiles([int(v) for v in lags.to_list()])

    unknown_rows: list[list[str]] = []
    for column in GROUP_COLUMNS:
        unknown = categoricals.filter(pl.col(column) == UNKNOWN).height
        unknown_rows.append(
            [
                f"`{column}`",
                f"{categoricals.get_column(column).n_unique():,}",
                f"{unknown:,}",
                _fmt(unknown / categoricals.height, 4),
            ]
        )
    coverage = _table(["group definition", "distinct values", UNKNOWN, "rate"], unknown_rows)

    reading = (
        "**The as-of value and the row's own value never disagree.** Community area and ZIP "
        "are attributes of a fixed premises, so carrying the last observed value forward "
        "reproduces the contemporaneous one exactly wherever both exist. The temporally safe "
        "choice therefore costs nothing in accuracy, and it reuses a frame Component 8 "
        "already validates strictly as-of on every row "
        f"(minimum observed lag {lag_stats['min']:.0f} day, median {lag_stats['median']:.0f}). "
        "A zero-day lag would mean a row had supplied its own attributes."
        + BREAK
        + "**Ward fails the same test.** The two published ward layers assign different region "
        f"ids to **{ward_disagree:,} of {ward.height:,}** rows ({_fmt(ward_rate, 4)}). The "
        "dataset publishing two ward vintages at all is the point: a ward identifier is a "
        "property of a boundary version, not of a place, so attaching the current ward to a "
        "2019 row attributes that row to a district that did not exist when it was inspected. "
        "Ward is refused. Chicago's 77 community areas have been fixed since the 1920s, which "
        "is exactly why they are the unit the city publishes statistics against."
    )
    return (
        "**As-of geography versus the row's own recorded geography**"
        + BREAK
        + agreement
        + BREAK
        + "**Coverage of the as-of frame**"
        + BREAK
        + coverage
        + BREAK
        + reading
    )


def group_join_integrity(sources: dict[str, pl.DataFrame]) -> str:
    """Is the predictions-to-group join total, one-to-one and unambiguous?

    A fairness audit that quietly dropped rows on a join would report metrics over a
    population nobody chose. Four things are checked: the group frame has one row per key,
    every scored id is present in it, every model scores the same ids, and no key maps to
    two group values.
    """
    predictions = sources["predictions"]
    categoricals = sources["categoricals"]

    keys = categoricals.get_column("target_inspection_id")
    duplicate_keys = keys.len() - keys.n_unique()

    scored = set(predictions.get_column("target_inspection_id").to_list())
    missing = len(scored - set(keys.to_list()))

    per_model = (
        predictions.group_by("model_name")
        .agg(pl.col("target_inspection_id").n_unique().alias("ids"))
        .sort("model_name")
    )
    identical_id_sets = per_model.get_column("ids").n_unique() == 1

    ambiguous = 0
    for column in GROUP_COLUMNS:
        counts = categoricals.group_by("target_inspection_id").agg(
            pl.col(column).n_unique().alias("values")
        )
        ambiguous += counts.filter(pl.col("values") > 1).height

    rows = [
        ["group frame rows", f"{categoricals.height:,}"],
        ["group frame distinct keys", f"{keys.n_unique():,}"],
        ["duplicate keys in the group frame", f"**{duplicate_keys}**"],
        ["distinct scored ids", f"{len(scored):,}"],
        ["scored ids absent from the group frame", f"**{missing}**"],
        ["models in the calibrated artifact", f"{per_model.height}"],
        ["every model scores an identical id set", "yes" if identical_id_sets else "**no**"],
        ["keys mapping to more than one group value", f"**{ambiguous}**"],
    ]
    note = (
        "The join key is `target_inspection_id`, which is Component 3's identifier, "
        "Component 4's primary key and the raw Socrata `inspection_id` at the same time. "
        "That is why no key has to be invented for this audit and why the join is exact "
        "rather than approximate. Every model scoring an identical id set is what makes a "
        "cross-model comparison here a comparison of models rather than of populations."
    )
    return _table(["property", "measured"], rows) + BREAK + note


def group_support_population(sources: dict[str, pl.DataFrame]) -> str:
    """How much data does each (fold, group) cell actually hold?

    This is the profile that decides the shape of the whole component. If the per-fold cells
    are large, a fairness table can be per fold. If they are not, the honest reporting grain
    is the pooled fold set and the per-fold table exists mainly to make the shortage visible.
    """
    audited = _audited(sources).filter(pl.col("fold_set") == "quarterly")

    sections: list[str] = []
    for column in GROUP_COLUMNS:
        cells = audited.group_by("fold_id", column).agg(
            pl.len().alias("n"), pl.col("target").sum().alias("positive")
        )
        cell_stats = _quantiles([int(v) for v in cells.get_column("n").to_list()])
        pooled = audited.group_by(column).agg(
            pl.len().alias("n"), pl.col("target").sum().alias("positive")
        )
        pooled = pooled.with_columns((pl.col("n") - pl.col("positive")).alias("negative"))

        distribution = _table(
            ["grain", "cells", "min", "p25", "median", "p75", "p90", "max"],
            [
                [
                    "per (fold, group)",
                    f"{cells.height:,}",
                    f"{cell_stats['min']:.0f}",
                    f"{cell_stats['p25']:.0f}",
                    f"**{cell_stats['median']:.0f}**",
                    f"{cell_stats['p75']:.0f}",
                    f"{cell_stats['p90']:.0f}",
                    f"{cell_stats['max']:.0f}",
                ]
            ],
        )

        sweep_rows = [
            [
                f"{floor}",
                f"{cells.filter(pl.col('n') >= floor).height:,} / {cells.height:,}",
                f"**{pooled.filter(pl.col('n') >= floor).height} / {pooled.height}**",
            ]
            for floor in SUPPORT_GRID
        ]
        sweep = _table(
            ["row floor", "per-fold cells clearing it", "pooled groups clearing it"], sweep_rows
        )

        class_rows = [
            [f"{floor}", f"{_both_classes(pooled, floor)} / {pooled.height}"]
            for floor in CLASS_GRID
        ]
        classes = _table(["min positives and negatives", "pooled groups clearing it"], class_rows)

        sections.append(
            f"**`{column}`** -- {pooled.height} distinct values over "
            f"{audited.height:,} quarterly test rows"
            + BREAK
            + distribution
            + BREAK
            + sweep
            + BREAK
            + classes
        )

    implied = CALIBRATION_BINS * 20
    reading = (
        "**The per-fold grain does not support group metrics, and the pooled grain does.** "
        "The median (fold, community area) cell is a couple of dozen rows; an ROC-AUC over "
        "that is noise wearing four decimal places, and an ECE over it is undefined in "
        "practice. Pooled over the 17 quarterly test windows the same groups become "
        "measurable. Both grains are computed and persisted anyway -- the per-fold table is "
        "what makes the shortage visible instead of hiding it, and it is what the drift "
        "analysis reads."
        + BREAK
        + "Pooling is legitimate here and is not the leak ADR 0025 forbids. Every pooled row "
        "is still strictly held out for its own fold: it was scored by a model that never saw "
        "it. What pooling does cost is that the 17 windows were scored by 17 different fitted "
        "models, so a pooled number describes *the system as operated over 2022Q2-2026Q2* "
        "rather than one estimator. That is the honest label and it travels on every pooled "
        "row rather than in a footnote."
        + BREAK
        + "**The calibration floor is arithmetic, not taste.** `evaluation.metrics.ece` uses "
        f"{CALIBRATION_BINS} equal-mass bins. Component 9 recorded 27-50 rows per bin as "
        f"already thin for a selection rule. Twenty rows per bin needs {implied} rows, which "
        "is the floor the group calibration gate is set at. The bin count is **not** reduced "
        "to make more groups qualify, because a group ECE at a different bin count would no "
        "longer be comparable with Component 9's global one -- and the whole question in "
        "section 18 is whether the global improvement reached the groups."
    )
    return "\n\n".join(sections) + BREAK + reading


def group_outcome_rates(sources: dict[str, pl.DataFrame]) -> str:
    """How far apart are the groups' outcome rates, before any model is involved?

    This is a property of the city and its inspection programme, not of Sentinel. It is
    measured first precisely so that a later disparity in model behaviour can be read
    against it rather than mistaken for it.
    """
    audited = _audited(sources).filter(pl.col("fold_set") == "quarterly")
    overall = _as_float(audited.get_column("target").mean())

    sections: list[str] = []
    for column in GROUP_COLUMNS:
        pooled = (
            audited.group_by(column)
            .agg(pl.len().alias("n"), pl.col("target").sum().alias("positive"))
            .with_columns((pl.col("positive") / pl.col("n")).alias("rate"))
        )
        supported = pooled.filter(pl.col("n") >= 200).sort("rate")
        if supported.is_empty():
            sections.append(f"**`{column}`** -- no group reaches the 200-row floor.")
            continue
        listing = [*supported.head(5).to_dicts(), *supported.tail(5).reverse().to_dicts()]
        rows = [
            [f"`{r[column]}`", f"{r['n']:,}", f"{r['positive']:,}", _fmt(_as_float(r["rate"]))]
            for r in listing
        ]
        lowest = _as_float(supported.get_column("rate").min())
        highest = _as_float(supported.get_column("rate").max())
        sections.append(
            f"**`{column}`** -- {supported.height} groups at or above 200 rows; outcome rate "
            f"spans **{_fmt(lowest)} to {_fmt(highest)}** against an overall {_fmt(overall)}, "
            f"a spread of {_fmt(highest - lowest)}"
            + BREAK
            + _table(["group", "rows", "positives", "outcome rate"], rows)
        )

    reading = (
        "**A large group disparity exists in the outcomes themselves.** Whatever Sentinel "
        "does, it is ranking a population whose measured citation rate differs by more than "
        "thirty points between neighbourhoods. Two things follow and both matter."
        + BREAK
        + "First, a difference in *selection rate* across these groups is the expected "
        "behaviour of a working risk model, not evidence of a defect. Equal selection rates "
        "would require ignoring a real difference in measured outcomes. This is why the "
        "component reports selection rate and capture separately and refuses to collapse "
        "either into a verdict."
        + BREAK
        + "Second -- and this is ADR 0019's consequence, inherited here rather than discovered "
        "-- the outcome being measured is that a violation was **cited**, not that an "
        "establishment was unsafe. The dataset publishes no inspector identifier, so the gap "
        "between those two contains inspector variation that nothing in this project can "
        "quantify. A neighbourhood-level difference in citation rate cannot be decomposed "
        "into establishment risk versus differential inspection practice. That limitation "
        "bounds every number in this component."
    )
    return "\n\n".join(sections) + BREAK + reading


def representation_drift(sources: dict[str, pl.DataFrame]) -> str:
    """Does each group's share of the evaluated population move across the folds?

    If it does, a disparity that appears to emerge over time may be a change in who is being
    evaluated rather than a change in how the model treats them.
    """
    audited = _audited(sources).filter(pl.col("fold_set") == "quarterly")
    totals = audited.group_by("fold_id").agg(pl.len().alias("total"))
    n_folds = totals.height

    sections: list[str] = []
    for column in GROUP_COLUMNS:
        shares = (
            audited.group_by("fold_id", column)
            .agg(pl.len().alias("n"))
            .join(totals, on="fold_id")
            .with_columns((pl.col("n") / pl.col("total")).alias("share"))
        )
        summary = (
            shares.group_by(column)
            .agg(
                pl.col("share").min().alias("min_share"),
                pl.col("share").max().alias("max_share"),
                pl.col("share").mean().alias("mean_share"),
                pl.len().alias("folds"),
            )
            .with_columns((pl.col("max_share") - pl.col("min_share")).alias("travel"))
            .sort("travel", descending=True)
        )
        rows = [
            [
                f"`{r[column]}`",
                f"{r['folds']}",
                _fmt(_as_float(r["mean_share"])),
                _fmt(_as_float(r["min_share"])),
                _fmt(_as_float(r["max_share"])),
                f"**{_fmt(_as_float(r['travel']))}**",
            ]
            for r in summary.head(DISPLAY_ROWS).to_dicts()
        ]
        absent = summary.filter(pl.col("folds") < n_folds).height
        sections.append(
            f"**`{column}`** -- {absent} of {summary.height} groups are absent from at least "
            f"one of the {n_folds} folds entirely; the largest share travel:"
            + BREAK
            + _table(["group", "folds present", "mean share", "min", "max", "travel"], rows)
        )

    reading = (
        "**The evaluated population is not stationary.** Groups appear and disappear between "
        "quarters and their shares move by several points. Any drift in a group metric must "
        "therefore be read beside this table: a group whose share halved and whose measured "
        "ECE moved has two candidate explanations, and this component can separate neither. "
        "Reporting the drift in the model's behaviour without the drift in its evaluated "
        "population would attribute one to the other for free."
    )
    return "\n\n".join(sections) + BREAK + reading


def capacity_and_k(sources: dict[str, pl.DataFrame]) -> str:
    """What top-k levels are meaningful, given the capacity the city actually worked?

    Component 5 already derives k from the observed inspection calendar rather than choosing
    it. This profile reuses that derivation so the priority audit inherits it instead of
    inventing a second answer.
    """
    frame = sources["features"]

    rows: list[list[str]] = []
    for fold in _folds(frame):
        stats = folds_module.fold_stats(frame, fold)
        test = folds_module.window_frame(frame, fold)
        if test.is_empty():
            continue
        window = simulate.build_window(
            ids=test["target_inspection_id"].to_list(),
            labels=test["target"].to_list(),
            dates=test["rd"].to_list(),
        )
        median_daily = max(1, int(stats.test_median_daily_capacity or 1))
        k_values = simulate.capacity_k_values(window, median_daily=median_daily)
        rows.append(
            [
                fold.fold_id,
                f"{window.n:,}",
                f"{median_daily}",
                f"{k_values['k_1_day']}",
                f"{k_values['k_1_week']}",
                f"{k_values['k_pct_01']}",
                f"{k_values['k_pct_05']}",
                f"{k_values['k_pct_10']}",
            ]
        )

    body = _table(
        [
            "fold",
            "test rows",
            "median daily",
            "k_1_day",
            "k_1_week",
            "k_pct_01",
            "k_pct_05",
            "k_pct_10",
        ],
        rows,
    )
    reading = (
        "**Both families are kept, and no probability threshold is introduced.** The "
        "percentage cutoffs make groups comparable across folds of very different sizes; the "
        "capacity cutoffs are what the city could actually work in a day and in a week, which "
        "is the operational question. Neither is invented: `capacity_k_values` derives them "
        "from the window's own measured median daily rate, and this component calls that "
        "function rather than reimplementing it."
        + BREAK
        + "A probability cutoff at 0.5 would be a number this project has never derived from "
        "anything, and reporting per-group error rates at one would read as a deployment "
        "policy. Component 13 owns decision policy. Every threshold figure here is labelled a "
        "descriptive threshold audit, and a probability threshold is refused in prose rather "
        "than declared and left unreachable."
    )
    return body + BREAK + reading


def missingness_by_group(sources: dict[str, pl.DataFrame]) -> str:
    """Is null-rule family missingness distributed evenly across groups?

    Component 11 found `missing_no_code_era_canvass` to be a top-three signal for two of four
    models -- the *absence* of a record is among the most informative things available. If
    absence is also distributed unevenly across neighbourhoods, then data availability is a
    fairness surface in its own right. That is a measurement, not an accusation.
    """
    audited = _audited(sources).filter(pl.col("fold_set") == "quarterly")
    features = sources["features"]

    indicators = [
        (family_indicator_name(rule), indicator_source_column(rule))
        for rule in null_families()
        if indicator_source_column(rule) in features.columns
    ]
    joined = audited.join(
        features.select("target_inspection_id", *[column for _, column in indicators]),
        on="target_inspection_id",
        how="left",
    )

    sections: list[str] = []
    for column in GROUP_COLUMNS:
        rows: list[list[str]] = []
        for indicator, source_column in indicators:
            per_group = (
                joined.group_by(column)
                .agg(pl.len().alias("n"), pl.col(source_column).is_null().mean().alias("rate"))
                .filter(pl.col("n") >= 200)
            )
            if per_group.is_empty():
                continue
            rates = per_group.get_column("rate")
            low = _as_float(rates.min())
            high = _as_float(rates.max())
            rows.append(
                [
                    f"`{indicator}`",
                    _fmt(_as_float(joined.get_column(source_column).is_null().mean())),
                    _fmt(low),
                    _fmt(high),
                    f"**{_fmt(high - low)}**",
                    f"{per_group.height}",
                ]
            )
        sections.append(
            f"**`{column}`**, groups at or above 200 rows"
            + BREAK
            + _table(["indicator", "overall", "min group", "max group", "spread", "groups"], rows)
        )

    unknown_rows = audited.filter(pl.col("community_area") == UNKNOWN).height
    reading = (
        "**Missingness is not evenly distributed, and the unknown-geography group is the "
        f"sharpest case.** {unknown_rows:,} quarterly test rows carry "
        f"`community_area = {UNKNOWN}`, and that group is a superset of the rows with no prior "
        "inspection of any type -- the same rows Component 4 cannot compute a recency for and "
        "the same rows a null-rule family indicator fires on. Geography, data availability and "
        "model reliance meet on one set of rows, which is why `__UNKNOWN__` is treated as a "
        "first-class group here rather than dropped as a null."
        + BREAK
        + "**This is a measurement and nothing more.** A missingness feature is not unfair by "
        "definition: 'we have never inspected this place' is a true and relevant fact, and "
        "removing it would not make the underlying inequality in inspection history go away. "
        "What the audit can say is that the fact is unevenly distributed, and that models "
        "Component 11 measured leaning hard on it are therefore leaning on something that "
        "varies by neighbourhood. It cannot say the model is wrong to, and it cannot say which "
        "way the causation runs."
    )
    return "\n\n".join(sections) + BREAK + reading


def attribution_support(sources: dict[str, pl.DataFrame]) -> str:
    """Does Component 11's bounded sample support a per-group attribution comparison?

    Component 11 explained 300 rows per (model, fold) under a frozen protocol, and re-running
    it at a larger sample to answer this question is explicitly forbidden -- it would change
    the rows every published number rests on. So the question is whether the existing sample
    can be grouped, and at what grain.
    """
    values = sources.get("explanations")
    if values is None:
        return "Component 11's artifact was not supplied; this profile did not run."

    cases = values.select("model_name", "fold_set", "fold_id", "target_inspection_id").unique()
    groups = sources["categoricals"].select("target_inspection_id", *GROUP_COLUMNS)
    joined = cases.join(groups, on="target_inspection_id", how="left").filter(
        pl.col("fold_set") == "quarterly"
    )

    sections: list[str] = []
    for column in GROUP_COLUMNS:
        per_model = joined.group_by("model_name", column).agg(pl.len().alias("n"))
        stats = _quantiles([int(v) for v in per_model.get_column("n").to_list()])
        sweep = [
            [f"{floor}", f"{per_model.filter(pl.col('n') >= floor).height} / {per_model.height}"]
            for floor in (30, 50, 100, 200)
        ]
        sections.append(
            f"**`{column}`** -- explained rows per (model, group), pooled over the quarterly "
            f"folds: median {stats['median']:.0f}, p90 {stats['p90']:.0f}, max {stats['max']:.0f}"
            + BREAK
            + _table(["row floor", "(model, group) cells clearing it"], sweep)
        )

    reading = (
        "**Supportable at the pooled grain, for the best-supported groups only, and "
        "descriptively.** Per fold the sample gives a handful of rows per group and answers "
        "nothing. Pooled over the quarterly folds the larger groups reach a few dozen to a "
        "couple of hundred explained rows, which is enough to compare a mean-absolute-"
        "attribution *profile* -- a 30-feature ranking, whose global statistic Component 11 "
        "measured converging far faster than any individual value (rank rho 0.9964 at 8 "
        "rounds against a 64-round reference)."
        + BREAK
        + "It is not enough for anything stronger, and nothing stronger is claimed. A "
        "difference between two groups' attribution profiles says the model's reliance on "
        "features differs across those populations. It is not evidence of discrimination, it "
        "is not causal, and per ADR 0030 an attribution is not a quality measure in the first "
        "place -- a model can lean hard on a feature that is misleading it, which Component 6 "
        "measured happening under distribution shift."
    )
    return "\n\n".join(sections) + BREAK + reading


def covid_support(sources: dict[str, pl.DataFrame]) -> str:
    """The same support question for `covid_shift`, answered separately and never pooled."""
    audited = _audited(sources)

    rows: list[list[str]] = []
    for column in GROUP_COLUMNS:
        for label in ("quarterly", "covid_shift"):
            frame = audited.filter(pl.col("fold_set") == label)
            pooled = frame.group_by(column).agg(pl.len().alias("n"))
            rows.append(
                [
                    f"`{column}`",
                    label,
                    f"{frame.height:,}",
                    f"{pooled.height}",
                    f"{pooled.filter(pl.col('n') >= 100).height}",
                    f"**{pooled.filter(pl.col('n') >= 200).height}**",
                    f"{pooled.filter(pl.col('n') >= 300).height}",
                    _fmt(_as_float(frame.get_column("target").mean())),
                ]
            )

    body = _table(
        [
            "group definition",
            "fold set",
            "test rows",
            "groups",
            "n>=100",
            "n>=200",
            "n>=300",
            "rate",
        ],
        rows,
    )
    reading = (
        "**COVID is one fold and it is thin.** Its test window is nineteen months rather than "
        "a quarter, so it holds more rows than any single quarterly fold -- and yet far fewer "
        "groups clear the support floor, because the inspection programme was suspended and "
        "restarted and the establishments inspected during it were not a cross-section of the "
        "city."
        + BREAK
        + "It is reported as a separate stress-test observation. No trend is claimed from it, "
        "it is never averaged into a quarterly mean, and a group disparity appearing only "
        "there is an observation about one abnormal period rather than a finding about "
        "Sentinel. Five components have now measured this fold diverging from the quarterly "
        "answer; a sixth divergence would be the expectation, not a surprise."
    )
    return body + BREAK + reading


PROFILES: dict[str, Callable[[dict[str, pl.DataFrame]], str]] = {
    "group_source_inventory": group_source_inventory,
    "group_temporal_stability": group_temporal_stability,
    "group_join_integrity": group_join_integrity,
    "group_support_population": group_support_population,
    "group_outcome_rates": group_outcome_rates,
    "representation_drift": representation_drift,
    "capacity_and_k": capacity_and_k,
    "missingness_by_group": missingness_by_group,
    "attribution_support": attribution_support,
    "covid_support": covid_support,
}


def _load(args: argparse.Namespace) -> tuple[dict[str, pl.DataFrame], list[str]]:
    settings = load_settings()
    features_path = args.features or duckdb_queries.latest_parquet(
        settings.features_processed_dir, prefix="as_of_features_"
    )
    predictions_path = args.calibrated_predictions or duckdb_queries.latest_parquet(
        settings.predictions_processed_dir, prefix="calibrated_predictions_"
    )
    categoricals_path = args.categoricals or duckdb_queries.latest_parquet(
        settings.neural_processed_dir, prefix="neural_categoricals_"
    )
    raw_path = args.raw or duckdb_queries.latest_parquet(
        settings.food_inspections_raw_dir, prefix="food_inspections_"
    )
    sources: dict[str, pl.DataFrame] = {
        "features": pl.read_parquet(features_path).with_columns(
            pl.col("inspection_date").str.to_date().alias("rd")
        ),
        "predictions": pl.read_parquet(predictions_path),
        "categoricals": pl.read_parquet(categoricals_path),
        "raw": pl.read_parquet(raw_path),
    }
    provenance = [
        f"features: {features_path.name}",
        f"predictions: {predictions_path.name}",
        f"categoricals: {categoricals_path.name}",
        f"raw: {raw_path.name}",
    ]
    try:
        explanations_path = args.explanations or duckdb_queries.latest_parquet(
            settings.explanations_processed_dir, prefix="explanation_values_"
        )
    except FileNotFoundError:
        provenance.append("explanations: absent")
    else:
        sources["explanations"] = pl.read_parquet(explanations_path)
        provenance.append(f"explanations: {explanations_path.name}")
    return sources, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Component 4 feature table.")
    parser.add_argument("--calibrated-predictions", type=Path, help="Component 9 artifact.")
    parser.add_argument("--categoricals", type=Path, help="Component 8 as-of categoricals.")
    parser.add_argument("--raw", type=Path, help="Component 1 raw snapshot.")
    parser.add_argument("--explanations", type=Path, help="Component 11 attributions.")
    parser.add_argument("--only", action="append", help="Profile to run; repeatable.")
    args = parser.parse_args(argv)

    requested = args.only or list(PROFILES)
    unknown = [name for name in requested if name not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}")

    sources, provenance = _load(args)

    print("<!-- generated by scripts/profile_fairness.py -->")
    for line in provenance:
        print(f"<!-- {line} -->")
    for name in requested:
        print()
        print(f"### {name}")
        print()
        print(PROFILES[name](sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
