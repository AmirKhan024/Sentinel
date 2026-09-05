"""Loading the authoritative artifacts, and assembling the windows a policy runs over.

The one module here that touches Parquet on the way in. It reads nine closed components'
outputs and produces nothing but typed structures, so every later module can be pure.

**This component fits nothing and re-derives nothing it can read.** Component 5's fold
construction is called rather than reimplemented, and the derived folds are then checked
against Component 5's own published fold table -- so a snapshot that has moved under the
evaluation artifact is caught here rather than producing a queue for windows nobody evaluated.

**The prediction contract is Component 5's.** ``read_predictions`` and ``validate_predictions``
police exact coverage of each test window, no nulls, unique ids and a declared training
horizon that does not run past the fold's calibration end. A policy layer is exactly the place
where a quietly incomplete prediction set would go unnoticed -- a model that dropped the rows
it found hard would produce a shorter queue and a better-looking precision -- so the contract
is enforced on the way in rather than assumed.

**Group labels are read, never derived.** The as-of community area comes from Component 8's
categoricals artifact and the support status from Component 12's published support table. This
component does not construct a group frame: Component 12 owns that, including which
geographies are refused and why, and a second construction here would be a second answer to a
question that already has an authoritative one. What is read is a label and a status, onto the
output rows, for a human reader -- never back into a score.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

import sentinel.evaluation.folds as folds_module
from sentinel.evaluation.contract import read_predictions, validate_predictions
from sentinel.evaluation.models import FoldSpec
from sentinel.policy.definitions import ELIGIBILITY_COLUMN, SECONDARY_FLAG_COLUMN
from sentinel.policy.eligibility import ELIGIBLE_FLAG, SECONDARY_FLAG, annotate
from sentinel.policy.governance import parse_overrides
from sentinel.policy.models import Override, PolicyWindow

#: The reference-date column Component 5's fold helpers expect.
DATE_COLUMN = "rd"

#: The geography Component 13 reports against. One, not both: Component 12 audited community
#: area and ZIP and found them to tell the same story, and a policy artifact that carried two
#: overlapping geographic labels per row would invite a reader to treat their disagreement as
#: information.
GROUP_DEFINITION = "community_area"


class PolicyInputError(ValueError):
    """Raised when an input artifact cannot be trusted enough to build a queue from."""


def load_features(path: Path) -> pl.DataFrame:
    """Component 4's as-of feature table, with the reference date parsed and flags attached.

    Eligibility is decided here, once, at the edge. That is what makes it checkable: the
    validator re-derives the flag from the same column and compares, which it could not do if
    the predicate were applied lazily somewhere inside the allocator.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    frame = pl.read_parquet(path)
    required = (
        "target_inspection_id",
        "establishment_id",
        "inspection_date",
        "target",
        ELIGIBILITY_COLUMN,
        SECONDARY_FLAG_COLUMN,
    )
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise PolicyInputError(
            f"{path.name}: feature table is missing {', '.join(missing)}. The policy layer "
            "needs the as-of history columns, not only the scores"
        )
    frame = frame.with_columns(pl.col("inspection_date").str.to_date().alias(DATE_COLUMN))
    return annotate(frame)


def load_folds(features: pl.DataFrame, published: Path | None = None) -> list[FoldSpec]:
    """Component 5's folds, derived by its own helpers and checked against its own artifact.

    Derived rather than parsed, so the folds a policy is built for are constructed by the code
    that defines what a fold is. Checked against the published table, so a feature snapshot
    that has drifted away from the evaluation run is an error here instead of a queue for
    windows that were never evaluated.
    """
    start = folds_module.min_date(features, DATE_COLUMN)
    end = folds_module.max_date(features, DATE_COLUMN)
    if start is None or end is None:
        raise PolicyInputError("feature table has no usable reference dates")
    derived = [
        *folds_module.quarterly_folds(data_start=start, data_end=end),
        *folds_module.covid_shift_fold(data_end=end),
    ]
    if published is not None:
        if not published.exists():
            raise FileNotFoundError(f"Evaluation fold table not found: {published}")
        table = pl.read_parquet(published)
        expected = set(table["fold_id"].to_list())
        got = {fold.fold_id for fold in derived}
        if expected != got:
            absent = sorted(expected - got)
            surplus = sorted(got - expected)
            raise PolicyInputError(
                f"folds derived from the feature table do not match {published.name}: "
                f"{len(absent)} published fold(s) missing ({', '.join(absent[:5])}), "
                f"{len(surplus)} extra ({', '.join(surplus[:5])}). The feature snapshot and "
                "the evaluation run describe different periods"
            )
    return derived


def median_daily_capacity(features: pl.DataFrame, fold: FoldSpec) -> int:
    """The window's measured median daily inspection rate, floored at one.

    Component 5's ``fold_stats`` computes it -- the median number of inspections Chicago
    actually performed on a day inside the test window. Every capacity cutoff descends from
    this number, so inventing one here would make the whole policy grid an assumption.
    """
    stats = folds_module.fold_stats(features, fold, date_column=DATE_COLUMN)
    return max(1, int(stats.test_median_daily_capacity or 1))


def load_predictions(path: Path, *, model_name: str) -> pl.DataFrame:
    """One calibrated model's scores, with the base score kept beside the calibrated one."""
    if not path.exists():
        raise FileNotFoundError(f"Calibrated predictions not found: {path}")
    frame = pl.read_parquet(path)
    required = ("target_inspection_id", "score", "base_score", "model_name", "fold_id")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise PolicyInputError(f"{path.name}: missing {', '.join(missing)}")
    subset = frame.filter(pl.col("model_name") == model_name)
    if subset.is_empty():
        available = ", ".join(sorted(frame["model_name"].unique().to_list()))
        raise PolicyInputError(
            f"{path.name}: model {model_name!r} is absent. Available: {available}"
        )
    return subset


def enforce_prediction_contract(
    path: Path, folds: Sequence[FoldSpec], features: pl.DataFrame, *, model_name: str
) -> int:
    """Offer every prediction set to Component 5's contract, fold by fold.

    Returns the number of sets accepted. Nothing is scored here; the point is that a policy is
    never built from a prediction artifact the evaluator would have refused. A model that
    silently dropped its hard rows would shorten the universe, shorten the queue, and improve
    every number in this component for a reason that has nothing to do with being better.
    """
    by_id = {fold.fold_id: fold for fold in folds}
    accepted = 0
    for prediction_set in read_predictions(path):
        if prediction_set.model_name != model_name:
            continue
        fold = by_id.get(prediction_set.fold_id)
        if fold is None:
            raise PolicyInputError(
                f"{prediction_set.model_name}: scores fold {prediction_set.fold_id!r}, which "
                "the feature table does not produce"
            )
        window = folds_module.window_frame(features, fold, date_column=DATE_COLUMN)
        validate_predictions(prediction_set, fold, window["target_inspection_id"].to_list())
        accepted += 1
    if accepted == 0:
        raise PolicyInputError(f"{path.name}: no prediction set for model {model_name!r}")
    return accepted


def load_group_labels(path: Path | None) -> dict[str, str]:
    """As-of community area per scored row, from Component 8's categoricals artifact.

    Read, not constructed. Component 12 owns what a group frame is, which geographies are
    admissible and why ward is refused; this reads the one column it audits, for annotation.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"As-of categoricals not found: {path}")
    frame = pl.read_parquet(path)
    if GROUP_DEFINITION not in frame.columns:
        raise PolicyInputError(f"{path.name}: no {GROUP_DEFINITION!r} column")
    return dict(
        zip(
            frame["target_inspection_id"].to_list(),
            frame[GROUP_DEFINITION].to_list(),
            strict=True,
        )
    )


def load_group_support(path: Path | None, *, fold_set: str) -> dict[str, str]:
    """Component 12's ranking-support status per group, at the pooled fold-set grain.

    The pooled grain rather than the per-fold one, deliberately. Component 12 measured that
    the median (fold, community area) cell holds 16 rows and that 4 of 1,288 clear the support
    floor -- so a per-fold status would mark almost everything unsupported and the warning
    would carry no information. The pooled status answers the question a reviewer is actually
    asking: *has anybody ever been able to measure how the model behaves here?*
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Fairness support table not found: {path}")
    frame = pl.read_parquet(path).filter(
        (pl.col("group_definition") == GROUP_DEFINITION)
        & (pl.col("grain") == "fold_set")
        & (pl.col("fold_set") == fold_set)
    )
    return dict(zip(frame["group_value"].to_list(), frame["ranking_status"].to_list(), strict=True))


def read_override_file(path: Path | None) -> list[Override]:
    """Decode and validate a human override file, or return nothing.

    JSON rather than Parquet because a person edits it. A list of objects, each carrying every
    required field; the whole file is refused if any row is malformed, because a partially
    applied override file produces a queue nobody authorised.
    """
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Override file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyInputError(f"{path.name}: not valid JSON -- {exc}") from exc
    if not isinstance(payload, list):
        raise PolicyInputError(
            f"{path.name}: the override contract is a JSON list of override objects, got "
            f"{type(payload).__name__}"
        )
    return parse_overrides(payload)


def build_window(
    features: pl.DataFrame,
    predictions: pl.DataFrame,
    fold: FoldSpec,
    *,
    median_daily: int,
) -> PolicyWindow:
    """One fold's scored test window, in Component 5's canonical order.

    The join is inner on ``target_inspection_id`` and then checked for exact coverage, rather
    than left-joined and tolerated. A left join would put a null score on any row the model did
    not cover, and a null score sorted anywhere at all is an establishment being ranked by an
    accident of the sort implementation.
    """
    window = folds_module.window_frame(features, fold, date_column=DATE_COLUMN)
    scored = window.join(
        predictions.select("target_inspection_id", "score", "base_score"),
        on="target_inspection_id",
        how="inner",
    ).sort([DATE_COLUMN, "target_inspection_id"])
    if scored.height != window.height:
        raise PolicyInputError(
            f"{fold.fold_id}: {window.height} rows in the test window but {scored.height} "
            "carry a score. The policy layer never imputes a missing score -- a gap would "
            "silently rank that establishment last and call it a recommendation"
        )
    if scored.is_empty():
        raise PolicyInputError(f"{fold.fold_id}: empty test window")

    return PolicyWindow(
        fold_set=fold.fold_set,
        fold_id=fold.fold_id,
        ids=tuple(scored["target_inspection_id"].to_list()),
        scores=tuple(float(v) for v in scored["score"].to_list()),
        base_scores=tuple(float(v) for v in scored["base_score"].to_list()),
        labels=tuple(int(v) for v in scored["target"].to_list()),
        dates=tuple(scored[DATE_COLUMN].to_list()),
        eligible=tuple(bool(v) for v in scored[ELIGIBLE_FLAG].to_list()),
        secondary_no_history=tuple(bool(v) for v in scored[SECONDARY_FLAG].to_list()),
        median_daily_capacity=median_daily,
    )


def establishment_ids(features: pl.DataFrame) -> dict[str, str]:
    """Stable establishment identifier per scored row, for the recommendation artifact.

    A queue keyed only by ``target_inspection_id`` would name an *opportunity* rather than a
    business, and the person reading a recommendation wants to know which establishment it is.
    """
    return dict(
        zip(
            features["target_inspection_id"].to_list(),
            features["establishment_id"].to_list(),
            strict=True,
        )
    )


__all__ = [
    "DATE_COLUMN",
    "GROUP_DEFINITION",
    "PolicyInputError",
    "build_window",
    "enforce_prediction_contract",
    "establishment_ids",
    "load_features",
    "load_folds",
    "load_group_labels",
    "load_group_support",
    "load_predictions",
    "median_daily_capacity",
    "read_override_file",
]
