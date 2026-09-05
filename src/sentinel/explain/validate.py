"""Every check an explanation run must pass.

The checks fall into four groups, and the grouping is the argument for why this component
can be trusted at all.

**Identity.** Did we explain the model that actually produced the committed prediction?
That is the bit-identity gate, re-executed fits compared to the committed artifact with
``==``. Everything else is downstream of it: an attribution computed on a model no artifact
contains explains nothing, however tidy its arithmetic.

**Alignment.** Is each value attached to the right feature, the right establishment, the
right fold and the right model? This is the group that matters most and looks least
interesting, because every failure in it produces an artifact that passes arithmetic
checks. Component 6 and 7 order the same thirty columns differently at 19 of 30 positions;
a sum is invariant to a permutation of its terms; and a mis-joined row is internally
consistent about the wrong establishment.

**Arithmetic.** Does ``base + sum(phi)`` reconstruct the model's output? Real, and
necessary, and *weaker than it looks* -- the permutation method's path telescopes, so its
additivity holds at any round count and is not evidence that its values are accurate. Said
here as well as in ``attribute`` because a reader of a passing report will otherwise draw
the stronger conclusion.

**Temporal safety.** Did any reference row post-date the horizon of the model it is a
reference for? A background is part of an explanation, and a background containing future
rows answers a counterfactual the model was never asked. The check re-derives every date
from the feature frame rather than reading a recorded field, because a field can be written
by the same bug that would need to be caught.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from sentinel.evaluation import folds as folds_module
from sentinel.evaluation.models import FoldSpec
from sentinel.explain.background import background_ids, background_is_safe
from sentinel.explain.definitions import (
    KNOWN_FEATURE_NAMES,
    SUPPORTED_MODELS,
    ExplanationStatus,
    OutputSpace,
    spec_for,
    tolerance_for,
)
from sentinel.explain.models import (
    ExplanationSample,
    FoldAttribution,
    RefitModel,
    ReproductionOutcome,
    ValidationCheck,
)

logger = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"
MAX_OFFENDERS = 20

#: Names that look machine-generated rather than meaningful. The brief's example is
#: ``feature_127``; this also catches ``f12``, ``x_3``, ``column_9`` and a bare index.
#: Belt and braces beside the ``KNOWN_FEATURE_NAMES`` membership test, because membership
#: alone would silently start passing if someone ever added a generated name to the map.
ANONYMOUS_NAME = re.compile(r"^(?:f|x|v|col(?:umn)?|feature)?[_-]?\d+$", re.IGNORECASE)


def _check(
    name: str,
    passed: bool,
    detail: str,
    *,
    severity: str = SEVERITY_ERROR,
    offenders: Sequence[str] = (),
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail,
        offenders=tuple(offenders[:MAX_OFFENDERS]),
    )


# --- identity ----------------------------------------------------------------


def regenerated_scores_reproduce_the_committed_artifact(
    outcomes: Sequence[ReproductionOutcome],
) -> ValidationCheck:
    """The gate ADR 0029 rests on: is this the model the artifact was written by?

    ``==`` on floats, via ``calibration.basescores.reproduction_mismatches``. A tolerance
    would convert the one check that makes re-execution safe into a check that passes when
    the models differ.
    """
    if not outcomes:
        return _check(
            "regenerated_scores_reproduce_the_committed_artifact",
            False,
            "no (model, fold) was compared against a committed artifact, so nothing "
            "establishes that the explained models are the committed ones",
        )
    total = sum(o.rows for o in outcomes)
    bad = sum(o.mismatches for o in outcomes)
    offenders = [line for o in outcomes for line in o.offenders]
    return _check(
        "regenerated_scores_reproduce_the_committed_artifact",
        bad == 0,
        f"{total - bad}/{total} re-executed test scores are bit-identical to the committed "
        f"Component 6/7/8 artifacts across {len(outcomes)} (model, fold) pairs",
        offenders=offenders,
    )


def committed_prediction_artifacts_are_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> ValidationCheck:
    """Component 11 read the prediction artifacts and rewrote none of them.

    Checksummed before the run and again after everything was written, so "predictions are
    unchanged" is a measurement rather than an assurance.
    """
    offenders = [
        f"{name}: {before[name][:16]}... -> {after.get(name, 'missing')[:16]}..."
        for name in sorted(before)
        if after.get(name) != before[name]
    ]
    return _check(
        "committed_prediction_artifacts_are_unchanged",
        not offenders,
        f"{len(before) - len(offenders)}/{len(before)} prediction artifacts have the same "
        "sha256 after the run as before it",
        offenders=offenders,
    )


def every_explanation_maps_to_a_committed_prediction(
    values: pl.DataFrame, committed: pl.DataFrame
) -> ValidationCheck:
    """Every ``(model, fold, inspection)`` explained exists in a committed artifact.

    An explanation of a prediction nobody made is the failure mode this component could
    ship without anyone noticing: the values would be finite, additive and about a row that
    does not exist downstream.
    """
    if values.height == 0:
        return _check(
            "every_explanation_maps_to_a_committed_prediction",
            False,
            "no explanation rows were produced",
        )
    key = ["model_name", "fold_id", "target_inspection_id"]
    explained = values.select(key).unique()
    known = committed.select(key).unique().with_columns(pl.lit(True).alias("_found"))
    joined = explained.join(known, on=key, how="left")
    orphans = joined.filter(pl.col("_found").is_null())
    offenders = [
        f"{r['model_name']}/{r['fold_id']}/{r['target_inspection_id']}"
        for r in orphans.head(MAX_OFFENDERS).to_dicts()
    ]
    return _check(
        "every_explanation_maps_to_a_committed_prediction",
        orphans.height == 0,
        f"{explained.height - orphans.height}/{explained.height} explained predictions "
        "appear in a committed prediction artifact",
        offenders=offenders,
    )


def prediction_values_match_the_committed_scores(
    cases: pl.DataFrame, committed: pl.DataFrame
) -> ValidationCheck:
    """The probability recorded beside an explanation is the committed one, bit for bit.

    Distinct from the reproduction gate, which compares the *whole* test window as the fit
    produced it. This compares what was actually written into the explanation artifact, and
    so catches a defect the gate cannot: a correct model whose scores were mis-joined onto
    the explained rows on the way out.
    """
    if cases.height == 0:
        return _check("prediction_values_match_the_committed_scores", False, "no explanation cases")
    key = ["model_name", "fold_id", "target_inspection_id"]
    joined = cases.select([*key, "base_score"]).join(
        committed.select([*key, pl.col("score").alias("_committed")]), on=key, how="left"
    )
    bad = joined.filter(
        pl.col("_committed").is_null() | (pl.col("base_score") != pl.col("_committed"))
    )
    offenders = [
        f"{r['model_name']}/{r['fold_id']}/{r['target_inspection_id']}: "
        f"{r['base_score']!r} != {r['_committed']!r}"
        for r in bad.head(MAX_OFFENDERS).to_dicts()
    ]
    return _check(
        "prediction_values_match_the_committed_scores",
        bad.height == 0,
        f"{joined.height - bad.height}/{joined.height} recorded base scores equal the "
        "committed artifact's score exactly",
        offenders=offenders,
    )


def model_identity_is_preserved(values: pl.DataFrame) -> ValidationCheck:
    """Every row names a registered model, at the version its spec declares."""
    if values.height == 0:
        return _check("model_identity_is_preserved", False, "no explanation rows")
    offenders: list[str] = []
    for row in values.select(["model_name", "model_version", "family"]).unique().to_dicts():
        try:
            spec = spec_for(str(row["model_name"]))
        except KeyError:
            offenders.append(f"{row['model_name']}: not in the explain registry")
            continue
        if row["model_version"] != spec.version:
            offenders.append(
                f"{spec.name}: version {row['model_version']!r} != registry {spec.version!r}"
            )
        if row["family"] != spec.family.value:
            offenders.append(
                f"{spec.name}: family {row['family']!r} != registry {spec.family.value!r}"
            )
    return _check(
        "model_identity_is_preserved",
        not offenders,
        "every explanation row names a registered model at its registered version and family",
        offenders=offenders,
    )


# --- alignment ---------------------------------------------------------------


def explanations_belong_to_their_prediction_fold(
    values: pl.DataFrame, folds: Sequence[FoldSpec]
) -> ValidationCheck:
    """Every row's ``fold_set`` is the one its ``fold_id`` belongs to.

    Cheap, and it catches the class of bug where a fold's attributions are written under a
    neighbouring fold's id -- which would pass every arithmetic check and quietly attribute
    one quarter's reasoning to another.
    """
    if values.height == 0:
        return _check("explanations_belong_to_their_prediction_fold", False, "no rows")
    expected = {f.fold_id: f.fold_set for f in folds}
    offenders = [
        f"{r['fold_id']}: fold_set {r['fold_set']!r} != {expected.get(str(r['fold_id']))!r}"
        for r in values.select(["fold_id", "fold_set"]).unique().to_dicts()
        if expected.get(str(r["fold_id"])) != r["fold_set"]
    ]
    return _check(
        "explanations_belong_to_their_prediction_fold",
        not offenders,
        f"every explained fold_id maps to its declared fold_set across {len(expected)} folds",
        offenders=offenders,
    )


def explained_rows_lie_in_the_test_window(
    frame: pl.DataFrame,
    samples: Sequence[ExplanationSample],
    folds: Sequence[FoldSpec],
    *,
    date_column: str = "rd",
) -> ValidationCheck:
    """Every explained row sits inside its fold's test window, re-derived from the frame.

    Not read from a column: the window is rebuilt with ``evaluation.folds.window_frame``,
    the same function every ``score_window`` requires, so this fails if the sampler and the
    fold definition ever disagree.
    """
    by_id = {f.fold_id: f for f in folds}
    offenders: list[str] = []
    checked = 0
    for sample in samples:
        fold = by_id.get(sample.fold_id)
        if fold is None:
            offenders.append(f"{sample.fold_id}: no such fold")
            continue
        window = set(
            str(v)
            for v in folds_module.window_frame(frame, fold, date_column=date_column)[
                "target_inspection_id"
            ].to_list()
        )
        checked += len(sample.ids)
        for row_id in sample.ids:
            if row_id not in window:
                offenders.append(f"{sample.fold_id}/{row_id}: outside the test window")
    return _check(
        "explained_rows_lie_in_the_test_window",
        not offenders,
        f"{checked - len(offenders)}/{checked} explained rows sit inside the test window "
        f"their fold defines, across {len(samples)} (model-shared) samples",
        offenders=offenders,
    )


def every_feature_maps_to_a_known_representation(values: pl.DataFrame) -> ValidationCheck:
    """No anonymous columns, and every name traces back to Component 4.

    Two tests, not one. Membership in ``KNOWN_FEATURE_NAMES`` is the real check; the
    anonymous-name pattern is a second one that would still fire if a generated name were
    ever added to the origin map, which is how the first check could quietly stop working.
    """
    if values.height == 0:
        return _check("every_feature_maps_to_a_known_representation", False, "no rows")
    names = values.select(["feature_name", "original_feature_name", "derived_from"]).unique()
    offenders: list[str] = []
    for row in names.to_dicts():
        name = str(row["feature_name"])
        if ANONYMOUS_NAME.match(name):
            offenders.append(f"{name}: an anonymous, machine-generated feature name")
        if name not in KNOWN_FEATURE_NAMES:
            offenders.append(f"{name}: not a known feature representation")
        if not row["original_feature_name"] or not row["derived_from"]:
            offenders.append(f"{name}: incomplete origin mapping")
    distinct = names["feature_name"].n_unique()
    return _check(
        "every_feature_maps_to_a_known_representation",
        not offenders,
        f"{distinct} distinct feature names, all present in Component 4's contract or its "
        "null-rule indicator set, none anonymous",
        offenders=offenders,
    )


def feature_names_match_the_declared_name_source(
    models: Sequence[RefitModel],
) -> ValidationCheck:
    """Each model's column names come from the function its registry entry names.

    The check that would have caught the single most likely defect in this component. The
    two candidate name lists are permutations of one another, disagreeing at 19 of 30
    positions, so a wrong choice mislabels most of the table while every sum still
    reconciles. Here the names are re-derived independently and compared position by
    position.
    """
    offenders: list[str] = []
    for model in models:
        expected = _expected_columns(model)
        if tuple(model.matrix_columns) != expected:
            wrong = sum(1 for a, b in zip(model.matrix_columns, expected, strict=False) if a != b)
            offenders.append(
                f"{model.spec.name}/{model.fold_id}: {wrong} column name(s) differ from "
                f"{model.spec.name_source}"
            )
    return _check(
        "feature_names_match_the_declared_name_source",
        not offenders,
        f"all {len(models)} re-executed fits name their columns with the function their "
        "registry entry declares",
        offenders=offenders,
    )


def _expected_columns(model: RefitModel) -> tuple[str, ...]:
    """Re-derive one model's column names from its own component, independently."""
    from sentinel.calibration.definitions import Family

    if model.spec.family is Family.LOGISTIC:
        from sentinel.modeling import preprocess as modeling_preprocess
        from sentinel.modeling.definitions import spec_for as modeling_spec_for

        return modeling_preprocess.ordered_matrix_columns(modeling_spec_for(model.spec.name))
    if model.spec.family is Family.BOOSTED:
        from sentinel.boosting import preprocess as boosting_preprocess
        from sentinel.boosting.definitions import spec_for as boosting_spec_for

        return boosting_preprocess.matrix_columns(boosting_spec_for(model.spec.name))
    from sentinel.neural import preprocess as neural_preprocess
    from sentinel.neural.definitions import spec_for as neural_spec_for

    fitted, _ = model.estimator
    return neural_preprocess.transformed_columns(neural_spec_for(model.spec.name), fitted.encoding)


def no_duplicate_explanation_keys(values: pl.DataFrame, cases: pl.DataFrame) -> ValidationCheck:
    """One attribution per (model, fold, inspection, feature); one case per prediction."""
    value_key = ["model_name", "fold_id", "target_inspection_id", "feature_name"]
    case_key = ["model_name", "fold_id", "target_inspection_id"]
    value_dupes = values.height - values.select(value_key).unique().height
    case_dupes = cases.height - cases.select(case_key).unique().height
    offenders: list[str] = []
    if value_dupes:
        offenders.append(f"explanation_values: {value_dupes} duplicate key(s)")
    if case_dupes:
        offenders.append(f"explanation_cases: {case_dupes} duplicate key(s)")
    return _check(
        "no_duplicate_explanation_keys",
        not offenders,
        f"{values.height} value rows and {cases.height} case rows are each uniquely keyed",
        offenders=offenders,
    )


def every_supported_model_covers_every_fold(
    values: pl.DataFrame, folds: Sequence[FoldSpec], expected_models: Sequence[str]
) -> ValidationCheck:
    """No supported model silently skipped a fold.

    A missing fold would not fail any other check -- the rows that exist would all be
    correct -- and it would quietly remove a quarter from every stability statistic.
    """
    offenders: list[str] = []
    expected_folds = {f.fold_id for f in folds}
    for model in expected_models:
        present = set(
            str(v)
            for v in values.filter(pl.col("model_name") == model)["fold_id"].unique().to_list()
        )
        for missing in sorted(expected_folds - present):
            offenders.append(f"{model}: no attributions for fold {missing}")
    return _check(
        "every_supported_model_covers_every_fold",
        not offenders,
        f"{len(expected_models)} supported model(s) x {len(expected_folds)} folds all present",
        offenders=offenders,
    )


# --- arithmetic --------------------------------------------------------------


def additivity_reconstructs_the_model_output(
    attributions: Sequence[FoldAttribution],
) -> ValidationCheck:
    """``base + sum(phi)`` equals the model's output, per method tolerance.

    **Necessary, and weaker than it looks for one of the three methods.** The permutation
    game's path telescopes, so its additivity holds at one round and at sixty-four alike --
    passing here says its arithmetic is sound, not that its credit split is accurate. The
    exactness of a method is recorded separately, on every row, as ``is_exact``.
    """
    if not attributions:
        return _check("additivity_reconstructs_the_model_output", False, "no attributions")
    offenders: list[str] = []
    worst = 0.0
    rows = 0
    for attribution in attributions:
        tolerance = tolerance_for(attribution.method)
        residual = attribution.residual
        rows += len(residual)
        worst = max(worst, float(residual.max()) if len(residual) else 0.0)
        bad = np.flatnonzero(residual > tolerance)
        for index in bad[:MAX_OFFENDERS]:
            offenders.append(
                f"{attribution.model_name}/{attribution.fold_id}/"
                f"{attribution.row_ids[index]}: residual {residual[index]:.3e} > "
                f"{tolerance:.0e}"
            )
    return _check(
        "additivity_reconstructs_the_model_output",
        not offenders,
        f"{rows - len(offenders)}/{rows} decompositions reconstruct their model output "
        f"within the method's frozen tolerance; worst residual {worst:.3e}",
        offenders=offenders,
    )


def shap_values_are_finite(values: pl.DataFrame) -> ValidationCheck:
    """No NaN and no infinity anywhere in the attributions or their base values."""
    if values.height == 0:
        return _check("shap_values_are_finite", False, "no rows")
    bad = values.filter(
        pl.col("shap_value").is_null()
        | pl.col("shap_value").is_nan()
        | pl.col("shap_value").is_infinite()
        | pl.col("base_value").is_null()
        | pl.col("base_value").is_nan()
        | pl.col("prediction_value").is_null()
        | pl.col("prediction_value").is_nan()
    )
    offenders = [
        f"{r['model_name']}/{r['fold_id']}/{r['target_inspection_id']}/{r['feature_name']}"
        for r in bad.head(MAX_OFFENDERS).to_dicts()
    ]
    return _check(
        "shap_values_are_finite",
        bad.height == 0,
        f"{values.height - bad.height}/{values.height} attribution rows carry finite values",
        offenders=offenders,
    )


def output_space_labels_are_declared(values: pl.DataFrame) -> ValidationCheck:
    """Every row's output space is a declared member of the enum.

    An attribution whose space is unstated, or stated as something the project does not
    define, is a number without units -- and mixing two spaces in one table without a field
    to separate them is the specific failure this column exists to prevent.
    """
    if values.height == 0:
        return _check("output_space_labels_are_declared", False, "no rows")
    declared = {space.value for space in OutputSpace}
    seen = set(str(v) for v in values["output_space"].unique().to_list())
    offenders = [f"{value}: not a declared OutputSpace" for value in sorted(seen - declared)]
    return _check(
        "output_space_labels_are_declared",
        not offenders,
        f"every row declares one of {sorted(declared)}; observed {sorted(seen)}",
        offenders=offenders,
    )


# --- temporal safety ---------------------------------------------------------


def background_rows_precede_the_training_horizon(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    backgrounds: Mapping[str, pl.DataFrame],
    *,
    date_column: str = "rd",
) -> ValidationCheck:
    """No reference row post-dates the horizon of the model it is a reference for.

    The background is part of the explanation. A reference set drawn from the test window
    would still produce additive, finite, plausible values -- answering "how does this row
    differ from the period the model is being judged on", which is not a question the model
    was ever asked and not one a regulator should be shown.

    Dates are re-derived from the rows themselves, never read from a recorded field: a field
    can be written by the same bug that needs catching.
    """
    by_id = {f.fold_id: f for f in folds}
    offenders: list[str] = []
    checked = 0
    for fold_id, background in sorted(backgrounds.items()):
        fold = by_id.get(fold_id)
        if fold is None:
            offenders.append(f"{fold_id}: background for a fold that does not exist")
            continue
        checked += background.height
        safe, late = background_is_safe(background, fold, date_column=date_column)
        if not safe:
            offenders.extend(late)
    return _check(
        "background_rows_precede_the_training_horizon",
        not offenders,
        f"{checked} background rows across {len(backgrounds)} fold(s) are all dated on or "
        "before their fold's train_end",
        offenders=offenders,
    )


def background_is_drawn_from_the_training_window(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    backgrounds: Mapping[str, pl.DataFrame],
) -> ValidationCheck:
    """Every background row is a training row of its own fold.

    Stronger than the date check and deliberately kept beside it. A row can be dated before
    ``train_end`` and still not be in the training window -- the split is defined by
    ``assign_split``, not by a date comparison a caller writes -- so this re-derives the
    training frame and asserts set containment.
    """
    from sentinel.modeling.train import training_frame

    by_id = {f.fold_id: f for f in folds}
    offenders: list[str] = []
    checked = 0
    for fold_id, background in sorted(backgrounds.items()):
        fold = by_id.get(fold_id)
        if fold is None or background.height == 0:
            continue
        allowed = {str(v) for v in training_frame(frame, fold)["target_inspection_id"].to_list()}
        stray = sorted(background_ids(background) - allowed)
        checked += background.height
        for row_id in stray[:MAX_OFFENDERS]:
            offenders.append(f"{fold_id}/{row_id}: not in the fold's training window")
    return _check(
        "background_is_drawn_from_the_training_window",
        not offenders,
        f"{checked} background rows are all members of their fold's training window as "
        "modeling.train.training_frame defines it",
        offenders=offenders,
    )


# --- honesty about what was not done -----------------------------------------


def unsupported_models_carry_no_attributions(
    values: pl.DataFrame, cases: pl.DataFrame, support: pl.DataFrame
) -> ValidationCheck:
    """An unsupported model has no values, no cases, and a stated reason.

    The check that makes "honest unsupported behaviour" checkable. A placeholder row of
    zeros would be worse than no row at all: zero is a legitimate attribution meaning "this
    feature did not move the score", and a table of them would read as a model that used no
    features rather than as a model nobody could explain.
    """
    unsupported = [
        s.name
        for s in (spec_for(n) for n in support["model_name"].to_list())
        if s.status is ExplanationStatus.UNSUPPORTED
    ]
    offenders: list[str] = []
    for name in unsupported:
        in_values = values.filter(pl.col("model_name") == name).height
        in_cases = cases.filter(pl.col("model_name") == name).height
        if in_values:
            offenders.append(f"{name}: {in_values} attribution row(s) for an unsupported model")
        if in_cases:
            offenders.append(f"{name}: {in_cases} case row(s) for an unsupported model")
        row = support.filter(pl.col("model_name") == name).to_dicts()
        if not row or not str(row[0].get("unsupported_reason") or "").strip():
            offenders.append(f"{name}: unsupported with no reason recorded")
        elif row[0].get("explanation_method") is not None:
            offenders.append(f"{name}: unsupported but records an explanation method")
    return _check(
        "unsupported_models_carry_no_attributions",
        not offenders,
        f"{len(unsupported)} unsupported model(s) carry no values, no cases and a stated reason",
        offenders=offenders,
    )


def covid_shift_is_never_pooled_into_a_quarterly_aggregate(
    importance: pl.DataFrame, stability: pl.DataFrame, drift: pl.DataFrame
) -> ValidationCheck:
    """No aggregate row mixes fold sets.

    Structural rather than conventional. Component 5 measured the model ordering *inverting*
    on ``covid_shift``, and a regime-shift fold averaged into seventeen ordinary ones moves
    the headline and leaves no trace of having done so.
    """
    offenders: list[str] = []
    aggregates = importance.filter(pl.col("scope") == "fold_set")
    for row in aggregates.to_dicts():
        if row["fold_id"] is not None:
            offenders.append(
                f"{row['model_name']}/{row['fold_set']}/{row['feature_name']}: an aggregate "
                f"row carries fold_id {row['fold_id']!r}"
            )
    covid = aggregates.filter(pl.col("fold_set") == "covid_shift")
    quarterly = aggregates.filter(pl.col("fold_set") == "quarterly")
    for name, table in (("stability", stability), ("drift", drift)):
        if table.height and table["fold_set"].n_unique() > 1:
            grouped = table.group_by("fold_set").len().to_dicts()
            detail = ", ".join(f"{g['fold_set']}={g['len']}" for g in grouped)
            logger.debug("%s table spans fold sets: %s", name, detail)
    return _check(
        "covid_shift_is_never_pooled_into_a_quarterly_aggregate",
        not offenders,
        f"{quarterly.height} quarterly and {covid.height} covid_shift aggregate rows are "
        "computed within their own fold set; no aggregate row spans both",
        offenders=offenders,
    )


def representative_cases_are_ordered_by_predicted_risk(
    cases: pl.DataFrame,
) -> ValidationCheck:
    """The high tier outscores the medium, which outscores the low.

    The executable form of "cases were selected on the prediction, not the outcome". If a
    tier were ever picked by whether the model was right, this ordering would break -- and
    it holds by construction only as long as the selection really is a score quantile.
    """
    if cases.height == 0:
        return _check(
            "representative_cases_are_ordered_by_predicted_risk",
            False,
            "no representative cases were selected",
        )
    offenders: list[str] = []
    checked = 0
    for (model, fold_id), group in cases.group_by(["model_name", "fold_id"]):
        scores = {str(r["tier"]): float(r["base_score"]) for r in group.to_dicts()}
        if len(scores) < 3:
            continue
        checked += 1
        if not scores["low"] <= scores["medium"] <= scores["high"]:
            offenders.append(
                f"{model}/{fold_id}: low={scores['low']:.4f} medium={scores['medium']:.4f} "
                f"high={scores['high']:.4f} are not in predicted-risk order"
            )
    return _check(
        "representative_cases_are_ordered_by_predicted_risk",
        not offenders,
        f"{checked - len(offenders)}/{checked} representative triples are ordered "
        "low <= medium <= high by the model's own committed score",
        offenders=offenders,
    )


def every_feature_was_used_somewhere(values: pl.DataFrame) -> ValidationCheck:
    """Advisory: which features no model ever moved a score with.

    Not an error. A feature with zero attribution everywhere is a real and useful finding --
    the model declined to use it -- and it is worth surfacing because Component 4 spent
    effort building it and a later component may want to know.
    """
    if values.height == 0:
        return _check("every_feature_was_used_somewhere", False, "no rows", severity=SEVERITY_WARN)
    used = (
        values.group_by("feature_name")
        .agg(pl.col("shap_value").abs().max().alias("peak"))
        .filter(pl.col("peak") > 0.0)["feature_name"]
        .to_list()
    )
    unused = sorted(KNOWN_FEATURE_NAMES - set(str(v) for v in used))
    return _check(
        "every_feature_was_used_somewhere",
        not unused,
        f"{len(used)}/{len(KNOWN_FEATURE_NAMES)} feature representations received a "
        "non-zero attribution from at least one model on at least one fold",
        severity=SEVERITY_WARN,
        offenders=[f"{name}: never moved any model's score" for name in unused],
    )


# --- entry point -------------------------------------------------------------


def validate_explanations(
    frame: pl.DataFrame,
    folds: Sequence[FoldSpec],
    models: Sequence[RefitModel],
    attributions: Sequence[FoldAttribution],
    samples: Sequence[ExplanationSample],
    reproductions: Sequence[ReproductionOutcome],
    backgrounds: Mapping[str, pl.DataFrame],
    tables: Mapping[str, pl.DataFrame],
    committed: pl.DataFrame,
    sha_before: Mapping[str, str],
    sha_after: Mapping[str, str],
    *,
    expected_models: Sequence[str] = SUPPORTED_MODELS,
    date_column: str = "rd",
) -> list[ValidationCheck]:
    """Every check an explanation run must pass, in one list."""
    values = tables["explanation_values"]
    cases = tables["explanation_cases"]
    return [
        regenerated_scores_reproduce_the_committed_artifact(reproductions),
        committed_prediction_artifacts_are_unchanged(sha_before, sha_after),
        every_explanation_maps_to_a_committed_prediction(values, committed),
        prediction_values_match_the_committed_scores(cases, committed),
        model_identity_is_preserved(values),
        explanations_belong_to_their_prediction_fold(values, folds),
        explained_rows_lie_in_the_test_window(frame, samples, folds, date_column=date_column),
        every_feature_maps_to_a_known_representation(values),
        feature_names_match_the_declared_name_source(models),
        no_duplicate_explanation_keys(values, cases),
        every_supported_model_covers_every_fold(values, folds, expected_models),
        additivity_reconstructs_the_model_output(attributions),
        shap_values_are_finite(values),
        output_space_labels_are_declared(values),
        background_rows_precede_the_training_horizon(
            frame, folds, backgrounds, date_column=date_column
        ),
        background_is_drawn_from_the_training_window(frame, folds, backgrounds),
        unsupported_models_carry_no_attributions(values, cases, tables["explanation_support"]),
        covid_shift_is_never_pooled_into_a_quarterly_aggregate(
            tables["explanation_importance"],
            tables["explanation_stability"],
            tables["explanation_drift"],
        ),
        representative_cases_are_ordered_by_predicted_risk(
            tables["explanation_representative_cases"]
        ),
        every_feature_was_used_somewhere(values),
    ]


# --- reporting ---------------------------------------------------------------


def format_report(checks: Sequence[ValidationCheck]) -> str:
    """A human-readable report, errors first."""
    lines: list[str] = []
    ordered = sorted(checks, key=lambda c: (c.passed, c.severity != SEVERITY_ERROR, c.name))
    for check in ordered:
        mark = "PASS" if check.passed else ("FAIL" if check.severity == SEVERITY_ERROR else "WARN")
        lines.append(f"[{mark}] {check.name}: {check.detail}")
        for offender in check.offenders:
            lines.append(f"         - {offender}")
    return "\n".join(lines)


def has_failures(checks: Sequence[ValidationCheck]) -> bool:
    """True when any error-severity check failed. Warnings never fail a run."""
    return any(not c.passed and c.severity == SEVERITY_ERROR for c in checks)


__all__ = [
    "ANONYMOUS_NAME",
    "SEVERITY_ERROR",
    "SEVERITY_WARN",
    "additivity_reconstructs_the_model_output",
    "background_is_drawn_from_the_training_window",
    "background_rows_precede_the_training_horizon",
    "committed_prediction_artifacts_are_unchanged",
    "covid_shift_is_never_pooled_into_a_quarterly_aggregate",
    "every_explanation_maps_to_a_committed_prediction",
    "every_feature_maps_to_a_known_representation",
    "every_feature_was_used_somewhere",
    "every_supported_model_covers_every_fold",
    "explained_rows_lie_in_the_test_window",
    "explanations_belong_to_their_prediction_fold",
    "feature_names_match_the_declared_name_source",
    "format_report",
    "has_failures",
    "model_identity_is_preserved",
    "no_duplicate_explanation_keys",
    "output_space_labels_are_declared",
    "prediction_values_match_the_committed_scores",
    "regenerated_scores_reproduce_the_committed_artifact",
    "representative_cases_are_ordered_by_predicted_risk",
    "shap_values_are_finite",
    "unsupported_models_carry_no_attributions",
    "validate_explanations",
]
