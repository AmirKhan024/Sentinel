"""Component 12 orchestration. The only module here that touches the filesystem or the clock.

Reads five artifacts, writes ten tables and a manifest, and changes nothing. This is the
first component in the project that re-executes no fit at all: Component 9 had to regenerate
scores that were never recorded and Component 11 had to regenerate the models themselves,
both behind ADR 0026's bit-identity gate. Every input this component needs already exists on
disk, so the integrity claim it makes is the opposite one -- **nothing moved** -- and it is
checked by re-reading every input's sha256 after the last table is written.

The order of work is the component's argument:

```text
group frame        who is who, and can we prove the label predates the row?
      |
support            is there enough data to compare them at all?
      |
metrics            ranking, probability and threshold behaviour, per group
      |
priority           who reaches the top k, and what did it capture for them?
      |
missingness        is the data itself distributed evenly?
      |
attribution        does the model rely on different features for different groups?
      |
disparity + drift  how far apart, and is that moving?
```

Support comes before metrics rather than after, and that is not an implementation detail. The
question "how do the groups compare" is only answerable once "is there enough data to compare
them" has been answered, and doing it the other way round is how an audit ends up quoting a
dramatic ratio from a group of twelve rows.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from sentinel import __version__
from sentinel.calibration import metrics as calibration_metrics
from sentinel.config import Settings
from sentinel.evaluation import folds as folds_module
from sentinel.evaluation import metrics as canonical_metrics
from sentinel.evaluation.models import FoldSpec
from sentinel.fairness import (
    attribution,
    disparity,
    drift,
    groups,
    missingness,
    priority,
    validate,
    writer,
)
from sentinel.fairness import (
    metrics as group_metrics,
)
from sentinel.fairness import (
    support as support_module,
)
from sentinel.fairness.definitions import (
    ADVISORY_REPRESENTATION_TRAVEL,
    ATTRIBUTION_MIN_ROWS,
    BLOCKED_EXPERIMENTS,
    BOOTSTRAP_METRICS,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SCHEMES,
    BOOTSTRAP_SEED,
    CALIBRATION_MIN_ROWS,
    CI_LEVEL,
    DISPARITY_REFERENCE,
    DOES_NOT_ESTABLISH,
    FAIRNESS_DEFINITION_VERSION,
    GROUP_CALIBRATION_BINS,
    GROUP_DEFINITION_REGISTRY,
    K_LEVELS,
    SUPPORT_MIN_NEGATIVE,
    SUPPORT_MIN_POSITIVE,
    SUPPORT_MIN_ROWS,
    THRESHOLD_POLICY,
    UNKNOWN_GROUP,
    Grain,
    GroupDefinitionSpec,
    GroupDefinitionStatus,
    GroupStatus,
    Stage,
)
from sentinel.fairness.models import (
    ArtifactRecord,
    FairnessManifest,
    FairnessStats,
    GroupSupport,
    ValidationCheck,
)
from sentinel.manifest import compute_sha256, manifest_path_for, write_manifest

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Where figures land unless a caller says otherwise, matching every other component.
FIGURES_DIR = Path("docs/analysis/figures")

#: The limitation this component inherits rather than discovers, carried in every manifest.
#: ADR 0019 named it in advance and said it should be stated here rather than found here.
INHERITED_LIMITATIONS: tuple[str, ...] = (
    "ADR 0019 -- the dataset publishes 22 columns and none identifies an inspector, so a "
    "group difference in citation rate cannot be decomposed into establishment risk versus "
    "differential inspection practice. Chicago assigns inspectors by district, which makes "
    "geography close to the strongest available proxy for who inspected.",
    "ADR 0008 -- the target is that a Priority violation was CITED, not that an "
    "establishment was unsafe.",
    "ADR 0012 -- this is a re-ordering study over inspections that actually happened. No "
    "establishment nobody inspected has a label, so nothing here speaks to coverage.",
    "ADR 0023 -- community area and ZIP correlate strongly with race and income by "
    "construction, and a correlate is not the attribute. No protected characteristic is "
    "observed anywhere in this project.",
)


class FairnessBuildError(RuntimeError):
    """A fairness audit could not be produced honestly from the inputs it was given."""


@dataclass(slots=True)
class FairnessResult:
    """Everything one run produced, whether or not it was written."""

    #: The definitions this run actually audited, which is not necessarily every audited
    #: definition in the registry -- a caller may restrict it. The summary reports these
    #: rather than the registry, because naming what was not measured is how a partial run
    #: gets read as a complete one.
    definitions: list[str]
    tables: dict[str, pl.DataFrame]
    checks: list[ValidationCheck]
    manifest: FairnessManifest
    stats: FairnessStats
    advisories: list[str]
    metrics_path: Path | None = None
    manifest_path: Path | None = None
    written: list[Path] = field(default_factory=list)
    figure_paths: list[Path] = field(default_factory=list)
    dry_run: bool = False


# --- input loading -----------------------------------------------------------


def _load_features(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if "inspection_date" not in frame.columns or "target" not in frame.columns:
        raise FairnessBuildError(
            f"{path.name} is not a Component 4 feature table: it must carry "
            "'inspection_date' and 'target'"
        )
    return frame.with_columns(pl.col("inspection_date").str.to_date().alias("rd"))


def _build_folds(frame: pl.DataFrame) -> list[FoldSpec]:
    """Folds re-derived from the data, never read back from disk.

    Every component does this, and the reason is the same each time: a fold table on disk is a
    record of what a previous run computed, and reading it would make this run agree with that
    record rather than with the data.
    """
    start = folds_module.min_date(frame, "rd")
    end = folds_module.max_date(frame, "rd")
    if start is None or end is None:
        raise FairnessBuildError("feature table has no usable reference dates")
    quarterly = folds_module.quarterly_folds(data_start=start, data_end=end)
    return [*quarterly, *folds_module.covid_shift_fold(data_end=end)]


def _fold_assignments(frame: pl.DataFrame, folds: Sequence[FoldSpec]) -> dict[str, str]:
    """Which fold's *test* window each row falls in, re-derived from ``assign_split``.

    Used by ``validate.every_row_is_in_its_declared_fold`` so that the fold label on a
    prediction row is checked against the data rather than trusted. Quarterly test windows do
    not overlap; ``covid_shift`` overlaps nothing in the quarterly set because its test window
    ends before the first quarterly one opens, so one row maps to at most one fold per set.
    """
    out: dict[str, str] = {}
    for fold in folds:
        labelled = folds_module.assign_split(frame, fold)
        test = labelled.filter(pl.col("split") == "test")
        for key in test.get_column("target_inspection_id").to_list():
            out.setdefault(str(key), fold.fold_id)
    return out


# --- per-grain measurement ---------------------------------------------------


def _metric_row(
    *,
    model_name: str,
    stage: Stage,
    spec: GroupDefinitionSpec,
    entry: GroupSupport,
    grain: Grain,
    fold_set: str,
    fold_id: str,
    metric: str,
    value: float | None,
    k_name: str = "",
    k: int = 0,
) -> dict[str, object]:
    """One metrics row, with the support decision applied to the value rather than the row.

    The value is nulled when the group missed its floor; the counts are always real and the
    row is always emitted. That is the shape the whole component depends on -- an unsupported
    group is a visible null with a stated reason, never an absent row.
    """
    kind = group_metrics.kind_of(metric)
    status = support_module.status_for(entry, kind)
    supported = status is GroupStatus.SUPPORTED
    return {
        "model_name": model_name,
        "stage": stage.value,
        "group_definition": spec.name,
        "group_value": entry.group_value,
        "grain": grain.value,
        "fold_set": fold_set,
        "fold_id": fold_id,
        "metric": metric,
        "metric_kind": kind.value,
        "k_name": k_name,
        "k": k,
        "value": value if supported else None,
        "n_rows": entry.n_rows,
        "n_positive": entry.n_positive,
        "n_negative": entry.n_negative,
        "group_status": status.value,
        "insufficient_reason": "" if supported else entry.insufficient_reason,
        "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
    }


def _group_values(
    frame: pl.DataFrame, spec: GroupDefinitionSpec, value: str
) -> tuple[list[int], list[float], list[float], list[str], list[date]]:
    """One group's labels, both probabilities, ids and dates, in frame order."""
    subset = frame.filter(pl.col(spec.source_column) == value)
    return (
        [int(v) for v in subset.get_column("target").to_list()],
        [float(v) for v in subset.get_column("base_probability").to_list()],
        [float(v) for v in subset.get_column("calibrated_probability").to_list()],
        [str(v) for v in subset.get_column("target_inspection_id").to_list()],
        [d for d in subset.get_column("rd").to_list() if isinstance(d, date)],
    )


def _measure_grain(
    frame: pl.DataFrame,
    spec: GroupDefinitionSpec,
    supports: Sequence[GroupSupport],
    *,
    model_name: str,
    grain: Grain,
    fold_set: str,
    fold_id: str,
    k_values: Mapping[str, int],
    emit_unsupported: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Metric and calibration-comparison rows for one (model, definition, grain) cell.

    ``emit_unsupported`` is the one place the artifact's size is traded against its
    completeness. At the pooled grain -- the reporting grain -- a row is emitted for every
    observed group including the unsupported ones, so a reader can count what was excluded
    from any comparison. At the per-fold grain only supported groups get metric rows, because
    the profiler measured 4 of 1,288 (fold, community area) cells clearing the floor and a
    complete null table there would be roughly eleven million rows of nothing.

    **The per-fold shortage stays visible either way**: ``fairness_group_support`` carries a
    row for every observed group at every grain, and ``validate.no_group_disappeared`` checks
    it against the data.
    """
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    for entry in supports:
        ranking_ok = entry.ranking_status is GroupStatus.SUPPORTED
        calibration_ok = entry.calibration_status is GroupStatus.SUPPORTED
        if not (ranking_ok or calibration_ok) and not emit_unsupported:
            continue

        labels, base, calibrated, ids, dates = _group_values(frame, spec, entry.group_value)
        per_stage: dict[Stage, dict[str, float | None]] = {}

        for stage, scores in ((Stage.BASE, base), (Stage.CALIBRATED, calibrated)):
            values: dict[str, float | None] = {}
            if ranking_ok:
                values.update(group_metrics.ranking_metrics(labels, scores, dates, ids))
            if calibration_ok:
                values.update(group_metrics.probability_metrics(labels, scores))
            per_stage[stage] = values

            for metric in group_metrics.RANKING_METRICS:
                metric_rows.append(
                    _metric_row(
                        model_name=model_name,
                        stage=stage,
                        spec=spec,
                        entry=entry,
                        grain=grain,
                        fold_set=fold_set,
                        fold_id=fold_id,
                        metric=metric,
                        value=values.get(metric),
                    )
                )
            for metric in group_metrics.PROBABILITY_METRICS:
                metric_rows.append(
                    _metric_row(
                        model_name=model_name,
                        stage=stage,
                        spec=spec,
                        entry=entry,
                        grain=grain,
                        fold_set=fold_set,
                        fold_id=fold_id,
                        metric=metric,
                        value=values.get(metric),
                    )
                )
            for k_name, k in sorted(k_values.items()):
                threshold = (
                    group_metrics.threshold_metrics(labels, scores, ids, k=k)
                    if ranking_ok
                    else dict.fromkeys(group_metrics.THRESHOLD_METRICS)
                )
                for metric in group_metrics.THRESHOLD_METRICS:
                    metric_rows.append(
                        _metric_row(
                            model_name=model_name,
                            stage=stage,
                            spec=spec,
                            entry=entry,
                            grain=grain,
                            fold_set=fold_set,
                            fold_id=fold_id,
                            metric=metric,
                            value=threshold.get(metric),
                            k_name=k_name,
                            k=k,
                        )
                    )

        for metric in group_metrics.PROBABILITY_METRICS:
            base_value = per_stage[Stage.BASE].get(metric)
            calibrated_value = per_stage[Stage.CALIBRATED].get(metric)
            calibration_rows.append(
                {
                    "model_name": model_name,
                    "group_definition": spec.name,
                    "group_value": entry.group_value,
                    "grain": grain.value,
                    "fold_set": fold_set,
                    "fold_id": fold_id,
                    "metric": metric,
                    "base_value": base_value,
                    "calibrated_value": calibrated_value,
                    "delta": (
                        calibrated_value - base_value
                        if base_value is not None and calibrated_value is not None
                        else None
                    ),
                    "improved": group_metrics.improved(metric, base_value, calibrated_value),
                    "n_rows": entry.n_rows,
                    "n_positive": entry.n_positive,
                    "group_status": entry.calibration_status.value,
                    "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
                }
            )

    return metric_rows, calibration_rows


def _pooled_reference(frame: pl.DataFrame, *, score_column: str) -> dict[str, float | None]:
    """Every metric over the un-grouped rows: the reference each disparity is measured from.

    Computed from the rows rather than from the group values. A mean of group means would be
    a different quantity -- weighted by group count rather than by rows -- and wrong by more
    the more uneven the groups are, which is exactly the situation being measured.
    """
    labels = [int(v) for v in frame.get_column("target").to_list()]
    scores = [float(v) for v in frame.get_column(score_column).to_list()]
    ids = [str(v) for v in frame.get_column("target_inspection_id").to_list()]
    dates = [d for d in frame.get_column("rd").to_list() if isinstance(d, date)]
    out: dict[str, float | None] = {}
    out.update(group_metrics.ranking_metrics(labels, scores, dates, ids))
    out.update(group_metrics.probability_metrics(labels, scores))
    return out


# --- orchestration -----------------------------------------------------------


def run_fairness_audit(
    settings: Settings,
    *,
    features_path: Path,
    calibrated_path: Path,
    categoricals_path: Path,
    explanations_path: Path | None = None,
    output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    group_definitions: Sequence[str] | None = None,
    figures_dir: Path | None = None,
    write_figures: bool = True,
    dry_run: bool = False,
) -> FairnessResult:
    """Audit every requested model's held-out behaviour across every audited group."""
    started = datetime.now(UTC)
    stamp = started.strftime(TIMESTAMP_FORMAT)

    read_paths: dict[str, Path] = {
        "features": features_path,
        "calibrated_predictions": calibrated_path,
        "categoricals": categoricals_path,
    }
    if explanations_path is not None:
        read_paths["explanations"] = explanations_path
    sha_before = {name: compute_sha256(path) for name, path in read_paths.items()}

    features = _load_features(features_path)
    predictions = pl.read_parquet(calibrated_path)
    categoricals = pl.read_parquet(categoricals_path)
    explanations = pl.read_parquet(explanations_path) if explanations_path else None

    specs = groups.resolve_definitions(group_definitions)
    available_models = sorted(predictions.get_column("model_name").unique().to_list())
    chosen_models = list(models) if models else available_models
    unknown = [m for m in chosen_models if m not in available_models]
    if unknown:
        raise FairnessBuildError(
            f"model(s) {', '.join(unknown)} are not in {calibrated_path.name}; available: "
            f"{', '.join(available_models)}"
        )
    predictions = predictions.filter(pl.col("model_name").is_in(chosen_models))

    folds = _build_folds(features)
    labels = features.select("target_inspection_id", "target", "rd")
    group_frame = groups.build_group_frame(predictions, categoricals, labels, specs)
    audited = groups.audited_frame(group_frame)
    source = groups.group_source(categoricals, specs)

    stats = FairnessStats(
        models=len(chosen_models),
        group_definitions=len(specs),
        folds=len(folds),
        audited_rows=audited.height,
    )

    rows: dict[str, list[dict[str, object]]] = {name: [] for name in writer.SCHEMAS}
    supports_by_key: dict[tuple[str, str, str], list[GroupSupport]] = {}

    fold_sets = sorted(audited.get_column("fold_set").unique().to_list())
    probe_model = chosen_models[0]

    # --- support and representation, model-independent ------------------------
    for fold_set in fold_sets:
        scoped = audited.filter(
            (pl.col("fold_set") == fold_set) & (pl.col("model_name") == probe_model)
        )
        for spec in specs:
            pooled = support_module.measure(
                scoped, spec, grain=Grain.FOLD_SET, fold_set=fold_set, fold_id=""
            )
            supports_by_key[(spec.name, fold_set, "")] = pooled
            rows["fairness_group_support"].extend(_support_dict(r) for r in pooled)

            for fold_id in sorted(scoped.get_column("fold_id").unique().to_list()):
                per_fold = support_module.measure(
                    scoped.filter(pl.col("fold_id") == fold_id),
                    spec,
                    grain=Grain.FOLD,
                    fold_set=fold_set,
                    fold_id=str(fold_id),
                )
                supports_by_key[(spec.name, fold_set, str(fold_id))] = per_fold
                rows["fairness_group_support"].extend(_support_dict(r) for r in per_fold)

    summary = support_module.summarise(
        [r for key, group in supports_by_key.items() if key[2] == "" for r in group]
    )
    stats.groups_observed = summary["observed"]
    stats.groups_supported = summary["supported_ranking"]
    stats.groups_insufficient = summary["insufficient"]

    # --- metrics, priority, missingness ---------------------------------------
    for fold_set in fold_sets:
        fold_ids = sorted(
            audited.filter(pl.col("fold_set") == fold_set).get_column("fold_id").unique().to_list()
        )
        capacity = _capacity_by_fold(features, folds, fold_set)

        for model_name in chosen_models:
            scoped = audited.filter(
                (pl.col("fold_set") == fold_set) & (pl.col("model_name") == model_name)
            )
            if scoped.is_empty():
                continue
            pooled_k = _pooled_k(scoped, capacity)

            for spec in specs:
                pooled_support = supports_by_key[(spec.name, fold_set, "")]
                metric_rows, calibration_rows = _measure_grain(
                    scoped,
                    spec,
                    pooled_support,
                    model_name=model_name,
                    grain=Grain.FOLD_SET,
                    fold_set=fold_set,
                    fold_id="",
                    k_values=pooled_k,
                    emit_unsupported=True,
                )
                rows["fairness_group_metrics"].extend(metric_rows)
                rows["fairness_group_calibration"].extend(calibration_rows)

                for fold_id in fold_ids:
                    per_fold_support = supports_by_key[(spec.name, fold_set, str(fold_id))]
                    fold_frame = scoped.filter(pl.col("fold_id") == fold_id)
                    fold_k = capacity.get(str(fold_id), {})
                    fold_metrics, fold_calibration = _measure_grain(
                        fold_frame,
                        spec,
                        per_fold_support,
                        model_name=model_name,
                        grain=Grain.FOLD,
                        fold_set=fold_set,
                        fold_id=str(fold_id),
                        k_values=fold_k,
                        emit_unsupported=False,
                    )
                    rows["fairness_group_metrics"].extend(fold_metrics)
                    rows["fairness_group_calibration"].extend(fold_calibration)

                rows["fairness_priority_audit"].extend(
                    _priority_rows(
                        scoped,
                        spec,
                        pooled_support,
                        model_name=model_name,
                        grain=Grain.FOLD_SET,
                        fold_set=fold_set,
                        fold_id="",
                        k_values=pooled_k,
                    )
                )
                for fold_id in fold_ids:
                    rows["fairness_priority_audit"].extend(
                        _priority_rows(
                            scoped.filter(pl.col("fold_id") == fold_id),
                            spec,
                            supports_by_key[(spec.name, fold_set, str(fold_id))],
                            model_name=model_name,
                            grain=Grain.FOLD,
                            fold_set=fold_set,
                            fold_id=str(fold_id),
                            k_values=capacity.get(str(fold_id), {}),
                        )
                    )

        # Missingness is model-independent: it describes the data, not a prediction. Measured
        # once per (fold set, definition) against the probe model's rows, which cover the
        # same ids as every other model's by the check above.
        probe = audited.filter(
            (pl.col("fold_set") == fold_set) & (pl.col("model_name") == probe_model)
        )
        rows["fairness_group_missingness"].extend(
            _missingness_rows(probe, features, specs, supports_by_key, fold_set, pooled_k=pooled_k)
        )

    # --- attribution profiles -------------------------------------------------
    if explanations is not None:
        lookup = source.select("target_inspection_id", *[spec.source_column for spec in specs])
        for fold_set in fold_sets:
            for spec in specs:
                for row in attribution.profiles(
                    explanations, lookup, spec, fold_set=fold_set, min_rows=ATTRIBUTION_MIN_ROWS
                ):
                    rows["fairness_attribution_profiles"].append(_attribution_dict(row))

    # --- disparity, bootstrap and drift ---------------------------------------
    metrics_frame = writer.finalize(rows["fairness_group_metrics"], "fairness_group_metrics")
    priority_frame = writer.finalize(rows["fairness_priority_audit"], "fairness_priority_audit")
    references = _reference_cache(audited)
    rows["fairness_disparity"] = _disparity_rows(metrics_frame, priority_frame, references)
    disparity_frame = writer.finalize(rows["fairness_disparity"], "fairness_disparity")
    rows["fairness_drift"] = drift.series(disparity_frame)
    rows["fairness_bootstrap"] = _bootstrap_rows(
        audited, metrics_frame, priority_frame, specs, chosen_models
    )
    rows["fairness_group_definitions"] = _definition_rows(audited, specs)

    tables = {name: writer.finalize(rows[name], name) for name in writer.SCHEMAS}

    stats.metric_rows = tables["fairness_group_metrics"].height
    stats.metric_rows_null = (
        tables["fairness_group_metrics"].filter(pl.col("value").is_null()).height
    )
    stats.priority_rows = tables["fairness_priority_audit"].height
    stats.calibration_rows = tables["fairness_group_calibration"].height
    stats.missingness_rows = tables["fairness_group_missingness"].height
    stats.attribution_rows = tables["fairness_attribution_profiles"].height
    stats.disparity_rows = tables["fairness_disparity"].height
    stats.drift_rows = tables["fairness_drift"].height
    stats.bootstrap_rows = tables["fairness_bootstrap"].height

    # --- write ----------------------------------------------------------------
    destination = output_dir or settings.fairness_processed_dir
    written: list[Path] = []
    metrics_path: Path | None = None
    if not dry_run:
        for name, frame in sorted(tables.items()):
            path = destination / f"{name}_{stamp}.parquet"
            writer.write_table(frame, path)
            written.append(path)
            if name == writer.DATASET_SLUG:
                metrics_path = path

    sha_after = {name: compute_sha256(path) for name, path in read_paths.items()}
    stats.inputs_unchanged = sha_after == sha_before

    checks = _validate(
        audited=audited,
        predictions=predictions,
        source=source,
        features=features,
        folds=folds,
        specs=specs,
        tables=tables,
        observed=group_frame.observed_values,
        sha_before=sha_before,
        sha_after=sha_after,
    )
    advisories = validate.advisory_findings(checks)
    # The drift advisories are computed rather than checked, because "this disparity widened"
    # is a measurement about a series rather than an assertion that can pass or fail. They
    # join the check-derived advisories in the manifest and the report; none of them can fail
    # a run. ADR 0034.
    quarterly_support = [
        record
        for key, group in supports_by_key.items()
        if key[1] == "quarterly"
        for record in group
    ]
    advisories.extend(
        drift.advisory_lines(
            rows["fairness_drift"],
            drift.representation_travel(quarterly_support),
            representation_threshold=ADVISORY_REPRESENTATION_TRAVEL,
        )
    )
    stats.advisories = len(advisories)
    stats.seconds = (datetime.now(UTC) - started).total_seconds()

    figure_paths: list[Path] = []
    if write_figures and not dry_run:
        from sentinel.fairness.figures import render

        figure_paths = render(tables, destination=figures_dir or FIGURES_DIR)

    manifest = _build_manifest(
        started=started,
        read_paths=read_paths,
        sha_before=sha_before,
        sha_after=sha_after,
        features=features,
        specs=specs,
        audited=audited,
        chosen_models=chosen_models,
        predictions=predictions,
        fold_sets=fold_sets,
        summary=summary,
        tables=tables,
        checks=checks,
        advisories=advisories,
        written=written,
        stats=stats,
    )

    manifest_path: Path | None = None
    if not dry_run and metrics_path is not None:
        manifest_path = manifest_path_for(metrics_path)
        write_manifest(manifest, manifest_path)

    return FairnessResult(
        definitions=[spec.name for spec in specs],
        tables=tables,
        checks=checks,
        manifest=manifest,
        stats=stats,
        advisories=advisories,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        written=written,
        figure_paths=figure_paths,
        dry_run=dry_run,
    )


# --- row builders ------------------------------------------------------------


def _support_dict(record: GroupSupport) -> dict[str, object]:
    return {
        "group_definition": record.group_definition,
        "group_value": record.group_value,
        "grain": record.grain,
        "fold_set": record.fold_set,
        "fold_id": record.fold_id,
        "n_rows": record.n_rows,
        "n_positive": record.n_positive,
        "n_negative": record.n_negative,
        "base_rate": record.base_rate,
        "representation_share": record.representation_share,
        "ranking_status": record.ranking_status.value,
        "calibration_status": record.calibration_status.value,
        "insufficient_reason": record.insufficient_reason,
        "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
    }


def _attribution_dict(row: object) -> dict[str, object]:
    record = row
    return {
        "model_name": record.model_name,  # type: ignore[attr-defined]
        "group_definition": record.group_definition,  # type: ignore[attr-defined]
        "group_value": record.group_value,  # type: ignore[attr-defined]
        "fold_set": record.fold_set,  # type: ignore[attr-defined]
        "feature_name": record.feature_name,  # type: ignore[attr-defined]
        "mean_abs_shap": record.mean_abs_shap,  # type: ignore[attr-defined]
        "mean_shap": record.mean_shap,  # type: ignore[attr-defined]
        "rank": record.rank,  # type: ignore[attr-defined]
        "overall_rank": record.overall_rank,  # type: ignore[attr-defined]
        "rank_delta": record.rank_delta,  # type: ignore[attr-defined]
        "n_rows": record.n_rows,  # type: ignore[attr-defined]
        "profile_spearman": record.profile_spearman,  # type: ignore[attr-defined]
        "is_exact": record.is_exact,  # type: ignore[attr-defined]
        "group_status": record.group_status.value,  # type: ignore[attr-defined]
        "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
    }


def _capacity_by_fold(
    features: pl.DataFrame, folds: Sequence[FoldSpec], fold_set: str
) -> dict[str, dict[str, int]]:
    """Each fold's top-k cutoffs, from Component 5's ``capacity_k_values``.

    Derived from the fold's *own* measured median daily capacity rather than a shared number,
    because capacity moved from 26 to 45 inspections a day across the study period and a
    fixed k would mean a different fraction of each window.
    """
    out: dict[str, dict[str, int]] = {}
    for fold in folds:
        if fold.fold_set != fold_set:
            continue
        stats = folds_module.fold_stats(features, fold)
        test = folds_module.window_frame(features, fold)
        if test.is_empty():
            continue
        median_daily = max(1, int(stats.test_median_daily_capacity or 1))
        out[fold.fold_id] = priority.k_values_for(
            test.select("target_inspection_id", "target", "rd"), fold, median_daily
        )
    return out


def _pooled_k(frame: pl.DataFrame, capacity: Mapping[str, dict[str, int]]) -> dict[str, int]:
    """Cutoffs for the pooled grain: the sum of the folds' cutoffs.

    Summed rather than recomputed over the pooled rows, because the pooled set is not a window
    the city ever worked -- it is 17 quarters laid end to end. Summing the per-fold capacities
    keeps "the top 5%" meaning the same operational thing it means inside a quarter.
    """
    totals: dict[str, int] = {}
    fold_ids = set(frame.get_column("fold_id").unique().to_list())
    for fold_id, values in capacity.items():
        if fold_id not in fold_ids:
            continue
        for name, k in values.items():
            totals[name] = totals.get(name, 0) + k
    return {name: min(k, frame.height) for name, k in totals.items() if k > 0}


def _priority_rows(
    frame: pl.DataFrame,
    spec: GroupDefinitionSpec,
    supports: Sequence[GroupSupport],
    *,
    model_name: str,
    grain: Grain,
    fold_set: str,
    fold_id: str,
    k_values: Mapping[str, int],
) -> list[dict[str, object]]:
    """Priority-audit rows for every stage and every cutoff."""
    index = {record.group_value: record for record in supports}
    out: list[dict[str, object]] = []
    for stage in (Stage.BASE, Stage.CALIBRATED):
        column = groups.stage_column(stage.value)
        for k_name, k in sorted(k_values.items()):
            for row in priority.audit(
                frame,
                spec,
                index,
                model_name=model_name,
                stage=stage,
                score_column=column,
                grain=grain,
                fold_set=fold_set,
                fold_id=fold_id,
                k_name=k_name,
                k=k,
            ):
                out.append(
                    {
                        "model_name": row.model_name,
                        "stage": row.stage.value,
                        "group_definition": row.group_definition,
                        "group_value": row.group_value,
                        "grain": row.grain,
                        "fold_set": row.fold_set,
                        "fold_id": row.fold_id,
                        "k_name": row.k_name,
                        "k": row.k,
                        "n_rows": row.n_rows,
                        "n_positive": row.n_positive,
                        "population_share": row.population_share,
                        "n_selected": row.n_selected,
                        "selected_share": row.selected_share,
                        "selection_rate": row.selection_rate,
                        "selection_rate_ratio": row.selection_rate_ratio,
                        "positives_selected": row.positives_selected,
                        "precision_in_selected": row.precision_in_selected,
                        "capture_rate": row.capture_rate,
                        "overall_capture_rate": row.overall_capture_rate,
                        "group_status": row.group_status.value,
                        "insufficient_reason": row.insufficient_reason,
                        "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
                    }
                )
    return out


def _missingness_rows(
    probe: pl.DataFrame,
    features: pl.DataFrame,
    specs: Sequence[GroupDefinitionSpec],
    supports_by_key: Mapping[tuple[str, str, str], list[GroupSupport]],
    fold_set: str,
    *,
    pooled_k: Mapping[str, int],
) -> list[dict[str, object]]:
    """Missingness rows at the pooled grain, with the top-5% set as the comparison."""
    if probe.is_empty():
        return []
    indicator_sources = [source for _, source in missingness.indicators(features.columns)]
    enriched = probe.join(
        features.select("target_inspection_id", *indicator_sources),
        on="target_inspection_id",
        how="left",
    )
    k_name = "k_pct_05" if "k_pct_05" in pooled_k else next(iter(sorted(pooled_k)), "")
    selected_ids: list[str] = []
    if k_name:
        selected_ids = (
            priority.select_top_k(enriched, "calibrated_probability", pooled_k[k_name])
            .get_column("target_inspection_id")
            .to_list()
        )

    out: list[dict[str, object]] = []
    for spec in specs:
        index = {r.group_value: r for r in supports_by_key[(spec.name, fold_set, "")]}
        for row in missingness.measure(
            enriched,
            spec,
            index,
            selected_ids,
            grain=Grain.FOLD_SET,
            fold_set=fold_set,
            fold_id="",
            k_name=k_name,
        ):
            out.append(
                {
                    "group_definition": row.group_definition,
                    "group_value": row.group_value,
                    "grain": row.grain,
                    "fold_set": row.fold_set,
                    "fold_id": row.fold_id,
                    "indicator": row.indicator,
                    "source_column": row.source_column,
                    "n_rows": row.n_rows,
                    "n_missing": row.n_missing,
                    "missing_rate": row.missing_rate,
                    "overall_missing_rate": row.overall_missing_rate,
                    "deviation": row.deviation,
                    "missing_rate_in_top_k": row.missing_rate_in_top_k,
                    "k_name": row.k_name,
                    "group_status": row.group_status.value,
                    "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
                }
            )
    return out


def _disparity_rows(
    metrics: pl.DataFrame,
    priority_frame: pl.DataFrame,
    references: Mapping[tuple[str, str, str, str, str], float | None],
) -> list[dict[str, object]]:
    """Disparity summaries over the metric table and the priority table's capture column.

    ``capture_rate`` is folded in from the priority table because it is a group metric in every
    respect except which table it lives in -- and section 10 of the brief asks for its
    disparity specifically. Both are reduced to ``disparity.COMPARABLE_COLUMNS`` so that one
    implementation of each measure serves both, rather than two that could disagree inside the
    same comparison.
    """
    frames: list[pl.DataFrame] = []
    if not metrics.is_empty():
        frames.append(metrics.select(disparity.COMPARABLE_COLUMNS))
    if not priority_frame.is_empty():
        frames.append(
            priority_frame.select(
                "model_name",
                "stage",
                "group_definition",
                "group_value",
                "grain",
                "fold_set",
                "fold_id",
                pl.lit("capture_rate").alias("metric"),
                "k_name",
                pl.col("capture_rate").alias("value"),
                "n_rows",
                "group_status",
            )
        )
    if not frames:
        return []
    return disparity.summarise(pl.concat(frames, how="vertical"), references)


def _reference_cache(
    audited: pl.DataFrame,
) -> dict[tuple[str, str, str, str, str], float | None]:
    """Pooled metric values per (model, stage, fold set, fold), for the disparity reference."""
    out: dict[tuple[str, str, str, str, str], float | None] = {}
    for model_name in sorted(audited.get_column("model_name").unique().to_list()):
        for fold_set in sorted(audited.get_column("fold_set").unique().to_list()):
            scoped = audited.filter(
                (pl.col("model_name") == model_name) & (pl.col("fold_set") == fold_set)
            )
            if scoped.is_empty():
                continue
            for stage in (Stage.BASE, Stage.CALIBRATED):
                column = groups.stage_column(stage.value)
                pooled = _pooled_reference(scoped, score_column=column)
                for metric, value in pooled.items():
                    out[(str(model_name), stage.value, str(fold_set), "", metric)] = value
    return out


def _bootstrap_rows(
    audited: pl.DataFrame,
    metrics: pl.DataFrame,
    priority_frame: pl.DataFrame,
    specs: Sequence[GroupDefinitionSpec],
    models: Sequence[str],
) -> list[dict[str, object]]:
    """Deterministic intervals for the two metrics where sampling variability changes a reading.

    Only the calibrated stage, only the pooled grain, only supported groups. Bootstrapping
    every cell would triple the runtime to decorate numbers nobody would read differently, and
    a small group stays flagged by its support regardless of any interval.

    The seed is derived from the model's position in the list plus a frozen base, never from
    ``hash()`` of a name: Python salts ``str`` hashing per process, which is what made
    Component 9's bootstrap non-reproducible until the key changed. MEMORY invariant 92.
    """
    if metrics.is_empty():
        return []
    out: list[dict[str, object]] = []
    for position, model_name in enumerate(sorted(models)):
        seed = BOOTSTRAP_SEED + position
        for fold_set in sorted(audited.get_column("fold_set").unique().to_list()):
            scoped = audited.filter(
                (pl.col("model_name") == model_name) & (pl.col("fold_set") == fold_set)
            )
            if scoped.is_empty():
                continue
            for spec in specs:
                eligible = metrics.filter(
                    (pl.col("model_name") == model_name)
                    & (pl.col("fold_set") == str(fold_set))
                    & (pl.col("group_definition") == spec.name)
                    & (pl.col("grain") == Grain.FOLD_SET.value)
                    & (pl.col("stage") == Stage.CALIBRATED.value)
                    & (pl.col("metric") == "ece")
                    & (pl.col("group_status") == GroupStatus.SUPPORTED.value)
                ).sort("group_value")
                for offset, record in enumerate(eligible.to_dicts()):
                    value = str(record["group_value"])
                    subset = scoped.filter(pl.col(spec.source_column) == value)
                    labels = [int(v) for v in subset.get_column("target").to_list()]
                    probabilities = [
                        float(v) for v in subset.get_column("calibrated_probability").to_list()
                    ]
                    blocks = [str(v) for v in subset.get_column(groups.ENTITY_COLUMN).to_list()]
                    for scheme_index, scheme in enumerate(BOOTSTRAP_SCHEMES):
                        interval = calibration_metrics.bootstrap(
                            labels,
                            probabilities,
                            metric=_group_ece,
                            metric_name="ece",
                            scheme=scheme,
                            # Structured integers, never hash() of a name: Python salts str
                            # hashing per process, which is what made Component 9's bootstrap
                            # non-reproducible until its key became a registry position.
                            seed_key=(seed, position, offset, scheme_index),
                            groups=blocks,
                            replications=BOOTSTRAP_REPLICATIONS,
                            level=CI_LEVEL,
                        )
                        out.append(
                            {
                                "model_name": model_name,
                                "stage": Stage.CALIBRATED.value,
                                "group_definition": spec.name,
                                "group_value": value,
                                "grain": Grain.FOLD_SET.value,
                                "fold_set": str(fold_set),
                                "metric": "ece",
                                "k_name": "",
                                "point_estimate": interval.point_estimate,
                                "lower": interval.lower,
                                "upper": interval.upper,
                                "replications": BOOTSTRAP_REPLICATIONS,
                                "level": CI_LEVEL,
                                "seed": seed,
                                "n_rows": subset.height,
                                "scheme": scheme,
                                "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
                            }
                        )
    return out


def _group_ece(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """The canonical ECE at the canonical bin count, as a named function.

    Named rather than a lambda so the bootstrap's metric is the same callable object on every
    replication and the same one a reader can go and look at -- and so the bin count is bound
    once here rather than captured from a loop variable.
    """
    return canonical_metrics.ece(labels, probabilities, n_bins=GROUP_CALIBRATION_BINS)


def _definition_rows(
    audited: pl.DataFrame, specs: Sequence[GroupDefinitionSpec]
) -> list[dict[str, object]]:
    """One row per candidate definition, audited and refused alike.

    The refusals travel in the artifact rather than only in ADR 0033, so a reader who opens
    the Parquet instead of the document still finds out why there is no ward breakdown -- and
    finds the measurement that decided it, not an assertion.
    """
    audited_names = {spec.name for spec in specs}
    out: list[dict[str, object]] = []
    for spec in GROUP_DEFINITION_REGISTRY:
        is_audited = spec.name in audited_names
        distinct = 0
        unknown = 0
        rows_counted = 0
        if is_audited and spec.source_column in audited.columns:
            column = audited.get_column(spec.source_column)
            distinct = int(column.n_unique())
            unknown = audited.filter(pl.col(spec.source_column) == UNKNOWN_GROUP).height
            rows_counted = audited.height
        out.append(
            {
                "group_definition": spec.name,
                "status": (
                    GroupDefinitionStatus.AUDITED.value
                    if is_audited
                    else GroupDefinitionStatus.REFUSED.value
                ),
                "source_column": spec.source_column,
                "provenance": spec.provenance,
                "rationale": spec.rationale,
                "is_model_feature": spec.is_model_feature,
                "refusal_reason": spec.refusal_reason
                or ("" if is_audited else "not requested in this run"),
                "distinct_values": distinct,
                "unknown_rows": unknown,
                "audited_rows": rows_counted,
                "fairness_definition_version": FAIRNESS_DEFINITION_VERSION,
            }
        )
    return out


# --- validation and manifest -------------------------------------------------


def _validate(
    *,
    audited: pl.DataFrame,
    predictions: pl.DataFrame,
    source: pl.DataFrame,
    features: pl.DataFrame,
    folds: Sequence[FoldSpec],
    specs: Sequence[GroupDefinitionSpec],
    tables: Mapping[str, pl.DataFrame],
    observed: Mapping[str, Sequence[str]],
    sha_before: Mapping[str, str],
    sha_after: Mapping[str, str],
) -> list[ValidationCheck]:
    columns = [spec.source_column for spec in specs]
    assignments = _fold_assignments(features, folds)
    return [
        validate.every_audited_row_has_a_prediction(audited, predictions),
        validate.every_group_value_comes_from_the_source(audited, source, columns),
        validate.group_mapping_predates_every_row(source),
        validate.group_mapping_is_unambiguous(source, columns),
        validate.every_row_is_in_its_declared_fold(audited, assignments),
        validate.stages_are_not_confused(audited, predictions),
        validate.no_group_disappeared(tables["fairness_group_support"], observed),
        validate.every_metric_carries_support(tables["fairness_group_metrics"]),
        validate.support_decisions_are_reproducible(tables["fairness_group_support"]),
        validate.no_outcome_or_feature_column_leaked(tables),
        validate.tables_are_deterministically_sorted(tables, writer.SORT_KEYS),
        validate.inputs_were_not_modified(sha_before, sha_after),
        validate.covid_was_not_pooled(tables),
        validate.group_calibration_spread_is_modest(tables["fairness_disparity"]),
        validate.selection_rates_are_proportionate(tables["fairness_priority_audit"]),
        validate.capture_is_even_across_groups(tables["fairness_disparity"]),
    ]


def _build_manifest(
    *,
    started: datetime,
    read_paths: Mapping[str, Path],
    sha_before: Mapping[str, str],
    sha_after: Mapping[str, str],
    features: pl.DataFrame,
    specs: Sequence[GroupDefinitionSpec],
    audited: pl.DataFrame,
    chosen_models: Sequence[str],
    predictions: pl.DataFrame,
    fold_sets: Sequence[str],
    summary: Mapping[str, int],
    tables: Mapping[str, pl.DataFrame],
    checks: Sequence[ValidationCheck],
    advisories: Sequence[str],
    written: Sequence[Path],
    stats: FairnessStats,
) -> FairnessManifest:
    experimental = sorted(
        predictions.filter(pl.col("is_experimental")).get_column("model_name").unique().to_list()
    )
    artifacts = [
        ArtifactRecord(
            path=str(path),
            bytes=path.stat().st_size,
            sha256=compute_sha256(path),
            row_count=tables[path.stem.rsplit("_", 1)[0]].height,
            schema=writer.schema_of(tables[path.stem.rsplit("_", 1)[0]]),
        )
        for path in written
    ]
    feature_version = ""
    if "feature_definition_version" in features.columns:
        feature_version = str(features.get_column("feature_definition_version").head(1).item())

    return FairnessManifest(
        code_version=__version__,
        fairness_definition_version=FAIRNESS_DEFINITION_VERSION,
        built_at=started.isoformat(),
        features_path=str(read_paths["features"]),
        features_sha256=sha_before["features"],
        feature_definition_version=feature_version,
        calibrated_predictions_path=str(read_paths["calibrated_predictions"]),
        calibrated_predictions_sha256=sha_before["calibrated_predictions"],
        categoricals_path=str(read_paths["categoricals"]),
        categoricals_sha256=sha_before["categoricals"],
        explanations_path=str(read_paths["explanations"]) if "explanations" in read_paths else None,
        explanations_sha256=sha_before.get("explanations"),
        inputs_unchanged=stats.inputs_unchanged,
        input_sha256_after=dict(sha_after),
        audited_group_definitions=[spec.name for spec in specs],
        refused_group_definitions={
            spec.name: spec.refusal_reason
            for spec in GROUP_DEFINITION_REGISTRY
            if spec.status is GroupDefinitionStatus.REFUSED
        },
        group_provenance={spec.name: spec.provenance for spec in specs},
        group_source_is_a_model_feature={spec.name: spec.is_model_feature for spec in specs},
        support_min_rows=SUPPORT_MIN_ROWS,
        support_min_positive=SUPPORT_MIN_POSITIVE,
        support_min_negative=SUPPORT_MIN_NEGATIVE,
        calibration_min_rows=CALIBRATION_MIN_ROWS,
        calibration_bins=GROUP_CALIBRATION_BINS,
        groups_observed=summary["observed"],
        groups_supported=summary["supported_ranking"],
        groups_insufficient=summary["insufficient"],
        models=list(chosen_models),
        experimental_models=experimental,
        stages=[stage.value for stage in Stage],
        fold_sets=list(fold_sets),
        k_levels=list(K_LEVELS),
        threshold_policy=THRESHOLD_POLICY,
        disparity_reference=DISPARITY_REFERENCE,
        bootstrap_replications=BOOTSTRAP_REPLICATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_metrics=list(BOOTSTRAP_METRICS),
        ci_level=CI_LEVEL,
        does_not_establish=list(DOES_NOT_ESTABLISH),
        blocked=list(BLOCKED_EXPERIMENTS),
        inherited_limitations=list(INHERITED_LIMITATIONS),
        checks=[
            {
                "name": c.name,
                "passed": c.passed,
                "severity": c.severity,
                "detail": c.detail,
            }
            for c in checks
        ],
        advisories=list(advisories),
        artifacts=artifacts,
        row_counts={name: frame.height for name, frame in sorted(tables.items())},
        seconds=stats.seconds,
    )


def summarize(result: FairnessResult) -> str:
    """A fixed-width block for the CLI, ending with what a green run does not mean."""
    stats = result.stats
    lines = [
        "",
        "Component 12 -- fairness and geographic equity audit",
        "",
        f"  models                {stats.models}",
        f"  group definitions     {stats.group_definitions} ({', '.join(result.definitions)})",
        f"  folds                 {stats.folds}",
        f"  audited rows          {stats.audited_rows:,}",
        "",
        f"  groups observed       {stats.groups_observed}",
        f"  groups supported      {stats.groups_supported} "
        f"(>= {SUPPORT_MIN_ROWS} rows, {SUPPORT_MIN_POSITIVE}/{SUPPORT_MIN_NEGATIVE} classes)",
        f"  insufficient support  {stats.groups_insufficient} "
        "(recorded with counts and a reason, never dropped)",
        "",
        f"  metric rows           {stats.metric_rows:,} "
        f"({stats.metric_rows_null:,} null for insufficient support)",
        f"  priority rows         {stats.priority_rows:,}",
        f"  calibration rows      {stats.calibration_rows:,}",
        f"  missingness rows      {stats.missingness_rows:,}",
        f"  attribution rows      {stats.attribution_rows:,}",
        f"  disparity rows        {stats.disparity_rows:,}",
        f"  drift rows            {stats.drift_rows:,}",
        f"  bootstrap rows        {stats.bootstrap_rows:,}",
        "",
        f"  inputs unchanged      {stats.inputs_unchanged}",
        f"  advisory findings     {stats.advisories}",
        f"  seconds               {stats.seconds:.1f}",
    ]
    if result.advisories:
        lines.append("")
        lines.append("  ADVISORY (measured disparities -- evidence, not implementation errors):")
        lines.extend(f"    - {note}" for note in result.advisories)
    lines.append("")
    lines.append("  DOES NOT ESTABLISH:")
    lines.extend(f"    - {claim}" for claim in DOES_NOT_ESTABLISH)
    if result.dry_run:
        lines.extend(["", "  DRY RUN -- nothing was written."])
    else:
        lines.append("")
        lines.extend(f"  wrote {path}" for path in result.written)
        if result.manifest_path:
            lines.append(f"  wrote {result.manifest_path}")
        lines.extend(f"  wrote {path}" for path in result.figure_paths)
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "FIGURES_DIR",
    "INHERITED_LIMITATIONS",
    "TIMESTAMP_FORMAT",
    "FairnessBuildError",
    "FairnessResult",
    "run_fairness_audit",
    "summarize",
]
