"""Component 8's experimental categorical join. The riskiest module in the component.

Component 4's feature table has 26 columns and **not one of them is categorical**. The
four families this component embeds -- chain, facility type, community area and zip --
are not features it can ask for; they have to be brought in from elsewhere. This module
is that elsewhere, and it is deliberately quarantined:

* Its output is a **separate artifact under a separate layer**
  (``data/processed/neural/``), never a column added to Component 4's table.
* ``feature_definition_version`` stays ``v1``. Nothing here is a Component 4 feature and
  nothing here may be treated as one. See ADR 0022.
* Every model that compares against Components 6 and 7 -- ``neural_numeric_only`` -- is
  fitted without any of it.

**The as-of rule applies here exactly as it does in Component 4.** A categorical is taken
from the establishment's most recent inspection *strictly before* the row's own
``inspection_date``, never from the row itself. The distinction matters more than it
looks. Facility type and address are genuinely known before an inspection happens, so
reading them off the target row would arguably be legitimate -- but they are *recorded on
the inspection record*, which is written at inspection time, and this project does not
build features from the row being predicted. Carrying the last observed value forward
needs no exception and is directly testable: ``source_inspection_date`` is emitted beside
every value and ``validate`` asserts it is strictly earlier, on every row.

The cost is stated rather than hidden: an establishment with no prior inspection of any
type has nothing to carry forward, and gets ``UNKNOWN``. That is a real category with a
learned embedding, not a null and not an imputation -- "we have never seen this place
before" is a fact about the establishment, and it is the same fact Component 4 encodes
as a null-rule family indicator.

``chain`` is not emitted here. This module emits ``chain_key`` -- the as-of normalised
name from Component 2 -- and ``encode`` turns that into a chain category using membership
derived from one fold's training rows only. Chain membership is a property of a *set* of
establishments, so computing it globally would let an establishment's second location,
opened years later, reach backwards and change what the model saw. See the ``encode``
docstring.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import polars as pl

from sentinel.neural.definitions import UNKNOWN_CATEGORY

logger = logging.getLogger(__name__)

#: The Socrata computed-region column carrying Chicago's 77 community areas. It is a
#: spatial join Socrata performs against a boundary layer, not city-supplied source data,
#: and it is absent whenever the row has no coordinates -- both facts are recorded in
#: ``docs/data_contracts/food_inspections_raw.md`` and neither is worked around here.
COMMUNITY_AREA_COLUMN = ":@computed_region_vrxf_vc4k"

#: Raw columns this module reads. Everything else in the 22-column snapshot is ignored;
#: naming them explicitly means a schema change surfaces as a missing-column error rather
#: than as a silently absent category.
RAW_COLUMNS: tuple[str, ...] = (
    "inspection_id",
    "inspection_date",
    "facility_type",
    "zip",
    COMMUNITY_AREA_COLUMN,
)

#: Columns the assignment table supplies. ``name_key`` is Component 2's normalised
#: ``dba_name``; reusing it rather than re-normalising here means "the same name" means
#: the same thing in Component 8 as it does in entity resolution.
ASSIGNMENT_COLUMNS: tuple[str, ...] = ("inspection_id", "establishment_id", "name_key")

#: The categorical columns this module emits, in declared order.
EMITTED_CATEGORICALS: tuple[str, ...] = (
    "chain_key",
    "facility_type",
    "community_area",
    "zip",
)


class CategoricalBuildError(ValueError):
    """Raised when the experimental categorical table cannot be built."""


def _normalize_text(column: str) -> pl.Expr:
    """Upper-case, trim and collapse internal whitespace; blank becomes UNKNOWN.

    ``facility_type`` is free text with observed blanks and case variation, so two rows
    naming the same kind of business must not become two vocabulary entries. The
    transform is deliberately conservative -- it does not merge synonyms, because
    deciding that "GROCERY STORE" and "GROCERY" are the same thing is a judgement this
    module has no basis for and would quietly bake into every result.
    """
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"\s+", " ")
        .replace("", None)
        .fill_null(UNKNOWN_CATEGORY)
    )


def _normalize_zip() -> pl.Expr:
    """Five-digit zip as a string. Anything else becomes UNKNOWN.

    The raw column is Socrata type ``number`` stored as text, so a leading zero survives
    but a ZIP+4 or a stray value does not become a silent third format.
    """
    return (
        pl.col("zip")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.extract(r"^(\d{5})", 1)
        .fill_null(UNKNOWN_CATEGORY)
    )


def _normalize_community_area() -> pl.Expr:
    """The computed region as a bare integer string, or UNKNOWN.

    Absent whenever the raw row had no usable coordinates, which is the documented
    behaviour of a Socrata computed region and not a defect to patch over.
    """
    return (
        pl.col(COMMUNITY_AREA_COLUMN)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.extract(r"^(\d+)", 1)
        .fill_null(UNKNOWN_CATEGORY)
    )


def load_history(raw_path: Path, assignments_path: Path) -> pl.DataFrame:
    """The per-establishment attribute history, one row per raw inspection.

    Joined on ``inspection_id`` to Component 2's assignments, because an establishment is
    Component 2's concept and re-deriving it here would put a second identity definition
    in the repository.
    """
    if not raw_path.exists():
        raise CategoricalBuildError(f"raw food inspections not found: {raw_path}")
    if not assignments_path.exists():
        raise CategoricalBuildError(f"establishment assignments not found: {assignments_path}")

    raw = pl.read_parquet(raw_path)
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise CategoricalBuildError(
            f"raw snapshot is missing column(s) {', '.join(missing)}. Component 8's "
            "categoricals cannot be built from a snapshot that lacks them."
        )

    assignments = pl.read_parquet(assignments_path)
    missing = [c for c in ASSIGNMENT_COLUMNS if c not in assignments.columns]
    if missing:
        raise CategoricalBuildError(
            f"establishment assignments are missing column(s) {', '.join(missing)}"
        )

    history = (
        raw.select(RAW_COLUMNS)
        .with_columns(
            pl.col("inspection_id").cast(pl.Utf8),
            # The raw column is ISO-8601 with a zero time component. Truncating to 10
            # characters rather than parsing a timestamp keeps this identical to how
            # Components 3 and 4 read the same field.
            pl.col("inspection_date").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("rd"),
            _normalize_text("facility_type").alias("facility_type"),
            _normalize_zip().alias("zip"),
            _normalize_community_area().alias("community_area"),
        )
        .join(
            assignments.select(ASSIGNMENT_COLUMNS).with_columns(
                pl.col("inspection_id").cast(pl.Utf8)
            ),
            on="inspection_id",
            how="inner",
        )
        .with_columns(
            pl.col("name_key").cast(pl.Utf8).fill_null(UNKNOWN_CATEGORY).alias("chain_key")
        )
        .drop_nulls("rd")
        .select(
            "inspection_id",
            "establishment_id",
            "rd",
            *EMITTED_CATEGORICALS,
        )
    )
    if history.height == 0:
        raise CategoricalBuildError("attribute history is empty after joining assignments")
    logger.info(
        "Attribute history: %d rows over %d establishments",
        history.height,
        history["establishment_id"].n_unique(),
    )
    return history


def build_categoricals(features: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Carry each categorical forward to every prediction row, strictly as-of.

    One backward as-of join per establishment, with exact date matches **excluded**. That
    exclusion is the whole temporal argument: an exact match would be the target
    inspection itself, and reading a value off the row being predicted is precisely what
    Component 4's as-of rule forbids.
    """
    required = ("target_inspection_id", "establishment_id", "inspection_date")
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise CategoricalBuildError(f"feature table is missing column(s) {', '.join(missing)}")

    left = (
        features.select(required)
        .with_columns(pl.col("inspection_date").str.to_date().alias("rd"))
        .sort(["rd", "target_inspection_id"])
    )

    right = history.rename({"inspection_id": "source_inspection_id", "rd": "source_rd"}).sort(
        ["source_rd", "source_inspection_id"]
    )

    # polars cannot verify sortedness when ``by`` groups are supplied and warns about it
    # on every call. Both frames ARE sorted by their as-of key -- the sorts are two lines
    # above -- so the warning is about what polars can check, not about what is true.
    # Silenced narrowly, around this one call, rather than globally.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        joined = left.join_asof(
            right,
            left_on="rd",
            right_on="source_rd",
            by="establishment_id",
            strategy="backward",
            # The one flag that makes this a *strict* as-of join. With exact matches
            # allowed the target inspection would supply its own categoricals on every
            # row that has one, and every leakage test in this component would still
            # pass while the model read the present.
            allow_exact_matches=False,
        )

    resolved = joined.with_columns(
        [pl.col(name).fill_null(UNKNOWN_CATEGORY) for name in EMITTED_CATEGORICALS]
    ).with_columns(
        pl.col("source_inspection_id").cast(pl.Utf8),
        (pl.col("rd") - pl.col("source_rd"))
        .dt.total_days()
        .cast(pl.Int32)
        .alias("days_since_source"),
        pl.col("source_rd").alias("source_inspection_date"),
    )

    out = resolved.select(
        "target_inspection_id",
        "establishment_id",
        "inspection_date",
        *EMITTED_CATEGORICALS,
        "source_inspection_id",
        "source_inspection_date",
        "days_since_source",
    ).sort("target_inspection_id")

    if out.height != features.height:
        raise CategoricalBuildError(
            f"categorical join produced {out.height} rows for {features.height} feature "
            "rows. The join must be one-to-one on target_inspection_id."
        )
    return out


def coverage(table: pl.DataFrame) -> dict[str, float]:
    """Fraction of rows carrying a real (non-UNKNOWN) value, per family.

    Reported rather than optimised. A family with poor coverage is a family whose
    embedding is mostly the UNKNOWN row, and that is a finding about the data rather than
    a problem to engineer around.
    """
    height = table.height
    if height == 0:
        return {name: 0.0 for name in EMITTED_CATEGORICALS}
    return {
        name: float((table[name] != UNKNOWN_CATEGORY).sum()) / height
        for name in EMITTED_CATEGORICALS
    }


def cardinality(table: pl.DataFrame) -> dict[str, int]:
    """Distinct values per family, UNKNOWN included."""
    return {name: int(table[name].n_unique()) for name in EMITTED_CATEGORICALS}


__all__ = [
    "ASSIGNMENT_COLUMNS",
    "COMMUNITY_AREA_COLUMN",
    "EMITTED_CATEGORICALS",
    "RAW_COLUMNS",
    "CategoricalBuildError",
    "build_categoricals",
    "cardinality",
    "coverage",
    "load_history",
]
