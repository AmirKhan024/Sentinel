"""Offset/limit pagination over an already-filtered, already-sorted frame."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class PageParams:
    offset: int = 0
    limit: int = 50
    sort_column: str | None = None
    descending: bool = False


def slice_frame(frame: pl.DataFrame, page: PageParams) -> tuple[list[dict[str, object]], int]:
    """Return one page of rows as dicts, plus the total row count before slicing."""
    total = frame.height
    sliced = frame.slice(page.offset, page.limit)
    return sliced.to_dicts(), total


__all__ = ["PageParams", "slice_frame"]
