# ADR 0001 — Python 3.12 with uv, and a strict one-component-at-a-time scope

**Status:** Accepted · **Date:** 2026-08-15

## Context

Sentinel will eventually span data engineering, gradient-boosted and neural
models, calibration, fairness auditing, constrained optimisation, routing, and
LLM orchestration. The temptation at project start is to scaffold directories
and dependencies for all of it.

## Decision

**Python 3.12**, with **uv** for environment and dependency management, and a
`src/` layout package built by hatchling.

Dependencies are added **only when the component that needs them is being
built**. Component 1 therefore declares exactly six runtime dependencies:
httpx, pydantic, pydantic-settings, polars, pyarrow, duckdb.

No directories are created for future components.

## Rationale

Python is where the entire downstream ecosystem lives (scikit-learn, XGBoost,
SHAP, OR-Tools). Choosing anything else for ingestion would force a language
boundary later.

uv over pip/poetry/conda: it resolves and installs an order of magnitude faster,
produces a committed `uv.lock` for reproducibility, and manages the Python
version itself, so `uv sync` is the whole setup story.

`src/` layout because it makes the installed package the thing under test,
rather than whatever happens to be in the working directory.

The dependency discipline is the important half of this decision. A
`pyproject.toml` listing torch and OR-Tools on day one would be a lie about what
the project does, and a maintenance burden for code that does not exist.

## Alternatives rejected

* **Poetry** — slower, and its lockfile / PEP 621 story has been in flux.
* **conda** — heavyweight; no geospatial or compiled-science dependency yet
  justifies it. Reconsider if PostGIS or heavy geo work arrives.
* **Scaffolding all components now** — creates empty directories that imply
  work which does not exist, and invites premature abstraction.

## Consequences

* Setup is `uv sync`. CI uses the same command.
* Each future component adds its own dependencies in its own commit, so the
  `pyproject.toml` history reads as a record of what was actually built.
* `uv.lock` is committed and must be kept in sync.
