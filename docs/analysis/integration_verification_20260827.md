# End-to-end integration verification — 2026-08-27

Not a component's own findings document. This records what was actually executed, against real
committed artifacts, to answer one question: does Sentinel work as an integrated system through
its real interfaces, not just through unit tests?

## Scope

Executed, not just inspected: `sentinel decide`, `sentinel schedule`, `sentinel review` against
the real `data/processed/` artifacts; `sentinel serve` (real HTTP requests via curl against every
major endpoint, including negative/failure cases); the frontend dev server and its own test
suite, against the real running API.

## Result

The core deterministic pipeline (Components 1–14, 16) works end-to-end through its real
interfaces. Two real integration bugs were found and fixed; both were invisible to the existing
unit-test suite because a test fixture on one side of each bug encoded the same wrong assumption
as the code on the other side.

## Bug 1 — calibrated model name leaked into Component 11's lookup

**Symptom:** every real establishment's explanation lookup, through
`GET /v1/establishments/{id}`, reported `"'xgboost_platt' is not a recognised model in the
explanation support table"` — for a model Component 11 genuinely explains.

**Root cause:** Component 13/14 carry Component 9's *calibrated* model name
(`xgboost_platt`); Component 11's own tables carry the *base* name (`xgboost`) and never a
calibrated one — `docs/data_contracts/explanations.md` §0a already documents this exact mismatch
as a known footgun ("caused a silent failure once"). `establishment_service.py` passed the
calibrated name straight through to the explanation lookup.

**Why unit tests missed it:** `tests/api/conftest.py`'s `explanation_support_row()` /
`explanation_case_row()` / `explanation_value_row()` fixtures defaulted `model_name` to
`"lightgbm_platt"` — the *same wrong* calibrated-name assumption the buggy code made. The
fixture and the bug agreed with each other, so the test that exercised this path
(`test_establishment_bundle_includes_explanation_when_sampled`) passed for the wrong reason.

**Fix:** `sentinel.api.services.explain_service.base_model_name_of()` strips a known
`sentinel.calibration.definitions.Method` suffix (`_platt`, `_isotonic`) before the lookup;
`establishment_service.py` calls it once, at the one place a calibrated name crosses into
Component 11's namespace. The explanation contract itself (`/v1/explanations/{id}`) is
unchanged — it still expects the base name, exactly as documented.

**Fixtures corrected** to the real Component 11 convention (bare model names); a regression
test (`test_establishment_bundle_resolves_the_calibrated_model_name_to_the_base_name`) and a
direct unit test of `base_model_name_of` were added.

**Verified:** live, against real data — `GET /v1/establishments/EST-00000327239?...` now returns
a full 30-feature SHAP explanation instead of a false "not explainable."

## Bug 2 — frontend pointed at a port nothing serves

**Symptom:** the frontend's own test suite failed 12 of 33 tests with "Could not reach the
Sentinel API"; the dev server, if opened in a browser, would have failed identically.

**Root cause:** `frontend/.env` (a local, untracked file) set
`VITE_SENTINEL_API_BASE_URL=http://127.0.0.1:8010`. Nothing has ever listened on 8010 in this
environment. `frontend/.env.example`, the MSW test fixtures, and `Settings.api_port`'s own
default all agree on `8000`; only the local `.env` disagreed.

**Fix:** corrected `frontend/.env` to `http://127.0.0.1:8000`, matching `.env.example` and the
real `sentinel serve` default.

**Verified:** frontend test suite went from 12 failed / 21 passed to 33 / 33 passed; the live
dev server, restarted, resolved `VITE_SENTINEL_API_BASE_URL` to `8000` and received real,
CORS-permitted 200 responses from the running API for `/healthz`, `/v1/runs`,
`/v1/manifests/policy`, `/v1/manifests/scheduling`.

## What was executed, not merely inspected

- `sentinel decide --dry-run --report` against the real, committed recommendation chain: 22
  checks, 0 errors, 4 advisories (matching Component 13's own recorded findings).
- `sentinel review` twice, dry-run, against real data: 70,791 cases flagged both times,
  byte-identical — determinism confirmed live, not only in the test suite.
- `sentinel serve`, real HTTP requests: recommendations, schedule, backlog, review queue, review
  resolutions (including a real staged `POST`), establishment bundles, explanations, manifests,
  run discovery, staged-requests reconciliation.
- Negative cases: ambiguous scope (422), unknown row (404), unknown component (404), duplicate
  staged id with a different payload (409), invalid `resolution_action` (422), a
  `refer_to_override` resolution missing its required pointer (422), a resolution file with a
  duplicate `review_id` refused by the CLI (exit 1), a schedule command fed a
  deliberately-truncated recommendations file (exit 1, refused rather than silently scheduling a
  broken queue).
- Frontend dev server (Vite) and its own Vitest suite, against the real running API.

## A real limitation found, not fixed here

`sentinel review` does not cross-validate that a manually overridden `--recommendations` file
corresponds to the same run the auto-discovered (or explicitly passed) `--schedule` /
`--execution` artifacts were built from. Feeding it a deliberately mismatched, truncated
recommendations file while the schedule auto-discovered the real, full one produced a
plausible-looking but cross-run-inconsistent queue rather than a refusal. Under normal operation
(no manual `--recommendations` override) this cannot arise, because the CLI always discovers the
latest of every artifact, and the latest recommendations, schedule and execution log are
naturally the ones that were actually built from each other. Recorded as an open question rather
than fixed here, because closing it requires a design decision (error vs. advisory, and against
which provenance field) that this verification pass is not the place to make unilaterally.

## Regression, after both fixes

`pytest`: 3,190 passed, 3 deselected, 0 failed (up from 3,181 — 9 new tests: 1 establishment
regression test, 6 `base_model_name_of` unit cases, 2 review-manifest/run-discovery tests added
during this same pass). `ruff check .`: clean. `ruff format --check .`: the same 10 pre-existing
Component 9 files, none touched by this work. `mypy src/sentinel`: clean, 191 files. Frontend:
`vitest run` 33/33, `tsc -b --noEmit` clean.
