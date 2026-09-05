# Frontend product clarity pass — 2026-08-28

Not a new backend capability. This is a UI/UX pass over the existing, already-verified
end-to-end system (see `docs/analysis/integration_verification_20260827.md`), addressing a
product clarity problem the integration verification did not: the frontend was technically
correct but assumed the visitor already understood Sentinel's internal vocabulary.

## What was wrong

The first frontend pass (built for API product-testing) led every page with the same technical
form: a raw scope selector (`policy_id`, `fold_set`, `fold_id`, `k_name`...), tables whose column
headers were literal API field names (`decision_mechanism`, `schedule_status`,
`backlog_reason`), and an Overview page whose only content was a manifest dump (row counts,
sha256-adjacent checks, a raw JSON `advisories` list). Nothing was wrong — every value was real —
but nothing translated it. A first-time visitor had to already know what a "fold" or a "policy
id" was before the page would show them anything at all.

## What changed

**Progressive disclosure, not removal.** Every technical field, code, and identifier that existed
before still exists — moved into a collapsed "Technical details" section on each page rather than
deleted. `src/lib/copy.ts` is the one place a technical code gets a plain-language explanation
beside it (e.g. `no_execution_record_on_scheduled_row` → "This scheduled inspection does not
currently have a matching record of what happened"); the raw code is never hidden, only
re-explained.

**An automatically chosen, real starting scope.** `useDefaultScope` fills in a full, valid
decision scope the moment the policy/scheduling manifests load — the most recent fold, the
manifest's own `selected_model`/`primary_k_level`, the scheduling manifest's own default
configuration — so a first-time visitor sees real data immediately instead of an empty form. It
never overwrites a scope the visitor (or a bookmarked URL) already set, and the full manual scope
form remains one click away under "Advanced options."

**A new page for an existing API surface.** Component 16's human-review queue had no frontend
page at all before this pass (only the API endpoints existed, verified directly in the 2026-08-27
integration pass). `HumanReviewPage`, `api/review.ts`, and the `ReviewCaseOut`/
`ResolutionLogRowOut` types are new; the underlying `/v1/review/*` endpoints are unchanged.

**An establishment journey, not a field dump.** `EstablishmentDetailPage` now renders one
establishment's path through all five layers (available information → prioritization →
recommendation → schedule → human review) as a five-step, visually ordered journey, each step in
plain language, with the previous field-by-field technical view preserved underneath.

## A real bug found and fixed during this pass

`useDefaultScope`'s first version called `setScopeField` once per field (up to six times) inside
one effect. All six calls appeared to work in isolation but only the *last* field ever actually
landed in the URL: react-router's `setSearchParams` updater receives the currently *committed*
search params at the time each call is scheduled, not the result of a sibling call made earlier
in the same synchronous block, so repeated rapid calls do not compose the way `useState`'s
functional updater does. The page would get stuck showing "Preparing an inspection plan..."
forever, because only one of the required scope fields was ever actually set.

Caught by the frontend's own test suite, not by manual inspection: tests that asserted the page
eventually showed real data (rather than merely asserting the loading state appeared) timed out
consistently. Fixed by adding `setScopeFields` (a bulk setter that composes every field into one
`setSearchParams` call) to `useDecisionScope`, and having `useDefaultScope` use it instead of
repeated single-field calls. `useDecisionScope`'s existing single-field `setScopeField` is
unchanged and still correct for its normal use — one dropdown, one change, one call.

## What was not changed

No ML model, no Component 13 policy semantics, no Component 14 scheduling semantics, no
Component 16 trigger semantics, and no API contract changed. Every field shown anywhere in the
frontend still traces to a real API response field — `lib/copy.ts` adds a second, human-readable
string beside a technical value, it never replaces or recomputes one. No score is re-labelled
"High Risk" / "Low Risk" — the policy contract makes no such categorical claim, only a rank, and
the UI states that explicitly rather than inventing a category the backend does not support.

## Verified

Live, against the real running API (not only against mocks): the exact default scope the
frontend now computes automatically (`policy_id=pure_risk`,
`model_name=xgboost_platt` from the policy manifest's `selected_model`,
`k_name=k_1_day` from `primary_k_level`, `fold_id=quarterly-2026Q2` as the most recent real
fold, `schedule_config_id=strict_priority__observed_calendar` from the scheduling manifest's
default) returns 1,638 establishments considered, 28 recommended, 28 scheduled, 0 waiting for
capacity, and 28 flagged for human review — all real numbers, confirmed via direct requests to
the running `sentinel serve` process. The frontend's own test suite (43 tests, including one new
file for the Human Review page) passes against MSW-mocked versions of the same endpoints;
`tsc -b --noEmit` and `oxlint` are both clean. No backend file was modified in this pass; the
full backend suite (3,190 tests) and `ruff`/`mypy` were re-run and remain unchanged and clean.
