# Sentinel Frontend

A React + TypeScript + Vite frontend over the Sentinel API (`src/sentinel/api/`, run via
`uv run sentinel serve`). It began as cross-cutting infrastructure for the historical/backtest
pages (Overview, Inspection Priorities, Inspection Schedule, Waiting for Capacity, Human Review,
Establishment detail) and has since grown to include Component 21's own pages — **Field Plan**
(Component 20's geographic organization) and **Plan Review** (Component 21's supervisor review,
adjustment, and approval of the live, `planning_date`-scoped operational plan) — which are genuine
product surfaces for those two components, not a separate future deliverable. **Today** (the
landing page, `/`) is a third such live page: it reads the same `planning_date`-scoped endpoints
as Field Plan and Plan Review, and is never scoped by a historical evaluation fold. See
`lib/today.ts` and `STATUS.md`'s "The 'Today = April 1, 2026' bug" for why that distinction is
load-bearing, not stylistic.

**The product story, not just a data browser.** The first pass of this frontend (still available
in spirit under every page's "Technical details") exposed the API's raw shapes directly — policy
IDs, fold identifiers, manifest checks. A second pass rebuilt the primary experience around plain
language, so an inspection supervisor — not just an engineer — can open Sentinel and understand
within seconds what it recommends and why: **Overview** tells the story and shows real
operational counts; **Inspection Priorities**, **Inspection Schedule**, **Waiting for Capacity**
and **Human Review** each answer one operational question in plain language; the **Establishment
detail** page shows one establishment's journey through all four. Every technical concept (a
fold, a policy id, a raw reason code, a manifest) still exists and is still real — it just lives
one click away under "Technical details," never as the first thing a visitor has to parse. See
`docs/analysis/frontend_product_clarity_20260828.md` for the full before/after and rationale.

**Actionable, not read-only.** A third pass ("Actionability & Operational Workflow") closed the
gap between the UI telling a supervisor what to investigate and actually letting them act: the
Establishment detail page exposes real forms for four backtest-side write contracts —
`POST /v1/policy/overrides` (change the priority decision), `POST /v1/schedule/adjustments`
(move/cancel a planned inspection), `POST /v1/execution/events` (record an inspection outcome)
and `POST /v1/review/resolutions` (resolve a flagged case) — each submitting exactly the payload
its backend contract requires, validated the same way the batch CLI validates it. The **Plan
Review** page adds two more, for the live operational plan: `POST /v1/plan-review/decisions`
(`PlanDecisionForm` — keep, move to a later day, change field-work order, or do not proceed with
one establishment) and `POST /v1/plan-review/approve` (`PlanApprovalPanel` — approve the whole
plan). All six follow the same discipline. Every write is **staged, never applied** (ADR 0049):
the UI says so explicitly on every confirmation, and a "Decision history" section on each
establishment's page shows both committed and still-pending entries for the four backtest-side
contracts, reconciled against `/v1/staged-requests`. Turning a staged request into a new artifact
remains a manual `sentinel decide` / `sentinel schedule` / `sentinel review` / `sentinel
review-plan` / `sentinel approve-plan` run by an operator — the UI was not given a button that
pretends to do that, because no such capability exists on the backend to expose honestly.

## Running it

Two terminals, from the repository root:

```bash
# terminal 1 — the Sentinel API
uv run sentinel serve --reload

# terminal 2 — the frontend
cd frontend
npm install
cp .env.example .env   # edit VITE_SENTINEL_API_BASE_URL if the API isn't on 127.0.0.1:8000
npm run dev            # http://localhost:5173
```

The API has no data until the relevant `sentinel decide` / `sentinel schedule` / `sentinel
explain` runs have produced artifacts under `data/processed/`. If a page shows `artifact_not_found`,
that component hasn't been run yet — this is not a frontend bug.

## Commands

```bash
npm run dev         # Vite dev server
npm run build        # tsc -b && vite build
npm run typecheck    # tsc -b --noEmit
npm run lint         # oxlint
npm run test          # vitest (watch mode)
npm run test -- --run # vitest, single run — the CI-equivalent gate
```

## How decision scope works here

The **Field Plan** and **Plan Review** pages are scoped by `planning_date`, not by the
`policy_id`/`fold_set`/`fold_id`/`k_name`/`schedule_config_id` scope described below -- they read
the current live operational plan and take no scope selector at all. Everything in this section
applies to the historical/backtest pages only.

Every read endpoint on the backtest side requires an explicit decision scope (`policy_id`, `fold_set`, `fold_id`,
`k_name`, and `schedule_config_id` for schedule/backlog views) — the API returns `422
ambiguous_scope` rather than guessing (ADR 0050), and this frontend still refuses to fire an
incomplete request. What changed is who fills the scope in: `useDefaultScope` now picks a real,
verified-non-empty scope automatically the moment the policy/scheduling manifests load (the most
recent fold, the manifest's own `selected_model`/`primary_k_level`, the scheduling manifest's own
default configuration), so a first-time visitor sees real data immediately rather than an empty
form. It never overwrites a field the visitor (or a bookmarked URL) already set, and the full
manual scope form is always available under "Advanced options" for comparing a different plan.

Scope dropdowns are populated two ways:

- `policy_id`, `model_name`, `k_name`, `schedule_config_id` are read live from
  `GET /v1/manifests/policy` and `GET /v1/manifests/scheduling` — never hardcoded.
- `fold_set`/`fold_id` are the one exception: **no API endpoint enumerates them**, so
  `src/api/folds.ts` hardcodes the real, verified fold table (17 `quarterly` folds
  `quarterly-2022Q2`…`quarterly-2026Q2`, plus the single `covid_shift-2020H2-2021` fold) read
  directly from `data/processed/evaluation/evaluation_folds_*.parquet`. This is real data, not an
  invented placeholder — it is simply the one scope dimension this API doesn't expose a discovery
  route for.

## What was deliberately not built

- No routing or map — Component 15 is blocked (no inspector, no travel time), and no such
  endpoint exists to build against.
- No authentication — the API has none; this is a local prototype boundary. Every write form
  collects a free-text "actor" name (remembered per-browser in `localStorage`) because the
  contracts require one — this is attribution, not identity verification.
- No instant application of a write — every submission is staged only; nothing in the UI
  recomputes a queue or a schedule, because no such endpoint exists on the backend (ADR 0049).
- No client-side sort-column picker — the API fixes the sort column per endpoint server-side and
  only exposes `descending`; a column picker would contradict that determinism guarantee.
- No fabricated data — every field rendered traces to an actual API response field; a missing or
  null field renders as "—" or an explicit "not available" message, never invented or silently
  dropped. Execution-outcome options come from `GET /v1/execution/contract`, not a hardcoded list.
- No duplicated model/policy/scheduling logic — this frontend reads and writes through the API's
  existing contracts; it computes nothing itself (no percentile, capacity, or validation logic is
  reimplemented — `relativePriorityLabel` only formats `model_rank`/`n_universe`, both already
  computed server-side).
- No risk verdicts — `is_selected` is presented as "selected for this plan," never
  "recommended"/"dangerous," because the same establishment can flip between selected and not
  purely from a capacity change with nothing about it different (see `lib/copy.ts`).

## Architecture

```
src/
  api/        typed fetch client (apiFetch for GET, apiPost for the four write endpoints),
              per-resource modules (overrides.ts, adjustments.ts, execution.ts, review.ts, ...),
              error classification, TS types mirroring every schema in src/sentinel/api/schemas/
  hooks/      useApiQuery (fetch + cancel-in-flight), useDecisionScope (scope in URL params,
              plus a bulk setScopeFields), useDefaultScope (auto-fills a real scope from live
              manifests), useManifestOptions (cached manifest -> dropdown options),
              useStagedSubmit (the submit lifecycle every write form shares), useActor
              (the free-text, localStorage-remembered "who is deciding" field every write needs)
  lib/
    copy.ts   plain-language translations of every technical code the API returns (decision
              mechanisms/reasons, warnings, schedule statuses, review triggers, override/
              adjustment/execution actions), relative-priority and capacity-honesty framing, the
              one-time model-limitation disclosure (HOW_TO_USE_PRIORITY, now collapsed behind
              "How Sentinel prioritizes locations" rather than always-visible), and
              `planLabelForToday`/`planStalenessNote` (the operational-date honesty layer -- see
              `today.ts` below) -- the raw code is always still shown too, in Technical details;
              nothing here is hidden, only re-explained
    today.ts  the one place "today" is computed (`currentOperationalDate()`, from `new Date()`,
              never a hardcoded literal) and compared against a plan's own `planning_date` --
              this is what fixed the "Today" page silently showing April 1, 2026 (see STATUS.md's
              "The 'Today = April 1, 2026' bug" for the root cause and fix)
    ids.ts    generates a natural id (override_id, adjustment_id, ...) for a new write; the
              backend imposes no format beyond non-blank and unique
  components/
    scope/    InspectionPlanSelector (the plain-language front end for scope), ScopeSelector
    actions/  OverrideForm, AdjustmentForm, ExecutionOutcomeForm, ResolutionForm (one component
              per backtest-side write contract), PlanDecisionForm, PlanApprovalPanel (the two
              live-plan write contracts), DecisionHistory (committed + staged entries for one
              establishment, across the four backtest-side contracts), StagedReceiptNotice (the
              honest "staged, not applied" confirmation every write form shares)
    common/   TechnicalDetails (progressive disclosure), SummaryCard, ManifestChecksPanel, plus
              the generic table/pagination/state components
  pages/      **Today** (the live, `planning_date`-scoped operational plan -- Components 17-21;
              reads the same endpoints as Field Plan/Plan Review, never a historical fold),
              Field Plan (Component 20's geographic organization), Plan Review (Component 21's
              supervisor review/adjustment/approval of the live plan), Schedule Day View
              (`/schedule/day` -- the historical, fold-scoped day-by-day schedule Today used to
              be; moved here, not deleted), Overview/"Backtest Summary" (the historical-analysis
              hub), Inspection Priorities (recommendations), Establishment detail (a seven-step
              journey: information, priority position + evidence, selection, schedule, human
              review, decision history, current field plan -- each step ending in the relevant
              action form or link), Inspection Schedule, Waiting for Capacity (backlog), Human
              Review (two sections:
              Decision Review and Missing Outcomes, filtered server-side by trigger)
  test/       msw mock server with a stateful in-memory staging store (mirrors the real
              StagingService so a POST in a test is reflected by a following GET), fixtures
              matching the real API contract, setup
```

See the root `README.md`'s "The Sentinel Frontend" section for how this fits into the project as
a whole, and `docs/data_contracts/sentinel_api.md` for the API contract this frontend is built
against.
