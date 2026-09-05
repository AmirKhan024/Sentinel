# The Sentinel API — a product boundary over Components 1-16

Plain language. No prior machine-learning knowledge assumed.

*Updated after Component 16: the API now also serves the human-review queue and resolution log,
through the same stage-only write pattern described below.*

---

## 1. What problem does this solve?

By Component 14, Sentinel can answer a rich set of questions — who should be inspected, why,
under which policy, on which day, and what happened when they were. The answers live as
timestamped Parquet files under `data/processed/`, each with a JSON manifest recording exactly
how it was produced.

That is everything a data scientist needs. It is nothing a product needs. A frontend cannot open
a Parquet file, does not know which of eleven policy tables and thirteen scheduling tables to
join for a given question, and has no way to know whether "the recommendation for establishment
E1" means this quarter's or last quarter's, this policy's or another one's, without reading this
repository's internals.

The Sentinel API is the boundary that closes that gap: a validated HTTP interface that turns
"artifacts on disk" into "a documented product contract," without adding a single new thing that
computes anything.

## 2. Why doesn't this get a component number?

Because the roadmap already has a name for "Component 15" — OR-Tools routing, and it is blocked
on the same missing data (no inspector, no travel time) that blocked Component 10. Calling this
work "Component 15" too would mean two different things share one name depending on which
document you open. So this is built and documented as **cross-cutting infrastructure that sits
beside the numbered pipeline** — its own ADRs, its own data contract, one paragraph in `README.md`
after the roadmap table, not a row inside it.

## 3. Why did you add an API layer instead of letting the frontend read Parquet?

Three reasons, in order of how much damage skipping them would do:

**Decision scope.** The same establishment appears under many folds, many policies, many
capacities, and — once a schedule has been re-planned — many planning runs. "Give me the
recommendation for E1" is not a well-formed question until enough of those are pinned down. A
library of query helpers could enforce that too, in principle; in practice, every consumer would
have to remember to call it, and one that forgot would silently pick "the first row it found" —
exactly the silent policy decision ADR 0050 refuses to allow.

**A single place errors are consistent.** Missing artifact, ambiguous scope, unknown component,
a write payload that fails validation — every one of these needs to become a specific,
predictable status code and message. Scattered across every frontend consumer, that consistency
is impossible to guarantee.

**A place to draw the "never applies" line.** Components 13 and 14 already accept human input
files. Exposing that over HTTP raises the question of whether a `POST` should trigger a rebuild.
Having one place that answers "no, and here is exactly what it does instead" (ADR 0049) is safer
than leaving that question to whoever builds the first frontend feature that needs it.

## 4. How do you prevent mixing decisions from different folds?

Every endpoint states which scope fields it requires before it will answer at all —
`policy_id`, `fold_set`, `fold_id`, `k_name` for a recommendation, plus `schedule_config_id` for
a schedule. A request that leaves one out gets a `422` naming exactly what's missing, never a
best-guess row. And even a fully-specified scope can still match more than one row — an
establishment inspected twice in the same window — and that case is checked too: the response is
`422 ambiguous_scope` with the specific values (like the two `target_inspection_id`s) that would
resolve it, not an arbitrarily chosen first match. See ADR 0050.

## 5. Can a user edit a model prediction?

No, and there is no code path where they could. `score` and `base_score` on every response are
read verbatim from Component 9's calibrated predictions, carried through Component 13's queue and
Component 14's schedule unchanged. No endpoint accepts a score, a rank, a probability, or a
decision mechanism as input. The three things a person *can* submit — an override, an adjustment,
an execution event — each change a narrowly scoped fact (who's in the queue, when an approved row
is worked, what happened) and none of them can touch a model's output. That boundary is not
policed by the API; it is structural, because the write schemas simply have no field for it.

## 6. How are overrides (and adjustments, and execution events) handled?

All three follow the same shape, and it is not the shape of a normal REST write:

1. The payload is validated against the *exact* pydantic contract Component 13 or 14 already
   defines (`Override`, `Adjustment`, `ExecutionEvent`), by running it through the real parser
   (`parse_overrides`, `parse_adjustments`, `parse_execution_events`) as a one-row file. If the
   batch CLI would refuse it, the API refuses it with the same message.
2. If valid, it is appended — never rewritten — to a JSON-lines file the API owns, in the exact
   list-of-objects shape `sentinel decide --overrides` / `sentinel schedule --adjustments/
   --execution` already read.
3. The response is a receipt: `{request_id, kind, natural_id, status: "pending", staged_at}`.
   **Not** a new schedule. Nothing has been recomputed.

Turning that pending file into an actual new recommendation queue or schedule is still a manual
step — an operator runs the CLI against it, same as always. `GET /v1/staged-requests` lets a
caller see what's pending versus what has since shown up in a committed log (`"applied"`), purely
by comparing ids — the API never does the applying itself.

## 7. Why didn't you make writes apply immediately?

Because "apply immediately" is a much bigger claim than it sounds. `sentinel schedule` doesn't
process one adjustment in isolation — it rebuilds slots, utilization and priority-preservation
numbers for the *whole cell* in one batch, checksums every input before and after, and writes one
manifest for the run. An endpoint that tried to apply a single adjustment on its own would have
to either reimplement all of that inside a request handler (a second, thinner, unaudited copy of
`run_schedule`) or run the real batch command synchronously on every `POST` — turning a write into
a slow, all-or-nothing operation with no natural place to report partial progress. Staging avoids
both: it's instant, it's exactly as safe as writing a file by hand, and it changes nothing about
how Components 13 and 14 compute anything. See ADR 0049 for the full argument, including the
alternatives that were considered and rejected.

## 8. Why didn't you build routing?

Because the dataset still has no inspector, no shift, no duration, no travel time and no road
network — the same reason Component 14 didn't build it, going back to ADR 0019. Adding an HTTP
layer doesn't create operational data that doesn't exist. There is no `/v1/routes` endpoint, by
omission, and no plan to add one until the data does.

## 9. Why didn't you add PostgreSQL, Redis, or a microservice architecture?

Because nothing here needs them yet. Every read is a Parquet file this API already knows how to
find and filter with Polars; every write is one line appended to a flat file. Introducing a
database, a cache, or a second service would be exactly the kind of dependency this project's own
convention refuses — one added because it sounds appropriate for "an API," not because a measured
need exists. If the staging store or the read volume ever outgrows a flat file, that's a decision
to make when the measurement says so, the same way every other dependency in this repository
arrived.

## 10. What would be required to productionize this API?

Being honest about what is *not* here: authentication and authorization (there is none — this is
explicitly a prototype boundary, and the code does not pretend otherwise), an operational process
to drain the staging store on a schedule instead of by hand, monitoring/alerting on artifact
staleness, and load-testing the flat-file staging store under real concurrent write volume. None
of these were skipped by accident; each is named here and in ADR 0048/0049 as a gap a real
deployment would need to close, not a claim this system already makes.

## 11. How does this prepare Sentinel for a frontend?

Every page a product designer would ask for — a dashboard, a recommendation queue, an
establishment detail page, a policy comparison, a schedule view, a backlog view, a decision
timeline — now has a documented endpoint or a small composition of them
(`docs/data_contracts/sentinel_api.md`), each with its required scope, its pagination contract,
and its error shapes specified up front. A frontend engineer can build against that document
without ever opening this repository's Python or Parquet.
