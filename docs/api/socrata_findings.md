# Chicago Data Portal / Socrata API — verified findings

Everything in this document was verified by issuing real requests against the
live API on **2026-08-15**. Nothing here is inferred from documentation alone.
Where a value can drift over time (row counts, dates) it is marked as
*as of* that date.

---

## 1. Endpoint and dataset

| | |
|---|---|
| Portal | Chicago Data Portal, `data.cityofchicago.org` |
| Platform | Socrata, SODA 2.1 |
| Dataset | Food Inspections |
| Dataset ID (4x4) | `4ijn-s7e5` |
| Resource endpoint | `https://data.cityofchicago.org/resource/4ijn-s7e5.json` |
| Metadata endpoint | `https://data.cityofchicago.org/api/views/4ijn-s7e5.json` |
| Authentication | **None required.** Verified with anonymous requests. |

A Socrata **application token** is optional. It does not grant additional data
access; it moves the caller out of the shared anonymous throttling pool. The
project supports it via `SENTINEL_SOCRATA_APP_TOKEN`, sent as the `X-App-Token`
header, but does not require it.

### Verified

```
GET /resource/4ijn-s7e5.json?$limit=2   ->  HTTP 200
```

---

## 2. Row count

```
GET /resource/4ijn-s7e5.json?$select=count(*)
->  [{"count":"314245"}]
```

**314,245 rows** as of 2026-08-15. Note that even `count(*)` is returned as a
*string*, which is the first hint of the encoding behaviour described in §5.

---

## 3. Schema discovery

Every response carries the dataset's declared schema in two headers:

```
X-SODA2-Fields: ["inspection_id","dba_name",...]
X-SODA2-Types:  ["number","text",...]
```

The two arrays are positionally aligned. Capturing them means we get the
source-declared schema for free with each request, without a second metadata
call. Sentinel records both in every ingestion manifest.

The `/api/views/{id}.json` metadata endpoint gives the same information in a
richer form (`columns[].fieldName`, `columns[].dataTypeName`, plus human
labels) and also exposes `rowsUpdatedAt`. Sentinel does not currently call it;
the headers are sufficient.

### The 22 fields

17 source columns:

| # | Field | Socrata type | Human label |
|---|---|---|---|
| 1 | `inspection_id` | number | Inspection ID |
| 2 | `dba_name` | text | DBA Name |
| 3 | `aka_name` | text | AKA Name |
| 4 | `license_` | number | License # |
| 5 | `facility_type` | text | Facility Type |
| 6 | `risk` | text | Risk |
| 7 | `address` | text | Address |
| 8 | `city` | text | City |
| 9 | `state` | text | State |
| 10 | `zip` | number | Zip |
| 11 | `inspection_date` | calendar_date / floating_timestamp | Inspection Date |
| 12 | `inspection_type` | text | Inspection Type |
| 13 | `results` | text | Results |
| 14 | `violations` | text | Violations |
| 15 | `latitude` | number | Latitude |
| 16 | `longitude` | number | Longitude |
| 17 | `location` | location | Location |

Plus 5 Socrata-generated spatial annotations, produced by joining the row's
point geometry against boundary layers:

| Field | Meaning |
|---|---|
| `:@computed_region_awaf_s7ux` | Historical Wards 2003-2015 |
| `:@computed_region_6mkv_f3dw` | Zip Codes |
| `:@computed_region_vrxf_vc4k` | Community Areas |
| `:@computed_region_bdys_3d7i` | Census Tracts |
| `:@computed_region_43wa_7qmu` | Wards |

The `:@` prefix marks them as system-generated rather than city-supplied.

> Note: the `inspection_date` type is reported as `calendar_date` by the
> metadata endpoint but as `floating_timestamp` by the `X-SODA2-Types` header.
> Both describe a timezone-less date/time. Sentinel stores the raw string and
> does not attempt to reconcile them at ingestion time.

---

## 4. Pagination

Parameters: `$limit`, `$offset`, `$order`.

### Page size

```
GET ...?$select=inspection_id&$limit=60000   ->  60,000 rows returned
```

There is **no 50,000-row cap** on this endpoint, contrary to what older Socrata
documentation suggests for SODA 2.0. Sentinel defaults to a page size of 50,000
as a comfortable, conservative value.

### `$order` is mandatory for correctness

Socrata does not guarantee a stable row order between requests. Without an
explicit total order, `$offset` paging can return overlapping or skipped rows.
Verified that ordering produces contiguous, non-overlapping pages:

```
?$select=inspection_id&$order=inspection_id&$limit=3&$offset=0
->  44247, 44248, 44249

?$select=inspection_id&$order=inspection_id&$limit=3&$offset=3
->  44250, 44251, 44252
```

Sentinel always sends `$order=inspection_id`. This is not optional and is
enforced in `build_params()`, which raises if the order column is empty.

### Termination

The API does not return a "has more pages" flag or a cursor. Pagination ends
when a page comes back with fewer rows than requested, or with zero rows.
Sentinel treats both as terminal.

---

## 5. Everything is a string

This is the single most important finding for the raw data contract.

**Every value in the JSON response is a string**, regardless of the type the
dataset declares:

```json
{
  "inspection_id": "2641210",       // declared "number"
  "license_": "1771904",            // declared "number"
  "zip": "60644",                   // declared "number"
  "latitude": "41.86568627741837",  // declared "number"
  "inspection_date": "2026-08-14T00:00:00.000"   // declared calendar_date
}
```

The one exception is `location`, which is a nested JSON **object**:

```json
"location": {
  "latitude": "41.86568627741837",
  "longitude": "-87.76566985156941",
  "human_address": "{\"address\": \"\", \"city\": \"\", \"state\": \"\", \"zip\": \"\"}"
}
```

Note that `human_address` is itself a JSON string containing JSON, and in the
observed records its fields are empty.

Consequence for Sentinel: the raw Parquet layer stores every column as `Utf8`,
and `location` is re-serialized to its compact JSON string. See
[the raw data contract](../data_contracts/food_inspections_raw.md) and
[ADR 0002](../decisions/0002-parquet-raw-storage.md).

There is a live test (`tests/test_smoke_live.py`) that asserts this encoding
still holds, so if Chicago ever changes it, we find out.

---

## 6. `$order` suppresses the computed_region columns

Discovered while implementing pagination, and reproducible:

| Request | Fields in `X-SODA2-Fields` | computed_region present |
|---|---|---|
| `?$limit=2` | 22 | yes |
| `?$limit=2&$order=inspection_id` | **17** | **no** |
| `?$limit=2&$order=inspection_id&$offset=0` | 17 | no |
| `?$limit=2&$order=inspection_id DESC` | 17 | no |

Adding `$order` — which correctness requires — silently drops the five
`:@computed_region_*` columns from both the header and the records.

They can be recovered by naming them in `$select`:

```
?$select=<all 22 fields>&$order=inspection_id&$limit=2
->  22 fields, computed_region columns present
```

### How Sentinel handles this

Hardcoding a 22-name `$select` would work but would create a worse problem: if
Chicago ever adds a column, a stale `$select` would silently exclude it, which
is exactly the kind of quiet data loss this project is trying to avoid.

Instead, ingestion **discovers the field list at runtime**:

1. One unordered request (`?$limit=1`) — the only shape that reveals all 22
   field names in `X-SODA2-Fields`.
2. Pagination then sends `$select=<discovered fields>&$order=inspection_id`.

Cost: one extra HTTP request per ingestion run. Benefit: the full column set is
captured, and a new upstream column is picked up automatically.

This is controlled by `SENTINEL_INCLUDE_COMPUTED_REGIONS` (default `true`).
Setting it to `false` skips discovery and accepts the 17-column projection.

---

## 7. Error behaviour

Errors are returned as JSON with a machine-readable `errorCode`:

```
GET ...?$select=nope_col   ->  HTTP 400
{
  "message": "Query coordinator error: query.soql.no-such-column; No such column: nope_col; ...",
  "errorCode": "query.soql.no-such-column",
  "data": { "column": "nope_col", "dataset": "foxtrot.7116", ... }
}
```

Sentinel's retry policy follows from this:

| Condition | Behaviour | Reason |
|---|---|---|
| 429 | retry with backoff | throttling, transient |
| 500, 502, 503, 504 | retry with backoff | upstream fault, transient |
| timeout / transport error | retry with backoff | network, transient |
| **400, 401, 403, 404, other 4xx** | **raise immediately** | our query is wrong; retrying hides the bug |

The response body is attached to the raised error so the `errorCode` above
reaches the operator rather than being swallowed.

---

## 8. Other observed response headers

```
X-SODA2-Truth-Last-Modified: Sat, 15 Aug 2026 09:06:01 GMT
X-SODA2-Data-Out-Of-Date:    false
Last-Modified:               Sat, 15 Aug 2026 09:06:01 GMT
ETag:                        W/"..."
Access-Control-Allow-Origin: *
```

`X-SODA2-Truth-Last-Modified` and `ETag` are potentially useful for incremental
or conditional fetching. **Not used yet** — Component 1 does full pulls only.
Recorded here as a lead for a future incremental-ingestion component.

---

## 9. Things deliberately not investigated

* `$where` / `$group` / SoQL filtering — not needed until Component 2 defines
  which rows are in scope.
* CSV and GeoJSON output formats — JSON is sufficient.
* The `/api/views` metadata endpoint as a schema source — the response headers
  already provide it.
* Incremental / conditional fetching via `ETag` or `Last-Modified`.
* Rate limits under sustained load. Anonymous access was sufficient for a
  5,000-row development pull with no throttling observed.
