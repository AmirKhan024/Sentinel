"""Tests for the Socrata client: request construction, pagination, retries, errors."""

from __future__ import annotations

import httpx
import pytest
import respx

from sentinel.ingest.socrata import (
    SocrataClient,
    SocrataRequestError,
    SocrataResponseError,
    build_params,
)
from tests.conftest import (
    FIELD_NAMES,
    FIELD_TYPES,
    SODA_HEADERS,
    TEST_RESOURCE_URL,
    make_records,
)


def make_client(**overrides: object) -> SocrataClient:
    kwargs: dict[str, object] = {
        "resource_url": TEST_RESOURCE_URL,
        "page_size": 10,
        "order_column": "inspection_id",
        "max_retries": 2,
        "retry_backoff": 0.0,
        "sleep": lambda _seconds: None,
    }
    kwargs.update(overrides)
    return SocrataClient(**kwargs)  # type: ignore[arg-type]


# --- request construction ------------------------------------------------


def test_build_params_includes_order_for_stable_pagination() -> None:
    params = build_params(limit=500, offset=1000, order_column="inspection_id")
    assert params == {"$limit": "500", "$offset": "1000", "$order": "inspection_id"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": 0, "offset": 0, "order_column": "id"}, "limit must be positive"),
        ({"limit": -1, "offset": 0, "order_column": "id"}, "limit must be positive"),
        ({"limit": 10, "offset": -5, "order_column": "id"}, "offset must be non-negative"),
        ({"limit": 10, "offset": 0, "order_column": ""}, "order_column is required"),
    ],
)
def test_build_params_rejects_invalid_input(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_params(**kwargs)  # type: ignore[arg-type]


@respx.mock
def test_fetch_page_sends_expected_query_parameters() -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(3), headers=SODA_HEADERS)
    )
    with make_client() as client:
        client.fetch_page(offset=20, limit=10)

    request = route.calls.last.request
    assert dict(request.url.params) == {
        "$limit": "10",
        "$offset": "20",
        "$order": "inspection_id",
    }


@respx.mock
def test_app_token_sent_as_header_when_configured() -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=[], headers=SODA_HEADERS)
    )
    with make_client(app_token="secret-token") as client:
        client.fetch_page(offset=0, limit=10)

    assert route.calls.last.request.headers["X-App-Token"] == "secret-token"


@respx.mock
def test_no_app_token_header_when_unset() -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=[], headers=SODA_HEADERS)
    )
    with make_client() as client:
        client.fetch_page(offset=0, limit=10)

    assert "X-App-Token" not in route.calls.last.request.headers


@respx.mock
def test_fetch_page_captures_declared_schema_from_headers() -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(2), headers=SODA_HEADERS)
    )
    with make_client() as client:
        page = client.fetch_page(offset=0, limit=10)

    assert page.field_names == FIELD_NAMES
    assert page.field_types == FIELD_TYPES
    assert page.limit == 10
    assert page.offset == 0


# --- pagination ------------------------------------------------------------


@respx.mock
def test_iter_pages_walks_offsets_until_short_page() -> None:
    """Three pages: full, full, then short. Offsets must advance by rows received."""
    responses = [
        httpx.Response(200, json=make_records(10, start=0), headers=SODA_HEADERS),
        httpx.Response(200, json=make_records(10, start=10), headers=SODA_HEADERS),
        httpx.Response(200, json=make_records(4, start=20), headers=SODA_HEADERS),
    ]
    route = respx.get(TEST_RESOURCE_URL).mock(side_effect=responses)

    with make_client() as client:
        pages = list(client.iter_pages())

    assert [len(p.records) for p in pages] == [10, 10, 4]
    assert [p.offset for p in pages] == [0, 10, 20]
    assert [c.request.url.params["$offset"] for c in route.calls] == ["0", "10", "20"]
    # The short page terminates the loop: no fourth request.
    assert route.call_count == 3


@respx.mock
def test_iter_pages_stops_on_empty_page() -> None:
    """A full page followed by an empty one ends pagination without yielding empty."""
    responses = [
        httpx.Response(200, json=make_records(10), headers=SODA_HEADERS),
        httpx.Response(200, json=[], headers=SODA_HEADERS),
    ]
    respx.get(TEST_RESOURCE_URL).mock(side_effect=responses)

    with make_client() as client:
        pages = list(client.iter_pages())

    assert len(pages) == 1
    assert len(pages[0].records) == 10


@respx.mock
def test_total_limit_truncates_final_page_request() -> None:
    """With total_limit=25 and page_size=10, the third request asks for only 5 rows."""
    responses = [
        httpx.Response(200, json=make_records(10, start=0), headers=SODA_HEADERS),
        httpx.Response(200, json=make_records(10, start=10), headers=SODA_HEADERS),
        httpx.Response(200, json=make_records(5, start=20), headers=SODA_HEADERS),
    ]
    route = respx.get(TEST_RESOURCE_URL).mock(side_effect=responses)

    with make_client() as client:
        pages = list(client.iter_pages(total_limit=25))

    assert sum(len(p.records) for p in pages) == 25
    assert [c.request.url.params["$limit"] for c in route.calls] == ["10", "10", "5"]
    assert route.call_count == 3


@respx.mock
def test_total_limit_smaller_than_page_size_makes_one_request() -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(3), headers=SODA_HEADERS)
    )
    with make_client() as client:
        pages = list(client.iter_pages(total_limit=3))

    assert route.call_count == 1
    assert route.calls.last.request.url.params["$limit"] == "3"
    assert sum(len(p.records) for p in pages) == 3


@respx.mock
def test_total_limit_zero_makes_no_requests() -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(3), headers=SODA_HEADERS)
    )
    with make_client() as client:
        assert list(client.iter_pages(total_limit=0)) == []
    assert route.call_count == 0


# --- retry behaviour -------------------------------------------------------


@respx.mock
def test_retries_on_500_then_succeeds() -> None:
    responses = [
        httpx.Response(500, text="upstream boom"),
        httpx.Response(200, json=make_records(2), headers=SODA_HEADERS),
    ]
    route = respx.get(TEST_RESOURCE_URL).mock(side_effect=responses)

    with make_client() as client:
        page = client.fetch_page(offset=0, limit=10)

    assert route.call_count == 2
    assert len(page.records) == 2


@respx.mock
def test_retries_on_429_throttling() -> None:
    responses = [
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(200, json=make_records(1), headers=SODA_HEADERS),
    ]
    route = respx.get(TEST_RESOURCE_URL).mock(side_effect=responses)

    with make_client() as client:
        client.fetch_page(offset=0, limit=10)

    assert route.call_count == 2


@respx.mock
def test_retries_on_timeout_then_succeeds() -> None:
    responses: list[object] = [
        httpx.TimeoutException("timed out"),
        httpx.Response(200, json=make_records(1), headers=SODA_HEADERS),
    ]
    route = respx.get(TEST_RESOURCE_URL).mock(side_effect=responses)

    with make_client() as client:
        page = client.fetch_page(offset=0, limit=10)

    assert route.call_count == 2
    assert len(page.records) == 1


@respx.mock
def test_retry_budget_is_bounded_and_then_raises(no_sleep: object) -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(503, text="down"))

    with make_client(sleep=no_sleep) as client, pytest.raises(SocrataRequestError) as exc:
        client.fetch_page(offset=0, limit=10)

    # max_retries=2 means 3 attempts total, then a hard failure.
    assert route.call_count == 3
    assert exc.value.status_code == 503
    assert "down" in str(exc.value)
    assert no_sleep.delays == [0.0, 0.0]  # type: ignore[attr-defined]


@respx.mock
def test_backoff_is_exponential() -> None:
    delays: list[float] = []
    respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(500, text="boom"))

    client = make_client(max_retries=3, retry_backoff=1.0, sleep=delays.append)
    with client, pytest.raises(SocrataRequestError):
        client.fetch_page(offset=0, limit=10)

    assert delays == [1.0, 2.0, 4.0]


@respx.mock
def test_persistent_timeout_raises_after_budget() -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(side_effect=httpx.TimeoutException("nope"))

    with make_client() as client, pytest.raises(SocrataRequestError, match="after 3 attempt"):
        client.fetch_page(offset=0, limit=10)

    assert route.call_count == 3


# --- non-retryable failures ------------------------------------------------


@respx.mock
def test_400_raises_immediately_without_retry() -> None:
    """A bad query is our bug. Retrying it would hide the defect."""
    body = '{"errorCode":"query.soql.no-such-column","message":"No such column: nope"}'
    route = respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(400, text=body))

    with make_client() as client, pytest.raises(SocrataRequestError) as exc:
        client.fetch_page(offset=0, limit=10)

    assert route.call_count == 1
    assert exc.value.status_code == 400
    assert "query.soql.no-such-column" in str(exc.value)


@respx.mock
def test_404_raises_immediately_without_retry() -> None:
    route = respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(404, text="not found"))

    with make_client() as client, pytest.raises(SocrataRequestError):
        client.fetch_page(offset=0, limit=10)

    assert route.call_count == 1


# --- malformed responses ---------------------------------------------------


@respx.mock
def test_non_json_response_raises() -> None:
    respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(200, text="<html>nope</html>"))

    with make_client() as client, pytest.raises(SocrataResponseError, match="not valid JSON"):
        client.fetch_page(offset=0, limit=10)


@respx.mock
def test_json_object_instead_of_array_raises() -> None:
    respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(200, json={"error": True}))

    with (
        make_client() as client,
        pytest.raises(SocrataResponseError, match="Expected a JSON array"),
    ):
        client.fetch_page(offset=0, limit=10)


@respx.mock
def test_array_of_non_objects_raises() -> None:
    respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(200, json=[1, 2, 3]))

    with make_client() as client, pytest.raises(SocrataResponseError, match="expected object"):
        client.fetch_page(offset=0, limit=10)


@respx.mock
def test_missing_schema_headers_is_tolerated() -> None:
    """Header capture is best-effort; its absence must not break ingestion."""
    respx.get(TEST_RESOURCE_URL).mock(return_value=httpx.Response(200, json=make_records(2)))

    with make_client() as client:
        page = client.fetch_page(offset=0, limit=10)

    assert page.field_names == []
    assert len(page.records) == 2


# --- field discovery and $select ------------------------------------------


@respx.mock
def test_discover_fields_sends_no_order() -> None:
    """This endpoint only reveals its computed_region columns without $order."""
    route = respx.get(TEST_RESOURCE_URL).mock(
        return_value=httpx.Response(200, json=make_records(1), headers=SODA_HEADERS)
    )
    with make_client() as client:
        fields = client.discover_fields()

    assert fields == FIELD_NAMES
    params = dict(route.calls.last.request.url.params)
    assert params == {"$limit": "1"}
    assert "$order" not in params


def test_build_params_includes_select_when_given() -> None:
    params = build_params(
        limit=10,
        offset=0,
        order_column="inspection_id",
        select=["inspection_id", ":@computed_region_43wa_7qmu"],
    )
    assert params["$select"] == "inspection_id,:@computed_region_43wa_7qmu"


def test_build_params_omits_select_when_empty() -> None:
    assert "$select" not in build_params(limit=1, offset=0, order_column="id", select=[])
    assert "$select" not in build_params(limit=1, offset=0, order_column="id")


@respx.mock
def test_iter_pages_forwards_select_to_every_request() -> None:
    respx.get(TEST_RESOURCE_URL).mock(
        side_effect=[
            httpx.Response(200, json=make_records(10, start=0), headers=SODA_HEADERS),
            httpx.Response(200, json=make_records(2, start=10), headers=SODA_HEADERS),
        ]
    )
    route = respx.routes[0]

    with make_client() as client:
        list(client.iter_pages(select=["inspection_id", "results"]))

    assert all(c.request.url.params["$select"] == "inspection_id,results" for c in route.calls)
