"""CORS is additive infrastructure for the local frontend under `frontend/`.

It changes response headers on cross-origin/OPTIONS requests only -- same-origin requests
(and every other test in this suite, which sends no ``Origin`` header) are unaffected.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_allowed_origin_preflight_gets_cors_headers(client: TestClient) -> None:
    response = client.options(
        "/v1/recommendations",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_disallowed_origin_gets_no_cors_header(client: TestClient) -> None:
    response = client.options(
        "/v1/recommendations",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers
