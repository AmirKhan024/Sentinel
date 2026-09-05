"""API error types and their mapping to HTTP responses.

One exception class per failure mode the spec requires distinguishing, so a router never has to
guess a status code and a service never has to import FastAPI. ``install_exception_handlers``
does the only translation from "what went wrong" to "what HTTP status that is", in one place, so
every endpoint fails the same way for the same reason.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Base class for every error this API raises on purpose.

    Never leaks a stack trace: the message on this exception is exactly the message a caller
    receives, so nothing here may repeat internal paths, tracebacks or implementation detail.
    """

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra

    def body(self) -> dict[str, Any]:
        return {"error": self.error_code, "detail": self.detail, **self.extra}


class ArtifactNotFound(ApiError):
    """No artifact exists at all for the requested component/run."""

    status_code = 404
    error_code = "artifact_not_found"


class RowNotFound(ApiError):
    """The artifact exists, but no row matches the requested key under the given scope."""

    status_code = 404
    error_code = "row_not_found"


class AmbiguousScope(ApiError):
    """The caller's decision scope does not pick out exactly one artifact cell or row.

    Carries ``missing_scope_fields`` and/or ``candidate_values`` in ``extra`` so the response
    body tells a caller what to add, rather than only that something is wrong. The API never
    guesses "latest" or "first" on the caller's behalf -- ADR 0050.
    """

    status_code = 422
    error_code = "ambiguous_scope"


class ValidationRefused(ApiError):
    """A write payload failed the existing pydantic/parser contract for its layer.

    Wraps ``Override``/``Adjustment``/``ExecutionEvent`` parser refusals (``AdjustmentError``,
    ``ExecutionError``, pydantic ``ValidationError``) so their message reaches the caller
    unchanged rather than being replaced with a generic one.
    """

    status_code = 422
    error_code = "validation_refused"


class DuplicateKey(ApiError):
    """A write payload's natural id collides with a pending or already-committed record."""

    status_code = 409
    error_code = "duplicate_key"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Never leak internals. The unexpected case is logged by the ASGI server; the caller
        # gets a message that says only that it happened.
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "An internal error occurred."},
        )


__all__ = [
    "AmbiguousScope",
    "ApiError",
    "ArtifactNotFound",
    "DuplicateKey",
    "RowNotFound",
    "ValidationRefused",
    "install_exception_handlers",
]
