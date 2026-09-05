"""The Sentinel API application factory.

``create_app`` wires routers and error handling; it holds no state of its own. ``Settings`` is
resolved per request through ``sentinel.api.deps.get_settings``, which tests override with
``app.dependency_overrides`` to point at a ``tmp_path`` fixture -- there is no module-level
``Settings`` instance here to make that override race against.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sentinel import __version__
from sentinel.api.errors import install_exception_handlers
from sentinel.api.routers import (
    adjustments,
    establishments,
    execution,
    explanations,
    meta,
    overrides,
    plan_review,
    recommendations,
    review,
    schedule,
    staged_requests,
)
from sentinel.config import load_settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentinel API",
        version=__version__,
        description=(
            "A validated read/write HTTP boundary over Components 1-16 and 20-21's artifacts. "
            "Computes nothing; writes are staged, never applied. See ADR 0048 and ADR 0049."
        ),
    )
    # CORS is for the local product-testing frontend under `frontend/` only -- it changes
    # response headers on cross-origin/OPTIONS requests and nothing else; same-origin and
    # no-Origin-header requests (like TestClient's) are unaffected.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=load_settings().api_cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    install_exception_handlers(app)
    for router_module in (
        meta.router,
        recommendations.router,
        overrides.router,
        schedule.router,
        adjustments.router,
        execution.router,
        review.router,
        explanations.router,
        establishments.router,
        plan_review.router,
        staged_requests.router,
    ):
        app.include_router(router_module)
    return app


__all__ = ["create_app"]
