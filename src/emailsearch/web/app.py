"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from emailsearch import __version__
from emailsearch.config import get_settings

log = logging.getLogger(__name__)


def _frontend_dist_dir() -> Path | None:
    """Resolve the built frontend dir, if present.

    In dev the React app runs on :5173 (proxied to us). In prod we run
    `npm run build` and FastAPI serves the static files directly.
    """
    here = Path(__file__).resolve().parent
    candidate = here.parent.parent.parent / "frontend" / "dist"
    return candidate if candidate.is_dir() else None


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Server lifecycle hooks. On shutdown we signal cooperative cancel to
    every in-flight sync job and return **immediately** — we do NOT wait for
    the worker threads.

    Workers are daemon threads (see `sync.loader.spawn_load_job`), so the
    process exit kills them instantly. Waiting here was the old bug: an
    Outlook COM `Restrict()` call can block for tens of seconds, during
    which the worker can't check its cancel flag, and the user thinks
    Ctrl+C is broken.
    """
    yield
    from emailsearch.sync.jobs import get_registry

    try:
        cancelled = get_registry().request_cancel_all_active()
        if cancelled:
            log.info("shutdown: signaled cancel on %d active job(s)", len(cancelled))
    except Exception:
        log.exception("shutdown: failed to signal active jobs")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="EmailSearch",
        version=__version__,
        description="Local Outlook email search.",
        lifespan=_lifespan,
    )

    # In dev the React frontend runs on :5173 and proxies /api → here. CORS lets the
    # browser talk to us during npm run dev.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "version": __version__,
            "db_path": str(settings.resolved_db_path),
        }

    # Routers
    from emailsearch.web.routes import search, status, sync

    app.include_router(status.router)
    app.include_router(sync.router)
    app.include_router(search.router)

    # Serve built frontend from / in prod (no-op in dev — React is on :5173).
    dist = _frontend_dist_dir()
    if dist is not None:
        # Vite emits assets under /assets/...
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        index_html = dist / "index.html"

        @app.get("/")
        @app.get("/{full_path:path}")
        def spa(full_path: str = "") -> FileResponse:
            # Don't shadow API routes (FastAPI matches more specific routes first).
            return FileResponse(index_html)

    return app
