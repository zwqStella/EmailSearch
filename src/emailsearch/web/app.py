"""FastAPI application factory."""

from __future__ import annotations

import asyncio
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

    In dev the React app runs on :5173 (proxied to us); in prod we run
    ``npm run build`` and FastAPI serves the static files directly.
    """
    here = Path(__file__).resolve().parent
    candidate = here.parent.parent.parent / "frontend" / "dist"
    return candidate if candidate.is_dir() else None


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Server lifecycle hooks.

    Startup: eagerly load the HuggingFace embedding model so the first
    user-facing search doesn't pay the model-load latency. Dispatched
    to a worker thread (sentence-transformers does blocking I/O) but
    awaited so FastAPI delays accepting requests until the model is
    ready. Preload failure is logged but not fatal — the lazy code
    path in ``encoder._get_embed_model`` is the safety net.

    Shutdown: signal cooperative cancel to every in-flight sync job and
    return. Workers are daemon threads so process exit kills them.
    """
    from emailsearch.embed.encoder import preload_models

    try:
        await asyncio.to_thread(preload_models)
    except Exception:
        log.exception(
            "startup: embedding preload failed; will lazy-load on first use"
        )

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

    # In dev the React frontend runs on :5173 and proxies /api → here.
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
    from emailsearch.web.routes import ask, search, status, sync

    app.include_router(status.router)
    app.include_router(sync.router)
    app.include_router(search.router)
    app.include_router(ask.router)

    # Serve built frontend from / in prod (no-op in dev — React is on :5173).
    dist = _frontend_dist_dir()
    if dist is not None:
        # Vite emits assets under /assets/...
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        index_html = dist / "index.html"

        @app.get("/")
        @app.get("/{full_path:path}")
        def spa(full_path: str = "") -> FileResponse:
            # FastAPI matches more specific routes first, so API routes win.
            return FileResponse(index_html)

    return app
