"""Search + email-detail routes.

The /api/search/stream endpoint is the primary search entrypoint. It runs
each leg (keyword / semantic_fts / semantic_knn) in parallel via
``asyncio.to_thread`` and emits one NDJSON line per leg as soon as that
leg finishes — no waiting for the slowest leg, no cross-leg fusion. The
browser inserts incoming hits at their correct position by score (see
``frontend/src/pages/SearchPage.tsx``).

NDJSON over SSE because ``fetch`` + ``ReadableStream`` gives us native
``AbortController`` cancellation when the user changes query mid-flight,
and one JSON object per line is trivially parseable.

Wire format (one JSON object per line, separated by ``\n``):
  {"type":"meta","query":"...","mode":"...","sources":["keyword",...]}
  {"type":"hits","source":"keyword","hits":[...],"trace":{...}}
  {"type":"hits","source":"semantic_knn","hits":[...],"trace":{...}}
  {"type":"hits","source":"semantic_fts","hits":[...],"trace":{...}}
  {"type":"done","duration_ms":1234}

The ``hits`` events arrive in leg-completion order (usually FTS legs
first, embedding leg last).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from emailsearch.config import get_settings
from emailsearch.db.connection import connect, open_connection
from emailsearch.db.repositories import (
    count_chunks,
    count_emails,
    get_email,
    list_filter_facets,
)
from emailsearch.search.service import (
    LegResult,
    SearchFilters,
    SearchMode,
    legs_for_mode,
    run_leg,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


def _ndjson(payload: dict) -> bytes:
    """Encode one NDJSON record (compact JSON + trailing newline)."""
    return (json.dumps(payload, default=str, ensure_ascii=False) + "\n").encode("utf-8")


async def _run_leg_in_thread(
    source: str,
    db_path: str,
    query: str,
    *,
    limit: int,
    filters: SearchFilters,
    debug: bool,
) -> LegResult:
    """Run a single search leg on a worker thread with its own SQLite connection.

    A dedicated connection per leg is the simplest way to get true
    parallelism out of SQLite + WAL mode for read-only queries. The
    embedding leg's slow LLM hop holds no DB lock, so FTS legs return in
    milliseconds even when augmentation takes a couple of seconds.
    """

    def _work() -> LegResult:
        conn: sqlite3.Connection = open_connection(db_path)
        try:
            return run_leg(
                source,  # type: ignore[arg-type]
                conn,
                query,
                limit=limit,
                filters=filters,
                debug=debug,
            )
        finally:
            conn.close()

    return await asyncio.to_thread(_work)


@router.get("/search/stream")
async def search_stream_endpoint(
    q: str = Query(..., description="Search query"),
    mode: Literal["keyword", "semantic", "hybrid"] = Query("hybrid"),
    limit: int = Query(20, ge=1, le=100),
    # Hard filters — all optional. Omit a param to leave that dimension unfiltered.
    start_at: int | None = Query(
        None, ge=0, description="Inclusive lower bound on received_at (epoch seconds)."
    ),
    end_at: int | None = Query(
        None, ge=0, description="Exclusive upper bound on received_at (epoch seconds)."
    ),
    from_address: str | None = Query(
        None, description="Exact (case-insensitive) match on from_address."
    ),
    folder_id: str | None = Query(
        None, description="Exact match on folder_id."
    ),
) -> StreamingResponse:
    """Stream per-leg search results as NDJSON.

    See module docstring for the wire format. The response uses
    ``application/x-ndjson``; each ``yield`` is flushed to the socket as
    soon as it lands.
    """
    settings = get_settings()
    debug = settings.debug_enabled
    filters = SearchFilters(
        start_at=start_at,
        end_at=end_at,
        from_address=from_address,
        folder_id=folder_id,
    )
    sources = legs_for_mode(mode)
    db_path = str(settings.resolved_db_path)

    log.info(
        "search/stream: query=%r mode=%s legs=%s filters_active=%s",
        q, mode, sources, filters.is_active(),
    )

    async def _generate() -> AsyncIterator[bytes]:
        started_at = time.perf_counter()

        # Meta first so the browser knows how many legs to expect before
        # any results land.
        yield _ndjson({
            "type": "meta",
            "query": q,
            "mode": mode,
            "sources": sources,
            "filters": filters.model_dump(),
            "debug_enabled": debug,
        })

        if not q.strip():
            # Empty query → no legs. Emit done immediately so the browser
            # stops its "Searching..." spinner.
            yield _ndjson({"type": "done", "duration_ms": 0})
            return

        # Kick off every leg in parallel. Each leg is wrapped in a tagged
        # coroutine so the as_completed loop can identify which source
        # failed without relying on task identity.
        async def _tagged(src: str) -> tuple[str, LegResult | BaseException]:
            try:
                result = await _run_leg_in_thread(
                    src, db_path, q,
                    limit=limit, filters=filters, debug=debug,
                )
                return src, result
            except BaseException as exc:  # noqa: BLE001 — surfaced via stream
                return src, exc

        tasks = [
            asyncio.create_task(_tagged(src), name=f"leg:{src}")
            for src in sources
        ]
        try:
            for coro in asyncio.as_completed(tasks):
                src, payload = await coro
                if isinstance(payload, BaseException):
                    # One leg crashing must not take the others down.
                    log.exception(
                        "search/stream: leg %s failed", src,
                        exc_info=(type(payload), payload, payload.__traceback__),
                    )
                    yield _ndjson({
                        "type": "error",
                        "source": src,
                        "message": f"{type(payload).__name__}: {payload}",
                    })
                    continue
                yield _ndjson({
                    "type": "hits",
                    "source": payload.source,
                    "hits": [h.model_dump() for h in payload.hits],
                    "trace": payload.trace,
                })
        finally:
            # Cancel any still-running legs if the client disconnects
            # mid-stream so we don't leak DB connections / threadpool slots.
            for t in tasks:
                if not t.done():
                    t.cancel()

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        yield _ndjson({"type": "done", "duration_ms": duration_ms})

    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        # Disable intermediate buffering — without this some reverse proxies
        # (and Starlette's gzip middleware) coalesce per-leg yields into a
        # single response chunk, defeating streaming.
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/filters")
def filters_endpoint() -> dict:
    """Distinct sender + folder values with per-value counts.

    Populates the Sender / Folder dropdowns on the search page. Cheap
    GROUP-BY (~ms) so the UI can refetch whenever a load job finishes.
    """
    with connect(get_settings().resolved_db_path) as conn:
        return list_filter_facets(conn)


@router.get("/emails/{email_id}")
def get_email_endpoint(email_id: str) -> dict:
    with connect(get_settings().resolved_db_path) as conn:
        email = get_email(conn, email_id)
        if email is None:
            raise HTTPException(status_code=404, detail="email not found")
        return email.model_dump()


@router.get("/stats")
def stats() -> dict:
    with connect(get_settings().resolved_db_path) as conn:
        return {
            "emails": count_emails(conn),
            "chunks": count_chunks(conn),
        }
