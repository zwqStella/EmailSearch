"""Ask agent HTTP endpoint — POST /api/ask/stream.

The request body carries the question; the response is an NDJSON stream
of :class:`AskEvent` objects. The event protocol is documented in
:mod:`emailsearch.ask.service`.

Threading model: :func:`ask_question` is a *synchronous* generator (the
LLM client + SQLite are both blocking). We bridge it to async by
running the generator on a worker thread and putting each yielded event
on an :class:`asyncio.Queue` that the response generator drains.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from emailsearch.ask.service import AskEvent, ask_question
from emailsearch.config import get_settings
from emailsearch.db.connection import open_connection
from emailsearch.web.routes import ndjson_line

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ask"])


# Sentinel pushed onto the bridge queue when the generator is done.
_END: object = object()


class AskRequest(BaseModel):
    """Request body for POST /api/ask/stream.

    ``mode`` defaults to ``hybrid`` for best recall when the question
    type is unknown. ``limit`` is optional — omit it to use
    ``settings.ask_retrieval_limit``. Bounded so a pathological request
    can't blow the synthesis prompt past the model's context window.
    """

    question: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    limit: int | None = Field(default=None, ge=1, le=50)


def _event_to_payload(event: AskEvent) -> dict[str, Any]:
    """Flatten ``{type, data: {...}}`` into ``{type, ...data}`` so the
    frontend's discriminated-union TS types can be flat per-event."""
    return {"type": event.type, **event.data}


@router.post("/ask/stream")
async def ask_stream_endpoint(req: AskRequest) -> StreamingResponse:
    """Stream the agent's events as NDJSON over a single POST response."""
    settings = get_settings()
    db_path = str(settings.resolved_db_path)
    limit = req.limit  # may be None — passed through to the agent

    log.info(
        "ask/stream: question=%r mode=%s limit=%s",
        req.question, req.mode, limit,
    )

    # Bridge queue between the (blocking) generator thread and the
    # (async) response writer. ``maxsize=64`` is well above any
    # realistic per-question event count.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)

    def _worker() -> None:
        """Run the sync generator, push each event onto the queue.

        ``ask_question`` never raises (every step is try/except wrapped),
        but we still belt-and-braces an outer guard so an unexpected
        crash surfaces as an error event rather than wedging the queue.
        """
        conn: sqlite3.Connection | None = None
        try:
            conn = open_connection(db_path)
            for event in ask_question(
                conn, req.question, mode=req.mode, limit=limit,
            ):
                asyncio.run_coroutine_threadsafe(
                    queue.put(event), loop,
                ).result()
        except Exception as exc:
            log.exception("ask/stream: worker crashed")
            fallback = AskEvent(
                type="error",
                data={"message": f"worker crashed: {type(exc).__name__}: {exc}"},
            )
            try:
                asyncio.run_coroutine_threadsafe(
                    queue.put(fallback), loop,
                ).result()
            except Exception:  # pragma: no cover — loop already closed
                pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # pragma: no cover — close failure non-fatal
                    pass
            asyncio.run_coroutine_threadsafe(queue.put(_END), loop).result()

    async def _generate() -> AsyncIterator[bytes]:
        started_at = time.perf_counter()
        worker_task = asyncio.create_task(asyncio.to_thread(_worker))
        try:
            while True:
                event = await queue.get()
                if event is _END:
                    break
                yield ndjson_line(_event_to_payload(event))
        finally:
            # Client disconnect → cancel the worker. The worker owns its
            # own DB connection cleanup in its ``finally`` block.
            if not worker_task.done():
                worker_task.cancel()
            log.info(
                "ask/stream: response generator exiting after %d ms",
                int((time.perf_counter() - started_at) * 1000),
            )

    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        # Same headers as /api/search/stream — disables intermediate
        # buffering so each per-event yield reaches the browser
        # immediately.
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
