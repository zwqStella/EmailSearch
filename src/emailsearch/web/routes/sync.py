"""Sync routes: Load button + job polling + folder list + clear index."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from emailsearch.config import get_settings
from emailsearch.db.connection import connect
from emailsearch.db.repositories import clear_all_data, count_chunks, count_emails
from emailsearch.sync.jobs import get_registry
from emailsearch.sync.loader import list_folders, spawn_load_job
from emailsearch.util import to_utc

router = APIRouter(prefix="/api", tags=["sync"])


class LoadRequest(BaseModel):
    """Inputs from the Load page."""

    start: datetime = Field(description="Inclusive start of the date range (UTC).")
    end: datetime = Field(description="Exclusive end of the date range (UTC).")
    folder_ids: list[str] | None = None


class LoadResponse(BaseModel):
    job_id: str


@router.post("/sync/load", response_model=LoadResponse)
async def start_load(req: LoadRequest) -> LoadResponse:
    if req.end <= req.start:
        raise HTTPException(status_code=400, detail="end must be after start")

    start_at = int(to_utc(req.start).timestamp())
    end_at = int(to_utc(req.end).timestamp())
    job = get_registry().create(
        start_at=start_at,
        end_at=end_at,
        folder_ids=req.folder_ids,
    )
    # Daemon thread → process exit kills the worker instantly on Ctrl+C,
    # no waiting for an in-flight Outlook COM call to return.
    spawn_load_job(job.job_id)
    return LoadResponse(job_id=job.job_id)


@router.get("/sync/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = get_registry().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job.to_dict()


@router.get("/sync/jobs")
def list_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in get_registry().list_recent()]}


@router.post("/sync/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Request cooperative cancellation of a pending or running job.

    The worker checks the cancel flag between messages, so the in-flight
    message finishes cleanly before the job stops. Returns 404 if the job is
    unknown, 409 if it's already finished.
    """
    reg = get_registry()
    job = reg.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Job already {job.status}; nothing to cancel.",
        )
    reg.request_cancel(job_id)
    return {"ok": True, "job_id": job_id}


@router.delete("/sync/jobs")
def clear_job_history() -> dict:
    """Wipe all finished jobs (succeeded / failed / cancelled) from history.

    Pending or still-running jobs are preserved — cancel them first if you
    want them gone.
    """
    deleted = get_registry().clear_history()
    return {"ok": True, "deleted": deleted}


@router.get("/folders")
async def folders() -> dict:
    items = await asyncio.to_thread(list_folders)
    return {"folders": items}


@router.delete("/index")
def clear_index() -> dict:
    """Drop every indexed email + chunk and rebuild the schema from scratch.

    Use this after changing the embedding model, FTS tokenizer, or any other
    schema-affecting setting — the next Load will rebuild from your Outlook
    mailbox with the new settings.
    """
    with connect(get_settings().resolved_db_path) as conn:
        before = {"emails": count_emails(conn), "chunks": count_chunks(conn)}
        clear_all_data(conn)
        return {"ok": True, "deleted": before}
