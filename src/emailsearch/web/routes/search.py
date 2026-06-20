"""Search + email-detail routes."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from emailsearch.config import get_settings
from emailsearch.db.connection import connect
from emailsearch.db.repositories import count_chunks, count_emails, get_email
from emailsearch.search.service import SearchResponse, search

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: str = Query(..., description="Search query"),
    mode: Literal["keyword", "semantic", "hybrid"] = Query("hybrid"),
    limit: int = Query(20, ge=1, le=100),
) -> SearchResponse:
    with connect(get_settings().resolved_db_path) as conn:
        return search(conn, q, mode=mode, limit=limit)


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
