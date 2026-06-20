"""Search service: combines repos + chunk-grouping into the API-facing model."""

from __future__ import annotations

import logging
import sqlite3
from typing import Literal

from pydantic import BaseModel

from emailsearch.db.models import EmailRow
from emailsearch.db.repositories import (
    get_emails_by_ids,
    search_fts,
    search_vec,
)
from emailsearch.embed.encoder import embed_query

log = logging.getLogger(__name__)

SearchMode = Literal["keyword", "semantic", "hybrid"]
MatchedIn = Literal["body", "attachment", "both"]


class SearchHit(BaseModel):
    email_id: str
    subject: str
    from_address: str
    from_name: str | None
    received_at: int
    snippet: str
    score: float                       # higher = better
    matched_in: MatchedIn
    matched_attachment_name: str | None = None
    web_link: str | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    mode: SearchMode
    query: str


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    mode: SearchMode = "hybrid",
    limit: int = 20,
) -> SearchResponse:
    query = query.strip()
    if not query:
        return SearchResponse(hits=[], mode=mode, query="")

    if mode == "keyword":
        hits = _keyword(conn, query, limit)
    elif mode == "semantic":
        hits = _semantic(conn, query, limit)
    else:
        hits = _hybrid(conn, query, limit)
    return SearchResponse(hits=hits, mode=mode, query=query)


# ---------------------------------------------------------------------------
# keyword
# ---------------------------------------------------------------------------


def _keyword(conn: sqlite3.Connection, query: str, limit: int) -> list[SearchHit]:
    fts_query = _to_fts_query(query)
    if not fts_query:
        return []
    fts_hits = search_fts(conn, fts_query, limit=limit)
    if not fts_hits:
        return []

    emails = get_emails_by_ids(conn, [h.email_id for h in fts_hits])
    out: list[SearchHit] = []
    for h in fts_hits:
        e = emails.get(h.email_id)
        if e is None:
            continue
        # Higher score = better. bm25 is "lower better"; invert.
        score = 1.0 / (1.0 + max(0.0, h.rank))
        matched_in, matched_att = _classify_keyword_match(e, query)
        out.append(
            SearchHit(
                email_id=h.email_id,
                subject=h.subject,
                from_address=h.from_address,
                from_name=h.from_name,
                received_at=h.received_at,
                snippet=h.snippet,
                score=score,
                matched_in=matched_in,
                matched_attachment_name=matched_att,
                web_link=e.web_link,
            )
        )
    return out


# ---------------------------------------------------------------------------
# semantic
# ---------------------------------------------------------------------------


def _semantic(conn: sqlite3.Connection, query: str, limit: int) -> list[SearchHit]:
    qvec = embed_query(query)
    # Over-fetch chunks so we have headroom after grouping by email.
    chunks = search_vec(conn, qvec, limit=limit * 4)
    if not chunks:
        return []

    # Group by email — keep min distance per email.
    best: dict[str, tuple[float, str, str | None, str]] = {}  # email_id -> (dist, src_type, src_name, text)
    for c in chunks:
        prev = best.get(c.email_id)
        if prev is None or c.distance < prev[0]:
            best[c.email_id] = (c.distance, c.source_type, c.source_name, c.chunk_text)

    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:limit]
    emails = get_emails_by_ids(conn, [eid for eid, _ in ranked])

    out: list[SearchHit] = []
    for eid, (dist, src_type, src_name, chunk_text) in ranked:
        e = emails.get(eid)
        if e is None:
            continue
        score = 1.0 / (1.0 + max(0.0, dist))
        matched_in: MatchedIn = "attachment" if src_type == "attachment" else "body"
        out.append(
            SearchHit(
                email_id=eid,
                subject=e.subject,
                from_address=e.from_address,
                from_name=e.from_name,
                received_at=e.received_at,
                snippet=_make_snippet(chunk_text, query),
                score=score,
                matched_in=matched_in,
                matched_attachment_name=src_name if src_type == "attachment" else None,
                web_link=e.web_link,
            )
        )
    return out


# ---------------------------------------------------------------------------
# hybrid (RRF)
# ---------------------------------------------------------------------------


_RRF_K = 60


def _hybrid(conn: sqlite3.Connection, query: str, limit: int) -> list[SearchHit]:
    # Pull more from each side so RRF has signal.
    over = max(limit * 2, 20)
    kw = _keyword(conn, query, over)
    sem = _semantic(conn, query, over)

    # Build {email_id -> SearchHit} by RRF, mutating only via model_copy so
    # there's a single update style in this function.
    by_id: dict[str, SearchHit] = {}
    rrf: dict[str, float] = {}

    for rank, hit in enumerate(kw, start=1):
        rrf[hit.email_id] = rrf.get(hit.email_id, 0.0) + 1.0 / (_RRF_K + rank)
        by_id[hit.email_id] = hit  # prefer keyword's snippet (it's highlighted)

    for rank, hit in enumerate(sem, start=1):
        rrf[hit.email_id] = rrf.get(hit.email_id, 0.0) + 1.0 / (_RRF_K + rank)
        existing = by_id.get(hit.email_id)
        if existing is None:
            by_id[hit.email_id] = hit
            continue
        # Both sides found this email; merge metadata via a single model_copy.
        updates: dict[str, object] = {}
        if existing.matched_in != hit.matched_in:
            updates["matched_in"] = "both"
        if existing.matched_attachment_name is None and hit.matched_attachment_name:
            updates["matched_attachment_name"] = hit.matched_attachment_name
        if updates:
            by_id[hit.email_id] = existing.model_copy(update=updates)

    ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [by_id[eid].model_copy(update={"score": score}) for eid, score in ranked]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_fts_query(query: str) -> str:
    """Conservative pass-through. Strip FTS5 special characters; an all-special
    query is reported as empty so the caller can skip the search rather than
    raise an FTS5 syntax error."""
    bad = {'"', "(", ")", "*"}
    return "".join(ch if ch not in bad else " " for ch in query).strip()


def _classify_keyword_match(email: EmailRow, query: str) -> tuple[MatchedIn, str | None]:
    """Single pass over attachments: returns (matched_in, attachment_name)."""
    q = query.lower()
    in_body = q in (email.body_text or "").lower() or q in (email.subject or "").lower()
    matched_att: str | None = None
    for a in email.attachments:
        if q in (a.extracted_text or "").lower():
            matched_att = a.name
            break
    if in_body and matched_att is not None:
        return "both", matched_att
    if matched_att is not None:
        return "attachment", matched_att
    return "body", None


def _make_snippet(text: str, query: str, *, window: int = 80) -> str:
    """Cheap snippet around the query term; falls back to the start of text."""
    if not text:
        return ""
    lower = text.lower()
    qlower = query.lower()
    i = lower.find(qlower)
    if i < 0:
        return (text[: window * 2] + ("…" if len(text) > window * 2 else "")).strip()
    start = max(0, i - window)
    end = min(len(text), i + len(query) + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}".strip()
