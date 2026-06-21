"""Data-access helpers over the 3-table schema."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from typing import Any

from sqlite_vec import serialize_float32

from emailsearch.db.models import (
    Chunk,
    EmailRow,
    FtsHit,
    VecHit,
    addresses_from_json,
    addresses_to_json,
    attachments_from_json,
    attachments_to_json,
)


def _insert_chunk(conn: sqlite3.Connection, ch: Chunk) -> None:
    """INSERT one chunk into ``vec_email_chunks``. Caller owns the txn."""
    conn.execute(
        """
        INSERT INTO vec_email_chunks (
            chunk_id, embedding,
            email_id, source_type, source_name, chunk_index, chunk_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ch.chunk_id,
            serialize_float32(ch.embedding),
            ch.email_id,
            ch.source_type,
            ch.source_name,
            ch.chunk_index,
            ch.chunk_text,
        ),
    )


# ---------------------------------------------------------------------------
# emails
# ---------------------------------------------------------------------------


def email_exists(conn: sqlite3.Connection, email_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM emails WHERE id = ? LIMIT 1", (email_id,)).fetchone()
    return row is not None


def insert_email_with_chunks(
    conn: sqlite3.Connection,
    email: EmailRow,
    chunks: Iterable[Chunk],
) -> None:
    """Atomically insert the email row + its chunks (FTS triggers fire automatically)."""
    chunks = list(chunks)
    now = int(time.time())
    with conn:  # transaction
        conn.execute(
            """
            INSERT INTO emails (
                id, subject, from_address, from_name,
                to_addresses, cc_addresses,
                received_at, sent_at,
                folder_id, folder_name, conversation_id,
                body_text, body_html, summary, web_link,
                attachments, searchable_text,
                has_attachments, body_ocr_used,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                email.id,
                email.subject,
                email.from_address,
                email.from_name,
                addresses_to_json(email.to_addresses),
                addresses_to_json(email.cc_addresses),
                email.received_at,
                email.sent_at,
                email.folder_id,
                email.folder_name,
                email.conversation_id,
                email.body_text,
                email.body_html,
                email.summary,
                email.web_link,
                attachments_to_json(email.attachments),
                email.searchable_text,
                int(email.has_attachments),
                int(email.body_ocr_used),
                now,
            ),
        )
        for ch in chunks:
            _insert_chunk(conn, ch)


def clear_all_data(conn: sqlite3.Connection) -> None:
    """Wipe every indexed email + chunk and rebuild the schema from scratch.

    Re-creates tables so any schema changes (FTS tokenizer, vec0 dimension)
    take effect on the next load. Callers don't need to re-apply the schema
    afterwards — it's already applied here.
    """
    from importlib import resources

    with conn:
        # Drop triggers first so dropping FTS doesn't fire them.
        conn.execute("DROP TRIGGER IF EXISTS emails_ai")
        conn.execute("DROP TRIGGER IF EXISTS emails_ad")
        conn.execute("DROP TRIGGER IF EXISTS emails_au")
        # Dropping a vec0 / FTS5 virtual table also drops its shadow tables.
        conn.execute("DROP TABLE IF EXISTS emails_fts")
        conn.execute("DROP TABLE IF EXISTS vec_email_chunks")
        conn.execute("DROP TABLE IF EXISTS emails")
    sql = resources.files("emailsearch.db").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)


def get_email(conn: sqlite3.Connection, email_id: str) -> EmailRow | None:
    row = conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()
    if row is None:
        return None
    return _row_to_email(row)


def get_emails_by_ids(conn: sqlite3.Connection, ids: list[str]) -> dict[str, EmailRow]:
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    rows = conn.execute(f"SELECT * FROM emails WHERE id IN ({placeholders})", ids).fetchall()
    return {r["id"]: _row_to_email(r) for r in rows}


def count_emails(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM emails").fetchone()
    return int(row["n"])


def count_chunks(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM vec_email_chunks").fetchone()
    return int(row["n"])


def set_email_summary(conn: sqlite3.Connection, email_id: str, summary: str) -> bool:
    """Set the LLM-generated summary on an existing email row.

    Three coordinated writes in one transaction:

    1. UPDATE ``emails`` (FTS5 ``emails_au`` trigger re-syncs the keyword
       index automatically).
    2. DELETE prior summary chunks (so re-summarization doesn't collide
       on the deterministic chunk id).
    3. INSERT fresh summary chunks so semantic search can match the
       summary's embedding in a single KNN pass.

    Returns ``False`` when the email id doesn't exist.
    """
    # Re-derive searchable_text from current body + (new) summary +
    # attachments. Read the columns directly to avoid round-tripping a
    # full EmailRow through pydantic.
    row = conn.execute(
        "SELECT body_text, attachments FROM emails WHERE id = ?",
        (email_id,),
    ).fetchone()
    if row is None:
        return False

    # Local import to avoid a top-level cycle (build_chunks imports the
    # encoder, which would pull the embedding stack into every DB consumer).
    from emailsearch.embed.build_chunks import build_summary_chunks

    parts: list[str] = []
    if row["body_text"]:
        parts.append(row["body_text"])
    if summary:
        parts.append(summary)
    for a in attachments_from_json(row["attachments"]):
        if a.extracted_text:
            parts.append(a.extracted_text)
    searchable_text = " ".join(parts)

    # Build summary chunks BEFORE opening the txn — embedding is the
    # slow step and we don't want it holding the write lock.
    summary_chunks = build_summary_chunks(email_id, summary)

    with conn:  # transaction
        result = conn.execute(
            "UPDATE emails SET summary = ?, searchable_text = ? WHERE id = ?",
            (summary, searchable_text, email_id),
        )
        conn.execute(
            "DELETE FROM vec_email_chunks WHERE email_id = ? AND source_type = 'summary'",
            (email_id,),
        )
        for ch in summary_chunks:
            _insert_chunk(conn, ch)
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    *,
    start_at: int | None = None,
    end_at: int | None = None,
    from_address: str | None = None,
    folder_id: str | None = None,
) -> list[FtsHit]:
    """Keyword search via FTS5. Returns hits ordered by bm25 (best first).

    Column weights ``bm25(emails_fts, 5.0, 1.0, 1.0)`` bias toward subject
    hits. ``bm25`` returns lower-is-better. Optional hard filters
    (date range, sender, folder) are applied in the WHERE clause so the
    LIMIT returns the top-N matching hits.
    """
    rows = conn.execute(
        """
        SELECT
            e.id            AS email_id,
            e.subject       AS subject,
            e.from_address  AS from_address,
            e.from_name     AS from_name,
            e.received_at   AS received_at,
            snippet(emails_fts, 2, '<mark>', '</mark>', '…', 12) AS snippet,
            bm25(emails_fts, 5.0, 1.0, 1.0) AS rank
        FROM emails_fts
        JOIN emails e ON e.rowid = emails_fts.rowid
        WHERE emails_fts MATCH :q
          AND (:start_at IS NULL OR e.received_at >= :start_at)
          AND (:end_at   IS NULL OR e.received_at <  :end_at)
          AND (:from_address IS NULL OR LOWER(e.from_address) = LOWER(:from_address))
          AND (:folder_id    IS NULL OR e.folder_id = :folder_id)
        ORDER BY rank
        LIMIT :limit
        """,
        {
            "q": query,
            "start_at": start_at,
            "end_at": end_at,
            "from_address": from_address,
            "folder_id": folder_id,
            "limit": limit,
        },
    ).fetchall()
    return [FtsHit(**r) for r in rows]


def search_vec(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 20,
) -> list[VecHit]:
    """Semantic KNN over vec0. Returns chunk-level hits ordered by distance."""
    rows = conn.execute(
        """
        SELECT
            chunk_id,
            email_id,
            source_type,
            source_name,
            chunk_text,
            distance
        FROM vec_email_chunks
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (serialize_float32(query_embedding), limit),
    ).fetchall()
    return [VecHit(**r) for r in rows]


# ---------------------------------------------------------------------------
# filter facets — populates the search-page filter dropdowns
# ---------------------------------------------------------------------------


def list_filter_facets(
    conn: sqlite3.Connection,
    *,
    sender_limit: int = 500,
    folder_limit: int = 500,
) -> dict[str, list[dict[str, Any]]]:
    """Return distinct sender + folder values with per-value counts.

    Used by the frontend to populate the Sender / Folder filter dropdowns.
    Senders are deduplicated case-insensitively and returned in their
    highest-frequency casing; the display name is the most-recently-seen
    ``from_name`` for that address. Both lists are capped to prevent
    multi-MB JSON payloads on huge corpora.

    Empty / NULL values are excluded — they can't be filtered against
    meaningfully.
    """
    sender_rows = conn.execute(
        """
        SELECT
            MAX(from_address) AS address,
            MAX(from_name)    AS name,
            COUNT(*)          AS count
        FROM emails
        WHERE from_address IS NOT NULL AND from_address != ''
        GROUP BY LOWER(from_address)
        ORDER BY count DESC, address ASC
        LIMIT ?
        """,
        (sender_limit,),
    ).fetchall()
    folder_rows = conn.execute(
        """
        SELECT
            folder_id,
            MAX(folder_name) AS folder_name,
            COUNT(*)         AS count
        FROM emails
        WHERE folder_id IS NOT NULL AND folder_id != ''
        GROUP BY folder_id
        ORDER BY count DESC, folder_name ASC
        LIMIT ?
        """,
        (folder_limit,),
    ).fetchall()
    return {
        "senders": [
            {
                "address": r["address"],
                "name": r["name"],
                "count": int(r["count"]),
            }
            for r in sender_rows
        ],
        "folders": [
            {
                "folder_id": r["folder_id"],
                # Fall back to the id when the name is missing so the dropdown
                # never shows a blank label.
                "folder_name": r["folder_name"] or r["folder_id"],
                "count": int(r["count"]),
            }
            for r in folder_rows
        ],
    }


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _row_to_email(row: dict[str, Any]) -> EmailRow:
    return EmailRow(
        id=str(row["id"]),
        subject=str(row["subject"] or ""),
        from_address=str(row["from_address"] or ""),
        from_name=row["from_name"],
        to_addresses=addresses_from_json(row["to_addresses"]),
        cc_addresses=addresses_from_json(row["cc_addresses"]),
        received_at=int(row["received_at"]),
        sent_at=int(row["sent_at"]) if row["sent_at"] is not None else None,
        folder_id=row["folder_id"],
        folder_name=row["folder_name"],
        conversation_id=row["conversation_id"],
        body_text=str(row["body_text"] or ""),
        body_html=str(row["body_html"] or ""),
        web_link=row["web_link"],
        # `summary` may be absent on a legacy DB opened before
        # `_migrate_legacy_columns` runs — fall back defensively.
        summary=row.get("summary"),
        attachments=attachments_from_json(row["attachments"]),
        has_attachments=bool(row["has_attachments"]),
        body_ocr_used=bool(row["body_ocr_used"]),
    )
