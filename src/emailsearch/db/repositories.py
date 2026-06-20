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
                body_text, body_html, web_link,
                attachments, searchable_text,
                has_attachments, body_ocr_used,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                email.web_link,
                attachments_to_json(email.attachments),
                email.searchable_text,
                int(email.has_attachments),
                int(email.body_ocr_used),
                now,
            ),
        )
        for ch in chunks:
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


def clear_all_data(conn: sqlite3.Connection) -> None:
    """Wipe every indexed email + chunk and rebuild the schema from scratch.

    Re-creates tables so any schema changes (e.g. FTS tokenizer, vec0 dimension)
    take effect on the next load. Callers must apply the schema themselves
    after this returns — open_connection's default `connect()` helper already
    does that.
    """
    from importlib import resources

    with conn:
        # Drop triggers first so dropping FTS doesn't trigger them.
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


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[FtsHit]:
    """Keyword search via FTS5. Returns hits ordered by bm25 (best first)."""
    rows = conn.execute(
        """
        SELECT
            e.id            AS email_id,
            e.subject       AS subject,
            e.from_address  AS from_address,
            e.from_name     AS from_name,
            e.received_at   AS received_at,
            snippet(emails_fts, 2, '<mark>', '</mark>', '…', 12) AS snippet,
            bm25(emails_fts) AS rank
        FROM emails_fts
        JOIN emails e ON e.rowid = emails_fts.rowid
        WHERE emails_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
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
        attachments=attachments_from_json(row["attachments"]),
        has_attachments=bool(row["has_attachments"]),
        body_ocr_used=bool(row["body_ocr_used"]),
    )
