"""SQLite connection helpers with sqlite-vec extension loaded."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

import sqlite_vec

from emailsearch.config import get_settings


def _row_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def open_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded and pragmas applied.

    Caller is responsible for closing. Use `connect()` context manager when possible.
    """
    if db_path is None:
        db_path = get_settings().resolved_db_path
    db_path = str(db_path)

    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = _row_factory
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Apply pragmas (cheap if already set).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply schema.sql idempotently (CREATE IF NOT EXISTS throughout)."""
    sql = resources.files("emailsearch.db").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager that opens, applies schema, yields, and closes."""
    conn = open_connection(db_path)
    try:
        apply_schema(conn)
        yield conn
    finally:
        conn.close()
