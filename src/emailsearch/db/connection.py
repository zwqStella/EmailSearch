"""SQLite connection helpers with sqlite-vec extension loaded."""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

import sqlite_vec

from emailsearch.config import get_settings

log = logging.getLogger(__name__)


def _row_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


# Schema is idempotent (CREATE IF NOT EXISTS throughout), but re-running it
# on every connect() still parses ~10 statements and probes the system
# catalog — wasted work for the read-heavy search path that opens many
# short-lived connections. Track which DB paths have had schema applied
# this process so subsequent connects skip it.
#
# ``:memory:`` is intentionally NOT cached — each ``:memory:`` connection
# is a fresh, distinct database in SQLite, so caching that name would
# leave new in-memory DBs unschema'd. Tests already call apply_schema()
# explicitly for their in-memory connections.
_schema_applied_paths: set[str] = set()
_schema_lock = threading.Lock()


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
    #
    # Durability / concurrency:
    #   journal_mode=WAL   — readers don't block writers; required for the
    #                        per-leg parallel search threads.
    #   synchronous=NORMAL — WAL-safe; fsync only on checkpoint, not per txn.
    #   busy_timeout=5000  — 5s wait on lock contention instead of immediate
    #                        SQLITE_BUSY. Search has up to 3 concurrent
    #                        readers and a loader writer can land mid-search.
    #   foreign_keys=ON    — schema cosmetic; we don't actually have FK relations.
    #
    # Performance:
    #   cache_size=-65536  — 64 MB per-connection page cache (negative = KB).
    #                        FTS5 + vec0 reads are page-heavy; 64 MB easily
    #                        holds the hot working set on a typical mailbox.
    #   mmap_size=1 GiB    — memory-map the DB so SQLite reads pages via the
    #                        OS page cache without per-read syscall overhead.
    #   temp_store=MEMORY  — keep transient sort / materialized subquery state
    #                        in RAM instead of spilling to a temp file.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA cache_size = -65536")
    conn.execute("PRAGMA mmap_size = 1073741824")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply schema.sql idempotently (CREATE IF NOT EXISTS throughout)."""
    sql = resources.files("emailsearch.db").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    _migrate_legacy_columns(conn)


def _migrate_legacy_columns(conn: sqlite3.Connection) -> None:
    """Add columns present in the latest schema but missing from older DBs.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op when the table already exists,
    so new columns added to schema.sql don't reach pre-existing databases.
    We backfill them here with ``ALTER TABLE ... ADD COLUMN`` so users don't
    have to clear-and-resync just to pick up a new column.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(emails)")}
    if "summary" not in cols:
        conn.execute("ALTER TABLE emails ADD COLUMN summary TEXT")


def _ensure_schema_once(conn: sqlite3.Connection, db_path: str) -> None:
    """Apply schema for ``db_path`` the first time we see it this process.

    Double-checked under a lock so concurrent first-connects race-free.
    ``:memory:`` is excluded — every in-memory connection is its own DB,
    so caching the schema-applied flag would leave later ones empty.
    """
    if db_path == ":memory:":
        apply_schema(conn)
        return
    if db_path in _schema_applied_paths:
        return
    with _schema_lock:
        if db_path in _schema_applied_paths:
            return
        apply_schema(conn)
        _schema_applied_paths.add(db_path)


def reset_schema_cache() -> None:
    """Forget which DB paths have been schema'd this process.

    Used by tests that swap DB paths between cases, and by ``clear_all_data``
    after it tears down and rebuilds the tables.
    """
    with _schema_lock:
        _schema_applied_paths.clear()


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager that opens, applies schema, yields, and closes.

    Schema application is memoized per process: the first call for a given
    ``db_path`` runs it; subsequent calls skip the work. Cuts ~5-10 ms of
    catalog churn off every short-lived web-route connection.
    """
    if db_path is None:
        db_path = get_settings().resolved_db_path
    db_path_str = str(db_path)
    conn = open_connection(db_path_str)
    try:
        _ensure_schema_once(conn, db_path_str)
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Connection pool for the hot read path.
#
# Each ``/api/search/stream`` request opens 3 fresh connections (one per
# leg), and each open pays for: a sqlite3.connect syscall, sqlite-vec
# extension load (dlopen + symbol resolution), and 7 PRAGMA round-trips.
# That's ~5-15 ms x 3 legs of pure overhead before any query work runs.
#
# A bounded per-path pool reuses warmed-up connections across requests.
# On checkout exhaustion we open a transient connection rather than
# blocking — a sudden burst of users should not queue waiting for a slot.
# On checkin, transients are closed if the pool is already full.
# ---------------------------------------------------------------------------

# Sized for the realistic concurrent-reader count: 3 search legs +
# 1 ask agent + 1 frontend detail fetch + headroom for the next user.
_POOL_MAX_PER_PATH = 8

_pool_lock = threading.Lock()
_pools: dict[str, queue.Queue[sqlite3.Connection]] = {}


def _get_pool(db_path: str) -> queue.Queue[sqlite3.Connection]:
    """Return the LIFO pool for ``db_path``, lazily creating it."""
    pool = _pools.get(db_path)
    if pool is not None:
        return pool
    with _pool_lock:
        pool = _pools.get(db_path)
        if pool is None:
            pool = queue.LifoQueue(maxsize=_POOL_MAX_PER_PATH)
            _pools[db_path] = pool
        return pool


@contextmanager
def pooled_connection(
    db_path: Path | str | None = None,
) -> Iterator[sqlite3.Connection]:
    """Check out a pooled connection; return it (or close it) on exit.

    Best for hot, short-lived reads where the per-open overhead is
    measurable vs. the query work itself (the per-leg search threads in
    particular).

    Behavior:
      - On checkout: pop a pooled connection if one is available, else
        open a transient one. Never blocks the caller.
      - On normal exit: try to put the connection back. If the pool is
        full (max already in flight), the surplus is closed.
      - On exception: the connection is closed (it may be in a half-
        committed state) and the exception re-raises. Pools should
        only hold connections in a known-good state.
    """
    if db_path is None:
        db_path = get_settings().resolved_db_path
    db_path_str = str(db_path)
    pool = _get_pool(db_path_str)

    try:
        conn = pool.get_nowait()
    except queue.Empty:
        conn = open_connection(db_path_str)
        _ensure_schema_once(conn, db_path_str)

    error: BaseException | None = None
    try:
        yield conn
    except BaseException as exc:
        error = exc
        raise
    finally:
        if error is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover — close failure non-fatal
                log.exception("pooled_connection: close after error failed")
        else:
            try:
                pool.put_nowait(conn)
            except queue.Full:
                # Pool already saturated — drop this one instead of growing
                # unbounded. Common after a burst that opened transients.
                try:
                    conn.close()
                except Exception:  # pragma: no cover
                    log.exception("pooled_connection: close on full pool failed")


def clear_connection_pools() -> None:
    """Close every pooled connection and forget every pool.

    Used by tests to reset state between cases that use different DB
    paths. Production code can call this on shutdown if it wants to
    release file handles eagerly (process exit handles it otherwise).
    """
    with _pool_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        while True:
            try:
                conn = pool.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:  # pragma: no cover
                log.exception("clear_connection_pools: close failed")
