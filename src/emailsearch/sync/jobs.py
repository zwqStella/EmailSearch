"""Persistent job registry for the Load button.

State is mirrored to the ``sync_jobs`` SQLite table so:

  1. The Recent jobs list survives a server restart.
  2. A still-running job can be cooperatively cancelled (the loader
     checks ``is_cancel_requested()`` between messages).
  3. The user can wipe the entire history with one click.

In-memory state remains the source of truth for the running job's
counters (low-latency mid-job updates) and the cancel flag (set by HTTP,
read by worker thread). On every update we mirror to SQLite. On startup
we load the most recent N jobs from SQLite into memory and force-cancel
anything that looked "running" at shutdown.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from emailsearch.config import get_settings
from emailsearch.db.connection import connect

log = logging.getLogger(__name__)

JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]

MAX_HISTORY = 100


@dataclass
class JobState:
    job_id: str
    status: JobStatus = "pending"
    start_at: int = 0  # unix epoch — date-range start (NOT job-start time)
    end_at: int = 0  # unix epoch — date-range end
    folder_ids: list[str] | None = None

    started_at: int = 0  # when run_load_job actually started executing
    finished_at: int | None = None

    count_added: int = 0
    count_skipped: int = 0
    count_errors: int = 0
    count_attachments_processed: int = 0
    last_message_id: str | None = None
    error: str | None = None

    # Non-persisted: cooperative-cancel flag flipped by HTTP handler, read by
    # the worker thread between messages.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def is_cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "folder_ids": self.folder_ids,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "count_added": self.count_added,
            "count_skipped": self.count_skipped,
            "count_errors": self.count_errors,
            "count_attachments_processed": self.count_attachments_processed,
            "last_message_id": self.last_message_id,
            "error": self.error,
            "cancel_requested": self.is_cancel_requested(),
        }


# Fields stored in the DB — `cancel_event` is intentionally excluded.
_DB_FIELDS: tuple[str, ...] = (
    "job_id",
    "status",
    "start_at",
    "end_at",
    "folder_ids",
    "started_at",
    "finished_at",
    "count_added",
    "count_skipped",
    "count_errors",
    "count_attachments_processed",
    "last_message_id",
    "error",
)


class JobRegistry:
    def __init__(self, db_path: str | None = None, max_history: int = MAX_HISTORY) -> None:
        self._jobs: dict[str, JobState] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._max_history = max_history
        self._db_path = db_path or str(get_settings().resolved_db_path)
        self._load_from_disk()

    # ------------------------------------------------------------------ disk

    def _load_from_disk(self) -> None:
        """Rehydrate recent jobs from SQLite + reconcile dangling 'running' rows."""
        try:
            with connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM sync_jobs ORDER BY created_at DESC LIMIT ?",
                    (self._max_history,),
                ).fetchall()
        except Exception as exc:  # pragma: no cover
            log.warning("job registry: could not open DB on startup: %s", exc)
            return

        ids_to_reconcile: list[str] = []
        with self._lock:
            # Insert oldest-first so `_order` reflects insertion order.
            for row in reversed(rows):
                job = self._row_to_state(row)
                # A job that was "running" at shutdown can never resume —
                # mark it cancelled with a clear reason so the UI shows
                # something honest.
                if job.status in ("pending", "running"):
                    job.status = "cancelled"
                    job.error = job.error or "Server restarted while job was running."
                    job.finished_at = job.finished_at or int(time.time())
                    ids_to_reconcile.append(job.job_id)
                self._jobs[job.job_id] = job
                self._order.append(job.job_id)
        # Push reconciled status back to disk so the row matches memory.
        for jid in ids_to_reconcile:
            self._persist(self._jobs[jid])

    def _persist(self, job: JobState) -> None:
        try:
            with connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO sync_jobs (
                        job_id, status, start_at, end_at, folder_ids,
                        started_at, finished_at,
                        count_added, count_skipped, count_errors, count_attachments_processed,
                        last_message_id, error, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        status = excluded.status,
                        started_at = excluded.started_at,
                        finished_at = excluded.finished_at,
                        count_added = excluded.count_added,
                        count_skipped = excluded.count_skipped,
                        count_errors = excluded.count_errors,
                        count_attachments_processed = excluded.count_attachments_processed,
                        last_message_id = excluded.last_message_id,
                        error = excluded.error
                    """,
                    (
                        job.job_id,
                        job.status,
                        job.start_at,
                        job.end_at,
                        json.dumps(job.folder_ids) if job.folder_ids is not None else None,
                        job.started_at,
                        job.finished_at,
                        job.count_added,
                        job.count_skipped,
                        job.count_errors,
                        job.count_attachments_processed,
                        job.last_message_id,
                        job.error,
                        int(time.time()),
                    ),
                )
        except Exception as exc:  # pragma: no cover
            log.warning("job registry: persist failed: %s", exc)

    def _delete_persisted(self, job_ids: list[str]) -> None:
        if not job_ids:
            return
        try:
            with connect(self._db_path) as conn:
                placeholders = ",".join(["?"] * len(job_ids))
                conn.execute(
                    f"DELETE FROM sync_jobs WHERE job_id IN ({placeholders})",
                    job_ids,
                )
        except Exception as exc:  # pragma: no cover
            log.warning("job registry: delete failed: %s", exc)

    @staticmethod
    def _row_to_state(row: dict) -> JobState:
        folder_ids_raw = row.get("folder_ids")
        return JobState(
            job_id=str(row["job_id"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            start_at=int(row["start_at"]),
            end_at=int(row["end_at"]),
            folder_ids=json.loads(folder_ids_raw) if folder_ids_raw else None,
            started_at=int(row.get("started_at") or 0),
            finished_at=int(row["finished_at"]) if row.get("finished_at") is not None else None,
            count_added=int(row.get("count_added") or 0),
            count_skipped=int(row.get("count_skipped") or 0),
            count_errors=int(row.get("count_errors") or 0),
            count_attachments_processed=int(row.get("count_attachments_processed") or 0),
            last_message_id=row.get("last_message_id"),
            error=row.get("error"),
        )

    # ------------------------------------------------------------- public API

    def create(
        self,
        *,
        start_at: int,
        end_at: int,
        folder_ids: list[str] | None,
    ) -> JobState:
        with self._lock:
            job_id = uuid.uuid4().hex
            job = JobState(
                job_id=job_id,
                start_at=start_at,
                end_at=end_at,
                folder_ids=folder_ids,
            )
            self._jobs[job_id] = job
            self._order.append(job_id)
            evicted = self._evict_old()
        self._persist(job)
        # DB I/O outside the lock — eviction can lag the create slightly.
        self._delete_persisted(evicted)
        return job

    def _evict_old(self) -> list[str]:
        # Caller is responsible for persisting the eviction (so DB I/O can
        # happen outside the lock). Returns the IDs that were dropped from
        # memory so they can be deleted from `sync_jobs` too — otherwise the
        # on-disk table grows unbounded.
        evicted: list[str] = []
        while len(self._order) > self._max_history:
            old_id = self._order.pop(0)
            self._jobs.pop(old_id, None)
            evicted.append(old_id)
        return evicted

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self) -> list[JobState]:
        with self._lock:
            # newest first
            return [self._jobs[j] for j in reversed(self._order) if j in self._jobs]

    def update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                if k in _DB_FIELDS:
                    setattr(job, k, v)
            job_snapshot = job
        # Persist outside the lock to keep DB I/O off the hot path.
        self._persist(job_snapshot)

    def mark_started(self, job_id: str) -> None:
        self.update(job_id, status="running", started_at=int(time.time()))

    def mark_succeeded(self, job_id: str) -> None:
        self.update(job_id, status="succeeded", finished_at=int(time.time()))

    def mark_failed(self, job_id: str, error: str) -> None:
        self.update(
            job_id,
            status="failed",
            error=error,
            finished_at=int(time.time()),
        )

    def mark_cancelled(self, job_id: str, reason: str = "Cancelled by user.") -> None:
        self.update(
            job_id,
            status="cancelled",
            error=reason,
            finished_at=int(time.time()),
        )

    def request_cancel(self, job_id: str) -> bool:
        """Set the cancel flag for a running job. Returns True if a running
        job was signaled, False otherwise."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("pending", "running"):
                return False
            job.cancel_event.set()
            return True

    def request_cancel_all_active(self) -> list[str]:
        """Signal cancel on every pending/running job. Returns the job IDs signaled.

        Used by the server-shutdown path so background worker threads exit
        cooperatively (otherwise non-daemon `asyncio.to_thread` threads keep
        the interpreter alive after Ctrl+C).
        """
        with self._lock:
            cancelled: list[str] = []
            for jid, job in self._jobs.items():
                if job.status in ("pending", "running"):
                    job.cancel_event.set()
                    cancelled.append(jid)
            return cancelled

    def clear_history(self) -> int:
        """Drop all *terminal* jobs from memory + disk. Running/pending jobs
        survive (don't kill jobs out from under the worker thread).

        Returns the number of jobs deleted.
        """
        with self._lock:
            keep_ids = {
                jid
                for jid, job in self._jobs.items()
                if job.status in ("pending", "running")
            }
            deleted = [jid for jid in self._order if jid not in keep_ids]
            self._jobs = {jid: self._jobs[jid] for jid in keep_ids if jid in self._jobs}
            self._order = [jid for jid in self._order if jid in keep_ids]

        self._delete_persisted(deleted)
        return len(deleted)


# ----- module-level singleton (lazy so test code can swap it cleanly) -------

_registry: JobRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> JobRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry


def reset_registry_for_tests() -> None:
    global _registry
    with _registry_lock:
        _registry = None
