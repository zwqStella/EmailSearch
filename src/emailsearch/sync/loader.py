"""Loader: pulls messages from local Outlook (Classic) and writes to SQLite.

Idempotency: if ``email_exists(id)`` for the message's stable id
(Internet Message-Id when available, else Outlook EntryID), skip it.

The COM client is synchronous. We run the whole job inline; the caller
puts it on a daemon thread via ``spawn_load_job`` so Ctrl+C exits
immediately. Cooperative cancel (``job.is_cancel_requested()``) is
honored between messages for responsive jobs.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from emailsearch.config import get_settings
from emailsearch.db.connection import connect
from emailsearch.db.repositories import email_exists, insert_email_with_chunks
from emailsearch.embed.build_chunks import build_chunks
from emailsearch.extract.pipeline import extract_email
from emailsearch.outlook.com_client import OutlookClient, OutlookUnavailableError
from emailsearch.outlook.raw import RawMessage
from emailsearch.summarize import summarize_email
from emailsearch.sync.jobs import JobRegistry, get_registry

log = logging.getLogger(__name__)


# Anything that produces RawMessages — overridable in tests.
MessageSource = OutlookClient


def run_load_job(
    job_id: str,
    *,
    registry: JobRegistry | None = None,
    source_factory: type[MessageSource] | None = None,
) -> None:
    """Execute a load job inline. Updates the registry as it goes; never raises.

    Synchronous on purpose. Tests call this directly to assert final state;
    the HTTP layer goes through `spawn_load_job` instead.
    """
    registry = registry or get_registry()
    factory = source_factory or OutlookClient
    _run_blocking(job_id, registry, factory)


def spawn_load_job(
    job_id: str,
    *,
    registry: JobRegistry | None = None,
    source_factory: type[MessageSource] | None = None,
) -> threading.Thread:
    """Fire-and-forget version: kicks off ``run_load_job`` on a daemon
    thread (process exit kills it instantly on Ctrl+C). Job persistence
    + the on-restart reconciler keep the UI consistent.
    """
    thread = threading.Thread(
        target=run_load_job,
        args=(job_id,),
        kwargs={"registry": registry, "source_factory": source_factory},
        name=f"emailsearch-loader-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return thread


def _run_blocking(job_id: str, registry: JobRegistry, factory: type[MessageSource]) -> None:
    job = registry.get(job_id)
    if job is None:
        log.warning("run_load_job: unknown job_id=%s", job_id)
        return

    registry.mark_started(job_id)
    settings = get_settings()
    max_size_bytes = settings.max_attachment_mb * 1024 * 1024

    try:
        start_dt = datetime.fromtimestamp(job.start_at, tz=UTC)
        end_dt = datetime.fromtimestamp(job.end_at, tz=UTC)

        try:
            client = factory(max_attachment_bytes=max_size_bytes)
        except OutlookUnavailableError as exc:
            registry.mark_failed(job_id, f"Outlook unavailable: {exc}")
            return

        with connect(settings.resolved_db_path) as conn, client:
            for raw in client.iter_messages(
                start=start_dt, end=end_dt, folder_ids=job.folder_ids
            ):
                # Cooperative cancel: checked between messages so the
                # in-flight COM call completes (no half-extracted state).
                if job.is_cancel_requested():
                    registry.mark_cancelled(job_id)
                    return
                _process_one_message(conn, raw, job, registry, job_id)

        # Iter may exit because cancel was requested at the last item.
        if job.is_cancel_requested():
            registry.mark_cancelled(job_id)
        else:
            registry.mark_succeeded(job_id)
    except Exception as exc:
        log.exception("loader: job %s failed", job_id)
        registry.mark_failed(job_id, str(exc))


def _process_one_message(
    conn,
    raw: RawMessage,
    job,
    registry: JobRegistry,
    job_id: str,
) -> None:
    if not raw.id:
        return
    if email_exists(conn, raw.id):
        registry.update(
            job_id,
            count_skipped=job.count_skipped + 1,
            last_message_id=raw.id,
        )
        return

    try:
        email = extract_email(raw)
        # Best-effort LLM summarization. ``summarize_email`` returns None
        # on disabled/failed LLM — the email is indexed either way.
        email.summary = summarize_email(email)
        chunks = build_chunks(email)
        insert_email_with_chunks(conn, email, chunks)
        job.count_added += 1
        job.count_attachments_processed += len(raw.attachments)
        job.last_message_id = raw.id
    except Exception:
        log.exception("loader: failed message %s", raw.id)
        job.count_errors += 1

    registry.update(
        job_id,
        count_added=job.count_added,
        count_skipped=job.count_skipped,
        count_errors=job.count_errors,
        count_attachments_processed=job.count_attachments_processed,
        last_message_id=job.last_message_id,
    )


def list_folders(*, source_factory: type[MessageSource] | None = None) -> list[dict]:
    """Synchronous helper for the folder-picker route."""
    factory = source_factory or OutlookClient
    try:
        client = factory()
    except OutlookUnavailableError as exc:
        log.warning("list_folders: outlook unavailable: %s", exc)
        return []
    try:
        with client:
            return [fi.to_dict() for fi in client.list_folders()]
    except Exception as exc:
        log.warning("list_folders: %s", exc)
        return []
