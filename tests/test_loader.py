"""Loader integration tests with a fake Outlook source.

The real `OutlookClient` opens COM connections; we substitute a tiny stub that
yields hand-built `RawMessage`s so the test exercises the full extract → chunk
→ DB pipeline without Windows / Outlook.

Embedding is also stubbed (the real model is heavy and tested elsewhere).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from emailsearch.config import Settings
from emailsearch.db.connection import apply_schema, open_connection
from emailsearch.db.repositories import count_chunks, count_emails, get_email
from emailsearch.embed import build_chunks as build_chunks_mod
from emailsearch.outlook.raw import RawAttachment, RawMessage
from emailsearch.sync.jobs import JobRegistry
from emailsearch.sync.loader import run_load_job

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fake source — duck-types `OutlookClient`'s context-manager + iter_messages.
# ---------------------------------------------------------------------------


class FakeSource:
    """Tiny in-memory replacement for `OutlookClient` in tests."""

    _global_messages: ClassVar[list[RawMessage]] = []

    def __init__(self, *, max_attachment_bytes: int = 25 * 1024 * 1024) -> None:
        self._max = max_attachment_bytes

    def __enter__(self) -> FakeSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def close(self) -> None:  # parity with OutlookClient
        return None

    def list_folders(self) -> list[Any]:
        return []

    def iter_messages(
        self,
        *,
        start: datetime,
        end: datetime,
        folder_ids: list[str] | None = None,
    ) -> Iterator[RawMessage]:
        for m in FakeSource._global_messages:
            if not (start.timestamp() <= m.received_at < end.timestamp()):
                continue
            if folder_ids and m.folder_id not in folder_ids:
                continue
            yield m


def _seed(messages: Iterable[RawMessage]) -> None:
    FakeSource._global_messages = list(messages)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the embedding pipeline — keeps tests fast and deterministic."""

    def fake_chunk_text(text: str) -> list[str]:
        return [text] if text.strip() else []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[(len(t) % 100) * 0.001 + i * 0.0001 for i in range(384)] for t in texts]

    monkeypatch.setattr(build_chunks_mod, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(build_chunks_mod, "embed_texts", fake_embed)


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Use a tmp DB so each test has a clean slate, and disable real OCR."""
    s = Settings(
        data_dir=tmp_path,
        db_path=tmp_path / "test.db",
        ocr_enabled=False,
        max_attachment_mb=25,
    )
    s.ensure_dirs()
    monkeypatch.setattr("emailsearch.config.get_settings", lambda: s)
    monkeypatch.setattr("emailsearch.sync.loader.get_settings", lambda: s)
    monkeypatch.setattr("emailsearch.sync.jobs.get_settings", lambda: s)
    monkeypatch.setattr("emailsearch.extract.inline_images.get_settings", lambda: s)
    monkeypatch.setattr("emailsearch.extract.extractors.get_settings", lambda: s)
    return s


def _msg(msg_id: str, subject: str = "hi", *, ts: int = 1736942400) -> RawMessage:
    return RawMessage(
        id=msg_id,
        subject=subject,
        from_address="alice@example.com",
        from_name="Alice",
        to=[("bob@example.com", "Bob")],
        received_at=ts,
        body_html=f"<p>Body of {subject}</p>",
        folder_id="inbox",
        folder_name="Inbox",
        web_link=f"outlook:{msg_id}",
    )


def _msg_with_text_attachment(msg_id: str, text: str = "hello attachment world") -> RawMessage:
    m = _msg(msg_id, subject="with-att")
    m.has_attachments = True
    m.attachments = [
        RawAttachment(
            att_id="att-1",
            name="notes.txt",
            content_type="text/plain",
            size=len(text),
            is_inline=False,
            content_bytes=text.encode("utf-8"),
        )
    ]
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_loader_inserts_emails_and_is_idempotent(isolated_settings: Settings) -> None:
    _seed([_msg("m1", "alpha"), _msg("m2", "beta")])

    registry = JobRegistry()
    job = registry.create(
        start_at=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
        end_at=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp()),
        folder_ids=None,
    )
    run_load_job(job.job_id, registry=registry, source_factory=FakeSource)

    final = registry.get(job.job_id)
    assert final.status == "succeeded"
    assert final.count_added == 2
    assert final.count_skipped == 0
    assert final.count_errors == 0

    conn = open_connection(isolated_settings.resolved_db_path)
    apply_schema(conn)
    try:
        assert count_emails(conn) == 2
        assert get_email(conn, "m1").subject == "alpha"
    finally:
        conn.close()

    # --- Re-run: should skip both ---
    job2 = registry.create(start_at=job.start_at, end_at=job.end_at, folder_ids=None)
    run_load_job(job2.job_id, registry=registry, source_factory=FakeSource)
    final2 = registry.get(job2.job_id)
    assert final2.status == "succeeded"
    assert final2.count_added == 0
    assert final2.count_skipped == 2
    assert final2.count_errors == 0


async def test_loader_handles_attachments(isolated_settings: Settings) -> None:
    _seed([_msg_with_text_attachment("m99")])

    registry = JobRegistry()
    job = registry.create(
        start_at=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
        end_at=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp()),
        folder_ids=None,
    )
    run_load_job(job.job_id, registry=registry, source_factory=FakeSource)

    final = registry.get(job.job_id)
    assert final.status == "succeeded"
    assert final.count_added == 1
    assert final.count_attachments_processed == 1

    conn = open_connection(isolated_settings.resolved_db_path)
    apply_schema(conn)
    try:
        email = get_email(conn, "m99")
        assert email is not None
        assert len(email.attachments) == 1
        att = email.attachments[0]
        assert att.name == "notes.txt"
        assert att.status == "ok"
        assert "hello attachment world" in att.extracted_text
        assert "hello attachment world" in email.searchable_text
        assert count_chunks(conn) >= 2  # body + attachment chunks
    finally:
        conn.close()


async def test_loader_marks_failed_when_outlook_unavailable(
    isolated_settings: Settings,
) -> None:
    """If the Outlook client can't be constructed, the job ends as 'failed'."""

    class DeadSource:
        def __init__(self, **_kw: Any) -> None:
            from emailsearch.outlook.com_client import OutlookUnavailableError

            raise OutlookUnavailableError("simulated: no Outlook")

    registry = JobRegistry()
    job = registry.create(
        start_at=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
        end_at=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp()),
        folder_ids=None,
    )
    run_load_job(job.job_id, registry=registry, source_factory=DeadSource)

    final = registry.get(job.job_id)
    assert final.status == "failed"
    assert "Outlook" in (final.error or "")


async def test_loader_filters_by_folder(isolated_settings: Settings) -> None:
    other = _msg("m-other", "other")
    other.folder_id = "archive"
    _seed([_msg("m-inbox", "inbox-msg"), other])

    registry = JobRegistry()
    job = registry.create(
        start_at=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
        end_at=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp()),
        folder_ids=["inbox"],
    )
    run_load_job(job.job_id, registry=registry, source_factory=FakeSource)

    final = registry.get(job.job_id)
    assert final.status == "succeeded"
    assert final.count_added == 1

    conn = open_connection(isolated_settings.resolved_db_path)
    apply_schema(conn)
    try:
        assert get_email(conn, "m-inbox") is not None
        assert get_email(conn, "m-other") is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Persistence + cancellation
# ---------------------------------------------------------------------------


async def test_registry_persists_and_reloads_jobs(isolated_settings: Settings) -> None:
    """Jobs survive a registry rebuild (proxy for a server restart)."""
    _seed([_msg("m1", "alpha")])
    reg = JobRegistry()
    job = reg.create(
        start_at=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
        end_at=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp()),
        folder_ids=None,
    )
    run_load_job(job.job_id, registry=reg, source_factory=FakeSource)

    # Simulate restart: brand-new registry against the same DB.
    reg2 = JobRegistry()
    rehydrated = reg2.get(job.job_id)
    assert rehydrated is not None
    assert rehydrated.status == "succeeded"
    assert rehydrated.count_added == 1


async def test_registry_reconciles_dangling_running_jobs(
    isolated_settings: Settings,
) -> None:
    """A row that says 'running' after restart can't actually be running —
    the worker thread is gone — so the registry marks it cancelled with a
    'server restarted' reason.

    Defined as a plain (sync) function despite the file-level asyncio mark
    because there's nothing to await; pytest-asyncio tolerates this.
    """
    reg = JobRegistry()
    job = reg.create(start_at=0, end_at=1, folder_ids=None)
    reg.mark_started(job.job_id)
    assert reg.get(job.job_id).status == "running"

    reg2 = JobRegistry()
    final = reg2.get(job.job_id)
    assert final is not None
    assert final.status == "cancelled"
    assert "restarted" in (final.error or "").lower()


async def test_loader_honors_cancel_request(isolated_settings: Settings) -> None:
    """Setting cancel before the loop starts ends the job as cancelled with
    zero work done. Idempotency is preserved (no partial inserts)."""
    _seed([_msg("m1"), _msg("m2"), _msg("m3")])
    reg = JobRegistry()
    job = reg.create(
        start_at=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()),
        end_at=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp()),
        folder_ids=None,
    )
    reg.request_cancel(job.job_id)

    run_load_job(job.job_id, registry=reg, source_factory=FakeSource)

    final = reg.get(job.job_id)
    assert final.status == "cancelled"
    assert final.count_added == 0


async def test_clear_history_preserves_running_jobs(isolated_settings: Settings) -> None:
    """clear_history() drops terminal jobs but leaves an in-flight job alone."""
    reg = JobRegistry()
    done = reg.create(start_at=0, end_at=1, folder_ids=None)
    reg.mark_succeeded(done.job_id)
    failed = reg.create(start_at=0, end_at=1, folder_ids=None)
    reg.mark_failed(failed.job_id, "x")
    running = reg.create(start_at=0, end_at=1, folder_ids=None)
    reg.mark_started(running.job_id)

    deleted = reg.clear_history()
    assert deleted == 2
    assert reg.get(done.job_id) is None
    assert reg.get(failed.job_id) is None
    assert reg.get(running.job_id) is not None
    # And the running job survives a registry rebuild too.
    reg2 = JobRegistry()
    # Note: rehydration reconciles it to 'cancelled' (no live worker), which
    # is the correct behavior we tested above.
    assert reg2.get(running.job_id) is not None


async def test_request_cancel_all_active(isolated_settings: Settings) -> None:
    """Signaling cancel-all should set the flag on every pending/running job."""
    reg = JobRegistry()
    a = reg.create(start_at=0, end_at=1, folder_ids=None)
    b = reg.create(start_at=0, end_at=1, folder_ids=None)
    reg.mark_started(b.job_id)
    done = reg.create(start_at=0, end_at=1, folder_ids=None)
    reg.mark_succeeded(done.job_id)

    cancelled = reg.request_cancel_all_active()
    assert set(cancelled) == {a.job_id, b.job_id}
    assert reg.get(a.job_id).is_cancel_requested()
    assert reg.get(b.job_id).is_cancel_requested()
    assert not reg.get(done.job_id).is_cancel_requested()


async def test_lifespan_shutdown_cancels_active_jobs(
    isolated_settings: Settings,
) -> None:
    """Server lifespan shutdown should signal cancel + mark stuck jobs
    cancelled-on-disk via the eventual rehydration reconciler."""
    from fastapi import FastAPI

    from emailsearch.web.app import _lifespan

    reg = JobRegistry()
    job = reg.create(start_at=0, end_at=1, folder_ids=None)
    reg.mark_started(job.job_id)

    # Patch the singleton accessor so the lifespan sees our test registry.
    import emailsearch.sync.jobs as jobs_mod

    original = jobs_mod._registry
    jobs_mod._registry = reg
    try:
        async with _lifespan(FastAPI()):
            pass  # exit the context → triggers shutdown branch
    finally:
        jobs_mod._registry = original

    # The job's cancel flag should have been set during shutdown.
    assert reg.get(job.job_id).is_cancel_requested()
