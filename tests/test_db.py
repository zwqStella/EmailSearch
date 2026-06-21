"""Unit tests for the DB schema + repositories."""

from __future__ import annotations

import time

import pytest

from emailsearch.db.connection import apply_schema, open_connection
from emailsearch.db.models import (
    AttachmentRecord,
    Chunk,
    EmailAddress,
    EmailRow,
)
from emailsearch.db.repositories import (
    clear_all_data,
    count_chunks,
    count_emails,
    email_exists,
    get_email,
    get_emails_by_ids,
    insert_email_with_chunks,
    list_filter_facets,
    search_fts,
    search_vec,
    set_email_summary,
)


@pytest.fixture()
def conn():
    c = open_connection(":memory:")
    apply_schema(c)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def _stub_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy chunker + embedder used by `set_email_summary`.

    `set_email_summary` now chunks + embeds the summary via
    `emailsearch.embed.build_chunks.build_summary_chunks`. Real embedding
    would load the 80 MB sentence-transformer model into every test process
    — we'd rather use a tiny deterministic stub. Patches go on the
    `build_chunks` module's globals (not `encoder`) because that module
    captured the names at import time.
    """
    from emailsearch.embed import build_chunks as build_chunks_mod

    def fake_chunk_text(text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [text.strip()]

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [
            [(len(t) % 100) * 0.001 + (i * 0.0001) for i in range(384)]
            for t in texts
        ]

    monkeypatch.setattr(build_chunks_mod, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(build_chunks_mod, "embed_texts", fake_embed)


def _make_email(
    email_id: str = "msg-1",
    subject: str = "Project Atlas weekly sync",
    body: str = "Discussion of Q3 milestones and a draft of the new roadmap.",
    attachments: list[AttachmentRecord] | None = None,
    *,
    summary: str | None = None,
) -> EmailRow:
    return EmailRow(
        id=email_id,
        subject=subject,
        from_address="alice@example.com",
        from_name="Alice",
        to_addresses=[EmailAddress(address="bob@example.com", name="Bob")],
        received_at=int(time.time()),
        body_text=body,
        body_html=f"<p>{body}</p>",
        web_link=f"https://outlook.office.com/mail/{email_id}",
        summary=summary,
        attachments=attachments or [],
        has_attachments=bool(attachments),
    )


def _make_chunk(
    email_id: str,
    idx: int,
    text: str,
    *,
    source_type: str = "body",
    source_name: str | None = None,
    dim: int = 384,
) -> Chunk:
    # deterministic fake embedding so distance ordering is meaningful
    seed = (idx + 1) * 0.01
    embedding = [seed + (i * 0.0001) for i in range(dim)]
    return Chunk(
        chunk_id=f"{email_id}:{source_type}:{idx}"
        + (f":{source_name}" if source_name else ""),
        email_id=email_id,
        source_type=source_type,  # type: ignore[arg-type]
        source_name=source_name,
        chunk_index=idx,
        chunk_text=text,
        embedding=embedding,
    )


def test_idempotency_email_exists(conn) -> None:
    assert not email_exists(conn, "msg-1")
    insert_email_with_chunks(conn, _make_email("msg-1"), [])
    assert email_exists(conn, "msg-1")


def test_insert_with_chunks_and_fts_search(conn) -> None:
    email = _make_email(
        "msg-1",
        subject="Roadmap review",
        body="Discussion of Q3 milestones.",
    )
    chunks = [_make_chunk("msg-1", 0, email.body_text)]
    insert_email_with_chunks(conn, email, chunks)

    # FTS5 finds the email by a body word
    hits = search_fts(conn, "milestones", limit=10)
    assert len(hits) == 1
    assert hits[0].email_id == "msg-1"
    assert "milestones" in hits[0].snippet.lower()


def test_attachment_text_indexed_in_fts(conn) -> None:
    """Attachment extracted_text is concatenated into searchable_text and visible to FTS."""
    att = AttachmentRecord(
        att_id="att-1",
        name="quarterly-report.pdf",
        content_type="application/pdf",
        size=1024,
        extracted_text="Revenue grew 17% with strong cloud bookings.",
        status="ok",
    )
    email = _make_email("msg-2", subject="Q3 report", body="See attached.", attachments=[att])
    insert_email_with_chunks(conn, email, [])

    hits = search_fts(conn, "bookings", limit=10)
    assert len(hits) == 1
    assert hits[0].email_id == "msg-2"


def test_vec_search_returns_aux_columns(conn) -> None:
    email = _make_email("msg-3", body="Some body text")
    chunks = [
        _make_chunk("msg-3", 0, "alpha chunk"),
        _make_chunk("msg-3", 1, "beta chunk", source_type="attachment", source_name="report.pdf"),
    ]
    insert_email_with_chunks(conn, email, chunks)

    # Query close to chunk 0 (seed=0.01)
    qvec = [0.01 + (i * 0.0001) for i in range(384)]
    hits = search_vec(conn, qvec, limit=2)
    assert len(hits) == 2
    assert hits[0].chunk_id.endswith("body:0")
    assert hits[0].source_type == "body"
    # The attachment hit carries its source_name back via aux column
    att_hit = next(h for h in hits if h.source_type == "attachment")
    assert att_hit.source_name == "report.pdf"


def test_round_trip_email(conn) -> None:
    att = AttachmentRecord(
        att_id="att-9",
        name="notes.txt",
        content_type="text/plain",
        size=42,
        extracted_text="hello",
        status="ok",
    )
    email = _make_email("msg-4", attachments=[att])
    insert_email_with_chunks(conn, email, [])

    got = get_email(conn, "msg-4")
    assert got is not None
    assert got.subject == email.subject
    assert len(got.attachments) == 1
    assert got.attachments[0].name == "notes.txt"

    by_ids = get_emails_by_ids(conn, ["msg-4", "missing"])
    assert set(by_ids.keys()) == {"msg-4"}


def test_counts(conn) -> None:
    insert_email_with_chunks(conn, _make_email("a"), [_make_chunk("a", 0, "x")])
    insert_email_with_chunks(conn, _make_email("b"), [_make_chunk("b", 0, "y"), _make_chunk("b", 1, "z")])
    assert count_emails(conn) == 2
    assert count_chunks(conn) == 3


def test_clear_all_data_wipes_index(conn) -> None:
    insert_email_with_chunks(conn, _make_email("a"), [_make_chunk("a", 0, "x")])
    insert_email_with_chunks(conn, _make_email("b"), [_make_chunk("b", 0, "y")])
    assert count_emails(conn) == 2
    assert count_chunks(conn) == 2

    clear_all_data(conn)

    # Tables exist and are empty after rebuild
    assert count_emails(conn) == 0
    assert count_chunks(conn) == 0
    # FTS still wired up — insert + search should work again
    insert_email_with_chunks(conn, _make_email("c", body="reborn"), [])
    assert [h.email_id for h in search_fts(conn, "reborn")] == ["c"]


def test_fts_trigram_matches_cjk_substring(conn) -> None:
    """Trigram tokenizer makes CJK substring search work for queries >= 3 chars.

    Shorter queries (like 2-char '\u4e2a\u7a0e') won't form a trigram and won't match — they
    must rely on semantic search instead. That's a property of FTS5 trigram, not a bug.
    """
    insert_email_with_chunks(
        conn,
        _make_email(
            "tax",
            subject="2025\u5e74\u4e2a\u4eba\u6240\u5f97\u7a0e\u6c47\u7b97\u6e05\u7f34",
            body="annual personal income tax filing",
        ),
        [],
    )
    # 3-char substring inside the subject -> matches via trigrams
    assert [h.email_id for h in search_fts(conn, "\u6240\u5f97\u7a0e")] == ["tax"]
    assert [h.email_id for h in search_fts(conn, "\u4e2a\u4eba\u6240")] == ["tax"]
    # 2-char queries are below the trigram threshold -> no FTS hit (semantic covers these)
    assert search_fts(conn, "\u4e2a\u7a0e") == []


def test_summary_roundtrips_through_db(conn) -> None:
    """The LLM-generated `summary` column persists and reads back via get_email."""
    email = _make_email(
        "with-summary",
        subject="Q3 plan",
        body="long detailed body",
        summary="Q3 budget approved; ship by Aug 15.",
    )
    insert_email_with_chunks(conn, email, [])
    got = get_email(conn, "with-summary")
    assert got is not None
    assert got.summary == "Q3 budget approved; ship by Aug 15."


def test_summary_is_indexed_in_fts(conn) -> None:
    """Summary text is part of `searchable_text`, so a keyword that appears
    only in the summary (not the body) still matches via FTS."""
    email = _make_email(
        "fts-summary",
        subject="status",
        body="see details below",  # 'reorg' is NOT in body
        summary="Team reorg announced for next quarter.",
    )
    insert_email_with_chunks(conn, email, [])
    hits = search_fts(conn, "reorg", limit=10)
    assert [h.email_id for h in hits] == ["fts-summary"]


def test_set_email_summary_updates_row_and_fts(conn) -> None:
    """`set_email_summary` is the backfill primitive — it writes the summary
    AND re-derives `searchable_text` so a keyword in the summary (but not
    in the body) becomes findable via FTS without a re-ingest."""
    email = _make_email(
        "to-backfill",
        subject="status",
        body="see details below",  # 'reorg' is NOT in body
        summary=None,
    )
    insert_email_with_chunks(conn, email, [])
    # Sanity: 'reorg' isn't findable before the backfill.
    assert search_fts(conn, "reorg", limit=5) == []

    ok = set_email_summary(
        conn, "to-backfill", "Team reorg announced for next quarter."
    )
    assert ok is True

    # Row picked up the summary.
    got = get_email(conn, "to-backfill")
    assert got is not None
    assert got.summary == "Team reorg announced for next quarter."

    # FTS re-synced via the existing UPDATE trigger — 'reorg' now matches.
    hits = search_fts(conn, "reorg", limit=5)
    assert [h.email_id for h in hits] == ["to-backfill"]


def test_set_email_summary_inserts_summary_chunk(conn) -> None:
    """Beyond updating the row + FTS, set_email_summary now also embeds the
    summary into `vec_email_chunks` with `source_type='summary'` so the
    semantic search can match against it directly."""
    email = _make_email("sc-1", subject="status", body="body text", summary=None)
    insert_email_with_chunks(conn, email, [])

    # Before backfill: no summary chunk exists yet.
    pre = conn.execute(
        "SELECT COUNT(*) AS n FROM vec_email_chunks "
        "WHERE email_id = ? AND source_type = 'summary'",
        ("sc-1",),
    ).fetchone()["n"]
    assert pre == 0

    ok = set_email_summary(conn, "sc-1", "Team reorg announced for next quarter.")
    assert ok is True

    rows = conn.execute(
        "SELECT chunk_id, source_type, source_name, chunk_text "
        "FROM vec_email_chunks WHERE email_id = ? AND source_type = 'summary'",
        ("sc-1",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "sc-1::summary::0"
    assert rows[0]["source_name"] is None
    assert rows[0]["chunk_text"] == "Team reorg announced for next quarter."


def test_set_email_summary_replaces_existing_summary_chunk(conn) -> None:
    """Re-running `set_email_summary` with a different text must REPLACE the
    summary chunk (not append). The chunk-id is deterministic, so without
    the DELETE-before-INSERT pass a re-summarization would either collide
    on the primary key or produce stale chunks alongside the new one."""
    email = _make_email("sc-2", subject="status", body="body text", summary=None)
    insert_email_with_chunks(conn, email, [])

    assert set_email_summary(conn, "sc-2", "Original summary text.") is True
    assert set_email_summary(conn, "sc-2", "Updated summary text.") is True

    rows = conn.execute(
        "SELECT chunk_text FROM vec_email_chunks "
        "WHERE email_id = ? AND source_type = 'summary'",
        ("sc-2",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "Updated summary text."


def test_set_email_summary_empty_purges_summary_chunks(conn) -> None:
    """Explicitly clearing the summary (empty string) drops any existing
    summary chunks — prevents stale topical vectors from outliving the
    text the user / operator intentionally cleared."""
    email = _make_email("sc-3", subject="status", body="body text", summary=None)
    insert_email_with_chunks(conn, email, [])
    assert set_email_summary(conn, "sc-3", "Initial summary.") is True
    # Sanity: chunk exists.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM vec_email_chunks WHERE email_id = ? AND source_type = 'summary'",
        ("sc-3",),
    ).fetchone()["n"] == 1

    # Clearing the summary purges the chunk.
    assert set_email_summary(conn, "sc-3", "") is True
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM vec_email_chunks WHERE email_id = ? AND source_type = 'summary'",
        ("sc-3",),
    ).fetchone()["n"] == 0


def test_set_email_summary_returns_false_for_missing_email(conn) -> None:
    """An unknown email id is a soft no-op: caller can log + skip rather
    than handling an exception."""
    assert set_email_summary(conn, "does-not-exist", "anything") is False


def test_migration_adds_summary_column_to_legacy_db(conn) -> None:
    """A DB created before the `summary` column existed gets back-filled by
    `apply_schema` so users don't have to clear-and-resync. The trigger here
    is a connection that already has the `emails` table but is missing the
    new column — we simulate it by dropping the column then re-applying."""
    from emailsearch.db.connection import _migrate_legacy_columns, apply_schema

    # Drop the summary column to simulate a pre-migration DB. SQLite supports
    # DROP COLUMN as of 3.35; on older runtimes this would need a rebuild,
    # but pytest envs ship modern SQLite.
    conn.execute("ALTER TABLE emails DROP COLUMN summary")
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(emails)")}
    assert "summary" not in cols

    # apply_schema (CREATE IF NOT EXISTS) on its own wouldn't add the column —
    # the migration helper is what brings it back.
    apply_schema(conn)
    cols_after = {row["name"] for row in conn.execute("PRAGMA table_info(emails)")}
    assert "summary" in cols_after

    # Idempotent: running again is a no-op.
    _migrate_legacy_columns(conn)
    cols_again = {row["name"] for row in conn.execute("PRAGMA table_info(emails)")}
    assert "summary" in cols_again


# ---------------------------------------------------------------------------
# list_filter_facets — powers the search-page filter dropdowns
# ---------------------------------------------------------------------------


def _seed_facet_email(
    conn,
    eid: str,
    *,
    from_address: str,
    from_name: str | None,
    folder_id: str | None,
    folder_name: str | None,
) -> None:
    email = EmailRow(
        id=eid,
        subject="x",
        from_address=from_address,
        from_name=from_name,
        received_at=int(time.time()),
        folder_id=folder_id,
        folder_name=folder_name,
        body_text="body",
        body_html="<p>body</p>",
    )
    insert_email_with_chunks(conn, email, [])


def test_facets_groups_senders_case_insensitively(conn) -> None:
    """Alice@x vs alice@x are the same sender; the facet returns ONE row
    with count=2 (so the dropdown shows one entry, not two near-duplicates)."""
    _seed_facet_email(conn, "a", from_address="Alice@example.com", from_name="Alice", folder_id=None, folder_name=None)
    _seed_facet_email(conn, "b", from_address="alice@example.com", from_name="Alice", folder_id=None, folder_name=None)
    _seed_facet_email(conn, "c", from_address="bob@example.com", from_name="Bob", folder_id=None, folder_name=None)

    facets = list_filter_facets(conn)
    senders = facets["senders"]
    by_lower = {s["address"].lower(): s for s in senders}
    assert by_lower["alice@example.com"]["count"] == 2
    assert by_lower["bob@example.com"]["count"] == 1


def test_facets_orders_senders_by_count_desc(conn) -> None:
    """Highest-volume sender first — that's the most useful default for the dropdown."""
    for i in range(3):
        _seed_facet_email(
            conn, f"a{i}",
            from_address="alice@example.com", from_name="Alice",
            folder_id=None, folder_name=None,
        )
    _seed_facet_email(
        conn, "b", from_address="bob@example.com", from_name="Bob",
        folder_id=None, folder_name=None,
    )

    senders = list_filter_facets(conn)["senders"]
    assert [s["address"].lower() for s in senders][:2] == ["alice@example.com", "bob@example.com"]


def test_facets_folder_falls_back_to_id_when_name_missing(conn) -> None:
    """A folder with no display name still shows up — labeled by its id so
    the dropdown isn't blank."""
    _seed_facet_email(
        conn, "x", from_address="a@b.com", from_name=None,
        folder_id="opaque-folder-id", folder_name=None,
    )
    folders = list_filter_facets(conn)["folders"]
    assert len(folders) == 1
    assert folders[0]["folder_id"] == "opaque-folder-id"
    assert folders[0]["folder_name"] == "opaque-folder-id"
    assert folders[0]["count"] == 1


def test_facets_excludes_null_and_empty_values(conn) -> None:
    """An email with NULL/empty sender or folder shouldn't produce a blank
    dropdown entry — there's nothing meaningful to filter against."""
    _seed_facet_email(
        conn, "no-folder", from_address="alice@example.com", from_name="Alice",
        folder_id=None, folder_name=None,
    )
    _seed_facet_email(
        conn, "with-folder", from_address="alice@example.com", from_name="Alice",
        folder_id="inbox", folder_name="Inbox",
    )

    facets = list_filter_facets(conn)
    folder_ids = [f["folder_id"] for f in facets["folders"]]
    assert folder_ids == ["inbox"]  # only the non-NULL folder
    # No empty-string sender row even if the corpus only has one sender.
    assert all(s["address"] for s in facets["senders"])


def test_facets_respects_limits(conn) -> None:
    """Per-call caps prevent multi-MB payloads on huge corpora."""
    for i in range(5):
        _seed_facet_email(
            conn, f"s{i}", from_address=f"user{i}@example.com", from_name=None,
            folder_id=f"folder-{i}", folder_name=f"Folder {i}",
        )
    facets = list_filter_facets(conn, sender_limit=2, folder_limit=3)
    assert len(facets["senders"]) == 2
    assert len(facets["folders"]) == 3


def test_search_fts_respects_filters(conn) -> None:
    """Repo-level smoke: filter kwargs reach the SQL WHERE clause."""
    e1 = _make_email("e1", subject="alpha", body="x")
    e1.received_at = 1_000_000
    e2 = _make_email("e2", subject="alpha", body="y")
    e2.received_at = 2_000_000
    insert_email_with_chunks(conn, e1, [])
    insert_email_with_chunks(conn, e2, [])

    # No filter → both
    assert {h.email_id for h in search_fts(conn, "alpha")} == {"e1", "e2"}
    # start_at trims older
    assert {h.email_id for h in search_fts(conn, "alpha", start_at=1_500_000)} == {"e2"}
    # end_at trims newer (half-open)
    assert {h.email_id for h in search_fts(conn, "alpha", end_at=1_500_000)} == {"e1"}
