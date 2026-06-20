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
    search_fts,
    search_vec,
)


@pytest.fixture()
def conn():
    c = open_connection(":memory:")
    apply_schema(c)
    try:
        yield c
    finally:
        c.close()


def _make_email(
    email_id: str = "msg-1",
    subject: str = "Project Atlas weekly sync",
    body: str = "Discussion of Q3 milestones and a draft of the new roadmap.",
    attachments: list[AttachmentRecord] | None = None,
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
