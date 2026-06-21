"""Tests for `embed.build_chunks` using a stubbed encoder."""

from __future__ import annotations

import time

import pytest

from emailsearch.db.models import AttachmentRecord, EmailAddress, EmailRow
from emailsearch.embed import build_chunks as build_chunks_mod


@pytest.fixture(autouse=True)
def stub_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy encoder calls with deterministic stubs.

    Real chunker is fine — it's pure-Python and fast — but real embedding
    requires the 80 MB sentence-transformer model. We stub it with a fixed-dim
    deterministic vector that depends only on text length so equal strings
    produce equal vectors (good for asserting chunk IDs map to texts).
    """

    def fake_chunk_text(text: str) -> list[str]:
        # tiny chunks so multi-chunk paths exercise
        words = text.split()
        chunks = []
        size = 8
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i : i + size]))
        return chunks or ([text] if text.strip() else [])

    def fake_embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            base = (len(t) % 100) * 0.001
            out.append([base + (i * 0.0001) for i in range(384)])
        return out

    monkeypatch.setattr(build_chunks_mod, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(build_chunks_mod, "embed_texts", fake_embed)


def _email(
    body: str = "this is the body of the email",
    attachments: list[AttachmentRecord] | None = None,
    *,
    subject: str = "hello",
    summary: str | None = None,
) -> EmailRow:
    return EmailRow(
        id="msg-1",
        subject=subject,
        from_address="alice@example.com",
        from_name="Alice",
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=int(time.time()),
        body_text=body,
        body_html=f"<p>{body}</p>",
        summary=summary,
        attachments=attachments or [],
    )


def test_empty_email_returns_no_chunks() -> None:
    # Empty subject + empty body → no embeddable content at all.
    email = _email(body="", subject="")
    assert build_chunks_mod.build_chunks(email) == []


def test_body_only_chunks() -> None:
    email = _email(body="alpha " * 30)  # ~30 words → 4 chunks of 8
    chunks = build_chunks_mod.build_chunks(email)
    assert len(chunks) >= 2
    assert all(c.source_type == "body" for c in chunks)
    assert all(c.email_id == "msg-1" for c in chunks)
    # chunk IDs are deterministic
    assert chunks[0].chunk_id == "msg-1::body::0"
    assert all(len(c.embedding) == 384 for c in chunks)


def test_attachment_chunks_carry_source_name() -> None:
    att = AttachmentRecord(
        att_id="att-1",
        name="report.pdf",
        content_type="application/pdf",
        size=100,
        extracted_text="alpha beta gamma " * 5,  # 15 words → 2 chunks
        status="ok",
    )
    email = _email(body="short body", attachments=[att])
    chunks = build_chunks_mod.build_chunks(email)
    body_chunks = [c for c in chunks if c.source_type == "body"]
    att_chunks = [c for c in chunks if c.source_type == "attachment"]
    assert len(body_chunks) >= 1
    assert len(att_chunks) >= 1
    assert all(c.source_name == "report.pdf" for c in att_chunks)
    # Chunk IDs use the attachment's position (att=0 here) — NOT the filename,
    # because long reply chains often have multiple identically-named attachments.
    assert att_chunks[0].chunk_id == "msg-1::att::0::0"


def test_chunk_ids_unique_when_attachments_share_filename() -> None:
    """Two attachments named 'image001.png' must not collide on chunk_id."""
    atts = [
        AttachmentRecord(
            att_id="att-a",
            name="image001.png",
            content_type="image/png",
            size=100,
            extracted_text="text from first image",
            status="ok",
        ),
        AttachmentRecord(
            att_id="att-b",
            name="image001.png",
            content_type="image/png",
            size=100,
            extracted_text="text from second image",
            status="ok",
        ),
    ]
    email = _email(body="see images", attachments=atts)
    chunks = build_chunks_mod.build_chunks(email)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"duplicate chunk IDs: {ids}"
    att_ids = sorted(c.chunk_id for c in chunks if c.source_type == "attachment")
    assert att_ids == ["msg-1::att::0::0", "msg-1::att::1::0"]


def test_skips_empty_attachment_text() -> None:
    att = AttachmentRecord(
        att_id="att-1",
        name="empty.bin",
        content_type="application/octet-stream",
        size=10,
        extracted_text="",
        status="unsupported",
    )
    email = _email(attachments=[att])
    chunks = build_chunks_mod.build_chunks(email)
    assert all(c.source_type != "attachment" for c in chunks)


def test_subject_and_from_prepended_to_body_chunks() -> None:
    """The first body chunk should contain the subject + from prefix."""
    email = _email(body="zzz " * 2)  # one chunk (2 words)
    chunks = build_chunks_mod.build_chunks(email)
    assert "Subject: hello" in chunks[0].chunk_text
    assert "alice@example.com" in chunks[0].chunk_text


def test_attachment_first_chunk_has_subject_from_filename_header() -> None:
    """First attachment chunk should carry Subject/From/Attachment header for
    embedding context — later chunks of the same attachment should NOT, to
    avoid having the header dominate short chunks' semantics."""
    att = AttachmentRecord(
        att_id="att-1",
        name="quarterly-report.pdf",
        content_type="application/pdf",
        size=100,
        extracted_text="alpha beta gamma " * 10,  # 30 words → 4 chunks of 8
        status="ok",
    )
    email = _email(attachments=[att])
    chunks = build_chunks_mod.build_chunks(email)
    att_chunks = [c for c in chunks if c.source_type == "attachment"]
    assert len(att_chunks) >= 2

    first = att_chunks[0]
    assert "Subject: hello" in first.chunk_text
    assert "alice@example.com" in first.chunk_text
    assert "Attachment: quarterly-report.pdf" in first.chunk_text

    # Later chunks of the same attachment carry no header — keeps the
    # embedding focused on the actual chunk text.
    later = att_chunks[1]
    assert "Subject:" not in later.chunk_text
    assert "Attachment:" not in later.chunk_text


def test_header_skips_missing_fields() -> None:
    """No empty 'Subject: ' lines when fields are missing."""
    att = AttachmentRecord(
        att_id="att-1",
        name="report.pdf",
        content_type="application/pdf",
        size=100,
        extracted_text="some content",
        status="ok",
    )
    email = EmailRow(
        id="msg-x",
        subject="",  # no subject
        from_address="",  # no sender
        received_at=int(time.time()),
        body_text="body content",
        body_html="",
        attachments=[att],
        has_attachments=True,
    )
    chunks = build_chunks_mod.build_chunks(email)
    body_chunk = next(c for c in chunks if c.source_type == "body")
    att_chunk = next(c for c in chunks if c.source_type == "attachment")
    assert "Subject:" not in body_chunk.chunk_text
    assert "From:" not in body_chunk.chunk_text
    assert "Subject:" not in att_chunk.chunk_text
    assert "From:" not in att_chunk.chunk_text
    # Filename header still present even when subject/from are empty
    assert "Attachment: report.pdf" in att_chunk.chunk_text


def test_summary_is_emitted_as_summary_chunk() -> None:
    """The LLM-generated summary now gets its own embedded chunk with
    `source_type='summary'`. This is what lets semantic search treat
    "matches the email's topical summary" as a first-class KNN hit
    (and promote those matches above body/attachment-only matches)
    instead of paying for a per-candidate rerank embedding at query time.
    """
    email = _email(
        body="some body content",
        subject="Q3 budget review",
        summary="Q3 budget approved; deliverable due Aug 15.",
    )
    chunks = build_chunks_mod.build_chunks(email)
    summary_chunks = [c for c in chunks if c.source_type == "summary"]
    assert summary_chunks, "expected at least one summary chunk"
    s = summary_chunks[0]
    # Stable chunk-id format: no source_index in the slug because summary
    # is a singleton per email.
    assert s.chunk_id == "msg-1::summary::0"
    assert s.source_name is None
    # No header prepended — summary text is already self-contained.
    assert s.chunk_text == "Q3 budget approved; deliverable due Aug 15."
    assert "Subject:" not in s.chunk_text
    assert "From:" not in s.chunk_text
    assert len(s.embedding) == 384


def test_no_summary_chunk_when_summary_absent() -> None:
    """An email without a summary (LLM disabled / failed at ingest) must
    not emit a summary chunk — empty/None summary is a no-op so we don't
    burn a vector on whitespace."""
    email = _email(body="body content", summary=None)
    chunks = build_chunks_mod.build_chunks(email)
    assert not any(c.source_type == "summary" for c in chunks)

    email_empty = _email(body="body content", summary="   ")
    chunks_empty = build_chunks_mod.build_chunks(email_empty)
    assert not any(c.source_type == "summary" for c in chunks_empty)


def test_build_summary_chunks_standalone() -> None:
    """`build_summary_chunks` is the per-email helper used by
    `set_email_summary` after the loader has already inserted body +
    attachment chunks. It returns ONLY summary chunks, with the same
    chunk-id format `build_chunks` uses for the summary slice."""
    chunks = build_chunks_mod.build_summary_chunks(
        "msg-1", "Q3 budget approved; deliverable due Aug 15."
    )
    assert len(chunks) >= 1
    assert all(c.source_type == "summary" for c in chunks)
    assert chunks[0].chunk_id == "msg-1::summary::0"
    assert chunks[0].email_id == "msg-1"
    assert chunks[0].source_name is None
    assert len(chunks[0].embedding) == 384


def test_build_summary_chunks_returns_empty_for_blank_input() -> None:
    """None / empty / whitespace summary -> no chunks. Lets callers no-op."""
    assert build_chunks_mod.build_summary_chunks("msg-1", None) == []
    assert build_chunks_mod.build_summary_chunks("msg-1", "") == []
    assert build_chunks_mod.build_summary_chunks("msg-1", "   ") == []
