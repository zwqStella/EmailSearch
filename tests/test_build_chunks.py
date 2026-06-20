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
) -> EmailRow:
    return EmailRow(
        id="msg-1",
        subject="hello",
        from_address="alice@example.com",
        from_name="Alice",
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=int(time.time()),
        body_text=body,
        body_html=f"<p>{body}</p>",
        attachments=attachments or [],
    )


def test_empty_email_returns_no_chunks() -> None:
    email = _email(body="")
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
