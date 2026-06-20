"""Search service tests with stubbed embeddings."""

from __future__ import annotations

import time

import pytest

from emailsearch.db.connection import apply_schema, open_connection
from emailsearch.db.models import AttachmentRecord, Chunk, EmailAddress, EmailRow
from emailsearch.db.repositories import insert_email_with_chunks
from emailsearch.search import service as search_service


@pytest.fixture()
def conn():
    c = open_connection(":memory:")
    apply_schema(c)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def stub_query_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the query encoder. We design hits/embeddings together so distance
    ordering is predictable."""

    def fake_embed_query(text: str) -> list[float]:
        # Match seed encoding used in `_chunk_with_seed` below.
        # "alpha" → seed 0.01, "beta" → 0.02, "gamma" → 0.03
        seed = {"alpha": 0.01, "beta": 0.02, "gamma": 0.03}.get(text.strip().lower(), 0.5)
        return _seed_vec(seed)

    monkeypatch.setattr(search_service, "embed_query", fake_embed_query)


def _seed_vec(seed: float, dim: int = 384) -> list[float]:
    return [seed + i * 0.0001 for i in range(dim)]


def _chunk(email_id: str, idx: int, text: str, *, seed: float, source_type="body", source_name=None) -> Chunk:
    cid = f"{email_id}::{source_type}::{idx}" + (f"::{source_name}" if source_name else "")
    return Chunk(
        chunk_id=cid,
        email_id=email_id,
        source_type=source_type,
        source_name=source_name,
        chunk_index=idx,
        chunk_text=text,
        embedding=_seed_vec(seed),
    )


def _email(eid: str, subject: str, body: str, atts=None) -> EmailRow:
    return EmailRow(
        id=eid,
        subject=subject,
        from_address="alice@example.com",
        from_name="Alice",
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=int(time.time()),
        body_text=body,
        body_html=f"<p>{body}</p>",
        attachments=atts or [],
        has_attachments=bool(atts),
    )


def _seed_corpus(conn) -> None:
    # m1: body about "alpha"
    insert_email_with_chunks(
        conn,
        _email("m1", "Project alpha", "we discussed alpha milestones"),
        [_chunk("m1", 0, "we discussed alpha milestones", seed=0.01)],
    )
    # m2: body about "beta"
    insert_email_with_chunks(
        conn,
        _email("m2", "Project beta", "the beta plan is ready"),
        [_chunk("m2", 0, "the beta plan is ready", seed=0.02)],
    )
    # m3: PDF attachment mentions "gamma"
    att = AttachmentRecord(
        att_id="att-3",
        name="report.pdf",
        content_type="application/pdf",
        size=100,
        extracted_text="gamma analysis details",
        status="ok",
    )
    insert_email_with_chunks(
        conn,
        _email("m3", "Q3", "see attached", atts=[att]),
        [
            _chunk("m3", 0, "see attached", seed=0.5),
            _chunk("m3", 0, "gamma analysis details", seed=0.03, source_type="attachment", source_name="report.pdf"),
        ],
    )


def test_keyword_search_finds_body(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="keyword")
    ids = [h.email_id for h in resp.hits]
    assert "m1" in ids


def test_keyword_search_finds_attachment(conn) -> None:
    """Attachment text is in searchable_text → FTS finds it; matched_in='attachment'."""
    _seed_corpus(conn)
    resp = search_service.search(conn, "gamma", mode="keyword")
    ids = [h.email_id for h in resp.hits]
    assert "m3" in ids
    hit = next(h for h in resp.hits if h.email_id == "m3")
    assert hit.matched_in == "attachment"
    assert hit.matched_attachment_name == "report.pdf"


def test_semantic_search_returns_attachment_match(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "gamma", mode="semantic")
    assert resp.hits, "expected at least one semantic hit"
    top = resp.hits[0]
    assert top.email_id == "m3"
    assert top.matched_in == "attachment"
    assert top.matched_attachment_name == "report.pdf"


def test_hybrid_search_unions_results(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="hybrid")
    ids = [h.email_id for h in resp.hits]
    assert "m1" in ids


def test_empty_query_returns_no_hits(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "   ", mode="hybrid")
    assert resp.hits == []
