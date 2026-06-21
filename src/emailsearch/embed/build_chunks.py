"""Build the list of `Chunk` rows for an `EmailRow`.

Body, each attachment with non-empty extracted_text, AND the LLM
summary (when present) each become their own chunks with deterministic
IDs so re-ingest is idempotent at the DB layer.
"""

from __future__ import annotations

from emailsearch.db.models import Chunk, EmailRow
from emailsearch.embed.encoder import chunk_text, embed_texts

# Sentinel source-index values inside `_make_chunk_id` to keep chunk IDs
# unambiguous. Attachments use their 0-based position within the email;
# body / summary are singletons per email so they get fixed sentinels.
_BODY_INDEX = -1
_SUMMARY_INDEX = -2


def build_chunks(email: EmailRow) -> list[Chunk]:
    """Build chunks ready for `insert_email_with_chunks`.

    Strategy:
      - body → one or more `source_type='body'` chunks
      - each attachment with non-empty `extracted_text` →
        `source_type='attachment'` chunks (`source_name=att.name`)
      - the LLM summary (when set) → `source_type='summary'` chunks

    Indexing the summary as its own vector lets semantic search treat
    "this email's topic matches the query" as a first-class KNN hit.
    Summary chunks carry NO leading header — they're already a
    self-contained topical description.

    A short "Subject: ... From: ..." header IS prepended to the FIRST
    chunk of body and each attachment for embedding context. Subsequent
    chunks of the same source carry no header — repeating it would
    dominate short chunks' semantics.

    Chunk IDs use the attachment's *position* within the email (not its
    filename) because long reply chains routinely produce multiple
    `image001.png` attachments.
    """
    header_body = _header_text(email)

    # (source_type, source_name, source_index, raw_text, leading_header)
    #   source_index = _BODY_INDEX for body, 0..N for attachments, _SUMMARY_INDEX for summary.
    sources: list[tuple[str, str | None, int, str, str]] = []

    if email.body_text and email.body_text.strip():
        sources.append(("body", None, _BODY_INDEX, email.body_text.strip(), header_body))

    for att_pos, att in enumerate(email.attachments):
        if att.extracted_text and att.extracted_text.strip():
            header_att = _header_text(email, attachment_name=att.name)
            sources.append(
                ("attachment", att.name, att_pos, att.extracted_text.strip(), header_att)
            )

    if email.summary and email.summary.strip():
        # No header for summary — it's already a self-contained topical
        # sentence. Adding "Subject: ..." in front would dilute the embedding.
        sources.append(("summary", None, _SUMMARY_INDEX, email.summary.strip(), ""))

    if not sources:
        return []

    return _embed_and_assemble(email.id, sources)


def build_summary_chunks(email_id: str, summary: str | None) -> list[Chunk]:
    """Embed just the LLM summary as `source_type='summary'` chunks.

    Used by `set_email_summary` when a summary is written or rewritten
    onto an already-indexed email — body + attachment chunks already
    exist, we only refresh the summary slice. Returns [] for empty input.
    Chunk IDs match `build_chunks` so deleting all summary rows before
    re-inserting keeps the table consistent.
    """
    if not summary or not summary.strip():
        return []
    sources: list[tuple[str, str | None, int, str, str]] = [
        ("summary", None, _SUMMARY_INDEX, summary.strip(), "")
    ]
    return _embed_and_assemble(email_id, sources)


def _embed_and_assemble(
    email_id: str,
    sources: list[tuple[str, str | None, int, str, str]],
) -> list[Chunk]:
    """Shared body for `build_chunks` / `build_summary_chunks`.

    Chunks each source separately to keep boundaries clean, batches
    every piece into a single `embed_texts` call (one model invocation
    per ingest), then maps embeddings back to Chunk rows.
    """
    # (source_type, source_name, source_index, chunk_index, text_for_embedding)
    expanded: list[tuple[str, str | None, int, int, str]] = []
    for source_type, source_name, source_index, text, header in sources:
        pieces = chunk_text(text)
        for chunk_index, piece in enumerate(pieces):
            # Only the first chunk of each source carries the header —
            # later-chunk embeddings stay dominated by their content. An
            # empty header (summary) collapses to just ``piece``.
            text_for_embedding = (
                f"{header}\n\n{piece}".strip() if chunk_index == 0 and header else piece
            )
            expanded.append(
                (source_type, source_name, source_index, chunk_index, text_for_embedding)
            )

    if not expanded:
        return []

    embeddings = embed_texts([piece for *_, piece in expanded])

    out: list[Chunk] = []
    for (source_type, source_name, source_index, chunk_index, piece), emb in zip(
        expanded, embeddings, strict=True
    ):
        chunk_id = _make_chunk_id(email_id, source_type, source_index, chunk_index)
        out.append(
            Chunk(
                chunk_id=chunk_id,
                email_id=email_id,
                source_type=source_type,  # type: ignore[arg-type]
                source_name=source_name,
                chunk_index=chunk_index,
                chunk_text=piece,
                embedding=emb,
            )
        )
    return out


def _header_text(email: EmailRow, *, attachment_name: str | None = None) -> str:
    """Compact provenance header prepended to the first chunk of each source.
    Kept short so it doesn't drown out the actual content in the embedding.
    """
    parts: list[str] = []
    if email.subject:
        parts.append(f"Subject: {email.subject}")
    if email.from_address:
        parts.append(f"From: {email.from_address}")
    if attachment_name:
        parts.append(f"Attachment: {attachment_name}")
    return "\n".join(parts)


def _make_chunk_id(
    email_id: str,
    source_type: str,
    source_index: int,
    chunk_index: int,
) -> str:
    """Stable, collision-free per-chunk primary key.

    For attachments, `source_index` is the attachment's position within
    the email (0-based) — unique regardless of filename collisions.
    Body and summary are singletons per email so they get fixed slugs.
    """
    if source_type == "attachment":
        return f"{email_id}::att::{source_index}::{chunk_index}"
    if source_type == "summary":
        return f"{email_id}::summary::{chunk_index}"
    return f"{email_id}::body::{chunk_index}"
