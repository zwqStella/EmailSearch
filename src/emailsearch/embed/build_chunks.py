"""Build the list of `Chunk` rows for an `EmailRow`.

Body, each attachment with non-empty extracted_text, AND the LLM-generated
summary (when present) each become their own chunks with deterministic IDs
so re-ingest is idempotent at the DB layer.
"""

from __future__ import annotations

from emailsearch.db.models import Chunk, EmailRow
from emailsearch.embed.encoder import chunk_text, embed_texts

# Sentinel source-index values used inside `_make_chunk_id` to keep chunk
# IDs unambiguous across source types. Attachments use their 0-based
# position within the email; body / summary are singletons per email so
# they get fixed sentinels.
_BODY_INDEX = -1
_SUMMARY_INDEX = -2


def build_chunks(email: EmailRow) -> list[Chunk]:
    """Returns chunks ready for `insert_email_with_chunks`.

    Strategy:
      - body becomes one or more chunks with `source_type='body'`
      - each attachment with non-empty `extracted_text` becomes one or more
        chunks with `source_type='attachment'` and `source_name=attachment.name`
      - the LLM-generated `email.summary` (when set) becomes one or more
        chunks with `source_type='summary'` and no `source_name`

    Indexing the summary as its own vector lets the semantic-search service
    treat "this email's topic matches the query" as a first-class KNN hit.
    Summary chunks carry NO leading header — the summary is already a
    self-contained topical description.

    A short "Subject: ... From: ..." header IS prepended to the **first**
    chunk of body and each attachment so those embeddings carry topic +
    sender context. Subsequent chunks of the same source carry no header
    — repeating it would dominate short chunks' semantics.

    Chunk IDs are deterministic and use the attachment's *position* within
    the email (not its filename) — long reply chains routinely produce
    multiple `image001.png` attachments, and a name-based ID would collide
    on the `vec_email_chunks` primary key.
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
    exist, we only refresh the summary slice. Returns [] for empty / None
    input.

    Chunk IDs match the format produced by `build_chunks`, so deleting all
    rows where ``source_type='summary' AND email_id=?`` before re-inserting
    keeps the table consistent across re-summarizations.
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

    Chunks each source separately to keep boundaries clean, batches every
    piece into a single `embed_texts` call (one model invocation per
    ingest, not one per source), then maps the embeddings back to Chunk rows.
    """
    # (source_type, source_name, source_index, chunk_index, text_for_embedding)
    expanded: list[tuple[str, str | None, int, int, str]] = []
    for source_type, source_name, source_index, text, header in sources:
        pieces = chunk_text(text)
        for chunk_index, piece in enumerate(pieces):
            # Only the first chunk of each source carries the header — keeps
            # later-chunk embeddings dominated by their actual content. An
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
    """Compact provenance header prepended to first chunk of each source.

    Kept short so it doesn't drown out the actual content in the embedding.
    Skips fields when missing rather than emitting empty labels.
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

    For attachments, `source_index` is the attachment's position within the
    email (0-based) — guaranteed unique regardless of filename collisions.
    Body and summary are singletons per email so they get fixed slugs (the
    ``_BODY_INDEX`` / ``_SUMMARY_INDEX`` sentinels never appear in the ID).
    """
    if source_type == "attachment":
        return f"{email_id}::att::{source_index}::{chunk_index}"
    if source_type == "summary":
        return f"{email_id}::summary::{chunk_index}"
    return f"{email_id}::body::{chunk_index}"
