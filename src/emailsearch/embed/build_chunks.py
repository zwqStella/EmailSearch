"""Build the list of `Chunk` rows for an `EmailRow`.

Body and each attachment with non-empty extracted_text become separate chunks
with deterministic IDs so re-ingest is idempotent at the DB layer.
"""

from __future__ import annotations

from emailsearch.db.models import Chunk, EmailRow
from emailsearch.embed.encoder import chunk_text, embed_texts


def build_chunks(email: EmailRow) -> list[Chunk]:
    """Returns chunks ready for `insert_email_with_chunks`.

    Strategy:
      - body becomes one or more chunks with `source_type='body'`
      - each attachment with non-empty `extracted_text` becomes one or more
        chunks with `source_type='attachment'` and `source_name=attachment.name`

    A short "Subject: ... From: ..." header is prepended to the **first** chunk
    of every source (body + each attachment) so the embedding carries the
    email's topic + sender context. Attachment chunks also include the
    filename in the header. Subsequent chunks of the same source carry no
    header — repeating it would dominate short chunks' semantics.

    Chunk IDs are deterministic and use the attachment's *position* within the
    email (not its filename) — long reply chains routinely produce multiple
    `image001.png` attachments, and a name-based ID would collide on the
    `vec_email_chunks` primary key.
    """
    header_body = _header_text(email)

    # (source_type, source_name, source_index, raw_text, leading_header)
    #   source_index = -1 for body, 0..N for attachments (position within email)
    sources: list[tuple[str, str | None, int, str, str]] = []

    if email.body_text and email.body_text.strip():
        sources.append(("body", None, -1, email.body_text.strip(), header_body))

    for att_pos, att in enumerate(email.attachments):
        if att.extracted_text and att.extracted_text.strip():
            header_att = _header_text(email, attachment_name=att.name)
            sources.append(
                ("attachment", att.name, att_pos, att.extracted_text.strip(), header_att)
            )

    if not sources:
        return []

    # Chunk each source separately to keep boundaries clean; embed in one batch.
    # (source_type, source_name, source_index, chunk_index, text_for_embedding)
    expanded: list[tuple[str, str | None, int, int, str]] = []
    for source_type, source_name, source_index, text, header in sources:
        pieces = chunk_text(text)
        for chunk_index, piece in enumerate(pieces):
            # Only the first chunk of each source carries the header — keeps
            # later-chunk embeddings dominated by their actual content.
            text_for_embedding = f"{header}\n\n{piece}".strip() if chunk_index == 0 else piece
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
        chunk_id = _make_chunk_id(email.id, source_type, source_index, chunk_index)
        out.append(
            Chunk(
                chunk_id=chunk_id,
                email_id=email.id,
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
    The attachment's display name lives in the `source_name` column for the UI.
    """
    if source_type == "attachment":
        return f"{email_id}::att::{source_index}::{chunk_index}"
    return f"{email_id}::body::{chunk_index}"
