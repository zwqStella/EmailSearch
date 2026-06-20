"""Backend-agnostic 'raw message' model.

Whoever pulls mail from a source (Outlook COM today; could be IMAP / Graph
/ EML files later) produces these. The downstream extract pipeline only
talks to `RawMessage` so swapping backends is a one-file change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RecipientKind = Literal["to", "cc", "bcc"]


class RawAttachment(BaseModel):
    """One attachment as fetched from the source.

    `content_bytes` is None when we couldn't (or chose not to) load the
    payload — e.g. size-cap exceeded, COM `SaveAsFile` failed, or the type
    is something we don't support.
    """

    att_id: str
    name: str
    content_type: str
    size: int
    is_inline: bool = False
    content_id: str | None = None
    content_bytes: bytes | None = None
    skipped_reason: str | None = None  # "too_large:N", "unsupported_type:...", "decode_failed", "save_failed"


class RawMessage(BaseModel):
    """One mail message in source-neutral form."""

    id: str  # stable idempotency key (Internet message id when available, else EntryID)
    subject: str = ""
    from_address: str = ""
    from_name: str | None = None
    to: list[tuple[str, str | None]] = Field(default_factory=list)  # (address, name)
    cc: list[tuple[str, str | None]] = Field(default_factory=list)
    received_at: int  # unix epoch seconds
    sent_at: int | None = None
    body_html: str = ""
    body_text: str = ""  # populated when source has no HTML body
    body_preview: str = ""
    conversation_id: str | None = None
    folder_id: str | None = None
    folder_name: str | None = None
    web_link: str | None = None  # outlook:<EntryID> for COM ("Open in Outlook")
    has_attachments: bool = False
    attachments: list[RawAttachment] = Field(default_factory=list)
