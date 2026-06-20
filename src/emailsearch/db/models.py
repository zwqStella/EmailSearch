"""Pydantic models for the persisted entities."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

ExtractionStatus = Literal["ok", "skipped_too_large", "unsupported", "failed", "empty"]
SourceType = Literal["body", "attachment"]


class EmailAddress(BaseModel):
    address: str
    name: str | None = None


class AttachmentRecord(BaseModel):
    """One row in `emails.attachments` JSON array."""

    att_id: str
    name: str
    content_type: str
    size: int
    is_inline: bool = False
    content_id: str | None = None
    extracted_text: str = ""
    status: ExtractionStatus = "ok"
    ocr_used: bool = False
    error: str | None = None


class EmailRow(BaseModel):
    """A row of the `emails` table."""

    id: str
    subject: str = ""
    from_address: str = ""
    from_name: str | None = None
    to_addresses: list[EmailAddress] = Field(default_factory=list)
    cc_addresses: list[EmailAddress] = Field(default_factory=list)
    received_at: int  # unix epoch seconds
    sent_at: int | None = None
    folder_id: str | None = None
    folder_name: str | None = None
    conversation_id: str | None = None
    body_text: str = ""
    body_html: str = ""
    web_link: str | None = None
    attachments: list[AttachmentRecord] = Field(default_factory=list)
    has_attachments: bool = False
    body_ocr_used: bool = False

    @property
    def searchable_text(self) -> str:
        parts = [self.body_text]
        for a in self.attachments:
            if a.extracted_text:
                parts.append(a.extracted_text)
        return " ".join(p for p in parts if p)


class Chunk(BaseModel):
    """One row destined for `vec_email_chunks`."""

    chunk_id: str
    email_id: str
    source_type: SourceType
    source_name: str | None = None  # attachment filename for source_type='attachment'
    chunk_index: int
    chunk_text: str
    embedding: list[float]


class FtsHit(BaseModel):
    email_id: str
    subject: str
    from_address: str
    from_name: str | None
    received_at: int
    snippet: str
    rank: float  # bm25, lower = better


class VecHit(BaseModel):
    email_id: str
    chunk_id: str
    source_type: SourceType
    source_name: str | None
    chunk_text: str
    distance: float  # lower = better


# ---------- helpers ----------


def addresses_to_json(addrs: list[EmailAddress]) -> str:
    return json.dumps([a.model_dump() for a in addrs], ensure_ascii=False)


def addresses_from_json(s: str | None) -> list[EmailAddress]:
    if not s:
        return []
    return [EmailAddress(**a) for a in json.loads(s)]


def attachments_to_json(atts: list[AttachmentRecord]) -> str:
    return json.dumps([a.model_dump() for a in atts], ensure_ascii=False)


def attachments_from_json(s: str | None) -> list[AttachmentRecord]:
    if not s:
        return []
    return [AttachmentRecord(**a) for a in json.loads(s)]
