"""Email-level extraction pipeline.

Combines a `RawMessage` (from any source — Outlook COM today) with per-attachment
text extraction and inline-image OCR into an `EmailRow` ready for chunking + DB
insertion.

Source-agnostic by design: nothing here knows about COM, Graph, IMAP, etc.
"""

from __future__ import annotations

import logging

from emailsearch.db.models import (
    AttachmentRecord,
    EmailAddress,
    EmailRow,
)
from emailsearch.extract.extractors import extract_attachment
from emailsearch.extract.inline_images import augment_body_with_ocr
from emailsearch.outlook.raw import RawAttachment, RawMessage

log = logging.getLogger(__name__)


def extract_email(raw: RawMessage) -> EmailRow:
    """Build an `EmailRow` from a `RawMessage` (with its attachments inline)."""
    # ---------- per-attachment text extraction ----------
    att_records: list[AttachmentRecord] = []
    inline_atts_for_splicer: list[dict] = []
    for att in raw.attachments:
        if att.is_inline and att.content_id and att.content_bytes:
            # The splicer takes a dict-shape (kept stable so it's source-neutral).
            inline_atts_for_splicer.append(
                {
                    "isInline": True,
                    "contentId": att.content_id,
                    "_bytes": att.content_bytes,
                }
            )
        att_records.append(_extract_one_attachment(att))

    # ---------- body: HTML → plain text, splicing OCR for cid: images ----------
    if raw.body_html:
        body_html = raw.body_html
        body_text, ocr_used = augment_body_with_ocr(body_html, inline_atts_for_splicer)
    else:
        body_text = (raw.body_text or "").strip()
        body_html = ""
        ocr_used = False

    # Empty body → use the source's preview.
    if not body_text:
        body_text = (raw.body_preview or "").strip()

    # body_html drives the iframe preview; synthesize one whenever
    # body_text has content so the preview is never blank.
    if not body_html and body_text:
        body_html = f"<pre>{_escape_html(body_text)}</pre>"

    return EmailRow(
        id=raw.id,
        subject=raw.subject,
        from_address=raw.from_address,
        from_name=raw.from_name,
        to_addresses=[EmailAddress(address=a, name=n) for a, n in raw.to],
        cc_addresses=[EmailAddress(address=a, name=n) for a, n in raw.cc],
        received_at=raw.received_at,
        sent_at=raw.sent_at,
        folder_id=raw.folder_id,
        folder_name=raw.folder_name,
        conversation_id=raw.conversation_id,
        body_text=body_text,
        body_html=body_html,
        web_link=raw.web_link,
        attachments=att_records,
        has_attachments=raw.has_attachments or bool(att_records),
        body_ocr_used=ocr_used,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extract_one_attachment(att: RawAttachment) -> AttachmentRecord:
    base = AttachmentRecord(
        att_id=att.att_id,
        name=att.name,
        content_type=att.content_type,
        size=att.size,
        is_inline=att.is_inline,
        content_id=att.content_id,
    )

    if att.skipped_reason is not None or att.content_bytes is None:
        reason = att.skipped_reason or "missing_content"
        if reason.startswith("too_large"):
            base.status = "skipped_too_large"
        elif reason.startswith("unsupported_type"):
            base.status = "unsupported"
        else:
            base.status = "failed"
            base.error = reason
        return base

    text, status, error = extract_attachment(att.content_type, att.content_bytes)
    base.extracted_text = text
    base.status = status
    base.error = error
    base.ocr_used = status == "ok" and att.content_type.lower().startswith("image/")
    return base


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
