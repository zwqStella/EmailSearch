"""Inline-image OCR splicer.

Outlook bodies often contain `<img src="cid:XYZ">` references that point at
inline attachments. We OCR each cited image and replace the `<img>` with a
`<span data-ocr-cid>` containing the recognized text, then run html2text to
produce the final body_text. body_html is preserved verbatim for the preview.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import html2text
from bs4 import BeautifulSoup, Tag

from emailsearch.config import get_settings

log = logging.getLogger(__name__)


def html_to_plain_text(html: str) -> str:
    """Convert HTML to plain text — Markdown-ish but readable. No image alts as text."""
    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_links = False
    h.body_width = 0  # no wrapping
    return h.handle(html or "").strip()


def augment_body_with_ocr(
    body_html: str,
    inline_attachments: Iterable[dict],
) -> tuple[str, bool]:
    """Returns `(body_text, ocr_used)`.

    `inline_attachments` is the list of attachment dicts (Graph format, augmented
    with `_bytes`) that have `isInline=True`. Their `contentId` is used to match
    `<img src="cid:...">` references.
    """
    if not body_html:
        return "", False
    if not get_settings().ocr_enabled:
        return html_to_plain_text(body_html), False

    cid_to_bytes = {
        att.get("contentId"): att.get("_bytes")
        for att in inline_attachments
        if att.get("isInline") and att.get("contentId") and att.get("_bytes")
    }
    if not cid_to_bytes:
        return html_to_plain_text(body_html), False

    # Avoid importing OCR engine if no cid images actually appear in the HTML.
    soup = BeautifulSoup(body_html, "lxml")
    img_tags = soup.find_all("img")
    if not img_tags:
        return html_to_plain_text(body_html), False

    from emailsearch.extract.extractors import extract_image_bytes

    ocr_used = False
    for img in img_tags:
        if not isinstance(img, Tag):
            continue
        src = img.get("src", "")
        if not isinstance(src, str) or not src.lower().startswith("cid:"):
            continue
        cid = src[4:].strip().lstrip("<").rstrip(">")
        data = cid_to_bytes.get(cid)
        if not data:
            continue
        text = extract_image_bytes(data)
        if not text:
            continue
        ocr_used = True
        span = soup.new_tag("span")
        span["data-ocr-cid"] = cid
        span.string = text
        img.replace_with(span)

    augmented_html = str(soup)
    return html_to_plain_text(augmented_html), ocr_used
