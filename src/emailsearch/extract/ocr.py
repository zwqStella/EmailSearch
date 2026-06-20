"""Lazy OCR singleton using rapidocr-onnxruntime.

The first call downloads ~30 MB of ONNX models into the rapidocr cache; later
calls reuse the loaded engine.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR  # noqa: F401

log = logging.getLogger(__name__)

_engine: object | None = None
_engine_lock = threading.Lock()

# Skip tiny images (signature icons, spacer pixels) — pure waste of OCR cycles.
MIN_OCR_DIMENSION = 100


def _get_engine() -> object:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR

                log.info("loading rapidocr (first call downloads ~30 MB ONNX models)")
                _engine = RapidOCR()
    return _engine


def ocr_image(image_bytes: bytes) -> str:
    """OCR an image. Returns the concatenated detected text, or '' on failure / tiny image."""
    if not image_bytes:
        return ""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            w, h = img.size
            if w < MIN_OCR_DIMENSION and h < MIN_OCR_DIMENSION:
                return ""
            # rapidocr accepts a numpy array or a path; passing bytes directly works in 1.4+.
            engine = _get_engine()
    except Exception as exc:
        log.warning("ocr: failed to open image: %s", exc)
        return ""

    try:
        result, _elapsed = engine(image_bytes)  # type: ignore[operator]
    except Exception as exc:
        log.warning("ocr: engine error: %s", exc)
        return ""

    if not result:
        return ""
    # result is list of [box, text, score]
    return " ".join(item[1] for item in result if len(item) >= 2)
