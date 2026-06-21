"""Tiny cross-layer helpers. Keep this module dependency-free."""

from __future__ import annotations

from datetime import UTC, datetime

# CJK Unicode ranges used by both search and ask for tokenizer / token-budget
# decisions. ASCII tokens use ``\b`` word boundaries; CJK tokens use plain
# substring (no inter-character word boundaries in the script).
_CJK_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u3400", "\u4dbf"),  # CJK Unified Ideographs Extension A
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\uac00", "\ud7af"),  # Hangul Syllables
    ("\uf900", "\ufaff"),  # CJK Compatibility Ideographs
)


def to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC. Naive inputs are assumed to be UTC already."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def is_cjk_char(ch: str) -> bool:
    """True iff ``ch`` falls in any CJK Unicode range."""
    return any(lo <= ch <= hi for lo, hi in _CJK_RANGES)


def contains_cjk(text: str) -> bool:
    """True iff ``text`` contains at least one CJK character."""
    return any(is_cjk_char(c) for c in text)
