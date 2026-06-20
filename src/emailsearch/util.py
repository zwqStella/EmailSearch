"""Tiny cross-layer helpers. Keep this module dependency-free."""

from __future__ import annotations

from datetime import UTC, datetime


def to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC. Naive inputs are assumed to be UTC already."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
