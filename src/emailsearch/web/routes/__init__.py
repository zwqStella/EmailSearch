"""HTTP route helpers shared across routers."""

from __future__ import annotations

import json
from typing import Any


def ndjson_line(payload: dict[str, Any]) -> bytes:
    """Encode one NDJSON record (compact JSON + trailing newline)."""
    return (json.dumps(payload, default=str, ensure_ascii=False) + "\n").encode("utf-8")
