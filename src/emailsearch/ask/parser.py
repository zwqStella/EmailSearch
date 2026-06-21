"""Question parser for the Ask agent.

ONE LLM hop turns a natural-language question into a structured
:class:`ParsedAskRequest` containing the distilled search query AND the
hard filters (date range, sender) inferred from the question. This is
what lets the user type "yesterday's email from alice@example.com about
budget" and have the date + sender filters land on the search without a
second LLM round-trip.

Lives in ``ask/`` rather than as a method on ``summarize/`` because the
existing :func:`summarize.distill_query` produces a bilingual bag for
OR-joined FTS, while the Ask agent feeds the whole ``search()``
machinery (hybrid mode) and needs a clean topical phrase that works for
AND-joined FTS, OR-joined FTS, and embedding lookup alike.

Failure semantics: any error path returns ``ParsedAskRequest(query=question)``
so the agent always has *some* query to run.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date

from pydantic import BaseModel

from emailsearch.config import get_settings
from emailsearch.summarize.client import _call_chat

log = logging.getLogger(__name__)

# Conservative email-shape filter — we only want to populate
# SearchFilters.from_address when we're sure the string looks like an
# address. Anything else falls back to None.
_EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Strip Markdown code fences if the model wraps the JSON. Some local
# models default to ``` ```json ... ``` ``` blocks even when told not to.
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)

# Designed to coexist with ``summarize.client._DISTILL_PROMPT`` (FTS
# bilingual bag) — see module docstring. Keep the filler-stripping +
# container-noun-dropping rules in sync; this prompt differs by emitting
# a single clean phrase and three extra structured fields.
_ASK_PARSE_PROMPT = (
    "You are an email-search query parser. The user typed a question\n"
    "about their email inbox. Parse it into a strict JSON object with\n"
    "four fields:\n"
    "\n"
    "  {{\n"
    '    "query":        <string — the topical search phrase>,\n'
    '    "start_at":     <unix epoch seconds | null>,\n'
    '    "end_at":       <unix epoch seconds | null>,\n'
    '    "from_address": <email address string | null>\n'
    "  }}\n"
    "\n"
    "Today's date (for resolving relative dates) is {today_iso}.\n"
    "Time zone: server local.\n"
    "\n"
    "FIELD RULES\n"
    "===========\n"
    "\n"
    'query (REQUIRED):\n'
    "  A short topical phrase describing what the email is ABOUT — the\n"
    "  same content a person would search for. It must work as a\n"
    "  bm25/keyword/embedding query simultaneously, so DO NOT emit a\n"
    "  bag of alternatives or translations — pick the single best\n"
    "  phrase.\n"
    "  - Preserve proper nouns, product names, technical terms,\n"
    "    version numbers, project codes verbatim.\n"
    "  - DROP filler that describes the search itself (every result\n"
    "    IS an email): 'email', 'message', 'mail', '邮件', '信件',\n"
    "    'help me find', 'show me', 'I'm looking for', '帮我找',\n"
    "    '搜索', '查一下', '找一下'.\n"
    "  - DROP relative-time phrases — those go in start_at/end_at,\n"
    "    not the query: 'last month', 'yesterday', 'this week',\n"
    "    'today', 'recent', '上个月', '昨天', '上周', '最近', '今天'.\n"
    "  - DROP 'from <person>' / '<email-address> 发的' phrases — that\n"
    "    goes in from_address, not the query.\n"
    "  - Keep the query in the SAME LANGUAGE as the question. Do NOT\n"
    "    translate. (The downstream search has its own bilingual\n"
    "    expansion for the FTS leg.)\n"
    "\n"
    "start_at / end_at:\n"
    "  Unix epoch seconds. Define a half-open interval [start_at, end_at)\n"
    "  matching the time scope of the question. Both null = no date\n"
    "  filter. Common patterns relative to {today_iso}:\n"
    "    - 'today'        → [today 00:00, tomorrow 00:00)\n"
    "    - 'yesterday'    → [yesterday 00:00, today 00:00)\n"
    "    - 'this week'    → [Monday 00:00, next Monday 00:00)\n"
    "    - 'this month'   → [day 1 00:00, day 1 of next month 00:00)\n"
    "    - 'last month'   → [day 1 last month 00:00, day 1 this month 00:00)\n"
    "    - 'past 7 days'  → [today minus 7 days 00:00, tomorrow 00:00)\n"
    "    - 'in 2024'      → [Jan 1 2024 00:00, Jan 1 2025 00:00)\n"
    "  When the question has no temporal phrasing at all, leave BOTH null.\n"
    "  Never invent a date out of thin air.\n"
    "\n"
    "from_address:\n"
    "  ONLY when the question contains a literal email address (string\n"
    "  matching <localpart>@<domain>). Copy it verbatim. If the user\n"
    "  said only a name ('from alice'), leave this null — name-based\n"
    "  fuzzy matching is not supported.\n"
    "\n"
    "OUTPUT FORMAT\n"
    "=============\n"
    "Reply with EXACTLY ONE JSON object on a single line. No preamble,\n"
    "no commentary, no Markdown code fence. Use null (lowercase) for\n"
    "missing fields, never omit a key.\n"
    "\n"
    "EXAMPLES\n"
    "========\n"
    "Today is 2026-06-21 in all examples.\n"
    "\n"
    "Q: when is the garage day this month?\n"
    'A: {{"query":"garage day","start_at":1748736000,"end_at":1751414400,"from_address":null}}\n'
    "\n"
    "Q: yesterday's email from alice@example.com about Q3 budget\n"
    'A: {{"query":"Q3 budget","start_at":1750464000,"end_at":1750550400,"from_address":"alice@example.com"}}\n'
    "\n"
    "Q: what did Bob say about the security incident last week?\n"
    'A: {{"query":"security incident Bob","start_at":1749945600,"end_at":1750550400,"from_address":null}}\n'
    "\n"
    "Q: summarize the rollout plan\n"
    'A: {{"query":"rollout plan","start_at":null,"end_at":null,"from_address":null}}\n'
    "\n"
    "Q: 上个月工会发的开心麻花活动是什么时候\n"
    'A: {{"query":"工会 开心麻花 活动","start_at":1746979200,"end_at":1748736000,"from_address":null}}\n'
    "\n"
    "Now parse:\n"
    "\n"
    "Q: {question}\n"
    "A: "
)


class ParsedAskRequest(BaseModel):
    """Structured form of a user's natural-language question.

    Produced by :func:`parse_ask_question` from a single LLM hop. The
    Ask agent passes the four fields directly into the existing
    :class:`SearchFilters` + ``search()`` machinery. All filter fields
    default to ``None`` (no constraint); ``query`` is always populated
    — on parse failure we fall back to the original question.
    """

    query: str
    start_at: int | None = None
    end_at: int | None = None
    from_address: str | None = None


def parse_ask_question(question: str) -> ParsedAskRequest:
    """Parse a natural-language question into query + filters.

    Single LLM hop. On any failure (LLM disabled, network, bad JSON,
    missing required field) returns ``ParsedAskRequest(query=question)``
    — worst case the agent runs the raw question against an unfiltered
    search, still useful.

    ``today_iso`` is the server's local date so relative phrases
    ("this month", "yesterday") resolve against the same clock that
    produced the indexed ``received_at`` epochs. Per-user timezones are
    out of scope.
    """
    question = (question or "").strip()
    if not question:
        return ParsedAskRequest(query="")

    settings = get_settings()
    fallback = ParsedAskRequest(query=question)
    if not settings.llm_enabled:
        log.info("ask parse: LLM disabled — falling back to raw question")
        return fallback

    today_iso = date.today().isoformat()
    prompt = _ASK_PARSE_PROMPT.format(today_iso=today_iso, question=question)
    response = _call_chat(
        prompt=prompt,
        max_tokens=settings.ask_parse_max_tokens,
        log_label=f"ask_parse({question[:40]!r})",
    )
    if response is None:
        return fallback

    parsed = _extract_parsed_request(response, fallback_query=question)
    log.info(
        "ask parse: question=%r → query=%r start_at=%s end_at=%s from=%r",
        question, parsed.query, parsed.start_at, parsed.end_at, parsed.from_address,
    )
    return parsed


def _extract_parsed_request(
    response: str, *, fallback_query: str,
) -> ParsedAskRequest:
    """Defensive JSON extraction + field validation.

    Handles three real failure modes: (1) ``` ```json ... ``` ``` wrapped
    output, (2) trailing prose after the JSON object (uses
    :func:`json.JSONDecoder.raw_decode`), (3) wrong field types (string
    ``"null"``, malformed email shape) — coerce or drop per field.
    Never raises; defaults to fallback on any failure.
    """
    stripped = response.strip()
    fence = _JSON_FENCE_RE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()

    # raw_decode recovers when the model appends commentary after the JSON.
    decoder = json.JSONDecoder()
    try:
        obj, _consumed = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        log.warning("ask parse: failed to decode JSON: %s — raw=%r", exc, stripped[:200])
        return ParsedAskRequest(query=fallback_query)

    if not isinstance(obj, dict):
        log.warning("ask parse: top-level JSON not an object — got %r", type(obj))
        return ParsedAskRequest(query=fallback_query)

    query = obj.get("query")
    if not isinstance(query, str) or not query.strip():
        query = fallback_query
    else:
        query = query.strip()

    return ParsedAskRequest(
        query=query,
        start_at=_coerce_epoch(obj.get("start_at")),
        end_at=_coerce_epoch(obj.get("end_at")),
        from_address=_coerce_email(obj.get("from_address")),
    )


def _coerce_epoch(value: object) -> int | None:
    """Accept JSON null, ints, and numeric strings; reject everything
    else. Drops zero / negative epochs (indexed mail is post-1970)."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool subclasses int — reject up-front
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value <= 0 or value != value:  # NaN check
            return None
        return int(value)
    if isinstance(value, str):
        try:
            n = int(value.strip())
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None
    return None


def _coerce_email(value: object) -> str | None:
    """Only return strings that look like email addresses. A model that
    fills ``from_address`` with a person's name would otherwise silently
    filter to zero results — name-based matching is out of scope."""
    if not isinstance(value, str):
        return None
    s = value.strip().strip("<>").lower()
    if not s or s in {"null", "none"}:
        return None
    if not _EMAIL_SHAPE.match(s):
        return None
    return s
