"""Ask agent: parse → search tool → streaming synthesis.

A single-step RAG loop framed as a one-tool agent:

1. :func:`parse_ask_question` turns the user's natural-language question
   into a :class:`ParsedAskRequest` (distilled query + hard filters).
2. :func:`search_emails_tool` is the agent's one tool — a thin façade over
   ``search.service.search()`` that fans out all three legs.
3. The retrieved hits are formatted into a numbered grounding prompt and
   streamed through :func:`_call_chat_stream`; each content fragment
   becomes one ``answer_delta`` event.

Event protocol — yielded in this order:

  - ``meta``         : ``{question, mode, limit}``
  - ``parsed``       : ``{query, filters: {start_at, end_at, from_address}}``
  - ``sources``      : ``{hits: [...]}``  (BEFORE answer streaming so
    inline ``[N]`` clicks work mid-stream)
  - ``triage``       : ``{selected_indexes, triaged}``
  - ``answer_delta`` : ``{text}``  (zero or more)
  - ``done``         : ``{duration_ms}``
  - ``error``        : ``{message}``  (one-of, ends the stream early)

The agent NEVER raises — every step is wrapped so the HTTP layer can
just iterate and forward events.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from emailsearch.ask.parser import ParsedAskRequest, parse_ask_question
from emailsearch.config import get_settings
from emailsearch.db.models import EmailRow
from emailsearch.db.repositories import get_emails_by_ids
from emailsearch.search.service import (
    SearchFilters,
    SearchHit,
    SearchMode,
    search,
)
from emailsearch.summarize.client import _call_chat, _call_chat_stream
from emailsearch.util import is_cjk_char

log = logging.getLogger(__name__)

# Discriminator on the event union — frontend keys off ``type``.
AskEventType = Literal[
    "meta", "parsed", "sources", "triage", "answer_delta", "done", "error"
]


class AskEvent(BaseModel):
    """One event in the agent's output stream. Loose-typed payload so the
    wire format can evolve without breaking this model — the frontend's
    TS types are the schema of record (see ``frontend/src/api/types.ts``).
    """

    type: AskEventType
    data: dict[str, Any]


# Synthesis prompt. Three rules carry the weight:
#   1. "Use ONLY the listed emails" — without this the model extrapolates
#      from world knowledge and hallucinates confident wrong answers.
#   2. Cite every claim with ``[N]`` — the frontend renders markers as
#      clickable buttons that open the email.
#   3. "Reply in the same language as the question" — without it CJK
#      questions reliably get English answers from English-trained models.
_ANSWER_PROMPT_TEMPLATE = (
    "You are answering a question about the user's email inbox.\n"
    "\n"
    "RULES (must follow exactly):\n"
    "  1. Use ONLY information from the SOURCES listed below. Do NOT\n"
    "     use outside knowledge or guess. If the sources don't answer\n"
    "     the question, reply with a short sentence saying you couldn't\n"
    "     find the answer in the user's emails (translated into the\n"
    "     same language as the question). Do NOT speculate.\n"
    "  2. Cite the sources you used with inline bracket markers like\n"
    "     [1], [2], etc., placed RIGHT AFTER the claim they support.\n"
    "     Cite multiple sources for one claim as [1][2] (no spaces, no\n"
    "     commas). Only cite source numbers that actually appear below.\n"
    "  3. Reply in the SAME LANGUAGE as the question. Be concise —\n"
    "     ideally 1-3 sentences. Skip greetings and preamble.\n"
    "\n"
    "SOURCES (each block starts with its citation number):\n"
    "{sources_block}\n"
    "\n"
    "QUESTION: {question}\n"
    "\n"
    "ANSWER:"
)


# Triage prompt: picks which retrieved hits to read fully for synthesis.
# Output is strictly machine-parseable (comma-separated 1-based indexes
# or "NONE"); we hand-parse defensively so accidental prose gets stripped.
_TRIAGE_PROMPT_TEMPLATE = (
    "You are a search-result triager. Below is the user's question and\n"
    "a numbered list of email previews returned by an email search.\n"
    "Your job: pick up to {max_pick} emails most likely to contain the\n"
    "ANSWER to the question. Output ONLY the comma-separated 1-based\n"
    "indexes of the chosen emails, most relevant first. If NO email\n"
    "looks relevant, output the single word: NONE\n"
    "\n"
    "Do not output anything else — no prose, no explanation, no\n"
    "Markdown. Just the indexes (e.g. '3,1,5') or 'NONE'.\n"
    "\n"
    "QUESTION: {question}\n"
    "\n"
    "EMAIL PREVIEWS:\n"
    "{previews_block}\n"
    "\n"
    "INDEXES:"
)


def search_emails_tool(
    conn: sqlite3.Connection,
    query: str,
    *,
    mode: SearchMode,
    limit: int,
    filters: SearchFilters,
) -> list[SearchHit]:
    """The agent's one tool: search the email index, return ranked hits.

    Thin façade over :func:`search.service.search` so the call site
    reads as a tool invocation. ``hybrid`` is the default for Ask
    because the agent has no a priori knowledge of whether the question
    is keyword-friendly or paraphrase-heavy.
    """
    log.info(
        "ask tool: search_emails(query=%r, mode=%s, limit=%d, filters_active=%s)",
        query, mode, limit, filters.is_active(),
    )
    return search(conn, query, mode=mode, limit=limit, filters=filters).hits


def ask_question(
    conn: sqlite3.Connection,
    question: str,
    *,
    mode: SearchMode = "hybrid",
    limit: int | None = None,
) -> Iterator[AskEvent]:
    """Run the agent and yield events as each step completes.

    The order of yields IS the protocol. We never raise: every step is
    wrapped in try/except and emits an ``error`` event on failure.
    """
    started = time.perf_counter()
    settings = get_settings()
    if limit is None:
        limit = settings.ask_retrieval_limit

    question = (question or "").strip()
    yield AskEvent(
        type="meta",
        data={"question": question, "mode": mode, "limit": limit},
    )

    if not question:
        yield AskEvent(type="done", data={"duration_ms": _ms_since(started)})
        return

    # 1) parse — single LLM hop, extracts query + filters.
    try:
        parsed = parse_ask_question(question)
    except Exception as exc:
        log.exception("ask: parser raised — surfacing as error event")
        yield AskEvent(
            type="error",
            data={"message": f"parse failed: {type(exc).__name__}: {exc}"},
        )
        return
    yield AskEvent(
        type="parsed",
        data={
            "query": parsed.query,
            "filters": {
                "start_at": parsed.start_at,
                "end_at": parsed.end_at,
                "from_address": parsed.from_address,
            },
        },
    )

    # 2) search — fan out all three legs and merge.
    try:
        hits = search_emails_tool(
            conn,
            parsed.query,
            mode=mode,
            limit=limit,
            filters=_to_filters(parsed),
        )
    except Exception as exc:
        log.exception("ask: search tool raised — surfacing as error event")
        yield AskEvent(
            type="error",
            data={"message": f"search failed: {type(exc).__name__}: {exc}"},
        )
        return
    yield AskEvent(
        type="sources",
        data={"hits": [h.model_dump() for h in hits]},
    )

    # 3) triage — re-fetch full ``EmailRow`` per hit, then ask the LLM
    #    which subset to read in full. Triage trades one small LLM call
    #    for a 60-70% shrink of the synthesis prompt. Skipped when
    #    ``len(hits) <= ask_triage_limit`` or when the LLM is disabled.
    #
    # ``SearchHit.snippet`` is only an 80-char keyword window — too thin
    # to answer specific factual questions. The full body lookup is
    # cheap (already-indexed ids).
    emails_by_id: dict[str, EmailRow] = {}
    if hits:
        try:
            emails_by_id = get_emails_by_ids(conn, [h.email_id for h in hits])
        except Exception as exc:
            log.exception("ask: hydration raised — surfacing as error event")
            yield AskEvent(
                type="error",
                data={"message": f"email lookup failed: {type(exc).__name__}: {exc}"},
            )
            return

    selected_indexes: list[int] | None = None
    if hits and len(hits) > settings.ask_triage_limit:
        try:
            picked = _triage_emails(
                question=question,
                hits=hits,
                emails_by_id=emails_by_id,
                max_pick=settings.ask_triage_limit,
            )
        except Exception as exc:
            log.exception("ask: triage raised — surfacing as error event")
            yield AskEvent(
                type="error",
                data={"message": f"triage failed: {type(exc).__name__}: {exc}"},
            )
            return
        if not picked:
            # LLM said NONE relevant. Read the top-N by score anyway so
            # synthesis has SOMETHING to ground on — let the synthesis
            # LLM emit the "couldn't find that" fallback.
            picked = list(range(1, min(settings.ask_triage_limit, len(hits)) + 1))
            log.info(
                "ask triage: NONE response — falling back to top-%d by score",
                len(picked),
            )
        selected_indexes = picked

    yield AskEvent(
        type="triage",
        data={
            "selected_indexes": (
                selected_indexes
                if selected_indexes is not None
                else list(range(1, len(hits) + 1))
            ),
            "triaged": selected_indexes is not None,
        },
    )

    # 4) synthesize — stream the answer. Zero deltas is a valid outcome
    #    (``_call_chat_stream`` yields nothing on LLM disabled/network/
    #    malformed SSE); the frontend simply shows no answer with the
    #    sources still visible.
    prompt = _build_answer_prompt(
        question=question,
        hits=hits,
        emails_by_id=emails_by_id,
        max_prompt_tokens=settings.ask_max_prompt_tokens,
        selected_indexes=selected_indexes,
    )
    try:
        any_delta = False
        for fragment in _call_chat_stream(
            prompt=prompt,
            max_tokens=settings.ask_max_answer_tokens,
            log_label=f"ask_answer({question[:40]!r})",
        ):
            any_delta = True
            yield AskEvent(type="answer_delta", data={"text": fragment})
        if not any_delta:
            log.info("ask: synthesis stream yielded zero fragments")
    except Exception as exc:
        log.exception("ask: synthesis raised — surfacing as error event")
        yield AskEvent(
            type="error",
            data={"message": f"synthesis failed: {type(exc).__name__}: {exc}"},
        )
        return

    yield AskEvent(
        type="done",
        data={"duration_ms": _ms_since(started)},
    )


def _to_filters(parsed: ParsedAskRequest) -> SearchFilters:
    """Project the parser's flat fields into a :class:`SearchFilters`."""
    return SearchFilters(
        start_at=parsed.start_at,
        end_at=parsed.end_at,
        from_address=parsed.from_address,
        folder_id=None,  # folder inference deferred — see ask/parser.py
    )


def _triage_emails(
    *,
    question: str,
    hits: list[SearchHit],
    emails_by_id: dict[str, EmailRow],
    max_pick: int,
    preview_chars: int = 200,
) -> list[int]:
    """Ask the LLM which hits to read FULLY for synthesis.

    Returns a list of 1-based hit indexes (into ``hits``), up to
    ``max_pick``. Empty list means "LLM said NONE relevant" — the caller
    falls back to "read top-N by search score" so the user still gets an
    answer even when the LLM is over-conservative.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return list(range(1, len(hits) + 1))

    previews = _build_triage_previews(
        hits=hits, emails_by_id=emails_by_id, preview_chars=preview_chars,
    )
    prompt = _TRIAGE_PROMPT_TEMPLATE.format(
        max_pick=max_pick,
        question=question,
        previews_block=previews,
    )
    log.info(
        "ask triage: asking LLM to pick %d of %d hit(s) (preview=%d chars/hit)",
        max_pick, len(hits), preview_chars,
    )
    response = _call_chat(
        prompt=prompt,
        max_tokens=settings.ask_triage_max_tokens,
        log_label=f"ask_triage({question[:40]!r})",
    )
    if response is None:
        # LLM unreachable / failed — treat as "read all" since we have
        # no signal for narrowing. Caller's truncation logic still
        # applies.
        log.info("ask triage: LLM returned None — falling back to read all")
        return list(range(1, len(hits) + 1))

    selected = _parse_triage_response(
        response, hit_count=len(hits), max_pick=max_pick,
    )
    log.info(
        "ask triage: response=%r → selected indexes %s (of %d)",
        response.strip()[:80], selected, len(hits),
    )
    return selected


def _build_triage_previews(
    *,
    hits: list[SearchHit],
    emails_by_id: dict[str, EmailRow],
    preview_chars: int,
) -> str:
    """Compact per-hit metadata (subject + summary + body glimpse) for the
    triage prompt — small enough to stay under ~2K tokens for 8 hits."""
    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        summary = (hit.summary or "").strip()
        if len(summary) > 200:
            summary = summary[:200] + "…"
        email = emails_by_id.get(hit.email_id)
        if email is not None and email.body_text:
            preview = _collapse_blank_lines(email.body_text.strip())[:preview_chars]
        else:
            preview = (hit.snippet or "").strip()[:preview_chars]
        preview = preview.replace("\n", " ").strip()
        if preview and len(preview) >= preview_chars:
            preview += "…"

        lines = [f"[{idx}] Subject: {hit.subject or '(no subject)'}"]
        if hit.from_name:
            lines.append(f"    From: {hit.from_name}")
        elif hit.from_address:
            lines.append(f"    From: {hit.from_address}")
        if summary:
            lines.append(f"    Summary: {summary}")
        if preview:
            lines.append(f"    Preview: {preview}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_triage_response(
    response: str, *, hit_count: int, max_pick: int,
) -> list[int]:
    """Pull a list of 1-based indexes out of the LLM's triage reply.

    Accepts ``"1,3,5"``, ``"3, 1, 5"``, ``"NONE"``, or any of these
    with prose appended. Returns ``[]`` for ``NONE`` or completely
    garbled output (caller falls back to top-N). Drops out-of-range
    indexes silently and de-duplicates while preserving order.
    """
    text = response.strip()
    if text.upper().startswith("NONE"):
        return []

    # Greedy digit extraction handles any layout (commas, spaces, prose).
    raw = re.findall(r"\d+", text)
    seen: set[int] = set()
    out: list[int] = []
    for s in raw:
        n = int(s)
        if 1 <= n <= hit_count and n not in seen:
            seen.add(n)
            out.append(n)
            if len(out) >= max_pick:
                break
    return out


def _build_answer_prompt(
    *,
    question: str,
    hits: list[SearchHit],
    emails_by_id: dict[str, EmailRow],
    max_prompt_tokens: int,
    selected_indexes: list[int] | None = None,
) -> str:
    """Format the numbered sources block + the question.

    Numbering is 1-based to match the ``[N]`` citation convention.
    When ``selected_indexes`` is provided (triage path), only those hits
    are rendered but the rendered numbers stay their ORIGINAL 1-based
    positions so inline citations resolve against the frontend's full
    sources list. Body+attachment text is allocated via max-min
    fair-share over a TOKEN budget (CJK-aware — char-based allocation
    busts model context limits on bilingual corpora).
    """
    if not hits:
        return _ANSWER_PROMPT_TEMPLATE.format(
            sources_block="  (No emails matched the search — answer that you couldn't find anything.)",
            question=question,
        )

    if selected_indexes is None:
        rendering = list(enumerate(hits, start=1))
    else:
        rendering = [
            (i, hits[i - 1]) for i in selected_indexes
            if 1 <= i <= len(hits)
        ]
    if not rendering:
        return _ANSWER_PROMPT_TEMPLATE.format(
            sources_block="  (No emails matched the search — answer that you couldn't find anything.)",
            question=question,
        )

    full_bodies = [
        _hit_body_candidate(hit=hit, email=emails_by_id.get(hit.email_id))
        for _, hit in rendering
    ]
    full_token_counts = [_estimate_tokens(b) for b in full_bodies]
    budgets = _allocate_body_budgets(full_token_counts, max_prompt_tokens)
    log.info(
        "ask: prompt budget %d tokens across %d rendered hit(s) "
        "(of %d retrieved); full_tokens=%s allocated_tokens=%s",
        max_prompt_tokens, len(rendering), len(hits),
        full_token_counts, budgets,
    )

    blocks: list[str] = []
    for (idx, hit), full_body, budget in zip(
        rendering, full_bodies, budgets, strict=True,
    ):
        date_str = datetime.fromtimestamp(hit.received_at).strftime("%Y-%m-%d %H:%M")
        sender = (
            f"{hit.from_name} <{hit.from_address}>"
            if hit.from_name else (hit.from_address or "(unknown)")
        )

        summary = (hit.summary or "").strip()
        if summary and len(summary) > 240:
            summary = summary[:240] + "…"

        if _estimate_tokens(full_body) <= budget:
            body_block = full_body
        else:
            body_block = _truncate_to_tokens(full_body, budget) + "…"

        lines = [
            f"[{idx}] Subject: {hit.subject or '(no subject)'}",
            f"    From:    {sender}",
            f"    Date:    {date_str}",
        ]
        if summary:
            lines.append(f"    Summary: {summary}")
        if body_block:
            # Indent continuation lines so the block visually attaches
            # to its source number.
            indented = body_block.replace("\n", "\n             ")
            lines.append(f"    Content: {indented}")
        blocks.append("\n".join(lines))

    return _ANSWER_PROMPT_TEMPLATE.format(
        sources_block="\n\n".join(blocks),
        question=question,
    )


def _hit_body_candidate(
    *, hit: SearchHit, email: EmailRow | None,
) -> str:
    """Assemble the FULL body + attachment text for one hit (no truncation
    — that's the caller's job via :func:`_allocate_body_budgets`).

    Falls back to ``hit.snippet`` (sans ``<mark>`` tags) when the
    :class:`EmailRow` is missing. Attachments get a clear
    ``--- Attachment: <name> ---`` delimiter as a weak prompt-injection
    guard. Runs of blank lines are collapsed to save tokens.
    """
    if email is None:
        return (hit.snippet or "").replace("<mark>", "").replace("</mark>", "").strip()

    body = _collapse_blank_lines((email.body_text or "").strip())
    parts: list[str] = []
    if body:
        parts.append(body)
    for att in email.attachments:
        text = (att.extracted_text or "").strip()
        if not text:
            continue
        parts.append(f"\n--- Attachment: {att.name} ---\n{_collapse_blank_lines(text)}")
    return "".join(parts)


def _estimate_tokens(s: str) -> int:
    """Rough CJK-aware token estimate (conservative over-estimate).

    GPT-class tokenizers use ~1-2 tokens per CJK char and ~4 chars per
    token for ASCII; we use 1 token/CJK char and 1 token/4 ASCII chars.
    Drives :func:`_allocate_body_budgets` (token-based fair-share) and
    :func:`_truncate_to_tokens` (token-aware slicing).
    """
    if not s:
        return 0
    cjk = 0
    other = 0
    for ch in s:
        if is_cjk_char(ch):
            cjk += 1
        else:
            other += 1
    # ``ceil(other / 4)`` via integer rounding.
    return cjk + (other + 3) // 4


def _truncate_to_tokens(s: str, token_budget: int) -> str:
    """Longest prefix of ``s`` whose estimated tokens fit ``budget``.

    Binary search on slice length — total cost O(n log n) since
    :func:`_estimate_tokens` is O(n). Returns ``""`` for non-positive
    budgets; returns ``s`` unchanged when it already fits.
    """
    if not s or token_budget <= 0:
        return ""
    if _estimate_tokens(s) <= token_budget:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _estimate_tokens(s[:mid]) <= token_budget:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo]


def _allocate_body_budgets(
    lengths: list[int], total_budget: int,
) -> list[int]:
    """Max-min fair-share allocation of ``total_budget`` across items.

    Each item gets the smaller of (its full length, the even share of
    the remaining budget). Items that fit within the share leave their
    unused budget for larger items; iterates to a fixed point.
    Unit-agnostic; called with TOKEN counts. Returns budgets summing to
    ``<= total_budget``.

    Why fair-share here: a typical Ask question retrieves one long
    detailed email plus several short reminders. A flat per-hit cap
    truncates the long one past the answer or wastes budget on the
    short ones; fair-share gives the long email most of the budget when
    the short ones don't need it.
    """
    n = len(lengths)
    if n == 0:
        return []
    if total_budget <= 0:
        return [0] * n
    budgets = [0] * n
    remaining = total_budget
    pending: set[int] = set(range(n))
    while pending:
        share = remaining // len(pending)
        if share <= 0:
            break
        small = [i for i in pending if lengths[i] <= share]
        if not small:
            # Everyone left exceeds the even share — they all get it.
            for i in pending:
                budgets[i] = share
            break
        for i in small:
            budgets[i] = lengths[i]
            remaining -= lengths[i]
            pending.discard(i)
    return budgets


def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive newlines down to two. Saves prompt tokens
    on HTML-derived bodies (Outlook quoted threads) without losing the
    paragraph structure the model needs."""
    if not text:
        return ""
    out: list[str] = []
    blank_run = 0
    for line in text.split("\n"):
        if line.strip():
            blank_run = 0
            out.append(line)
        else:
            blank_run += 1
            if blank_run <= 1:
                out.append(line)
    return "\n".join(out)


def _ms_since(start: float) -> int:
    """Wall-clock milliseconds since ``start`` (from ``time.perf_counter``)."""
    return int((time.perf_counter() - start) * 1000)
