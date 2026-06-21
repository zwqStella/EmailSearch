"""Tests for :mod:`emailsearch.ask.service`.

The agent has three moving parts: the parser, the search tool, and the
streaming synthesis call. We stub the parser and the streamer (so the
HTTP layer is not exercised) but use a real in-memory SQLite + FTS5
index for the search tool — same pattern as `test_search.py`. That
keeps the agent's plumbing honest end-to-end while staying offline /
deterministic.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from emailsearch.ask import service as ask_service
from emailsearch.ask.parser import ParsedAskRequest
from emailsearch.ask.service import AskEvent, ask_question
from emailsearch.db.connection import apply_schema, open_connection
from emailsearch.db.models import Chunk, EmailAddress, EmailRow
from emailsearch.db.repositories import insert_email_with_chunks


@pytest.fixture()
def conn():
    c = open_connection(":memory:")
    apply_schema(c)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def stub_search_llm_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the LLM hooks inside the search legs so the agent's own
    LLM calls (parser + synthesis) are the only thing under test.

    Without this, the semantic_fts / semantic_knn legs would try to
    reach a real LLM endpoint via ``distill_query`` / ``augment_query``
    when the agent calls ``search()``."""
    from emailsearch.search import service as search_service

    monkeypatch.setattr(search_service, "distill_query", lambda _q: None)
    monkeypatch.setattr(search_service, "augment_query", lambda _q: None)

    def fake_embed_query(_text: str) -> list[float]:
        # All zeros — vec0 still ranks chunks by distance from this,
        # which is fine for our tests (we don't assert on the embedding
        # leg's relative ordering).
        return [0.0] * 384

    monkeypatch.setattr(search_service, "embed_query", fake_embed_query)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_vec(seed: float, dim: int = 384) -> list[float]:
    return [seed + i * 0.0001 for i in range(dim)]


def _email(eid: str, subject: str, body: str, **kw: Any) -> EmailRow:
    return EmailRow(
        id=eid,
        subject=subject,
        from_address=kw.pop("from_address", "alice@example.com"),
        from_name=kw.pop("from_name", "Alice"),
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=kw.pop("received_at", int(time.time())),
        body_text=body,
        body_html=f"<p>{body}</p>",
        attachments=[],
        has_attachments=False,
        **kw,
    )


def _chunk(eid: str, idx: int, text: str, *, seed: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id=f"{eid}::body::{idx}",
        email_id=eid,
        source_type="body",
        source_name=None,
        chunk_index=idx,
        chunk_text=text,
        embedding=_seed_vec(seed),
    )


def _seed_corpus(conn) -> None:
    """Three emails the agent can find with single-word queries."""
    insert_email_with_chunks(
        conn,
        _email("m1", "Garage day announcement", "garage day is on Saturday April 5"),
        [_chunk("m1", 0, "garage day is on Saturday April 5", seed=0.01)],
    )
    insert_email_with_chunks(
        conn,
        _email("m2", "Q3 budget", "the Q3 budget review is on Friday"),
        [_chunk("m2", 0, "the Q3 budget review is on Friday", seed=0.02)],
    )
    insert_email_with_chunks(
        conn,
        _email("m3", "Random update", "company picnic this summer"),
        [_chunk("m3", 0, "company picnic this summer", seed=0.03)],
    )


def _collect(it: Iterator[AskEvent]) -> list[AskEvent]:
    return list(it)


def _stub_parser(monkeypatch: pytest.MonkeyPatch, result: ParsedAskRequest) -> None:
    """Replace the parser with a constant — the parser's own behaviour
    is covered in test_ask_parser.py, here we just want a known output
    so we can assert downstream wiring."""
    monkeypatch.setattr(ask_service, "parse_ask_question", lambda _q: result)


def _stub_stream(
    monkeypatch: pytest.MonkeyPatch,
    fragments: list[str] | Exception | None = None,
) -> list[dict[str, Any]]:
    """Replace the streaming synthesis call.

    - ``list[str]`` → yields each fragment in order (an empty list
      yields nothing — use this to simulate an LLM failure handled
      inside ``_call_chat_stream``).
    - ``Exception`` → raised on iteration start.
    - ``None`` (default) → yields a fixed two-fragment answer with one
      citation so the generic happy-path tests don't need to spell out
      a stub answer every time.

    Returns a list capturing every call's kwargs, so tests can assert
    on the prompt + token budget shape.
    """
    captured: list[dict[str, Any]] = []
    # Resolve the iteration source up-front so an explicit empty list
    # stays empty (``[] or DEFAULT`` evaluates to DEFAULT — a classic
    # falsy-list trap).
    if isinstance(fragments, Exception):
        to_yield: list[str] = []
    elif fragments is None:
        to_yield = ["The answer is X.", " See [1]."]
    else:
        to_yield = fragments

    def fake_stream(**kw: Any) -> Iterator[str]:
        captured.append(kw)
        if isinstance(fragments, Exception):
            raise fragments
        yield from to_yield

    monkeypatch.setattr(ask_service, "_call_chat_stream", fake_stream)
    return captured


def _stub_triage(
    monkeypatch: pytest.MonkeyPatch, response: str | None = "",
) -> list[dict[str, Any]]:
    """Replace the (non-streaming) triage LLM call.

    The triage hop calls :func:`_call_chat` (sibling to
    ``_call_chat_stream``) — patch that, NOT ``urlopen``, so the
    failure-routing inside ``_call_chat`` isn't part of the test.

    - ``response="..."`` → return that string verbatim from the
      patched ``_call_chat``.
    - ``response=None`` → simulate "LLM unreachable" — the triage
      helper falls back to "read all".

    Returns a list capturing every call's kwargs so tests can assert
    the triage prompt was built correctly.

    NOTE: triage is SKIPPED entirely when ``len(hits) <=
    ask_triage_limit``. Tests that want to exercise the triage path
    need to seed more than ``ask_triage_limit`` (default 3) emails.
    """
    captured: list[dict[str, Any]] = []

    def fake_call_chat(**kw: Any) -> str | None:
        captured.append(kw)
        return response

    monkeypatch.setattr(ask_service, "_call_chat", fake_call_chat)
    return captured


# ---------------------------------------------------------------------------
# Event ordering + happy path
# ---------------------------------------------------------------------------


def test_events_emit_in_documented_order(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The protocol is meta → parsed → sources → triage → answer_delta+ → done.
    The frontend's state machine assumes this order — break it and the
    inline `[N]` citations stop working mid-stream and the stage
    indicator labels get out of order."""
    _seed_corpus(conn)
    _stub_parser(monkeypatch, ParsedAskRequest(query="garage day"))
    _stub_stream(monkeypatch, ["The garage day is ", "April 5 [1]."])

    events = _collect(ask_question(conn, "when is the garage day this month?"))
    types = [e.type for e in events]
    assert types[0] == "meta"
    assert types[1] == "parsed"
    assert types[2] == "sources"
    assert types[3] == "triage"
    # Two answer_delta events (one per fragment from the stub).
    assert types[4:6] == ["answer_delta", "answer_delta"]
    assert types[-1] == "done"


def test_parsed_event_surfaces_filters_from_parser(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the parser extracts filters they must land on the `parsed`
    event so the frontend can render the filter pills."""
    _seed_corpus(conn)
    _stub_parser(
        monkeypatch,
        ParsedAskRequest(
            query="budget",
            start_at=1750464000,
            end_at=1750550400,
            from_address="alice@example.com",
        ),
    )
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "yesterday's email from alice about budget"))
    parsed = next(e for e in events if e.type == "parsed")
    assert parsed.data == {
        "query": "budget",
        "filters": {
            "start_at": 1750464000,
            "end_at": 1750550400,
            "from_address": "alice@example.com",
        },
    }


def test_sources_event_carries_search_hits(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent must surface its retrieved sources BEFORE answer
    streaming starts — otherwise inline `[N]` citations have nothing
    to resolve against in the UI."""
    _seed_corpus(conn)
    _stub_parser(monkeypatch, ParsedAskRequest(query="garage day"))
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "garage day?"))
    sources_idx = next(i for i, e in enumerate(events) if e.type == "sources")
    answer_idx = next(i for i, e in enumerate(events) if e.type == "answer_delta")
    assert sources_idx < answer_idx, "sources must arrive before answer_delta"

    sources_event = events[sources_idx]
    hits = sources_event.data["hits"]
    assert hits, "expected at least one hit for seeded 'garage day' email"
    assert any(h["email_id"] == "m1" for h in hits)


def test_synthesis_prompt_includes_numbered_sources_and_question(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The synthesis prompt is the contract between the agent and the
    LLM. It must (a) number the sources from 1 so `[N]` citations are
    valid, and (b) include the user's verbatim question."""
    _seed_corpus(conn)
    _stub_parser(monkeypatch, ParsedAskRequest(query="garage day"))
    captured = _stub_stream(monkeypatch, ["x"])

    _collect(ask_question(conn, "WHEN IS THE GARAGE DAY?"))
    assert len(captured) == 1
    prompt = captured[0]["prompt"]
    assert "[1]" in prompt
    assert "WHEN IS THE GARAGE DAY?" in prompt
    # The synthesis pass must use the answer token budget, not a
    # leaked default from elsewhere.
    from emailsearch.config import get_settings
    assert captured[0]["max_tokens"] == get_settings().ask_max_answer_tokens


def test_synthesis_prompt_includes_email_body_text(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the prompt must include each email's actual
    ``body_text`` so the LLM has the data to answer factual questions
    like "which dates can I pick?" against an email whose body is a
    table of dates.

    Previously we only sent an 80-char FTS snippet around the matched
    keyword, which surfaced the topic but never the specifics — the
    LLM correctly replied "I couldn't find that in your emails" to
    every factual question, even when the source email had the answer
    in plain sight.
    """
    # Body deliberately contains a fact ("Saturday April 5") that does
    # NOT overlap the query token, so the FTS snippet window would
    # historically miss it.
    insert_email_with_chunks(
        conn,
        _email("d1", "Garage day notice", "Garage day is on Saturday April 5 at 9am."),
        [_chunk("d1", 0, "Garage day Saturday April 5 9am", seed=0.01)],
    )
    _stub_parser(monkeypatch, ParsedAskRequest(query="garage day"))
    captured = _stub_stream(monkeypatch, ["x"])

    _collect(ask_question(conn, "what time is garage day?"))
    prompt = captured[0]["prompt"]
    # The body's full sentence — including the time-of-day that the
    # 80-char keyword window would never have captured — must land in
    # the prompt.
    assert "Saturday April 5" in prompt
    assert "9am" in prompt
    # Content header is the agent's, not the search leg's `Snippet:`.
    assert "Content:" in prompt


def test_long_email_gets_lions_share_of_budget(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the typical "1 long announcement + N short reminders"
    pattern must keep the long email's full content intact.

    With a flat per-hit cap, a 50KB HTML event announcement gets
    truncated to the same N chars as a 200-char reminder — and the
    date table that answers "which days can I pick?" lives past the
    truncation. Max-min fair-share gives short emails their full
    length and lets the long one absorb the remaining budget.

    Setup: one 30KB body containing a UNIQUE marker past the 5K mark,
    plus two short reminders. Keeping the hit count at 3 (the default
    ``ask_triage_limit``) means triage is SKIPPED — so this test
    isolates the fair-share allocator's behavior from the triage hop.
    With the default 60K total budget, the long email's marker
    should land in the prompt.
    """
    # 6KB of filler, then the marker, then more filler so the marker
    # is well past any small per-hit cap but well within the
    # fair-share allocation.
    filler = "Lots of preamble text. " * 300  # ~6900 chars
    marker = "PICKABLE_DATE_TOKEN_2026_06_15"
    long_body = filler + marker + " more content " * 1000  # ~21k chars
    insert_email_with_chunks(
        conn,
        _email("long1", "Event announcement", long_body),
        [_chunk("long1", 0, "Event announcement details", seed=0.01)],
    )
    # Two short reminders — same topic so they all surface.
    # Keeping the total at 3 (≤ ask_triage_limit) skips the triage hop
    # so the test focuses on the fair-share allocator.
    for i in range(2):
        insert_email_with_chunks(
            conn,
            _email(f"short{i}", f"Reminder {i}", "Don't forget the event."),
            [_chunk(f"short{i}", 0, "Event reminder", seed=0.01)],
        )

    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))

    def boom_triage(**_kw: Any) -> str | None:
        raise AssertionError("triage must NOT run for 3 hits at limit=3")

    monkeypatch.setattr(ask_service, "_call_chat", boom_triage)
    captured = _stub_stream(monkeypatch, ["x"])
    _collect(ask_question(conn, "when is the event?"))
    prompt = captured[0]["prompt"]
    assert marker in prompt, (
        "fair-share allocation should preserve the long email's full "
        "content when the other hits are short — the answer lives past "
        "any flat per-hit cap"
    )


# ---------------------------------------------------------------------------
# Token estimator + token-aware truncation
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty_is_zero() -> None:
    assert ask_service._estimate_tokens("") == 0


def test_estimate_tokens_ascii_uses_4_chars_per_token() -> None:
    """4 ASCII chars ~= 1 token (the OpenAI rule of thumb). Uses
    ceiling division so anything non-empty is at least 1 token."""
    assert ask_service._estimate_tokens("abcd") == 1
    assert ask_service._estimate_tokens("abcdefgh") == 2
    # Short strings still cost at least 1 token (ceiling).
    assert ask_service._estimate_tokens("a") == 1
    assert ask_service._estimate_tokens("ab") == 1


def test_estimate_tokens_cjk_is_one_token_per_char() -> None:
    """CJK characters are 1+ tokens each in GPT tokenizers — we use
    1/char as a conservative (under-estimating) baseline."""
    # 4 CJK chars → 4 tokens (not 1 like ASCII).
    assert ask_service._estimate_tokens("同乐日活") == 4
    # Two-char CJK phrases that bust the trigram FTS limit.
    assert ask_service._estimate_tokens("工会") == 2


def test_estimate_tokens_mixed_cjk_and_ascii() -> None:
    """Mixed strings (Chinese phrase + English translation) add the
    two scores. This is the typical Ask prompt content."""
    # 4 CJK chars + 8 ASCII chars = 4 + 2 = 6 tokens.
    assert ask_service._estimate_tokens("同乐日活abcdefgh") == 6


def test_truncate_to_tokens_returns_empty_for_zero_budget() -> None:
    assert ask_service._truncate_to_tokens("hello world", 0) == ""
    assert ask_service._truncate_to_tokens("hello world", -1) == ""


def test_truncate_to_tokens_returns_original_when_already_fits() -> None:
    """A short string with a generous budget passes through unchanged
    — no allocation, no slicing."""
    assert ask_service._truncate_to_tokens("abcd", 100) == "abcd"


def test_truncate_to_tokens_finds_largest_fitting_prefix() -> None:
    """Binary search must land on the LARGEST prefix that fits; no
    off-by-one that drops the last fitting character."""
    s = "a" * 100  # 25 tokens
    # Budget of 5 tokens → 20 ASCII chars fit.
    out = ask_service._truncate_to_tokens(s, 5)
    assert out == "a" * 20


def test_truncate_to_tokens_handles_cjk() -> None:
    """CJK truncation: 1 char = 1 token, so an N-token budget yields
    exactly N CJK chars (the simplest case)."""
    s = "同乐日活动通知"  # 7 CJK chars, 7 tokens
    assert ask_service._truncate_to_tokens(s, 3) == "同乐日"
    assert ask_service._truncate_to_tokens(s, 7) == s


def test_build_answer_prompt_respects_token_budget(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for HTTP 400 from copilot-api ('prompt token count
    13578 exceeds limit of 12288'). With a tight token budget, the
    assembled prompt's estimated tokens must stay near (not blow
    past) the configured ceiling — even on heavily-CJK content.

    Setup: a long CJK body (each char ≈ 1 token) at 5000 chars =
    5000 tokens. Budget = 1000 tokens. Without token-aware
    truncation, char-based slicing would let the full 5000 chars
    through (busting the limit by 4-5x).
    """
    long_cjk = "同" * 5000  # 5000 tokens by our estimator
    insert_email_with_chunks(
        conn,
        _email("cjk1", "同乐日通知", long_cjk),
        [_chunk("cjk1", 0, "同乐日", seed=0.01)],
    )
    _stub_parser(monkeypatch, ParsedAskRequest(query="同乐日"))
    captured = _stub_stream(monkeypatch, ["x"])

    # Override the prompt-token budget to 1000 (well under the
    # default 8000) so we can assert the cap was enforced.
    from emailsearch import config as _config

    s = _config.Settings(
        llm_enabled=False,  # disable triage LLM (1 hit anyway)
        ask_max_prompt_tokens=1000,
    )
    monkeypatch.setattr(ask_service, "get_settings", lambda: s)

    _collect(ask_question(conn, "今年同乐日有哪些选项?"))
    prompt = captured[0]["prompt"]
    # Allow a small overshoot for the prompt template + question
    # headers (~200 tokens). The CRITICAL property: we don't ship
    # the full 5000-token body when the budget is 1000.
    estimated = ask_service._estimate_tokens(prompt)
    assert estimated <= 1500, (
        f"prompt was {estimated} tokens — token-aware truncation "
        f"failed to enforce the {1000}-token budget for the body"
    )


# ---------------------------------------------------------------------------
# Fair-share allocator
# ---------------------------------------------------------------------------


def test_allocate_budgets_empty_returns_empty() -> None:
    assert ask_service._allocate_body_budgets([], 1000) == []


def test_allocate_budgets_zero_total_returns_zeros() -> None:
    assert ask_service._allocate_body_budgets([100, 200, 300], 0) == [0, 0, 0]


def test_allocate_budgets_everyone_fits_returns_full_lengths() -> None:
    """When the sum of lengths fits in the budget, no truncation
    happens — every item gets its full length."""
    out = ask_service._allocate_body_budgets([100, 200, 300], 10000)
    assert out == [100, 200, 300]


def test_allocate_budgets_nobody_fits_returns_even_share() -> None:
    """When every item exceeds the even share, the budget splits
    evenly. Integer division means total may be slightly under
    budget — that's intentional, we never overshoot."""
    out = ask_service._allocate_body_budgets([10000, 20000, 30000], 9000)
    assert out == [3000, 3000, 3000]
    assert sum(out) <= 9000


def test_allocate_budgets_small_items_donate_to_large() -> None:
    """Two tiny items and two huge ones: the tiny ones take their
    full length, and the huge ones split the leftover budget."""
    out = ask_service._allocate_body_budgets([50, 50, 10000, 10000], 1100)
    # Tiny ones get full length: 50 + 50 = 100.
    # Remaining budget 1000 splits between the two huge items: 500 each.
    assert out == [50, 50, 500, 500]
    assert sum(out) == 1100


def test_allocate_budgets_iterates_to_fixed_point() -> None:
    """Multi-stage donation: after the first pass donates from items
    that fit the initial share, the new (larger) share may let
    more items fit. Verify the allocator iterates."""
    # Budget 1000, lengths [100, 200, 400, 600]. Initial share = 250.
    # Items <= 250: [100, 200]. They take 300, leaving 700 for [400, 600].
    # New share = 700 / 2 = 350. Items <= 350: none (400 > 350).
    # Both get 350. Total = 100 + 200 + 350 + 350 = 1000.
    out = ask_service._allocate_body_budgets([100, 200, 400, 600], 1000)
    assert out == [100, 200, 350, 350]


# ---------------------------------------------------------------------------
# Empty inputs / empty hits
# ---------------------------------------------------------------------------


def test_empty_question_short_circuits(conn, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty question must not call the parser or the streamer — they
    would either crash or waste an LLM round-trip."""
    def boom_parser(_q: str) -> ParsedAskRequest:
        raise AssertionError("parser must not be called for empty question")

    def boom_stream(**_kw: Any) -> Iterator[str]:
        raise AssertionError("streamer must not be called for empty question")
        yield  # pragma: no cover

    monkeypatch.setattr(ask_service, "parse_ask_question", boom_parser)
    monkeypatch.setattr(ask_service, "_call_chat_stream", boom_stream)

    events = _collect(ask_question(conn, "   "))
    assert [e.type for e in events] == ["meta", "done"]


def test_no_hits_still_streams_an_answer(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the search returns nothing the synthesis prompt has an
    explicit "say you couldn't find anything" rule. The agent must
    still call the streamer (sources event has an empty list) so the
    LLM emits a graceful fallback.

    Force the zero-hit path with a from_address filter that doesn't
    match any seeded sender — a query that whiffs on every leg is
    surprisingly hard to construct, since vec0 always returns top-K
    nearest neighbours regardless of semantic match.
    """
    _seed_corpus(conn)
    _stub_parser(
        monkeypatch,
        ParsedAskRequest(
            query="garage day",
            from_address="nobody@nowhere.invalid",
        ),
    )
    captured = _stub_stream(
        monkeypatch, ["I couldn't find anything about that."]
    )

    events = _collect(ask_question(conn, "what about a topic that isn't there?"))
    sources_event = next(e for e in events if e.type == "sources")
    assert sources_event.data["hits"] == []
    # Streamer was called even with zero hits.
    assert len(captured) == 1
    # The fallback prompt template kicks in when hits are empty.
    assert "No emails matched" in captured[0]["prompt"]


# ---------------------------------------------------------------------------
# Failure surfaces — every step must emit an error event, never raise
# ---------------------------------------------------------------------------


def test_parser_exception_surfaces_as_error_event(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parser raising must NOT bubble out of the generator — the HTTP
    layer would convert it to a 500, killing the stream connection
    instead of letting the client render the error inline."""
    def boom(_q: str) -> ParsedAskRequest:
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(ask_service, "parse_ask_question", boom)

    events = _collect(ask_question(conn, "anything"))
    assert events[-1].type == "error"
    assert "parser exploded" in events[-1].data["message"]
    # No `done` after an error — the stream terminates on the error.
    assert "done" not in [e.type for e in events]


def test_streamer_failure_surfaces_as_error_event(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the synthesis call itself raises (proxy died mid-stream),
    the agent emits an error event with the sources still visible."""
    _seed_corpus(conn)
    _stub_parser(monkeypatch, ParsedAskRequest(query="garage day"))
    _stub_stream(monkeypatch, RuntimeError("LLM proxy disconnected"))

    events = _collect(ask_question(conn, "garage day?"))
    types = [e.type for e in events]
    # Sources land before the failure.
    assert "sources" in types
    assert types[-1] == "error"
    assert "LLM proxy disconnected" in events[-1].data["message"]


def test_streamer_yielding_nothing_still_emits_done(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A streamer that returns immediately (LLM disabled / network
    glitch handled inside ``_call_chat_stream``) yields zero deltas.
    The agent must still emit `done` so the UI's spinner stops."""
    _seed_corpus(conn)
    _stub_parser(monkeypatch, ParsedAskRequest(query="garage day"))
    _stub_stream(monkeypatch, [])

    events = _collect(ask_question(conn, "garage day?"))
    types = [e.type for e in events]
    assert "answer_delta" not in types
    assert types[-1] == "done"


# ---------------------------------------------------------------------------
# Tool wiring — filters flow through to search()
# ---------------------------------------------------------------------------


def test_parser_filters_reach_search_tool(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A from_address filter from the parser must narrow the search
    candidate set. We seed two emails with different senders and assert
    the filter is honored."""
    insert_email_with_chunks(
        conn,
        _email("a1", "garage day notice", "this Saturday", from_address="alice@example.com"),
        [_chunk("a1", 0, "garage day this Saturday", seed=0.01)],
    )
    insert_email_with_chunks(
        conn,
        _email("b1", "garage day reminder", "also this Saturday", from_address="bob@example.com"),
        [_chunk("b1", 0, "garage day this Saturday", seed=0.01)],
    )
    _stub_parser(
        monkeypatch,
        ParsedAskRequest(query="garage day", from_address="alice@example.com"),
    )
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "garage day from alice@example.com?"))
    sources_event = next(e for e in events if e.type == "sources")
    ids = {h["email_id"] for h in sources_event.data["hits"]}
    assert "a1" in ids
    assert "b1" not in ids, "from_address filter should exclude bob's email"


# ---------------------------------------------------------------------------
# Triage hop — narrow the synthesis prompt to LLM-picked relevant emails
# ---------------------------------------------------------------------------


def _seed_n_emails(conn, n: int) -> None:
    """Helper: seed N findable emails so triage has > ask_triage_limit
    hits to choose from (default limit is 3)."""
    for i in range(n):
        insert_email_with_chunks(
            conn,
            _email(f"e{i}", f"Event {i} announcement", f"event {i} details"),
            [_chunk(f"e{i}", 0, f"event {i} details", seed=0.01 + i * 0.001)],
        )


def test_triage_skipped_when_hits_at_or_below_limit(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With 3 hits and ``ask_triage_limit=3``, narrowing buys nothing.
    The triage LLM hop MUST be skipped — otherwise we waste a
    round-trip + tokens against a constrained proxy."""
    _seed_corpus(conn)  # exactly 3 emails
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))

    def boom_triage(**_kw: Any) -> str | None:
        raise AssertionError("triage must NOT call _call_chat when hits <= limit")

    monkeypatch.setattr(ask_service, "_call_chat", boom_triage)
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    assert triage_event.data["triaged"] is False
    # selected_indexes lists EVERY retrieved hit when triage skipped.
    sources_event = next(e for e in events if e.type == "sources")
    n = len(sources_event.data["hits"])
    assert triage_event.data["selected_indexes"] == list(range(1, n + 1))


def test_triage_picks_subset_for_synthesis(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When triage runs and returns specific indexes, only those
    emails' bodies appear in the synthesis prompt — and they keep
    their ORIGINAL 1-based index so citations align with the
    sources event."""
    _seed_n_emails(conn, 6)  # 6 hits > triage_limit=3 → triage fires
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))
    _stub_triage(monkeypatch, "2,4")  # pick #2 and #4
    captured = _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    assert triage_event.data["triaged"] is True
    assert triage_event.data["selected_indexes"] == [2, 4]

    prompt = captured[0]["prompt"]
    # Per-source blocks start with "\n[N] Subject:" — use that to
    # avoid false positives from the citation-format example inside
    # the prompt template's RULES section.
    assert "[2] Subject:" in prompt
    assert "[4] Subject:" in prompt
    # Non-selected hits must NOT appear as source blocks.
    assert "[1] Subject:" not in prompt
    assert "[3] Subject:" not in prompt
    assert "[5] Subject:" not in prompt
    assert "[6] Subject:" not in prompt


def test_triage_response_with_prose_still_parses(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chat-tuned models occasionally append commentary after the
    indexes ('1,3 — these look most relevant'). The triage parser
    must extract the digits regardless."""
    _seed_n_emails(conn, 5)
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))
    _stub_triage(monkeypatch, "1, 3 — these are most relevant")
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    assert triage_event.data["selected_indexes"] == [1, 3]


def test_triage_none_response_falls_back_to_top_n_by_score(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When triage says NONE (no email relevant), we still send the
    top ``ask_triage_limit`` to synthesis — better to let the
    synthesizer emit the "couldn't find that" fallback than to
    silently skip the answer step."""
    _seed_n_emails(conn, 6)
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))
    _stub_triage(monkeypatch, "NONE")
    captured = _stub_stream(monkeypatch, ["I couldn't find that."])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    # NONE → fall back to first 3 by score (matches ask_triage_limit
    # default). The fallback IS still flagged as triaged so the UI
    # shows "Read 3 of 6" rather than "Read 6 of 6".
    assert triage_event.data["triaged"] is True
    assert triage_event.data["selected_indexes"] == [1, 2, 3]
    # Synthesis still gets called.
    assert len(captured) == 1


def test_triage_out_of_range_indexes_skipped(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triage occasionally hallucinates indexes past the hit count.
    The parser must drop them rather than crash synthesis."""
    _seed_n_emails(conn, 4)
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))
    # 99 is out-of-range; 2 is valid.
    _stub_triage(monkeypatch, "99, 2, 100")
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    assert triage_event.data["selected_indexes"] == [2]


def test_triage_garbage_response_falls_back_to_top_n(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completely unparseable triage output (no digits at all) → empty
    selection → same fallback as NONE."""
    _seed_n_emails(conn, 5)
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))
    _stub_triage(monkeypatch, "completely garbled response")
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    assert triage_event.data["selected_indexes"] == [1, 2, 3]


def test_triage_llm_unreachable_reads_all(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``_call_chat`` returns None (LLM unreachable), triage
    must NOT silently drop sources — read everything and let the
    synthesizer cope. The UI shows "Read 6 of 6" so the user knows
    we didn't narrow."""
    _seed_n_emails(conn, 6)
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))
    _stub_triage(monkeypatch, None)
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    assert triage_event.data["selected_indexes"] == [1, 2, 3, 4, 5, 6]


def test_triage_caps_at_ask_triage_limit(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if the LLM picks 5 indexes, the parser must cap at
    ``ask_triage_limit`` (default 3) to bound the synthesis prompt
    size."""
    _seed_n_emails(conn, 8)
    _stub_parser(monkeypatch, ParsedAskRequest(query="event"))
    _stub_triage(monkeypatch, "1,2,3,4,5")
    _stub_stream(monkeypatch, ["x"])

    events = _collect(ask_question(conn, "any event?"))
    triage_event = next(e for e in events if e.type == "triage")
    assert triage_event.data["selected_indexes"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Triage parser unit tests (cheap, no in-memory DB)
# ---------------------------------------------------------------------------


def test_parse_triage_response_comma_separated() -> None:
    out = ask_service._parse_triage_response("1,3,5", hit_count=8, max_pick=5)
    assert out == [1, 3, 5]


def test_parse_triage_response_preserves_order() -> None:
    """'Most relevant first' must survive parsing."""
    out = ask_service._parse_triage_response("3,1,5", hit_count=8, max_pick=5)
    assert out == [3, 1, 5]


def test_parse_triage_response_deduplicates() -> None:
    out = ask_service._parse_triage_response("1,3,1,5,3", hit_count=8, max_pick=5)
    assert out == [1, 3, 5]


def test_parse_triage_response_none_returns_empty() -> None:
    assert ask_service._parse_triage_response("NONE", hit_count=8, max_pick=5) == []
    assert ask_service._parse_triage_response("none", hit_count=8, max_pick=5) == []
    # Even with surrounding prose — anything starting with NONE counts.
    assert ask_service._parse_triage_response(
        "NONE - nothing matched", hit_count=8, max_pick=5
    ) == []
