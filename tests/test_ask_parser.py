"""Tests for :mod:`emailsearch.ask.parser`.

Stubs the LLM HTTP call (mirrors test_summarize.py) so the parser's
behaviour is verified deterministically against fixed JSON responses.
We assert on the parsed output and on the prompt shape — the date
arithmetic itself is delegated to the LLM (it sees ``today_iso`` and
must do the math), so we don't try to verify epoch ranges in the
parser tests; that's checked end-to-end in the service tests.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from emailsearch.ask import parser as ask_parser
from emailsearch.ask.parser import ParsedAskRequest, parse_ask_question
from emailsearch.config import Settings


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | bytes) -> None:
        if isinstance(payload, bytes):
            self._raw = payload
        else:
            self._raw = json.dumps(payload).encode("utf-8")
        self._buf = io.BytesIO(self._raw)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._buf.read()


def _chat_response(text: str) -> dict[str, Any]:
    """One-shot /chat/completions response with the given content."""
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture()
def llm_enabled(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings with the LLM turned on but pointed at a fake endpoint.

    Both the parser AND its underlying ``_call_chat`` look up settings
    independently — patching both lookup sites keeps the test honest
    if either module's import structure changes.
    """
    s = Settings(
        llm_enabled=True,
        llm_base_url="http://127.0.0.1:4141/v1",
        llm_model="local-model",
        llm_timeout_s=5.0,
        ask_parse_max_tokens=200,
    )
    monkeypatch.setattr("emailsearch.ask.parser.get_settings", lambda: s)
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)
    return s


# ---------------------------------------------------------------------------
# Short-circuits
# ---------------------------------------------------------------------------


def test_empty_question_returns_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty question must not trigger an LLM call."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: pytest.fail("urlopen must not be called"),
    )
    out = parse_ask_question("   ")
    assert out == ParsedAskRequest(query="")


def test_disabled_llm_falls_back_to_raw_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM disabled (default in tests when not overridden) → no call,
    return question as-is for the search to use."""
    s = Settings(llm_enabled=False)
    monkeypatch.setattr("emailsearch.ask.parser.get_settings", lambda: s)
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called when llm_enabled=False")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    out = parse_ask_question("when is the garage day this month?")
    assert out == ParsedAskRequest(query="when is the garage day this month?")


# ---------------------------------------------------------------------------
# Happy path — well-formed JSON
# ---------------------------------------------------------------------------


def test_well_formed_json_parses_all_fields(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            _chat_response(
                '{"query":"Q3 budget","start_at":1750464000,"end_at":1750550400,'
                '"from_address":"alice@example.com"}'
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = parse_ask_question("yesterday's email from alice@example.com about Q3 budget")
    assert out == ParsedAskRequest(
        query="Q3 budget",
        start_at=1750464000,
        end_at=1750550400,
        from_address="alice@example.com",
    )
    # Prompt includes today's date (server local) so the LLM can resolve
    # relative-time phrases.
    prompt = captured["body"]["messages"][0]["content"]
    assert "Today's date" in prompt
    # Token cap is the ask-specific one, not the summarize default.
    assert captured["body"]["max_tokens"] == llm_enabled.ask_parse_max_tokens


def test_only_query_is_required(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No temporal / sender phrasing → all filter fields are null. The
    parser must accept this and surface query alone."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"rollout plan","start_at":null,"end_at":null,"from_address":null}'
            )
        ),
    )
    out = parse_ask_question("summarize the rollout plan")
    assert out == ParsedAskRequest(query="rollout plan")


# ---------------------------------------------------------------------------
# Defensive JSON handling
# ---------------------------------------------------------------------------


def test_strips_markdown_code_fence(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some local models wrap JSON in ``` ```json ... ``` ``` even when
    told not to. The parser must unwrap before decoding."""
    fenced = (
        "```json\n"
        '{"query":"rollout","start_at":null,"end_at":null,"from_address":null}\n'
        "```"
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(_chat_response(fenced)),
    )
    out = parse_ask_question("rollout?")
    assert out.query == "rollout"


def test_strips_unlabelled_code_fence(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fence without the `json` language tag must also be stripped."""
    fenced = (
        "```\n"
        '{"query":"rollout","start_at":null,"end_at":null,"from_address":null}\n'
        "```"
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(_chat_response(fenced)),
    )
    out = parse_ask_question("rollout?")
    assert out.query == "rollout"


def test_accepts_trailing_prose(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``raw_decode`` lets us peel off the leading JSON object even
    when the model appends commentary after it (a common chat-tuned
    failure mode)."""
    response = (
        '{"query":"budget","start_at":null,"end_at":null,"from_address":null}\n'
        "I hope this helps!"
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(_chat_response(response)),
    )
    out = parse_ask_question("budget?")
    assert out.query == "budget"


def test_malformed_json_falls_back_to_raw_question(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(_chat_response("not json at all")),
    )
    out = parse_ask_question("what about Q3?")
    assert out == ParsedAskRequest(query="what about Q3?")


def test_top_level_array_falls_back(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(_chat_response('["query","budget"]')),
    )
    out = parse_ask_question("budget?")
    assert out == ParsedAskRequest(query="budget?")


def test_missing_query_field_falls_back(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON object with no usable ``query`` falls back to the raw
    question — we never run an empty search."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response('{"start_at":null,"end_at":null,"from_address":null}')
        ),
    )
    out = parse_ask_question("garage day?")
    assert out.query == "garage day?"


def test_empty_query_string_falls_back(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response('{"query":"   ","start_at":null,"end_at":null,"from_address":null}')
        ),
    )
    out = parse_ask_question("garage day?")
    assert out.query == "garage day?"


def test_llm_returns_none_falls_back(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Underlying ``_call_chat`` returns None on any network failure;
    parser must surface the fallback request."""
    monkeypatch.setattr(ask_parser, "_call_chat", lambda **_kw: None)
    out = parse_ask_question("Q3 budget?")
    assert out == ParsedAskRequest(query="Q3 budget?")


# ---------------------------------------------------------------------------
# Field coercion
# ---------------------------------------------------------------------------


def test_string_epoch_coerced_to_int(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some models emit epoch values as strings. We accept those."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"x","start_at":"1750464000","end_at":null,"from_address":null}'
            )
        ),
    )
    out = parse_ask_question("x?")
    assert out.start_at == 1750464000


def test_zero_or_negative_epoch_treated_as_no_filter(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An epoch of 0 would silently match everything (real mail is
    post-1970). The coercer treats it as null."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"x","start_at":0,"end_at":-1,"from_address":null}'
            )
        ),
    )
    out = parse_ask_question("x?")
    assert out.start_at is None
    assert out.end_at is None


def test_string_null_in_from_address_treated_as_null(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some models emit the literal string ``"null"`` instead of the
    JSON ``null`` token. Treat it as missing."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"x","start_at":null,"end_at":null,"from_address":"null"}'
            )
        ),
    )
    out = parse_ask_question("x?")
    assert out.from_address is None


def test_non_email_string_in_from_address_rejected(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the question said only a name ('from alice'), the model
    must NOT populate from_address with a non-address — name-based
    matching is out of scope and would silently filter to zero hits."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"x","start_at":null,"end_at":null,"from_address":"alice"}'
            )
        ),
    )
    out = parse_ask_question("from alice")
    assert out.from_address is None


def test_email_address_lowercased(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Search filter is case-insensitive but we normalize at the
    boundary so the filter trace shows a canonical form."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"x","start_at":null,"end_at":null,"from_address":"Alice@Example.COM"}'
            )
        ),
    )
    out = parse_ask_question("x?")
    assert out.from_address == "alice@example.com"


def test_angle_brackets_stripped_from_email(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some models echo the question's literal '<alice@example.com>'
    formatting. Strip the brackets before validating."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"x","start_at":null,"end_at":null,'
                '"from_address":"<bob@example.com>"}'
            )
        ),
    )
    out = parse_ask_question("x?")
    assert out.from_address == "bob@example.com"


def test_bool_in_epoch_rejected(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``True`` is technically ``int(1)`` in Python — but it's an
    obvious type confusion and would resolve to 1970-01-01. Reject."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _FakeResponse(
            _chat_response(
                '{"query":"x","start_at":true,"end_at":null,"from_address":null}'
            )
        ),
    )
    out = parse_ask_question("x?")
    assert out.start_at is None
