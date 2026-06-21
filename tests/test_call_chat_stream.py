"""Tests for :func:`emailsearch.summarize.client._call_chat_stream`.

The OpenAI SSE wire format has a few sharp edges:
  - ``data: {...}\\n\\n`` frames may arrive in arbitrary chunk sizes
    (one frame per read, multiple per read, or one frame split across
    several reads).
  - ``data: [DONE]\\n\\n`` is the explicit termination sentinel — some
    proxies close the socket after, some keep it open.
  - Non-``data:`` lines (``event:`` / ``id:`` / comments) must be
    ignored, not crash the parser.
  - Malformed JSON in a single frame must skip that frame, not abort
    the whole stream.

Each test isolates one of these behaviors by stubbing ``urlopen`` with
a controllable byte source.
"""

from __future__ import annotations

import io
import urllib.error
from collections.abc import Iterator

import pytest

from emailsearch.config import Settings
from emailsearch.summarize import client as summarize_client


@pytest.fixture()
def llm_enabled(monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(
        llm_enabled=True,
        llm_base_url="http://127.0.0.1:4141/v1",
        llm_model="local-model",
        llm_timeout_s=5.0,
        ask_max_answer_tokens=600,
    )
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)
    return s


class _ChunkedResponse:
    """``urlopen`` stand-in that returns bytes in operator-chosen chunks.

    The real ``HTTPResponse`` returns whatever the OS has buffered on a
    given read — we replicate that by serving from a fixed list of
    chunks so each test can dial in the frame-boundary behaviour it
    cares about.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def __enter__(self) -> _ChunkedResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def _frame(content: str) -> bytes:
    """Build one OpenAI-style SSE frame carrying ``content`` as delta."""
    body = (
        '{"choices":[{"delta":{"content":'
        f'"{content}"'
        '}}]}'
    )
    return f"data: {body}\n\n".encode()


def _collect(it: Iterator[str]) -> list[str]:
    return list(it)


# ---------------------------------------------------------------------------
# Short-circuits
# ---------------------------------------------------------------------------


def test_disabled_llm_yields_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM disabled → no HTTP call, generator yields zero fragments."""
    s = Settings(llm_enabled=False)
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)

    def boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == []


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_yields_each_frame_content_in_order(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [
        _frame("Hello") + _frame(" "),
        _frame("world"),
        b"data: [DONE]\n\n",
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse(chunks),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert "".join(out) == "Hello world"


def test_request_shape_uses_stream_true(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming variant must set ``stream: true`` on the request
    body — without it the proxy returns a single non-SSE response and
    the parser yields nothing."""
    captured: dict[str, object] = {}

    def fake_urlopen(req: object, timeout: float = 0.0) -> _ChunkedResponse:
        import json as _json
        captured["body"] = _json.loads(req.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["headers"] = {
            k.lower(): v for k, v in req.header_items()  # type: ignore[attr-defined]
        }
        return _ChunkedResponse([_frame("x"), b"data: [DONE]\n\n"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=42, log_label="test"
        )
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["stream"] is True
    assert body["max_tokens"] == 42
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("accept") == "text/event-stream"


# ---------------------------------------------------------------------------
# Frame-boundary edge cases
# ---------------------------------------------------------------------------


def test_frame_split_across_reads(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One SSE frame straddling two underlying reads must still parse —
    the buffer is preserved across iterations."""
    full = _frame("split-frame") + b"data: [DONE]\n\n"
    # Split mid-frame at an arbitrary byte boundary that lands inside
    # the JSON payload.
    cut = len(_frame("split-frame")) - 5
    chunks = [full[:cut], full[cut:]]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse(chunks),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert "".join(out) == "split-frame"


def test_multiple_frames_in_one_read(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several SSE frames in a single read chunk all get parsed."""
    burst = _frame("a") + _frame("b") + _frame("c") + b"data: [DONE]\n\n"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse([burst]),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == ["a", "b", "c"]


def test_done_sentinel_terminates_stream_early(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[DONE] must end iteration even when more bytes are available
    afterwards — some proxies leave trailing bytes in the socket and
    we shouldn't try to parse them."""
    chunks = [
        _frame("first"),
        b"data: [DONE]\n\n",
        _frame("never-yielded"),
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse(chunks),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == ["first"]


def test_socket_close_without_done_still_yields_received_fragments(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proxies that close the socket on completion (no explicit [DONE])
    must still yield everything received before the close."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse([_frame("only-frame")]),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == ["only-frame"]


# ---------------------------------------------------------------------------
# Resilience: skip bad lines / frames, don't crash
# ---------------------------------------------------------------------------


def test_non_data_lines_are_ignored(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``event:``, ``id:``, comments (``: ...``), and blank separator
    lines must all be skipped silently."""
    mixed = (
        b": this is a comment\n"
        b"event: message\n"
        b"id: 42\n"
        + _frame("real-content")
        + b"data: [DONE]\n\n"
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse([mixed]),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == ["real-content"]


def test_malformed_json_frame_is_skipped(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single bad frame must not abort the stream — the next good
    frame keeps yielding."""
    chunks = [
        b"data: {not valid json}\n\n",
        _frame("recovered"),
        b"data: [DONE]\n\n",
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse(chunks),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == ["recovered"]


def test_frame_missing_content_is_skipped(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenAI sends an empty ``{"choices":[{"delta":{}}]}`` frame at
    the start of streaming (the role frame) and at the end (the
    finish_reason frame). Both must be skipped without yielding empty
    strings — empty deltas would clutter the UI."""
    chunks = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
        _frame("hello"),
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_kw: _ChunkedResponse(chunks),
    )

    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == ["hello"]


# ---------------------------------------------------------------------------
# HTTP / network failures: yield nothing, never raise
# ---------------------------------------------------------------------------


def test_http_error_yields_nothing(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _ChunkedResponse:
        # Fake URL is only used to construct the HTTPError — never
        # dialed. Use https:// to avoid the no-TLS lint.
        raise urllib.error.HTTPError(
            "https://fake", 503, "Service Unavailable",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error":"overloaded"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == []


def test_network_error_yields_nothing(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _ChunkedResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == []


def test_timeout_yields_nothing(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _ChunkedResponse:
        raise TimeoutError("read timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = _collect(
        summarize_client._call_chat_stream(
            prompt="hi", max_tokens=10, log_label="test"
        )
    )
    assert out == []
