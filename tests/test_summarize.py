"""Tests for `summarize.client.summarize_email` with stubbed HTTP."""

from __future__ import annotations

import io
import json
import time
import urllib.error
from typing import Any

import pytest

from emailsearch.config import Settings
from emailsearch.db.models import AttachmentRecord, EmailAddress, EmailRow
from emailsearch.summarize import client as summarize_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _email(
    body: str = "Body about budget for Q3.",
    subject: str = "Q3 plan",
    *,
    attachments: list[AttachmentRecord] | None = None,
) -> EmailRow:
    return EmailRow(
        id="msg-x",
        subject=subject,
        from_address="alice@example.com",
        from_name="Alice",
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=int(time.time()),
        body_text=body,
        body_html=f"<p>{body}</p>",
        attachments=attachments or [],
        has_attachments=bool(attachments),
    )


def _attachment(name: str, text: str, *, att_id: str = "att-1") -> AttachmentRecord:
    return AttachmentRecord(
        att_id=att_id,
        name=name,
        content_type="application/pdf",
        size=len(text),
        extracted_text=text,
        status="ok",
    )


class _FakeResponse:
    """Minimal stand-in for `urllib.request.urlopen`'s context-manager value."""

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


@pytest.fixture()
def llm_enabled(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings with LLM summarization turned on, pointed at a fake host.

    Summarize vs augment caps are deliberately distinct so tests can assert
    the right one is being applied to each call.
    """
    s = Settings(
        llm_enabled=True,
        llm_base_url="http://127.0.0.1:4141/v1",
        llm_model="local-model",
        llm_timeout_s=5.0,
        llm_max_tokens=180,
        llm_augment_max_tokens=40,
        llm_distill_max_tokens=20,
        llm_max_input_chars=200,
    )
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)
    return s


# ---------------------------------------------------------------------------
# Disabled / no-input short-circuits
# ---------------------------------------------------------------------------


def test_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm_enabled=False is the default — summarizer must not touch the network."""
    s = Settings(llm_enabled=False)
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called when llm_enabled=False")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert summarize_client.summarize_email(_email()) is None


def test_returns_none_for_empty_body(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No body text → no point asking the LLM, no HTTP call."""

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called when body is empty")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert summarize_client.summarize_email(_email(body="   ")) is None


# ---------------------------------------------------------------------------
# Happy path + request shape
# ---------------------------------------------------------------------------


def test_happy_path_returns_summary_and_sends_correct_request(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {"choices": [{"message": {"content": "  Budget review for Q3.  "}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = summarize_client.summarize_email(
        _email(body="Long discussion of budget.", subject="Q3 plan")
    )
    assert out == "Budget review for Q3."  # whitespace trimmed

    # Request shape: hits the /chat/completions route on the configured base.
    assert captured["url"] == "http://127.0.0.1:4141/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["timeout"] == llm_enabled.llm_timeout_s

    body = captured["body"]
    assert body["model"] == "local-model"
    # Summary uses the summary-specific token cap.
    assert body["max_tokens"] == llm_enabled.llm_max_tokens
    assert body["stream"] is False
    # Prompt contains the subject and from address so the model has context.
    prompt = body["messages"][0]["content"]
    assert "Q3 plan" in prompt
    assert "alice@example.com" in prompt
    assert "Long discussion of budget." in prompt


def test_long_body_is_truncated_to_max_input_chars(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bodies past llm_max_input_chars get truncated before prompting so a
    single huge email can't bloat the request or hang the model."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    # Use a body marker that can't appear anywhere else in the prompt
    # template, subject, or from address — keeps the assertion robust to
    # template tweaks.
    marker = "\u2603"  # snowman
    big_body = marker * (llm_enabled.llm_max_input_chars * 3)
    summarize_client.summarize_email(_email(body=big_body))

    prompt = captured["body"]["messages"][0]["content"]
    # Exactly llm_max_input_chars of the marker should reach the prompt.
    assert prompt.count(marker) == llm_enabled.llm_max_input_chars


# ---------------------------------------------------------------------------
# Failure modes — must all return None, never raise
# ---------------------------------------------------------------------------


def test_network_error_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.summarize_email(_email()) is None


def test_timeout_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        raise TimeoutError("read timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.summarize_email(_email()) is None


def test_malformed_json_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        return _FakeResponse(b"<html>nginx error</html>")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.summarize_email(_email()) is None


def test_unexpected_response_shape_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server returns valid JSON but missing the OpenAI 'choices' field."""

    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        return _FakeResponse({"error": "model not loaded"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.summarize_email(_email()) is None


def test_empty_summary_string_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that returns whitespace-only content is treated as a failure
    so we don't store an empty summary."""

    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        return _FakeResponse({"choices": [{"message": {"content": "   \n  "}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.summarize_email(_email()) is None


# ---------------------------------------------------------------------------
# Attachment inclusion in the summary prompt
# ---------------------------------------------------------------------------


def test_summary_prompt_includes_attachment_text(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the email has an attachment with extracted text, that text plus a
    per-attachment header reaches the model — so the summary can ground in
    PDF/DOCX content even when the body just says "see attached"."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    att = _attachment(
        "Q3-budget.pdf",
        "Q3 budget total approved at $2.5M with 17% YoY growth.",
    )
    summarize_client.summarize_email(
        _email(body="see attached", subject="Q3 budget", attachments=[att])
    )

    prompt = captured["body"]["messages"][0]["content"]
    # Body content reaches the model.
    assert "see attached" in prompt
    # Per-attachment header naming the file.
    assert "--- Attachment: Q3-budget.pdf ---" in prompt
    # Extracted attachment text reaches the model.
    assert "$2.5M with 17% YoY growth" in prompt


def test_summary_skips_attachments_without_extracted_text(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Image attachments we couldn't OCR (or any att with empty
    extracted_text) must NOT contribute a per-attachment block — there's
    nothing to summarize and the empty header would be noise."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    no_text = AttachmentRecord(
        att_id="att-blank",
        name="screenshot.png",
        content_type="image/png",
        size=100,
        extracted_text="",
        status="empty",
    )
    summarize_client.summarize_email(
        _email(body="see image", attachments=[no_text])
    )

    prompt = captured["body"]["messages"][0]["content"]
    assert "Attachment: screenshot.png" not in prompt


def test_summary_runs_with_empty_body_when_attachment_has_content(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An email with an empty body BUT a non-empty attachment is still
    summarizable — the body short-circuit no longer wins when there's
    attachment text to ground in."""

    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        return _FakeResponse({"choices": [{"message": {"content": "summary"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = summarize_client.summarize_email(
        _email(body="", attachments=[_attachment("doc.pdf", "real content here")])
    )
    assert out == "summary"


def test_summary_returns_none_when_body_and_attachments_both_empty(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No body, no attachment text → no HTTP call, return None."""

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called for empty content")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert summarize_client.summarize_email(_email(body="")) is None


def test_summary_truncates_oversized_attachment(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A huge attachment gets trimmed to fit the per-call budget. The body
    still gets HALF the budget when attachments are present, so a short
    body isn't dropped just because the attachment is large. Truncated
    content carries a marker so the model knows it was cut."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    budget = llm_enabled.llm_max_input_chars  # 200 in the fixture
    huge = "\u2603" * (budget * 5)  # 1000 snowmen — way over budget
    summarize_client.summarize_email(
        _email(body="short body", attachments=[_attachment("big.pdf", huge)])
    )

    prompt = captured["body"]["messages"][0]["content"]
    # Body present.
    assert "short body" in prompt
    # Attachment header present.
    assert "--- Attachment: big.pdf ---" in prompt
    # Snowmen count is well under the raw input length.
    snowmen = prompt.count("\u2603")
    assert 0 < snowmen < len(huge), f"expected truncation; got {snowmen} snowmen"
    # Truncation marker is emitted so the model knows content was cut.
    assert "[attachment truncated]" in prompt


# ---------------------------------------------------------------------------
# augment_query
# ---------------------------------------------------------------------------


def test_augment_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm_enabled=False → augment_query short-circuits without HTTP."""
    s = Settings(llm_enabled=False)
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called when llm_enabled=False")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert summarize_client.augment_query("anything") is None


def test_augment_returns_none_for_empty_query(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blank query → no LLM call, no augmented string."""

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called on empty query")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert summarize_client.augment_query("   ") is None


def test_augment_happy_path_uses_augment_token_cap(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: returns the model's expanded query (trimmed) and uses the
    AUGMENT token cap — distinct from the summary cap so the two operations
    can be tuned independently."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": "  alpha rollout progress update  "}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = summarize_client.augment_query("alpha update")
    assert out == "alpha rollout progress update"  # whitespace trimmed

    assert captured["url"] == "http://127.0.0.1:4141/v1/chat/completions"
    # Augment uses its own (smaller) token cap, not llm_max_tokens.
    assert captured["body"]["max_tokens"] == llm_enabled.llm_augment_max_tokens
    assert captured["body"]["max_tokens"] != llm_enabled.llm_max_tokens
    # The user query is in the prompt verbatim.
    assert "alpha update" in captured["body"]["messages"][0]["content"]


def test_augment_network_error_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.augment_query("alpha") is None


def test_augment_empty_response_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only augmentation is treated as a failure so the caller
    falls back to the raw query rather than embedding an empty string."""

    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        return _FakeResponse({"choices": [{"message": {"content": "\n  \n"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.augment_query("alpha") is None


# ---------------------------------------------------------------------------
# distill_query
# ---------------------------------------------------------------------------


def test_distill_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm_enabled=False → distill_query short-circuits without HTTP."""
    s = Settings(llm_enabled=False)
    monkeypatch.setattr("emailsearch.summarize.client.get_settings", lambda: s)

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called when llm_enabled=False")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert summarize_client.distill_query("anything") is None


def test_distill_returns_none_for_empty_query(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blank query → no LLM call."""

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("urlopen must not be called on empty query")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert summarize_client.distill_query("   ") is None


def test_distill_happy_path_uses_distill_token_cap(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: returns the model's distilled phrase (trimmed) and uses
    the DISTILL token cap — distinct from augment / summary caps so the
    three operations can be tuned independently."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": "  Q3 budget approval  "}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = summarize_client.distill_query("help me find the email about Q3 budget approval")
    assert out == "Q3 budget approval"  # whitespace trimmed

    assert captured["url"] == "http://127.0.0.1:4141/v1/chat/completions"
    # Distill uses its own (smaller) token cap, distinct from augment / summary.
    assert captured["body"]["max_tokens"] == llm_enabled.llm_distill_max_tokens
    assert captured["body"]["max_tokens"] != llm_enabled.llm_augment_max_tokens
    assert captured["body"]["max_tokens"] != llm_enabled.llm_max_tokens
    # The raw user query reaches the prompt verbatim.
    prompt = captured["body"]["messages"][0]["content"]
    assert "help me find the email about Q3 budget approval" in prompt


def test_distill_network_error_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.distill_query("help me find alpha") is None


def test_distill_empty_response_returns_none(
    llm_enabled: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that returns whitespace-only content → caller falls back to raw query."""

    def fake_urlopen(*_a: object, **_kw: object) -> _FakeResponse:
        return _FakeResponse({"choices": [{"message": {"content": "\n   \n"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert summarize_client.distill_query("anything") is None
