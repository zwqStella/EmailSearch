"""Best-effort LLM helpers against a local OpenAI-compatible endpoint.

Every public function:
  - returns the result string on success
  - returns ``None`` on any failure (network, timeout, HTTP non-2xx,
    malformed JSON, empty content) — the caller treats it as a no-op and
    falls back to its non-LLM path.

Uses stdlib ``urllib`` rather than ``httpx`` because: one POST per call,
on a localhost socket — no need for connection pooling. Test stubbing is
trivial (``monkeypatch.setattr`` on ``urllib.request.urlopen``).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator

from emailsearch.config import get_settings
from emailsearch.db.models import EmailRow

log = logging.getLogger(__name__)

# Kept language-neutral on purpose — the corpus includes CJK. Asking for
# the source language avoids the bilingual-summary failure mode where
# Chinese emails get summarized in English and vice versa.
_SUMMARIZE_PROMPT = (
    "Summarize the email below in 1-3 short sentences (no more than 60 words).\n"
    "Focus on the main topic and the most important specifics (decisions,\n"
    "deadlines, names, numbers). Reply in the SAME LANGUAGE as the email.\n"
    "Output ONLY the summary — no preamble, no 'This email is about...'.\n"
    "Attachments (when present) are extracted text shown after the body;\n"
    "treat them as part of the same email when picking the topic.\n"
    "\n"
    "---\n"
    "Subject: {subject}\n"
    "From: {from_addr}\n"
    "\n"
    "{body}\n"
)

# Query augmentation for the SEMANTIC (embedding) leg. Output is matched
# via cosine similarity against indexed chunk embeddings and is never
# shown to the user — optimize for sentence-level overlap with email
# prose rather than keyword recall.
_AUGMENT_PROMPT = (
    "Rewrite the email search query below as a SEMANTIC EMBEDDING TARGET\n"
    "— 1-2 short NATURAL-LANGUAGE SENTENCES that read like a line from\n"
    "an email actually discussing this topic, with related vocabulary\n"
    "woven in. The output is matched via cosine similarity against\n"
    "email subject / body / summary sentences and is NEVER shown to the\n"
    "user, so optimize for sentence-level semantic overlap with how the\n"
    "topic would be DISCUSSED in prose, not for keyword recall.\n"
    "\n"
    "PRESERVE verbatim:\n"
    "  - proper nouns (people, products, organizations)\n"
    "  - technical terms, version numbers, project codes, identifiers\n"
    "  - the original topical keywords (expand AROUND them, don't replace)\n"
    "\n"
    "WEAVE IN related vocabulary that emails on this topic actually use:\n"
    "  - synonyms ('budget' -> 'spending / funding / allocation')\n"
    "  - related concepts ('rollout' -> 'deployment / launch / release')\n"
    "  - common cooccurring words ('incident' -> 'outage / postmortem / RCA')\n"
    "\n"
    "DROP everything that describes the SEARCH, not the email's content:\n"
    "  - filler verbs: 'help me find', 'show me', 'I'm looking for',\n"
    "    '帮我找', '搜索', '查一下', '找一下'\n"
    "  - container nouns: 'email', 'message', 'mail', 'thread',\n"
    "    '邮件', '信件', '消息'\n"
    "  - relative-time expressions: 'last month', 'yesterday', 'this week',\n"
    "    'recent', '上个月', '昨天', '上周', '最近', '今天'\n"
    "  - meta-references: 'that email about', 'the one mentioning',\n"
    "    '关于...的邮件', '...发的邮件'\n"
    "\n"
    "Output 1-2 short sentences, under 40 words total. Reply in the SAME\n"
    "LANGUAGE as the query. Output ONLY the sentence(s) — no preamble,\n"
    "no quotes, no list formatting, no bullet points.\n"
    "\n"
    "Examples:\n"
    "  'help me find the email about Q3 budget approval'\n"
    "    -> 'The Q3 budget approval is being reviewed, covering funding allocation, fiscal quarter spending, and final sign-off on the deliverable numbers.'\n"
    "  'show me anything from alice mentioning the rollout'\n"
    "    -> 'Alice shared an update on the rollout status — deployment progress, launch readiness, and go-live plans across customer regions.'\n"
    "  'find the security incident report from last week'\n"
    "    -> 'The security incident report covers the outage timeline, postmortem findings, RCA, mitigation steps, and the investigation into the root cause.'\n"
    "  'emails about the new payment integration with Stripe'\n"
    "    -> 'The new Stripe payment integration covers checkout, charges, webhooks, subscriptions, and the billing gateway processor setup.'\n"
    "  '帮我找一下关于个税汇算清缴的邮件'\n"
    "    -> '关于个人所得税年度汇算清缴的通知,涉及退税、补税、专项附加扣除以及工资薪金的申报流程。'\n"
    "  '上个月工会发的开心麻花的邮件'\n"
    "    -> '工会组织的开心麻花话剧演出团建活动,包含团体票购买、福利发放和报名安排的通知。'\n"
    "\n"
    "Query: {query}\n"
)

# Query distillation for the FTS (BM25) leg. Strips natural-language
# filler down to topical content. Output is BILINGUAL when the query
# isn't English — original-language keywords PLUS English translations —
# so BM25 matches emails written in either language.
_DISTILL_PROMPT = (
    "Rewrite the email search query below as a SHORT keyword phrase\n"
    "containing ONLY the topical content of the email itself — names of\n"
    "people / organizations / products, technical terms, specific subjects,\n"
    "and identifiers (version numbers, project codes, concrete dates that\n"
    "name an event).\n"
    "\n"
    "BILINGUAL OUTPUT: the output is fed to a full-text (BM25) search\n"
    "over emails that may be written in EITHER English OR the user's\n"
    "language. If the query is NOT in English, APPEND the English\n"
    "translation of the topical keywords after the original-language\n"
    "ones (space-separated, same phrase) so the same FTS query matches\n"
    "emails on the topic regardless of which language they're written\n"
    "in. Proper nouns, identifiers, and already-English fragments stay\n"
    "as-is — don't translate 'Stripe', 'Q3', or 'v2.1'. English-only\n"
    "queries stay English-only (no translation needed).\n"
    "\n"
    "DROP everything that describes the search rather than the email:\n"
    "  - filler verbs: 'help me find', 'show me', 'I'm looking for',\n"
    "    'find', 'search for', '帮我找', '搜索', '查一下', '找一下'\n"
    "  - container nouns (every result IS an email): 'email', 'message',\n"
    "    'mail', 'thread', '邮件', '信件', '消息'\n"
    "  - relative-time expressions (handled by a separate date filter, NOT\n"
    "    matched semantically): 'last month', 'yesterday', 'this week',\n"
    "    'recent', 'today', '上个月', '昨天', '上周', '最近', '今天',\n"
    "    '前几天', '这个月'\n"
    "  - meta-references: 'that email about', 'the one mentioning',\n"
    "    'an email from', '关于...的邮件', '...发的邮件'\n"
    "\n"
    "Preserve specific terms verbatim. Output ONLY the distilled phrase\n"
    "— no preamble, no quotes, no formatting, no labels like 'EN:' /\n"
    "'ZH:'. Just the keywords on one line, space-separated.\n"
    "\n"
    "Examples:\n"
    "  'help me find the email about Q3 budget approval' -> 'Q3 budget approval'\n"
    "  'show me anything from alice mentioning the rollout' -> 'alice rollout'\n"
    "  'last month email from HR about benefits enrollment' -> 'HR benefits enrollment'\n"
    "  '帮我找一下关于个税汇算清缴的邮件' -> '个税 汇算清缴 personal income tax annual settlement reconciliation'\n"
    "  '上个月工会发的开心麻花的邮件' -> '工会 开心麻花 labor union Mahua FunAge theater performance'\n"
    "  '最近收到的关于Q3预算审批的邮件' -> 'Q3 预算审批 Q3 budget approval'\n"
    "\n"
    "Query: {query}\n"
)


def summarize_email(email: EmailRow) -> str | None:
    """Return a short summary of `email` or `None` on any failure / when disabled.

    Combines subject, sender, body, and extracted attachment text so the
    model picks up the topic even when it lives in a PDF / DOCX. The
    combined payload is capped at ``settings.llm_max_input_chars``.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return None

    body = (email.body_text or "").strip()
    attachment_blocks = _build_attachment_blocks(email)

    if not body and not attachment_blocks:
        return None

    content = _truncate_to_budget(
        body=body,
        attachment_blocks=attachment_blocks,
        budget=settings.llm_max_input_chars,
    )

    prompt = _SUMMARIZE_PROMPT.format(
        subject=email.subject or "(no subject)",
        from_addr=email.from_address or "(unknown)",
        body=content,
    )
    return _call_chat(
        prompt=prompt,
        max_tokens=settings.llm_max_tokens,
        log_label=f"summarize({email.id})",
    )


def _build_attachment_blocks(email: EmailRow) -> list[str]:
    """Per-attachment text blocks with a clear delimiter. Skips attachments
    without extracted text. The delimiter also acts as a weak
    prompt-injection guard — content is clearly fenced as content."""
    blocks: list[str] = []
    for att in email.attachments:
        text = (att.extracted_text or "").strip()
        if not text:
            continue
        blocks.append(f"--- Attachment: {att.name} ---\n{text}")
    return blocks


def _truncate_to_budget(
    *, body: str, attachment_blocks: list[str], budget: int
) -> str:
    """Concatenate body + attachments under a hard character budget.

    Body gets the first share (capped at half the budget when
    attachments are present); attachments are appended in order until
    the budget is exhausted. The last attachment may be partially
    truncated and gets a ``[truncated]`` marker.
    """
    pieces: list[str] = []
    remaining = budget

    if body:
        body_cap = budget // 2 if attachment_blocks else budget
        body_part = body[:body_cap]
        if len(body) > body_cap:
            body_part += "\n[body truncated]"
        pieces.append(body_part)
        remaining -= len(body_part)

    for block in attachment_blocks:
        if remaining <= 0:
            break
        if len(block) <= remaining:
            pieces.append(block)
            remaining -= len(block) + 2  # +2 for the join "\n\n"
            continue
        # Last block doesn't fit fully — trim and tag.
        pieces.append(block[:remaining] + "\n[attachment truncated]")
        remaining = 0
        break

    return "\n\n".join(pieces)


def augment_query(query: str) -> str | None:
    """Return an LLM-expanded version of `query` or `None` if disabled / failed.

    Used by the semantic_knn leg before embedding. Bridges the gap
    between a terse user query ("budget update") and richer indexed
    text. Caller falls back to embedding the raw query on None.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return None

    query = (query or "").strip()
    if not query:
        return None

    prompt = _AUGMENT_PROMPT.format(query=query)
    return _call_chat(
        prompt=prompt,
        max_tokens=settings.llm_augment_max_tokens,
        log_label=f"augment({query[:40]!r})",
    )


def distill_query(query: str) -> str | None:
    """Return a filler-stripped version of `query` or `None` if disabled / failed.

    Used by the semantic_fts leg. Strips filler ("help me find...") to a
    content-only phrase ("Q3 budget approval"). Caller falls back to the
    raw query on None.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return None

    query = (query or "").strip()
    if not query:
        return None

    prompt = _DISTILL_PROMPT.format(query=query)
    return _call_chat(
        prompt=prompt,
        max_tokens=settings.llm_distill_max_tokens,
        log_label=f"distill({query[:40]!r})",
    )


def _call_chat(*, prompt: str, max_tokens: int, log_label: str) -> str | None:
    """Shared POST to /chat/completions. Returns content text or None on
    failure. All exception paths funnel here so callers don't need to
    know the HTTP-vs-JSON-vs-shape failure modes."""
    settings = get_settings()
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Low temperature: outputs should be deterministic-ish and grounded.
        "temperature": 0.2,
        "stream": False,
    }
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # nosec B310 — URL is operator-controlled (env config), not user input.
        with urllib.request.urlopen(req, timeout=settings.llm_timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # Include body up to 500 bytes so "unknown model" / "rate limit" /
        # "auth failed" are diagnosable from the log.
        try:
            body = exc.read(500).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover — read() can fail on closed body
            body = "<unreadable>"
        log.warning(
            "LLM call failed for %s: HTTP %d %s — body=%r",
            log_label, exc.code, exc.reason, body,
        )
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("LLM call failed for %s: %s", log_label, exc)
        return None

    try:
        parsed = json.loads(raw.decode("utf-8"))
        text = parsed["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, UnicodeDecodeError) as exc:
        log.warning("LLM: unexpected response shape for %s: %s", log_label, exc)
        return None

    out = (text or "").strip()
    # INFO-level prompt/response log (trimmed to 200 chars) so the user
    # can tail the server output and see what each call produced.
    log.info(
        "LLM %s: prompt=%r response=%r",
        log_label,
        prompt[:200] + ("…" if len(prompt) > 200 else ""),
        out[:200] + ("…" if len(out) > 200 else ""),
    )
    return out or None


def _call_chat_stream(
    *,
    prompt: str,
    max_tokens: int,
    log_label: str,
) -> Iterator[str]:
    """Stream content fragments from /chat/completions. Yields nothing on failure.

    Sibling to :func:`_call_chat` for the Ask agent. Wire format follows
    the OpenAI SSE convention used by copilot-api / LM Studio / Ollama:

    .. code-block:: text

        data: {"choices":[{"delta":{"content":"Hello"}}]}\\n\\n
        data: {"choices":[{"delta":{"content":" world"}}]}\\n\\n
        data: [DONE]\\n\\n

    A single ``urlopen`` read may straddle frame boundaries, so we
    buffer raw bytes, split on ``\\n``, and parse complete lines. The
    ``[DONE]`` sentinel ends the stream explicitly — some backends
    close the socket immediately after, treating it as authoritative
    avoids a hang in the other case.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return

    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            # Some proxies require this hint to keep the connection open
            # and flush per-event rather than buffering.
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    yielded_chars = 0
    try:
        # nosec B310 — URL is operator-controlled (env config), not user input.
        with urllib.request.urlopen(
            req, timeout=settings.llm_timeout_s
        ) as resp:
            buffer = b""
            while True:
                # Small chunk size keeps latency low — each SSE frame is
                # tens to a few hundred bytes.
                chunk = resp.read(1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    if not text.startswith("data:"):
                        # ``event:`` / ``id:`` / comment lines — ignore.
                        continue
                    payload_str = text[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        log.info(
                            "LLM %s stream: [DONE] after %d char(s)",
                            log_label, yielded_chars,
                        )
                        return
                    try:
                        frame = json.loads(payload_str)
                        delta = frame["choices"][0].get("delta", {})
                        content = delta.get("content")
                    except (
                        json.JSONDecodeError, KeyError, IndexError, TypeError,
                    ):
                        # Per-frame parse failures are non-fatal — skip
                        # the bad frame and keep streaming.
                        continue
                    if not content:
                        continue
                    yielded_chars += len(content)
                    yield content
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(500).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover — read() can fail on closed body
            body = "<unreadable>"
        log.warning(
            "LLM stream failed for %s: HTTP %d %s — body=%r",
            log_label, exc.code, exc.reason, body,
        )
        return
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("LLM stream failed for %s: %s", log_label, exc)
        return

    # Stream ended without [DONE] (proxy closed socket on completion).
    log.info(
        "LLM %s stream: socket closed after %d char(s) (no [DONE])",
        log_label, yielded_chars,
    )

