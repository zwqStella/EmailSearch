"""Best-effort LLM helpers against a local OpenAI-compatible endpoint.

Why stdlib `urllib` and not `httpx`:
  - one POST per call, on a localhost socket — no need for connection pooling
  - keeps the dep footprint identical to the rest of the project
  - test stubbing is trivial (`monkeypatch.setattr` on `urllib.request.urlopen`)

Contract for every public function in this module:
  - returns the result string on success
  - returns `None` on any failure (network, timeout, HTTP non-2xx, malformed
    JSON, empty content) — the caller treats it as a no-op and falls back to
    its non-LLM path. Functions are designed so the caller can invoke them
    unconditionally without an extra `if settings.llm_enabled` guard.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from emailsearch.config import get_settings
from emailsearch.db.models import EmailRow

log = logging.getLogger(__name__)

# Kept language-neutral on purpose — the corpus includes CJK. Asking the model
# to reply in the source language avoids the bilingual-summary failure mode
# where Chinese emails get summarized in English (and vice versa), which
# hurts both readability and downstream semantic recall.
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

# Query augmentation prompt for the SEMANTIC (embedding) leg of search.
#
# Output is fed straight into ``embed_query`` and matched via cosine
# similarity against indexed chunk embeddings; it's never shown to the
# user, so it should be optimized for content-overlap with how someone
# would actually WRITE about the topic. Design rules:
#
#   1. SENTENCE FORM, not a keyword bag. Sentence-transformer embeddings
#      cluster by surface form as well as content, so a sentence-shaped
#      query lands much closer to a sentence-shaped chunk.
#   2. DROP filler that describes the SEARCH (container nouns "email" /
#      "邮件", filler verbs "find me", relative-time expressions). They
#      never appear in email content and just drag the vector toward
#      generic semantics.
#   3. WEAVE IN related vocabulary that DOES appear in emails on the
#      topic — synonyms, related concepts, common co-occurring words.
#
# Proper nouns, identifiers, numbers, and the user's own keywords are
# preserved verbatim — augmentation expands AROUND them.
_AUGMENT_PROMPT = (
    "Rewrite the email search query below as a SEMANTIC EMBEDDING TARGET\n"
    "— 1-2 short NATURAL-LANGUAGE SENTENCES that read like a line from\n"
    "an email actually discussing this topic, with related vocabulary\n"
    "woven in. The output is matched via cosine similarity against\n"
    "email subject / body / summary sentences and is NEVER shown to the\n"
    "user, so optimize for sentence-level semantic overlap with how the\n"
    "topic would be DISCUSSED in prose, not for keyword recall.\n"
    "\n"
    "Why sentences and not keywords: the indexed chunks are sentences\n"
    "from real emails, and sentence-transformer embeddings cluster by\n"
    "surface form as well as content — a sentence-shaped query lands\n"
    "much closer to a sentence-shaped chunk than a bag of keywords does.\n"
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

# Query distillation prompt for the FTS (BM25) leg of semantic search.
# Strips natural-language filler ("help me find that email about ...")
# down to the topical content so the FTS query targets the topic itself,
# not the words describing the search.
#
# Two filler categories need explicit drops because they describe the
# SEARCH and would otherwise survive (especially in CJK where they appear
# mid-phrase):
#   - container nouns ("email" / "邮件"): every indexed item IS an email.
#   - relative-time expressions ("last month" / "上个月"): handled by
#     SearchFilters.start_at / end_at, never matched as text.
#
# Output is BILINGUAL when the query is not in English — original-language
# keywords PLUS their English translations — because the corpus may be
# written in either language and bm25 only matches token-equal text.
# Proper nouns and already-English fragments are left alone.
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
    combined payload is capped at ``settings.llm_max_input_chars`` so a
    single multi-MB attachment can't blow the context window; oversized
    attachment text is trimmed with a ``[truncated]`` marker.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return None

    body = (email.body_text or "").strip()
    attachment_blocks = _build_attachment_blocks(email)

    # Nothing useful to summarize — bail before paying for an LLM round-trip.
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
    """Per-attachment text blocks with a clear delimiter for the model.

    Skips attachments without extracted text. The delimiter also acts as
    a weak prompt-injection guard — attachment content is clearly fenced
    as content rather than as user instructions.
    """
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
    """Concatenate body + attachment blocks under a hard character budget.

    Body gets the first share of the budget (up to its full length, capped
    at half the budget when attachments are present); attachments are
    appended in order until the budget is exhausted. The last attachment
    may be partially truncated and gets a ``[truncated]`` marker.
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

    Used by the semantic_knn leg before embedding. Bridges the gap between
    a terse user query ("budget update") and the richer text in indexed
    chunks ("Q3 budget review: deliverable approved..."). A None return
    means the caller falls back to embedding the raw query.
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

    Used by the semantic_fts leg. Natural-language queries like "help me
    find that email about Q3 budget approval" contain filler that dilutes
    the sentence-transformer vector; distillation produces a content-only
    phrase ("Q3 budget approval"). A None return falls back to the raw
    query.
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
    """Shared POST to /chat/completions. Returns content text or None on failure.

    All exception paths are funneled here so callers don't need to know the
    HTTP-vs-JSON-vs-shape failure modes — they just get None on anything
    that prevents a usable response.
    """
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
        with urllib.request.urlopen(req, timeout=settings.llm_timeout_s) as resp:  # noqa: S310
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # 4xx/5xx — include the response body. Default str(HTTPError) is
        # just "HTTP Error 400:" which is useless for diagnosing "unknown
        # model name" / "rate limit" / "auth failed". Read up to 500 bytes
        # so a runaway HTML error page doesn't flood the logs.
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
    # INFO-level prompt/response log so the user can tail the server output
    # and see exactly what each call to the model produced. Trimmed to 200
    # chars to keep one prompt/response on a couple of lines.
    log.info(
        "LLM %s: prompt=%r response=%r",
        log_label,
        prompt[:200] + ("…" if len(prompt) > 200 else ""),
        out[:200] + ("…" if len(out) > 200 else ""),
    )
    return out or None

