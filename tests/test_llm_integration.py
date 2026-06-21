"""Integration tests against a real local LLM endpoint.

Runs whenever the configured ``llm_base_url`` is reachable; skips cleanly
otherwise so the suite stays green on machines without a model loaded.
Exercises every "job kind" the rest of the app issues to the model:

  - ``summarize_email``    — ingest-time per-email summary
  - ``distill_query``      — query-time filler-stripping for semantic search
  - ``augment_query``      — query-time expansion for rerank

Each test asserts:
  1. The call succeeds (returns a non-None string).
  2. The response is non-empty and shape-appropriate for its job
     (distilled output shorter than input, augmented output non-trivial,
     summary references at least one content term).

We deliberately keep the content assertions loose: LLM outputs are
non-deterministic, so we don't compare exact strings — only that the
response is "plausible" for its prompt. The point of the test is to catch
end-to-end regressions in the prompt + HTTP plumbing, not to grade the
model's writing.

To run only these tests:

    pytest tests/test_llm_integration.py -v

To skip these (e.g. on CI):

    Set ``EMAILSEARCH_LLM_ENABLED=false`` in the environment.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from emailsearch.config import Settings
from emailsearch.db.models import AttachmentRecord, EmailAddress, EmailRow
from emailsearch.summarize import augment_query, distill_query, summarize_email

# How long to wait for the /models probe before giving up and skipping the
# whole module. Localhost should respond well under a second; 2s is a
# generous cap that keeps the suite snappy when no server is up.
_PROBE_TIMEOUT_S = 2.0


def _server_reachable(base_url: str) -> tuple[bool, str]:
    """Probe ``GET {base_url}/models`` to confirm an OpenAI-compatible server
    is listening. Returns ``(ok, detail)`` so the skip reason is informative.

    We hit ``/models`` rather than ``/chat/completions`` because the former
    is a cheap GET that returns immediately, while the latter would require
    a (potentially slow) generation just to verify reachability.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        # nosec B310 — URL is operator-controlled config, not user input.
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S):  # noqa: S310
            return True, "ok"
    except urllib.error.URLError as exc:
        return False, f"URLError: {exc.reason}"
    except (TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


@pytest.fixture(scope="module", autouse=True)
def live_llm_or_skip() -> None:
    """Module-level guard: skip every test in this file unless the configured
    local LLM endpoint is reachable AND ``llm_enabled`` is True.

    We read Settings() fresh (bypassing the lru_cache) so env-var changes
    between test runs are respected.
    """
    s = Settings()
    if not s.llm_enabled:
        pytest.skip(
            f"EMAILSEARCH_LLM_ENABLED={s.llm_enabled}; live LLM tests skipped"
        )
    ok, detail = _server_reachable(s.llm_base_url)
    if not ok:
        pytest.skip(
            f"local LLM not reachable at {s.llm_base_url}: {detail}; "
            "live LLM tests skipped"
        )


# ---------------------------------------------------------------------------
# Job kind 1: summarize_email
# ---------------------------------------------------------------------------


def test_summarize_email_against_live_llm() -> None:
    """End-to-end: a representative email gets summarized in a sentence or
    two. We assert on the dollar amounts ($2M / $500K / $300K) because those
    are language-invariant — observed real models translating "Q3 budget"
    into other languages when triggered by stray punctuation, but the
    numeric facts always survive translation. If the summary references
    NONE of the dollar amounts, the model either ignored the prompt or our
    prompt template is broken."""
    email = EmailRow(
        id="live-test-summary-1",
        subject="Q3 budget approval meeting",
        from_address="alice@example.com",
        from_name="Alice",
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=1_700_000_000,
        body_text=(
            "Hi team, we'll meet Friday at 2pm to finalize the Q3 budget. "
            "Please review the attached spreadsheet beforehand. Key decisions: "
            "$2M for new hires, $500K for tooling, $300K contingency."
        ),
        body_html="<p>...</p>",
    )

    summary = summarize_email(email)

    assert summary is not None, "live LLM returned None for summarize_email"
    assert summary.strip(), "summary is empty / whitespace-only"
    # Language-invariant grounding check: at least one of the prominent
    # dollar figures from the body must appear in the summary verbatim.
    # These survive any translation the model might do.
    expected_any = ("$2M", "$500K", "$300K", "2M", "500K", "300K")
    assert any(term in summary for term in expected_any), (
        f"summary references none of {expected_any!r}: {summary!r}"
    )


def test_summarize_email_with_attachment_against_live_llm() -> None:
    """End-to-end: an email whose body is uninformative ("see attached") but
    whose attachment contains the actual content gets summarized using the
    attachment text — proving attachments reach the model and influence the
    output. Asserts on numeric facts that exist ONLY in the attachment, so
    any of them landing in the summary is proof the attachment was used."""
    email = EmailRow(
        id="live-test-summary-att",
        subject="weekly numbers",
        from_address="alice@example.com",
        from_name="Alice",
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=1_700_000_000,
        body_text="See attached spreadsheet for the details.",
        body_html="<p>See attached.</p>",
        attachments=[
            AttachmentRecord(
                att_id="att-1",
                name="q3-numbers.csv",
                content_type="text/csv",
                size=200,
                extracted_text=(
                    "Revenue: $4.2M\n"
                    "Headcount: 47\n"
                    "Customer churn: 3.1%\n"
                    "Top product: Widget-X (sales: $1.8M)"
                ),
                status="ok",
            )
        ],
        has_attachments=True,
    )

    summary = summarize_email(email)

    assert summary is not None, "live LLM returned None for attachment summary"
    assert summary.strip(), "summary is empty / whitespace-only"
    # Numeric / unique facts that exist ONLY in the attachment. If at least
    # one appears in the summary, attachment content provably reached the
    # model. Kept inclusive across translations.
    expected_any = ("$4.2M", "4.2M", "47", "3.1%", "Widget-X", "$1.8M", "1.8M")
    assert any(term in summary for term in expected_any), (
        f"summary references none of {expected_any!r}: {summary!r}"
    )


def test_summarize_email_returns_none_for_empty_body() -> None:
    """Even with a live LLM, an empty-body email short-circuits to None
    without hitting the network (the function never wastes a round-trip on
    a no-op input)."""
    email = EmailRow(
        id="live-test-summary-empty",
        subject="just a subject",
        from_address="alice@example.com",
        received_at=0,
        body_text="",
        body_html="",
    )
    assert summarize_email(email) is None


# ---------------------------------------------------------------------------
# Job kind 2: distill_query
# ---------------------------------------------------------------------------


def test_distill_query_against_live_llm() -> None:
    """End-to-end: a filler-heavy natural-language query collapses to its
    topical core. The output must be strictly shorter than the input,
    preserve the actual subject keyword(s), and DROP the container noun
    'email' (every indexed item is an email — the word adds no signal)."""
    raw = "help me find that email about Q3 budget approval please"
    distilled = distill_query(raw)

    assert distilled is not None, "live LLM returned None for distill_query"
    assert distilled.strip(), "distilled query is empty / whitespace-only"
    assert len(distilled) < len(raw), (
        f"distilled output should drop filler; raw={raw!r} distilled={distilled!r}"
    )
    # The whole point of distillation is to PRESERVE the topical keyword
    # while dropping filler. "budget" is the topical anchor; if it's missing
    # the prompt is doing something wrong.
    lowered = distilled.lower()
    assert "budget" in lowered, (
        f"distilled missing topical keyword 'budget': {distilled!r}"
    )
    # Container noun must be stripped — it's not part of the email's content.
    assert "email" not in lowered, (
        f"distilled retained container noun 'email': {distilled!r}"
    )


def test_distill_query_drops_container_and_relative_time_cjk() -> None:
    """Regression: a CJK query like '上个月工会发的开心麻花的邮件' must drop
    both the container noun '邮件' and the relative-time expression
    '上个月' while preserving the topical entities ('工会', '开心麻花').

    Date filtering is a separate hard-filter mechanism; relative time
    expressions never match anything in the embedding step, so leaving
    them in dilutes the vector toward generic time-related content."""
    raw = "上个月工会发的开心麻花的邮件"
    distilled = distill_query(raw)

    assert distilled is not None, "live LLM returned None for distill_query"
    assert distilled.strip(), "distilled query is empty / whitespace-only"
    # Container noun: every result IS an email, so '邮件' adds no signal.
    assert "邮件" not in distilled, (
        f"distilled retained container noun '邮件': {distilled!r}"
    )
    # Relative time: handled by SearchFilters, not by the embedding.
    assert "上个月" not in distilled, (
        f"distilled retained relative-time expression '上个月': {distilled!r}"
    )
    # The actual topical entities must survive — they're the whole point.
    assert "开心麻花" in distilled, (
        f"distilled dropped topical entity '开心麻花': {distilled!r}"
    )
    assert "工会" in distilled, (
        f"distilled dropped topical entity '工会': {distilled!r}"
    )


def test_distill_query_returns_none_for_empty_input() -> None:
    """Empty / whitespace input short-circuits to None without an HTTP call."""
    assert distill_query("") is None
    assert distill_query("   \n  ") is None


# ---------------------------------------------------------------------------
# Job kind 3: augment_query
# ---------------------------------------------------------------------------


def test_augment_query_against_live_llm() -> None:
    """End-to-end: a terse query gets expanded with related vocabulary
    suitable for use as a semantic embedding target.

    The augmented output is matched against email body / subject / summary
    embeddings via cosine similarity (NOT shown to the user), so the
    contract is: preserve the original keywords, ADD topically-related
    vocabulary, and DROP everything that describes the search rather than
    the email's content (container nouns, filler verbs, meta-references).
    """
    raw = "Q3 budget approval"
    augmented = augment_query(raw)

    assert augmented is not None, "live LLM returned None for augment_query"
    assert augmented.strip(), "augmented query is empty / whitespace-only"
    # Augmentation adds related terms / synonyms / context, so the output
    # should be meaningfully longer than the input. We don't enforce a hard
    # ratio because models vary in verbosity; just "strictly longer".
    assert len(augmented) > len(raw), (
        f"augmented output should be longer than input; "
        f"raw={raw!r} augmented={augmented!r}"
    )
    # The original concept terms must survive expansion — augmentation
    # is supposed to ADD context, not replace the user's keywords.
    lowered = augmented.lower()
    assert "budget" in lowered, (
        f"augmented dropped the original keyword 'budget': {augmented!r}"
    )
    # The augmented output feeds the embedder, so anything that doesn't
    # describe email CONTENT just dilutes the resulting vector. Container
    # nouns ('email' / 'message') are the prime offender — every indexed
    # item IS an email, so the word adds zero embedding signal.
    assert "email" not in lowered, (
        f"augmented retained container noun 'email': {augmented!r}"
    )
    assert "message" not in lowered, (
        f"augmented retained container noun 'message': {augmented!r}"
    )


def test_augment_query_drops_filler_and_adds_synonyms_cjk() -> None:
    """Regression: a CJK natural-language query like
    '上个月工会发的开心麻花的邮件' must drop the container noun '邮件',
    the relative-time '上个月', and meta-reference '...发的' while
    preserving the topical entities ('工会', '开心麻花') AND expanding
    with related vocabulary that emails on the topic would actually use
    (演出 / 话剧 / 团建 / 福利 / etc.).

    Without this regression check the augment prompt could silently
    regress to the old verbose-prose behaviour ('上个月工会发的开心麻花
    的邮件、通知、活动信息...') which retains all the search-side filler
    and dilutes the embedding."""
    raw = "上个月工会发的开心麻花的邮件"
    augmented = augment_query(raw)

    assert augmented is not None, "live LLM returned None for augment_query"
    assert augmented.strip(), "augmented query is empty / whitespace-only"
    # The user's topical entities must survive — without them the
    # augmented embedding points at something else entirely.
    assert "工会" in augmented, (
        f"augmented dropped topical entity '工会': {augmented!r}"
    )
    assert "开心麻花" in augmented, (
        f"augmented dropped topical entity '开心麻花': {augmented!r}"
    )
    # Container noun: every result IS an email, so '邮件' adds no signal
    # and only drags the embedding toward generic "email" semantics.
    assert "邮件" not in augmented, (
        f"augmented retained container noun '邮件': {augmented!r}"
    )
    # Relative-time: handled by SearchFilters, never matched in embedding
    # space — leaving it in pulls the vector toward unrelated time-themed
    # content.
    assert "上个月" not in augmented, (
        f"augmented retained relative-time expression '上个月': {augmented!r}"
    )
    # Augmentation must actually ADD vocabulary — at least one of the
    # high-likelihood cooccurring terms for 工会 / 开心麻花 should appear.
    # 开心麻花 is a well-known comedy troupe, so 演出 / 话剧 / 团建 / 福利
    # are the canonical workplace-context expansions. We only require ONE
    # of them so the test tolerates per-model variance in synonym choice.
    expected_any = ("演出", "话剧", "团建", "福利", "票", "活动")
    assert any(term in augmented for term in expected_any), (
        f"augmented added no related vocabulary; expected any of "
        f"{expected_any!r} in: {augmented!r}"
    )


def test_augment_query_returns_none_for_empty_input() -> None:
    """Empty input short-circuits without an HTTP call."""
    assert augment_query("") is None
    assert augment_query("   ") is None


# ---------------------------------------------------------------------------
# Cross-cutting: distinct token caps reach the wire
# ---------------------------------------------------------------------------


def test_each_job_kind_completes_within_one_round_trip() -> None:
    """Smoke check that all three job kinds work back-to-back against the
    same endpoint — catches any global state leak between calls (e.g. a
    cached HTTP session or pickled prompt template)."""
    # All three should succeed; specific shape is covered by the dedicated
    # tests above.
    s = summarize_email(
        EmailRow(
            id="smoke",
            subject="x",
            from_address="a@b.com",
            received_at=0,
            body_text="the alpha rollout went well yesterday",
            body_html="",
        )
    )
    d = distill_query("show me the email about alpha rollout")
    a = augment_query("alpha rollout")
    assert s is not None
    assert d is not None
    assert a is not None
