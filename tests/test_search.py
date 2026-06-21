"""Search service tests with stubbed embeddings."""

from __future__ import annotations

import time

import pytest

from emailsearch.db.connection import apply_schema, open_connection
from emailsearch.db.models import AttachmentRecord, Chunk, EmailAddress, EmailRow
from emailsearch.db.repositories import insert_email_with_chunks
from emailsearch.search import service as search_service


@pytest.fixture()
def conn():
    c = open_connection(":memory:")
    apply_schema(c)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def stub_query_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the query encoder + LLM hooks so tests run offline / deterministic.

    Defaults applied to every test:
      - ``embed_query``: deterministic seed-based vector for known keywords.
      - ``distill_query``: returns ``None`` so the FTS leg inside semantic
        mode falls back to the raw query.
      - ``augment_query``: returns ``None`` so the KNN leg inside semantic
        mode falls back to embedding the raw query.

    Both LLM stubs return ``None`` by default so the autouse path doesn't
    spuriously transform queries — individual tests that exercise the
    distilled-FTS / augmented-KNN behaviour patch their own returns on top
    of this.

    Note: the old LLM-rerank pass (per-candidate ``embed_texts``) is gone —
    semantic ranking now promotes emails whose INDEXED summary chunk matched
    the query (inside the ``semantic_knn`` leg), and ``augment_query`` is
    used at query time to expand the embedding target, not to rerank a
    fetched candidate list.
    """

    def fake_embed_query(text: str) -> list[float]:
        # Match seed encoding used in `_chunk_with_seed` below.
        # "alpha" → seed 0.01, "beta" → 0.02, "gamma" → 0.03
        seed = {"alpha": 0.01, "beta": 0.02, "gamma": 0.03}.get(text.strip().lower(), 0.5)
        return _seed_vec(seed)

    monkeypatch.setattr(search_service, "embed_query", fake_embed_query)
    monkeypatch.setattr(search_service, "distill_query", lambda _q: None)
    monkeypatch.setattr(search_service, "augment_query", lambda _q: None)


def _seed_vec(seed: float, dim: int = 384) -> list[float]:
    return [seed + i * 0.0001 for i in range(dim)]


def _chunk(email_id: str, idx: int, text: str, *, seed: float, source_type="body", source_name=None) -> Chunk:
    cid = f"{email_id}::{source_type}::{idx}" + (f"::{source_name}" if source_name else "")
    return Chunk(
        chunk_id=cid,
        email_id=email_id,
        source_type=source_type,
        source_name=source_name,
        chunk_index=idx,
        chunk_text=text,
        embedding=_seed_vec(seed),
    )


def _email(
    eid: str,
    subject: str,
    body: str,
    atts=None,
    *,
    summary: str | None = None,
    from_address: str = "alice@example.com",
    from_name: str | None = "Alice",
    folder_id: str | None = None,
    folder_name: str | None = None,
    received_at: int | None = None,
) -> EmailRow:
    return EmailRow(
        id=eid,
        subject=subject,
        from_address=from_address,
        from_name=from_name,
        to_addresses=[EmailAddress(address="bob@example.com")],
        received_at=received_at if received_at is not None else int(time.time()),
        folder_id=folder_id,
        folder_name=folder_name,
        body_text=body,
        body_html=f"<p>{body}</p>",
        summary=summary,
        attachments=atts or [],
        has_attachments=bool(atts),
    )


def _seed_corpus(conn) -> None:
    # m1: body about "alpha"
    insert_email_with_chunks(
        conn,
        _email("m1", "Project alpha", "we discussed alpha milestones"),
        [_chunk("m1", 0, "we discussed alpha milestones", seed=0.01)],
    )
    # m2: body about "beta"
    insert_email_with_chunks(
        conn,
        _email("m2", "Project beta", "the beta plan is ready"),
        [_chunk("m2", 0, "the beta plan is ready", seed=0.02)],
    )
    # m3: PDF attachment mentions "gamma"
    att = AttachmentRecord(
        att_id="att-3",
        name="report.pdf",
        content_type="application/pdf",
        size=100,
        extracted_text="gamma analysis details",
        status="ok",
    )
    insert_email_with_chunks(
        conn,
        _email("m3", "Q3", "see attached", atts=[att]),
        [
            _chunk("m3", 0, "see attached", seed=0.5),
            _chunk("m3", 0, "gamma analysis details", seed=0.03, source_type="attachment", source_name="report.pdf"),
        ],
    )


def test_keyword_search_finds_body(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="keyword")
    ids = [h.email_id for h in resp.hits]
    assert "m1" in ids


def test_keyword_search_finds_attachment(conn) -> None:
    """Attachment text is in searchable_text → FTS finds it; matched_in='attachment'."""
    _seed_corpus(conn)
    resp = search_service.search(conn, "gamma", mode="keyword")
    ids = [h.email_id for h in resp.hits]
    assert "m3" in ids
    hit = next(h for h in resp.hits if h.email_id == "m3")
    assert hit.matched_in == "attachment"
    assert hit.matched_attachment_name == "report.pdf"


def test_semantic_search_returns_attachment_match(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "gamma", mode="semantic")
    assert resp.hits, "expected at least one semantic hit"
    top = resp.hits[0]
    assert top.email_id == "m3"
    assert top.matched_in == "attachment"
    assert top.matched_attachment_name == "report.pdf"


def test_hybrid_search_unions_results(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="hybrid")
    ids = [h.email_id for h in resp.hits]
    assert "m1" in ids


def test_empty_query_returns_no_hits(conn) -> None:
    _seed_corpus(conn)
    resp = search_service.search(conn, "   ", mode="hybrid")
    assert resp.hits == []


def test_keyword_query_with_apostrophe_does_not_raise(conn) -> None:
    """Apostrophes (contractions, names like 'O'Brien', or transliterated CJK
    like 'kai'xin') used to crash FTS5 with 'syntax error near "\\''" — the
    parser treats ``'`` as a phrase delimiter. The query sanitiser now wraps
    each token as a quoted phrase, making any embedded punctuation literal.
    """
    insert_email_with_chunks(
        conn,
        _email("apos", "don't forget the kai'xin meeting", "details inside"),
        [_chunk("apos", 0, "don't forget the kai'xin meeting", seed=0.01)],
    )
    # Each of these used to raise sqlite3.OperationalError.
    for q in ["don't", "kai'xin", "O'Brien", "上个月kai'xin"]:
        resp = search_service.search(conn, q, mode="keyword")
        # We only assert no-crash + that the parens didn't smuggle in a
        # phantom operator; whether each query matches the seeded corpus is
        # tokenizer-specific (trigram needs >=3 contiguous chars).
        assert isinstance(resp.hits, list)


def test_keyword_query_with_parens_and_star_does_not_raise(conn) -> None:
    """Parentheses and ``*`` are FTS5 syntax — wrapping each token in
    quotes makes them literal so a search for 'plan*' or 'foo(bar)' no
    longer crashes."""
    _seed_corpus(conn)
    for q in ["plan*", "foo(bar)", "a:b", "x-y"]:
        resp = search_service.search(conn, q, mode="keyword")
        assert isinstance(resp.hits, list)


def test_keyword_query_with_embedded_quote(conn) -> None:
    """A literal double-quote in the query gets escaped per FTS5 rules
    (``"`` → ``""``) and the search runs without error."""
    insert_email_with_chunks(
        conn,
        _email("quoted", 'the "alpha" rollout', "see attached"),
        [_chunk("quoted", 0, 'the "alpha" rollout', seed=0.01)],
    )
    resp = search_service.search(conn, 'say "hi"', mode="keyword")
    assert isinstance(resp.hits, list)


def test_keyword_query_finds_apostrophe_term(conn) -> None:
    """End-to-end: an indexed token containing an apostrophe IS findable
    via the keyword path now that the sanitiser preserves it."""
    insert_email_with_chunks(
        conn,
        _email("ob", "meeting with O'Brien", "agenda inside"),
        [_chunk("ob", 0, "meeting with O'Brien", seed=0.01)],
    )
    resp = search_service.search(conn, "O'Brien", mode="keyword")
    assert [h.email_id for h in resp.hits] == ["ob"]


# ---------------------------------------------------------------------------
# FTS query builder: joiner + sub-trigram filtering
# ---------------------------------------------------------------------------


def test_to_fts_query_default_joiner_is_and() -> None:
    """Two tokens with the default AND joiner produce juxtaposed quoted
    phrases — FTS5's implicit-AND syntax. Bare whitespace between phrases
    means 'both must match'."""
    built = search_service._to_fts_query("alpha beta")
    assert built.joiner == "AND"
    assert built.query == '"alpha" "beta"'
    assert built.kept_tokens == ["alpha", "beta"]
    assert built.dropped_short_tokens == []


def test_to_fts_query_or_joiner_uses_explicit_or() -> None:
    """With joiner='OR' the phrases are joined by the uppercase ``OR``
    operator — required for the semantic_fts leg to treat the LLM-emitted
    bag of alternative keywords as 'any of these is enough', not 'all of
    these must appear'."""
    built = search_service._to_fts_query("alpha beta gamma", joiner="OR")
    assert built.joiner == "OR"
    assert built.query == '"alpha" OR "beta" OR "gamma"'
    assert built.kept_tokens == ["alpha", "beta", "gamma"]


def test_to_fts_query_drops_sub_trigram_tokens() -> None:
    """The trigram tokenizer generates no tokens for strings shorter
    than 3 chars, so phrases like ``"工会"`` (2 chars) or ``"Q3"`` (2
    chars) are unmatchable AND poison any AND-joined query they appear
    in. The builder drops them up-front and surfaces them in
    ``dropped_short_tokens`` so the trace can explain the omission."""
    built = search_service._to_fts_query("Q3 budget 工会 微软工会")
    assert built.kept_tokens == ["budget", "微软工会"]
    assert built.dropped_short_tokens == ["Q3", "工会"]
    # The standalone short phrases never appear as their own quoted FTS
    # phrase; only the surviving tokens are emitted.
    assert built.query == '"budget" "微软工会"'


def test_to_fts_query_all_short_tokens_returns_empty_query() -> None:
    """When every token is below the trigram minimum, the builder
    returns an empty ``query`` so the caller short-circuits the search
    rather than running an empty MATCH expression."""
    built = search_service._to_fts_query("Q3 工会 a", joiner="OR")
    assert built.query == ""
    assert built.kept_tokens == []
    assert built.dropped_short_tokens == ["Q3", "工会", "a"]


def test_to_fts_query_or_joiner_with_mixed_lengths() -> None:
    """OR joiner + sub-trigram filter combined: short tokens are dropped
    and the survivors are OR-joined. The combination is what makes the
    semantic_fts leg recover hits for a bilingual distill like
    'Q3 budget approval 工会' (Q3 + 工会 dropped, OR-join of the rest)."""
    built = search_service._to_fts_query("Q3 budget approval 工会", joiner="OR")
    assert built.kept_tokens == ["budget", "approval"]
    assert built.dropped_short_tokens == ["Q3", "工会"]
    assert built.query == '"budget" OR "approval"'


def test_semantic_fts_leg_or_joins_distilled_alternatives(conn, monkeypatch) -> None:
    """The semantic_fts leg must OR-join distilled tokens. We seed an
    email that contains only ONE of the distilled alternatives ('union')
    and confirm the leg surfaces it — proving the leg is not AND-ing the
    bag and requiring every alternative to be present.

    Regression: prior to the OR fix this leg would AND-join a bilingual
    distill like '工会 labor union Mahua FunAge' and return 0 hits for any
    email that didn't contain ALL terms (i.e. every real email)."""
    insert_email_with_chunks(
        conn,
        _email("union1", "trade union members welcome", "see the notice attached"),
        [_chunk("union1", 0, "trade union members welcome", seed=0.5)],
    )
    # The other distilled alternatives ('labor', 'Mahua', 'FunAge') are
    # NOT present in the corpus. AND-joining would return 0; OR-joining
    # returns the row because 'union' matches.
    monkeypatch.setattr(
        search_service,
        "distill_query",
        lambda _q: "工会 labor union Mahua FunAge",
    )
    resp = search_service.search(conn, "上个月工会发的开心麻花的邮件", mode="semantic")
    ids = [h.email_id for h in resp.hits]
    assert "union1" in ids, (
        f"semantic_fts should OR-join the distilled bag; got hits={ids}"
    )


def test_semantic_fts_leg_trace_records_or_joiner_and_dropped_tokens(
    conn, monkeypatch
) -> None:
    """The leg trace must expose the joiner + the per-token bookkeeping
    so the user can see why a short keyword (``工会``) was dropped and
    confirm the leg is OR-joining (not AND-joining) the surviving
    alternatives."""
    insert_email_with_chunks(
        conn,
        _email("u1", "trade union notice", "members welcome"),
        [_chunk("u1", 0, "trade union notice", seed=0.5)],
    )
    monkeypatch.setattr(
        search_service,
        "distill_query",
        lambda _q: "工会 union labor",
    )
    resp = search_service.search(conn, "anything", mode="semantic")
    assert resp.debug is not None
    fts_trace = resp.debug["legs"]["semantic_fts"]
    assert fts_trace["fts_joiner"] == "OR"
    assert "工会" in fts_trace["fts_dropped_short_tokens"]
    assert "union" in fts_trace["fts_kept_tokens"]
    assert "labor" in fts_trace["fts_kept_tokens"]
    # The final MATCH expression is OR-joined and excludes the dropped
    # 2-char token.
    assert " OR " in fts_trace["fts_query"]
    assert "工会" not in fts_trace["fts_query"]


def test_keyword_leg_trace_records_and_joiner_and_dropped_tokens(conn) -> None:
    """The keyword leg keeps its AND semantics (verbatim user query —
    every typed term must match) but still surfaces short-token drops so
    a user who typed 'Q3 budget' can see why 'Q3' had no effect."""
    insert_email_with_chunks(
        conn,
        _email("b1", "budget review", "Q3 numbers attached"),
        [_chunk("b1", 0, "budget review Q3 numbers attached", seed=0.5)],
    )
    resp = search_service.search(conn, "Q3 budget", mode="keyword")
    assert resp.debug is not None
    kw_trace = resp.debug["legs"]["keyword"]
    assert kw_trace["fts_joiner"] == "AND"
    assert kw_trace["fts_dropped_short_tokens"] == ["Q3"]
    assert kw_trace["fts_kept_tokens"] == ["budget"]
    # AND-joined survivors — implicit AND is just whitespace between
    # phrases, no explicit operator.
    assert kw_trace["fts_query"] == '"budget"'


# ---------------------------------------------------------------------------
# Word-boundary post-filter: reject trigram substring false positives
# ---------------------------------------------------------------------------


def test_is_cjk_token_detects_cjk_and_rejects_ascii() -> None:
    """CJK characters in the Unified Ideographs / Hiragana / Katakana /
    Hangul ranges are flagged; pure-ASCII tokens are not. Mixed tokens
    (e.g. an English suffix on a Chinese stem) are flagged CJK — the
    verifier uses substring matching for them, which is correct since
    the CJK portion has no word boundary in the western sense."""
    assert search_service._is_cjk_token("工会") is True
    assert search_service._is_cjk_token("微软工会") is True
    assert search_service._is_cjk_token("こんにちは") is True   # Hiragana
    assert search_service._is_cjk_token("カタカナ") is True     # Katakana
    assert search_service._is_cjk_token("한국어") is True       # Hangul
    assert search_service._is_cjk_token("labor") is False
    assert search_service._is_cjk_token("Mahua") is False
    assert search_service._is_cjk_token("") is False
    # Mixed: any CJK char wins.
    assert search_service._is_cjk_token("工会2026") is True


def test_token_appears_with_boundary_ascii_requires_word_boundary() -> None:
    """ASCII tokens use ``\\b`` semantics: the token matches when
    preceded by a word boundary. This rejects trigram's substring false
    positives (``labor`` inside ``collaboration``) but accepts legitimate
    suffixes (``labor`` inside ``laborer`` / ``labors`` / ``laboring``).
    """
    fn = search_service._token_appears_with_boundary
    # Positive — bare word.
    assert fn("labor", "they discussed labor relations") is True
    # Positive — suffixed words still start at a word boundary.
    assert fn("labor", "the laborer arrived") is True
    assert fn("labor", "laboring on the report") is True
    assert fn("labor", "labors of love") is True
    # Negative — embedded inside another word (the bug we're fixing).
    assert fn("labor", "interdepartmental collaboration") is False
    assert fn("labor", "belabor the point") is False
    assert fn("labor", "laboratory equipment") is True  # 'lab' at boundary, but 'labor' prefix is also at boundary
    # Negative — prefix-of-something with no boundary.
    assert fn("union", "the communion ceremony") is False
    assert fn("union", "trade union members") is True
    # Punctuation counts as a word boundary.
    assert fn("alice", "ping (alice) on this") is True
    assert fn("alice", "alice@example.com") is True


def test_token_appears_with_boundary_cjk_uses_substring() -> None:
    """CJK tokens fall back to substring matching — ``\\b`` is defined on
    ``\\w`` which excludes CJK characters in Python's default regex
    mode, so the word-boundary anchor would never match for them. The
    indexed FTS trigrams use substring semantics for CJK too, so this
    matches what the user sees in the snippet."""
    fn = search_service._token_appears_with_boundary
    # CJK substring matches "anywhere" — including inside a longer term.
    assert fn("工会", "微软工会邀您参加活动") is True
    assert fn("微软工会", "微软工会邀您参加活动") is True
    # Absent → no match.
    assert fn("开心麻花", "微软工会邀您参加活动") is False


def test_keyword_leg_drops_collaboration_for_labor_query(conn) -> None:
    """Regression: trigram tokenizer matches 'labor' as a substring inside
    'collaboration', surfacing irrelevant emails. The word-boundary
    post-filter rejects those hits before they reach the user."""
    # False positive — 'labor' only appears inside 'collaboration'.
    insert_email_with_chunks(
        conn,
        _email("collab", "Team collaboration tools", "Improve team collaboration today."),
        [_chunk("collab", 0, "Improve team collaboration today.", seed=0.5)],
    )
    # Legitimate hit — 'labor' appears as a standalone word.
    insert_email_with_chunks(
        conn,
        _email("real", "Labor day announcement", "Office closed for labor day."),
        [_chunk("real", 0, "Office closed for labor day.", seed=0.51)],
    )
    resp = search_service.search(conn, "labor", mode="keyword")
    ids = [h.email_id for h in resp.hits]
    assert "real" in ids, f"legitimate labor match dropped: {ids}"
    assert "collab" not in ids, (
        f"substring false-positive 'collaboration' should have been filtered: {ids}"
    )


def test_semantic_fts_leg_drops_collaboration_for_distilled_labor(conn, monkeypatch) -> None:
    """Same regression but for the semantic_fts leg, where the LLM's
    distilled bag of alternatives ('labor union Mahua FunAge') would
    otherwise surface every 'collaboration' mention via the substring
    false positive on 'labor'.

    Reproduces the screenshot the user filed: '[PROD] Sev 4 ... [Fabric]
    [Notebook][Collaboration]' was appearing because the FTS leg
    substring-matched 'labor' inside 'collaboration', not because it had
    anything to do with union / labor topics.

    We exercise ``run_leg('semantic_fts', ...)`` directly instead of the
    full ``search(..., mode='semantic')`` wrapper — the embedding leg has
    its own retrieval path that's unaffected by the word-boundary filter,
    and the stub embedder in conftest treats both seeded chunks as
    near-identical, which would let 'collab' leak into a merged result
    via the KNN leg even when the FTS leg correctly rejected it."""
    insert_email_with_chunks(
        conn,
        _email("collab", "Fabric Notebook Collaboration sev 4", "Improve collaboration in the notebook."),
        [_chunk("collab", 0, "Improve collaboration in the notebook.", seed=0.5)],
    )
    insert_email_with_chunks(
        conn,
        _email("union", "Checkout instructions from MS China", "Trade Union Committee notice"),
        [_chunk("union", 0, "Trade Union Committee notice", seed=0.51)],
    )
    monkeypatch.setattr(
        search_service,
        "distill_query",
        lambda _q: "labor union Mahua FunAge",
    )
    leg = search_service.run_leg(
        "semantic_fts",
        conn,
        "上个月工会发的开心麻花的邮件",
        limit=20,
        filters=search_service.SearchFilters(),
        debug=True,
    )
    ids = [h.email_id for h in leg.hits]
    assert "union" in ids, f"legitimate 'Trade Union' match dropped: {ids}"
    assert "collab" not in ids, (
        f"'collaboration' substring false-positive not filtered: {ids}"
    )
    # And the trace records the filter actually ran.
    assert leg.trace is not None
    assert leg.trace["fts_substring_filtered_count"] >= 1
    assert leg.trace["fts_joiner"] == "OR"


def test_keyword_leg_preserves_cjk_substring_matches(conn) -> None:
    """CJK matches still use substring semantics (no word boundaries in
    CJK), so a 3-char query like '微软工会' still finds emails containing
    '微软工会邀您参加...' even though the full subject is longer than the
    query. The word-boundary filter is ASCII-only by design."""
    insert_email_with_chunks(
        conn,
        _email("cjk", "微软工会邀您参加2026北京同乐日活动", "see attached"),
        [_chunk("cjk", 0, "微软工会邀您参加2026北京同乐日活动", seed=0.5)],
    )
    resp = search_service.search(conn, "微软工会", mode="keyword")
    ids = [h.email_id for h in resp.hits]
    assert ids == ["cjk"], f"CJK substring match dropped: {ids}"


def test_fts_trace_records_substring_filter_counts(conn) -> None:
    """The trace exposes the raw FTS count + the substring-filtered count
    so a user inspecting the debug overlay can see WHY their result list
    is shorter than the FTS hit count suggests."""
    # 5 collaboration distractors, 1 real hit. Raw FTS returns all 6
    # (over-fetch is 3x of default limit 20 = 60, plenty of headroom);
    # the post-filter drops 5.
    for i in range(5):
        insert_email_with_chunks(
            conn,
            _email(f"collab-{i}", "Team collaboration", "collaboration is key"),
            [_chunk(f"collab-{i}", 0, "collaboration is key", seed=0.5 + i * 0.001)],
        )
    insert_email_with_chunks(
        conn,
        _email("real", "Labor news", "labor relations update"),
        [_chunk("real", 0, "labor relations update", seed=0.6)],
    )
    resp = search_service.search(conn, "labor", mode="keyword")
    assert resp.debug is not None
    kw_trace = resp.debug["legs"]["keyword"]
    assert kw_trace["fts_raw_count"] == 6
    assert kw_trace["fts_substring_filtered_count"] == 5
    assert len(kw_trace["fts_hits"]) == 1
    assert kw_trace["fts_hits"][0]["email_id"] == "real"


def test_keyword_search_prefers_subject_match(conn) -> None:
    """bm25 column weight on the subject column ranks subject hits above
    body-only hits for the same term."""
    # 'budget' appears only in the body
    insert_email_with_chunks(
        conn,
        _email("body-hit", "team meeting", "we discussed the budget for next quarter"),
        [_chunk("body-hit", 0, "we discussed the budget for next quarter", seed=0.4)],
    )
    # 'budget' is the subject
    insert_email_with_chunks(
        conn,
        _email("subj-hit", "budget review", "see attached deck for the numbers"),
        [_chunk("subj-hit", 0, "see attached deck for the numbers", seed=0.41)],
    )
    resp = search_service.search(conn, "budget", mode="keyword")
    ids = [h.email_id for h in resp.hits]
    assert ids[:2] == ["subj-hit", "body-hit"], (
        f"subject match should outrank body-only match; got {ids}"
    )


def test_search_hit_carries_llm_summary(conn) -> None:
    """When `email.summary` is set, every SearchHit for that email exposes it
    so the UI can render a 'topical summary' card above the snippet."""
    insert_email_with_chunks(
        conn,
        _email(
            "sum1",
            "Project alpha",
            "the alpha rollout went well",
            summary="Alpha rolled out successfully; no incidents.",
        ),
        [_chunk("sum1", 0, "the alpha rollout went well", seed=0.01)],
    )
    resp = search_service.search(conn, "alpha", mode="hybrid")
    assert resp.hits
    top = next(h for h in resp.hits if h.email_id == "sum1")
    assert top.summary == "Alpha rolled out successfully; no incidents."


# ---------------------------------------------------------------------------
# Semantic ranking: summary-chunk match promotes above body-only matches
# ---------------------------------------------------------------------------


def test_summary_chunk_match_promotes_above_better_body_match(conn) -> None:
    """The new semantic strategy: an email whose SUMMARY chunk matched the
    query KNN beats an email whose only matching chunk was body/attachment,
    EVEN IF the body chunk has a smaller distance to the query than the
    summary chunk does. This is the entire point of indexing the summary —
    "topical match" is a stronger signal than "incidental body mention",
    and we encode that as a hard bucket separation inside the
    ``semantic_knn`` leg.
    """
    # Email A: only a body chunk that matches the 'alpha' query nearly perfectly.
    insert_email_with_chunks(
        conn,
        _email("a-body-only", "unrelated subject", "alpha discussion text"),
        [_chunk("a-body-only", 0, "alpha discussion text", seed=0.01)],
    )
    # Email B: body chunk is FAR from 'alpha', BUT a summary chunk is
    # somewhat close. Promotion should put it on top of A despite A's
    # tighter body match.
    insert_email_with_chunks(
        conn,
        _email(
            "b-summary-match",
            "weekly status",
            "irrelevant body text",
            summary="alpha topic summary",
        ),
        [
            _chunk("b-summary-match", 0, "irrelevant body text", seed=0.5),
            # seed=0.02 is *worse* than A's body seed=0.01 but still in the
            # KNN top-K so the summary-match path triggers.
            _chunk(
                "b-summary-match",
                0,
                "alpha topic summary",
                seed=0.02,
                source_type="summary",
                source_name=None,
            ),
        ],
    )

    resp = search_service.search(conn, "alpha", mode="semantic")
    ids = [h.email_id for h in resp.hits]
    assert ids[0] == "b-summary-match", (
        f"summary-match email must outrank body-only match; got {ids}"
    )
    # Sanity: the promoted hit's score is strictly above 1.0 (the promotion
    # base) and the non-promoted hit's score is in [0, 1].
    promoted = next(h for h in resp.hits if h.email_id == "b-summary-match")
    body_only = next(h for h in resp.hits if h.email_id == "a-body-only")
    assert promoted.score > 1.0
    assert 0.0 <= body_only.score <= 1.0


def test_summary_match_ordering_within_bucket_by_summary_distance(conn) -> None:
    """Two emails both have a summary-chunk hit; the one with the closer
    summary chunk ranks first. Inside the "promoted" bucket the ranking is
    still distance-driven so close topical matches beat far topical matches.
    """
    insert_email_with_chunks(
        conn,
        _email("close", "s1", "body1", summary="alpha summary close"),
        [
            _chunk("close", 0, "body1", seed=0.5),
            _chunk("close", 0, "alpha summary close", seed=0.01,
                   source_type="summary", source_name=None),
        ],
    )
    insert_email_with_chunks(
        conn,
        _email("far", "s2", "body2", summary="alpha summary far"),
        [
            _chunk("far", 0, "body2", seed=0.5),
            _chunk("far", 0, "alpha summary far", seed=0.05,
                   source_type="summary", source_name=None),
        ],
    )

    resp = search_service.search(conn, "alpha", mode="semantic")
    ids = [h.email_id for h in resp.hits]
    assert ids.index("close") < ids.index("far"), (
        f"closer summary chunk should rank higher within the promoted bucket; got {ids}"
    )


def test_no_promotion_falls_back_to_chunk_distance_order(conn) -> None:
    """When NO email in the candidate pool has a matching summary chunk,
    ranking reduces to the original chunk-distance order — the new
    strategy doesn't change behaviour on legacy/unsummarized corpora."""
    insert_email_with_chunks(
        conn,
        _email("near", "subject A", "body about alpha"),
        [_chunk("near", 0, "body about alpha", seed=0.01)],
    )
    insert_email_with_chunks(
        conn,
        _email("far", "subject B", "body about beta"),
        [_chunk("far", 0, "body about beta", seed=0.5)],
    )

    resp = search_service.search(conn, "alpha", mode="semantic")
    ids = [h.email_id for h in resp.hits]
    assert ids[0] == "near"
    # Both scores live in [0, 1] since neither was promoted.
    assert all(0.0 <= h.score <= 1.0 for h in resp.hits)


def test_snippet_prefers_body_text_even_when_summary_matched(conn) -> None:
    """Snippet selection: when an email has BOTH a body hit and a summary
    hit, the user-facing snippet comes from the body (which carries the
    actual quoted content). The summary is shown separately via the
    `summary` field — duplicating it as snippet would waste a row of UI.
    """
    insert_email_with_chunks(
        conn,
        _email(
            "both-hit",
            "alpha topic",
            "the alpha rollout went smoothly across all regions",
            summary="alpha summary text",
        ),
        [
            _chunk("both-hit", 0,
                   "the alpha rollout went smoothly across all regions",
                   seed=0.01),
            _chunk("both-hit", 0, "alpha summary text", seed=0.01,
                   source_type="summary", source_name=None),
        ],
    )
    resp = search_service.search(conn, "alpha", mode="semantic")
    top = next(h for h in resp.hits if h.email_id == "both-hit")
    # Snippet is from the body chunk, not the summary chunk.
    assert "rollout" in top.snippet
    assert "summary text" not in top.snippet
    # Summary is still surfaced separately for the UI to render above.
    assert top.summary == "alpha summary text"


def test_summary_only_match_uses_summary_text_as_snippet(conn) -> None:
    """Edge case: an email where the body chunk fell out of the KNN top-K
    but the summary chunk made it in. There's no body chunk to draw a
    snippet from, so we fall back to the summary text — better than
    showing an empty snippet."""
    # Use an out-of-vocab embed query so the body chunk (seed 0.5) is the
    # only thing in the corpus; we explicitly only insert a summary chunk
    # for the email of interest to simulate "summary survived, body didn't".
    insert_email_with_chunks(
        conn,
        _email("summary-only", "s", "body about alpha", summary="topical alpha summary"),
        [
            _chunk("summary-only", 0, "topical alpha summary", seed=0.01,
                   source_type="summary", source_name=None),
            # No body chunk inserted in vec0 — simulates body falling out
            # of KNN headroom on a real corpus.
        ],
    )
    resp = search_service.search(conn, "alpha", mode="semantic")
    top = next(h for h in resp.hits if h.email_id == "summary-only")
    assert "topical alpha summary" in top.snippet
    # Hit is still promoted, since the summary chunk matched.
    assert top.score > 1.0


# ---------------------------------------------------------------------------
# Hard filters (time range, sender, folder)
# ---------------------------------------------------------------------------


def _seed_filterable_corpus(conn) -> None:
    """Three emails with varied received_at / sender / folder so each filter
    dimension can be exercised independently."""
    # Two epoch anchors a year apart so any reasonable date filter cleanly
    # separates "old" from "new".
    OLD = 1_600_000_000   # 2020-09-13
    NEW = 1_700_000_000   # 2023-11-14
    insert_email_with_chunks(
        conn,
        _email(
            "f-old-alice-inbox",
            "alpha discussion",
            "we talked about alpha milestones",
            from_address="alice@example.com",
            folder_id="inbox",
            folder_name="Inbox",
            received_at=OLD,
        ),
        [_chunk("f-old-alice-inbox", 0, "we talked about alpha milestones", seed=0.01)],
    )
    insert_email_with_chunks(
        conn,
        _email(
            "f-new-alice-projects",
            "alpha rollout plan",
            "rolling out alpha to all customers",
            from_address="alice@example.com",
            folder_id="projects",
            folder_name="Projects",
            received_at=NEW,
        ),
        [_chunk("f-new-alice-projects", 0, "rolling out alpha to all customers", seed=0.01)],
    )
    insert_email_with_chunks(
        conn,
        _email(
            "f-new-bob-inbox",
            "alpha question",
            "do you have an alpha update?",
            from_address="bob@example.com",
            folder_id="inbox",
            folder_name="Inbox",
            received_at=NEW,
        ),
        [_chunk("f-new-bob-inbox", 0, "do you have an alpha update?", seed=0.01)],
    )


def test_filters_date_range_excludes_old(conn) -> None:
    """A start_at after the OLD email's timestamp drops it from results."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(start_at=1_650_000_000)
    resp = search_service.search(conn, "alpha", mode="keyword", filters=filters)
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-new-alice-projects", "f-new-bob-inbox"}


def test_filters_date_range_excludes_new(conn) -> None:
    """An end_at before the NEW emails' timestamps drops both."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(end_at=1_650_000_000)
    resp = search_service.search(conn, "alpha", mode="keyword", filters=filters)
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-old-alice-inbox"}


def test_filters_sender_is_case_insensitive(conn) -> None:
    """from_address filter matches case-insensitively against e.from_address."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(from_address="ALICE@example.COM")
    resp = search_service.search(conn, "alpha", mode="keyword", filters=filters)
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-old-alice-inbox", "f-new-alice-projects"}


def test_filters_folder_id(conn) -> None:
    """folder_id filter is an exact match."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(folder_id="projects")
    resp = search_service.search(conn, "alpha", mode="keyword", filters=filters)
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-new-alice-projects"}


def test_filters_combine_with_AND_semantics(conn) -> None:
    """All filter fields are ANDed: sender + folder narrows to one email."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(
        from_address="alice@example.com",
        folder_id="inbox",
    )
    resp = search_service.search(conn, "alpha", mode="keyword", filters=filters)
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-old-alice-inbox"}


def test_filters_apply_to_semantic_mode(conn) -> None:
    """Filters narrow the semantic-mode result set too — vec0 has no WHERE
    clause for aux columns, so this is done Python-side after KNN."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(folder_id="inbox")
    resp = search_service.search(conn, "alpha", mode="semantic", filters=filters)
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-old-alice-inbox", "f-new-bob-inbox"}
    # The projects-folder email — same chunk seed, would otherwise match — is gone.
    assert "f-new-alice-projects" not in ids


def test_filters_apply_to_hybrid_mode(conn) -> None:
    """Hybrid mode applies filters to both keyword and semantic legs."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(from_address="bob@example.com")
    resp = search_service.search(conn, "alpha", mode="hybrid", filters=filters)
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-new-bob-inbox"}


def test_filters_default_returns_everything(conn) -> None:
    """No filters arg / empty SearchFilters() == no narrowing."""
    _seed_filterable_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="keyword")
    ids = {h.email_id for h in resp.hits}
    assert ids == {"f-old-alice-inbox", "f-new-alice-projects", "f-new-bob-inbox"}


def test_filters_overly_restrictive_returns_empty(conn) -> None:
    """An empty result set when filters reject everything — not an error."""
    _seed_filterable_corpus(conn)
    filters = search_service.SearchFilters(from_address="nobody@example.com")
    resp = search_service.search(conn, "alpha", mode="hybrid", filters=filters)
    assert resp.hits == []


def test_filters_is_active() -> None:
    """SearchFilters.is_active() is False only for the all-None default."""
    assert search_service.SearchFilters().is_active() is False
    assert search_service.SearchFilters(start_at=1).is_active() is True
    assert search_service.SearchFilters(end_at=1).is_active() is True
    assert search_service.SearchFilters(from_address="a@b").is_active() is True
    assert search_service.SearchFilters(folder_id="x").is_active() is True


# ---------------------------------------------------------------------------
# Query transformations — semantic mode: distilled -> FTS, augmented -> KNN
# ---------------------------------------------------------------------------


def test_semantic_embeds_augmented_query_not_distilled(conn, monkeypatch) -> None:
    """In the dual-query design, ``augment_query`` (NOT ``distill_query``)
    drives the embedding leg. We verify by stubbing distill='foo' (a value
    that wouldn't match anything) and augment='alpha' (a value that maps
    via the fake embedder to m1's chunk seed). If augment drove the KNN,
    m1 surfaces; if distill had been used by mistake, the result set would
    be wrong."""

    captured: list[str] = []

    def fake_embed_query(text: str) -> list[float]:
        captured.append(text)
        # 'alpha' (augmented) → tight to m1's chunk seed=0.01.
        # Anything else lands at the generic bucket far from all seeds.
        seed = 0.01 if text.strip().lower() == "alpha" else 0.5
        return _seed_vec(seed)

    monkeypatch.setattr(search_service, "embed_query", fake_embed_query)
    monkeypatch.setattr(search_service, "distill_query", lambda _q: "foo-distilled")
    monkeypatch.setattr(search_service, "augment_query", lambda _q: "alpha")

    _seed_corpus(conn)
    raw_query = "help me find the email about alpha"
    resp = search_service.search(conn, raw_query, mode="semantic")
    assert resp.hits

    # Augmented phrase reached the embedder; raw filler did not.
    assert "alpha" in captured
    assert raw_query not in captured
    # Distilled phrase MUST NOT have been embedded — that's the FTS leg's job.
    assert "foo-distilled" not in captured

    # And the email that the augmented embedding pointed at surfaces.
    assert any(h.email_id == "m1" for h in resp.hits)


def test_semantic_uses_distilled_query_for_fts_leg(conn, monkeypatch) -> None:
    """The distilled phrase drives the keyword (FTS) leg inside semantic
    mode. We spy on ``search_fts`` directly to confirm it received the
    distilled query (post-sanitisation) and NOT the raw natural-language
    string — this is the contract the user asked for: distilled goes to
    FTS, augmented goes to embeddings.
    """
    fts_calls: list[str] = []
    real_search_fts = search_service.search_fts

    def spy_search_fts(conn, query, **kwargs):
        fts_calls.append(query)
        return real_search_fts(conn, query, **kwargs)

    monkeypatch.setattr(search_service, "search_fts", spy_search_fts)
    monkeypatch.setattr(search_service, "distill_query", lambda _q: "alpha")
    monkeypatch.setattr(search_service, "augment_query", lambda _q: "alpha rollout")

    _seed_corpus(conn)
    raw = "help me find the email about alpha"
    search_service.search(conn, raw, mode="semantic")

    # search_fts was called by the semantic FTS leg with the distilled
    # query (wrapped as an FTS phrase by ``_to_fts_query``). The raw
    # natural-language query MUST NOT have reached FTS.
    assert fts_calls, "FTS leg never ran inside semantic mode"
    assert any('"alpha"' in q for q in fts_calls), (
        f"distilled phrase missing from FTS calls: {fts_calls!r}"
    )
    assert not any(raw in q for q in fts_calls), (
        f"raw filler-heavy query leaked into FTS: {fts_calls!r}"
    )


def test_semantic_falls_back_to_raw_when_both_llm_paths_return_none(
    conn, monkeypatch
) -> None:
    """distill_query AND augment_query both returning None (LLM disabled /
    failure) must NOT break semantic search — the FTS leg uses the raw
    query and the KNN leg embeds the raw query directly."""

    captured: list[str] = []

    def fake_embed_query(text: str) -> list[float]:
        captured.append(text)
        # Match on the raw query so we can prove it was used.
        seed = 0.01 if "alpha" in text.lower() else 0.5
        return _seed_vec(seed)

    monkeypatch.setattr(search_service, "embed_query", fake_embed_query)
    monkeypatch.setattr(search_service, "distill_query", lambda _q: None)
    monkeypatch.setattr(search_service, "augment_query", lambda _q: None)

    _seed_corpus(conn)
    resp = search_service.search(conn, "anything about alpha", mode="semantic")
    assert resp.hits
    # Raw query reached embed_query — no augmented string available.
    assert "anything about alpha" in captured


def test_keyword_mode_does_not_call_llm(conn, monkeypatch) -> None:
    """Keyword mode is the LLM-free path. Neither ``distill_query`` nor
    ``augment_query`` must run for it — bm25 over FTS5 already handles
    term frequency natively, and dragging the LLM into a keyword search
    would (a) add latency and (b) risk dropping terms the user intentionally
    typed."""

    def boom_distill(_q: str) -> str | None:
        raise AssertionError("distill_query must not be called for keyword mode")

    def boom_augment(_q: str) -> str | None:
        raise AssertionError("augment_query must not be called for keyword mode")

    monkeypatch.setattr(search_service, "distill_query", boom_distill)
    monkeypatch.setattr(search_service, "augment_query", boom_augment)

    _seed_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="keyword")
    assert resp.hits  # didn't raise, didn't call either LLM hook


# ---------------------------------------------------------------------------
# Debug payload — diagnostic trace for surprising search results
# ---------------------------------------------------------------------------


def _patch_debug(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    """Force the ``debug_enabled`` setting for one test, leaving llm_enabled
    untouched (defaults to True). The autouse stubs already neutralise
    distill/augment so no real HTTP happens."""
    from emailsearch.config import Settings

    s = Settings(debug_enabled=enabled)
    monkeypatch.setattr(search_service, "get_settings", lambda: s)


def test_debug_disabled_omits_trace(conn, monkeypatch) -> None:
    """When ``debug_enabled=False`` in config, response.debug is None so the
    wire payload stays lean."""
    _patch_debug(monkeypatch, enabled=False)
    _seed_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="hybrid")
    assert resp.debug is None


def test_debug_default_populates_trace_for_semantic(conn, monkeypatch) -> None:
    """Default config (debug_enabled=True) → semantic-mode response carries
    the full transformation chain for every leg so the user can see what
    each LLM hook produced AND which input drove each search leg."""
    monkeypatch.setattr(search_service, "distill_query", lambda _q: "alpha")
    monkeypatch.setattr(search_service, "augment_query", lambda _q: "alpha rollout release")
    _seed_corpus(conn)

    resp = search_service.search(conn, "help me find alpha", mode="semantic")
    assert resp.debug is not None
    assert resp.debug["raw_query"] == "help me find alpha"
    assert resp.debug["mode"] == "semantic"

    # Per-leg traces live under `debug.legs.<source>`. Semantic mode
    # runs two legs — distilled FTS + augmented KNN — so both must be
    # present.
    legs = resp.debug["legs"]
    assert set(legs.keys()) == {"semantic_fts", "semantic_knn"}

    # FTS leg trace: surfaces the LLM-distilled input + the post-
    # sanitization FTS query string + the hit list.
    fts = legs["semantic_fts"]
    assert fts["distilled_query"] == "alpha"
    assert fts["fts_input"] == "alpha"
    assert "fts_query" in fts
    assert isinstance(fts["fts_hits"], list)

    # KNN leg trace: surfaces the augmented input + the threshold +
    # per-chunk top-N + per-email final scores.
    knn = legs["semantic_knn"]
    assert knn["augmented_query"] == "alpha rollout release"
    assert knn["embedded_query"] == "alpha rollout release"
    assert "score_threshold" in knn
    assert knn["summary_promotion_base"] == search_service.SUMMARY_PROMOTION_BASE
    assert isinstance(knn["vec_top_chunks"], list)
    if knn["vec_top_chunks"]:
        top = knn["vec_top_chunks"][0]
        assert {
            "email_id", "source_type", "distance", "score", "chunk_text_preview",
        } <= top.keys()
    assert isinstance(knn["final_scores"], list)
    if knn["final_scores"]:
        fs = knn["final_scores"][0]
        # Per-email row: per-source scores + the promotion flag + the
        # final per-leg score. No cross-leg rank fields — fusion is the
        # frontend's job now.
        assert {
            "email_id", "body_score", "summary_score",
            "summary_promoted", "final_score", "snippet_source",
        } <= fs.keys()


def test_debug_records_distillation_fallback(conn, monkeypatch) -> None:
    """When distill_query / augment_query return None we fall back to the
    raw query — and each leg's trace must reflect that so the user knows
    the LLM hooks didn't actually run (e.g. local LLM server is down)."""
    monkeypatch.setattr(search_service, "distill_query", lambda _q: None)
    monkeypatch.setattr(search_service, "augment_query", lambda _q: None)
    _seed_corpus(conn)

    resp = search_service.search(conn, "alpha", mode="semantic")
    assert resp.debug is not None
    legs = resp.debug["legs"]
    # FTS leg fell back to the raw query.
    assert legs["semantic_fts"]["distilled_query"] is None
    assert legs["semantic_fts"]["fts_input"] == "alpha"
    # KNN leg fell back to embedding the raw query directly.
    assert legs["semantic_knn"]["augmented_query"] is None
    assert legs["semantic_knn"]["embedded_query"] == "alpha"


def test_debug_records_summary_promotion(conn) -> None:
    """When a candidate email matched via its summary chunk, the KNN leg's
    trace marks `summary_promoted=True` and shows a non-None
    `summary_score` so the user can see exactly which results won via
    topical match vs. body match."""
    insert_email_with_chunks(
        conn,
        _email("promoted", "s", "irrelevant", summary="alpha summary"),
        [
            _chunk("promoted", 0, "irrelevant", seed=0.5),
            _chunk("promoted", 0, "alpha summary", seed=0.02,
                   source_type="summary", source_name=None),
        ],
    )
    insert_email_with_chunks(
        conn,
        _email("body-only", "s", "alpha discussion text"),
        [_chunk("body-only", 0, "alpha discussion text", seed=0.01)],
    )

    resp = search_service.search(conn, "alpha", mode="semantic")
    knn = resp.debug["legs"]["semantic_knn"]
    by_id = {row["email_id"]: row for row in knn["final_scores"]}
    assert by_id["promoted"]["summary_promoted"] is True
    assert by_id["promoted"]["summary_score"] is not None
    assert by_id["body-only"]["summary_promoted"] is False
    assert by_id["body-only"]["summary_score"] is None


def test_debug_hybrid_records_all_three_legs(conn) -> None:
    """Hybrid debug carries one trace per leg under `debug.legs.<source>` —
    no RRF / fusion trace, since fusion now happens client-side."""
    _seed_corpus(conn)
    resp = search_service.search(conn, "alpha", mode="hybrid")
    assert resp.debug is not None
    legs = resp.debug["legs"]
    # All three legs ran.
    assert set(legs.keys()) == {"keyword", "semantic_fts", "semantic_knn"}
    # Each leg trace has at least its input and an fts-or-knn hit list.
    assert "fts_hits" in legs["keyword"]
    assert "fts_hits" in legs["semantic_fts"]
    assert "vec_top_chunks" in legs["semantic_knn"]


def test_debug_preview_truncates_long_chunk_text(conn) -> None:
    """The chunk_text_preview field on the KNN leg must be capped so a
    50k-char body doesn't bloat the debug payload."""
    long_body = "alpha " * 5000  # ~30k chars
    insert_email_with_chunks(
        conn,
        _email("long", "alpha", long_body),
        [_chunk("long", 0, long_body, seed=0.01)],
    )
    resp = search_service.search(conn, "alpha", mode="semantic")
    knn = resp.debug["legs"]["semantic_knn"]
    previews = [c["chunk_text_preview"] for c in knn["vec_top_chunks"]]
    # Ellipsis adds 1 char; cap is _DEBUG_PREVIEW_CHARS=160.
    assert all(len(p) <= search_service._DEBUG_PREVIEW_CHARS + 1 for p in previews)
    # And at least one preview was actually truncated.
    assert any(p.endswith("…") for p in previews)


# ---------------------------------------------------------------------------
# Leg architecture — per-leg API surface used by the streaming endpoint
# ---------------------------------------------------------------------------


def test_legs_for_mode_dispatch() -> None:
    """The mode→legs mapping is the contract the streaming endpoint relies
    on: keyword mode → 1 leg, semantic → 2 legs, hybrid → 3 legs."""
    assert search_service.legs_for_mode("keyword") == ["keyword"]
    assert search_service.legs_for_mode("semantic") == ["semantic_fts", "semantic_knn"]
    assert search_service.legs_for_mode("hybrid") == [
        "keyword", "semantic_fts", "semantic_knn",
    ]


def test_run_leg_keyword_returns_scored_hits(conn) -> None:
    """``run_leg`` is the per-leg entry point the streaming endpoint
    invokes from worker threads. Each leg produces a self-scored hit list
    with no cross-leg merging."""
    _seed_corpus(conn)
    result = search_service.run_leg(
        "keyword", conn, "alpha",
        limit=20, filters=search_service.SearchFilters(), debug=False,
    )
    assert result.source == "keyword"
    assert result.hits
    # Every hit has a positive score; bm25-derived scores live in (0, 1].
    assert all(0.0 < h.score <= 1.0 for h in result.hits)
    # Trace is suppressed when debug=False.
    assert result.trace is None


def test_run_leg_semantic_knn_drops_low_score_chunks(conn, monkeypatch) -> None:
    """The semantic_knn leg drops chunks whose per-source similarity is
    below ``settings.semantic_score_threshold``. We crank the threshold
    high enough that the only matching chunk is below it — the leg must
    return zero hits, proving the filter ran, while the FTS legs (which
    are NOT gated by the threshold) still match the same query."""
    # Threshold of 0.95 means distance must be < ~0.053 to pass — only
    # a near-identical embedding makes the cut.
    from emailsearch.config import Settings

    s = Settings(semantic_score_threshold=0.95)
    monkeypatch.setattr(search_service, "get_settings", lambda: s)

    # Seed an email whose chunk has seed=0.5: distance to query (seed=0.01)
    # is ~9.6, score ~0.094 — well below the 0.95 threshold.
    insert_email_with_chunks(
        conn,
        _email("weak-match", "subj", "body about alpha"),
        [_chunk("weak-match", 0, "body about alpha", seed=0.5)],
    )

    knn = search_service.run_leg(
        "semantic_knn", conn, "alpha",
        limit=20, filters=search_service.SearchFilters(), debug=True,
    )
    assert knn.hits == [], (
        f"weak-match chunk should have been dropped by threshold; got {knn.hits}"
    )
    # FTS legs are NOT gated — verify the same query still matches via
    # the keyword leg, proving the threshold only affects embeddings.
    kw = search_service.run_leg(
        "keyword", conn, "alpha",
        limit=20, filters=search_service.SearchFilters(), debug=False,
    )
    assert [h.email_id for h in kw.hits] == ["weak-match"]
    # Threshold + drop count are surfaced in the trace.
    assert knn.trace is not None
    assert knn.trace["score_threshold"] == 0.95
    assert knn.trace["dropped_by_threshold"] >= 1


def test_run_leg_empty_query_returns_empty(conn) -> None:
    """An all-whitespace query short-circuits each leg before any work
    runs — important so the streaming endpoint can fan out without first
    checking for emptiness itself."""
    _seed_corpus(conn)
    for src in ("keyword", "semantic_fts", "semantic_knn"):
        result = search_service.run_leg(
            src, conn, "   ",
            limit=20, filters=search_service.SearchFilters(), debug=False,
        )
        assert result.hits == []
        assert result.source == src


# ---------------------------------------------------------------------------
# Streaming endpoint — NDJSON wire format + parallel-leg behaviour
# ---------------------------------------------------------------------------


def _parse_ndjson(body: str) -> list[dict]:
    """Split an NDJSON response body into parsed JSON records."""
    import json as _json

    return [_json.loads(line) for line in body.splitlines() if line.strip()]


def test_search_stream_emits_meta_hits_done(tmp_path, monkeypatch) -> None:
    """End-to-end smoke test: /api/search/stream emits a meta event, one
    hits event per leg, and a done event. Confirms the NDJSON wire
    format matches what the frontend parser expects."""
    from fastapi.testclient import TestClient

    from emailsearch.config import Settings
    from emailsearch.web.app import create_app

    # Point the app at a fresh temp DB and seed it with one email that
    # matches both the keyword and semantic legs.
    db_path = tmp_path / "stream-test.db"
    s = Settings(db_path=db_path, debug_enabled=True)
    # Override get_settings everywhere it's imported. The route opens
    # the DB via settings.resolved_db_path, and the service reads
    # config inside each leg.
    monkeypatch.setattr("emailsearch.web.routes.search.get_settings", lambda: s)
    monkeypatch.setattr(search_service, "get_settings", lambda: s)
    # LLM hops are no-ops in the test — both legs fall back to the raw
    # query, so we don't need a live model server.
    monkeypatch.setattr(search_service, "distill_query", lambda _q: None)
    monkeypatch.setattr(search_service, "augment_query", lambda _q: None)
    # Same fake embedder as the rest of the suite — "alpha" → seed 0.01.
    monkeypatch.setattr(
        search_service, "embed_query",
        lambda text: _seed_vec(0.01 if text.strip().lower() == "alpha" else 0.5),
    )

    conn = open_connection(db_path)
    apply_schema(conn)
    try:
        insert_email_with_chunks(
            conn,
            _email("stream-1", "alpha rollout", "we shipped alpha last week"),
            [_chunk("stream-1", 0, "we shipped alpha last week", seed=0.01)],
        )
    finally:
        conn.close()

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/search/stream?q=alpha&mode=hybrid&limit=5")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = _parse_ndjson(resp.text)

    # Expected event sequence: meta, then one `hits` event per leg
    # (order undefined — depends on which leg finishes first), then done.
    assert events[0]["type"] == "meta"
    assert events[0]["query"] == "alpha"
    assert events[0]["mode"] == "hybrid"
    assert set(events[0]["sources"]) == {"keyword", "semantic_fts", "semantic_knn"}

    assert events[-1]["type"] == "done"
    assert isinstance(events[-1]["duration_ms"], int)

    hit_events = [e for e in events if e["type"] == "hits"]
    assert {e["source"] for e in hit_events} == {
        "keyword", "semantic_fts", "semantic_knn",
    }
    # Every hits event echoes its source + a list of hits with a score.
    for he in hit_events:
        assert isinstance(he["hits"], list)
        for h in he["hits"]:
            assert "score" in h and "email_id" in h
    # And our seeded email made it into at least one leg's output.
    all_ids = {h["email_id"] for he in hit_events for h in he["hits"]}
    assert "stream-1" in all_ids


def test_search_stream_empty_query_short_circuits(tmp_path, monkeypatch) -> None:
    """An empty / whitespace query emits meta + done with no hits events
    — important so the frontend's "Searching..." spinner clears
    immediately when the user clears the input."""
    from fastapi.testclient import TestClient

    from emailsearch.config import Settings
    from emailsearch.web.app import create_app

    db_path = tmp_path / "stream-empty.db"
    s = Settings(db_path=db_path, debug_enabled=True)
    monkeypatch.setattr("emailsearch.web.routes.search.get_settings", lambda: s)
    monkeypatch.setattr(search_service, "get_settings", lambda: s)

    # Touch the DB to apply schema (the route opens the file; bare
    # missing-DB would 500).
    conn = open_connection(db_path)
    apply_schema(conn)
    conn.close()

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/search/stream?q=%20%20%20&mode=hybrid")
        assert resp.status_code == 200
        events = _parse_ndjson(resp.text)

    types = [e["type"] for e in events]
    assert types == ["meta", "done"], f"unexpected event sequence: {types}"
    assert events[-1]["duration_ms"] == 0
