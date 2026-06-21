"""Search service: per-leg search helpers + a sync merge wrapper.

Three independent **legs** — ``keyword``, ``semantic_fts``, and
``semantic_knn`` — each produce a self-scored list of :class:`SearchHit`.
There is no cross-leg fusion: the caller (the streaming HTTP endpoint or
the sync ``search`` wrapper used by tests) merges per-email scores when
the same email surfaces in multiple legs. The HTTP layer fans the legs
out in parallel and streams each leg's results to the browser as soon as
it lands; the browser inserts new items by score.

Score conventions (all higher = better):
  - ``keyword`` / ``semantic_fts``: ``1 / (1 + bm25_rank)``, in (0, 1].
  - ``semantic_knn``: ``1 / (1 + distance)`` for body/attachment matches
    (in [0, 1]); summary-matched emails are promoted by
    :data:`SUMMARY_PROMOTION_BASE` so their scores land in [1, 2]. The
    promotion is intra-leg — it preserves the "topical match beats
    incidental body mention" signal inside this leg only.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

from emailsearch.config import get_settings
from emailsearch.db.models import EmailRow
from emailsearch.db.repositories import (
    get_emails_by_ids,
    search_fts,
    search_vec,
)
from emailsearch.embed.encoder import embed_query
from emailsearch.summarize import augment_query, distill_query

log = logging.getLogger(__name__)

SearchMode = Literal["keyword", "semantic", "hybrid"]
MatchedIn = Literal["body", "attachment", "both"]
LegSource = Literal["keyword", "semantic_fts", "semantic_knn"]

# Max characters of any single text field we include in the debug payload.
_DEBUG_PREVIEW_CHARS = 160

# Constant added to a summary-matched email's KNN score so it always
# outranks any non-summary match WITHIN the embedding leg. Per-source
# scores live in [0, 1] (from ``1 / (1 + distance)``), so a base of 1.0
# cleanly separates the two buckets while keeping inner ordering
# distance-driven. Intra-leg only — cross-leg merging is the frontend's
# job.
SUMMARY_PROMOTION_BASE = 1.0

# Minimum token length that the FTS5 trigram tokenizer can match. The
# tokenizer produces no tokens for strings shorter than 3 chars, so a
# phrase like "工会" or "Q3" returns 0 hits AND poisons any AND-joined
# query it appears in. We strip such tokens before building the MATCH
# expression and surface them in the per-leg trace.
_TRIGRAM_MIN_CHARS = 3

# How many extra FTS hits to fetch from SQL to leave headroom for the
# word-boundary post-filter (:func:`_verify_fts_hit`), which can drop a
# large fraction of hits when an ASCII fragment substring-matches a longer
# word (e.g. "labor" matches every "collaboration"). Raw bm25 over a
# contentless FTS5 table is sub-ms, so the over-fetch cost is negligible.
_FTS_OVERFETCH_FACTOR = 3

# CJK code-point ranges. Used by :func:`_is_cjk_token` to pick matching
# semantics: ASCII tokens use ``\b`` word boundaries; CJK tokens use
# plain substring (no inter-character word boundaries in the script).
_CJK_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u3400", "\u4dbf"),  # CJK Unified Ideographs Extension A
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\uac00", "\ud7af"),  # Hangul Syllables
    ("\uf900", "\ufaff"),  # CJK Compatibility Ideographs
)

FtsJoiner = Literal["AND", "OR"]


class SearchHit(BaseModel):
    email_id: str
    subject: str
    from_address: str
    from_name: str | None
    received_at: int
    snippet: str
    score: float                       # higher = better
    matched_in: MatchedIn
    matched_attachment_name: str | None = None
    web_link: str | None = None
    # LLM-generated topical summary when available, surfaced so the UI can
    # render it above the matched-text snippet. None when summarization is
    # disabled or generation failed for that email.
    summary: str | None = None


class SearchFilters(BaseModel):
    """Optional hard filters applied to every search mode.

    Empty filter = all fields ``None``. Activating a filter narrows the
    candidate set BEFORE ranking, so e.g. a sender filter doesn't just
    rerank — it excludes everyone else outright.

    Semantics:
      - ``start_at`` / ``end_at`` (epoch seconds, UTC): half-open interval
        ``[start_at, end_at)``. Either side may be omitted for an open range.
      - ``from_address``: case-insensitive exact match.
      - ``folder_id``: exact match (folder ids are stable opaque strings).
    """

    start_at: int | None = None
    end_at: int | None = None
    from_address: str | None = None
    folder_id: str | None = None

    def is_active(self) -> bool:
        return any(
            v is not None
            for v in (self.start_at, self.end_at, self.from_address, self.folder_id)
        )

    def matches(self, email: EmailRow) -> bool:
        """Check an EmailRow against active filters (used for the semantic path
        where we filter Python-side after the vec0 KNN)."""
        if self.start_at is not None and email.received_at < self.start_at:
            return False
        if self.end_at is not None and email.received_at >= self.end_at:
            return False
        if (
            self.from_address is not None
            and (email.from_address or "").lower() != self.from_address.lower()
        ):
            return False
        if self.folder_id is not None and email.folder_id != self.folder_id:
            return False
        return True


class LegResult(BaseModel):
    """One search leg's output. Streamed to the browser as a single chunk
    when the leg finishes, or aggregated into a :class:`SearchResponse` by
    the sync wrapper.

    ``trace`` mirrors the per-leg portion of the debug payload — populated
    only when ``debug_enabled`` is on so the lean path doesn't allocate.
    """

    source: LegSource
    hits: list[SearchHit]
    trace: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    """Aggregated response used by the sync wrapper + the tests.

    The streaming endpoint emits :class:`LegResult` chunks instead — it
    never builds one of these.
    """

    hits: list[SearchHit]
    mode: SearchMode
    query: str
    # Per-search transformation + ranking trace. Populated whenever
    # ``settings.debug_enabled`` is True (the default); ``None`` when
    # explicitly disabled server-side via ``EMAILSEARCH_DEBUG_ENABLED=false``.
    debug: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Mode → legs dispatch
# ---------------------------------------------------------------------------


def legs_for_mode(mode: SearchMode) -> list[LegSource]:
    """The set of legs that should run for a given search mode.

    - ``keyword``: the raw-query FTS leg only. No LLM hops, no embeddings.
    - ``semantic``: distilled FTS + augmented KNN. Both LLM hops fire,
      then run independently against the indexes.
    - ``hybrid``: all three legs. The keyword leg adds a safety net for
      verbatim terms the LLM transforms may have stripped.
    """
    if mode == "keyword":
        return ["keyword"]
    if mode == "semantic":
        return ["semantic_fts", "semantic_knn"]
    return ["keyword", "semantic_fts", "semantic_knn"]


def run_leg(
    source: LegSource,
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: SearchFilters,
    debug: bool,
) -> LegResult:
    """Dispatch entry point so the streaming endpoint can fan out legs by
    name. The actual work lives in private per-leg helpers below.
    """
    query = query.strip()
    if not query:
        return LegResult(source=source, hits=[], trace=None)
    if source == "keyword":
        return _run_keyword_leg(conn, query, limit=limit, filters=filters, debug=debug)
    if source == "semantic_fts":
        return _run_semantic_fts_leg(conn, query, limit=limit, filters=filters, debug=debug)
    if source == "semantic_knn":
        return _run_semantic_knn_leg(conn, query, limit=limit, filters=filters, debug=debug)
    raise ValueError(f"unknown leg source: {source!r}")


# ---------------------------------------------------------------------------
# Sync wrapper — same merge the frontend does, used by tests / CLI callers
# ---------------------------------------------------------------------------


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    mode: SearchMode = "hybrid",
    limit: int = 20,
    filters: SearchFilters | None = None,
) -> SearchResponse:
    """Run every leg for ``mode`` sequentially and merge by max score.

    Sync convenience wrapper used by tests and any in-process caller that
    wants a single fused result. The HTTP layer uses :func:`run_leg`
    directly + streams.

    Merge rule (mirrors the frontend exactly): one entry per ``email_id``;
    if an email surfaces in multiple legs, keep the hit with the highest
    score. Snippets / ``matched_in`` ride along with that winning hit — we
    don't cross-merge metadata from losing legs.
    """
    query = query.strip()
    filters = filters or SearchFilters()
    settings = get_settings()
    debug = settings.debug_enabled

    trace: dict[str, Any] | None = (
        {"raw_query": query, "mode": mode, "filters": filters.model_dump(), "legs": {}}
        if debug
        else None
    )

    if not query:
        return SearchResponse(hits=[], mode=mode, query="", debug=trace)

    log.info(
        "search: query=%r mode=%s limit=%d filters_active=%s debug=%s",
        query, mode, limit, filters.is_active(), debug,
    )

    # Run each leg sequentially. Order doesn't matter for the merge; we
    # use the mode-order so the trace reads top-down like the user would
    # expect.
    merged: dict[str, SearchHit] = {}
    for src in legs_for_mode(mode):
        leg = run_leg(src, conn, query, limit=limit, filters=filters, debug=debug)
        if trace is not None and leg.trace is not None:
            trace["legs"][src] = leg.trace
        for hit in leg.hits:
            existing = merged.get(hit.email_id)
            if existing is None or hit.score > existing.score:
                merged[hit.email_id] = hit

    ranked = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:limit]
    log.info("search: returning %d hit(s) for query=%r", len(ranked), query)
    if trace is not None:
        # Mirror the full structured trace to the server log so it's
        # visible in the uvicorn console without depending on the browser.
        log.info("search trace: %s", json.dumps(trace, default=str, ensure_ascii=False))
    return SearchResponse(hits=ranked, mode=mode, query=query, debug=trace)


# ---------------------------------------------------------------------------
# Leg 1: keyword — raw query → FTS
# ---------------------------------------------------------------------------


def _run_keyword_leg(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: SearchFilters,
    debug: bool,
) -> LegResult:
    leg_trace: dict[str, Any] | None = {"input": query} if debug else None
    built = _to_fts_query(query, joiner="AND")
    if leg_trace is not None:
        leg_trace["fts_query"] = built.query
        leg_trace["fts_joiner"] = built.joiner
        leg_trace["fts_kept_tokens"] = built.kept_tokens
        leg_trace["fts_dropped_short_tokens"] = built.dropped_short_tokens
    if built.dropped_short_tokens:
        log.info(
            "keyword: dropped %d sub-trigram token(s): %r",
            len(built.dropped_short_tokens), built.dropped_short_tokens,
        )
    if not built.query:
        log.info("keyword: empty fts query after sanitization (raw=%r)", query)
        return LegResult(source="keyword", hits=[], trace=leg_trace)
    # Over-fetch so the substring-false-positive filter has headroom; a
    # common fragment like "labor" can otherwise drown legitimate matches
    # out of the top-N.
    fts_limit = max(limit * _FTS_OVERFETCH_FACTOR, 20)
    raw_fts_hits = search_fts(
        conn,
        built.query,
        limit=fts_limit,
        start_at=filters.start_at,
        end_at=filters.end_at,
        from_address=filters.from_address,
        folder_id=filters.folder_id,
    )
    log.info(
        "keyword: fts raw returned %d hit(s) (over-fetch=%d) for %r",
        len(raw_fts_hits), fts_limit, built.query,
    )
    if not raw_fts_hits:
        if leg_trace is not None:
            leg_trace["fts_raw_count"] = 0
            leg_trace["fts_substring_filtered_count"] = 0
            leg_trace["fts_hits"] = []
        return LegResult(source="keyword", hits=[], trace=leg_trace)

    # Reject trigram substring false positives ("labor" inside
    # "collaboration") before scoring. We need the full EmailRow because
    # the FtsHit only carries subject + sender — the body / attachment
    # text where the false positive lives isn't in the FTS hit projection.
    emails = get_emails_by_ids(conn, [h.email_id for h in raw_fts_hits])
    verified: list = []
    for h in raw_fts_hits:
        e = emails.get(h.email_id)
        if e is None:
            continue
        if _verify_fts_hit(e, built.kept_tokens, built.joiner):
            verified.append(h)
    substring_filtered = len(raw_fts_hits) - len(verified)
    if substring_filtered:
        log.info(
            "keyword: substring filter dropped %d/%d hit(s)",
            substring_filtered, len(raw_fts_hits),
        )
    verified = verified[:limit]
    log.info(
        "keyword: returning %d hit(s) after filter + trim (limit=%d)",
        len(verified), limit,
    )
    if leg_trace is not None:
        leg_trace["fts_raw_count"] = len(raw_fts_hits)
        leg_trace["fts_substring_filtered_count"] = substring_filtered
        leg_trace["fts_hits"] = [
            {
                "email_id": h.email_id,
                "subject": h.subject,
                "from_address": h.from_address,
                "rank": h.rank,
            }
            for h in verified
        ]
    if not verified:
        return LegResult(source="keyword", hits=[], trace=leg_trace)

    out: list[SearchHit] = []
    for h in verified:
        e = emails.get(h.email_id)
        if e is None:
            continue
        # Higher score = better. bm25 is "lower better"; invert.
        score = 1.0 / (1.0 + max(0.0, h.rank))
        matched_in, matched_att = _classify_keyword_match(e, query)
        out.append(
            SearchHit(
                email_id=h.email_id,
                subject=h.subject,
                from_address=h.from_address,
                from_name=h.from_name,
                received_at=h.received_at,
                snippet=h.snippet,
                score=score,
                matched_in=matched_in,
                matched_attachment_name=matched_att,
                web_link=e.web_link,
                summary=e.summary,
            )
        )
    return LegResult(source="keyword", hits=out, trace=leg_trace)


# ---------------------------------------------------------------------------
# Leg 2: semantic_fts — LLM-distilled query → FTS
# ---------------------------------------------------------------------------


def _run_semantic_fts_leg(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: SearchFilters,
    debug: bool,
) -> LegResult:
    """FTS leg of semantic mode. The user's natural-language query is run
    through :func:`distill_query` first to strip filler down to the bare
    keyword phrase, and that phrase drives the bm25 search.

    Joiner is **OR** (not AND like the keyword leg). The distilled output
    is a bag of alternatives by design — bilingual translations of the
    same concept (``工会 labor union``) plus related vocabulary — and any
    real email matches at most one of those phrasings. AND-ing them
    deterministically returns 0 hits; OR-ing lets each alternative
    contribute independently.

    Falls back to the raw query when ``distill_query`` returns ``None``
    (LLM disabled / unreachable / failed).
    """
    distilled = distill_query(query)
    fts_input = distilled if distilled else query
    log.info(
        "semantic_fts: raw=%r distilled=%r (fallback=%s)",
        query, distilled, distilled is None,
    )
    leg_trace: dict[str, Any] | None = (
        {
            "raw_input": query,
            "distilled_query": distilled,
            "fts_input": fts_input,
        }
        if debug
        else None
    )

    built = _to_fts_query(fts_input, joiner="OR")
    if leg_trace is not None:
        leg_trace["fts_query"] = built.query
        leg_trace["fts_joiner"] = built.joiner
        leg_trace["fts_kept_tokens"] = built.kept_tokens
        leg_trace["fts_dropped_short_tokens"] = built.dropped_short_tokens
    if built.dropped_short_tokens:
        log.info(
            "semantic_fts: dropped %d sub-trigram token(s): %r",
            len(built.dropped_short_tokens), built.dropped_short_tokens,
        )
    if not built.query:
        return LegResult(source="semantic_fts", hits=[], trace=leg_trace)

    fts_limit = max(limit * _FTS_OVERFETCH_FACTOR, 20)
    raw_fts_hits = search_fts(
        conn,
        built.query,
        limit=fts_limit,
        start_at=filters.start_at,
        end_at=filters.end_at,
        from_address=filters.from_address,
        folder_id=filters.folder_id,
    )
    log.info(
        "semantic_fts: fts raw (distilled=%r -> %r, over-fetch=%d) returned %d hit(s)",
        fts_input, built.query, fts_limit, len(raw_fts_hits),
    )
    if not raw_fts_hits:
        if leg_trace is not None:
            leg_trace["fts_raw_count"] = 0
            leg_trace["fts_substring_filtered_count"] = 0
            leg_trace["fts_hits"] = []
        return LegResult(source="semantic_fts", hits=[], trace=leg_trace)

    # Reject trigram substring false positives. With OR joiner, a single
    # token like "labor" inside an LLM-distilled bag would otherwise
    # surface every "collaboration" mention in the corpus.
    emails = get_emails_by_ids(conn, [h.email_id for h in raw_fts_hits])
    verified: list = []
    for h in raw_fts_hits:
        e = emails.get(h.email_id)
        if e is None:
            continue
        if _verify_fts_hit(e, built.kept_tokens, built.joiner):
            verified.append(h)
    substring_filtered = len(raw_fts_hits) - len(verified)
    if substring_filtered:
        log.info(
            "semantic_fts: substring filter dropped %d/%d hit(s)",
            substring_filtered, len(raw_fts_hits),
        )
    verified = verified[:limit]
    log.info(
        "semantic_fts: returning %d hit(s) after filter + trim (limit=%d)",
        len(verified), limit,
    )
    if leg_trace is not None:
        leg_trace["fts_raw_count"] = len(raw_fts_hits)
        leg_trace["fts_substring_filtered_count"] = substring_filtered
        leg_trace["fts_hits"] = [
            {
                "email_id": h.email_id,
                "subject": h.subject,
                "from_address": h.from_address,
                "rank": h.rank,
            }
            for h in verified
        ]
    if not verified:
        return LegResult(source="semantic_fts", hits=[], trace=leg_trace)

    out: list[SearchHit] = []
    for h in verified:
        e = emails.get(h.email_id)
        if e is None:
            continue
        score = 1.0 / (1.0 + max(0.0, h.rank))
        matched_in, matched_att = _classify_keyword_match(e, fts_input)
        out.append(
            SearchHit(
                email_id=h.email_id,
                subject=h.subject,
                from_address=h.from_address,
                from_name=h.from_name,
                received_at=h.received_at,
                snippet=h.snippet,
                score=score,
                matched_in=matched_in,
                matched_attachment_name=matched_att,
                web_link=e.web_link,
                summary=e.summary,
            )
        )
    return LegResult(source="semantic_fts", hits=out, trace=leg_trace)


# ---------------------------------------------------------------------------
# Leg 3: semantic_knn — LLM-augmented query → embedding → vec0 KNN
# ---------------------------------------------------------------------------


def _run_semantic_knn_leg(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: SearchFilters,
    debug: bool,
) -> LegResult:
    """Embedding leg of semantic mode.

    Pipeline: raw → ``augment_query`` (synonyms / related terms) →
    ``embed_query`` → vec0 KNN → per-email grouping → score-threshold
    filter → summary promotion.

    ``settings.semantic_score_threshold`` gates the per-chunk score so a
    small / quiet corpus doesn't surface incidentally-similar emails just
    because vec0 always returns the top-K nearest neighbours. The FTS
    legs are unaffected — a verbatim term match remains a strong signal
    even when the embedding similarity isn't.
    """
    settings = get_settings()
    threshold = settings.semantic_score_threshold

    augmented = augment_query(query)
    embed_input = augmented if augmented else query
    log.info(
        "semantic_knn: raw=%r augmented=%r (fallback=%s) threshold=%.3f",
        query, augmented, augmented is None, threshold,
    )
    leg_trace: dict[str, Any] | None = (
        {
            "raw_input": query,
            "augmented_query": augmented,
            "embedded_query": embed_input,
            "score_threshold": threshold,
            "summary_promotion_base": SUMMARY_PROMOTION_BASE,
        }
        if debug
        else None
    )

    # vec0 has no WHERE clause for aux columns, so tight filters get
    # extra headroom — most of the top-K may be excluded post-hoc by
    # ``filters.matches``.
    over = max(limit * 4, 20)
    knn_over = over * 4 if filters.is_active() else over

    qvec = embed_query(embed_input)
    chunks = search_vec(conn, qvec, limit=knn_over)
    log.info(
        "semantic_knn: vec0 returned %d chunk(s) (over=%d)",
        len(chunks), knn_over,
    )

    # Drop chunks whose per-source similarity is below the threshold.
    # Score is computed with the same formula used later for the
    # per-email score so a chunk that passes here also has a meaningful
    # final score.
    pre_filter_count = len(chunks)
    chunks = [
        c for c in chunks
        if (1.0 / (1.0 + max(0.0, c.distance))) >= threshold
    ]
    dropped_by_threshold = pre_filter_count - len(chunks)
    log.info(
        "semantic_knn: threshold dropped %d/%d chunk(s) (threshold=%.3f)",
        dropped_by_threshold, pre_filter_count, threshold,
    )
    if leg_trace is not None:
        leg_trace["dropped_by_threshold"] = dropped_by_threshold
        leg_trace["vec_top_chunks"] = [
            {
                "email_id": c.email_id,
                "source_type": c.source_type,
                "source_name": c.source_name,
                "distance": c.distance,
                "score": round(1.0 / (1.0 + max(0.0, c.distance)), 6),
                "chunk_text_preview": _preview(c.chunk_text),
            }
            for c in chunks[:limit]
        ]

    if not chunks:
        if leg_trace is not None:
            leg_trace["final_scores"] = []
        return LegResult(source="semantic_knn", hits=[], trace=leg_trace)

    # Group chunks by email — track the best body / attachment chunk
    # separately from the best summary chunk. A summary hit promotes the
    # email's score into the [1, 2] bucket; body-only stays in [0, 1].
    body_best: dict[str, tuple[float, str, str | None, str]] = {}
    summary_best: dict[str, tuple[float, str]] = {}
    for c in chunks:
        if c.source_type == "summary":
            prev_s = summary_best.get(c.email_id)
            if prev_s is None or c.distance < prev_s[0]:
                summary_best[c.email_id] = (c.distance, c.chunk_text)
        else:
            prev_b = body_best.get(c.email_id)
            if prev_b is None or c.distance < prev_b[0]:
                body_best[c.email_id] = (c.distance, c.source_type, c.source_name, c.chunk_text)

    candidate_ids = set(body_best) | set(summary_best)
    emails = get_emails_by_ids(conn, list(candidate_ids))

    # Hard filters — vec0 can't filter aux columns, so we do it here.
    filters_dropped = 0
    if filters.is_active():
        before = len(emails)
        emails = {eid: e for eid, e in emails.items() if filters.matches(e)}
        filters_dropped = before - len(emails)
        body_best = {eid: v for eid, v in body_best.items() if eid in emails}
        summary_best = {eid: v for eid, v in summary_best.items() if eid in emails}
        candidate_ids = set(body_best) | set(summary_best)
        log.info(
            "semantic_knn: filters dropped %d/%d candidate(s); %d remain",
            filters_dropped, before, len(emails),
        )
    if leg_trace is not None:
        leg_trace["filters_dropped"] = filters_dropped

    # Score + build hits.
    scored: list[tuple[float, SearchHit, dict[str, Any]]] = []
    for eid in candidate_ids:
        e = emails.get(eid)
        if e is None:
            continue
        body_entry = body_best.get(eid)
        summary_entry = summary_best.get(eid)
        promoted = summary_entry is not None

        if body_entry is not None:
            _, src_type, src_name, chunk_text = body_entry
        else:
            assert summary_entry is not None
            src_type, src_name, chunk_text = "summary", None, summary_entry[1]

        if promoted:
            summary_dist = summary_entry[0]  # type: ignore[index]
            summary_score = 1.0 / (1.0 + max(0.0, summary_dist))
            score = SUMMARY_PROMOTION_BASE + summary_score
        else:
            body_dist = body_entry[0]  # type: ignore[index]
            score = 1.0 / (1.0 + max(0.0, body_dist))

        matched_in: MatchedIn = "attachment" if src_type == "attachment" else "body"
        snippet = _make_snippet(chunk_text, query)
        attachment_name = src_name if src_type == "attachment" else None

        hit = SearchHit(
            email_id=eid,
            subject=e.subject,
            from_address=e.from_address,
            from_name=e.from_name,
            received_at=e.received_at,
            snippet=snippet,
            score=score,
            matched_in=matched_in,
            matched_attachment_name=attachment_name,
            web_link=e.web_link,
            summary=e.summary,
        )
        scored_row = {
            "email_id": eid,
            "subject": e.subject,
            "from_address": e.from_address,
            "body_score": (
                round(1.0 / (1.0 + max(0.0, body_best[eid][0])), 6)
                if eid in body_best else None
            ),
            "summary_score": (
                round(1.0 / (1.0 + max(0.0, summary_best[eid][0])), 6)
                if eid in summary_best else None
            ),
            "summary_promoted": promoted,
            "snippet_source": src_type,
            "final_score": round(score, 6),
        }
        scored.append((score, hit, scored_row))

    scored.sort(key=lambda t: t[0], reverse=True)
    ranked = scored[:limit]
    out = [hit for _, hit, _ in ranked]

    log.info("semantic_knn: %d candidate(s) -> %d hit(s)", len(scored), len(out))
    if leg_trace is not None:
        leg_trace["final_scores"] = [row for _, _, row in ranked]

    return LegResult(source="semantic_knn", hits=out, trace=leg_trace)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _preview(s: str | None) -> str:
    """Truncate `s` for the debug payload. ``None`` becomes ``""``."""
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    if len(s) <= _DEBUG_PREVIEW_CHARS:
        return s
    return s[: _DEBUG_PREVIEW_CHARS] + "…"


class _FtsQueryBuild(NamedTuple):
    """Result of :func:`_to_fts_query`. Bundles the SQL-ready MATCH
    expression with the per-token bookkeeping each search leg surfaces
    in its trace.

    Attributes:
      query: The final MATCH expression. ``""`` signals "skip the search"
        (no usable tokens survived).
      kept_tokens: Tokens that made it into ``query`` (in input order).
      dropped_short_tokens: Tokens dropped because they were shorter than
        :data:`_TRIGRAM_MIN_CHARS` and therefore unmatchable by the
        trigram tokenizer. Surfaced so the user can see why their 2-char
        keyword (``工会``, ``Q3``) had no effect.
      joiner: ``"AND"`` or ``"OR"`` — mirrors the operator the tokens were
        joined with. Informational; lets the trace explain the per-leg
        semantics without callers re-deriving them.
    """

    query: str
    kept_tokens: list[str]
    dropped_short_tokens: list[str]
    joiner: FtsJoiner


def _to_fts_query(
    query: str,
    *,
    joiner: FtsJoiner = "AND",
) -> _FtsQueryBuild:
    """Convert a user query string into a safe FTS5 MATCH expression.

    FTS5 has a punctuation-heavy mini-grammar — ``"`` and ``'`` both
    delimit phrases, ``*`` is a prefix operator, ``( ) : - + ^`` are
    syntax, and bare ``AND`` / ``OR`` / ``NOT`` are operators. Passing a
    raw user query straight to MATCH crashes with ``syntax error`` on
    anything from a contraction (``don't``) to an email address fragment
    (``kai'xin``) to a Windows path (``C:\\foo``).

    Strategy: tokenize on whitespace, drop tokens shorter than
    :data:`_TRIGRAM_MIN_CHARS` (the trigram tokenizer can't match them),
    escape any embedded ``"`` per FTS5 rules (doubled — ``"`` → ``""``),
    and wrap each surviving token as a quoted phrase. Returns an empty
    ``query`` (signalling "skip the search") when no token survives.

    ``joiner`` semantics:
      - ``"AND"`` (default): every token must match — right for the
        verbatim keyword leg.
      - ``"OR"``: any token matching is enough — right for the
        semantic_fts leg, whose input is the distilled bilingual bag of
        alternatives from :func:`distill_query`. AND-ing those would be
        contradictory and deterministically return 0 hits.
    """
    raw_tokens = query.split()
    kept: list[str] = []
    dropped: list[str] = []
    for t in raw_tokens:
        # Length check is on the raw whitespace-split token. A 2-char
        # token is unmatchable by trigram regardless of whether it
        # contains punctuation, so there's no need to strip quotes first.
        if len(t) < _TRIGRAM_MIN_CHARS:
            dropped.append(t)
        else:
            kept.append(t)

    if not kept:
        return _FtsQueryBuild(
            query="",
            kept_tokens=[],
            dropped_short_tokens=dropped,
            joiner=joiner,
        )

    # Per FTS5 docs: 'To include a double-quote character within a string,
    # escape it by doubling it (i.e. use "").' Apostrophe is also a phrase
    # delimiter but does NOT have a doubling-escape — wrapping in `"..."`
    # makes it literal, which is what we want.
    escaped = [t.replace('"', '""') for t in kept]
    phrases = [f'"{t}"' for t in escaped]
    # Implicit AND is just whitespace between phrases; OR must be spelled
    # out (uppercase — lowercase ``or`` would be parsed as a phrase token).
    sep = " " if joiner == "AND" else " OR "
    return _FtsQueryBuild(
        query=sep.join(phrases),
        kept_tokens=kept,
        dropped_short_tokens=dropped,
        joiner=joiner,
    )


def _is_cjk_token(token: str) -> bool:
    """True iff ``token`` contains any CJK character (see :data:`_CJK_RANGES`).

    Used by :func:`_verify_fts_hit` to decide which matching semantics to
    apply per token: CJK gets substring matching (no inter-character word
    boundaries in the script), ASCII gets ``\\b`` word-boundary matching
    (so ``labor`` doesn't match ``collaboration``).
    """
    for ch in token:
        for lo, hi in _CJK_RANGES:
            if lo <= ch <= hi:
                return True
    return False


def _token_appears_with_boundary(token_lower: str, text_lower: str) -> bool:
    """Does ``token_lower`` appear in ``text_lower`` with the right matching
    semantics for its script?

    Both arguments must already be lowercased.

    - **ASCII token**: requires a ``\\b`` word boundary BEFORE the match.
      Rejects trigram's substring false positives (``labor`` inside
      ``collaboration``) while still accepting legitimate suffixes
      (``labor`` matches ``labors`` / ``laborer`` / ``laboring``).
    - **CJK token**: plain substring match. Python's ``\\b`` is defined on
      ``\\w`` which excludes CJK characters, so a regex boundary anchor
      would never match. Substring is also what the trigram index
      produces for CJK and what the user expects.
    """
    if _is_cjk_token(token_lower):
        return token_lower in text_lower
    return re.search(r"\b" + re.escape(token_lower), text_lower) is not None


def _verify_fts_hit(
    email: EmailRow,
    kept_tokens: list[str],
    joiner: FtsJoiner,
) -> bool:
    """Post-filter for FTS hits to compensate for the trigram tokenizer's
    pure-substring matching against ASCII text.

    Trigram indexes overlapping 3-character shingles. That's what makes
    CJK searchable (no whitespace tokens in CJK) but for ASCII it means
    a query token ``labor`` matches every email containing
    ``collaboration``, ``belabor``, ``laboratory``, etc.

    This filter checks each ASCII token against the email's combined
    searchable text with a regex ``\\b`` anchor; CJK tokens fall back to
    substring matching. The ``joiner`` mirrors the FTS query's semantics:
    ``"AND"`` requires every token to match legitimately (keyword leg);
    ``"OR"`` requires at least one (semantic_fts leg).

    Returns ``True`` (keep the hit) when ``kept_tokens`` is empty — the
    leg already short-circuits before calling us in that case, but the
    explicit guard makes this function safe to call standalone.
    """
    if not kept_tokens:
        return True
    # Haystack from every FTS-indexed column. ``searchable_text`` already
    # covers body + summary + attachment text; ``subject`` and
    # ``from_address`` are stored separately and indexed by their own FTS
    # columns.
    text_lower = " ".join(
        s.lower()
        for s in (email.subject, email.from_address, email.searchable_text)
        if s
    )
    check = all if joiner == "AND" else any
    return check(
        _token_appears_with_boundary(t.lower(), text_lower) for t in kept_tokens
    )


def _classify_keyword_match(email: EmailRow, query: str) -> tuple[MatchedIn, str | None]:
    """Single pass over attachments: returns (matched_in, attachment_name)."""
    q = query.lower()
    in_body = q in (email.body_text or "").lower() or q in (email.subject or "").lower()
    matched_att: str | None = None
    for a in email.attachments:
        if q in (a.extracted_text or "").lower():
            matched_att = a.name
            break
    if in_body and matched_att is not None:
        return "both", matched_att
    if matched_att is not None:
        return "attachment", matched_att
    return "body", None


def _make_snippet(text: str, query: str, *, window: int = 80) -> str:
    """Cheap snippet around the query term; falls back to the start of text."""
    if not text:
        return ""
    lower = text.lower()
    qlower = query.lower()
    i = lower.find(qlower)
    if i < 0:
        return (text[: window * 2] + ("…" if len(text) > window * 2 else "")).strip()
    start = max(0, i - window)
    end = min(len(text), i + len(query) + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}".strip()
