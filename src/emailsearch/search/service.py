"""Search service: per-leg search helpers + a sync merge wrapper.

Three independent **legs** — ``keyword``, ``semantic_fts``, and
``semantic_knn`` — each produce a self-scored list of :class:`SearchHit`.
There is no cross-leg fusion: the caller (the streaming HTTP endpoint or
the sync ``search`` wrapper used by tests) merges per-email scores when
the same email surfaces in multiple legs.

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
import time
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
from emailsearch.util import contains_cjk

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
# distance-driven.
SUMMARY_PROMOTION_BASE = 1.0

# Minimum token length the FTS5 trigram tokenizer can match. Strings
# shorter than 3 chars (e.g. "工会", "Q3") return 0 hits and poison any
# AND-joined query they appear in, so we strip them up-front.
_TRIGRAM_MIN_CHARS = 3

# Over-fetch factor for the substring-false-positive post-filter
# (:func:`_verify_fts_hit`), which can drop a large fraction of hits when
# an ASCII fragment substring-matches a longer word ("labor" inside
# "collaboration"). Raw bm25 is sub-ms so over-fetch cost is negligible.
_FTS_OVERFETCH_FACTOR = 3

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
    # LLM-generated topical summary; None when disabled or generation failed.
    summary: str | None = None


class SearchFilters(BaseModel):
    """Optional hard filters applied before ranking (active = at least one
    field set). Date range is half-open ``[start_at, end_at)``;
    ``from_address`` matches case-insensitively; ``folder_id`` matches
    exactly. Empty filter = all ``None``.
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
        """Check an EmailRow against active filters (used for the semantic
        path where we filter Python-side after the vec0 KNN)."""
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
    """One search leg's output. ``trace`` is populated only when
    ``debug_enabled`` is on so the lean path doesn't allocate."""

    source: LegSource
    hits: list[SearchHit]
    trace: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    """Aggregated response used by the sync wrapper + the tests. The
    streaming endpoint emits :class:`LegResult` chunks instead."""

    hits: list[SearchHit]
    mode: SearchMode
    query: str
    # Per-search trace. Populated whenever ``settings.debug_enabled`` is
    # True (default); ``None`` when explicitly disabled.
    debug: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Mode → legs dispatch
# ---------------------------------------------------------------------------


def legs_for_mode(mode: SearchMode) -> list[LegSource]:
    """Legs that should run for a given mode.

    - ``keyword``: raw-query FTS only (no LLM hops, no embeddings).
    - ``semantic``: distilled FTS + augmented KNN.
    - ``hybrid``: all three (keyword adds a safety net for verbatim
      terms the LLM transforms may have stripped).
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
    if source in ("keyword", "semantic_fts"):
        return _run_fts_leg(source, conn, query, limit=limit, filters=filters, debug=debug)
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

    Sync convenience wrapper used by tests / in-process callers; the HTTP
    layer uses :func:`run_leg` directly + streams. Merge rule: one entry
    per ``email_id``; keep the hit with the highest score.
    """
    search_started = time.perf_counter()
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
        if trace is not None:
            trace["overall_score"] = 0.0
            trace["timings_ms"] = {"total_ms": _ms_since(search_started)}
        return SearchResponse(hits=[], mode=mode, query="", debug=trace)

    log.info(
        "search: query=%r mode=%s limit=%d filters_active=%s debug=%s",
        query, mode, limit, filters.is_active(), debug,
    )

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
        trace["overall_score"] = max((h.score for h in ranked), default=0.0)
        trace["timings_ms"] = {"total_ms": _ms_since(search_started)}
        log.info("search trace: %s", json.dumps(trace, default=str, ensure_ascii=False))
    return SearchResponse(hits=ranked, mode=mode, query=query, debug=trace)


# ---------------------------------------------------------------------------
# FTS legs (keyword + semantic_fts) — share everything except input prep
# ---------------------------------------------------------------------------


def _run_fts_leg(
    source: Literal["keyword", "semantic_fts"],
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filters: SearchFilters,
    debug: bool,
) -> LegResult:
    """Run a bm25-based leg.

    ``keyword`` uses the raw query AND-joined (every typed term must match).
    ``semantic_fts`` runs the query through :func:`distill_query` first to
    strip filler and emit a bilingual bag of alternatives, then OR-joins
    them (any one is enough — AND-ing the bag is contradictory and
    returns 0 hits by construction). Falls back to the raw query when
    distillation returns ``None``.
    """
    leg_started = time.perf_counter()
    timings_ms: dict[str, float] = {}
    leg_trace: dict[str, Any] | None

    if source == "semantic_fts":
        llm_started = time.perf_counter()
        distilled = distill_query(query)
        timings_ms["llm_preprocess_ms"] = _ms_since(llm_started)
        fts_input = distilled if distilled else query
        joiner: FtsJoiner = "OR"
        log.info(
            "semantic_fts: raw=%r distilled=%r (fallback=%s)",
            query, distilled, distilled is None,
        )
        leg_trace = (
            {"raw_input": query, "distilled_query": distilled, "fts_input": fts_input}
            if debug else None
        )
    else:
        fts_input = query
        joiner = "AND"
        leg_trace = {"input": query} if debug else None

    built = _to_fts_query(fts_input, joiner=joiner)
    if leg_trace is not None:
        leg_trace["fts_query"] = built.query
        leg_trace["fts_joiner"] = built.joiner
        leg_trace["fts_kept_tokens"] = built.kept_tokens
        leg_trace["fts_dropped_short_tokens"] = built.dropped_short_tokens
    if built.dropped_short_tokens:
        log.info(
            "%s: dropped %d sub-trigram token(s): %r",
            source, len(built.dropped_short_tokens), built.dropped_short_tokens,
        )
    if not built.query:
        if source == "keyword":
            log.info("keyword: empty fts query after sanitization (raw=%r)", query)
        _finalize_leg_trace(leg_trace, leg_started, timings_ms, [])
        return LegResult(source=source, hits=[], trace=leg_trace)

    fts_limit = max(limit * _FTS_OVERFETCH_FACTOR, 20)
    db_started = time.perf_counter()
    raw_fts_hits = search_fts(
        conn,
        built.query,
        limit=fts_limit,
        start_at=filters.start_at,
        end_at=filters.end_at,
        from_address=filters.from_address,
        folder_id=filters.folder_id,
    )
    timings_ms["db_search_ms"] = _ms_since(db_started)
    log.info(
        "%s: fts raw (query=%r, over-fetch=%d) returned %d hit(s)",
        source, built.query, fts_limit, len(raw_fts_hits),
    )
    if not raw_fts_hits:
        if leg_trace is not None:
            leg_trace["fts_raw_count"] = 0
            leg_trace["fts_substring_filtered_count"] = 0
            leg_trace["fts_hits"] = []
        _finalize_leg_trace(leg_trace, leg_started, timings_ms, [])
        return LegResult(source=source, hits=[], trace=leg_trace)

    # Reject trigram substring false positives ("labor" inside
    # "collaboration"). Needs the full EmailRow because FtsHit only
    # carries subject + sender, not body / attachment text.
    emails = get_emails_by_ids(conn, [h.email_id for h in raw_fts_hits])
    verified = [
        h for h in raw_fts_hits
        if (e := emails.get(h.email_id)) is not None
        and _verify_fts_hit(e, built.kept_tokens, built.joiner)
    ]
    substring_filtered = len(raw_fts_hits) - len(verified)
    if substring_filtered:
        log.info(
            "%s: substring filter dropped %d/%d hit(s)",
            source, substring_filtered, len(raw_fts_hits),
        )
    verified = verified[:limit]
    log.info(
        "%s: returning %d hit(s) after filter + trim (limit=%d)",
        source, len(verified), limit,
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
        _finalize_leg_trace(leg_trace, leg_started, timings_ms, [])
        return LegResult(source=source, hits=[], trace=leg_trace)

    out: list[SearchHit] = []
    for h in verified:
        e = emails.get(h.email_id)
        if e is None:
            continue
        matched_in, matched_att = _classify_keyword_match(e, fts_input)
        out.append(
            SearchHit(
                email_id=h.email_id,
                subject=h.subject,
                from_address=h.from_address,
                from_name=h.from_name,
                received_at=h.received_at,
                snippet=h.snippet,
                score=_bm25_to_score(h.rank),
                matched_in=matched_in,
                matched_attachment_name=matched_att,
                web_link=e.web_link,
                summary=e.summary,
            )
        )
    _finalize_leg_trace(leg_trace, leg_started, timings_ms, out)
    return LegResult(source=source, hits=out, trace=leg_trace)


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
    leg_started = time.perf_counter()
    timings_ms: dict[str, float] = {}
    settings = get_settings()
    threshold = settings.semantic_score_threshold

    llm_started = time.perf_counter()
    augmented = augment_query(query)
    timings_ms["llm_preprocess_ms"] = _ms_since(llm_started)
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

    embed_started = time.perf_counter()
    qvec = embed_query(embed_input)
    timings_ms["embedding_ms"] = _ms_since(embed_started)
    db_started = time.perf_counter()
    chunks = search_vec(conn, qvec, limit=knn_over)
    timings_ms["db_search_ms"] = _ms_since(db_started)
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
        _finalize_leg_trace(leg_trace, leg_started, timings_ms, [])
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

    _finalize_leg_trace(leg_trace, leg_started, timings_ms, out)
    return LegResult(source="semantic_knn", hits=out, trace=leg_trace)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ms_since(start: float) -> float:
    """Milliseconds elapsed since ``start`` (from ``time.perf_counter``),
    rounded to 3 decimals.

    Integer milliseconds truncate sub-ms steps to ``0`` which makes the
    trace useless for fast paths (vec0 KNN, cached embed, empty FTS).
    """
    return round((time.perf_counter() - start) * 1000, 3)


def _finalize_leg_trace(
    leg_trace: dict[str, Any] | None,
    leg_started: float,
    timings_ms: dict[str, float],
    hits: list[SearchHit],
) -> None:
    """Stamp ``total_ms`` + ``overall_score`` on the leg trace before return.

    No-op when ``leg_trace`` is ``None``. Mutates ``timings_ms`` in place
    to append ``total_ms``. ``overall_score`` is the max hit score (0.0
    when ``hits`` is empty).
    """
    if leg_trace is None:
        return
    timings_ms["total_ms"] = _ms_since(leg_started)
    leg_trace["timings_ms"] = timings_ms
    leg_trace["overall_score"] = max((h.score for h in hits), default=0.0)


def _bm25_to_score(rank: float) -> float:
    """Convert an FTS5 ``bm25()`` rank into a (0, 1) score where higher = better.

    SQLite's bm25 returns NEGATIVE numbers (more negative = stronger).
    We negate to get a "goodness" magnitude then squash through
    ``g / (1 + g)`` so scores asymptote to 1.0 for very strong matches
    and decay toward 0.0 for weak ones. The floor at 0 protects against
    any spurious non-negative result.
    """
    goodness = max(0.0, -rank)
    return goodness / (1.0 + goodness)


def _preview(s: str | None) -> str:
    """Truncate `s` for the debug payload. ``None`` becomes ``""``."""
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    if len(s) <= _DEBUG_PREVIEW_CHARS:
        return s
    return s[: _DEBUG_PREVIEW_CHARS] + "…"


class _FtsQueryBuild(NamedTuple):
    """Result of :func:`_to_fts_query`.

    Bundles the SQL-ready MATCH expression with the per-token bookkeeping
    each search leg surfaces in its trace. ``query == ""`` signals
    "skip the search" (no usable tokens survived). ``dropped_short_tokens``
    is surfaced so the user can see why their 2-char keyword (e.g.
    ``工会``, ``Q3``) had no effect.
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

    Raw user input crashes FTS5's mini-grammar on anything from a
    contraction (``don't``) to a Windows path (``C:\\foo``). We
    tokenize on whitespace, drop tokens shorter than
    :data:`_TRIGRAM_MIN_CHARS` (unmatchable by the trigram tokenizer),
    escape embedded ``"`` per FTS5 rules (``"`` → ``""``), and wrap each
    survivor as a quoted phrase.

    ``joiner="AND"`` (every token required) suits the verbatim keyword
    leg; ``"OR"`` (any token enough) suits the semantic_fts leg whose
    input is a bag of alternatives.
    """
    raw_tokens = query.split()
    kept: list[str] = []
    dropped: list[str] = []
    for t in raw_tokens:
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

    # FTS5: doubled ``"`` is the escape for an embedded quote inside a
    # ``"..."`` phrase. Implicit AND is whitespace; OR must be uppercase
    # (lowercase ``or`` would be parsed as a phrase token).
    escaped = [t.replace('"', '""') for t in kept]
    phrases = [f'"{t}"' for t in escaped]
    sep = " " if joiner == "AND" else " OR "
    return _FtsQueryBuild(
        query=sep.join(phrases),
        kept_tokens=kept,
        dropped_short_tokens=dropped,
        joiner=joiner,
    )


def _is_cjk_token(token: str) -> bool:
    """True iff ``token`` contains any CJK character.

    Used by :func:`_verify_fts_hit` to choose matching semantics: CJK
    gets substring matching (no inter-character word boundaries); ASCII
    gets ``\\b`` word-boundary matching.
    """
    return contains_cjk(token)


def _token_appears_with_boundary(token_lower: str, text_lower: str) -> bool:
    """Does ``token_lower`` appear in ``text_lower`` with the right matching
    semantics for its script?

    Both arguments must already be lowercased. ASCII tokens require a
    ``\\b`` word boundary before the match (rejects ``labor`` inside
    ``collaboration``). CJK tokens fall back to substring — Python's
    ``\\b`` is defined on ``\\w`` which excludes CJK characters.
    """
    if _is_cjk_token(token_lower):
        return token_lower in text_lower
    return re.search(r"\b" + re.escape(token_lower), text_lower) is not None


def _verify_fts_hit(
    email: EmailRow,
    kept_tokens: list[str],
    joiner: FtsJoiner,
) -> bool:
    """Post-filter for FTS hits to compensate for trigram's pure-substring
    matching against ASCII text.

    Trigram indexes overlapping 3-char shingles — required for CJK (no
    whitespace tokens) but for ASCII it makes ``labor`` match
    ``collaboration``. We re-check each ASCII token against the email's
    searchable text with a ``\\b`` anchor; CJK tokens use substring.
    Returns ``True`` when ``kept_tokens`` is empty (defensive guard).
    """
    if not kept_tokens:
        return True
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
