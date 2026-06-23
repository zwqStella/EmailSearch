"""Orchestration: read the eval set, run every query in every mode,
collect per-query results, aggregate per mode (and per category).

We deliberately run modes *sequentially* per query — not in parallel.
The :func:`emailsearch.search.service.search` function already runs its
own legs in-process and we want the latency numbers to reflect what an
interactive user actually experiences, not artificially inflated
contention.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import defaultdict
from collections.abc import Iterable

from emailsearch.eval.metrics import (
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from emailsearch.eval.schema import (
    EvalQuery,
    EvalReport,
    EvalSet,
    ModeSummary,
    QueryResult,
)
from emailsearch.search.service import SearchFilters, SearchMode, search

log = logging.getLogger(__name__)

# Default modes evaluated — covers every code path in
# :func:`legs_for_mode`. Override via the CLI if you want a subset.
DEFAULT_MODES: tuple[SearchMode, ...] = ("keyword", "semantic", "hybrid")

# Fixed retrieval cap. Big enough to fit every query's relevant set
# (typical curated sets are < 15 IDs) while small enough to keep the
# semantic_knn over-fetch tractable.
RUN_LIMIT = 20


def run_eval(
    conn: sqlite3.Connection,
    eval_set: EvalSet,
    *,
    modes: Iterable[SearchMode] = DEFAULT_MODES,
    limit: int = RUN_LIMIT,
) -> EvalReport:
    """Run every (query, mode) pair and build the full report.

    The caller owns the connection — pass an already-open, schema-applied
    one (typically ``open_connection()`` against the user's real DB).
    """
    modes = tuple(modes)
    per_query: list[QueryResult] = []

    # Skip queries the operator forgot to label. They'd silently produce
    # 0.0 for every metric and poison the aggregate.
    usable_queries = [q for q in eval_set.queries if q.relevant]
    skipped = [q.id for q in eval_set.queries if not q.relevant]
    if skipped:
        log.warning("eval: skipping %d unlabeled query/ies: %s", len(skipped), skipped)

    for query in usable_queries:
        for mode in modes:
            per_query.append(_run_single(conn, query, mode, limit=limit))

    summaries = [_summarize(per_query, mode) for mode in modes]
    per_category = _summarize_by_category(per_query, usable_queries, modes)

    return EvalReport(
        eval_set_meta=eval_set.meta,
        modes=list(modes),
        summaries=summaries,
        per_query=per_query,
        per_category=per_category,
    )


def _run_single(
    conn: sqlite3.Connection,
    query: EvalQuery,
    mode: SearchMode,
    *,
    limit: int,
) -> QueryResult:
    filters = SearchFilters(
        start_at=query.filters.start_at,
        end_at=query.filters.end_at,
        from_address=query.filters.from_address,
        folder_id=query.filters.folder_id,
    )
    started = time.perf_counter()
    resp = search(conn, query.text, mode=mode, limit=limit, filters=filters)
    latency_ms = (time.perf_counter() - started) * 1000.0

    ranked_ids = [h.email_id for h in resp.hits]
    relevant = set(query.relevant)

    return QueryResult(
        query_id=query.id,
        mode=mode,
        ranked_ids=ranked_ids,
        latency_ms=latency_ms,
        precision_at_5=precision_at_k(ranked_ids, relevant, 5),
        precision_at_10=precision_at_k(ranked_ids, relevant, 10),
        precision_at_20=precision_at_k(ranked_ids, relevant, 20),
        recall_at_5=recall_at_k(ranked_ids, relevant, 5),
        recall_at_10=recall_at_k(ranked_ids, relevant, 10),
        recall_at_20=recall_at_k(ranked_ids, relevant, 20),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant),
        ndcg_at_10=ndcg_at_k(ranked_ids, relevant, 10),
        n_relevant=len(relevant),
    )


def _summarize(results: list[QueryResult], mode: SearchMode) -> ModeSummary:
    rows = [r for r in results if r.mode == mode]
    latencies = [r.latency_ms for r in rows]
    return ModeSummary(
        mode=mode,
        n_queries=len(rows),
        mean_precision_at_5=mean(r.precision_at_5 for r in rows),
        mean_precision_at_10=mean(r.precision_at_10 for r in rows),
        mean_precision_at_20=mean(r.precision_at_20 for r in rows),
        mean_recall_at_5=mean(r.recall_at_5 for r in rows),
        mean_recall_at_10=mean(r.recall_at_10 for r in rows),
        mean_recall_at_20=mean(r.recall_at_20 for r in rows),
        mrr=mean(r.reciprocal_rank for r in rows),
        mean_ndcg_at_10=mean(r.ndcg_at_10 for r in rows),
        p50_latency_ms=percentile(latencies, 0.5),
        p95_latency_ms=percentile(latencies, 0.95),
    )


def _summarize_by_category(
    results: list[QueryResult],
    queries: list[EvalQuery],
    modes: tuple[SearchMode, ...],
) -> dict[str, dict[SearchMode, ModeSummary]]:
    """Group results by query category, then re-aggregate per mode.

    Reveals e.g. that keyword excels on *verbatim* queries and gets
    crushed on *semantic*. Returns empty dict if every query shares the
    same category (the breakdown would be redundant with the overall
    summary).
    """
    cat_by_qid = {q.id: q.category for q in queries}
    categories = {cat_by_qid[r.query_id] for r in results}
    if len(categories) <= 1:
        return {}

    by_category: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_category[cat_by_qid[r.query_id]].append(r)

    out: dict[str, dict[SearchMode, ModeSummary]] = {}
    for category, rows in by_category.items():
        out[category] = {mode: _summarize(rows, mode) for mode in modes}
    return out
