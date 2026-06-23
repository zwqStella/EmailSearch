"""Pydantic models for the eval query set and per-run report.

The query set is the *input* (a TOML file authored by the operator).
The report is the *output* (markdown rendered by :mod:`.report` plus an
optional JSON dump for downstream tooling).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from emailsearch.search.service import SearchMode

QueryCategory = Literal[
    "verbatim",      # exact tokens appear in subject/body
    "topic",         # paraphrased but stays within domain vocabulary
    "semantic",      # requires conceptual generalization
    "multilingual",  # CJK or mixed-script query
    "person",        # sender or recipient lookup
    "attachment",    # answer lives in an attachment
    "thread",        # one logical conversation; many message IDs
]


class FiltersSpec(BaseModel):
    """Optional hard filters passed through to ``SearchFilters``.

    Mirrors :class:`emailsearch.search.service.SearchFilters` so a query
    can fix a date / sender / folder before ranking.
    """

    start_at: int | None = None
    end_at: int | None = None
    from_address: str | None = None
    folder_id: str | None = None


class EvalQuery(BaseModel):
    """One labeled query.

    ``relevant`` is the *ground truth*: a (small, hand-curated) set of
    email IDs we have deemed relevant by inspecting the corpus, not by
    looking at what the search system returns. Keep this list short
    (typically 1-10 IDs) — Precision@K and nDCG@K only count what's in
    here, so spurious entries silently penalize the metric.
    """

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: QueryCategory
    description: str = ""
    relevant: list[str] = Field(default_factory=list)
    filters: FiltersSpec = Field(default_factory=FiltersSpec)


class EvalSet(BaseModel):
    """Top-level container deserialized from ``queries.toml``."""

    version: int = 1
    meta: dict[str, str] = Field(default_factory=dict)
    queries: list[EvalQuery]


class QueryResult(BaseModel):
    """Per-(query, mode) outcome — feeds both the per-query appendix and
    the aggregated mode summary.

    Metrics are computed at K ∈ {5, 10, 20}. K=10 is the conventional IR
    cutoff (the "first page" of typical search UIs), K=5 captures the
    user's typical foveal scan, and K=20 matches the retrieval ``limit``
    so every returned hit is accounted for in at least one column.
    """

    query_id: str
    mode: SearchMode
    ranked_ids: list[str]  # length == returned hits, in score order
    latency_ms: float
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    reciprocal_rank: float
    ndcg_at_10: float
    n_relevant: int  # |relevant| for the query (sanity column in the report)


class ModeSummary(BaseModel):
    """Aggregated metrics for one search mode across all queries."""

    mode: SearchMode
    n_queries: int
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_precision_at_20: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_recall_at_20: float
    mrr: float                 # mean reciprocal rank
    mean_ndcg_at_10: float
    p50_latency_ms: float
    p95_latency_ms: float


class EvalReport(BaseModel):
    """Full report payload — what the CLI writes out as JSON next to
    the markdown file (handy for diffing across commits)."""

    eval_set_meta: dict[str, str]
    modes: list[SearchMode]
    summaries: list[ModeSummary]
    per_query: list[QueryResult]
    # Per-category breakdown: category -> mode -> ModeSummary.
    # Lets the markdown report show "where each mode shines".
    per_category: dict[str, dict[SearchMode, ModeSummary]] = Field(default_factory=dict)
