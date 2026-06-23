"""Pure ranking-metric implementations.

All metrics treat relevance as binary (relevant / not-relevant) — graded
relevance would need a richer ``EvalQuery.relevant`` schema and isn't
worth the labeling cost for this corpus size.

Conventions:
  - ``ranked``: list of email IDs in score order (best first).
  - ``relevant``: set of email IDs hand-labeled as relevant.
  - ``k``: cutoff; metrics consume ``ranked[:k]`` only.

Edge cases (matching common IR conventions):
  - Empty ``relevant`` → recall is undefined; we return 0.0 and the
    caller is expected to skip the query from the aggregate.
  - Empty ``ranked`` → all metrics return 0.0.
  - ``k`` larger than ``len(ranked)`` is fine: the slice just truncates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-``k`` results that are in ``relevant``.

    Denominator is always ``k`` (not ``min(k, len(ranked))``) — this is
    the standard P@K convention so a short result list is correctly
    penalized.
    """
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for e in top if e in relevant)
    return hits / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items present in the top-``k`` results."""
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0
    top = set(ranked[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant hit, or 0 if none present.

    Rank is 1-based — the first item has rank 1, not 0.
    """
    if not relevant:
        return 0.0
    for i, e in enumerate(ranked, start=1):
        if e in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain @ ``k`` with binary relevance.

    Using the standard formula ``rel_i / log2(i + 1)`` (1-based ``i``)
    so the top position contributes 1.0 — matches both Wikipedia and
    sklearn ``dcg_score``.
    """
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return sum(
        1.0 / math.log2(i + 1)
        for i, e in enumerate(top, start=1)
        if e in relevant
    )


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG @ ``k`` (binary relevance).

    The ideal ordering places ``min(|relevant|, k)`` relevant items
    first. nDCG = DCG / IDCG ∈ [0, 1]; 1.0 = ideal ranking.
    """
    if not relevant or k <= 0:
        return 0.0
    dcg = dcg_at_k(ranked, relevant, k)
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def mean(values: Iterable[float]) -> float:
    """Plain arithmetic mean. Returns 0.0 for empty input — callers that
    care about "no data" should check the source list length, not the
    return value."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values: Iterable[float], p: float) -> float:
    """Linear-interpolation percentile (numpy-default behavior).

    ``p`` is in [0, 1]; e.g. p=0.95 returns the p95. Empty input → 0.0.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"percentile p must be in [0, 1], got {p}")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[int(idx)]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
