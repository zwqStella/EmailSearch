"""Unit tests for the ranking-metric implementations.

All math here is deterministic and offline — no DB, no LLM, no
embedding. Anchored to hand-computed examples so a future refactor that
silently changes the formula will fail loudly.
"""

from __future__ import annotations

import math

import pytest

from emailsearch.eval.metrics import (
    dcg_at_k,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_all_relevant_top_k(self) -> None:
        # Top 3 are all relevant → P@3 = 1.0.
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(ranked, relevant, 3) == pytest.approx(1.0)

    def test_none_relevant_top_k(self) -> None:
        ranked = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(ranked, relevant, 3) == pytest.approx(0.0)

    def test_mixed_top_k(self) -> None:
        # 2 of top 5 are relevant → P@5 = 0.4.
        ranked = ["a", "x", "b", "y", "z"]
        relevant = {"a", "b", "q"}
        assert precision_at_k(ranked, relevant, 5) == pytest.approx(0.4)

    def test_denominator_is_k_not_min(self) -> None:
        # Short result list MUST be penalized — denominator is k, not
        # len(ranked). One relevant hit in a 2-long list at k=5 gives
        # 1/5 = 0.2, not 1/2.
        ranked = ["a", "x"]
        relevant = {"a"}
        assert precision_at_k(ranked, relevant, 5) == pytest.approx(0.2)

    def test_precision_shrinks_at_larger_k(self) -> None:
        # When K grows but the relevant set is small, P@K mechanically
        # shrinks because the denominator grows. 2 relevant hits at the
        # top: P@5 = 0.4, P@10 = 0.2, P@20 = 0.1. This is the IR-textbook
        # reason small relevant sets are reported with K=5/10, not 20.
        ranked = ["a", "b"] + [f"x{i}" for i in range(30)]
        relevant = {"a", "b"}
        assert precision_at_k(ranked, relevant, 5) == pytest.approx(0.4)
        assert precision_at_k(ranked, relevant, 10) == pytest.approx(0.2)
        assert precision_at_k(ranked, relevant, 20) == pytest.approx(0.1)

    def test_k_zero(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_empty_ranked(self) -> None:
        assert precision_at_k([], {"a"}, 5) == 0.0


class TestRecallAtK:
    def test_full_recall_at_k(self) -> None:
        ranked = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(ranked, relevant, 3) == pytest.approx(1.0)

    def test_partial_recall_at_k(self) -> None:
        # 1 of 2 relevant in top 3 → R@3 = 0.5.
        ranked = ["a", "x", "y", "b"]
        relevant = {"a", "b"}
        assert recall_at_k(ranked, relevant, 3) == pytest.approx(0.5)

    def test_zero_when_no_relevant_in_top_k(self) -> None:
        ranked = ["x", "y", "z", "a"]
        relevant = {"a"}
        assert recall_at_k(ranked, relevant, 3) == 0.0

    def test_empty_relevant_set(self) -> None:
        # Convention: undefined → 0.0 (caller filters these out before
        # aggregating).
        assert recall_at_k(["a", "b"], set(), 3) == 0.0


class TestReciprocalRank:
    def test_first_hit_at_rank_1(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_first_hit_at_rank_3(self) -> None:
        # 1-based rank — third position → 1/3.
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)

    def test_no_hit_returns_zero(self) -> None:
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_only_first_counts(self) -> None:
        # Second item is relevant → 1/2. The third, also relevant, is
        # ignored — that's what *reciprocal rank* means.
        assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == pytest.approx(0.5)


class TestDcgAtK:
    def test_single_hit_top(self) -> None:
        # rel at position 1 → 1 / log2(2) = 1.
        assert dcg_at_k(["a", "x"], {"a"}, 5) == pytest.approx(1.0)

    def test_single_hit_pos_2(self) -> None:
        # rel at position 2 → 1 / log2(3).
        expected = 1.0 / math.log2(3)
        assert dcg_at_k(["x", "a"], {"a"}, 5) == pytest.approx(expected)

    def test_multiple_hits_summed(self) -> None:
        # Hits at positions 1 and 3: 1/log2(2) + 1/log2(4) = 1 + 0.5.
        ranked = ["a", "x", "b"]
        assert dcg_at_k(ranked, {"a", "b"}, 5) == pytest.approx(1.5)


class TestNdcgAtK:
    def test_ideal_ordering_is_one(self) -> None:
        # All relevant items at the top → nDCG = 1.
        ranked = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(1.0)

    def test_reversed_ordering_drops_score(self) -> None:
        # Same relevant set, but pushed to the back — nDCG < 1.
        ranked = ["x", "y", "a", "b", "c"]
        relevant = {"a", "b", "c"}
        score = ndcg_at_k(ranked, relevant, 10)
        assert 0.0 < score < 1.0

    def test_no_relevant_returns_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), 10) == 0.0

    def test_idcg_caps_at_k(self) -> None:
        # 5 relevant items but k=2 — IDCG considers only the first 2
        # positions, so 2 hits at top-2 gives nDCG = 1.0 (not 2/5).
        ranked = ["a", "b", "x", "y", "z"]
        relevant = {"a", "b", "c", "d", "e"}
        assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(1.0)


class TestAggregates:
    def test_mean_empty(self) -> None:
        assert mean([]) == 0.0

    def test_mean_basic(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_percentile_single_value(self) -> None:
        assert percentile([42.0], 0.5) == 42.0

    def test_percentile_p50_odd_length(self) -> None:
        # Median of [1,2,3,4,5] is 3.
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == pytest.approx(3.0)

    def test_percentile_p50_even_length(self) -> None:
        # Linear-interp median of [1,2,3,4] is 2.5.
        assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)

    def test_percentile_p95(self) -> None:
        # For [10, 20, ..., 100] (n=10) the p95 lands at index
        # 0.95 * 9 = 8.55, interpolating between 90 (idx 8) and
        # 100 (idx 9): 90 * 0.45 + 100 * 0.55 = 95.5.
        vals = [10.0 * (i + 1) for i in range(10)]
        assert percentile(vals, 0.95) == pytest.approx(95.5)

    def test_percentile_unsorted_input(self) -> None:
        # Implementation must sort internally; caller can pass any order.
        assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == pytest.approx(3.0)

    def test_percentile_out_of_range_p_raises(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], 1.5)
