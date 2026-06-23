"""Markdown report renderer.

Produces a single self-contained ``report.md`` with three sections:

1. **Headline table** — one row per mode with the four canonical IR
   metrics plus latency percentiles. The number that goes in the
   teacher-facing claim.
2. **Per-category breakdown** — same metrics, sliced by query category,
   so you can say *"keyword wins verbatim, semantic wins semantic"*.
3. **Per-query appendix** — a row per query with the rank of each
   relevant ID. Lets the reader sanity-check that the aggregates aren't
   masking pathological cases.

No external templating dep — straight f-strings + a tiny formatter.
"""

from __future__ import annotations

from emailsearch.eval.schema import EvalReport, ModeSummary, QueryResult
from emailsearch.search.service import SearchMode

# Column widths chosen so the rendered tables stay readable in a plain
# Markdown preview; values that overflow just wrap.
_HEADLINE_COLS = (
    "Mode",
    "n",
    "P@5",
    "P@10",
    "P@20",
    "R@5",
    "R@10",
    "R@20",
    "MRR",
    "nDCG@10",
    "p50 (ms)",
    "p95 (ms)",
)


def render(report: EvalReport) -> str:
    """Return the full markdown document as a single string."""
    parts: list[str] = []
    parts.append(_header(report))
    parts.append(_headline_table(report))
    if report.per_category:
        parts.append(_per_category_section(report))
    parts.append(_per_query_section(report))
    return "\n".join(parts) + "\n"


def _header(report: EvalReport) -> str:
    meta = report.eval_set_meta
    lines = ["# Search-quality evaluation report", ""]
    if "description" in meta:
        lines.append(f"_{meta['description']}_")
        lines.append("")
    if "authored" in meta:
        lines.append(f"- **Eval set authored:** {meta['authored']}")
    n_queries = len({r.query_id for r in report.per_query})
    lines.append(f"- **Queries:** {n_queries}")
    lines.append(f"- **Modes:** {', '.join(report.modes)}")
    if "notes" in meta:
        lines.append("")
        lines.append(meta["notes"].strip())
    lines.append("")
    return "\n".join(lines)


def _headline_table(report: EvalReport) -> str:
    lines = ["## Overall metrics", ""]
    lines.append("| " + " | ".join(_HEADLINE_COLS) + " |")
    lines.append("|" + "|".join(["---"] * len(_HEADLINE_COLS)) + "|")

    # Identify the winner per metric column so we can bold it. Higher
    # is better for everything except latency.
    higher_better = ["P@5", "P@10", "P@20", "R@5", "R@10", "R@20", "MRR", "nDCG@10"]
    lower_better = ["p50 (ms)", "p95 (ms)"]
    metric_values = {m: _metric_values(report.summaries, m) for m in higher_better + lower_better}
    winners = {
        m: _argmax(metric_values[m]) for m in higher_better
    }
    winners.update({m: _argmin(metric_values[m]) for m in lower_better})

    for i, s in enumerate(report.summaries):
        row = [
            s.mode,
            str(s.n_queries),
            _emph(_fmt(s.mean_precision_at_5), winners["P@5"] == i),
            _emph(_fmt(s.mean_precision_at_10), winners["P@10"] == i),
            _emph(_fmt(s.mean_precision_at_20), winners["P@20"] == i),
            _emph(_fmt(s.mean_recall_at_5), winners["R@5"] == i),
            _emph(_fmt(s.mean_recall_at_10), winners["R@10"] == i),
            _emph(_fmt(s.mean_recall_at_20), winners["R@20"] == i),
            _emph(_fmt(s.mrr), winners["MRR"] == i),
            _emph(_fmt(s.mean_ndcg_at_10), winners["nDCG@10"] == i),
            _emph(_fmt_ms(s.p50_latency_ms), winners["p50 (ms)"] == i),
            _emph(_fmt_ms(s.p95_latency_ms), winners["p95 (ms)"] == i),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "_Bold = best in column. Higher = better for quality metrics; "
        "lower = better for latency._"
    )
    lines.append("")
    lines.append(
        "_K choice: P@5 captures the user's foveal scan; P@10 is the conventional "
        "IR \"first-page\" cutoff; P@20 covers every hit the system returns "
        "(``RUN_LIMIT = 20``). Precision shrinks at larger K because "
        "the denominator grows but the relevant set is small \u2014 nDCG@10 corrects "
        "for that by rewarding higher-ranked hits more strongly than lower-ranked ones._"
    )
    lines.append("")
    return "\n".join(lines)


def _per_category_section(report: EvalReport) -> str:
    lines = ["## Per-category breakdown", ""]
    lines.append("Same metrics, sliced by query category — exposes where each mode shines.\n")
    cols = ("Category", "Mode", "n", "P@5", "P@10", "MRR", "nDCG@10")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for category, by_mode in sorted(report.per_category.items()):
        # Bold the per-row winner inside each category (using nDCG@10 as
        # the tie-breaker since it captures both precision and rank).
        best_mode = max(by_mode.items(), key=lambda kv: kv[1].mean_ndcg_at_10)[0]
        for mode, s in by_mode.items():
            row = [
                category,
                mode,
                str(s.n_queries),
                _fmt(s.mean_precision_at_5),
                _fmt(s.mean_precision_at_10),
                _fmt(s.mrr),
                _fmt(s.mean_ndcg_at_10),
            ]
            if mode == best_mode:
                row = [_emph(c, True) for c in row]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _per_query_section(report: EvalReport) -> str:
    lines = ["## Per-query detail", ""]
    lines.append("_Rank columns show the 1-based position of the first relevant hit, "
                 "or `–` if no relevant hit landed in the top 20._")
    lines.append("")

    by_qid: dict[str, dict[SearchMode, QueryResult]] = {}
    for r in report.per_query:
        by_qid.setdefault(r.query_id, {})[r.mode] = r

    cols = ["Query", "|rel|"] + [f"{m}: rank/lat" for m in report.modes]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for qid in sorted(by_qid):
        per_mode = by_qid[qid]
        first_mode = next(iter(per_mode.values()))
        cells = [qid, str(first_mode.n_relevant)]
        for mode in report.modes:
            r = per_mode.get(mode)
            if r is None:
                cells.append("–")
                continue
            rank = _first_relevant_rank(r)
            rank_str = str(rank) if rank else "–"
            cells.append(f"{rank_str} / {_fmt_ms(r.latency_ms)}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# small helpers — kept inline because they're presentation-only
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _fmt_ms(x: float) -> str:
    if x >= 1000.0:
        return f"{x / 1000.0:.2f}s"
    return f"{x:.0f}"


def _emph(text: str, on: bool) -> str:
    return f"**{text}**" if on else text


def _metric_values(summaries: list[ModeSummary], col: str) -> list[float]:
    getter = {
        "P@5": lambda s: s.mean_precision_at_5,
        "P@10": lambda s: s.mean_precision_at_10,
        "P@20": lambda s: s.mean_precision_at_20,
        "R@5": lambda s: s.mean_recall_at_5,
        "R@10": lambda s: s.mean_recall_at_10,
        "R@20": lambda s: s.mean_recall_at_20,
        "MRR": lambda s: s.mrr,
        "nDCG@10": lambda s: s.mean_ndcg_at_10,
        "p50 (ms)": lambda s: s.p50_latency_ms,
        "p95 (ms)": lambda s: s.p95_latency_ms,
    }[col]
    return [getter(s) for s in summaries]


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i]) if values else -1


def _argmin(values: list[float]) -> int:
    return min(range(len(values)), key=lambda i: values[i]) if values else -1


def _first_relevant_rank(result: QueryResult) -> int | None:
    """1-based rank of the first relevant hit, or None if absent."""
    if result.reciprocal_rank == 0.0:
        return None
    # Reciprocal rank already encodes the position; invert it back.
    return round(1.0 / result.reciprocal_rank)
