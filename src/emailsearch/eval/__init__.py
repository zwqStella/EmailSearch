"""Search-quality evaluation harness.

A small offline IR-style framework that runs a curated set of queries
through :func:`emailsearch.search.service.search` in every supported
mode and computes Precision@K, Recall@K, MRR, and nDCG@K plus latency
percentiles. Designed to answer a single question: *given a labeled set
of (query, relevant-emails) pairs, which mode wins, and by how much?*

Public surface:

- :mod:`emailsearch.eval.schema` — Pydantic models for the query set and
  the per-run report.
- :mod:`emailsearch.eval.metrics` — pure ranking-metric functions.
- :mod:`emailsearch.eval.runner` — runs queries against a live DB and
  collects per-query results.
- :mod:`emailsearch.eval.report` — markdown rendering.

CLI entry point: ``python -m emailsearch.eval run``.
"""
