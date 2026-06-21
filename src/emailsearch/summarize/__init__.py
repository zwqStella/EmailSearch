"""LLM-backed helpers via a local OpenAI-compatible endpoint.

Three operations are exposed:
  - `summarize_email`: ingest-time per-email summary, stored on the email row.
  - `distill_query`: query-time filler-stripping for the semantic_fts leg.
  - `augment_query`: query-time expansion of the user's search query,
    used by the semantic_knn leg before embedding.

All three are best-effort and return `None` on any failure / when disabled.
"""

from emailsearch.summarize.client import augment_query, distill_query, summarize_email

__all__ = ["augment_query", "distill_query", "summarize_email"]
