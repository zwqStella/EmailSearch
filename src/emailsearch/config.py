"""Application configuration loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. All env vars are prefixed `EMAILSEARCH_`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EMAILSEARCH_",
        extra="ignore",
    )

    # --- Storage ---
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".emailsearch")
    db_path: Path | None = None  # defaults to data_dir / "emails.db"

    # --- Embedding ---
    # Multilingual: handles English + Chinese + 50 other languages out of the box.
    # 384-dim (same as the English-only all-MiniLM-L6-v2 it replaced) so existing
    # vec0 schemas keep working — only re-embedding is needed after a change.
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384  # must match vec0 column width

    # --- OCR / extraction ---
    ocr_enabled: bool = True
    max_attachment_mb: int = 25

    # --- LLM summarization + query helpers ---
    # Best-effort: every call gracefully no-ops to None on any failure, so
    # search still works without a server. Set llm_enabled=False to skip the
    # network round-trips entirely.
    #
    # LLM uses:
    #   - ingest: `summarize_email` produces a per-email summary stored on
    #     the row, indexed in FTS, and embedded as `source_type='summary'`
    #     chunks so semantic search can match them via KNN.
    #   - search: `distill_query` strips filler before FTS; `augment_query`
    #     expands the query before embedding for the KNN leg.
    #
    # Defaults target the `copilot-api` proxy (OpenAI-compatible on :4141).
    # `llm_model` must be a name the endpoint actually serves — strict
    # backends (copilot-api, Azure OpenAI) reject unknown model names with
    # HTTP 400; LM Studio / Ollama / llama.cpp usually ignore it.
    llm_enabled: bool = True
    llm_base_url: str = "http://127.0.0.1:4141/v1"
    llm_model: str = "gpt-4o-mini"
    # Generous default — proxied GPT-class models can take 10-30s under load.
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 200             # cap on summary length
    llm_augment_max_tokens: int = 80      # cap on augmented-query length
    llm_distill_max_tokens: int = 40      # cap on distilled-query length
    llm_max_input_chars: int = 8000       # truncate body before summarizing

    # --- Semantic ranking ---
    # Per-chunk score floor for the embedding (KNN) leg. vec0 distances are
    # converted via ``1 / (1 + distance)`` (range [0, 1]); chunks scoring
    # below this are dropped before per-email grouping. Only the embedding
    # leg is gated — verbatim FTS matches remain a strong signal even when
    # the embedding similarity is incidental. Tune down for more recall, up
    # for higher confidence.
    semantic_score_threshold: float = 0.3

    # --- Diagnostics ---
    # When True, every /api/search response includes a `debug` field with
    # the per-leg query-transformation + ranking trace (logged to the
    # browser console). Cheap (capped previews, top-N entries) and very
    # useful for diagnosing surprising results.
    debug_enabled: bool = True

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path or (self.data_dir / "emails.db")

    @property
    def models_cache_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        """Create the data directory tree if missing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
