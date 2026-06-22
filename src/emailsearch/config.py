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
    # Best-effort: every call gracefully no-ops to None on any failure,
    # so search still works without a server. Set llm_enabled=False to
    # skip the network round-trips entirely.
    #
    # Defaults target the `copilot-api` proxy (OpenAI-compatible on
    # :4141). `llm_model` must be a name the endpoint serves — strict
    # backends (copilot-api, Azure OpenAI) reject unknown model names
    # with HTTP 400; LM Studio / Ollama / llama.cpp usually ignore it.
    # copilot-api transparently translates the OpenAI chat-completions
    # payload to Anthropic / Gemini for non-OpenAI model ids, so any id
    # from ``GET /v1/models`` is fair game.
    llm_enabled: bool = True
    llm_base_url: str = "http://127.0.0.1:4141/v1"
    llm_model: str = "claude-haiku-4.5"
    # Generous default — proxied LLM calls can take 10-30s under load.
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 200             # cap on summary length
    llm_augment_max_tokens: int = 80      # cap on augmented-query length
    llm_distill_max_tokens: int = 40      # cap on distilled-query length
    llm_max_input_chars: int = 8000       # truncate body before summarizing

    # --- Semantic ranking ---
    # Per-chunk score floor for the embedding (KNN) leg. vec0 distances
    # are converted via ``1 / (1 + distance)`` (range [0, 1]); chunks
    # below this are dropped before per-email grouping. Only the
    # embedding leg is gated — FTS verbatim matches remain a strong
    # signal even when embedding similarity is incidental.
    semantic_score_threshold: float = 0.3

    # --- Ask (RAG) tab ---
    # Retrieval breadth for the agent's single search tool-call. Tune up
    # for "summarize my recent emails about X" style questions that need
    # broader context.
    ask_retrieval_limit: int = 8
    # Max output tokens for the streaming synthesis call. The answer
    # plus its inline ``[N]`` citations has to fit in this budget.
    ask_max_answer_tokens: int = 600
    # Token budget for the synthesis prompt's sources block (all
    # selected emails combined). CJK-aware estimator + max-min
    # fair-share allocation: short emails get their full content, long
    # ones share what's left. 8000 tokens is a latency/cost cap, not a
    # capability limit — Claude Haiku / GPT-4.1 / Gemini Flash all have
    # 100K+ context windows. Raise it if Ask answers feel under-grounded
    # (and you're OK paying for the extra input tokens).
    ask_max_prompt_tokens: int = 8000
    # Max emails to read FULLY after triage. Keeping this small (3) is
    # what gets the synthesis prompt from ~24K tokens down to ~5-10K.
    # Triage is SKIPPED when ``len(hits) <= ask_triage_limit`` or when
    # the LLM is disabled — both cases fall through to "read all" with
    # the same fair-share allocator.
    ask_triage_limit: int = 3
    # Token cap for the triage hop. Output is a single short line of
    # comma-separated indexes or "NONE", so keep the budget tight.
    ask_triage_max_tokens: int = 60
    # Token cap for the question-parser hop (combined distill + filter
    # extraction). Output is a single short JSON object.
    ask_parse_max_tokens: int = 200

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
