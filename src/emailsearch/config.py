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
