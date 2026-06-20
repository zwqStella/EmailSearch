"""LlamaIndex used as a transform-only library: SentenceSplitter + HuggingFaceEmbedding.

No `IngestionPipeline`, no `VectorStoreIndex`. Plain function calls — the rest
of the codebase doesn't even need to know LlamaIndex exists.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from emailsearch.config import get_settings

if TYPE_CHECKING:
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

log = logging.getLogger(__name__)

_splitter: SentenceSplitter | None = None
_embed_model: HuggingFaceEmbedding | None = None
_init_lock = threading.Lock()

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _get_splitter() -> SentenceSplitter:
    global _splitter
    if _splitter is None:
        with _init_lock:
            if _splitter is None:
                from llama_index.core.node_parser import SentenceSplitter

                _splitter = SentenceSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
    return _splitter


def _get_embed_model() -> HuggingFaceEmbedding:
    global _embed_model
    if _embed_model is None:
        with _init_lock:
            if _embed_model is None:
                settings = get_settings()
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                log.info(
                    "loading embedding model %s (cache_folder=%s)",
                    settings.embed_model,
                    settings.models_cache_dir,
                )
                _embed_model = HuggingFaceEmbedding(
                    model_name=settings.embed_model,
                    cache_folder=str(settings.models_cache_dir),
                    embed_batch_size=32,
                )
    return _embed_model


def chunk_text(text: str) -> list[str]:
    """Split text into sentence-aware chunks. Returns [] for empty input."""
    if not text or not text.strip():
        return []
    from llama_index.core import Document

    nodes = _get_splitter().get_nodes_from_documents([Document(text=text)])
    return [n.get_content() for n in nodes]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns one float-list per input (384-dim with default model)."""
    if not texts:
        return []
    return _get_embed_model().get_text_embedding_batch(texts)


def embed_query(text: str) -> list[float]:
    """Embed a search query (HF embeddings use the same encoding for query/text)."""
    return _get_embed_model().get_query_embedding(text)
