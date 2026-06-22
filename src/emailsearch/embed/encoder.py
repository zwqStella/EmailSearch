"""LlamaIndex used as a transform-only library: SentenceSplitter + HuggingFaceEmbedding.

No `IngestionPipeline`, no `VectorStoreIndex`. Plain function calls — the rest
of the codebase doesn't even need to know LlamaIndex exists.
"""

from __future__ import annotations

import logging
import threading
import time
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


def preload_models() -> None:
    """Eagerly initialize the splitter + embedding model.

    Called from the web app's lifespan startup so the first user-facing
    search doesn't pay the model-load latency. Safe to call multiple
    times (the ``_get_*`` helpers are double-checked-locked singletons).
    A warm-up embedding call is issued at the end so sentence-transformers'
    lazy weight loading happens here, not on the first query.
    """
    t0 = time.monotonic()
    _get_splitter()
    model = _get_embed_model()
    # First forward pass moves weights onto the chosen device; a throwaway
    # query forces that to happen now.
    try:
        model.get_query_embedding("warmup")
    except Exception:
        # Don't let a warm-up failure poison startup — the next real call
        # will surface the error through the normal code path.
        log.exception("embedding warm-up call failed; continuing")
    log.info("preloaded embedding stack in %.2fs", time.monotonic() - t0)


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


# Pure function of (text, model name) — same input always yields the same
# vector. Repeat searches (debounced typing, query refinement, back-button
# navigation) hit the same string over and over, so caching saves 10-50 ms
# of CPU per call on the hot path. Bounded so a long-running server with
# wild query variety doesn't grow memory unbounded; cached lists are tiny
# (384 floats ≈ 3 KB) so 1024 entries ≈ 3 MB.
_EMBED_QUERY_CACHE_MAX = 1024
_embed_query_cache: dict[tuple[str, str], list[float]] = {}
_embed_cache_lock = threading.Lock()


def clear_embed_query_cache() -> None:
    """Drop every cached query embedding. Used by tests."""
    with _embed_cache_lock:
        _embed_query_cache.clear()


def embed_query(text: str) -> list[float]:
    """Embed a search query (HF embeddings use the same encoding for query/text).

    Cached per (text, model name) so debounced re-searches don't re-run the
    encoder. Returns the cached list directly — callers must NOT mutate it.
    """
    settings = get_settings()
    key = (text, settings.embed_model)
    with _embed_cache_lock:
        cached = _embed_query_cache.get(key)
    if cached is not None:
        return cached

    vector = _get_embed_model().get_query_embedding(text)
    with _embed_cache_lock:
        # FIFO eviction — dict preserves insertion order in CPython.
        if (
            len(_embed_query_cache) >= _EMBED_QUERY_CACHE_MAX
            and key not in _embed_query_cache
        ):
            _embed_query_cache.pop(next(iter(_embed_query_cache)))
        _embed_query_cache[key] = vector
    return vector
