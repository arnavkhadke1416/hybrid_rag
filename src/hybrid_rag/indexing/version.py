"""Shared version identity for a synchronized retrieval-index pair."""

from __future__ import annotations

import hashlib
import json

from hybrid_rag.config import Settings


CHUNK_SCHEMA_VERSION = 1
INDEX_BUILD_VERSION = 1


def make_index_version(settings: Settings, corpus_fingerprint: str) -> str:
    from hybrid_rag.indexing.bm25_store import (
        BM25_B,
        BM25_EPSILON,
        BM25_K1,
        BM25KeywordStore,
        TOKENIZER_NAME,
    )
    from hybrid_rag.indexing.faiss_store import FaissVectorStore

    definition = {
        "build_version": INDEX_BUILD_VERSION,
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_model": settings.embedding_model,
        "faiss_schema_version": FaissVectorStore.SCHEMA_VERSION,
        "bm25_schema_version": BM25KeywordStore.SCHEMA_VERSION,
        "bm25_tokenizer": TOKENIZER_NAME,
        "bm25_parameters": {
            "k1": BM25_K1,
            "b": BM25_B,
            "epsilon": BM25_EPSILON,
        },
    }
    serialized = json.dumps(
        definition,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
