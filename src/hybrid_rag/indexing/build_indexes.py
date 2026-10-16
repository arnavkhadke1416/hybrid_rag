"""Build the dense FAISS index from the persisted chunk snapshot."""

from __future__ import annotations

import numpy as np

from hybrid_rag.config import Settings
from hybrid_rag.indexing.bm25_store import BM25KeywordStore, BM25StoreError
from hybrid_rag.indexing.corpus import chunk_snapshot_fingerprint
from hybrid_rag.indexing.embeddings import EmbeddingEncoder, SentenceTransformerEmbeddings
from hybrid_rag.indexing.faiss_store import FaissStoreError, FaissVectorStore
from hybrid_rag.indexing.version import make_index_version
from hybrid_rag.ingestion.chunking import load_chunks


def build_faiss_index(
    settings: Settings | None = None,
    *,
    embedder: EmbeddingEncoder | None = None,
) -> FaissVectorStore:
    settings = settings or Settings.from_env()
    chunks_path = settings.processed_dir / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FaissStoreError(
            f"Processed chunks are missing at {chunks_path}; run the ingestion pipeline first."
        )
    chunks = load_chunks(chunks_path)

    encoder = embedder or SentenceTransformerEmbeddings(settings.embedding_model)
    if encoder.model_name != settings.embedding_model:
        raise FaissStoreError("Embedding encoder model does not match the configured model.")
    vectors = (
        encoder.encode([chunk.text for chunk in chunks])
        if chunks
        else np.empty((0, encoder.dimension), dtype=np.float32)
    )
    fingerprint = chunk_snapshot_fingerprint(chunks)
    store = FaissVectorStore.from_vectors(
        [chunk.chunk_id for chunk in chunks],
        vectors,
        embedding_model=encoder.model_name,
        corpus_fingerprint=fingerprint,
        index_version=make_index_version(settings, fingerprint),
    )
    store.save(settings.faiss_index_dir)
    return store


def build_bm25_index(settings: Settings | None = None) -> BM25KeywordStore:
    settings = settings or Settings.from_env()
    chunks_path = settings.processed_dir / "chunks.jsonl"
    if not chunks_path.is_file():
        raise BM25StoreError(
            f"Processed chunks are missing at {chunks_path}; run the ingestion pipeline first."
        )
    try:
        chunks = load_chunks(chunks_path)
    except (OSError, ValueError) as error:
        raise BM25StoreError(f"Could not load processed chunks from {chunks_path}.") from error
    fingerprint = chunk_snapshot_fingerprint(chunks)
    store = BM25KeywordStore.from_chunks(
        chunks,
        index_version=make_index_version(settings, fingerprint),
    )
    store.save(settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE)
    return store
