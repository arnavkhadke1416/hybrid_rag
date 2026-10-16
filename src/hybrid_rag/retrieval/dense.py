"""Dense candidate retrieval using the persisted FAISS index."""

from __future__ import annotations

from collections.abc import Sequence

from hybrid_rag.indexing.corpus import chunk_snapshot_fingerprint
from hybrid_rag.indexing.embeddings import EmbeddingEncoder
from hybrid_rag.indexing.faiss_store import FaissStoreError, FaissVectorStore
from hybrid_rag.schemas import Chunk, RetrievalResult


class DenseRetrievalError(ValueError):
    """Raised when dense retrieval inputs do not match the loaded index."""


def retrieve_dense(
    query: str,
    store: FaissVectorStore,
    embedder: EmbeddingEncoder,
    chunks: Sequence[Chunk],
    k: int,
) -> tuple[RetrievalResult, ...]:
    if not query.strip():
        raise DenseRetrievalError("Dense retrieval query must not be empty.")
    if k <= 0:
        raise DenseRetrievalError("Dense retrieval k must be a positive integer.")
    if embedder.model_name != store.embedding_model:
        raise DenseRetrievalError(
            "Query embedding model does not match the FAISS index; rebuild or load the matching index."
        )
    if embedder.dimension != store.dimension:
        raise DenseRetrievalError("Query embedding dimension does not match the FAISS index.")
    if store.corpus_fingerprint is None:
        raise DenseRetrievalError(
            "Dense index has no chunk snapshot metadata; rebuild it from the current chunks."
        )
    if chunk_snapshot_fingerprint(chunks) != store.corpus_fingerprint:
        raise DenseRetrievalError(
            "Dense index was built from a different chunk snapshot; rebuild it after ingestion."
        )

    chunks_by_id: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.chunk_id in chunks_by_id:
            raise DenseRetrievalError(f"Duplicate chunk ID in retrieval corpus: {chunk.chunk_id}")
        chunks_by_id[chunk.chunk_id] = chunk

    query_vector = embedder.encode([query])
    hits = store.search(query_vector, k)[0]
    results: list[RetrievalResult] = []
    for chunk_id, score in hits:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise FaissStoreError(
                f"FAISS index references missing chunk {chunk_id}; rebuild the dense index."
            )
        results.append(RetrievalResult(chunk=chunk, score=score, dense_score=score))
    return tuple(results)
