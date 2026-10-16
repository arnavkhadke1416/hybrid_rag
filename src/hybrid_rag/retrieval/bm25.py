"""Keyword retrieval using the persisted BM25 index."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hybrid_rag.indexing.bm25_store import BM25KeywordStore, BM25StoreError, tokenize_text
from hybrid_rag.indexing.corpus import chunk_snapshot_fingerprint
from hybrid_rag.schemas import Chunk, RetrievalResult


class BM25RetrievalError(ValueError):
    """Raised when BM25 retrieval inputs do not match the indexed corpus."""


def retrieve_bm25(
    query: str,
    store: BM25KeywordStore,
    chunks: Sequence[Chunk],
    k: int,
) -> tuple[RetrievalResult, ...]:
    if not query.strip():
        raise BM25RetrievalError("BM25 query must not be empty.")
    if k <= 0:
        raise BM25RetrievalError("BM25 k must be a positive integer.")
    query_tokens = tokenize_text(query)
    if not query_tokens:
        raise BM25RetrievalError("BM25 query must contain at least one searchable token.")
    if chunk_snapshot_fingerprint(chunks) != store.corpus_fingerprint:
        raise BM25StoreError(
            "BM25 index was built from a different chunk snapshot; rebuild it after ingestion."
        )

    scores = store.scores(query_tokens)
    if len(scores) != len(store.chunk_ids) or not np.isfinite(scores).all():
        raise BM25StoreError("BM25 index returned invalid scores for the current query.")

    query_token_set = set(query_tokens)
    matching_positions = [
        position
        for position, document_tokens in enumerate(store.tokenized_corpus)
        if query_token_set.intersection(document_tokens)
    ]
    ranked_positions = sorted(
        matching_positions,
        key=lambda position: (-float(scores[position]), position),
    )[:k]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return tuple(
        RetrievalResult(
            chunk=chunks_by_id[store.chunk_ids[position]],
            score=float(scores[position]),
            bm25_score=float(scores[position]),
        )
        for position in ranked_positions
    )
