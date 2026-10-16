import pytest

from hybrid_rag.indexing.bm25_store import BM25KeywordStore, BM25StoreError
from hybrid_rag.retrieval.bm25 import BM25RetrievalError, retrieve_bm25
from hybrid_rag.schemas import Chunk


def make_chunks() -> tuple[Chunk, ...]:
    return (
        Chunk(
            "a",
            "doc",
            "a.pdf",
            "a.pdf",
            "pdf",
            0,
            "Reference AES-256 and version v2.1.0 for product ZX-42.",
        ),
        Chunk("b", "doc", "a.pdf", "a.pdf", "pdf", 1, "AES-128 protects another record."),
        Chunk("c", "doc", "a.pdf", "a.pdf", "pdf", 2, "A document about apples and fruit."),
    )


def test_bm25_retrieval_returns_matching_chunk_records_for_exact_terms() -> None:
    chunks = make_chunks()
    store = BM25KeywordStore.from_chunks(chunks)

    result = retrieve_bm25("AES-256!", store, chunks, k=10)
    identifier_result = retrieve_bm25("ZX-42,", store, chunks, k=1)

    assert [item.chunk.chunk_id for item in result] == ["a"]
    assert result[0].bm25_score == result[0].score
    assert identifier_result[0].chunk.chunk_id == "a"
    assert retrieve_bm25("term-not-in-corpus", store, chunks, k=5) == ()


def test_empty_queries_and_invalid_k_are_rejected() -> None:
    chunks = make_chunks()
    store = BM25KeywordStore.from_chunks(chunks)

    with pytest.raises(BM25RetrievalError, match="must not be empty"):
        retrieve_bm25("   ", store, chunks, k=1)
    with pytest.raises(BM25RetrievalError, match="searchable token"):
        retrieve_bm25("!!!", store, chunks, k=1)
    with pytest.raises(BM25RetrievalError, match="positive integer"):
        retrieve_bm25("AES-256", store, chunks, k=0)


def test_retrieval_rejects_changed_chunk_snapshots() -> None:
    chunks = make_chunks()
    store = BM25KeywordStore.from_chunks(chunks)

    with pytest.raises(BM25StoreError, match="different chunk snapshot"):
        retrieve_bm25("AES-256", store, chunks[:-1], k=1)
