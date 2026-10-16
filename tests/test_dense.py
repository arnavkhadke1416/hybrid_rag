import numpy as np
import pytest

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL
from hybrid_rag.indexing.corpus import chunk_snapshot_fingerprint
from hybrid_rag.indexing.faiss_store import FaissVectorStore, FaissStoreError
from hybrid_rag.retrieval.dense import DenseRetrievalError, retrieve_dense
from hybrid_rag.schemas import Chunk


class FakeEmbedder:
    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = 2

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encoded.extend(texts)
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "query": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def make_chunks() -> tuple[Chunk, ...]:
    return (
        Chunk("a", "doc", "a.pdf", "a.pdf", "pdf", 0, "alpha"),
        Chunk("b", "doc", "a.pdf", "a.pdf", "pdf", 1, "beta"),
    )


def test_dense_retrieval_maps_top_k_to_chunk_records_and_reuses_index() -> None:
    chunks = make_chunks()
    embedder = FakeEmbedder()
    store = FaissVectorStore.from_vectors(
        [chunk.chunk_id for chunk in chunks],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        embedder.model_name,
        corpus_fingerprint=chunk_snapshot_fingerprint(chunks),
    )
    index_identity = id(store.index)

    first = retrieve_dense("query", store, embedder, chunks, k=10)
    second = retrieve_dense("query", store, embedder, chunks, k=1)

    assert [result.chunk.chunk_id for result in first] == ["a", "b"]
    assert first[0].dense_score == pytest.approx(1.0)
    assert [result.chunk.chunk_id for result in second] == ["a"]
    assert embedder.encoded == ["query", "query"]
    assert id(store.index) == index_identity


def test_dense_retrieval_rejects_invalid_or_mismatched_inputs() -> None:
    chunks = make_chunks()
    embedder = FakeEmbedder()
    store = FaissVectorStore.from_vectors(
        [chunk.chunk_id for chunk in chunks],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        embedder.model_name,
        corpus_fingerprint=chunk_snapshot_fingerprint(chunks),
    )

    with pytest.raises(DenseRetrievalError, match="must not be empty"):
        retrieve_dense(" ", store, embedder, chunks, 1)
    with pytest.raises(DenseRetrievalError, match="positive integer"):
        retrieve_dense("query", store, embedder, chunks, 0)
    with pytest.raises(DenseRetrievalError, match="model does not match"):
        retrieve_dense("query", store, FakeEmbedderWithOtherModel(), chunks, 1)
    with pytest.raises(DenseRetrievalError, match="different chunk snapshot"):
        retrieve_dense("query", store, embedder, chunks[:1], 2)

    mismapped_store = FaissVectorStore.from_vectors(
        ["a", "missing"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        embedder.model_name,
        corpus_fingerprint=chunk_snapshot_fingerprint(chunks),
    )
    with pytest.raises(FaissStoreError, match="missing chunk"):
        retrieve_dense("query", mismapped_store, embedder, chunks, 2)


class FakeEmbedderWithOtherModel(FakeEmbedder):
    model_name = "different/model"
