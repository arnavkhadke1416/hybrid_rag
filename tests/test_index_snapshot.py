from dataclasses import replace

import numpy as np
import pytest

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL, Settings
from hybrid_rag.indexing.bm25_store import BM25KeywordStore, BM25StoreError
from hybrid_rag.indexing.build_indexes import build_bm25_index, build_faiss_index
from hybrid_rag.indexing.faiss_store import FaissVectorStore
from hybrid_rag.ingestion.chunking import load_chunks, save_chunks
from hybrid_rag.retrieval.bm25 import retrieve_bm25
from hybrid_rag.retrieval.dense import DenseRetrievalError, retrieve_dense
from hybrid_rag.retrieval.hybrid import HybridIndexError, HybridRetriever
from hybrid_rag.schemas import Chunk


class FakeEmbedder:
    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = 2

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if "AES-128" in text:
                vectors.append([0.0, 1.0])
            elif "AES-192" in text:
                vectors.append([-1.0, 0.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def make_chunks() -> tuple[Chunk, ...]:
    return (
        Chunk("a", "doc", "a.pdf", "a.pdf", "pdf", 0, "AES-256 reference"),
        Chunk("b", "doc", "a.pdf", "a.pdf", "pdf", 1, "AES-128 reference"),
    )


def test_dense_and_bm25_snapshots_stay_synchronized_across_corpus_changes(tmp_path) -> None:
    (tmp_path / "documents").mkdir()
    settings = Settings(project_root=tmp_path)
    chunks_path = settings.processed_dir / "chunks.jsonl"
    chunks = make_chunks()
    save_chunks(chunks, chunks_path)
    embedder = FakeEmbedder()

    dense = build_faiss_index(settings, embedder=embedder)
    sparse = build_bm25_index(settings)
    assert dense.corpus_fingerprint == sparse.corpus_fingerprint
    assert dense.index_version == sparse.index_version
    loaded_dense = FaissVectorStore.load(
        settings.faiss_index_dir,
        expected_dimension=embedder.dimension,
    )
    loaded_sparse = BM25KeywordStore.load(
        settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE,
        expected_corpus_fingerprint=sparse.corpus_fingerprint,
    )
    assert retrieve_dense("query", loaded_dense, embedder, chunks, k=2)[0].chunk.chunk_id == "a"
    assert retrieve_bm25("AES-256", loaded_sparse, chunks, k=2)[0].chunk.chunk_id == "a"

    changed_corpora = (
        chunks[1:],
        (*chunks, Chunk("c", "doc", "b.pdf", "b.pdf", "pdf", 0, "AES-192 reference")),
    )
    for changed_chunks in changed_corpora:
        save_chunks(changed_chunks, chunks_path)
        current_chunks = load_chunks(chunks_path)
        with pytest.raises(BM25StoreError, match="different chunk snapshot"):
            retrieve_bm25("AES-256", sparse, current_chunks, k=2)
        with pytest.raises(DenseRetrievalError, match="different chunk snapshot"):
            retrieve_dense("AES-256", dense, embedder, current_chunks, k=2)

        dense = build_faiss_index(settings, embedder=embedder)
        sparse = build_bm25_index(settings)
        assert dense.corpus_fingerprint == sparse.corpus_fingerprint
        assert dense.index_version == sparse.index_version


def test_hybrid_retriever_rejects_index_version_mismatch(tmp_path) -> None:
    (tmp_path / "documents").mkdir()
    settings = Settings(project_root=tmp_path)
    chunks = make_chunks()
    save_chunks(chunks, settings.processed_dir / "chunks.jsonl")
    embedder = FakeEmbedder()
    build_faiss_index(settings, embedder=embedder)
    build_bm25_index(settings)

    changed_settings = replace(settings, chunk_size=settings.chunk_size + 1)
    with pytest.raises(HybridIndexError, match="index version"):
        HybridRetriever(changed_settings, embedder=embedder)
