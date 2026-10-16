import logging

import numpy as np
import pytest

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL, Settings
from hybrid_rag.indexing.build_indexes import build_bm25_index, build_faiss_index
from hybrid_rag.ingestion.chunking import save_chunks
from hybrid_rag.retrieval.hybrid import (
    HybridIndexError,
    HybridQueryError,
    HybridRetriever,
)
from hybrid_rag.schemas import Chunk


EVALUATION_QUERIES = (
    ("semantic", "How can I regain access to my account?", "account"),
    ("exact-keyword", "AES-256", "security"),
)


class FakeEmbedder:
    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = 2

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encoded.extend(texts)
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "aes-256" in normalized:
                vectors.append([1.0, 0.0])
            elif "account" in normalized:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([-1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def make_chunks() -> tuple[Chunk, ...]:
    return (
        Chunk(
            "security",
            "manual",
            "manual.pdf",
            "manual.pdf",
            "pdf",
            0,
            "Encrypted backups use the AES-256 procedure.",
            page_number=4,
            section_heading="Encryption",
        ),
        Chunk(
            "account",
            "manual",
            "manual.pdf",
            "manual.pdf",
            "pdf",
            1,
            "To restore account access, request a password reset.",
            page_number=9,
            section_heading="Account recovery",
        ),
        Chunk(
            "printer",
            "manual",
            "manual.pdf",
            "manual.pdf",
            "pdf",
            2,
            "The printer maintenance schedule is updated monthly.",
            page_number=13,
            section_heading="Maintenance",
        ),
    )


def create_retriever(tmp_path) -> tuple[HybridRetriever, FakeEmbedder, tuple[Chunk, ...]]:
    (tmp_path / "documents").mkdir()
    settings = Settings(
        project_root=tmp_path,
        dense_top_k=2,
        bm25_top_k=2,
        fusion_top_k=2,
        rrf_k=10,
    )
    chunks = make_chunks()
    save_chunks(chunks, settings.processed_dir / "chunks.jsonl")
    embedder = FakeEmbedder()
    build_bm25_index(settings)
    build_faiss_index(settings, embedder=embedder)
    embedder.encoded.clear()
    return HybridRetriever(settings, embedder=embedder), embedder, chunks


def test_hybrid_evaluation_queries_use_both_paths_and_return_fused_metadata(tmp_path) -> None:
    retriever, embedder, _ = create_retriever(tmp_path)

    for category, query, expected_chunk_id in EVALUATION_QUERIES:
        response = retriever.retrieve(query)

        assert response.results[0].chunk.chunk_id == expected_chunk_id, category
        assert response.results[0].dense_score is not None
        assert response.results[0].bm25_score is not None
        assert response.results[0].fused_score == response.results[0].score
        assert response.results[0].chunk.source_path == "manual.pdf"
        assert response.results[0].chunk.page_number is not None
    assert embedder.encoded == [query for _, query, _ in EVALUATION_QUERIES]


def test_hybrid_normalizes_whitespace_and_reports_rank_diagnostics(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    retriever, embedder, _ = create_retriever(tmp_path)

    with caplog.at_level(logging.INFO, logger="hybrid_rag.retrieval.hybrid"):
        response = retriever.retrieve(" \t AES-256 \n", diagnostic=True)

    assert embedder.encoded == ["AES-256"]
    assert response.results[0].chunk.chunk_id == "security"
    diagnostic = response.diagnostics[0]
    assert diagnostic.chunk_id == "security"
    assert diagnostic.source_path == "manual.pdf"
    assert diagnostic.dense_rank == 1
    assert diagnostic.bm25_rank == 1
    assert diagnostic.fused_rank == 1
    assert "dense_rank=1 bm25_rank=1 fused_rank=1" in caplog.text
    assert "source=manual.pdf chunk_id=security" in caplog.text


def test_hybrid_rejects_invalid_queries_and_stale_live_snapshots(tmp_path) -> None:
    retriever, _, chunks = create_retriever(tmp_path)

    with pytest.raises(HybridQueryError, match="must not be empty"):
        retriever.retrieve(" \t ")
    with pytest.raises(HybridQueryError, match="searchable token"):
        retriever.retrieve("!!!")

    changed_chunks = (
        *chunks,
        Chunk("new", "manual", "manual.pdf", "manual.pdf", "pdf", 3, "New section."),
    )
    save_chunks(changed_chunks, retriever.settings.processed_dir / "chunks.jsonl")
    with pytest.raises(HybridIndexError, match="changed after this retriever"):
        retriever.retrieve("AES-256")


def test_missing_hybrid_indexes_give_build_instruction(tmp_path) -> None:
    (tmp_path / "documents").mkdir()
    settings = Settings(project_root=tmp_path)
    save_chunks(make_chunks(), settings.processed_dir / "chunks.jsonl")

    with pytest.raises(HybridIndexError) as error:
        HybridRetriever(settings, embedder=FakeEmbedder())

    assert r"python scripts\ingest_documents.py" in str(error.value)
