"""End-to-end dense, keyword, and fused retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hybrid_rag.config import Settings
from hybrid_rag.indexing.bm25_store import BM25KeywordStore, BM25StoreError, tokenize_text
from hybrid_rag.indexing.corpus import chunk_snapshot_fingerprint
from hybrid_rag.indexing.embeddings import EmbeddingEncoder, SentenceTransformerEmbeddings
from hybrid_rag.indexing.faiss_store import FaissStoreError, FaissVectorStore
from hybrid_rag.indexing.version import make_index_version
from hybrid_rag.ingestion.chunking import load_chunks
from hybrid_rag.retrieval.bm25 import retrieve_bm25
from hybrid_rag.retrieval.dense import retrieve_dense
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.schemas import Chunk, RetrievalResult


LOGGER = logging.getLogger(__name__)
INGESTION_COMMAND = r"python scripts\ingest_documents.py"
REBUILD_COMMAND = r"python scripts\rebuild_indexes.py"


class HybridIndexError(RuntimeError):
    """Raised when hybrid retrieval indexes are missing or incompatible."""


class HybridQueryError(ValueError):
    """Raised when a query is empty or contains no searchable tokens."""


@dataclass(frozen=True)
class HybridRetrievalDiagnostic:
    chunk_id: str
    source_path: str
    dense_rank: int | None
    bm25_rank: int | None
    fused_rank: int


@dataclass(frozen=True)
class HybridRetrievalResponse:
    results: tuple[RetrievalResult, ...]
    diagnostics: tuple[HybridRetrievalDiagnostic, ...] = ()


class HybridRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingEncoder | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        chunks_path = self.settings.processed_dir / "chunks.jsonl"
        if not chunks_path.is_file():
            raise HybridIndexError(
                f"Processed chunks are missing at {chunks_path}; run {INGESTION_COMMAND}."
            )
        try:
            self.chunks: tuple[Chunk, ...] = load_chunks(chunks_path)
        except (OSError, ValueError) as error:
            raise HybridIndexError(
                f"Could not load processed chunks from {chunks_path}; run {INGESTION_COMMAND}."
            ) from error
        self.corpus_fingerprint = chunk_snapshot_fingerprint(self.chunks)
        expected_index_version = make_index_version(
            self.settings,
            self.corpus_fingerprint,
        )

        bm25_path = self.settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE
        if not (self.settings.faiss_index_dir / FaissVectorStore.INDEX_FILE).is_file() or not (
            self.settings.faiss_index_dir / FaissVectorStore.MAPPING_FILE
        ).is_file() or not bm25_path.is_file():
            raise HybridIndexError(
                f"Dense or BM25 indexes are missing; run {INGESTION_COMMAND}."
            )
        try:
            self.bm25_store = BM25KeywordStore.load(
                bm25_path,
                expected_corpus_fingerprint=self.corpus_fingerprint,
                expected_index_version=expected_index_version,
            )
        except BM25StoreError as error:
            raise HybridIndexError(
                f"{error} Run {REBUILD_COMMAND}."
            ) from error

        self.embedder = (
            embedder
            if embedder is not None
            else SentenceTransformerEmbeddings(self.settings.embedding_model)
        )
        if self.embedder.model_name != self.settings.embedding_model:
            raise HybridIndexError(
                "Embedding encoder does not match the configured model; rebuild indexes with "
                f"{REBUILD_COMMAND}."
            )
        try:
            self.dense_store = FaissVectorStore.load(
                self.settings.faiss_index_dir,
                expected_model=self.settings.embedding_model,
                expected_dimension=self.embedder.dimension,
                expected_index_version=expected_index_version,
            )
        except FaissStoreError as error:
            raise HybridIndexError(f"{error} Run {REBUILD_COMMAND}.") from error
        if self.dense_store.corpus_fingerprint != self.corpus_fingerprint:
            raise HybridIndexError(
                f"Dense index was built from a different chunk snapshot; run {REBUILD_COMMAND}."
            )
        if self.dense_store.corpus_fingerprint != self.bm25_store.corpus_fingerprint:
            raise HybridIndexError(
                f"Dense and BM25 indexes are out of sync; run {REBUILD_COMMAND}."
            )
        if self.dense_store.index_version != self.bm25_store.index_version:
            raise HybridIndexError(
                f"Dense and BM25 index versions differ; run {REBUILD_COMMAND}."
            )

    def _ensure_current_snapshot(self) -> None:
        chunks_path = self.settings.processed_dir / "chunks.jsonl"
        try:
            current_chunks = load_chunks(chunks_path)
        except (OSError, ValueError) as error:
            raise HybridIndexError(
                f"Could not reload processed chunks from {chunks_path}; run {INGESTION_COMMAND}."
            ) from error
        if chunk_snapshot_fingerprint(current_chunks) != self.corpus_fingerprint:
            raise HybridIndexError(
                "Processed chunks changed after this retriever was initialized; rebuild both "
                f"indexes with {REBUILD_COMMAND} and create a new HybridRetriever."
            )

    def retrieve(
        self,
        query: str,
        *,
        diagnostic: bool = False,
    ) -> HybridRetrievalResponse:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise HybridQueryError("Hybrid query must not be empty.")
        if not tokenize_text(normalized_query):
            raise HybridQueryError("Hybrid query must contain at least one searchable token.")
        self._ensure_current_snapshot()
        if not self.chunks:
            return HybridRetrievalResponse(results=())

        dense_results = retrieve_dense(
            normalized_query,
            self.dense_store,
            self.embedder,
            self.chunks,
            self.settings.dense_top_k,
        )
        bm25_results = retrieve_bm25(
            normalized_query,
            self.bm25_store,
            self.chunks,
            self.settings.bm25_top_k,
        )
        if diagnostic:
            fused_results = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                rrf_k=self.settings.rrf_k,
                top_k=max(len(dense_results) + len(bm25_results), 1),
            )
        else:
            fused_results = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                rrf_k=self.settings.rrf_k,
                top_k=self.settings.fusion_top_k,
            )
        diagnostics = (
            self._diagnostics(dense_results, bm25_results, fused_results)
            if diagnostic
            else ()
        )
        return HybridRetrievalResponse(
            results=fused_results[: self.settings.fusion_top_k],
            diagnostics=diagnostics,
        )

    @staticmethod
    def _diagnostics(
        dense_results: tuple[RetrievalResult, ...],
        bm25_results: tuple[RetrievalResult, ...],
        fused_results: tuple[RetrievalResult, ...],
    ) -> tuple[HybridRetrievalDiagnostic, ...]:
        dense_ranks: dict[str, int] = {}
        for rank, result in enumerate(dense_results, start=1):
            dense_ranks.setdefault(result.chunk.chunk_id, rank)
        bm25_ranks: dict[str, int] = {}
        for rank, result in enumerate(bm25_results, start=1):
            bm25_ranks.setdefault(result.chunk.chunk_id, rank)
        diagnostics = tuple(
            HybridRetrievalDiagnostic(
                chunk_id=result.chunk.chunk_id,
                source_path=result.chunk.source_path,
                dense_rank=dense_ranks.get(result.chunk.chunk_id),
                bm25_rank=bm25_ranks.get(result.chunk.chunk_id),
                fused_rank=rank,
            )
            for rank, result in enumerate(fused_results, start=1)
        )
        for item in diagnostics:
            LOGGER.info(
                "Hybrid retrieval diagnostic dense_rank=%s bm25_rank=%s fused_rank=%d "
                "source=%s chunk_id=%s",
                item.dense_rank,
                item.bm25_rank,
                item.fused_rank,
                item.source_path,
                item.chunk_id,
            )
        return diagnostics
