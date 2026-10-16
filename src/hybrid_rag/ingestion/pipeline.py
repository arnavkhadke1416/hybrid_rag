"""Public end-to-end ingestion entry point."""

from __future__ import annotations

from hybrid_rag.config import Settings
from hybrid_rag.indexing.embeddings import EmbeddingEncoder
from hybrid_rag.ingestion.automatic import IngestionRunSummary, run_ingestion


def ingest_documents(
    settings: Settings | None = None,
    *,
    force: bool = False,
    embedder: EmbeddingEncoder | None = None,
) -> IngestionRunSummary:
    return run_ingestion(settings, force=force, embedder=embedder)
