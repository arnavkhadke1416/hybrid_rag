from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from docx import Document

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL, Settings
from hybrid_rag.generation.gemini import GeminiGenerator
from hybrid_rag.ingestion.pipeline import ingest_documents
from hybrid_rag.retrieval.hybrid import HybridRetriever
from hybrid_rag.services.rag_service import RAGService


class FakeEmbedder:
    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = 2

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0] if "aes-256" in text.lower() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


class FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text="Backups use AES-256 [S1].")


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()


def test_service_ingests_indexes_retrieves_and_generates_with_mocked_gemini(
    tmp_path: Path,
) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    document = Document()
    document.add_paragraph("Backups are protected using AES-256 encryption.")
    document.save(documents_dir / "guide.docx")
    settings = Settings(
        project_root=tmp_path,
        gemini_model="gemini-3.7-flash",
        gemini_api_key=None,
    )

    embedder = FakeEmbedder()
    ingestion = ingest_documents(settings, embedder=embedder)
    assert ingestion.total_documents == 1
    assert len(ingestion.chunks) == 1

    client = FakeClient()
    service = RAGService(
        settings,
        retriever=HybridRetriever(settings, embedder=embedder),
        generator=GeminiGenerator(settings, client=client),
    )
    response = service.ask("How are backups protected with AES-256?", diagnostic=True)

    assert response.answer.answer == "Backups use AES-256 [S1]."
    assert [source.source_path for source in response.answer.sources] == ["guide.docx"]
    assert response.retrieval.results[0].chunk.source_path == "guide.docx"
    assert response.retrieval.diagnostics[0].fused_rank == 1
    assert client.models.calls[0]["model"] == "gemini-3.7-flash"
