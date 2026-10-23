from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hybrid_rag.config import Settings
from hybrid_rag.generation.gemini import CitedSource, GroundedAnswer
from hybrid_rag.retrieval.hybrid import HybridRetrievalDiagnostic
from hybrid_rag.schemas import Chunk, RetrievalResult
from hybrid_rag.services.rag_service import RAGServiceResponse
from hybrid_rag.web import create_app


def _settings(root: Path) -> Settings:
    (root / "documents").mkdir()
    return Settings(project_root=root)


def test_status_reports_manifest_files_and_index_readiness(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = settings.documents_dir / "guide.pdf"
    source.write_bytes(b"pdf")
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    settings.faiss_index_dir.mkdir(parents=True)
    (settings.faiss_index_dir / "index.faiss").touch()
    (settings.faiss_index_dir / "mapping.json").touch()
    settings.bm25_index_dir.mkdir(parents=True)
    (settings.bm25_index_dir / "index.json").touch()
    settings.manifest_path.parent.mkdir(parents=True)
    settings.manifest_path.write_text(
        json.dumps({"files": [{"source_path": "guide.pdf", "status": "loaded", "chunk_count": 1}]}),
        encoding="utf-8",
    )

    response = TestClient(create_app(settings)).get("/api/status")

    assert response.status_code == 200
    assert response.json()["index_ready"] is True
    assert response.json()["loaded_count"] == 1
    assert response.json()["documents"][0]["chunk_count"] == 1


def test_ask_returns_citations_and_hybrid_rank_details(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import hybrid_rag.web as web

    settings = _settings(tmp_path)
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_path="guide.pdf",
        file_name="guide.pdf",
        file_type="pdf",
        chunk_index=0,
        text="Backups use AES-256 encryption.",
        page_number=3,
    )
    result = RetrievalResult(chunk=chunk, score=0.02, dense_score=0.8, bm25_score=2.5)
    citation = CitedSource("[S1]", "chunk-1", "guide.pdf", "guide.pdf", 3, None, None)
    retrieval = SimpleNamespace(
        results=(result,),
        diagnostics=(HybridRetrievalDiagnostic("chunk-1", "guide.pdf", 1, 2, 1),),
    )
    service = SimpleNamespace(
        ask=lambda question, diagnostic: RAGServiceResponse(
            GroundedAnswer("Backups use AES-256 [S1].", (citation,)), retrieval
        )
    )
    monkeypatch.setattr(web, "RAGService", lambda configured: service)
    client = TestClient(create_app(settings))

    response = client.post("/api/ask", json={"question": "How are backups encrypted?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Backups use AES-256 [S1]."
    assert response.json()["sources"][0]["citation_label"] == "[S1]"
    assert response.json()["results"][0]["dense_rank"] == 1
    assert response.json()["results"][0]["bm25_rank"] == 2


def test_ingest_upload_sanitizes_path_and_refreshes_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import hybrid_rag.web as web

    settings = _settings(tmp_path)
    summary = SimpleNamespace(
        added=("guide.pdf",),
        changed=(),
        removed=(),
        unchanged=(),
        failures=(),
        total_documents=1,
        chunks=(object(),),
        rebuilt=True,
        embedded_chunks=1,
    )
    monkeypatch.setattr(web, "ingest_documents", lambda configured: summary)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/ingest",
        files={"files": ("../guide.pdf", b"pdf-content", "application/pdf")},
    )

    assert response.status_code == 200
    assert (settings.documents_dir / "guide.pdf").read_bytes() == b"pdf-content"
    assert response.json()["added"] == ["guide.pdf"]
    assert response.json()["chunk_count"] == 1