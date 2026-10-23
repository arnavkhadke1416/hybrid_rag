"""Local web interface for the document-grounded RAG workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from hybrid_rag.config import ConfigurationError, Settings
from hybrid_rag.generation.gemini import GeminiConfigurationError, GeminiGenerationError
from hybrid_rag.indexing.embeddings import EmbeddingError
from hybrid_rag.ingestion.loaders import discover_documents
from hybrid_rag.ingestion.pipeline import ingest_documents
from hybrid_rag.retrieval.hybrid import HybridIndexError, HybridQueryError
from hybrid_rag.schemas import Chunk
from hybrid_rag.services.rag_service import RAGService


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC_DIR = Path(__file__).with_name("static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    diagnostics: bool = True


def _chunk_count(chunks_path: Path) -> int:
    if not chunks_path.is_file():
        return 0
    try:
        with chunks_path.open(encoding="utf-8") as chunks_file:
            return sum(bool(line.strip()) for line in chunks_file)
    except OSError:
        return 0


def _manifest_entries(settings: Settings) -> dict[str, dict[str, Any]]:
    try:
        manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = manifest.get("files", []) if isinstance(manifest, dict) else []
    return {
        entry["source_path"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("source_path"), str)
    }


def _workspace_status(settings: Settings) -> dict[str, Any]:
    entries = _manifest_entries(settings)
    documents = []
    for path in discover_documents(settings.documents_dir):
        source_path = path.relative_to(settings.documents_dir).as_posix()
        entry = entries.get(source_path, {})
        documents.append(
            {
                "name": path.name,
                "source_path": source_path,
                "file_type": path.suffix.lstrip(".").lower(),
                "status": entry.get("status", "pending"),
                "chunk_count": entry.get("chunk_count", 0),
            }
        )

    chunks_path = settings.processed_dir / "chunks.jsonl"
    chunk_count = _chunk_count(chunks_path)
    indexes_exist = all(
        path.is_file()
        for path in (
            settings.faiss_index_dir / "index.faiss",
            settings.faiss_index_dir / "mapping.json",
            settings.bm25_index_dir / "index.json",
        )
    )
    index_ready = chunks_path.is_file() and indexes_exist and chunk_count > 0
    return {
        "ready": index_ready and settings.gemini_api_key is not None,
        "index_ready": index_ready,
        "gemini_configured": settings.gemini_api_key is not None,
        "model": settings.gemini_model,
        "embedding_model": settings.embedding_model,
        "document_count": len(documents),
        "loaded_count": sum(document["status"] == "loaded" for document in documents),
        "chunk_count": chunk_count,
        "documents": documents,
    }


def _safe_upload_name(filename: str | None) -> str:
    name = (filename or "").replace("\\", "/").rsplit("/", maxsplit=1)[-1].strip()
    if not name or name in {".", ".."} or Path(name).suffix.lower() not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Upload a PDF or DOCX document.")
    return name


def _available_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    stem, suffix = destination.stem, destination.suffix
    version = 2
    while (directory / f"{stem} ({version}){suffix}").exists():
        version += 1
    return directory / f"{stem} ({version}){suffix}"


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    app = FastAPI(title="Hybrid RAG", docs_url=None, redoc_url=None)
    app.state.settings = configured
    app.state.rag_service = None
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status(request: Request) -> dict[str, Any]:
        try:
            return _workspace_status(request.app.state.settings)
        except ConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/ask")
    def ask(payload: AskRequest, request: Request) -> dict[str, Any]:
        question = " ".join(payload.question.split())
        if not question:
            raise HTTPException(status_code=400, detail="Enter a question to continue.")
        try:
            service = request.app.state.rag_service
            if service is None:
                service = RAGService(request.app.state.settings)
                request.app.state.rag_service = service
            response = service.ask(question, diagnostic=payload.diagnostics)
        except (ConfigurationError, EmbeddingError, GeminiConfigurationError,
                GeminiGenerationError, HybridIndexError, HybridQueryError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        results = []
        diagnostic_by_chunk = {
            item.chunk_id: item for item in response.retrieval.diagnostics
        }
        for rank, result in enumerate(response.retrieval.results, start=1):
            chunk: Chunk = result.chunk
            diagnostic = diagnostic_by_chunk.get(chunk.chunk_id)
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "page_range": chunk.page_range,
                    "section_heading": chunk.section_heading,
                    "text": chunk.text,
                    "score": result.score,
                    "dense_score": result.dense_score,
                    "bm25_score": result.bm25_score,
                    "dense_rank": diagnostic.dense_rank if diagnostic else None,
                    "bm25_rank": diagnostic.bm25_rank if diagnostic else None,
                }
            )
        return {
            "question": question,
            "answer": response.answer.answer,
            "sources": [
                {
                    "citation_label": source.citation_label,
                    "file_name": source.file_name,
                    "source_path": source.source_path,
                    "page_number": source.page_number,
                    "page_range": source.page_range,
                    "section_heading": source.section_heading,
                }
                for source in response.answer.sources
            ],
            "results": results,
        }

    @app.post("/api/ingest")
    async def ingest(
        request: Request,
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="Choose at least one PDF or DOCX file.")
        settings: Settings = request.app.state.settings
        prepared: list[tuple[UploadFile, str]] = []
        for upload in files:
            filename = _safe_upload_name(upload.filename)
            prepared.append((upload, filename))

        saved_paths: list[Path] = []
        try:
            settings.documents_dir.mkdir(parents=True, exist_ok=True)
            for upload, filename in prepared:
                contents = await upload.read(MAX_UPLOAD_BYTES + 1)
                if len(contents) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{filename} exceeds the 20 MB upload limit.",
                    )
                destination = _available_destination(settings.documents_dir, filename)
                await run_in_threadpool(destination.write_bytes, contents)
                saved_paths.append(destination)

            summary = await run_in_threadpool(ingest_documents, settings)
        except HTTPException:
            for path in saved_paths:
                path.unlink(missing_ok=True)
            raise
        except (ConfigurationError, OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {error}") from error
        finally:
            for upload in files:
                await upload.close()

        request.app.state.rag_service = None
        return {
            "added": list(summary.added),
            "changed": list(summary.changed),
            "removed": list(summary.removed),
            "unchanged": list(summary.unchanged),
            "failed": [
                {"source_path": failure.source_path, "message": failure.message}
                for failure in summary.failures
            ],
            "document_count": summary.total_documents,
            "chunk_count": len(summary.chunks),
            "rebuilt": summary.rebuilt,
            "embedded_chunks": summary.embedded_chunks,
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("hybrid_rag.web:app", host="127.0.0.1", port=8000)