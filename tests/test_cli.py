from __future__ import annotations

from hybrid_rag.cli import main
from hybrid_rag.generation.gemini import CitedSource, GroundedAnswer
from hybrid_rag.retrieval.hybrid import (
    HybridIndexError,
    HybridRetrievalDiagnostic,
    HybridRetrievalResponse,
)
from hybrid_rag.schemas import Chunk, RetrievalResult
from hybrid_rag.services.rag_service import RAGServiceResponse


def make_response(answer: str = "AES-256 protects backups [S1].") -> RAGServiceResponse:
    chunk = Chunk(
        "chunk-a",
        "doc",
        "guide.docx",
        "guide.docx",
        "docx",
        0,
        "Backups use AES-256.",
        section_heading="Backups",
    )
    return RAGServiceResponse(
        answer=GroundedAnswer(
            answer,
            (
                CitedSource(
                    citation_label="[S1]",
                    chunk_id=chunk.chunk_id,
                    file_name=chunk.file_name,
                    source_path=chunk.source_path,
                    page_number=None,
                    page_range=None,
                    section_heading=chunk.section_heading,
                ),
            ),
        ),
        retrieval=HybridRetrievalResponse(
            results=(RetrievalResult(chunk=chunk, score=0.03),),
            diagnostics=(
                HybridRetrievalDiagnostic(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    dense_rank=1,
                    bm25_rank=2,
                    fused_rank=1,
                ),
            ),
        ),
    )


def test_ask_displays_citations_and_optional_diagnostics(monkeypatch, capsys) -> None:
    class FakeService:
        calls: list[tuple[str, bool]] = []

        def ask(self, question: str, *, diagnostic: bool = False) -> RAGServiceResponse:
            self.calls.append((question, diagnostic))
            return make_response()

    monkeypatch.setattr("hybrid_rag.cli.RAGService", FakeService)

    assert main(["ask", "Which encryption method?", "--diagnostics"]) == 0

    output = capsys.readouterr().out
    assert "AES-256 protects backups [S1]." in output
    assert "[S1] guide.docx | chunk chunk-a | section Backups" in output
    assert "dense=1 bm25=2 fused=1" in output
    assert FakeService.calls == [("Which encryption method?", True)]


def test_interactive_reuses_service_for_multiple_questions(
    monkeypatch,
    capsys,
) -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def ask(self, question: str, *, diagnostic: bool = False) -> RAGServiceResponse:
            self.calls.append(question)
            return make_response(f"Answer for {question} [S1].")

    service = FakeService()
    monkeypatch.setattr("hybrid_rag.cli.RAGService", lambda: service)
    questions = iter(("First question?", "Second question?", "quit"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))

    assert main(["interactive"]) == 0

    output = capsys.readouterr().out
    assert "Answer for First question? [S1]." in output
    assert "Answer for Second question? [S1]." in output
    assert service.calls == ["First question?", "Second question?"]


def test_missing_indexes_print_actionable_command_without_traceback(
    monkeypatch,
    capsys,
) -> None:
    def fail_to_initialize():
        raise HybridIndexError(
            r"Processed chunks are missing; run python scripts\ingest_documents.py."
        )

    monkeypatch.setattr("hybrid_rag.cli.RAGService", fail_to_initialize)

    assert main(["ask", "Question?"]) == 1

    error = capsys.readouterr().err
    assert r"python scripts\ingest_documents.py" in error
    assert "Traceback" not in error
