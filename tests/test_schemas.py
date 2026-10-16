from dataclasses import replace
from collections.abc import Callable
from typing import Literal, cast

import pytest

from hybrid_rag.schemas import Chunk, DocumentRecord, DocumentSegment, RetrievalResult


def make_chunk(
    *,
    chunk_index: int = 0,
    text: str = "Relevant evidence.",
    page_number: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    section_heading: str | None = None,
    page_range: tuple[int, int] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id="chunk-1",
        document_id="document-1",
        source_path="guide/sample.pdf",
        file_name="sample.pdf",
        file_type="pdf",
        chunk_index=chunk_index,
        text=text,
        page_number=page_number,
        char_start=char_start,
        char_end=char_end,
        section_heading=section_heading,
        page_range=page_range,
    )


def test_chunk_preserves_source_and_optional_location_metadata() -> None:
    chunk = make_chunk(page_number=2, char_start=10, char_end=28, section_heading="Overview")

    assert chunk.page_number == 2
    assert chunk.char_start == 10
    assert chunk.char_end == 28
    assert chunk.section_heading == "Overview"


def test_document_records_and_retrieval_results_are_typed() -> None:
    document = DocumentRecord(
        document_id="document-1",
        source_path="guide/sample.docx",
        file_name="sample.docx",
        file_type="docx",
        file_hash="sha256-value",
        segments=(DocumentSegment("Paragraph"),),
    )
    result = RetrievalResult(chunk=make_chunk(), score=0.5, fused_score=0.02)

    assert document.segments[0].text == "Paragraph"
    assert result.chunk.chunk_id == "chunk-1"
    assert result.fused_score == 0.02


@pytest.mark.parametrize(
    "create_chunk",
    [
        lambda: make_chunk(chunk_index=-1),
        lambda: make_chunk(text="   "),
        lambda: make_chunk(page_number=0),
        lambda: make_chunk(char_start=10, char_end=9),
        lambda: make_chunk(page_range=(3, 2)),
    ],
)
def test_invalid_chunk_metadata_is_rejected(create_chunk: Callable[[], Chunk]) -> None:
    with pytest.raises(ValueError):
        create_chunk()


def test_source_paths_cannot_escape_documents() -> None:
    with pytest.raises(ValueError, match="relative path under documents"):
        DocumentRecord(
            document_id="document-1",
            source_path="../outside.pdf",
            file_name="outside.pdf",
            file_type="pdf",
            file_hash="sha256-value",
        )


def test_unsupported_document_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="file_type must be"):
        replace(make_chunk(), file_type=cast(Literal["pdf", "docx"], "txt"))


def test_non_finite_retrieval_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        RetrievalResult(chunk=make_chunk(), score=float("inf"))
