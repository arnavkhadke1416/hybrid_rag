import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
from docx import Document

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL, Settings
from hybrid_rag.ingestion.chunking import chunk_document, load_chunks, save_chunks
from hybrid_rag.ingestion.cleaning import clean_text
from hybrid_rag.ingestion.pipeline import ingest_documents
from hybrid_rag.schemas import DocumentRecord, DocumentSegment, FileType


class FakeEmbedder:
    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = 2

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def make_settings(
    tmp_path: Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Settings:
    (tmp_path / "documents").mkdir(exist_ok=True)
    return Settings.from_env(
        environ={
            "CHUNK_SIZE": str(chunk_size),
            "CHUNK_OVERLAP": str(chunk_overlap),
        },
        project_root=tmp_path,
    )


def make_document(
    segments: tuple[DocumentSegment, ...],
    *,
    document_id: str = "document-id",
    source_path: str = "guide.pdf",
    file_type: FileType = "pdf",
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        source_path=source_path,
        file_name=Path(source_path).name,
        file_type=file_type,
        file_hash="a" * 64,
        segments=segments,
    )


def test_cleaning_preserves_identifiers_headings_punctuation_and_tables() -> None:
    source = "##  Policy\r\n\r\n AES-256\t  key  \r\nVersion  2.1: Keep 4.2.\r\nTable  A |  B\n\n\nEnd"

    assert clean_text(source) == (
        "## Policy\n\nAES-256 key\nVersion 2.1: Keep 4.2.\nTable A | B\n\nEnd"
    )


def test_chunk_size_paragraph_boundaries_and_overlap_are_respected(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, chunk_size=100, chunk_overlap=10)
    first_paragraph = "First paragraph has enough text to be its own complete section."
    second_paragraph = "Second paragraph also has enough text to continue in another section."
    document = make_document(
        (DocumentSegment(f"{first_paragraph}\n\n{second_paragraph}"),)
    )

    chunks = chunk_document(document, settings)

    assert len(chunks) >= 2
    assert all(0 < len(chunk.text) <= settings.chunk_size for chunk in chunks)
    assert chunks[0].text == first_paragraph
    assert chunks[0].char_end is not None
    assert chunks[1].char_start is not None
    assert chunks[1].char_start < chunks[0].char_end
    assert chunks[0].chunk_index == 0


def test_long_paragraph_falls_back_to_bounded_character_chunks(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, chunk_size=80, chunk_overlap=15)
    paragraph = "AES-256 remains an exact identifier in this deliberately long paragraph. " * 8
    document = make_document((DocumentSegment(paragraph),))

    chunks = chunk_document(document, settings)

    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 80 for chunk in chunks)
    assert all(chunks[index].char_start < chunks[index - 1].char_end for index in range(1, len(chunks)))
    for match in re.finditer("AES-256", clean_text(paragraph)):
        assert any(
            chunk.char_start is not None
            and chunk.char_end is not None
            and chunk.char_start <= match.start()
            and chunk.char_end >= match.end()
            for chunk in chunks
        )


def test_cross_page_chunks_keep_an_explicit_page_range(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, chunk_size=100, chunk_overlap=10)
    first_page = "Page one contains enough words to extend past the chunk boundary."
    second_page = "Page two continues with distinct source material and page attribution."
    document = make_document(
        (
            DocumentSegment(first_page, page_number=1),
            DocumentSegment(second_page, page_number=2),
        )
    )

    chunks = chunk_document(document, settings)

    cross_page = next(chunk for chunk in chunks if chunk.page_range is not None)
    assert cross_page.page_number is None
    assert cross_page.page_range == (1, 2)


def test_chunk_ids_are_stable_and_distinct_for_positions_and_sources(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, chunk_size=50, chunk_overlap=10)
    text = "Repeated evidence remains useful. " * 8
    first = make_document((DocumentSegment(text),))
    second = make_document(
        (DocumentSegment(text),),
        document_id="another-document",
        source_path="another.pdf",
    )

    first_chunks = chunk_document(first, settings)
    repeated_chunks = chunk_document(first, settings)
    second_chunks = chunk_document(second, settings)

    assert [chunk.chunk_id for chunk in first_chunks] == [
        chunk.chunk_id for chunk in repeated_chunks
    ]
    assert len({chunk.chunk_id for chunk in first_chunks}) == len(first_chunks)
    assert set(chunk.chunk_id for chunk in first_chunks).isdisjoint(
        chunk.chunk_id for chunk in second_chunks
    )


def test_empty_segments_are_counted_in_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = make_settings(tmp_path)
    document = make_document((DocumentSegment("  \r\n\t"),))

    with caplog.at_level(logging.INFO, logger="hybrid_rag.ingestion.chunking"):
        chunks = chunk_document(document, settings)

    assert chunks == ()
    assert "Filtered 1 empty source segment(s)" in caplog.text


def test_jsonl_round_trips_unicode_newlines_and_page_ranges(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    document = make_document(
        (
            DocumentSegment("Café keeps UTF-8 intact.\nSecond line.", page_number=1),
            DocumentSegment("信息 continues here.", page_number=2),
        )
    )
    chunks = chunk_document(document, settings)
    destination = save_chunks(chunks, tmp_path / "processed" / "chunks.jsonl")

    serialized = destination.read_text(encoding="utf-8")

    assert "Café" in serialized
    assert "信息" in serialized
    assert len(serialized.splitlines()) == len(chunks)
    assert load_chunks(destination) == chunks


def test_ingestion_pipeline_saves_chunks_and_versioned_manifest(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, chunk_size=60, chunk_overlap=10)
    document_path = settings.documents_dir / "memo.docx"
    docx = Document()
    docx.add_paragraph("A useful paragraph with AES-256 and version 2.4.")
    docx.save(document_path)

    result = ingest_documents(settings, embedder=FakeEmbedder())

    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    entries = {entry["source_path"]: entry for entry in manifest["files"]}
    assert result.total_documents == 1
    assert result.failures == ()
    assert result.chunks
    assert result.chunks_path == settings.processed_dir / "chunks.jsonl"
    assert load_chunks(result.chunks_path) == result.chunks
    assert manifest["schema_version"] == 3
    assert manifest["chunking"] == {
        "unit": "characters",
        "chunk_size": 60,
        "chunk_overlap": 10,
    }
    assert entries["memo.docx"]["chunk_count"] == len(result.chunks)
