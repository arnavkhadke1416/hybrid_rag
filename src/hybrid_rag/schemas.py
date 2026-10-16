"""Typed records shared by ingestion, indexing, retrieval, and generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal


FileType = Literal["pdf", "docx"]


def _validate_source_path(source_path: str) -> None:
    path = PurePath(source_path)
    if not source_path.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError("source_path must be a relative path under documents/.")


@dataclass(frozen=True)
class DocumentSegment:
    text: str
    page_number: int | None = None
    section_heading: str | None = None

    def __post_init__(self) -> None:
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be a positive integer.")


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source_path: str
    file_name: str
    file_type: FileType
    file_hash: str
    segments: tuple[DocumentSegment, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty.")
        if not self.file_name.strip():
            raise ValueError("file_name must not be empty.")
        if not self.file_hash.strip():
            raise ValueError("file_hash must not be empty.")
        if self.file_type not in {"pdf", "docx"}:
            raise ValueError("file_type must be 'pdf' or 'docx'.")
        _validate_source_path(self.source_path)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_path: str
    file_name: str
    file_type: FileType
    chunk_index: int
    text: str
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_heading: str | None = None
    page_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for name in ("chunk_id", "document_id", "source_path", "file_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty.")
        if self.file_type not in {"pdf", "docx"}:
            raise ValueError("file_type must be 'pdf' or 'docx'.")
        _validate_source_path(self.source_path)
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative.")
        if not self.text.strip():
            raise ValueError("chunk text must not be empty.")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be a positive integer.")
        if self.char_start is not None and self.char_start < 0:
            raise ValueError("char_start must be non-negative.")
        if self.char_end is not None and self.char_end < 0:
            raise ValueError("char_end must be non-negative.")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must not be less than char_start.")
        if self.page_range is not None:
            first_page, last_page = self.page_range
            if first_page < 1 or last_page < first_page:
                raise ValueError("page_range must contain valid, ascending page numbers.")


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None

    def __post_init__(self) -> None:
        for name in ("score", "dense_score", "bm25_score", "fused_score"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
