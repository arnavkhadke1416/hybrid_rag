"""Character-based chunking and JSONL persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from hybrid_rag.config import Settings
from hybrid_rag.ingestion.cleaning import clean_text
from hybrid_rag.schemas import Chunk, DocumentRecord, DocumentSegment


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SegmentSpan:
    start: int
    end: int
    segment: DocumentSegment


def _combine_segments(
    document: DocumentRecord,
) -> tuple[str, tuple[_SegmentSpan, ...], int]:
    parts: list[str] = []
    spans: list[_SegmentSpan] = []
    cursor = 0
    empty_segments = 0

    for segment in document.segments:
        text = clean_text(segment.text)
        if not text:
            empty_segments += 1
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(_SegmentSpan(start, cursor, segment))

    return "".join(parts), tuple(spans), empty_segments


def _chunk_end(text: str, start: int, size: int, overlap: int) -> int:
    limit = min(start + size, len(text))
    if limit == len(text):
        return limit

    minimum_end = start + max(size // 2, overlap + 1)
    paragraph_boundary = text.rfind("\n\n", minimum_end, limit)
    if paragraph_boundary >= minimum_end:
        return min(paragraph_boundary + 2, limit)

    whitespace_boundary = max(
        text.rfind(" ", minimum_end, limit),
        text.rfind("\n", minimum_end, limit),
    )
    if whitespace_boundary >= minimum_end:
        return whitespace_boundary + 1
    return limit


def chunk_document(
    document: DocumentRecord,
    settings: Settings | None = None,
) -> tuple[Chunk, ...]:
    settings = settings or Settings.from_env()
    text, spans, empty_segments = _combine_segments(document)
    if empty_segments:
        LOGGER.info(
            "Filtered %d empty source segment(s) from %s",
            empty_segments,
            document.source_path,
        )

    chunks: list[Chunk] = []
    removed_empty_chunks = 0
    start = 0
    while start < len(text):
        end = _chunk_end(text, start, settings.chunk_size, settings.chunk_overlap)
        content_start = start
        content_end = end
        while content_start < content_end and text[content_start].isspace():
            content_start += 1
        while content_end > content_start and text[content_end - 1].isspace():
            content_end -= 1

        if content_start >= content_end:
            removed_empty_chunks += 1
        else:
            chunk_text = text[content_start:content_end]
            page_numbers = sorted(
                {
                    span.segment.page_number
                    for span in spans
                    if span.segment.page_number is not None
                    and span.start < content_end
                    and span.end > content_start
                }
            )
            headings = {
                span.segment.section_heading
                for span in spans
                if span.segment.section_heading
                and span.start < content_end
                and span.end > content_start
            }
            chunk_index = len(chunks)
            chunk_id = hashlib.sha256(
                (
                    f"{document.document_id}\0{document.file_hash}\0"
                    f"{chunk_index}\0{chunk_text}"
                ).encode("utf-8")
            ).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source_path=document.source_path,
                    file_name=document.file_name,
                    file_type=document.file_type,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    page_number=page_numbers[0] if len(page_numbers) == 1 else None,
                    char_start=content_start,
                    char_end=content_end,
                    section_heading=next(iter(headings)) if len(headings) == 1 else None,
                    page_range=(page_numbers[0], page_numbers[-1])
                    if len(page_numbers) > 1
                    else None,
                )
            )

        if end >= len(text):
            break
        next_start = max(start + 1, end - settings.chunk_overlap)
        start = next_start

    if removed_empty_chunks:
        LOGGER.info(
            "Filtered %d empty chunk(s) from %s",
            removed_empty_chunks,
            document.source_path,
        )
    return tuple(chunks)


def save_chunks(chunks: Sequence[Chunk], path: Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for chunk in chunks:
                json.dump(asdict(chunk), temporary_file, ensure_ascii=False, separators=(",", ":"))
                temporary_file.write("\n")
        os.replace(temporary_path, destination)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_chunks(path: Path) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid chunk JSON on line {line_number}.") from error
            if not isinstance(value, dict):
                raise ValueError(f"Chunk JSON on line {line_number} must be an object.")
            page_range = value.get("page_range")
            if page_range is not None:
                if (
                    not isinstance(page_range, list)
                    or len(page_range) != 2
                    or not all(isinstance(page, int) for page in page_range)
                ):
                    raise ValueError(f"Invalid page_range in chunk JSON on line {line_number}.")
                value["page_range"] = tuple(page_range)
            try:
                chunks.append(Chunk(**value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid chunk record on line {line_number}.") from error
    return tuple(chunks)
