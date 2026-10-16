"""Discover and extract text from PDF and DOCX source files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from docx import Document as open_docx
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PageObject, PdfReader
from pypdf.errors import PdfReadError

from hybrid_rag.config import Settings
from hybrid_rag.schemas import DocumentRecord, DocumentSegment, FileType


LOGGER = logging.getLogger(__name__)
SUPPORTED_TYPES: dict[str, FileType] = {".pdf": "pdf", ".docx": "docx"}
TEMPORARY_SUFFIXES = {".tmp", ".temp", ".part", ".crdownload", ".download", ".lock"}


class DocumentLoadError(ValueError):
    """Raised when a source file cannot be safely loaded."""


@dataclass(frozen=True)
class DocumentLoadFailure:
    source_path: str
    file_type: FileType | None
    file_hash: str | None
    error: str


@dataclass(frozen=True)
class DocumentLoadBatch:
    documents: tuple[DocumentRecord, ...]
    failures: tuple[DocumentLoadFailure, ...]


def _is_temporary(name: str) -> bool:
    lowered = name.lower()
    return (
        name.startswith(("~$", "._"))
        or Path(lowered).suffix in TEMPORARY_SUFFIXES
        or lowered in {"thumbs.db", ".ds_store"}
        or lowered.endswith(("~", ".swp", ".swo"))
    )


def discover_documents(documents_dir: Path) -> tuple[Path, ...]:
    root = Path(documents_dir).expanduser().resolve()
    if not root.is_dir():
        raise DocumentLoadError(f"Document source directory does not exist: {root}")

    def fail_on_walk_error(error: OSError) -> None:
        raise DocumentLoadError(
            f"Could not scan document directory ({type(error).__name__})."
        ) from error

    discovered: list[Path] = []
    for current_dir, directories, file_names in os.walk(
        root,
        followlinks=False,
        onerror=fail_on_walk_error,
    ):
        directories.sort()
        kept_directories: list[str] = []
        for directory in directories:
            relative_path = (Path(current_dir) / directory).relative_to(root).as_posix()
            if _is_temporary(directory):
                LOGGER.info("Ignoring temporary directory under documents: %s", relative_path)
            else:
                kept_directories.append(directory)
        directories[:] = kept_directories

        for file_name in sorted(file_names, key=lambda name: (name.casefold(), name)):
            path = Path(current_dir) / file_name
            relative_path = path.relative_to(root).as_posix()
            if file_name == ".gitkeep":
                continue
            if _is_temporary(file_name):
                LOGGER.info("Ignoring temporary document file: %s", relative_path)
                continue
            if path.suffix.lower() not in SUPPORTED_TYPES:
                LOGGER.info("Ignoring unsupported file in documents directory: %s", relative_path)
                continue
            discovered.append(path)

    return tuple(
        sorted(
            discovered,
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )
    )


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DocumentLoadError(
            f"Cannot read source file to compute SHA-256 ({type(error).__name__})."
        ) from error
    return digest.hexdigest()


def _page_has_images(page: PageObject) -> bool:
    resources = page.get("/Resources")
    if resources is None:
        return False
    resources = resources.get_object()
    x_objects = resources.get("/XObject")
    if x_objects is None:
        return False
    x_objects = x_objects.get_object()
    return any(
        item.get_object().get("/Subtype") == "/Image"
        for item in x_objects.values()
    )


def _read_pdf(path: Path) -> tuple[DocumentSegment, ...]:
    try:
        with path.open("rb") as pdf_file:
            reader = PdfReader(pdf_file)
            if reader.is_encrypted:
                raise DocumentLoadError(
                    "PDF is password-protected; password handling is not implemented."
                )

            pages = reader.pages
            segments: list[DocumentSegment] = []
            for page_number, page in enumerate(pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    segments.append(DocumentSegment(text=text, page_number=page_number))

            if not segments:
                scanned = any(_page_has_images(page) for page in pages)
                if scanned:
                    raise DocumentLoadError(
                        "PDF contains image-only pages and no extractable text; it appears scanned. "
                        "OCR was not performed."
                    )
                raise DocumentLoadError(
                    "PDF has no extractable text; it may be empty or scanned. OCR was not performed."
                )
            return tuple(segments)
    except DocumentLoadError:
        raise
    except PdfReadError as error:
        raise DocumentLoadError(
            "Could not parse PDF; it may be corrupt, unreadable, or password-protected."
        ) from error
    except (OSError, ValueError, EOFError) as error:
        raise DocumentLoadError("Could not read PDF; it may be corrupt or unreadable.") from error
    except Exception as error:
        # Parser implementations expose additional format-specific exceptions.
        raise DocumentLoadError(
            f"PDF parser failed while reading this file ({type(error).__name__})."
        ) from error


def _read_docx(path: Path) -> tuple[DocumentSegment, ...]:
    try:
        document: DocxDocument = open_docx(str(path))
        segments: list[DocumentSegment] = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                if block.text.strip():
                    segments.append(DocumentSegment(text=block.text))
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [cell.text for cell in row.cells]
                    if any(cell.strip() for cell in cells):
                        segments.append(DocumentSegment(text=" | ".join(cells)))

        if not segments:
            raise DocumentLoadError("DOCX contains no extractable paragraph or table text.")
        return tuple(segments)
    except DocumentLoadError:
        raise
    except Exception as error:
        # Parser implementations expose additional format-specific exceptions.
        raise DocumentLoadError(
            "Could not parse DOCX; it may be corrupt, unreadable, or password-protected."
        ) from error


def _resolve_source(path: Path, root: Path) -> tuple[Path, Path]:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        source = candidate.resolve(strict=True)
    except OSError as error:
        raise DocumentLoadError("Source file is missing or unreadable.") from error
    try:
        relative_path = source.relative_to(root)
    except ValueError as error:
        raise DocumentLoadError("Source file must be located under the documents directory.") from error
    if not source.is_file():
        raise DocumentLoadError("Source path is not a regular file.")
    return source, relative_path


def _load_document(path: Path, root: Path, file_hash: str) -> DocumentRecord:
    source, relative_path = _resolve_source(path, root)
    file_type = SUPPORTED_TYPES.get(source.suffix.lower())
    if file_type is None:
        raise DocumentLoadError("Unsupported document type; only PDF and DOCX are supported.")

    segments = _read_pdf(source) if file_type == "pdf" else _read_docx(source)
    relative_source = relative_path.as_posix()
    document_id = hashlib.sha256(relative_source.encode("utf-8")).hexdigest()
    return DocumentRecord(
        document_id=document_id,
        source_path=relative_source,
        file_name=source.name,
        file_type=file_type,
        file_hash=file_hash,
        segments=segments,
    )


def load_document(
    path: Path,
    documents_dir: Path,
    *,
    file_hash: str | None = None,
) -> DocumentRecord:
    """Load one PDF or DOCX file and retain its source metadata."""
    root = Path(documents_dir).expanduser().resolve()
    if not root.is_dir():
        raise DocumentLoadError(f"Document source directory does not exist: {root}")
    source, _ = _resolve_source(Path(path), root)
    computed_hash = compute_file_hash(source) if file_hash is None else file_hash
    return _load_document(source, root, computed_hash)


def _manifest_entry_for_document(
    document: DocumentRecord,
    chunk_count: int | None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "source_path": document.source_path,
        "file_name": document.file_name,
        "file_type": document.file_type,
        "file_hash": document.file_hash,
        "status": "loaded",
        "segment_count": len(document.segments),
    }
    if chunk_count is not None:
        entry["chunk_count"] = chunk_count
    return entry


def write_load_manifest(
    manifest_path: Path,
    documents: tuple[DocumentRecord, ...],
    failures: tuple[DocumentLoadFailure, ...],
    *,
    chunk_counts: Mapping[str, int] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunks_path: Path | None = None,
) -> None:
    if chunk_counts is not None and (chunk_size is None or chunk_overlap is None):
        raise ValueError("Chunk manifest entries require both chunk_size and chunk_overlap.")

    entries: list[dict[str, object]] = [
        _manifest_entry_for_document(
            item,
            chunk_counts.get(item.document_id) if chunk_counts is not None else None,
        )
        for item in documents
    ]
    entries.extend(
        {
            "source_path": failure.source_path,
            "file_type": failure.file_type,
            "file_hash": failure.file_hash,
            "status": "failed",
            "error": failure.error,
        }
        for failure in failures
    )
    entries.sort(key=lambda item: str(item["source_path"]).casefold())
    manifest = {
        "schema_version": 2 if chunk_counts is not None else 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    if chunk_counts is not None:
        manifest["chunking"] = {
            "unit": "characters",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        manifest["chunks"] = {
            "path": str(chunks_path) if chunks_path is not None else None,
            "count": sum(chunk_counts.values()),
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(manifest, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, manifest_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_documents(
    documents_dir: Path | None = None,
    *,
    manifest_path: Path | None = None,
    write_manifest: bool = True,
    settings: Settings | None = None,
) -> DocumentLoadBatch:
    """Load supported files independently and persist per-file outcomes."""
    settings = settings or Settings.from_env()
    root = Path(documents_dir) if documents_dir is not None else settings.documents_dir
    root = root.expanduser().resolve()
    paths = discover_documents(root)
    destination = Path(manifest_path) if manifest_path is not None else settings.manifest_path
    if not destination.is_absolute():
        destination = settings.project_root / destination
    destination = destination.expanduser().resolve()

    documents: list[DocumentRecord] = []
    failures: list[DocumentLoadFailure] = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        file_type = SUPPORTED_TYPES[path.suffix.lower()]
        file_hash: str | None = None
        try:
            source, _ = _resolve_source(path, root)
            file_hash = compute_file_hash(source)
            documents.append(_load_document(source, root, file_hash))
        except DocumentLoadError as error:
            failure = DocumentLoadFailure(relative_path, file_type, file_hash, str(error))
            failures.append(failure)
            LOGGER.error("Document load failed for %s: %s", relative_path, failure.error)
        except OSError as error:
            failure = DocumentLoadFailure(
                relative_path,
                file_type,
                file_hash,
                f"Source file could not be read ({type(error).__name__}).",
            )
            failures.append(failure)
            LOGGER.error("Document load failed for %s: %s", relative_path, failure.error)

    batch = DocumentLoadBatch(tuple(documents), tuple(failures))
    if write_manifest:
        write_load_manifest(destination, batch.documents, batch.failures)
    return batch
