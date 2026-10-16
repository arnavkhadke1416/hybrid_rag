"""Incremental ingestion with staged, recoverable index publication."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from hybrid_rag.config import Settings
from hybrid_rag.indexing.bm25_store import BM25KeywordStore, BM25StoreError
from hybrid_rag.indexing.build_indexes import build_bm25_index, build_faiss_index
from hybrid_rag.indexing.corpus import chunk_snapshot_fingerprint
from hybrid_rag.indexing.embeddings import EmbeddingEncoder
from hybrid_rag.indexing.faiss_store import FaissStoreError, FaissVectorStore
from hybrid_rag.indexing.version import CHUNK_SCHEMA_VERSION, make_index_version
from hybrid_rag.ingestion.chunking import chunk_document, load_chunks, save_chunks
from hybrid_rag.ingestion.loaders import (
    SUPPORTED_TYPES,
    DocumentLoadError,
    DocumentLoadFailure,
    compute_file_hash,
    discover_documents,
    load_document,
)
from hybrid_rag.schemas import Chunk


LOGGER = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 3
TRANSACTION_SCHEMA_VERSION = 1


class IngestionTransactionError(RuntimeError):
    """Raised when an interrupted or failed build cannot be safely recovered."""


@dataclass(frozen=True)
class IngestionRunSummary:
    chunks: tuple[Chunk, ...]
    failures: tuple[DocumentLoadFailure, ...]
    added: tuple[str, ...]
    changed: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]
    index_version: str
    total_documents: int
    rebuilt: bool
    embedded_chunks: int
    chunks_path: Path
    manifest_path: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transaction_path(settings: Settings) -> Path:
    return settings.manifest_path.with_name(
        f".{settings.manifest_path.name}.transaction.json"
    )


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, ensure_ascii=False, indent=2, allow_nan=False)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _expected_artifacts(settings: Settings) -> tuple[tuple[Path, str], ...]:
    return (
        (settings.faiss_index_dir, "directory"),
        (settings.bm25_index_dir, "directory"),
        (
            settings.processed_dir / "chunks.jsonl",
            "file",
        ),
        (
            settings.manifest_path,
            "file",
        ),
    )


def _validate_journal(
    settings: Settings,
    journal: object,
) -> tuple[str, list[dict[str, object]], list[Path]]:
    if not isinstance(journal, dict) or journal.get("schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise IngestionTransactionError("Ingestion transaction journal has an unsupported schema.")
    state = journal.get("state")
    if state not in {"preparing", "committed"}:
        raise IngestionTransactionError("Ingestion transaction journal has an invalid state.")

    raw_roots = journal.get("stage_roots")
    if not isinstance(raw_roots, list) or len(raw_roots) != 4:
        raise IngestionTransactionError("Ingestion transaction journal has invalid staging paths.")
    expected_roots = [
        (settings.faiss_index_dir.parent, f".{settings.faiss_index_dir.name}.stage-"),
        (settings.bm25_index_dir.parent, f".{settings.bm25_index_dir.name}.stage-"),
        (settings.processed_dir.parent, f".{settings.processed_dir.name}.stage-"),
        (settings.manifest_path.parent, f".{settings.manifest_path.name}.stage-"),
    ]
    stage_roots: list[Path] = []
    for raw_root, (expected_parent, prefix) in zip(raw_roots, expected_roots, strict=True):
        if not isinstance(raw_root, str):
            raise IngestionTransactionError("Ingestion transaction staging path is invalid.")
        root = Path(raw_root).resolve()
        if root.parent != expected_parent or not root.name.startswith(prefix):
            raise IngestionTransactionError("Ingestion transaction staging path is unsafe.")
        stage_roots.append(root)

    raw_artifacts = journal.get("artifacts")
    expected = _expected_artifacts(settings)
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(expected):
        raise IngestionTransactionError("Ingestion transaction journal has invalid artifacts.")
    artifacts: list[dict[str, object]] = []
    seen_destinations: set[Path] = set()
    for raw_artifact, (destination, kind) in zip(
        raw_artifacts,
        expected,
        strict=True,
    ):
        if not isinstance(raw_artifact, dict):
            raise IngestionTransactionError("Ingestion transaction artifact is invalid.")
        expected_destination = destination.resolve()
        raw_destination = raw_artifact.get("destination")
        staged_value = raw_artifact.get("staged")
        backup_value = raw_artifact.get("backup")
        had_previous = raw_artifact.get("had_previous")
        if (
            not isinstance(raw_destination, str)
            or Path(raw_destination).resolve() != expected_destination
            or expected_destination in seen_destinations
            or not isinstance(staged_value, str)
            or not isinstance(backup_value, str)
            or not isinstance(had_previous, bool)
        ):
            raise IngestionTransactionError("Ingestion transaction artifact paths are invalid.")
        seen_destinations.add(expected_destination)

        staged = Path(staged_value).resolve()
        stage_root = next(
            (root for root in stage_roots if staged == root or staged.parent == root),
            None,
        )
        if stage_root is None:
            raise IngestionTransactionError("Ingestion transaction staging path is unsafe.")
        if kind == "directory" and staged != stage_root:
            raise IngestionTransactionError("Ingestion transaction directory stage is invalid.")
        if kind == "file" and (staged.parent != stage_root or staged.name != destination.name):
            raise IngestionTransactionError("Ingestion transaction file stage is invalid.")

        backup = Path(backup_value).resolve()
        if (
            backup.parent != expected_destination.parent
            or not backup.name.startswith(f".{destination.name}.backup-")
        ):
            raise IngestionTransactionError("Ingestion transaction backup path is unsafe.")
        artifacts.append(
            {
                "destination": expected_destination,
                "staged": staged,
                "backup": backup,
                "had_previous": had_previous,
            }
        )
    return state, artifacts, stage_roots


def _rollback(
    artifacts: Sequence[Mapping[str, object]],
) -> None:
    for artifact in reversed(artifacts):
        destination = artifact["destination"]
        backup = artifact["backup"]
        had_previous = artifact["had_previous"]
        if not isinstance(destination, Path) or not isinstance(backup, Path):
            raise IngestionTransactionError("Ingestion rollback metadata is invalid.")
        if not isinstance(had_previous, bool):
            raise IngestionTransactionError("Ingestion rollback metadata is invalid.")
        if _exists(backup):
            _remove_path(destination)
            os.replace(backup, destination)
        elif not had_previous:
            _remove_path(destination)
        elif not _exists(destination):
            raise IngestionTransactionError(
                f"Could not recover the previous ingestion artifact at {destination}."
            )


def _cleanup_transaction(
    artifacts: Sequence[Mapping[str, object]],
    stage_roots: Sequence[Path],
) -> None:
    for artifact in artifacts:
        backup = artifact["backup"]
        if isinstance(backup, Path):
            _remove_path(backup)
    for root in stage_roots:
        _remove_path(root)


def _recover_incomplete_build(settings: Settings) -> None:
    journal_path = _transaction_path(settings)
    if not journal_path.is_file():
        return
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IngestionTransactionError(
            f"Could not read interrupted-ingestion journal at {journal_path}."
        ) from error
    state, artifacts, stage_roots = _validate_journal(settings, journal)
    if state == "preparing":
        _rollback(artifacts)
        LOGGER.warning("Rolled back an interrupted ingestion build.")
    _cleanup_transaction(artifacts, stage_roots)
    journal_path.unlink()


def _create_stage_dir(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{name}.stage-", dir=parent))


def _publish_build(
    settings: Settings,
    artifacts: Sequence[tuple[Path, Path]],
    stage_roots: Sequence[Path],
) -> None:
    transaction_id = uuid.uuid4().hex
    records = [
        {
            "destination": str(destination.resolve()),
            "staged": str(staged.resolve()),
            "backup": str(
                destination.parent
                / f".{destination.name}.backup-{transaction_id}-{position}"
            ),
            "had_previous": _exists(destination),
        }
        for position, (destination, staged) in enumerate(artifacts)
    ]
    journal_path = _transaction_path(settings)
    journal: dict[str, object] = {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "state": "preparing",
        "stage_roots": [str(root.resolve()) for root in stage_roots],
        "artifacts": records,
    }
    _write_json_atomic(journal_path, journal)
    try:
        for record in records:
            destination = Path(str(record["destination"]))
            staged = Path(str(record["staged"]))
            backup = Path(str(record["backup"]))
            if record["had_previous"]:
                os.replace(destination, backup)
            os.replace(staged, destination)
        journal["state"] = "committed"
        _write_json_atomic(journal_path, journal)
    except OSError as error:
        try:
            _rollback(_validate_journal(settings, journal)[1])
        except (OSError, IngestionTransactionError) as rollback_error:
            raise IngestionTransactionError(
                "Index publication failed and automatic rollback was incomplete; "
                f"rerun ingestion to recover from {journal_path}."
            ) from rollback_error
        journal_path.unlink(missing_ok=True)
        _cleanup_transaction(
            _validate_journal(
                settings,
                {**journal, "state": "committed"},
            )[1],
            stage_roots,
        )
        raise IngestionTransactionError(
            "Index publication failed; the previous index set was restored."
        ) from error

    state, published_artifacts, validated_roots = _validate_journal(settings, journal)
    if state != "committed":
        raise IngestionTransactionError("Index publication did not reach the committed state.")
    _cleanup_transaction(published_artifacts, validated_roots)
    journal_path.unlink()


def _load_previous_manifest(path: Path) -> tuple[dict[str, object] | None, dict[str, dict[str, object]]]:
    if not path.is_file():
        return None, {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Ignoring unreadable prior ingestion manifest at %s: %s", path, error)
        return None, {}
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        LOGGER.warning("Ignoring prior ingestion manifest with an unsupported shape at %s.", path)
        return None, {}

    entries: dict[str, dict[str, object]] = {}
    for item in value["files"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("source_path"), str)
            or item["source_path"] in entries
        ):
            LOGGER.warning("Ignoring prior ingestion manifest with invalid file entries at %s.", path)
            return None, {}
        entries[item["source_path"]] = item
    return value, entries


def _load_previous_chunks(path: Path) -> tuple[Chunk, ...] | None:
    if not path.is_file():
        return None
    try:
        return load_chunks(path)
    except (OSError, ValueError) as error:
        LOGGER.warning("Could not reuse prior chunks from %s: %s", path, error)
        return None


def _chunking_definition(settings: Settings) -> dict[str, object]:
    return {
        "unit": "characters",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def _indexing_definition(settings: Settings, index_version: str) -> dict[str, object]:
    from hybrid_rag.indexing.bm25_store import TOKENIZER_NAME

    return {
        "index_version": index_version,
        "embedding_model": settings.embedding_model,
        "tokenizer": TOKENIZER_NAME,
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
    }


def _manifest_payload(
    settings: Settings,
    entries: Sequence[Mapping[str, object]],
    changes: Mapping[str, Sequence[str]],
    chunks: Sequence[Chunk],
    fingerprint: str,
    index_version: str,
) -> dict[str, object]:
    chunks_path = settings.processed_dir / "chunks.jsonl"
    try:
        recorded_chunks_path = chunks_path.relative_to(settings.project_root).as_posix()
    except ValueError:
        recorded_chunks_path = str(chunks_path)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "updated_at": _now(),
        "files": sorted(entries, key=lambda entry: str(entry["source_path"]).casefold()),
        "chunking": _chunking_definition(settings),
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "indexing": _indexing_definition(settings, index_version),
        "chunk_fingerprint": fingerprint,
        "chunks": {
            "path": recorded_chunks_path,
            "count": len(chunks),
        },
        "changes": {
            name: list(paths)
            for name, paths in changes.items()
        },
    }


def _indexes_are_current(
    settings: Settings,
    fingerprint: str,
    index_version: str,
) -> bool:
    try:
        dense = FaissVectorStore.load(
            settings.faiss_index_dir,
            expected_model=settings.embedding_model,
            expected_index_version=index_version,
        )
        sparse = BM25KeywordStore.load(
            settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE,
            expected_corpus_fingerprint=fingerprint,
            expected_index_version=index_version,
        )
    except (BM25StoreError, FaissStoreError):
        return False
    return (
        dense.corpus_fingerprint == fingerprint
        and sparse.corpus_fingerprint == fingerprint
        and dense.index_version == sparse.index_version == index_version
    )


def _validate_target_layout(settings: Settings) -> None:
    targets = [
        settings.faiss_index_dir,
        settings.bm25_index_dir,
        settings.processed_dir / "chunks.jsonl",
        settings.manifest_path,
    ]
    resolved = [target.resolve() for target in targets]
    for index, target in enumerate(resolved):
        if any(
            target == other or target in other.parents or other in target.parents
            for other in resolved[index + 1 :]
        ):
            raise ValueError("Processed data, index, and manifest paths must not overlap.")


def run_ingestion(
    settings: Settings | None = None,
    *,
    force: bool = False,
    embedder: EmbeddingEncoder | None = None,
) -> IngestionRunSummary:
    settings = settings or Settings.from_env()
    _validate_target_layout(settings)
    _recover_incomplete_build(settings)

    manifest, previous_entries = _load_previous_manifest(settings.manifest_path)
    previous_chunks = _load_previous_chunks(settings.processed_dir / "chunks.jsonl")
    previous_chunks_available = previous_chunks is not None
    chunks_by_source: dict[str, list[Chunk]] = {}
    for chunk in previous_chunks or ():
        chunks_by_source.setdefault(chunk.source_path, []).append(chunk)

    previous_chunking = manifest.get("chunking") if manifest is not None else None
    reprocess_all = (
        force
        or manifest is None
        or previous_chunking != _chunking_definition(settings)
        or manifest.get("chunk_schema_version") != CHUNK_SCHEMA_VERSION
        or not previous_chunks_available
    )

    current_paths = discover_documents(settings.documents_dir)
    current_source_paths = {
        path.relative_to(settings.documents_dir).as_posix()
        for path in current_paths
    }
    removed = tuple(sorted(set(previous_entries) - current_source_paths, key=str.casefold))
    added: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    failures: list[DocumentLoadFailure] = []
    entries: list[dict[str, object]] = []
    chunks: list[Chunk] = []
    timestamp = _now()

    for path in current_paths:
        relative_path = path.relative_to(settings.documents_dir).as_posix()
        file_type = SUPPORTED_TYPES[path.suffix.lower()]
        previous = previous_entries.get(relative_path)
        file_hash: str | None = None
        source_modified_at: str | None = None
        try:
            file_hash = compute_file_hash(path)
            source_modified_at = datetime.fromtimestamp(
                path.stat().st_mtime,
                timezone.utc,
            ).isoformat()
        except (DocumentLoadError, OSError) as error:
            message = str(error) if isinstance(error, DocumentLoadError) else (
                f"Source file metadata could not be read ({type(error).__name__})."
            )
            failure = DocumentLoadFailure(relative_path, file_type, file_hash, message)
            failures.append(failure)
            if previous is None:
                added.append(relative_path)
            elif previous.get("file_hash") == file_hash:
                unchanged.append(relative_path)
            else:
                changed.append(relative_path)
            entries.append(
                {
                    "source_path": relative_path,
                    "file_name": path.name,
                    "file_type": file_type,
                    "file_hash": file_hash,
                    "status": "failed",
                    "parse_status": "failed",
                    "error": message,
                    "chunk_count": 0,
                    "source_modified_at": source_modified_at,
                    "processed_at": timestamp,
                    "last_seen_at": timestamp,
                }
            )
            continue

        if previous is None:
            added.append(relative_path)
        elif previous.get("file_hash") != file_hash:
            changed.append(relative_path)
        else:
            unchanged.append(relative_path)

        if (
            not reprocess_all
            and previous is not None
            and previous.get("file_hash") == file_hash
            and previous.get("status") == "loaded"
        ):
            reused = chunks_by_source.get(relative_path, [])
            expected_count = previous.get("chunk_count")
            if isinstance(expected_count, int) and expected_count == len(reused):
                chunks.extend(reused)
                entry = dict(previous)
                entry["source_modified_at"] = source_modified_at
                entry["last_seen_at"] = timestamp
                entries.append(entry)
                continue
            LOGGER.warning(
                "Prior chunks for %s do not match the manifest; reprocessing this document.",
                relative_path,
            )

        try:
            document = load_document(
                path,
                settings.documents_dir,
                file_hash=file_hash,
            )
            document_chunks = chunk_document(document, settings)
            if compute_file_hash(path) != file_hash:
                raise DocumentLoadError(
                    "Source file changed during ingestion; rerun to process a stable snapshot."
                )
        except (DocumentLoadError, OSError) as error:
            message = str(error) if isinstance(error, DocumentLoadError) else (
                f"Source file could not be read ({type(error).__name__})."
            )
            failure = DocumentLoadFailure(relative_path, file_type, file_hash, message)
            failures.append(failure)
            entries.append(
                {
                    "source_path": relative_path,
                    "file_name": path.name,
                    "file_type": file_type,
                    "file_hash": file_hash,
                    "status": "failed",
                    "parse_status": "failed",
                    "error": message,
                    "chunk_count": 0,
                    "source_modified_at": source_modified_at,
                    "processed_at": timestamp,
                    "last_seen_at": timestamp,
                }
            )
            continue

        chunks.extend(document_chunks)
        entries.append(
            {
                "source_path": document.source_path,
                "file_name": document.file_name,
                "file_type": document.file_type,
                "file_hash": document.file_hash,
                "status": "loaded",
                "parse_status": "loaded",
                "segment_count": len(document.segments),
                "chunk_count": len(document_chunks),
                "source_modified_at": source_modified_at,
                "processed_at": timestamp,
                "last_seen_at": timestamp,
            }
        )

    for path, entry in zip(current_paths, entries, strict=True):
        expected_hash = entry.get("file_hash")
        if isinstance(expected_hash, str) and compute_file_hash(path) != expected_hash:
            relative_path = path.relative_to(settings.documents_dir).as_posix()
            raise DocumentLoadError(
                f"{relative_path} changed during ingestion; rerun to process a stable snapshot."
            )

    current_chunks = tuple(chunks)
    fingerprint = chunk_snapshot_fingerprint(current_chunks)
    index_version = make_index_version(settings, fingerprint)
    changes = {
        "added": tuple(added),
        "changed": tuple(changed),
        "removed": removed,
        "failed": tuple(failure.source_path for failure in failures),
        "unchanged": tuple(unchanged),
    }
    indexes_current = _indexes_are_current(settings, fingerprint, index_version)
    chunks_on_disk = _load_previous_chunks(settings.processed_dir / "chunks.jsonl")
    same_chunk_file = (
        chunks_on_disk is not None
        and chunk_snapshot_fingerprint(chunks_on_disk) == fingerprint
    )
    rebuild = force or not indexes_current or not same_chunk_file

    if not rebuild:
        _write_json_atomic(
            settings.manifest_path,
            _manifest_payload(
                settings,
                entries,
                changes,
                current_chunks,
                fingerprint,
                index_version,
            ),
        )
        return IngestionRunSummary(
            chunks=current_chunks,
            failures=tuple(failures),
            added=changes["added"],
            changed=changes["changed"],
            removed=changes["removed"],
            unchanged=changes["unchanged"],
            index_version=index_version,
            total_documents=sum(entry.get("status") == "loaded" for entry in entries),
            rebuilt=False,
            embedded_chunks=0,
            chunks_path=settings.processed_dir / "chunks.jsonl",
            manifest_path=settings.manifest_path,
        )

    settings.faiss_index_dir.parent.mkdir(parents=True, exist_ok=True)
    settings.bm25_index_dir.parent.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stage_faiss = _create_stage_dir(
        settings.faiss_index_dir.parent,
        settings.faiss_index_dir.name,
    )
    stage_bm25 = _create_stage_dir(
        settings.bm25_index_dir.parent,
        settings.bm25_index_dir.name,
    )
    stage_processed = _create_stage_dir(
        settings.processed_dir.parent,
        settings.processed_dir.name,
    )
    stage_manifest_root = _create_stage_dir(
        settings.manifest_path.parent,
        settings.manifest_path.name,
    )
    stage_roots = (stage_faiss, stage_bm25, stage_processed, stage_manifest_root)
    stage_chunks_path = stage_processed / "chunks.jsonl"
    stage_manifest_path = stage_manifest_root / settings.manifest_path.name
    stage_settings = replace(
        settings,
        processed_dir=stage_processed,
        faiss_index_dir=stage_faiss,
        bm25_index_dir=stage_bm25,
        manifest_path=stage_manifest_path,
    )
    journal_path = _transaction_path(settings)
    published = False
    try:
        save_chunks(current_chunks, stage_chunks_path)
        bm25_store = build_bm25_index(stage_settings)
        dense_store = build_faiss_index(stage_settings, embedder=embedder)
        if not (
            bm25_store.corpus_fingerprint
            == dense_store.corpus_fingerprint
            == fingerprint
        ):
            raise ValueError("Staged indexes do not match the prepared chunk snapshot.")
        if not (
            bm25_store.index_version
            == dense_store.index_version
            == index_version
        ):
            raise ValueError("Staged indexes do not share the expected indexing version.")

        _write_json_atomic(
            stage_manifest_path,
            _manifest_payload(
                settings,
                entries,
                changes,
                current_chunks,
                fingerprint,
                index_version,
            ),
        )
        artifacts = (
            (settings.faiss_index_dir, stage_faiss),
            (settings.bm25_index_dir, stage_bm25),
            (settings.processed_dir / "chunks.jsonl", stage_chunks_path),
            (settings.manifest_path, stage_manifest_path),
        )
        _publish_build(settings, artifacts, stage_roots)
        published = True
    finally:
        if not published and not journal_path.exists():
            for stage_root in stage_roots:
                _remove_path(stage_root)

    return IngestionRunSummary(
        chunks=current_chunks,
        failures=tuple(failures),
        added=changes["added"],
        changed=changes["changed"],
        removed=changes["removed"],
        unchanged=changes["unchanged"],
        index_version=index_version,
        total_documents=sum(entry.get("status") == "loaded" for entry in entries),
        rebuilt=True,
        embedded_chunks=len(current_chunks),
        chunks_path=settings.processed_dir / "chunks.jsonl",
        manifest_path=settings.manifest_path,
    )
