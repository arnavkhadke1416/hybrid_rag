from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from docx import Document

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL, Settings
from hybrid_rag.indexing.bm25_store import BM25KeywordStore
from hybrid_rag.indexing.faiss_store import FaissVectorStore
from hybrid_rag.indexing.version import make_index_version
from hybrid_rag.ingestion import automatic
from hybrid_rag.ingestion.automatic import (
    IngestionTransactionError,
    _create_stage_dir,
    _publish_build,
    _recover_incomplete_build,
    run_ingestion,
)
from hybrid_rag.retrieval.hybrid import HybridRetriever


class FakeEmbedder:
    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(tuple(texts))
        return np.asarray(
            [
                [1.0, 0.0] if "aes-256" in text.lower() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def make_settings(tmp_path: Path) -> Settings:
    documents = tmp_path / "documents"
    documents.mkdir()
    return Settings(project_root=tmp_path)


def write_docx(path: Path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(path)


def test_ingestion_detects_all_file_changes_and_keeps_index_versions_synchronized(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    write_docx(settings.documents_dir / "first.docx", "Backups use AES-256.")
    write_docx(settings.documents_dir / "second.docx", "Accounts use password resets.")
    embedder = FakeEmbedder()

    initial = run_ingestion(settings, embedder=embedder)
    assert set(initial.added) == {"first.docx", "second.docx"}
    assert initial.rebuilt
    assert len(embedder.calls) == 1

    unchanged = run_ingestion(settings, embedder=embedder)
    assert unchanged.unchanged == ("first.docx", "second.docx")
    assert unchanged.added == unchanged.changed == unchanged.removed == ()
    assert not unchanged.rebuilt
    assert len(embedder.calls) == 1

    write_docx(settings.documents_dir / "first.docx", "Backups now use AES-256 and rotation.")
    (settings.documents_dir / "second.docx").unlink()
    write_docx(settings.documents_dir / "third.docx", "Recovery requires an administrator.")
    updated = run_ingestion(settings, embedder=embedder)

    assert updated.added == ("third.docx",)
    assert updated.changed == ("first.docx",)
    assert updated.removed == ("second.docx",)
    assert {chunk.source_path for chunk in updated.chunks} == {"first.docx", "third.docx"}
    assert updated.rebuilt
    assert len(embedder.calls) == 2

    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["indexing"]["index_version"] == updated.index_version
    assert manifest["changes"]["removed"] == ["second.docx"]
    assert all(
        item["status"] == item["parse_status"] == "loaded"
        and item["source_modified_at"]
        and item["processed_at"]
        and item["file_hash"]
        and "chunk_count" in item
        for item in manifest["files"]
    )
    dense = FaissVectorStore.load(
        settings.faiss_index_dir,
        expected_model=settings.embedding_model,
        expected_index_version=updated.index_version,
    )
    sparse = BM25KeywordStore.load(
        settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE,
        expected_index_version=updated.index_version,
    )
    assert dense.corpus_fingerprint == sparse.corpus_fingerprint
    assert dense.index_version == sparse.index_version == updated.index_version
    assert set(dense.chunk_ids) == {chunk.chunk_id for chunk in updated.chunks}
    assert sparse.chunk_ids == dense.chunk_ids


def test_duplicate_file_contents_remain_separate_sources(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first = settings.documents_dir / "first.docx"
    second = settings.documents_dir / "duplicate.docx"
    write_docx(first, "The same source content is copied.")
    shutil.copyfile(first, second)

    result = run_ingestion(settings, embedder=FakeEmbedder())

    assert result.added == ("duplicate.docx", "first.docx")
    assert {chunk.source_path for chunk in result.chunks} == {
        "duplicate.docx",
        "first.docx",
    }
    assert len({chunk.chunk_id for chunk in result.chunks}) == 2


def test_force_rebuild_reprocesses_documents_and_empty_corpus_remains_indexable(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    source = settings.documents_dir / "memo.docx"
    write_docx(source, "AES-256 protects backup files.")
    embedder = FakeEmbedder()
    run_ingestion(settings, embedder=embedder)

    forced = run_ingestion(settings, force=True, embedder=embedder)
    assert forced.unchanged == ("memo.docx",)
    assert forced.rebuilt
    assert len(embedder.calls) == 2

    source.unlink()
    removed = run_ingestion(settings, embedder=embedder)
    assert removed.removed == ("memo.docx",)
    assert removed.chunks == ()
    assert removed.rebuilt
    assert len(embedder.calls) == 2
    assert FaissVectorStore.load(
        settings.faiss_index_dir,
        expected_model=settings.embedding_model,
        expected_index_version=removed.index_version,
    ).chunk_ids == ()
    assert BM25KeywordStore.load(
        settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE,
        expected_index_version=removed.index_version,
    ).chunk_ids == ()
    assert HybridRetriever(settings, embedder=embedder).retrieve("AES-256").results == ()


def test_chunk_configuration_change_reprocesses_and_versions_indexes(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    source = settings.documents_dir / "memo.docx"
    write_docx(source, "AES-256 protects backup files.")
    embedder = FakeEmbedder()
    initial = run_ingestion(settings, embedder=embedder)

    changed_settings = replace(settings, chunk_size=settings.chunk_size + 100)
    updated = run_ingestion(changed_settings, embedder=embedder)

    assert updated.unchanged == ("memo.docx",)
    assert updated.rebuilt
    assert updated.index_version != initial.index_version
    assert len(embedder.calls) == 2


def test_modified_unreadable_document_is_reported_and_old_chunks_are_excluded(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    source = settings.documents_dir / "memo.docx"
    write_docx(source, "Original policy uses AES-256.")
    embedder = FakeEmbedder()
    initial = run_ingestion(settings, embedder=embedder)
    assert initial.chunks

    source.write_bytes(b"not a valid DOCX package")
    updated = run_ingestion(settings, embedder=embedder)

    assert updated.changed == ("memo.docx",)
    assert [failure.source_path for failure in updated.failures] == ["memo.docx"]
    assert updated.chunks == ()
    assert updated.rebuilt
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"][0]["status"] == "failed"
    assert manifest["files"][0]["parse_status"] == "failed"
    assert manifest["files"][0]["file_hash"]
    assert manifest["files"][0]["error"]
    assert FaissVectorStore.load(
        settings.faiss_index_dir,
        expected_model=settings.embedding_model,
        expected_index_version=updated.index_version,
    ).chunk_ids == ()
    assert BM25KeywordStore.load(
        settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE,
        expected_index_version=updated.index_version,
    ).chunk_ids == ()


def test_failed_index_build_keeps_last_good_artifacts_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    source = settings.documents_dir / "memo.docx"
    write_docx(source, "Original backup policy.")
    embedder = FakeEmbedder()
    run_ingestion(settings, embedder=embedder)
    destinations = (
        settings.faiss_index_dir / FaissVectorStore.INDEX_FILE,
        settings.faiss_index_dir / FaissVectorStore.MAPPING_FILE,
        settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE,
        settings.processed_dir / "chunks.jsonl",
        settings.manifest_path,
    )
    originals = {path: path.read_bytes() for path in destinations}
    write_docx(source, "Updated backup policy with AES-256.")

    def fail_dense_build(*_args, **_kwargs):
        raise RuntimeError("synthetic dense build failure")

    monkeypatch.setattr(automatic, "build_faiss_index", fail_dense_build)
    with pytest.raises(RuntimeError, match="synthetic dense build failure"):
        run_ingestion(settings, embedder=embedder)

    assert {path: path.read_bytes() for path in destinations} == originals
    assert not automatic._transaction_path(settings).exists()
    assert not list(settings.faiss_index_dir.parent.glob(".faiss.stage-*"))
    assert not list(settings.bm25_index_dir.parent.glob(".bm25.stage-*"))


def test_interrupted_publication_recovers_previous_index_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    source = settings.documents_dir / "memo.docx"
    write_docx(source, "Original backup policy.")
    run_ingestion(settings, embedder=FakeEmbedder())
    original = {
        settings.faiss_index_dir / FaissVectorStore.INDEX_FILE:
            (settings.faiss_index_dir / FaissVectorStore.INDEX_FILE).read_bytes(),
        settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE:
            (settings.bm25_index_dir / BM25KeywordStore.INDEX_FILE).read_bytes(),
        settings.processed_dir / "chunks.jsonl":
            (settings.processed_dir / "chunks.jsonl").read_bytes(),
        settings.manifest_path: settings.manifest_path.read_bytes(),
    }

    stage_faiss = _create_stage_dir(settings.faiss_index_dir.parent, "faiss")
    stage_bm25 = _create_stage_dir(settings.bm25_index_dir.parent, "bm25")
    stage_processed = _create_stage_dir(settings.processed_dir.parent, "processed")
    stage_manifest = _create_stage_dir(settings.manifest_path.parent, settings.manifest_path.name)
    stages = (stage_faiss, stage_bm25, stage_processed, stage_manifest)
    (stage_faiss / "new-index.txt").write_text("new", encoding="utf-8")
    (stage_bm25 / "new-index.txt").write_text("new", encoding="utf-8")
    (stage_processed / "chunks.jsonl").write_text("new\n", encoding="utf-8")
    (stage_manifest / settings.manifest_path.name).write_text("new", encoding="utf-8")
    artifacts = (
        (settings.faiss_index_dir, stage_faiss),
        (settings.bm25_index_dir, stage_bm25),
        (settings.processed_dir / "chunks.jsonl", stage_processed / "chunks.jsonl"),
        (settings.manifest_path, stage_manifest / settings.manifest_path.name),
    )
    original_replace = automatic.os.replace
    fail_stage = True
    fail_restore = True

    def fail_during_publish(source_path, destination_path):
        nonlocal fail_stage, fail_restore
        source = Path(source_path)
        if source == stage_bm25 and fail_stage:
            fail_stage = False
            raise OSError("synthetic interrupted publication")
        if source.name.startswith(f".{settings.bm25_index_dir.name}.backup-") and fail_restore:
            fail_restore = False
            raise OSError("synthetic interrupted rollback")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(automatic.os, "replace", fail_during_publish)
    with pytest.raises(IngestionTransactionError, match="rollback was incomplete"):
        _publish_build(settings, artifacts, stages)
    monkeypatch.setattr(automatic.os, "replace", original_replace)

    assert automatic._transaction_path(settings).is_file()
    _recover_incomplete_build(settings)

    assert {path: path.read_bytes() for path in original} == original
    assert not automatic._transaction_path(settings).exists()
    assert all(not stage.exists() for stage in stages)
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["indexing"]["index_version"] == make_index_version(
        settings,
        manifest["chunk_fingerprint"],
    )
