import json

import pytest

from hybrid_rag.indexing.bm25_store import (
    BM25KeywordStore,
    BM25StoreError,
    tokenize_text,
)
from hybrid_rag.schemas import Chunk


def make_chunks() -> tuple[Chunk, ...]:
    return (
        Chunk(
            "a",
            "doc",
            "a.pdf",
            "a.pdf",
            "pdf",
            0,
            "Reference AES-256 and version v2.1.0 for product ZX-42.",
        ),
        Chunk("b", "doc", "a.pdf", "a.pdf", "pdf", 1, "AES-128 protects another record."),
        Chunk("c", "doc", "a.pdf", "a.pdf", "pdf", 2, "A document about apples and fruit."),
    )


def test_tokenizer_preserves_technical_terms_and_normalizes_case() -> None:
    assert tokenize_text("AES-256, v2.1.0; ZX-42; foo_bar; C++ C#!") == (
        "aes-256",
        "v2.1.0",
        "zx-42",
        "foo_bar",
        "c++",
        "c#",
    )


def test_bm25_index_round_trips_as_safe_json_and_preserves_mapping(tmp_path) -> None:
    chunks = make_chunks()
    store = BM25KeywordStore.from_chunks(chunks)
    index_path = tmp_path / "index.json"
    store.save(index_path)

    loaded = BM25KeywordStore.load(
        index_path,
        expected_corpus_fingerprint=store.corpus_fingerprint,
    )

    assert loaded.chunk_ids == ("a", "b", "c")
    assert loaded.tokenized_corpus == store.tokenized_corpus
    assert loaded.corpus_fingerprint == store.corpus_fingerprint
    assert len(loaded.scores(("aes-256",))) == len(chunks)
    assert list(tmp_path.iterdir()) == [index_path]
    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    assert metadata["tokenizer"] == "lowercase-tech-token-v1"
    assert metadata["chunk_ids"] == ["a", "b", "c"]


def test_missing_empty_and_malformed_bm25_indexes_are_rejected(tmp_path) -> None:
    with pytest.raises(BM25StoreError, match="missing"):
        BM25KeywordStore.load(tmp_path / "missing.json")
    empty = BM25KeywordStore.from_chunks(())
    assert empty.chunk_ids == ()
    assert empty.scores(("aes-256",)).size == 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(BM25StoreError, match="Could not read"):
        BM25KeywordStore.load(malformed)


def test_loading_index_for_different_chunk_snapshot_is_rejected(tmp_path) -> None:
    store = BM25KeywordStore.from_chunks(make_chunks())
    path = tmp_path / "index.json"
    store.save(path)

    with pytest.raises(BM25StoreError, match="different chunk snapshot"):
        BM25KeywordStore.load(path, expected_corpus_fingerprint="0" * 64)
