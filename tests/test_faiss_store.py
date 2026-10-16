import json
import shutil

import numpy as np
import pytest

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL
from hybrid_rag.indexing.faiss_store import FaissStoreError, FaissVectorStore


def make_store() -> FaissVectorStore:
    return FaissVectorStore.from_vectors(
        ["chunk-a", "chunk-b", "chunk-c"],
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
    )


def test_faiss_index_saves_reloads_and_maps_ranked_results(tmp_path) -> None:
    store = make_store()
    store.save(tmp_path)

    loaded = FaissVectorStore.load(
        tmp_path,
        expected_model=DEFAULT_EMBEDDING_MODEL,
        expected_dimension=2,
    )
    hits = loaded.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k=10)

    assert loaded.chunk_ids == ("chunk-a", "chunk-b", "chunk-c")
    assert [chunk_id for chunk_id, _ in hits[0]] == ["chunk-a", "chunk-b", "chunk-c"]
    assert [score for _, score in hits[0]] == pytest.approx([1.0, 0.0, -1.0])
    assert (tmp_path / "index.faiss").is_file()
    mapping = json.loads((tmp_path / "mapping.json").read_text(encoding="utf-8"))
    assert mapping["metric"] == "inner_product"
    assert mapping["index_type"] == "IndexFlatIP"


def test_empty_and_invalid_vector_corpora_are_handled() -> None:
    empty = FaissVectorStore.from_vectors(
        [],
        np.empty((0, 2), dtype=np.float32),
        corpus_fingerprint="0" * 64,
    )
    assert empty.chunk_ids == ()
    with pytest.raises(FaissStoreError, match="unit-normalized"):
        FaissVectorStore.from_vectors(["chunk-a"], np.asarray([[2.0, 0.0]]))
    with pytest.raises(FaissStoreError, match="unique"):
        FaissVectorStore.from_vectors(
            ["duplicate", "duplicate"],
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        )


def test_missing_index_and_incompatible_metadata_are_actionable(tmp_path) -> None:
    with pytest.raises(FaissStoreError, match="missing"):
        FaissVectorStore.load(tmp_path)

    store = make_store()
    store.save(tmp_path)
    with pytest.raises(FaissStoreError, match="does not match the configured model"):
        FaissVectorStore.load(tmp_path, expected_model="different-model")
    with pytest.raises(FaissStoreError, match="does not match embedding model dimension"):
        FaissVectorStore.load(tmp_path, expected_dimension=3)


def test_inconsistent_row_mapping_and_query_dimensions_are_rejected(tmp_path) -> None:
    store = make_store()
    store.save(tmp_path)
    mapping_path = tmp_path / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["chunk_ids"].pop()
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(FaissStoreError, match="row mapping"):
        FaissVectorStore.load(tmp_path)

    with pytest.raises(FaissStoreError, match="dimension must match"):
        store.search(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="positive integer"):
        store.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k=0)


def test_index_and_mapping_from_different_saves_are_rejected(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = make_store()
    second = FaissVectorStore.from_vectors(
        ["chunk-a", "chunk-b", "chunk-c"],
        np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    )
    first.save(first_dir)
    second.save(second_dir)
    shutil.copyfile(second_dir / "index.faiss", first_dir / "index.faiss")

    with pytest.raises(FaissStoreError, match="checksum"):
        FaissVectorStore.load(first_dir)
