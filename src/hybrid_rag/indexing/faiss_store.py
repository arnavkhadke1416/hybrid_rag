"""Normalized cosine-like dense index backed by FAISS IndexFlatIP."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL


class FaissStoreError(RuntimeError):
    """Raised for missing, invalid, or incompatible dense indexes."""


class FaissVectorStore:
    INDEX_FILE = "index.faiss"
    MAPPING_FILE = "mapping.json"
    SCHEMA_VERSION = 4

    def __init__(
        self,
        index: faiss.Index,
        chunk_ids: Sequence[str],
        embedding_model: str,
        corpus_fingerprint: str | None = None,
        index_version: str | None = None,
    ) -> None:
        self.index = index
        self.chunk_ids = tuple(chunk_ids)
        self.embedding_model = embedding_model
        self.corpus_fingerprint = corpus_fingerprint
        self.index_version = index_version
        self.dimension = int(index.d)
        self._validate_index()

    @classmethod
    def from_vectors(
        cls,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        corpus_fingerprint: str | None = None,
        index_version: str | None = None,
    ) -> FaissVectorStore:
        ids = tuple(chunk_ids)
        if any(not chunk_id for chunk_id in ids) or len(set(ids)) != len(ids):
            raise FaissStoreError("FAISS chunk IDs must be non-empty and unique.")
        if not embedding_model:
            raise FaissStoreError("An embedding model name is required for the FAISS index.")

        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise FaissStoreError("Embedding vectors must be a numeric matrix.") from error
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0:
            raise FaissStoreError("Embedding matrix rows and dimensions do not match the chunk mapping.")
        if not np.isfinite(matrix).all():
            raise FaissStoreError("Embedding matrix contains non-finite values.")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
            raise FaissStoreError("FAISS requires unit-normalized vectors for cosine-like ranking.")

        normalized_matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        index = faiss.IndexFlatIP(int(normalized_matrix.shape[1]))
        index.add(normalized_matrix)
        return cls(index, ids, embedding_model, corpus_fingerprint, index_version)

    def _validate_index(self) -> None:
        if not isinstance(self.index, faiss.IndexFlatIP):
            raise FaissStoreError("Dense index must use FAISS IndexFlatIP.")
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise FaissStoreError("Dense index must use inner-product similarity.")
        if self.dimension <= 0 or self.index.ntotal != len(self.chunk_ids):
            raise FaissStoreError("FAISS index dimensions or row mapping are inconsistent.")
        if not self.embedding_model:
            raise FaissStoreError("FAISS index metadata is missing its embedding model.")
        if self.corpus_fingerprint is not None and (
            not isinstance(self.corpus_fingerprint, str)
            or len(self.corpus_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.corpus_fingerprint)
        ):
            raise FaissStoreError("FAISS chunk snapshot fingerprint is invalid.")
        if self.index_version is not None and (
            not isinstance(self.index_version, str)
            or len(self.index_version) != 64
            or any(character not in "0123456789abcdef" for character in self.index_version)
        ):
            raise FaissStoreError("FAISS index version is invalid.")
        if any(not chunk_id for chunk_id in self.chunk_ids):
            raise FaissStoreError("FAISS index mapping contains an empty chunk ID.")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise FaissStoreError("FAISS index mapping contains duplicate chunk IDs.")

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[tuple[tuple[str, float], ...], ...]:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if not self.chunk_ids:
            raise FaissStoreError("FAISS index has no vectors; build it from a non-empty chunk corpus.")
        try:
            queries = np.asarray(query_vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise FaissStoreError("Query embeddings must be a numeric matrix.") from error
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise FaissStoreError(
                f"Query embedding dimension must match the index dimension ({self.dimension})."
            )
        if not np.isfinite(queries).all():
            raise FaissStoreError("Query embeddings contain non-finite values.")
        norms = np.linalg.norm(queries, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
            raise FaissStoreError("Query embeddings must be unit-normalized like indexed vectors.")

        scores, positions = self.index.search(
            np.ascontiguousarray(queries),
            min(k, len(self.chunk_ids)),
        )
        return tuple(
            tuple(
                (self.chunk_ids[int(position)], float(score))
                for score, position in zip(row_scores, row_positions, strict=True)
                if position >= 0
            )
            for row_scores, row_positions in zip(scores, positions, strict=True)
        )

    def save(self, index_dir: Path) -> None:
        destination = Path(index_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        index_path = destination / self.INDEX_FILE
        mapping_path = destination / self.MAPPING_FILE
        index_temp: Path | None = None
        mapping_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination,
                prefix=".index.",
                suffix=".faiss.tmp",
                delete=False,
            ) as temporary_index:
                index_temp = Path(temporary_index.name)
            faiss.write_index(self.index, str(index_temp))
            with index_temp.open("rb") as index_file:
                index_sha256 = hashlib.file_digest(index_file, "sha256").hexdigest()
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination,
                prefix=".mapping.",
                suffix=".json.tmp",
                delete=False,
            ) as temporary_mapping:
                mapping_temp = Path(temporary_mapping.name)
                json.dump(
                    {
                        "schema_version": self.SCHEMA_VERSION,
                        "index_type": "IndexFlatIP",
                        "metric": "inner_product",
                        "embedding_model": self.embedding_model,
                        "dimension": self.dimension,
                        "corpus_fingerprint": self.corpus_fingerprint,
                        "index_version": self.index_version,
                        "index_sha256": index_sha256,
                        "chunk_ids": self.chunk_ids,
                    },
                    temporary_mapping,
                    ensure_ascii=False,
                    indent=2,
                )
                temporary_mapping.write("\n")
            os.replace(index_temp, index_path)
            os.replace(mapping_temp, mapping_path)
        except (OSError, RuntimeError, ValueError) as error:
            for temporary_path in (index_temp, mapping_temp):
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            raise FaissStoreError(f"Could not save FAISS index under {destination}.") from error

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        expected_model: str = DEFAULT_EMBEDDING_MODEL,
        expected_dimension: int | None = None,
        expected_index_version: str | None = None,
    ) -> FaissVectorStore:
        source = Path(index_dir).expanduser().resolve()
        index_path = source / cls.INDEX_FILE
        mapping_path = source / cls.MAPPING_FILE
        if not index_path.is_file() or not mapping_path.is_file():
            raise FaissStoreError(
                f"FAISS index or chunk mapping is missing under {source}; "
                "build the dense index from the current chunks before searching."
            )

        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FaissStoreError(f"Could not read FAISS index files under {source}.") from error
        if not isinstance(mapping, dict):
            raise FaissStoreError("FAISS mapping metadata must be a JSON object.")
        if mapping.get("schema_version") != cls.SCHEMA_VERSION:
            raise FaissStoreError("FAISS mapping schema version is unsupported; rebuild the dense index.")
        try:
            with index_path.open("rb") as index_file:
                index_sha256 = hashlib.file_digest(index_file, "sha256").hexdigest()
        except OSError as error:
            raise FaissStoreError(f"Could not read FAISS index files under {source}.") from error
        if mapping.get("index_sha256") != index_sha256:
            raise FaissStoreError(
                "FAISS index checksum does not match its chunk mapping; rebuild the dense index."
            )
        try:
            index = faiss.read_index(str(index_path))
        except (OSError, RuntimeError) as error:
            raise FaissStoreError(f"Could not read FAISS index files under {source}.") from error
        if mapping.get("index_type") != "IndexFlatIP" or mapping.get("metric") != "inner_product":
            raise FaissStoreError("FAISS mapping uses an unsupported index type or metric.")
        if mapping.get("embedding_model") != expected_model:
            raise FaissStoreError(
                "FAISS embedding model does not match the configured model; rebuild the dense index."
            )
        dimension = mapping.get("dimension")
        chunk_ids = mapping.get("chunk_ids")
        if not isinstance(dimension, int) or dimension <= 0:
            raise FaissStoreError("FAISS mapping dimension is invalid.")
        if expected_dimension is not None and dimension != expected_dimension:
            raise FaissStoreError(
                f"FAISS dimension {dimension} does not match embedding model dimension "
                f"{expected_dimension}; rebuild the dense index."
            )
        if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
            raise FaissStoreError("FAISS mapping chunk IDs are invalid.")
        corpus_fingerprint = mapping.get("corpus_fingerprint")
        if corpus_fingerprint is not None and (
            not isinstance(corpus_fingerprint, str)
            or len(corpus_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in corpus_fingerprint)
        ):
            raise FaissStoreError("FAISS chunk snapshot fingerprint is invalid.")
        index_version = mapping.get("index_version")
        if index_version is not None and (
            not isinstance(index_version, str)
            or len(index_version) != 64
            or any(character not in "0123456789abcdef" for character in index_version)
        ):
            raise FaissStoreError("FAISS index version is invalid.")
        if expected_index_version is not None and index_version != expected_index_version:
            raise FaissStoreError(
                "FAISS index version does not match the current index configuration; rebuild both indexes."
            )
        if index.d != dimension:
            raise FaissStoreError("FAISS index dimension does not match its mapping metadata.")
        return cls(index, chunk_ids, expected_model, corpus_fingerprint, index_version)
