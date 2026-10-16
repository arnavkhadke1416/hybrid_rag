"""Safe JSON-persisted BM25 keyword index."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from hybrid_rag.indexing.corpus import chunk_snapshot_fingerprint
from hybrid_rag.schemas import Chunk


TOKENIZER_NAME = "lowercase-tech-token-v1"
TOKEN_PATTERN = re.compile(r"[\w]+(?:[-./+#][\w]+)*(?:\+\+|#)?", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25


class BM25StoreError(RuntimeError):
    """Raised when a BM25 index is missing, invalid, or incompatible."""


def tokenize_text(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in TOKEN_PATTERN.findall(text))


class BM25KeywordStore:
    INDEX_FILE = "index.json"
    SCHEMA_VERSION = 2

    def __init__(
        self,
        chunk_ids: Sequence[str],
        tokenized_corpus: Sequence[Sequence[str]],
        corpus_fingerprint: str,
        index_version: str | None = None,
    ) -> None:
        self.chunk_ids = tuple(chunk_ids)
        self.tokenized_corpus = tuple(tuple(tokens) for tokens in tokenized_corpus)
        self.corpus_fingerprint = corpus_fingerprint
        self.index_version = index_version

        if len(self.chunk_ids) != len(self.tokenized_corpus):
            raise BM25StoreError("BM25 corpus rows and chunk-ID mapping are inconsistent.")
        if any(not chunk_id for chunk_id in self.chunk_ids) or len(set(self.chunk_ids)) != len(
            self.chunk_ids
        ):
            raise BM25StoreError("BM25 chunk IDs must be non-empty and unique.")
        if any(
            not isinstance(token, str) or not token or token != token.lower()
            for document in self.tokenized_corpus
            for token in document
        ):
            raise BM25StoreError("BM25 corpus tokens must be non-empty lowercase strings.")
        if (
            not isinstance(corpus_fingerprint, str)
            or len(corpus_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in corpus_fingerprint)
        ):
            raise BM25StoreError("BM25 corpus fingerprint is invalid.")
        if index_version is not None and (
            not isinstance(index_version, str)
            or len(index_version) != 64
            or any(character not in "0123456789abcdef" for character in index_version)
        ):
            raise BM25StoreError("BM25 index version is invalid.")

        self._index = (
            BM25Okapi(
                [list(tokens) for tokens in self.tokenized_corpus],
                k1=BM25_K1,
                b=BM25_B,
                epsilon=BM25_EPSILON,
            )
            if self.chunk_ids and any(self.tokenized_corpus)
            else None
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        *,
        index_version: str | None = None,
    ) -> BM25KeywordStore:
        return cls(
            [chunk.chunk_id for chunk in chunks],
            [tokenize_text(chunk.text) for chunk in chunks],
            chunk_snapshot_fingerprint(chunks),
            index_version,
        )

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        if self._index is None:
            return np.zeros(len(self.chunk_ids), dtype=np.float64)
        return np.asarray(self._index.get_scores(list(query_tokens)), dtype=np.float64)

    def save(self, path: Path) -> None:
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
                json.dump(
                    {
                        "schema_version": self.SCHEMA_VERSION,
                        "tokenizer": TOKENIZER_NAME,
                        "corpus_fingerprint": self.corpus_fingerprint,
                        "index_version": self.index_version,
                        "chunk_ids": self.chunk_ids,
                        "tokenized_corpus": self.tokenized_corpus,
                        "bm25": {
                            "k1": BM25_K1,
                            "b": BM25_B,
                            "epsilon": BM25_EPSILON,
                        },
                    },
                    temporary_file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                temporary_file.write("\n")
            os.replace(temporary_path, destination)
        except (OSError, TypeError, ValueError) as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise BM25StoreError(f"Could not save BM25 index to {destination}.") from error

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_corpus_fingerprint: str | None = None,
        expected_index_version: str | None = None,
    ) -> BM25KeywordStore:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise BM25StoreError(
                f"BM25 index is missing at {source}; rebuild it from the current chunks."
            )
        try:
            metadata = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BM25StoreError(f"Could not read BM25 index at {source}.") from error
        if not isinstance(metadata, dict):
            raise BM25StoreError("BM25 index metadata must be a JSON object.")
        if metadata.get("schema_version") != cls.SCHEMA_VERSION:
            raise BM25StoreError("BM25 index schema version is unsupported; rebuild the index.")
        if metadata.get("tokenizer") != TOKENIZER_NAME:
            raise BM25StoreError("BM25 tokenizer does not match this implementation; rebuild the index.")
        if metadata.get("bm25") != {
            "k1": BM25_K1,
            "b": BM25_B,
            "epsilon": BM25_EPSILON,
        }:
            raise BM25StoreError("BM25 parameters are unsupported; rebuild the index.")

        chunk_ids = metadata.get("chunk_ids")
        tokenized_corpus = metadata.get("tokenized_corpus")
        corpus_fingerprint = metadata.get("corpus_fingerprint")
        if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
            raise BM25StoreError("BM25 chunk-ID mapping is invalid.")
        if not isinstance(tokenized_corpus, list) or any(
            not isinstance(document, list)
            or not all(isinstance(token, str) for token in document)
            for document in tokenized_corpus
        ):
            raise BM25StoreError("BM25 tokenized corpus is invalid.")
        if not isinstance(corpus_fingerprint, str):
            raise BM25StoreError("BM25 corpus fingerprint is missing.")
        index_version = metadata.get("index_version")
        if index_version is not None and (
            not isinstance(index_version, str)
            or len(index_version) != 64
            or any(character not in "0123456789abcdef" for character in index_version)
        ):
            raise BM25StoreError("BM25 index version is invalid.")
        if (
            expected_corpus_fingerprint is not None
            and corpus_fingerprint != expected_corpus_fingerprint
        ):
            raise BM25StoreError(
                "BM25 index was built from a different chunk snapshot; rebuild it after ingestion."
            )
        if expected_index_version is not None and index_version != expected_index_version:
            raise BM25StoreError(
                "BM25 index version does not match the current index configuration; rebuild both indexes."
            )
        try:
            return cls(chunk_ids, tokenized_corpus, corpus_fingerprint, index_version)
        except (ArithmeticError, TypeError, ValueError) as error:
            raise BM25StoreError(f"BM25 index metadata is invalid at {source}.") from error
