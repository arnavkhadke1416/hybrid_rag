"""Sentence Transformers embeddings for the configured exact model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding model cannot produce valid vectors."""


class EmbeddingEncoder(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class _SentenceTransformerModel(Protocol):
    def get_embedding_dimension(self) -> int | None: ...

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


class SentenceTransformerEmbeddings:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        batch_size: int = 32,
        device: str = "cpu",
        model: _SentenceTransformerModel | None = None,
    ) -> None:
        if model_name != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(
                f"The configured embedding model must remain {DEFAULT_EMBEDDING_MODEL!r}."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.model_name = model_name
        self.batch_size = batch_size

        try:
            self._model = (
                model
                if model is not None
                else SentenceTransformer(model_name, device=device, token=False)
            )
            dimension = self._model.get_embedding_dimension()
        except Exception as error:
            raise EmbeddingError(
                f"Could not load the exact embedding model {model_name!r} on {device}."
            ) from error
        if not isinstance(dimension, int) or dimension <= 0:
            raise EmbeddingError("The embedding model returned an invalid vector dimension.")
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = tuple(texts)
        if any(not text.strip() for text in values):
            raise ValueError("Embedding input texts must not be empty.")
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            encoded = self._model.encode(
                list(values),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            vectors = np.asarray(encoded, dtype=np.float32)
        except Exception as error:
            raise EmbeddingError("The configured embedding model failed to encode input text.") from error

        if vectors.ndim != 2 or vectors.shape != (len(values), self.dimension):
            raise EmbeddingError(
                f"Embedding output shape must be ({len(values)}, {self.dimension})."
            )
        if not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding output contains non-finite values.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingError("Embedding output contains a zero-length vector.")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)
