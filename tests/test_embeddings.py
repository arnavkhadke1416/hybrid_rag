import numpy as np
import pytest

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL
from hybrid_rag.indexing.embeddings import EmbeddingError, SentenceTransformerEmbeddings


class FakeSentenceTransformer:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.received_texts: list[str] = []

    def get_embedding_dimension(self) -> int:
        return len(self.vectors[0])

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.received_texts.extend(texts)
        return np.asarray(self.vectors, dtype=np.float32)


def test_embeddings_use_exact_model_and_normalize_output() -> None:
    model = FakeSentenceTransformer([[3.0, 4.0], [0.0, 2.0]])
    embeddings = SentenceTransformerEmbeddings(model=model)

    vectors = embeddings.encode(["first", "second"])

    assert embeddings.model_name == DEFAULT_EMBEDDING_MODEL
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])
    assert model.received_texts == ["first", "second"]


def test_embedding_model_cannot_be_silently_substituted() -> None:
    with pytest.raises(ValueError, match="must remain"):
        SentenceTransformerEmbeddings("another/model")


def test_empty_texts_return_an_empty_matrix_without_model_call() -> None:
    model = FakeSentenceTransformer([[1.0, 0.0]])
    embeddings = SentenceTransformerEmbeddings(model=model)

    vectors = embeddings.encode([])

    assert vectors.shape == (0, 2)
    assert model.received_texts == []


@pytest.mark.parametrize(
    "vectors",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
    ],
)
def test_invalid_embedding_outputs_are_rejected(vectors: list[list[float]]) -> None:
    model = FakeSentenceTransformer(vectors)
    embeddings = SentenceTransformerEmbeddings(model=model)

    with pytest.raises(EmbeddingError):
        embeddings.encode(["single input"])


def test_blank_input_and_invalid_batch_size_are_rejected() -> None:
    model = FakeSentenceTransformer([[1.0, 0.0]])
    embeddings = SentenceTransformerEmbeddings(model=model)

    with pytest.raises(ValueError, match="must not be empty"):
        embeddings.encode(["  "])
    with pytest.raises(ValueError, match="positive integer"):
        SentenceTransformerEmbeddings(batch_size=0, model=model)
