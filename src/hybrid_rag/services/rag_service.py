"""Coordinate hybrid retrieval and grounded answer generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from hybrid_rag.config import Settings
from hybrid_rag.generation.gemini import GeminiGenerator, GroundedAnswer
from hybrid_rag.retrieval.hybrid import HybridRetrievalResponse, HybridRetriever
from hybrid_rag.schemas import RetrievalResult


class _Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        diagnostic: bool = False,
    ) -> HybridRetrievalResponse: ...


class _AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GroundedAnswer: ...


@dataclass(frozen=True)
class RAGServiceResponse:
    answer: GroundedAnswer
    retrieval: HybridRetrievalResponse


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: _Retriever | None = None,
        generator: _AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.retriever = (
            retriever if retriever is not None else HybridRetriever(self.settings)
        )
        self.generator = (
            generator if generator is not None else GeminiGenerator(self.settings)
        )

    def ask(self, question: str, *, diagnostic: bool = False) -> RAGServiceResponse:
        retrieval = self.retriever.retrieve(question, diagnostic=diagnostic)
        answer = self.generator.generate(question, retrieval.results)
        return RAGServiceResponse(answer=answer, retrieval=retrieval)
