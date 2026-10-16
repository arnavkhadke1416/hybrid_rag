"""Reciprocal Rank Fusion for dense and keyword retrieval results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from hybrid_rag.schemas import Chunk, RetrievalResult


class FusionError(ValueError):
    """Raised when retrieval rankings cannot be safely fused."""


@dataclass
class _FusionCandidate:
    chunk: Chunk
    fused_score: float = 0.0
    dense_score: float | None = None
    bm25_score: float | None = None


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> tuple[RetrievalResult, ...]:
    if rrf_k <= 0:
        raise FusionError("rrf_k must be a positive integer.")
    if top_k <= 0:
        raise FusionError("top_k must be a positive integer.")

    candidates: dict[str, _FusionCandidate] = {}

    def add_ranking(
        results: Sequence[RetrievalResult],
        source: Literal["dense", "bm25"],
    ) -> None:
        seen_ids: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            candidate = candidates.get(chunk_id)
            if candidate is None:
                candidate = _FusionCandidate(chunk=result.chunk)
                candidates[chunk_id] = candidate
            elif candidate.chunk != result.chunk:
                raise FusionError(
                    f"Retrieval paths returned different chunk metadata for {chunk_id}."
                )

            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            candidate.fused_score += 1.0 / (rrf_k + rank)

            if source == "dense":
                if candidate.dense_score is None:
                    candidate.dense_score = (
                        result.dense_score if result.dense_score is not None else result.score
                    )
                if candidate.bm25_score is None and result.bm25_score is not None:
                    candidate.bm25_score = result.bm25_score
            else:
                if candidate.bm25_score is None:
                    candidate.bm25_score = (
                        result.bm25_score if result.bm25_score is not None else result.score
                    )
                if candidate.dense_score is None and result.dense_score is not None:
                    candidate.dense_score = result.dense_score

    add_ranking(dense_results, "dense")
    add_ranking(bm25_results, "bm25")

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.fused_score, candidate.chunk.chunk_id),
    )[:top_k]
    return tuple(
        RetrievalResult(
            chunk=candidate.chunk,
            score=candidate.fused_score,
            dense_score=candidate.dense_score,
            bm25_score=candidate.bm25_score,
            fused_score=candidate.fused_score,
        )
        for candidate in ranked
    )
