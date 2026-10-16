import pytest

from hybrid_rag.retrieval.fusion import FusionError, reciprocal_rank_fusion
from hybrid_rag.schemas import Chunk, RetrievalResult


def make_chunk(chunk_id: str, *, page_number: int | None = None) -> Chunk:
    return Chunk(
        chunk_id,
        "document",
        "manual.pdf",
        "manual.pdf",
        "pdf",
        ord(chunk_id[0]) - ord("a"),
        f"Text for {chunk_id}",
        page_number=page_number,
    )


def test_rrf_uses_one_based_ranks_deduplicates_and_preserves_scores() -> None:
    chunks = {chunk_id: make_chunk(chunk_id, page_number=2) for chunk_id in "abcd"}
    dense = (
        RetrievalResult(chunks["a"], score=0.93, dense_score=0.93),
        RetrievalResult(chunks["b"], score=0.77, dense_score=0.77),
        RetrievalResult(chunks["c"], score=0.51, dense_score=0.51),
    )
    bm25 = (
        RetrievalResult(chunks["b"], score=9.0, bm25_score=9.0),
        RetrievalResult(chunks["d"], score=7.0, bm25_score=7.0),
        RetrievalResult(chunks["a"], score=6.0, bm25_score=6.0),
    )

    results = reciprocal_rank_fusion(dense, bm25, rrf_k=60, top_k=10)

    assert [result.chunk.chunk_id for result in results] == ["b", "a", "d", "c"]
    assert len({result.chunk.chunk_id for result in results}) == len(results)
    assert results[0].fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].fused_score == pytest.approx(1 / 61 + 1 / 63)
    assert results[0].dense_score == pytest.approx(0.77)
    assert results[0].bm25_score == pytest.approx(9.0)
    assert results[1].dense_score == pytest.approx(0.93)
    assert results[1].bm25_score == pytest.approx(6.0)
    assert results[1].score == results[1].fused_score
    assert results[1].chunk.page_number == 2
    assert results[1].chunk.source_path == "manual.pdf"


def test_rrf_keeps_single_path_results_and_breaks_ties_by_chunk_id() -> None:
    dense = (RetrievalResult(make_chunk("z"), score=0.7, dense_score=0.7),)
    bm25 = (RetrievalResult(make_chunk("a"), score=4.0, bm25_score=4.0),)

    results = reciprocal_rank_fusion(dense, bm25, rrf_k=60, top_k=1)

    assert [result.chunk.chunk_id for result in results] == ["a"]
    assert results[0].dense_score is None
    assert results[0].bm25_score == pytest.approx(4.0)
    assert results[0].fused_score == pytest.approx(1 / 61)


def test_rrf_ignores_duplicate_chunk_ids_within_a_ranking() -> None:
    chunk = make_chunk("a")
    dense = (
        RetrievalResult(chunk, score=0.9, dense_score=0.9),
        RetrievalResult(chunk, score=0.8, dense_score=0.8),
    )

    results = reciprocal_rank_fusion(dense, (), rrf_k=10, top_k=5)

    assert len(results) == 1
    assert results[0].fused_score == pytest.approx(1 / 11)


def test_rrf_rejects_conflicting_metadata_and_invalid_configuration() -> None:
    dense = (RetrievalResult(make_chunk("a", page_number=1), score=1.0),)
    bm25 = (RetrievalResult(make_chunk("a", page_number=2), score=1.0),)

    with pytest.raises(FusionError, match="different chunk metadata"):
        reciprocal_rank_fusion(dense, bm25)
    with pytest.raises(FusionError, match="rrf_k"):
        reciprocal_rank_fusion((), (), rrf_k=0)
    with pytest.raises(FusionError, match="top_k"):
        reciprocal_rank_fusion((), (), top_k=0)
