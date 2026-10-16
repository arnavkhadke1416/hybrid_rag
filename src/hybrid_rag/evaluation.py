"""Reproducible dense, BM25, and hybrid retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from hybrid_rag.config import PROJECT_ROOT, Settings
from hybrid_rag.indexing.build_indexes import build_bm25_index, build_faiss_index
from hybrid_rag.indexing.embeddings import EmbeddingEncoder, SentenceTransformerEmbeddings
from hybrid_rag.ingestion.cleaning import clean_text
from hybrid_rag.ingestion.chunking import chunk_document, save_chunks
from hybrid_rag.retrieval.bm25 import retrieve_bm25
from hybrid_rag.retrieval.dense import retrieve_dense
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.retrieval.hybrid import HybridRetriever
from hybrid_rag.schemas import Chunk, DocumentRecord, DocumentSegment, RetrievalResult


DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"
METRIC_CUTOFFS = (1, 3, 5)
SWEEP_CANDIDATE_COUNTS = (3, 5, 10)
SWEEP_RRF_VALUES = (10, 30, 60)
SWEEP_CHUNK_CONFIGURATIONS = ((160, 0), (160, 40), (320, 80), (800, 120))


class EvaluationError(ValueError):
    """Raised when the benchmark dataset or evaluation run is invalid."""


def _load_dataset(path: Path) -> tuple[dict[str, Any], tuple[Chunk, ...]]:
    try:
        raw = path.read_bytes()
        dataset = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Could not read evaluation dataset at {path}.") from error
    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise EvaluationError("Evaluation dataset has an unsupported schema version.")
    corpus = dataset.get("corpus")
    cases = dataset.get("cases")
    if not isinstance(corpus, list) or not corpus or not isinstance(cases, list) or not cases:
        raise EvaluationError("Evaluation dataset must contain a non-empty corpus and case list.")

    try:
        chunks = tuple(Chunk(**item) for item in corpus if isinstance(item, dict))
    except (TypeError, ValueError) as error:
        raise EvaluationError("Evaluation corpus contains an invalid chunk record.") from error
    if len(chunks) != len(corpus) or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise EvaluationError("Evaluation corpus must have unique chunk IDs and valid records.")
    known_ids = {chunk.chunk_id for chunk in chunks}
    case_ids: set[str] = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("case_id"), str)
            or case["case_id"] in case_ids
            or not isinstance(case.get("query"), str)
            or not case["query"].strip()
            or not isinstance(case.get("relevant_chunk_ids"), list)
            or not all(isinstance(item, str) for item in case["relevant_chunk_ids"])
            or not isinstance(case.get("answerable"), bool)
        ):
            raise EvaluationError("Evaluation dataset contains an invalid query case.")
        relevant = set(case["relevant_chunk_ids"])
        if not relevant.issubset(known_ids) or case["answerable"] != bool(relevant):
            raise EvaluationError(
                f"Evaluation case {case['case_id']} has inconsistent relevance labels."
            )
        case_ids.add(case["case_id"])
    return dataset, chunks


def score_ranking(
    ranked_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    *,
    cutoff: int,
) -> dict[str, float]:
    if cutoff <= 0:
        raise ValueError("cutoff must be a positive integer.")
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return {"recall": 0.0, "reciprocal_rank": 0.0}
    recall = len(set(ranked_chunk_ids[:cutoff]).intersection(relevant)) / len(relevant)
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)
            if chunk_id in relevant
        ),
        0.0,
    )
    return {"recall": recall, "reciprocal_rank": reciprocal_rank}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_metrics(
    results: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, float | None]:
    answerable = [item for item in results if item["answerable"]]
    metrics: dict[str, float | None] = {
        "mrr": _mean(
            [
                score_ranking(
                    item["rankings"][method],
                    item["relevant_chunk_ids"],
                    cutoff=10_000,
                )["reciprocal_rank"]
                for item in answerable
            ]
        )
    }
    for cutoff in METRIC_CUTOFFS:
        metrics[f"recall_at_{cutoff}"] = _mean(
            [
                score_ranking(
                    item["rankings"][method],
                    item["relevant_chunk_ids"],
                    cutoff=cutoff,
                )["recall"]
                for item in answerable
            ]
        )
    unsupported = [item for item in results if not item["answerable"]]
    metrics["unsupported_nonempty_rate"] = _mean(
        [float(bool(item["rankings"][method])) for item in unsupported]
    )
    return metrics


def _ranking_records(results: Sequence[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": result.chunk.chunk_id,
            "source_path": result.chunk.source_path,
            "page_number": result.chunk.page_number,
            "section_heading": result.chunk.section_heading,
            "score": result.score,
        }
        for result in results
    ]


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _sweep_fusion(
    results: Sequence[Mapping[str, Any]],
    raw_rankings: Sequence[tuple[Sequence[RetrievalResult], Sequence[RetrievalResult]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    sweep: list[dict[str, Any]] = []
    for candidate_k in SWEEP_CANDIDATE_COUNTS:
        for rrf_k in SWEEP_RRF_VALUES:
            rankings = [
                [
                    result.chunk.chunk_id
                    for result in reciprocal_rank_fusion(
                        dense[:candidate_k],
                        bm25[:candidate_k],
                        rrf_k=rrf_k,
                        top_k=top_k,
                    )
                ]
                for dense, bm25 in raw_rankings
            ]
            evaluated = [
                {
                    **item,
                    "rankings": {"sweep": ranking},
                }
                for item, ranking in zip(results, rankings, strict=True)
            ]
            metrics = _aggregate_metrics(evaluated, "sweep")
            sweep.append(
                {
                    "candidate_k": candidate_k,
                    "rrf_k": rrf_k,
                    "metrics": metrics,
                }
            )
    best = min(
        sweep,
        key=lambda item: (
            -float(item["metrics"]["mrr"] or 0.0),
            -float(item["metrics"]["recall_at_3"] or 0.0),
            item["candidate_k"],
            item["rrf_k"],
        ),
    )
    return {
        "candidates": sweep,
        "best_by_mrr_then_recall_at_3": {
            "candidate_k": best["candidate_k"],
            "rrf_k": best["rrf_k"],
            "metrics": best["metrics"],
        },
    }


def _rechunk_corpus(
    chunks: Sequence[Chunk],
    settings: Settings,
) -> tuple[tuple[Chunk, ...], dict[str, tuple[tuple[str, int, int], ...]]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)

    rechunked: list[Chunk] = []
    source_spans: dict[str, tuple[tuple[str, int, int], ...]] = {}
    for document_chunks in grouped.values():
        segments: list[DocumentSegment] = []
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for chunk in document_chunks:
            text = clean_text(chunk.text)
            if not text:
                continue
            if segments:
                cursor += 2
            start = cursor
            cursor += len(text)
            spans.append((chunk.chunk_id, start, cursor))
            segments.append(
                DocumentSegment(
                    text=text,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                )
            )

        first = document_chunks[0]
        text_fingerprint = "\n\n".join(segment.text for segment in segments)
        document = DocumentRecord(
            document_id=first.document_id,
            source_path=first.source_path,
            file_name=first.file_name,
            file_type=first.file_type,
            file_hash=hashlib.sha256(text_fingerprint.encode("utf-8")).hexdigest(),
            segments=tuple(segments),
        )
        rechunked.extend(chunk_document(document, settings))
        source_spans[first.document_id] = tuple(spans)
    return tuple(rechunked), source_spans


def _relevant_rechunked_ids(
    relevant_chunk_ids: Sequence[str],
    chunks: Sequence[Chunk],
    source_spans: Mapping[str, Sequence[tuple[str, int, int]]],
) -> list[str]:
    relevant_spans = [
        (document_id, start, end)
        for document_id, spans in source_spans.items()
        for chunk_id, start, end in spans
        if chunk_id in relevant_chunk_ids
    ]
    return [
        chunk.chunk_id
        for chunk in chunks
        if any(
            chunk.document_id == document_id
            and max(
                0,
                min(
                    chunk.char_end if chunk.char_end is not None else len(chunk.text),
                    end,
                )
                - max(chunk.char_start if chunk.char_start is not None else 0, start),
            )
            >= (end - start + 1) // 2
            for document_id, start, end in relevant_spans
        )
    ]


def _sweep_chunking(
    dataset: Mapping[str, Any],
    original_chunks: Sequence[Chunk],
    *,
    embedder: EmbeddingEncoder,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="hybrid-rag-chunk-sweep-") as temporary_directory:
        root = Path(temporary_directory)
        (root / "documents").mkdir()
        for chunk_size, chunk_overlap in SWEEP_CHUNK_CONFIGURATIONS:
            corpus_settings = Settings(
                project_root=root,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                dense_top_k=10,
                bm25_top_k=10,
                fusion_top_k=5,
            )
            corpus, source_spans = _rechunk_corpus(original_chunks, corpus_settings)
            settings = Settings(
                project_root=root,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                dense_top_k=min(10, len(corpus)),
                bm25_top_k=min(10, len(corpus)),
                fusion_top_k=min(5, len(corpus)),
            )
            save_chunks(corpus, settings.processed_dir / "chunks.jsonl")
            dense_store = build_faiss_index(settings, embedder=embedder)
            sparse_store = build_bm25_index(settings)
            if dense_store.index_version != sparse_store.index_version:
                raise EvaluationError("Chunking sweep indexes have different versions.")
            retriever = HybridRetriever(settings, embedder=embedder)
            case_results: list[dict[str, Any]] = []
            for case in dataset["cases"]:
                response = retriever.retrieve(case["query"])
                relevant_ids = _relevant_rechunked_ids(
                    case["relevant_chunk_ids"],
                    corpus,
                    source_spans,
                )
                case_results.append(
                    {
                        "answerable": case["answerable"],
                        "relevant_chunk_ids": relevant_ids,
                        "rankings": {
                            "hybrid": [item.chunk.chunk_id for item in response.results]
                        },
                    }
                )
            candidates.append(
                {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunk_count": len(corpus),
                    "metrics": _aggregate_metrics(case_results, "hybrid"),
                }
            )

    best = min(
        candidates,
        key=lambda item: (
            -float(item["metrics"]["mrr"] or 0.0),
            -float(item["metrics"]["recall_at_3"] or 0.0),
            abs(item["chunk_size"] - 800) + abs(item["chunk_overlap"] - 120),
        ),
    )
    return {
        "candidates": candidates,
        "best_by_mrr_then_recall_at_3": {
            "chunk_size": best["chunk_size"],
            "chunk_overlap": best["chunk_overlap"],
            "metrics": best["metrics"],
        },
        "relevance_mapping": (
            "A re-chunked result is relevant when it overlaps at least half of a "
            "manually labeled source chunk."
        ),
    }


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    output_path: Path | None = None,
    embedder: EmbeddingEncoder | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset, chunks = _load_dataset(dataset_path)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    corpus_size = len(chunks)
    dense_k = min(10, corpus_size)
    fusion_k = min(5, corpus_size)
    rankings: list[dict[str, Any]] = []
    raw_rankings: list[tuple[Sequence[RetrievalResult], Sequence[RetrievalResult]]] = []
    build_times: dict[str, float] = {}
    query_times: dict[str, list[float]] = {"dense": [], "bm25": [], "hybrid": []}

    with TemporaryDirectory(prefix="hybrid-rag-evaluation-") as temporary_directory:
        root = Path(temporary_directory)
        (root / "documents").mkdir()
        settings = Settings(
            project_root=root,
            dense_top_k=dense_k,
            bm25_top_k=dense_k,
            fusion_top_k=fusion_k,
            rrf_k=60,
        )
        save_chunks(chunks, settings.processed_dir / "chunks.jsonl")
        model_load_started = time.perf_counter()
        encoder = embedder or SentenceTransformerEmbeddings(
            settings.embedding_model,
            device="cpu",
        )
        model_load_seconds = time.perf_counter() - model_load_started

        started = time.perf_counter()
        dense_store = build_faiss_index(settings, embedder=encoder)
        build_times["dense_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        sparse_store = build_bm25_index(settings)
        build_times["bm25_seconds"] = time.perf_counter() - started
        if dense_store.index_version != sparse_store.index_version:
            raise EvaluationError("Dense and BM25 evaluation indexes have different versions.")
        retriever = HybridRetriever(settings, embedder=encoder)

        for case in dataset["cases"]:
            query = case["query"]
            started = time.perf_counter()
            dense_results = retrieve_dense(
                query,
                dense_store,
                encoder,
                chunks,
                dense_k,
            )
            query_times["dense"].append(time.perf_counter() - started)

            started = time.perf_counter()
            sparse_results = retrieve_bm25(
                query,
                sparse_store,
                chunks,
                dense_k,
            )
            query_times["bm25"].append(time.perf_counter() - started)

            started = time.perf_counter()
            hybrid_response = retriever.retrieve(query, diagnostic=True)
            query_times["hybrid"].append(time.perf_counter() - started)
            raw_rankings.append((dense_results, sparse_results))
            dense_ids = [item.chunk.chunk_id for item in dense_results]
            sparse_ids = [item.chunk.chunk_id for item in sparse_results]
            hybrid_ids = [item.chunk.chunk_id for item in hybrid_response.results]
            relevant_ids = case["relevant_chunk_ids"]
            expected_page = case.get("expected_page")
            page_found = (
                None
                if expected_page is None
                else any(
                    item.chunk.chunk_id in relevant_ids
                    and item.chunk.page_number == expected_page
                    for item in hybrid_response.results
                )
            )
            rankings.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "query": query,
                    "answerable": case["answerable"],
                    "reference_answer": case["reference_answer"],
                    "relevant_chunk_ids": relevant_ids,
                    "rankings": {
                        "dense": dense_ids,
                        "bm25": sparse_ids,
                        "hybrid": hybrid_ids,
                    },
                    "ranked_sources": {
                        "dense": _ranking_records(dense_results),
                        "bm25": _ranking_records(sparse_results),
                        "hybrid": _ranking_records(hybrid_response.results),
                    },
                    "dense_bm25_disagreement": {
                        "different_order": dense_ids != sparse_ids,
                        "dense_only_top_k": [item for item in dense_ids if item not in sparse_ids],
                        "bm25_only_top_k": [item for item in sparse_ids if item not in dense_ids],
                    },
                    "page_citation_chunk_retrieved": page_found,
                    "latency_ms": {
                        "dense": query_times["dense"][-1] * 1000,
                        "bm25": query_times["bm25"][-1] * 1000,
                        "hybrid": query_times["hybrid"][-1] * 1000,
                    },
                }
            )

        metrics = {
            method: _aggregate_metrics(rankings, method)
            for method in ("dense", "bm25", "hybrid")
        }
        page_cases = [item for item in rankings if item["page_citation_chunk_retrieved"] is not None]
        page_retrieval_rate = _mean(
            [float(item["page_citation_chunk_retrieved"]) for item in page_cases]
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "platform": platform.platform(),
            "embedding_model": settings.embedding_model,
            "embedding_device": "cpu" if embedder is None else "custom (unspecified)",
            "index_version": dense_store.index_version,
            "index_versions_match": dense_store.index_version == sparse_store.index_version,
            "configuration": {
                "dense_top_k": settings.dense_top_k,
                "bm25_top_k": settings.bm25_top_k,
                "fusion_top_k": settings.fusion_top_k,
                "rrf_k": settings.rrf_k,
            },
            "timing_seconds": {
                "model_load": model_load_seconds,
                **build_times,
                "mean_query": {
                    method: _mean(values)
                    for method, values in query_times.items()
                },
            },
            "metrics": metrics,
            "page_citation_chunk_retrieval_rate": page_retrieval_rate,
            "rrf_parameter_sweep": _sweep_fusion(
                rankings,
                raw_rankings,
                top_k=fusion_k,
            ),
            "chunking_parameter_sweep": _sweep_chunking(
                dataset,
                chunks,
                embedder=encoder,
            ),
            "generation_quality": {
                "citation_validity": None,
                "faithfulness": None,
                "abstention_quality": None,
                "not_measured_reason": (
                    "This retrieval-only run makes no Gemini requests. Citation and answer "
                    "quality require a separately approved, authenticated generation evaluation."
                ),
            },
            "queries": rankings,
        }

    if output_path is not None:
        _write_json_atomic(Path(output_path).expanduser().resolve(), report)
    return report
