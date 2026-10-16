"""Run the synthetic retrieval benchmark and save detailed rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hybrid_rag.evaluation import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_PATH,
    EvaluationError,
    run_evaluation,
)
from hybrid_rag.indexing.bm25_store import BM25StoreError
from hybrid_rag.indexing.embeddings import EmbeddingError
from hybrid_rag.indexing.faiss_store import FaissStoreError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare dense, BM25, and hybrid retrieval on the reviewed synthetic set."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Evaluation-set JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Output JSON path for rankings, metrics, and timings.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(args.dataset, output_path=args.output)
    except (BM25StoreError, EmbeddingError, EvaluationError, FaissStoreError, OSError, ValueError) as error:
        print(f"hybrid-rag evaluation failed: {error}", file=sys.stderr)
        return 1

    print("Method    Recall@1  Recall@3  Recall@5  MRR")
    for method in ("dense", "bm25", "hybrid"):
        metrics = report["metrics"][method]
        values = (
            metrics["recall_at_1"],
            metrics["recall_at_3"],
            metrics["recall_at_5"],
            metrics["mrr"],
        )
        formatted = [
            "-" if value is None else f"{value:.3f}"
            for value in values
        ]
        print(f"{method:<9} " + "  ".join(formatted))
    print(
        "Best RRF sweep: "
        + json.dumps(report["rrf_parameter_sweep"]["best_by_mrr_then_recall_at_3"])
    )
    print(
        "Best chunking sweep: "
        + json.dumps(report["chunking_parameter_sweep"]["best_by_mrr_then_recall_at_3"])
    )
    print(f"Saved detailed results to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
