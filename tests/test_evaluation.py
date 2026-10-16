from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hybrid_rag.config import DEFAULT_EMBEDDING_MODEL
from hybrid_rag.evaluation import DEFAULT_DATASET_PATH, run_evaluation, score_ranking


class FakeEvaluationEmbedder:
    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = 5

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "backup" in normalized or "aes-256" in normalized or "cipher" in normalized:
                vectors.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif any(token in normalized for token in ("account", "password", "sign in")):
                vectors.append([0.0, 1.0, 0.0, 0.0, 0.0])
            elif any(token in normalized for token in ("rotation", "recovery key", "90 days")):
                vectors.append([0.0, 0.0, 1.0, 0.0, 0.0])
            elif any(token in normalized for token in ("emergency", "facilities", "555-0100")):
                vectors.append([0.0, 0.0, 0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_score_ranking_reports_recall_and_reciprocal_rank() -> None:
    assert score_ranking(
        ["noise", "relevant-b", "relevant-a"],
        ["relevant-a", "relevant-b"],
        cutoff=2,
    ) == {"recall": 0.5, "reciprocal_rank": 0.5}
    assert score_ranking(["noise"], [], cutoff=3) == {
        "recall": 0.0,
        "reciprocal_rank": 0.0,
    }
    with pytest.raises(ValueError, match="positive integer"):
        score_ranking([], [], cutoff=0)


def test_benchmark_compares_methods_saves_results_and_inspects_disagreements(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results.json"
    report = run_evaluation(
        DEFAULT_DATASET_PATH,
        output_path=output_path,
        embedder=FakeEvaluationEmbedder(),
    )

    assert report["index_versions_match"] is True
    assert report["embedding_device"] == "custom (unspecified)"
    assert set(report["metrics"]) == {"dense", "bm25", "hybrid"}
    assert len(report["queries"]) == 5
    assert len(report["rrf_parameter_sweep"]["candidates"]) == 9
    assert len(report["chunking_parameter_sweep"]["candidates"]) == 4
    assert report["chunking_parameter_sweep"]["relevance_mapping"]
    assert report["generation_quality"]["citation_validity"] is None
    page_case = next(
        item
        for item in report["queries"]
        if item["category"] == "page_citation"
    )
    assert page_case["page_citation_chunk_retrieved"] is True
    assert all("dense_bm25_disagreement" in item for item in report["queries"])
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8"))["dataset_sha256"] == report[
        "dataset_sha256"
    ]
