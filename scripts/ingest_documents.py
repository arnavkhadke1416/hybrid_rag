"""Discover documents, persist chunks, and build both retrieval indexes."""

from __future__ import annotations

import sys
import argparse

from hybrid_rag.config import ConfigurationError, Settings
from hybrid_rag.indexing.bm25_store import BM25StoreError
from hybrid_rag.indexing.embeddings import EmbeddingError
from hybrid_rag.indexing.faiss_store import FaissStoreError
from hybrid_rag.ingestion.automatic import IngestionTransactionError, run_ingestion
from hybrid_rag.ingestion.loaders import DocumentLoadError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover documents, detect changes, and synchronize both indexes."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess every source document and rebuild both indexes.",
    )
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        summary = run_ingestion(settings, force=args.force)
    except (
        BM25StoreError,
        ConfigurationError,
        DocumentLoadError,
        EmbeddingError,
        FaissStoreError,
        IngestionTransactionError,
        OSError,
        ValueError,
    ) as error:
        print(f"hybrid-rag ingestion failed: {error}", file=sys.stderr)
        return 1

    for failure in summary.failures:
        print(
            f"Failed to load {failure.source_path}: {failure.error}",
            file=sys.stderr,
        )
    print(
        f"{len(summary.unchanged)} unchanged, {len(summary.added)} added, "
        f"{len(summary.changed)} changed, {len(summary.removed)} removed, "
        f"{len(summary.failures)} failed; {len(summary.chunks)} chunks indexed."
    )
    if summary.rebuilt:
        print(
            f"FAISS and BM25 indexes rebuilt (index version {summary.index_version[:12]}; "
            f"{summary.embedded_chunks} chunk embeddings generated)."
        )
    else:
        print(f"FAISS and BM25 indexes already current ({summary.index_version[:12]}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
