"""Reprocess source documents and rebuild all persisted RAG data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ingest_documents import main as ingest_main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Force reprocess source documents and rebuild all RAG data."
    )
    parser.parse_args(argv)
    return ingest_main(["--force"])


if __name__ == "__main__":
    raise SystemExit(main())
