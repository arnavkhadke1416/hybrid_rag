"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from hybrid_rag.config import ConfigurationError
from hybrid_rag.generation.gemini import (
    CitedSource,
    GeminiConfigurationError,
    GeminiGenerationError,
)
from hybrid_rag.indexing.embeddings import EmbeddingError
from hybrid_rag.retrieval.hybrid import HybridIndexError, HybridQueryError
from hybrid_rag.services.rag_service import RAGService, RAGServiceResponse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-rag",
        description="Ask questions grounded in documents using hybrid retrieval.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="Ask one question using existing indexes.")
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Display dense, BM25, and fused ranks for retrieved chunks.",
    )
    interactive_parser = commands.add_parser(
        "interactive",
        help="Ask multiple questions using the existing indexes.",
    )
    interactive_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Display dense, BM25, and fused ranks for retrieved chunks.",
    )
    return parser


def _format_page(source: CitedSource) -> str | None:
    if source.page_number is not None:
        return str(source.page_number)
    if source.page_range is not None:
        return f"{source.page_range[0]}-{source.page_range[1]}"
    return None


def _display_response(response: RAGServiceResponse, *, diagnostics: bool) -> None:
    print(response.answer.answer)
    if response.answer.sources:
        print("\nSources:")
        for source in response.answer.sources:
            details = [source.source_path, f"chunk {source.chunk_id}"]
            page = _format_page(source)
            if page is not None:
                details.append(f"page {page}")
            if source.section_heading:
                details.append(f"section {source.section_heading}")
            print(f"  {source.citation_label} " + " | ".join(details))

    if diagnostics and response.retrieval.diagnostics:
        print("\nRetrieval diagnostics:")
        for item in response.retrieval.diagnostics:
            dense_rank = "-" if item.dense_rank is None else str(item.dense_rank)
            bm25_rank = "-" if item.bm25_rank is None else str(item.bm25_rank)
            print(
                f"  dense={dense_rank} bm25={bm25_rank} fused={item.fused_rank} "
                f"source={item.source_path} chunk={item.chunk_id}"
            )


_CLI_ERRORS = (
    ConfigurationError,
    EmbeddingError,
    GeminiConfigurationError,
    GeminiGenerationError,
    HybridIndexError,
    HybridQueryError,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        service = RAGService()
        if args.command == "ask":
            response = service.ask(args.question, diagnostic=args.diagnostics)
            _display_response(response, diagnostics=args.diagnostics)
            return 0
        return _run_interactive(service, diagnostics=args.diagnostics)
    except _CLI_ERRORS as error:
        print(f"hybrid-rag: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nhybrid-rag: interrupted", file=sys.stderr)
        return 130


def _run_interactive(service: RAGService, *, diagnostics: bool) -> int:
    print("Interactive Hybrid RAG. Enter 'quit' or 'exit' to stop.")
    while True:
        try:
            question = input("Question> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.strip().lower() in {"quit", "exit"}:
            return 0
        if not question.strip():
            continue
        try:
            response = service.ask(question, diagnostic=diagnostics)
            _display_response(response, diagnostics=diagnostics)
        except _CLI_ERRORS as error:
            print(f"hybrid-rag: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
