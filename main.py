#!/usr/bin/env python3
"""File Searcher — API entry point."""

from __future__ import annotations

import argparse
import os
import sys

from app.api import create_app, run_app
from app.logging_config import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic file searcher (API / web UI)",
    )
    parser.add_argument("folder", help="Root data directory to index")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--ocr-api-key", default=os.getenv("OCR_API_KEY"))
    parser.add_argument("--ocr-base-url", default=os.getenv("OCR_LLM_BASE_URL"))
    parser.add_argument("--ocr-model", default=os.getenv("OCR_LLM_MODEL"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "file_searcher"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--extensions", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--by-chunk", action="store_true")
    parser.add_argument("--snippet-chars", type=int, default=300)
    parser.add_argument("--lexical-weight", type=float, default=0.3)
    parser.add_argument(
        "--filebrowser-url",
        default=os.getenv("FILEBROWSER_URL", ""),
        help="FileBrowser base URL for file links",
    )

    # ── Logging ────────────────────────────────────────────────
    log_group = parser.add_argument_group("logging")
    log_group.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO). Set to DEBUG for detailed output.",
    )
    log_group.add_argument(
        "--verbose",
        action="store_true",
        default=os.getenv("VERBOSE", "").lower() in ("1", "true", "yes"),
        help=(
            "Enable verbose logging (DEBUG level). "
            "Shows per-file extraction timing, chunk detail, "
            "OCR page timings, embedding batch throughput, and ETA."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Initialise logging before anything else
    setup_logging(level=args.log_level, verbose=args.verbose)

    # Import here so logger is configured when app package code runs
    # (e.g. get_logger() calls)
    from app.logging_config import get_logger

    log = get_logger("main")

    if not args.api_key:
        log.critical("No API key provided. Set --api-key or OPENAI_API_KEY")
        sys.exit(1)
    if not args.model:
        log.critical("No embedding model specified. Set --model or EMBEDDING_MODEL")
        sys.exit(1)

    log.info("Starting File Searcher")
    log.info("  folder     : %s", args.folder)
    log.info("  model      : %s", args.model)
    log.info("  qdrant     : %s / %s", args.qdrant_url, args.qdrant_collection)
    log.info("  chunk_size : %d  overlap: %d", args.chunk_size, args.overlap)
    log.info("  batch_size : %d", args.batch_size)
    log.info("  log_level  : %s%s", args.log_level, " (verbose)" if args.verbose else "")

    app = create_app(
        folder=args.folder,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        ocr_api_key=args.ocr_api_key,
        ocr_base_url=args.ocr_base_url,
        ocr_model=args.ocr_model,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        host=args.host,
        port=args.port,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        extensions=args.extensions,
        top_k=args.top_k,
        by_chunk=args.by_chunk,
        snippet_chars=args.snippet_chars,
        lexical_weight=args.lexical_weight,
        filebrowser_url=args.filebrowser_url,
    )
    run_app(app)


if __name__ == "__main__":
    main()
