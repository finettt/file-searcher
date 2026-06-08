#!/usr/bin/env python3
"""File Searcher — Web UI gateway entry point."""

from __future__ import annotations

import argparse
import os

from app.logging_config import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Web UI gateway: serves HTML and proxies API calls to the indexer",
    )
    parser.add_argument(
        "--indexer-url",
        default=os.getenv("INDEXER_URL", "http://localhost:8002"),
        help="Internal URL of the indexer service",
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8001")))

    # ── Logging ────────────────────────────────────────────────
    log_group = parser.add_argument_group("logging")
    log_group.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    log_group.add_argument(
        "--verbose",
        action="store_true",
        default=os.getenv("VERBOSE", "").lower() in ("1", "true", "yes"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level, verbose=args.verbose)

    from app.logging_config import get_logger

    log = get_logger("main_webui")

    log.info("Starting Web UI Gateway")
    log.info("  host        : %s", args.host)
    log.info("  port        : %d", args.port)
    log.info("  indexer_url : %s", args.indexer_url)
    log.info("  log_level   : %s%s", args.log_level, " (verbose)" if args.verbose else "")

    from app.webui_api import create_webui_app, run_webui

    app = create_webui_app(
        indexer_url=args.indexer_url,
        host=args.host,
        port=args.port,
    )
    run_webui(app)


if __name__ == "__main__":
    main()
