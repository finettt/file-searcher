"""Logging configuration for File Searcher.

Usage:
    from .logging_config import setup_logging, get_logger
    setup_logging(level="DEBUG", verbose=True)
    log = get_logger(__name__)

Log levels:
    DEBUG   — internal state, every chunk detail, timing breakdowns
    INFO    — normal operational messages (default)
    WARNING — recoverable issues (skipped files, fallback paths)
    ERROR   — failures that affect output
    CRITICAL— fatal startup errors

Verbose mode (--verbose):
    Sets level to DEBUG and enables per-chunk logging with:
    - file index / total, elapsed time, ETA
    - chunk count and character range per file
    - OCR timing per page
    - embedding batch throughput (chunks/s, tokens estimated)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

# ── Logger name hierarchy ─────────────────────────────────────
#   file_searcher              — root
#   file_searcher.indexer      — index build
#   file_searcher.extractor    — text extraction / OCR
#   file_searcher.embedder     — embedding batches
#   file_searcher.searcher     — query / search
#   file_searcher.api          — HTTP routes
#   file_searcher.cache        — Qdrant operations
#   file_searcher.progress     — progress events

ROOT_LOGGER = "file_searcher"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _VerboseFormatter(logging.Formatter):
    """Coloured formatter for verbose/DEBUG output in a terminal."""

    _GREY = "\033[2;37m"
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _BOLD_RED = "\033[1;31m"
    _RESET = "\033[0m"

    _LEVEL_COLOURS = {
        logging.DEBUG: _GREY,
        logging.INFO: _GREEN,
        logging.WARNING: _YELLOW,
        logging.ERROR: _RED,
        logging.CRITICAL: _BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self._LEVEL_COLOURS.get(record.levelno, self._RESET)
        level = f"{colour}{record.levelname:<8}{self._RESET}"
        # Strip the root prefix for cleaner display
        name = record.name.replace(ROOT_LOGGER + ".", "").replace(ROOT_LOGGER, "root")
        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        line = f"{self._GREY}{ts}{self._RESET} {level} {self._CYAN}{name}{self._RESET}  {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class _PlainFormatter(logging.Formatter):
    """Plain formatter for production / non-TTY output."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        name = record.name.replace(ROOT_LOGGER + ".", "")
        msg = record.getMessage()
        line = f"{ts} [{record.levelname}] {name}: {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(
    level: str | int = "INFO",
    verbose: bool = False,
) -> None:
    """Configure the file_searcher logging hierarchy.

    Call once at application startup (main.py).

    Args:
        level:   Standard log level name or integer. Overridden to DEBUG
                 when *verbose* is True.
        verbose: Enable DEBUG level + coloured per-chunk output.
    """
    if verbose:
        level = "DEBUG"

    numeric = logging.getLevelName(level) if isinstance(level, str) else level

    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(numeric)

    # Avoid duplicate handlers on re-init (e.g. in tests)
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric)

    use_colour = verbose or (sys.stderr.isatty() and os.getenv("NO_COLOR") is None)
    handler.setFormatter(_VerboseFormatter() if use_colour else _PlainFormatter())
    root.addHandler(handler)

    # Silence noisy third-party loggers at WARNING unless verbose
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if verbose else logging.WARNING)

    root.debug(
        "Logging initialised — level=%s verbose=%s tty=%s",
        logging.getLevelName(numeric),
        verbose,
        sys.stderr.isatty(),
    )


def get_logger(module_name: str) -> logging.Logger:
    """Return a logger scoped under the file_searcher hierarchy.

    Args:
        module_name: Typically __name__ from the calling module.
                     'app.indexer' → 'file_searcher.indexer'
    """
    # Map 'app.foo' → 'file_searcher.foo', bare names pass through
    if module_name.startswith("app."):
        child = module_name[len("app.") :]
    elif module_name == "app":
        child = "root"
    else:
        child = module_name
    return logging.getLogger(f"{ROOT_LOGGER}.{child}")
