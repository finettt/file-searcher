"""Constants and configuration."""

from __future__ import annotations

import os
from pathlib import Path

# ── File filters ──────────────────────────────────────────────

DEFAULT_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx", ".xlsx", ".xls"}

SPECIAL_FILENAMES: set[str] = set()  # Specify filenames if required. By default we index only documents.

SKIP_DIRNAMES: set[str] = {
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "@eaDir",
    "#recycle",
}

# ── Defaults ──────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_CHARS = 300
DEFAULT_BATCH_SIZE = 32
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# ── Qdrant ────────────────────────────────────────────────────

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "file_searcher")

# ── Reranker ──────────────────────────────────────────────────

DEFAULT_RERANKER_URL = os.getenv("RERANKER_BASE_URL", "http://localhost:8004/v1")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen3-Reranker-0.6B")
DEFAULT_RERANKER_TOP_CANDIDATES = 50  # BM25 pool size fed to cross-encoder

# ── Paths ─────────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
