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
DEFAULT_LEXICAL_WEIGHT = 0.3
DEFAULT_BATCH_SIZE = 32
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# ── Qdrant ────────────────────────────────────────────────────

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "file_searcher")

# ── Paths ─────────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
