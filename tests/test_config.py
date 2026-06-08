"""Tests for app/config.py."""

from __future__ import annotations

from app.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EXTENSIONS,
    DEFAULT_HOST,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_OVERLAP,
    DEFAULT_PORT,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_URL,
    DEFAULT_SNIPPET_CHARS,
    DEFAULT_TOP_K,
    SKIP_DIRNAMES,
)


class TestConfigConstants:
    def test_default_extensions(self):
        assert ".pdf" in DEFAULT_EXTENSIONS
        assert ".txt" in DEFAULT_EXTENSIONS
        assert ".docx" in DEFAULT_EXTENSIONS
        assert ".xlsx" in DEFAULT_EXTENSIONS
        assert ".xls" in DEFAULT_EXTENSIONS

    def test_default_chunk_size(self):
        assert DEFAULT_CHUNK_SIZE == 1200

    def test_default_overlap(self):
        assert DEFAULT_OVERLAP == 200

    def test_default_top_k(self):
        assert DEFAULT_TOP_K == 5

    def test_default_snippet_chars(self):
        assert DEFAULT_SNIPPET_CHARS == 300

    def test_default_lexical_weight(self):
        assert DEFAULT_LEXICAL_WEIGHT == 0.3

    def test_default_batch_size(self):
        assert DEFAULT_BATCH_SIZE == 32

    def test_default_host(self):
        assert DEFAULT_HOST == "0.0.0.0"

    def test_default_port(self):
        assert DEFAULT_PORT == 8000

    def test_default_qdrant_url(self):
        assert DEFAULT_QDRANT_URL == "http://localhost:6333"

    def test_default_qdrant_collection(self):
        assert DEFAULT_QDRANT_COLLECTION == "file_searcher"

    def test_skip_dirnames(self):
        assert "node_modules" in SKIP_DIRNAMES
        assert "__pycache__" in SKIP_DIRNAMES
        assert "venv" in SKIP_DIRNAMES
        assert ".venv" in SKIP_DIRNAMES
