"""Tests for app/logging_config.py."""

from __future__ import annotations

from app.logging_config import get_logger, setup_logging


class TestLoggingConfig:
    def test_setup_logging_default(self):
        """Test logging setup with default level."""
        setup_logging()
        # Should not raise

    def test_setup_logging_debug(self):
        """Test logging setup with DEBUG level."""
        setup_logging(level="DEBUG")
        # Should not raise

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "file_searcher.test_module"

    def test_get_logger_root(self):
        """Test getting logger with __name__."""
        logger = get_logger(__name__)
        assert logger is not None
