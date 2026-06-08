"""Tests for app/indexer.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBuildClient:
    def test_build_client_with_base_url(self):
        """Test client builder with custom base URL."""
        mock_client = MagicMock()

        with patch("openai.OpenAI", return_value=mock_client) as mock_openai:
            from app.indexer import build_client

            build_client("test-key", "http://custom.example.com")
            mock_openai.assert_called_once_with(api_key="test-key", base_url="http://custom.example.com")

    def test_build_client_without_base_url(self):
        """Test client builder without base URL."""
        mock_client = MagicMock()

        with patch("openai.OpenAI", return_value=mock_client) as mock_openai:
            from app.indexer import build_client

            build_client("test-key")
            mock_openai.assert_called_once_with(api_key="test-key")


class TestEmbedTexts:
    def test_embed_texts_batches(self):
        """Test that embeddings are created in batches."""
        mock_client = MagicMock()

        # Mock response with proper index attributes for two calls
        def mock_create(model, input):
            mock_resp = MagicMock()
            mock_data = []
            for i, embedding in enumerate(input):
                mock_embedding = MagicMock()
                mock_embedding.embedding = [0.1 * (i + 1)] * 1536
                mock_embedding.index = i
                mock_data.append(mock_embedding)
            mock_resp.data = mock_data
            return mock_resp

        mock_client.embeddings.create.side_effect = mock_create

        with patch("openai.OpenAI", return_value=mock_client):
            from app.indexer import embed_texts

            texts = ["text one", "text two", "text three"]
            result = embed_texts(mock_client, "model", texts, batch_size=2)

            assert result.shape == (3, 1536)
            # Should be called twice: once for batch of 2, once for batch of 1
            assert mock_client.embeddings.create.call_count == 2


class TestL2Normalize:
    def test_l2_normalize_matrix(self):
        """Test matrix L2 normalization."""
        from app.indexer import l2_normalize_matrix

        matrix = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 1.0]])
        result = l2_normalize_matrix(matrix)

        # First row should be normalized to [0.6, 0.8]
        assert np.allclose(result[0], [0.6, 0.8])
        # Check that norms are 1
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0)

    def test_l2_normalize_vector(self):
        """Test vector L2 normalization."""
        from app.indexer import l2_normalize_vector

        vector = np.array([3.0, 4.0])
        result = l2_normalize_vector(vector)
        assert np.allclose(result, [0.6, 0.8])


class TestHelpers:
    def test_fmt_size(self):
        """Test byte formatting."""
        from app.indexer import _fmt_size

        assert _fmt_size(500) == "500B"
        assert "KB" in _fmt_size(1500)
        assert "MB" in _fmt_size(2 * 1024 * 1024)

    def test_fmt_time(self):
        """Test time formatting."""
        from app.indexer import _fmt_time

        assert _fmt_time(5.5) == "5.5s"
        assert "m" in _fmt_time(90)
        assert "h" in _fmt_time(3600)

    def test_fmt_eta(self):
        """Test ETA formatting."""
        from app.indexer import _fmt_eta

        assert _fmt_eta(0, 10, 0) == "calculating…"
        assert _fmt_eta(1, 10, 0) == "calculating…"
        result = _fmt_eta(5, 10, 10.0)
        assert "s" in result or "m" in result


class TestBuildCancelled:
    def test_exception_exists(self):
        """BuildCancelled exception is importable."""
        from app.indexer import BuildCancelled

        exc = BuildCancelled()
        assert isinstance(exc, Exception)

    def test_cancel_event_stops_build(self):
        """build_index raises BuildCancelled when cancel_event is set."""
        import threading

        from app.indexer import BuildCancelled, build_index

        cancel = threading.Event()
        cancel.set()

        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True

        root = Path(__file__).parent.parent
        gen = build_index(
            root,
            qdrant=mock_qdrant,
            model="test",
            api_key="test",
            cancel_event=cancel,
        )
        with pytest.raises(BuildCancelled):
            next(gen)
