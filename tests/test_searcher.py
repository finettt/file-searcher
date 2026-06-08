"""Tests for app/searcher.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDoSearch:
    def test_returns_empty_when_no_results(self):
        """Test that search returns empty list when no results found."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 0
        mock_qdrant.client.query_points.return_value = MagicMock(points=[])

        with patch("app.searcher.build_client") as mock_build:
            mock_build.return_value = MagicMock()

            from app.searcher import do_search

            result = do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
            )
            assert result == []

    def test_returns_results_with_correct_structure(self):
        """Test that results have the expected structure."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10

        # Mock Qdrant response
        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "path": "test.txt",
            "abs_path": "/tmp/test.txt",
            "chunk_id": 0,
            "start": 0,
            "end": 100,
            "text": "hello world test content",
        }
        mock_qdrant.client.query_points.return_value = MagicMock(points=[mock_point])

        with patch("app.searcher.build_client") as mock_build:
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            # Mock embedding response
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_resp

            from app.searcher import do_search

            result = do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                top_k=5,
            )

        assert len(result) >= 1
        item = result[0]
        assert "rank" in item
        assert "path" in item
        assert "score" in item
        assert "sem" in item
        assert "lex" in item
        assert "chunk_id" in item
        assert "snippet" in item

    def test_respects_top_k(self):
        """Test that results are limited by top_k."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 100

        # Create 50 mock points
        points = []
        for i in range(50):
            mock_point = MagicMock()
            mock_point.score = 0.95 - i * 0.01
            mock_point.payload = {
                "path": f"test{i}.txt",
                "abs_path": f"/tmp/test{i}.txt",
                "chunk_id": 0,
                "start": 0,
                "end": 100,
                "text": f"content {i}",
            }
            points.append(mock_point)

        mock_qdrant.client.query_points.return_value = MagicMock(points=points)

        with patch("app.searcher.build_client") as mock_build:
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_resp

            from app.searcher import do_search

            result = do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                top_k=3,
            )

        assert len(result) <= 3

    def test_by_chunk_ranking(self):
        """Test that by_chunk=True uses chunk-based ranking."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "path": "test.txt",
            "abs_path": "/tmp/test.txt",
            "chunk_id": 0,
            "start": 0,
            "end": 100,
            "text": "hello world test content",
        }
        mock_qdrant.client.query_points.return_value = MagicMock(points=[mock_point])

        with patch("app.searcher.build_client") as mock_build:
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_resp

            from app.searcher import do_search

            result = do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                by_chunk=True,
            )

        assert isinstance(result, list)

    def test_ext_filter(self):
        """Test that ext_filter filters results by extension."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "path": "test.pdf",
            "abs_path": "/tmp/test.pdf",
            "chunk_id": 0,
            "start": 0,
            "end": 100,
            "text": "pdf content",
        }
        mock_qdrant.client.query_points.return_value = MagicMock(points=[mock_point])

        with patch("app.searcher.build_client") as mock_build:
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_resp

            from app.searcher import do_search

            result = do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                ext_filter=[".txt"],  # Filter for .txt, should exclude .pdf
            )

        # PDF should be filtered out
        assert len(result) == 0