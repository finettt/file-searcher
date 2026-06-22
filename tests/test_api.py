"""Tests for app/api.py — backward-compatibility shim.

Verifies that the re-exported ``create_app`` factory (which delegates to
``create_indexer_app``) still works for existing callers.

Full coverage for the indexer service lives in:
  - tests/test_indexer_api.py
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestCreateApp:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        """Set up test fixtures."""
        self.tmp_path = tmp_path

    def test_create_app_returns_fastapi(self):
        """Test that create_app returns a FastAPI application."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )
            assert app is not None

    def test_health_no_index(self):
        """Test health endpoint when no index exists."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "no_index"
            assert data["chunks"] == 0
            assert data["model"] is None

    def test_status_no_index(self):
        """Test status endpoint when no index exists."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            response = client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert data["stale"] is True
            assert data["building"] is False

    def test_search_empty_query(self):
        """Test search endpoint with empty query."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            response = client.post("/api/search", json={"query": ""})
            assert response.status_code == 400

    def test_search_triggers_auto_build(self):
        """Test that search triggers auto-build when index is empty."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        mock_qdrant.count.return_value = 0

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            # Should return 503 when auto-build fails (no real Qdrant)
            response = client.post("/api/search", json={"query": "test"})
            assert response.status_code == 503

    def test_rebuild_returns_started(self):
        """Test rebuild endpoint returns started status."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            response = client.post("/api/rebuild")
            assert response.status_code == 200
            assert response.json()["status"] == "started"

    def test_rebuild_when_already_building(self):
        """Test rebuild returns conflict when already building."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            # Simulate building state via AppState (now IndexerState)
            app.state.ctx.rebuilding = True
            app.state.ctx.rebuild_lock = True
            client = TestClient(app)
            response = client.post("/api/rebuild")
            assert response.status_code == 409

    def test_rebuild_selective_no_paths(self):
        """Test selective rebuild returns 422 with no paths (Pydantic validation)."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            # Pydantic requires 'paths' with min_length=1, so {} returns 422
            response = client.post("/api/rebuild-selective", json={})
            assert response.status_code == 422

    def test_rebuild_selective_empty_paths(self):
        """Test selective rebuild returns 422 with empty paths list."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            response = client.post("/api/rebuild-selective", json={"paths": []})
            assert response.status_code == 422

    def test_diff_no_index(self):
        """Test diff endpoint when no index exists."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            response = client.get("/api/diff")
            assert response.status_code == 200
            data = response.json()
            assert data["has_index"] is False

    def test_file_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            response = client.get("/api/file", params={"path": "../../../etc/passwd"})
            assert response.status_code == 403

    def test_search_validates_top_k(self):
        """Test that search validates top_k bounds via Pydantic."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10

        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.api import create_app

            app = create_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

            client = TestClient(app)
            # top_k=0 should fail Pydantic validation (ge=1)
            response = client.post("/api/search", json={"query": "test", "top_k": 0})
            assert response.status_code == 422

            # top_k=999 should fail Pydantic validation (le=200)
            response = client.post("/api/search", json={"query": "test", "top_k": 999})
            assert response.status_code == 422
