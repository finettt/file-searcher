"""Tests for app/indexer_api.py — the indexer service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestIndexerApp:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.tmp_path = tmp_path

    def _make_app(self, mock_qdrant=None):
        if mock_qdrant is None:
            mock_qdrant = MagicMock()
            mock_qdrant.collection_exists.return_value = False
        with patch("app.cache.QdrantIndex", return_value=mock_qdrant):
            from app.indexer_api import create_indexer_app

            return create_indexer_app(
                folder=str(self.tmp_path),
                model="test-model",
                api_key="test-key",
            )

    def test_create_indexer_app_returns_fastapi(self):
        app = self._make_app()
        assert app is not None
        assert app.title == "File Searcher — Indexer"

    def test_health_no_index(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 503

    def test_health_with_index(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.get_info.return_value = {
            "points_count": 42,
            "model": "test-model",
        }
        app = self._make_app(mock_qdrant)
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["chunks"] == 42

    def test_status_no_index(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["stale"] is True
        assert data["building"] is False

    def test_search_empty_query(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10
        app = self._make_app(mock_qdrant)
        client = TestClient(app)
        response = client.post("/api/search", json={"query": ""})
        assert response.status_code == 400

    def test_search_triggers_auto_build(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        mock_qdrant.count.return_value = 0
        app = self._make_app(mock_qdrant)
        client = TestClient(app)
        response = client.post("/api/search", json={"query": "test"})
        assert response.status_code == 503

    def test_rebuild_returns_started(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.post("/api/rebuild")
        assert response.status_code == 200
        assert response.json()["status"] == "started"

    def test_rebuild_when_already_building(self):
        app = self._make_app()
        app.state.ctx.rebuilding = True
        app.state.ctx.rebuild_lock = True
        client = TestClient(app)
        response = client.post("/api/rebuild")
        assert response.status_code == 409

    def test_rebuild_selective_no_paths(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.post("/api/rebuild-selective", json={})
        assert response.status_code == 422

    def test_rebuild_selective_empty_paths(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.post("/api/rebuild-selective", json={"paths": []})
        assert response.status_code == 422

    def test_diff_no_index(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/api/diff")
        assert response.status_code == 200
        data = response.json()
        assert data["has_index"] is False

    def test_file_path_traversal_rejected(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/api/file", params={"path": "../../../etc/passwd"})
        assert response.status_code == 403

    def test_file_not_found(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/api/file", params={"path": "nonexistent.txt"})
        assert response.status_code == 404

    def test_search_validates_top_k_too_low(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10
        app = self._make_app(mock_qdrant)
        client = TestClient(app)
        response = client.post("/api/search", json={"query": "test", "top_k": 0})
        assert response.status_code == 422

    def test_search_validates_top_k_too_high(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10
        app = self._make_app(mock_qdrant)
        client = TestClient(app)
        response = client.post("/api/search", json={"query": "test", "top_k": 999})
        assert response.status_code == 422

    def test_no_root_html_endpoint(self):
        """Indexer should not serve the HTML template — no / route registered."""
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 404

    def test_state_has_indexer_state(self):
        """Verify app state is properly initialized."""
        app = self._make_app()
        ctx = app.state.ctx
        assert ctx.model == "test-model"
        assert ctx.api_key == "test-key"
        assert not ctx.rebuilding
        assert not ctx.rebuild_lock