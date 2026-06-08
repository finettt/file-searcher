"""Tests for app/webui_api.py — the web UI gateway service."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient


class TestWebUIApp:
    def _make_app(self, indexer_url="http://fake-indexer:8002"):
        from app.webui_api import create_webui_app

        return create_webui_app(indexer_url=indexer_url)

    def test_create_webui_app_returns_fastapi(self):
        app = self._make_app()
        assert app is not None
        assert app.title == "File Searcher — Web UI"

    def test_root_returns_html(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_export_csv_handled_locally(self):
        """Export is handled by the web UI itself, not proxied."""
        app = self._make_app()
        client = TestClient(app)
        response = client.post(
            "/api/export",
            json={
                "results": [{"rank": 1, "path": "test.txt", "score": 0.95}],
                "format": "csv",
            },
        )
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_export_json_handled_locally(self):
        app = self._make_app()
        client = TestClient(app)
        response = client.post(
            "/api/export",
            json={
                "results": [{"rank": 1, "path": "test.txt", "score": 0.95}],
                "format": "json",
            },
        )
        assert response.status_code == 200

    def test_proxy_health_success(self):
        """Test that /api/health is proxied to the indexer."""
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"status": "ok", "chunks": 10, "model": "test"},
            request=httpx.Request("GET", "http://fake-indexer:8002/api/health"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"

    def test_proxy_status_success(self):
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"stale": False, "building": False, "info": "ok", "filebrowser_url": ""},
            request=httpx.Request("GET", "http://fake-indexer:8002/api/status"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert data["stale"] is False

    def test_proxy_search(self):
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"results": [], "index_info": "test", "index_stale": False, "filebrowser_url": ""},
            request=httpx.Request("POST", "http://fake-indexer:8002/api/search"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.post("/api/search", json={"query": "hello"})
            assert response.status_code == 200

    def test_proxy_rebuild(self):
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"status": "started"},
            request=httpx.Request("POST", "http://fake-indexer:8002/api/rebuild"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.post("/api/rebuild")
            assert response.status_code == 200
            assert response.json()["status"] == "started"

    def test_proxy_diff(self):
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"has_index": False, "added": [], "removed": [], "modified": [], "total_current": 0},
            request=httpx.Request("GET", "http://fake-indexer:8002/api/diff"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.get("/api/diff")
            assert response.status_code == 200

    def test_proxy_file_with_query_string(self):
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"text": "hello world", "lines": 1, "truncated": False},
            request=httpx.Request("GET", "http://fake-indexer:8002/api/file?path=test.txt"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.get("/api/file", params={"path": "test.txt"})
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "hello world"

    def test_proxy_indexer_unavailable(self):
        """When indexer is down, proxy should return 503."""
        app = self._make_app()

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            client = TestClient(app)
            response = client.get("/api/status")
            assert response.status_code == 503
            assert "unavailable" in response.json()["detail"].lower()

    def test_proxy_indexer_timeout(self):
        """When indexer times out, proxy should return 504."""
        app = self._make_app()

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            client = TestClient(app)
            response = client.get("/api/status")
            assert response.status_code == 504
            assert "timed out" in response.json()["detail"].lower()

    def test_proxy_relays_error_status_codes(self):
        """Proxy should relay upstream error status codes unchanged."""
        app = self._make_app()

        mock_response = httpx.Response(
            409,
            json={"detail": "Rebuild already in progress"},
            request=httpx.Request("POST", "http://fake-indexer:8002/api/rebuild"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.post("/api/rebuild")
            assert response.status_code == 409

    def test_proxy_rebuild_selective(self):
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"status": "started", "paths": ["file.txt"]},
            request=httpx.Request("POST", "http://fake-indexer:8002/api/rebuild-selective"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.post("/api/rebuild-selective", json={"paths": ["file.txt"]})
            assert response.status_code == 200

    def test_proxy_cancel_rebuild(self):
        app = self._make_app()

        mock_response = httpx.Response(
            200,
            json={"status": "cancelling"},
            request=httpx.Request("POST", "http://fake-indexer:8002/api/cancel-rebuild"),
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            client = TestClient(app)
            response = client.post("/api/cancel-rebuild")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelling"
