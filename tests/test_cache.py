"""Tests for app/cache.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestQdrantIndex:
    def test_init_defaults(self):
        """Test QdrantIndex initialization with defaults."""
        mock_client = MagicMock()

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            assert qdrant.url == "http://localhost:6333"
            assert qdrant.collection == "file_searcher"

    def test_init_custom_params(self):
        """Test QdrantIndex with custom parameters."""
        mock_client = MagicMock()

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex(url="http://custom:9999", collection="custom_collection")
            assert qdrant.url == "http://custom:9999"
            assert qdrant.collection == "custom_collection"

    def test_client_property_lazy(self):
        """Test that client property creates connection lazily."""
        mock_client = MagicMock()
        mock_QdrantClient = MagicMock(return_value=mock_client)

        with patch("app.cache.QdrantClient", mock_QdrantClient):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            # Client should not be created yet
            assert qdrant._client is None
            # Access client property
            _ = qdrant.client
            # Now it should be created
            assert qdrant._client is not None
            mock_QdrantClient.assert_called_once_with(url="http://localhost:6333", timeout=120)

    def test_collection_exists_true(self):
        """Test collection_exists returns True when collection exists."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "file_searcher"
        mock_client.get_collections.return_value = MagicMock(collections=[mock_collection])

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            assert qdrant.collection_exists() is True

    def test_collection_exists_false(self):
        """Test collection_exists returns False when collection doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            assert qdrant.collection_exists() is False

    def test_count_no_collection(self):
        """Test count returns 0 when collection doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            assert qdrant.count() == 0

    def test_count_with_points(self):
        """Test count returns points count when collection exists."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "file_searcher"
        mock_client.get_collections.return_value = MagicMock(collections=[mock_collection])
        mock_client.get_collection.return_value = MagicMock(points_count=42)

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            assert qdrant.count() == 42

    def test_get_info_no_collection(self):
        """Test get_info returns empty dict when collection doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            assert qdrant.get_info() == {}

    def test_get_all_file_hashes_no_metadata(self):
        """Test get_all_file_hashes returns empty dict when no metadata."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[MagicMock(name="file_searcher")])
        mock_client.retrieve.return_value = []

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            assert qdrant.get_all_file_hashes() == {}

    def test_get_all_paths(self):
        """Test get_all_paths returns set of paths."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[MagicMock(name="file_searcher")])
        mock_client.scroll.return_value = (
            [
                MagicMock(payload={"path": "file1.txt"}),
                MagicMock(payload={"path": "file2.txt"}),
            ],
            None,  # next_offset
        )

        with patch("app.cache.QdrantClient", return_value=mock_client):
            from app.cache import QdrantIndex

            qdrant = QdrantIndex()
            paths = qdrant.get_all_paths()
            assert "file1.txt" in paths
            assert "file2.txt" in paths
