"""Tests for app/searcher.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# Deterministic reranker mock: returns items in original order with descending scores
def _mock_rerank(query, chunks, *, base_url, model="Qwen3-Reranker-0.6B", top_n=None):
    n = len(chunks)
    effective = n if top_n is None else min(top_n, n)
    return [(i, 1.0 - i * 0.05) for i in range(effective)]


RERANKER_KWARGS = dict(
    reranker_base_url="http://localhost:8004/v1",
    reranker_model="Qwen3-Reranker-0.6B",
)


class TestDoSearch:
    def test_returns_empty_when_no_results(self):
        """Test that search returns empty list when no results found."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 0
        mock_qdrant.client.query_points.return_value = MagicMock(points=[])

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
        ):
            mock_build.return_value = MagicMock()

            from app.searcher import do_search

            result = do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                **RERANKER_KWARGS,
            )
            assert result == []

    def test_returns_results_with_correct_structure(self):
        """Test that results have the expected structure including rerank field."""
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

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
        ):
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
                top_k=5,
                **RERANKER_KWARGS,
            )

        assert len(result) >= 1
        item = result[0]
        assert "rank" in item
        assert "path" in item
        assert "score" in item
        assert "sem" in item
        assert "lex" in item
        assert "rerank" in item
        assert "chunk_id" in item
        assert "snippet" in item

    def test_respects_top_k(self):
        """Test that results are limited by top_k."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 100

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

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
        ):
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
                **RERANKER_KWARGS,
            )

        assert len(result) <= 3

    def test_by_chunk_ranking(self):
        """Test that by_chunk=True works without errors."""
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

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
        ):
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
                **RERANKER_KWARGS,
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

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
        ):
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
                **RERANKER_KWARGS,
            )

        assert len(result) == 0

    def test_reranker_called_with_correct_args(self):
        """Test that rerank_chunks is called with query and pool of chunks."""
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "path": "doc.txt",
            "abs_path": "/tmp/doc.txt",
            "chunk_id": 0,
            "start": 0,
            "end": 100,
            "text": "important document content",
        }
        mock_qdrant.client.query_points.return_value = MagicMock(points=[mock_point])

        captured_calls = []

        def capturing_rerank(query, chunks, *, base_url, model, top_n=None):
            captured_calls.append({"query": query, "chunks": chunks, "base_url": base_url, "model": model})
            return _mock_rerank(query, chunks, base_url=base_url, model=model, top_n=top_n)

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=capturing_rerank),
        ):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_resp

            from app.searcher import do_search

            do_search(
                mock_qdrant,
                "my search query",
                model="test-model",
                api_key="test-key",
                reranker_base_url="http://localhost:8004/v1",
                reranker_model="Qwen3-Reranker-0.6B",
            )

        assert len(captured_calls) == 1
        call = captured_calls[0]
        assert call["query"] == "my search query"
        assert call["base_url"] == "http://localhost:8004/v1"
        assert call["model"] == "Qwen3-Reranker-0.6B"
        assert isinstance(call["chunks"], list)

    def test_rerank_score_exposed_in_output(self):
        """Test that the rerank score is included in each result item."""
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
            "text": "some text",
        }
        mock_qdrant.client.query_points.return_value = MagicMock(points=[mock_point])

        fixed_score = 0.8765

        def fixed_rerank(query, chunks, *, base_url, model, top_n=None):
            return [(0, fixed_score)]

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=fixed_rerank),
        ):
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
                **RERANKER_KWARGS,
            )

        assert len(result) == 1
        assert result[0]["rerank"] == round(fixed_score, 4)
        assert result[0]["score"] == round(fixed_score, 4)
