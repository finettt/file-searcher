"""Tests for app/searcher.py."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx


# Deterministic reranker mock: returns items in original order with descending scores
def _mock_rerank(query, chunks, *, base_url, model="Qwen3-Reranker-0.6B", top_n=None):
    n = len(chunks)
    effective = n if top_n is None else min(top_n, n)
    return [(i, 1.0 - i * 0.05) for i in range(effective)]


RERANKER_KWARGS = dict(
    reranker_base_url="http://localhost:8004/v1",
    reranker_model="Qwen3-Reranker-0.6B",
)


@contextmanager
def _legacy_search_patches(mock_points=None, embedding_dim=1536):
    """Patch helpers for legacy (dense-only, no sparse vectors) search path.

    Forces _collection_has_sparse to return False so the BM25 + client-side
    RRF fallback path is exercised.
    """
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.count.return_value = 10 if mock_points is None else max(10, len(mock_points))
    mock_qdrant.client.query_points.return_value = MagicMock(points=mock_points or [])

    with (
        patch("app.searcher.build_client") as mock_build,
        patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
        patch("app.searcher._collection_has_sparse", return_value=False),
    ):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * embedding_dim)]
        mock_resp.data[0].index = 0
        mock_client.embeddings.create.return_value = mock_resp
        yield mock_qdrant


def _make_mock_point(path="test.txt", text="hello world test content", score=0.95):
    p = MagicMock()
    p.score = score
    p.payload = {
        "path": path,
        "abs_path": f"/tmp/{path}",
        "chunk_id": 0,
        "start": 0,
        "end": 100,
        "text": text,
    }
    return p


class TestDoSearch:
    def test_returns_empty_when_no_results(self):
        """Test that search returns empty list when no results found."""
        from app.searcher import do_search

        with _legacy_search_patches(mock_points=[]) as mock_qdrant:
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
        from app.searcher import do_search

        points = [_make_mock_point()]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
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
        from app.searcher import do_search

        points = [_make_mock_point(path=f"test{i}.txt", text=f"content {i}", score=0.95 - i * 0.01) for i in range(50)]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
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
        from app.searcher import do_search

        points = [_make_mock_point()]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
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
        from app.searcher import do_search

        points = [_make_mock_point(path="test.pdf", text="pdf content")]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
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
        from app.searcher import do_search

        captured_calls = []

        def capturing_rerank(query, chunks, *, base_url, model, top_n=None):
            captured_calls.append({"query": query, "chunks": chunks, "base_url": base_url, "model": model})
            return _mock_rerank(query, chunks, base_url=base_url, model=model, top_n=top_n)

        points = [_make_mock_point(path="doc.txt", text="important document content")]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
            with patch("app.searcher.rerank_chunks", side_effect=capturing_rerank):
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

    def test_score_is_rrf_and_rerank_is_crossencoder(self):
        """Test that 'score' contains RRF and 'rerank' contains cross-encoder score."""
        from app.searcher import do_search

        fixed_rerank_score = 0.8765

        def fixed_rerank(query, chunks, *, base_url, model, top_n=None):
            return [(0, fixed_rerank_score)]

        points = [_make_mock_point(text="some text")]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
            with patch("app.searcher.rerank_chunks", side_effect=fixed_rerank):
                result = do_search(
                    mock_qdrant,
                    "test query",
                    model="test-model",
                    api_key="test-key",
                    **RERANKER_KWARGS,
                )

        assert len(result) == 1
        # rerank = cross-encoder score
        assert result[0]["rerank"] == round(fixed_rerank_score, 4)
        # score = RRF fusion score, which differs from the cross-encoder score
        assert result[0]["score"] != result[0]["rerank"]
        assert result[0]["score"] > 0

    def test_fallback_to_rrf_when_reranker_unavailable(self):
        """Test that search falls back to RRF ordering when reranker is down."""
        from app.searcher import do_search

        def failing_rerank(query, chunks, *, base_url, model, top_n=None):
            raise httpx.ConnectError("Connection refused")

        points = [_make_mock_point(text="test content for fallback")]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
            with patch("app.searcher.rerank_chunks", side_effect=failing_rerank):
                result = do_search(
                    mock_qdrant,
                    "test query",
                    model="test-model",
                    api_key="test-key",
                    **RERANKER_KWARGS,
                )

        # Should still return results (via RRF fallback)
        assert len(result) == 1
        assert result[0]["path"] == "test.txt"
        # rerank sentinel is -1.0 (outside [0,1]) — distinguishable from a real score
        assert result[0]["rerank"] == -1.0
        # reranker_used flag lets consumers know the cross-encoder was not active
        assert result[0]["reranker_used"] is False
        # RRF score should still be present and positive
        assert result[0]["score"] > 0

    def test_no_top_n_passed_to_reranker(self):
        """Verify reranker receives no top_n so ext_filter doesn't lose results."""
        from app.searcher import do_search

        captured_kwargs = []

        def capturing_rerank(query, chunks, *, base_url, model, top_n=None):
            captured_kwargs.append({"top_n": top_n})
            return _mock_rerank(query, chunks, base_url=base_url, model=model, top_n=top_n)

        points = [_make_mock_point(text="some text")]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
            with patch("app.searcher.rerank_chunks", side_effect=capturing_rerank):
                do_search(
                    mock_qdrant,
                    "query",
                    model="test-model",
                    api_key="test-key",
                    top_k=3,
                    **RERANKER_KWARGS,
                )

        assert len(captured_kwargs) == 1
        # top_n should NOT be passed (None) so all pool results come back
        assert captured_kwargs[0]["top_n"] is None


class TestNativeHybridSearch:
    """Tests for the native Qdrant hybrid search path (sparse vectors present)."""

    def test_hybrid_search_uses_prefetch(self):
        """Verify that query_points is called with prefetch when sparse vectors exist."""
        from app.searcher import do_search
        from qdrant_client.http import models as qmodels

        points = [_make_mock_point()]
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10
        mock_qdrant.client.query_points.return_value = MagicMock(points=points)

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
            patch("app.searcher._collection_has_sparse", return_value=True),
        ):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_resp.data[0].index = 0
            mock_client.embeddings.create.return_value = mock_resp

            do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                **RERANKER_KWARGS,
            )

        call_kwargs = mock_qdrant.client.query_points.call_args[1]
        # In hybrid mode, prefetch should be used
        assert "prefetch" in call_kwargs
        assert call_kwargs["prefetch"] is not None
        assert len(call_kwargs["prefetch"]) == 2
        # The top-level query should be a FusionQuery with RRF
        assert isinstance(call_kwargs["query"], qmodels.FusionQuery)
        assert call_kwargs["query"].fusion == qmodels.Fusion.RRF

    def test_hybrid_search_returns_correct_structure(self):
        """Hybrid path results have same structure as legacy path."""
        from app.searcher import do_search

        points = [_make_mock_point()]
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10
        mock_qdrant.client.query_points.return_value = MagicMock(points=points)

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
            patch("app.searcher._collection_has_sparse", return_value=True),
        ):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_resp.data[0].index = 0
            mock_client.embeddings.create.return_value = mock_resp

            result = do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                **RERANKER_KWARGS,
            )

        assert len(result) >= 1
        item = result[0]
        for key in ("rank", "path", "score", "sem", "lex", "rerank", "chunk_id", "snippet"):
            assert key in item

    def test_hybrid_search_prefetch_uses_correct_vector_names(self):
        """Dense leg uses 'text' and sparse leg uses 'text-sparse'."""
        from app.searcher import do_search

        from app.config import DEFAULT_DENSE_VECTOR_NAME, DEFAULT_SPARSE_VECTOR_NAME

        points = [_make_mock_point()]
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.count.return_value = 10
        mock_qdrant.client.query_points.return_value = MagicMock(points=points)

        with (
            patch("app.searcher.build_client") as mock_build,
            patch("app.searcher.rerank_chunks", side_effect=_mock_rerank),
            patch("app.searcher._collection_has_sparse", return_value=True),
        ):
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_resp.data[0].index = 0
            mock_client.embeddings.create.return_value = mock_resp

            do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                **RERANKER_KWARGS,
            )

        prefetch = mock_qdrant.client.query_points.call_args[1]["prefetch"]
        vector_names = {p.using for p in prefetch}
        assert DEFAULT_DENSE_VECTOR_NAME in vector_names
        assert DEFAULT_SPARSE_VECTOR_NAME in vector_names

    def test_legacy_path_does_not_use_prefetch(self):
        """When no sparse vectors exist, query_points is called without prefetch."""
        from app.searcher import do_search

        points = [_make_mock_point()]
        with _legacy_search_patches(mock_points=points) as mock_qdrant:
            do_search(
                mock_qdrant,
                "test query",
                model="test-model",
                api_key="test-key",
                **RERANKER_KWARGS,
            )

        call_kwargs = mock_qdrant.client.query_points.call_args[1]
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None

    def test_collection_has_sparse_returns_false_on_error(self):
        """_collection_has_sparse returns False when get_collection raises."""
        from app.searcher import _collection_has_sparse

        mock_qdrant = MagicMock()
        mock_qdrant.client.get_collection.side_effect = Exception("connection error")
        assert _collection_has_sparse(mock_qdrant) is False

    def test_collection_has_sparse_returns_false_for_dense_only(self):
        """_collection_has_sparse returns False for dense-only collection."""
        from app.searcher import _collection_has_sparse
        from app.config import DEFAULT_SPARSE_VECTOR_NAME

        mock_qdrant = MagicMock()
        # Simulate a dense-only collection with no sparse_vectors attribute
        mock_info = MagicMock()
        mock_info.config.params.sparse_vectors = {}
        mock_qdrant.client.get_collection.return_value = mock_info
        assert _collection_has_sparse(mock_qdrant) is False

    def test_collection_has_sparse_returns_true_when_sparse_present(self):
        """_collection_has_sparse returns True when sparse vector name is in config."""
        from app.searcher import _collection_has_sparse
        from app.config import DEFAULT_SPARSE_VECTOR_NAME

        mock_qdrant = MagicMock()
        mock_info = MagicMock()
        mock_info.config.params.sparse_vectors = {DEFAULT_SPARSE_VECTOR_NAME: MagicMock()}
        mock_qdrant.client.get_collection.return_value = mock_info
        assert _collection_has_sparse(mock_qdrant) is True
