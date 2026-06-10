"""Tests for app/reranker.py."""

# MAX_DOC_CHARS is 1800 — imported from the module under test

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestRerank:
    def _make_chunks(self, texts: list[str]) -> list[dict]:
        return [{"text": t, "path": f"file{i}.txt"} for i, t in enumerate(texts)]

    def test_returns_empty_for_empty_chunks(self):
        from app.reranker import rerank

        result = rerank(
            "query",
            [],
            base_url="http://localhost:8004/v1",
            model="test-model",
        )
        assert result == []

    def test_posts_to_correct_url(self):
        chunks = self._make_chunks(["doc one", "doc two"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            from app.reranker import rerank

            result = rerank(
                "my query",
                chunks,
                base_url="http://localhost:8004/v1",
                model="Qwen3-Reranker-0.6B",
            )

        # Verify URL formed correctly
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:8004/v1/rerank"

    def test_returns_sorted_by_score_descending(self):
        chunks = self._make_chunks(["low relevance", "high relevance", "medium relevance"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            from app.reranker import rerank

            result = rerank(
                "query",
                chunks,
                base_url="http://localhost:8004/v1",
            )

        assert len(result) == 3
        # Should be sorted descending: index 1 (0.9), index 2 (0.5), index 0 (0.1)
        assert result[0] == (1, 0.9)
        assert result[1] == (2, 0.5)
        assert result[2] == (0, 0.1)

    def test_result_tuples_have_correct_types(self):
        chunks = self._make_chunks(["hello world"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.75}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            from app.reranker import rerank

            result = rerank("query", chunks, base_url="http://localhost:8004/v1")

        assert len(result) == 1
        idx, score = result[0]
        assert isinstance(idx, int)
        assert isinstance(score, float)

    def test_sends_correct_payload(self):
        chunks = self._make_chunks(["text a", "text b"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.4},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            from app.reranker import rerank

            rerank(
                "my search query",
                chunks,
                base_url="http://localhost:8004/v1",
                model="test-reranker",
                top_n=2,
            )

        call_kwargs = mock_client.post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["query"] == "my search query"
        assert payload["model"] == "test-reranker"
        assert payload["documents"] == ["text a", "text b"]
        assert payload["top_n"] == 2

    def test_raises_on_http_error(self):
        import httpx

        chunks = self._make_chunks(["some text"])

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")

            from app.reranker import rerank

            with pytest.raises(httpx.ConnectError):
                rerank("query", chunks, base_url="http://localhost:8004/v1")

    def test_strips_trailing_slash_from_base_url(self):
        chunks = self._make_chunks(["doc"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.5}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            from app.reranker import rerank

            rerank("query", chunks, base_url="http://localhost:8004/v1/")

        called_url = mock_client.post.call_args[0][0]
        assert called_url == "http://localhost:8004/v1/rerank"

    def test_top_n_omitted_when_none(self):
        chunks = self._make_chunks(["doc"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.5}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            from app.reranker import rerank

            rerank("query", chunks, base_url="http://localhost:8004/v1", top_n=None)

        call_kwargs = mock_client.post.call_args[1]
        assert "top_n" not in call_kwargs["json"]

    def test_documents_truncated_to_max_chars(self):
        """Documents longer than MAX_DOC_CHARS are truncated before sending."""
        from app.reranker import MAX_DOC_CHARS, rerank

        long_text = "x" * (MAX_DOC_CHARS + 500)
        chunks = [{"text": long_text, "path": "big.txt"}]

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.6}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            rerank("query", chunks, base_url="http://localhost:8004/v1")

        sent_docs = mock_client.post.call_args[1]["json"]["documents"]
        assert len(sent_docs) == 1
        assert len(sent_docs[0]) == MAX_DOC_CHARS

    def test_short_documents_not_truncated(self):
        """Documents shorter than MAX_DOC_CHARS are sent as-is."""
        from app.reranker import MAX_DOC_CHARS, rerank

        short_text = "hello world"
        assert len(short_text) < MAX_DOC_CHARS
        chunks = [{"text": short_text, "path": "small.txt"}]

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.7}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_resp

            rerank("query", chunks, base_url="http://localhost:8004/v1")

        sent_docs = mock_client.post.call_args[1]["json"]["documents"]
        assert sent_docs[0] == short_text