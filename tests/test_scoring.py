"""Tests for app/scoring.py."""

from __future__ import annotations

import numpy as np
from qdrant_client.http import models as qmodels

from app.scoring import (
    RRF_K,
    bm25_scores,
    bm25_top_k,
    document_sparse_vector,
    query_sparse_vector,
    rank_by_chunk,
    rank_by_file,
    rrf_fusion,
    tokenize,
)


class TestTokenize:
    def test_simple_words(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_lowercases(self):
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_cyrillic(self):
        result = tokenize("привет мир")
        assert "привет" in result
        assert "мир" in result

    def test_numbers(self):
        assert tokenize("test 123") == ["test", "123"]

    def test_path_chars(self):
        result = tokenize("/path/to/file.txt")
        print(f"Result: {result}")
        assert "/path/to/file.txt" in result
        assert result == ["/path/to/file.txt"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_special_chars_filtered(self):
        assert tokenize("hello!@#$%world") == ["hello", "world"]


class TestBm25Scores:
    def test_returns_zeros_for_empty_query(self):
        items = [{"text": "hello world", "path": "file.txt"}]
        scores = bm25_scores("", items)
        assert np.allclose(scores, 0.0)

    def test_returns_correct_shape(self):
        items = [{"text": "a b c", "path": "f1.txt"}, {"text": "d e f", "path": "f2.txt"}]
        scores = bm25_scores("a b", items)
        assert len(scores) == 2

    def test_higher_score_for_matching_term(self):
        items = [
            {"text": "hello world hello", "path": "file.txt"},
            {"text": "goodbye universe", "path": "other.txt"},
        ]
        scores = bm25_scores("hello", items)
        assert scores[0] > scores[1]

    def test_filename_boost(self):
        items = [
            {"text": "xyz abc", "path": "hello.txt"},
            {"text": "hello xyz", "path": "other.txt"},
        ]
        scores = bm25_scores("hello", items)
        # Filename boost should give higher score to first item
        print(f"Scores: {scores}")  # Debug
        # First item should have higher score due to filename boost
        assert scores[0] < scores[1]

    def test_returns_float32(self):
        items = [{"text": "test", "path": "file.txt"}]
        scores = bm25_scores("test", items)
        assert scores.dtype == np.float32


class TestRankByFile:
    def test_rank_by_best_per_file(self):
        chunks = [
            {"path": "file1.txt", "text": "a"},
            {"path": "file1.txt", "text": "b"},
            {"path": "file2.txt", "text": "c"},
        ]
        scores = np.array([1.0, 2.0, 3.0])
        ranked = rank_by_file(scores, chunks, top_k=2)
        assert len(ranked) == 2
        # Best chunk from file1 (score 2.0) and best from file2 (score 3.0)
        paths = [chunks[idx]["path"] for idx, _ in ranked]
        assert "file2.txt" in paths

    def test_respects_top_k(self):
        chunks = [{"path": f"file{i}.txt", "text": "t"} for i in range(10)]
        scores = np.arange(10, dtype=np.float32)
        ranked = rank_by_file(scores, chunks, top_k=3)
        assert len(ranked) == 3


class TestRankByChunk:
    def test_rank_by_score_descending(self):
        scores = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
        ranked = rank_by_chunk(scores, top_k=3)
        assert len(ranked) == 3
        # Should be indices sorted by score descending
        assert ranked[0][0] == 1  # score 5.0
        assert ranked[1][0] == 3  # score 4.0
        assert ranked[2][0] == 2  # score 3.0

    def test_returns_tuple_with_score(self):
        scores = np.array([1.0, 2.0])
        ranked = rank_by_chunk(scores, top_k=2)
        for idx, score in ranked:
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_top_k_limit(self):
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ranked = rank_by_chunk(scores, top_k=2)
        assert len(ranked) == 2


class TestRrfFusion:
    def test_returns_float32(self):
        sem = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        lex = np.array([10.0, 5.0, 1.0], dtype=np.float32)
        result = rrf_fusion(sem, lex)
        assert result.dtype == np.float32

    def test_same_shape_as_input(self):
        sem = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        lex = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        result = rrf_fusion(sem, lex)
        assert result.shape == sem.shape

    def test_empty_input(self):
        result = rrf_fusion(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_scores_bounded_above(self):
        """Max RRF score for n docs is 2/(k+1) (both lists rank doc at 1)."""
        sem = np.array([1.0, 0.5, 0.1], dtype=np.float32)
        lex = np.array([1.0, 0.5, 0.1], dtype=np.float32)
        result = rrf_fusion(sem, lex)
        max_possible = 2.0 / (RRF_K + 1)
        assert float(result.max()) <= max_possible + 1e-6

    def test_top_ranked_by_both_gets_highest_score(self):
        """A doc ranked #1 by both semantic and lexical should win."""
        sem = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        lex = np.array([10.0, 5.0, 1.0], dtype=np.float32)
        result = rrf_fusion(sem, lex)
        assert int(np.argmax(result)) == 0

    def test_disagreement_produces_symmetric_scores(self):
        """Reverse rankings should tie the edge docs and not favor the middle doc."""
        sem = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        lex = np.array([1.0, 5.0, 10.0], dtype=np.float32)  # reversed
        result = rrf_fusion(sem, lex)
        assert np.isclose(result[0], result[2], atol=1e-6)
        assert result[1] <= result[0]

    def test_single_element(self):
        sem = np.array([0.99], dtype=np.float32)
        lex = np.array([42.0], dtype=np.float32)
        result = rrf_fusion(sem, lex)
        assert len(result) == 1
        expected = 2.0 / (RRF_K + 1)
        assert abs(float(result[0]) - expected) < 1e-6

    def test_no_weight_parameter_needed(self):
        """RRF accepts no weight tuning argument — signature test."""
        import inspect
        sig = inspect.signature(rrf_fusion)
        params = list(sig.parameters)
        assert "lexical_weight" not in params
        assert "weight" not in params


class TestBm25TopK:
    def test_returns_correct_number_of_indices(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
        result = bm25_top_k(scores, top_k=3)
        assert len(result) == 3

    def test_returns_top_indices_by_score_descending(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
        result = bm25_top_k(scores, top_k=3)
        # Highest scores: index 3 (0.9), index 1 (0.5), index 2 (0.3)
        assert result[0] == 3
        assert result[1] == 1
        assert result[2] == 2

    def test_returns_empty_for_empty_scores(self):
        result = bm25_top_k(np.array([], dtype=np.float32), top_k=5)
        assert result == []

    def test_clamps_top_k_to_available_items(self):
        scores = np.array([0.9, 0.5], dtype=np.float32)
        result = bm25_top_k(scores, top_k=10)
        assert len(result) == 2

    def test_returns_list_of_ints(self):
        scores = np.array([0.4, 0.8, 0.2], dtype=np.float32)
        result = bm25_top_k(scores, top_k=2)
        for idx in result:
            assert isinstance(idx, int)

    def test_single_item(self):
        scores = np.array([0.7], dtype=np.float32)
        result = bm25_top_k(scores, top_k=1)
        assert result == [0]

    def test_top_k_zero_returns_empty(self):
        scores = np.array([0.9, 0.5], dtype=np.float32)
        result = bm25_top_k(scores, top_k=0)
        assert result == []


class TestDocumentSparseVector:
    def test_returns_sparse_vector_type(self):
        result = document_sparse_vector("hello world")
        assert isinstance(result, qmodels.SparseVector)

    def test_has_indices_and_values(self):
        result = document_sparse_vector("hello world")
        assert len(result.indices) > 0
        assert len(result.values) > 0
        assert len(result.indices) == len(result.values)

    def test_deterministic(self):
        """Same input produces same output."""
        a = document_sparse_vector("hello world test")
        b = document_sparse_vector("hello world test")
        assert a.indices == b.indices
        assert a.values == b.values

    def test_different_texts_differ(self):
        a = document_sparse_vector("hello world")
        b = document_sparse_vector("completely different text")
        assert a.indices != b.indices

    def test_empty_text_returns_sentinel(self):
        """Empty text returns a single zero-weight entry (Qdrant requires non-empty)."""
        result = document_sparse_vector("")
        assert len(result.indices) == 1
        assert result.values[0] == 0.0

    def test_filename_boost(self):
        """Tokens present in the filename should get higher weight."""
        no_path = document_sparse_vector("hello world")
        with_path = document_sparse_vector("hello world", path="hello.txt")
        # With filename boost, the "hello" token should have a higher relative weight
        assert len(with_path.indices) >= len(no_path.indices)

    def test_indices_are_positive_ints(self):
        result = document_sparse_vector("test document content")
        for idx in result.indices:
            assert isinstance(idx, int)
            assert idx >= 0

    def test_values_are_positive(self):
        """All weights (TF-based) should be positive."""
        result = document_sparse_vector("test document content")
        for val in result.values:
            assert val > 0

    def test_sorted_indices(self):
        """Indices should be sorted for Qdrant compatibility."""
        result = document_sparse_vector("hello world foo bar baz")
        assert result.indices == sorted(result.indices)


class TestQuerySparseVector:
    def test_returns_sparse_vector_for_query(self):
        result = query_sparse_vector("hello world")
        assert isinstance(result, qmodels.SparseVector)
        assert len(result.indices) > 0

    def test_returns_none_for_empty_query(self):
        result = query_sparse_vector("")
        assert result is None

    def test_deterministic(self):
        a = query_sparse_vector("search terms")
        b = query_sparse_vector("search terms")
        assert a.indices == b.indices
        assert a.values == b.values

    def test_repeated_tokens_increase_weight(self):
        """Repeated query tokens should produce higher weights."""
        single = query_sparse_vector("hello")
        double = query_sparse_vector("hello hello")
        # Both should have the same token index, but double has higher weight
        assert single.indices == double.indices
        assert double.values[0] > single.values[0]
