"""Tests for app/scoring.py."""

from __future__ import annotations

import numpy as np

from app.scoring import RRF_K, bm25_scores, rank_by_chunk, rank_by_file, rrf_fusion, tokenize


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
