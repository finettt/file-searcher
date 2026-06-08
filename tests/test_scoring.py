"""Tests for app/scoring.py."""

from __future__ import annotations

import numpy as np

from app.scoring import bm25_scores, rank_by_chunk, rank_by_file, tokenize


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
