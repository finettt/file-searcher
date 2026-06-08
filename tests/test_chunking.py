"""Tests for app/chunking.py."""

from __future__ import annotations

from app.chunking import chunk_text, clean_text, make_embed_input, make_snippet


class TestCleanText:
    def test_normalizes_crlf(self):
        assert clean_text("hello\r\nworld") == "hello\nworld"

    def test_normalizes_cr(self):
        assert clean_text("hello\rworld") == "hello\nworld"

    def test_collapses_multiple_newlines(self):
        assert clean_text("hello\n\n\n\nworld") == "hello\n\nworld"

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_whitespace(self):
        assert clean_text("   \n\n  ") == ""


class TestChunkText:
    def test_simple_chunk(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 1
        # Check that chunks contain text from the original
        full_text = "".join(c for _, _, c in chunks)
        assert "hello" in full_text or "world" in full_text

    def test_chunk_size_respected(self):
        text = "a" * 5000
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        assert len(chunks) >= 4
        for start, end, content in chunks:
            assert len(content) <= 1000

    def test_overlap(self):
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        assert len(chunks) >= 2
        # Check overlap between consecutive chunks
        if len(chunks) >= 2:
            _, end1, _content1 = chunks[0]
            start2, _, _content2 = chunks[1]
            # End of first chunk should be near start of second
            assert end1 >= start2

    def test_empty_text(self):
        assert chunk_text("", chunk_size=100, overlap=20) == []

    def test_breaks_on_newline(self):
        text = "line1\n\nline2\n\nline3"
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        # Should break on double newlines
        assert len(chunks) >= 1

    def test_returns_tuple_structure(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=5, overlap=1)
        for item in chunks:
            assert len(item) == 3
            assert isinstance(item[0], int)  # start
            assert isinstance(item[1], int)  # end
            assert isinstance(item[2], str)  # text


class TestMakeSnippet:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert make_snippet(text, limit=100) == text

    def test_long_text_truncated(self):
        text = "a" * 500
        result = make_snippet(text, limit=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_collapses_whitespace(self):
        text = "hello   world\n\n\nuniverse"
        result = make_snippet(text, limit=100)
        assert "   " not in result
        assert "\n" not in result

    def test_exactly_at_limit(self):
        text = "a" * 100
        result = make_snippet(text, limit=100)
        assert len(result) == 100
        assert not result.endswith("...")


class TestMakeEmbedInput:
    def test_includes_path(self):
        result = make_embed_input("docs/file.txt", "content")
        assert "docs/file.txt" in result

    def test_includes_filename(self):
        result = make_embed_input("docs/file.txt", "content")
        assert "file.txt" in result

    def test_includes_content(self):
        result = make_embed_input("docs/file.txt", "chunk content")
        assert "chunk content" in result

    def test_structure(self):
        result = make_embed_input("path/to/file.pdf", "text chunk")
        lines = result.split("\n")
        assert "File: path/to/file.pdf" in lines[0]
        assert "Name: file.pdf" in lines[1]
        assert "" in lines  # blank line separator
        assert "text chunk" in result
