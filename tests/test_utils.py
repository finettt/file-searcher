"""Tests for app/utils.py."""

from __future__ import annotations

from pathlib import Path

from app import utils
from app.config import DEFAULT_EXTENSIONS


class TestParseExtensions:
    def test_none_returns_default(self):
        result = utils.parse_extensions(None)
        assert result == DEFAULT_EXTENSIONS

    def test_empty_string_returns_default(self):
        result = utils.parse_extensions("")
        assert result == DEFAULT_EXTENSIONS

    def test_single_extension(self):
        result = utils.parse_extensions("pdf")
        assert result == {".pdf"}

    def test_extension_with_dot(self):
        result = utils.parse_extensions(".pdf")
        assert result == {".pdf"}

    def test_multiple_extensions(self):
        result = utils.parse_extensions("pdf,txt,docx")
        assert result == {".pdf", ".txt", ".docx"}

    def test_mixed_dot_style(self):
        result = utils.parse_extensions(".pdf,txt,.docx")
        assert result == {".pdf", ".txt", ".docx"}

    def test_case_insensitive(self):
        result = utils.parse_extensions("PDF,TXT,DOCX")
        assert result == {".pdf", ".txt", ".docx"}

    def test_whitespace_handling(self):
        result = utils.parse_extensions("  pdf  ,  txt  ,  docx  ")
        assert result == {".pdf", ".txt", ".docx"}

    def test_empty_items_filtered(self):
        result = utils.parse_extensions("pdf,,txt,")
        assert result == {".pdf", ".txt"}


class TestSkipDir:
    def test_hidden_dir_skipped(self):
        assert utils.skip_dir(".git") is True

    def test_underscore_prefix_skipped(self):
        assert utils.skip_dir("@eaDir") is True

    def test_hash_prefix_skipped(self):
        assert utils.skip_dir("#recycle") is True

    def test_standard_skip_dirs(self):
        assert utils.skip_dir("node_modules") is True
        assert utils.skip_dir("__pycache__") is True
        assert utils.skip_dir("venv") is True
        assert utils.skip_dir(".venv") is True

    def test_normal_dir_not_skipped(self):
        assert utils.skip_dir("documents") is False
        assert utils.skip_dir("src") is False


class TestComputeFileHash:
    def test_returns_hash_for_existing_file(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = utils.compute_file_hash(test_file)
        assert len(result) == 64  # SHA-256 hex length
        assert result.isalnum()

    def test_different_files_different_hashes(self, tmp_path: Path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content a")
        file2.write_text("content b")
        hash1 = utils.compute_file_hash(file1)
        hash2 = utils.compute_file_hash(file2)
        assert hash1 != hash2

    def test_returns_empty_for_nonexistent_file(self):
        result = utils.compute_file_hash(Path("/nonexistent/file.txt"))
        assert result == ""


class TestComputeFsHash:
    def test_empty_dir_returns_hash(self, tmp_path: Path):
        result = utils.compute_fs_hash(tmp_path, {".txt"})
        assert len(result) == 64

    def test_single_file(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        result = utils.compute_fs_hash(tmp_path, {".txt"})
        assert len(result) == 64

    def test_ignores_non_matching_extensions(self, tmp_path: Path):
        txt_file = tmp_path / "test.txt"
        py_file = tmp_path / "test.py"
        txt_file.write_text("hello")
        py_file.write_text("print('hello')")
        result = utils.compute_fs_hash(tmp_path, {".txt"})
        # Only .txt file should affect hash
        assert len(result) == 64

    def test_different_content_different_hash(self, tmp_path: Path):
        file1 = tmp_path / "file.txt"
        file1.write_text("content a")
        hash1 = utils.compute_fs_hash(tmp_path, {".txt"})
        file1.write_text("content b")
        hash2 = utils.compute_fs_hash(tmp_path, {".txt"})
        assert hash1 != hash2


class TestComputeFsHashMap:
    def test_empty_dir_returns_empty_dict(self, tmp_path: Path):
        result = utils.compute_fs_hash_map(tmp_path, {".txt"})
        assert result == {}

    def test_single_file(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = utils.compute_fs_hash_map(tmp_path, {".txt"})
        assert "test.txt" in result
        assert len(result["test.txt"]) == 64

    def test_multiple_files(self, tmp_path: Path):
        for name in ["a.txt", "b.txt", "c.txt"]:
            (tmp_path / name).write_text(f"content {name}")
        result = utils.compute_fs_hash_map(tmp_path, {".txt"})
        assert len(result) == 3
        for name in ["a.txt", "b.txt", "c.txt"]:
            assert name in result

    def test_returns_relative_paths(self, tmp_path: Path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.txt"
        test_file.write_text("hello")
        result = utils.compute_fs_hash_map(tmp_path, {".txt"})
        assert "subdir/test.txt" in result
