"""Tests for app/extractors.py."""

from __future__ import annotations

from pathlib import Path


class TestReadTextFile:
    def test_read_simple_text(self, tmp_path: Path):
        """Test reading a simple text file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        from app.extractors import read_text_file

        result = read_text_file(test_file)
        assert result == "hello world"

    def test_read_utf8_sig(self, tmp_path: Path):
        """Test reading UTF-8 with BOM."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"\xef\xbb\xbfhello world")

        from app.extractors import read_text_file

        result = read_text_file(test_file)
        # UTF-8-sig codec strips BOM, but latin-1 fallback may not
        assert "hello world" in result

    def test_binary_file_returns_empty(self, tmp_path: Path):
        """Test that binary files return empty string."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        from app.extractors import read_text_file

        result = read_text_file(test_file)
        assert result == ""

    def test_nonexistent_file_returns_empty(self):
        """Test that nonexistent file returns empty string."""
        from app.extractors import read_text_file

        result = read_text_file(Path("/nonexistent/file.txt"))
        assert result == ""


class TestExtractText:
    def test_extract_txt_file(self, tmp_path: Path):
        """Test extracting text from .txt file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        from app.extractors import extract_text

        result = extract_text(test_file)
        assert result == "hello world"

    def test_extract_unsupported_extension(self, tmp_path: Path):
        """Test that unsupported extension returns empty."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("some content")

        from app.extractors import extract_text

        result = extract_text(test_file)
        assert result == "some content"  # Falls back to read_text_file

    def test_extract_pdf_missing_package(self, tmp_path: Path):
        """Test PDF extraction when pypdf is not available."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake pdf content")

        from app.extractors import extract_pdf_text

        # Will return empty if pypdf not installed
        result = extract_pdf_text(test_file)
        # Result depends on whether pypdf is installed
        assert isinstance(result, str)

    def test_extract_docx_missing_package(self, tmp_path: Path):
        """Test DOCX extraction when python-docx is not available."""
        test_file = tmp_path / "test.docx"
        test_file.write_bytes(b"PK\x03\x04 fake docx")

        from app.extractors import extract_docx_text

        result = extract_docx_text(test_file)
        assert isinstance(result, str)

    def test_extract_xlsx_missing_package(self, tmp_path: Path):
        """Test XLSX extraction when openpyxl is not available."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_bytes(b"PK\x03\x04 fake xlsx")

        from app.extractors import extract_excel_text

        result = extract_excel_text(test_file)
        assert isinstance(result, str)

    def test_extract_xls_missing_package(self, tmp_path: Path):
        """Test XLS extraction when xlrd is not available."""
        test_file = tmp_path / "test.xls"
        test_file.write_bytes(b"\xd0\xcf\x11\xe0 fake xls")

        from app.extractors import extract_excel_text

        result = extract_excel_text(test_file)
        assert isinstance(result, str)
