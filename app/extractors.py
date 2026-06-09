"""Text extraction from PDF, DOCX, XLSX, XLS, and plain text files."""

from __future__ import annotations

import base64
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .logging_config import get_logger

log = get_logger(__name__)
_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Log a warning only once per key."""
    if key not in _WARNED_KEYS:
        log.warning(message)
        _WARNED_KEYS.add(key)


# ── Plain text ────────────────────────────────────────────────


def read_text_file(path: Path) -> str:
    """Read a text file, trying multiple encodings. Skip binary files."""
    t0 = time.monotonic()
    try:
        raw = path.read_bytes()
    except Exception as e:
        log.warning("read error %s: %s", path, e)
        return ""

    if b"\x00" in raw[:8192]:
        log.debug("binary skip %s", path)
        return ""

    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            text = raw.decode(enc)
            log.debug(
                "text extracted %s  encoding=%s  bytes=%d  chars=%d  time=%.2fs",
                path,
                enc,
                len(raw),
                len(text),
                time.monotonic() - t0,
            )
            return text
        except UnicodeDecodeError:
            continue

    text = raw.decode("utf-8", errors="ignore")
    log.debug(
        "text extracted %s  encoding=utf-8(ignore)  bytes=%d  chars=%d  time=%.2fs",
        path,
        len(raw),
        len(text),
        time.monotonic() - t0,
    )
    return text


# ── PDF ───────────────────────────────────────────────────────


def ocr_pdf_page(
    pdf_path: Path,
    page_number: int,
    client,
    model: str,
    *,
    on_page_start: Callable[[int, int], None] | None = None,
    on_page_done: Callable[[int, int], None] | None = None,
    total_pages: int = 0,
) -> str:
    """Render one PDF page to 300 DPI PNG, send to vision LLM, return text.

    *page_number* is 0-based.  Callbacks receive ``(page_1based, total_pages)``.
    """
    page_no = page_number + 1
    page_t0 = time.monotonic()
    try:
        from pdf2image import convert_from_path
    except ImportError:
        _warn_once("pdf2image", "Для OCR нужен пакет pdf2image + poppler-utils")
        return ""

    try:
        render_t0 = time.monotonic()
        images = convert_from_path(
            str(pdf_path),
            first_page=page_no,
            last_page=page_no,
            dpi=300,
        )
        render_dt = time.monotonic() - render_t0
        if not images:
            log.debug("ocr render produced no image: %s page=%d", pdf_path, page_no)
            return ""

        img = images[0]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name, "PNG")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                payload = f.read()
            b64 = base64.b64encode(payload).decode("utf-8")

            if on_page_start:
                try:
                    on_page_start(page_no, total_pages)
                except Exception:
                    pass

            req_t0 = time.monotonic()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract all text from this image, preserving layout. "
                            "Return only the extracted text, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ],
                    },
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            ocr_dt = time.monotonic() - req_t0
            text = response.choices[0].message.content or ""
            total_dt = time.monotonic() - page_t0

            log.info(
                "ocr page done  file=%s page=%d chars=%d render=%.2fs ocr=%.2fs total=%.2fs",
                pdf_path,
                page_no,
                len(text),
                render_dt,
                ocr_dt,
                total_dt,
            )
            log.debug(
                "ocr page detail  file=%s page=%d image=%dx%d png=%dB",
                pdf_path,
                page_no,
                img.width,
                img.height,
                len(payload),
            )

            if on_page_done:
                try:
                    on_page_done(page_no, total_pages)
                except Exception:
                    pass

            return text
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        log.warning("ocr error %s page %d: %s", pdf_path, page_no, e)
        return ""


def extract_pdf_text(
    path: Path,
    ocr_client=None,
    ocr_model: str | None = None,
    *,
    on_ocr_page_start: Callable[[int, int], None] | None = None,
    on_ocr_page_done: Callable[[int, int], None] | None = None,
) -> str:
    """Extract text from PDF. Fall back to OCR for pages without text layer.

    *on_ocr_page_start* / *on_ocr_page_done* receive ``(page_1based, total_pages)``
    and are called around every vision-LLM request so callers can emit progress events.
    """
    t0 = time.monotonic()
    try:
        from pypdf import PdfReader
    except ImportError:
        _warn_once("pypdf", "Для PDF нужен пакет pypdf: pip install pypdf")
        return ""

    try:
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        parts: list[str] = []
        text_pages = 0
        ocr_pages = 0
        empty_pages = 0

        for page_num, page in enumerate(reader.pages, 1):
            page_t0 = time.monotonic()
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            if not text.strip() or len(text.strip()) < 10:
                if ocr_client and ocr_model:
                    log.debug("pdf page fallback to ocr  file=%s page=%d", path, page_num)
                    ocr_text = ocr_pdf_page(
                        path,
                        page_num - 1,
                        ocr_client,
                        ocr_model,
                        on_page_start=on_ocr_page_start,
                        on_page_done=on_ocr_page_done,
                        total_pages=total_pages,
                    )
                    if ocr_text:
                        parts.append(ocr_text)
                        ocr_pages += 1
                        log.info(
                            "pdf page extracted via ocr  file=%s page=%d chars=%d total=%.2fs",
                            path,
                            page_num,
                            len(ocr_text),
                            time.monotonic() - page_t0,
                        )
                        continue
                parts.append("")
                empty_pages += 1
                log.debug(
                    "pdf page empty  file=%s page=%d total=%.2fs",
                    path,
                    page_num,
                    time.monotonic() - page_t0,
                )
            else:
                parts.append(text)
                text_pages += 1
                log.debug(
                    "pdf page extracted via text layer  file=%s page=%d chars=%d total=%.2fs",
                    path,
                    page_num,
                    len(text),
                    time.monotonic() - page_t0,
                )

        output = "\n".join(parts)
        log.info(
            "pdf extracted  file=%s pages=%d text_pages=%d ocr_pages=%d empty_pages=%d chars=%d total=%.2fs",
            path,
            len(reader.pages),
            text_pages,
            ocr_pages,
            empty_pages,
            len(output),
            time.monotonic() - t0,
        )
        return output
    except Exception as e:
        log.warning("pdf error %s: %s", path, e)
        return ""


# ── DOCX ──────────────────────────────────────────────────────


def extract_docx_text(path: Path) -> str:
    """Extract text and tables from a DOCX file."""
    t0 = time.monotonic()
    try:
        from docx import Document
    except ImportError:
        _warn_once("docx", "Для DOCX нужен пакет python-docx: pip install python-docx")
        return ""

    try:
        doc = Document(str(path))
        parts: list[str] = [p.text for p in doc.paragraphs]
        table_rows = 0
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    parts.append(row_text)
                    table_rows += 1

        text = "\n".join(parts)
        log.debug(
            "docx extracted %s paragraphs=%d table_rows=%d chars=%d total=%.2fs",
            path,
            len(doc.paragraphs),
            table_rows,
            len(text),
            time.monotonic() - t0,
        )
        return text
    except Exception as e:
        log.warning("docx error %s: %s", path, e)
        return ""


# ── Excel ─────────────────────────────────────────────────────


def extract_excel_text(path: Path) -> str:
    """Extract text from XLSX or XLS files."""
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return _extract_xlsx(path)
    if ext == ".xls":
        return _extract_xls(path)
    return ""


def _extract_xlsx(path: Path) -> str:
    t0 = time.monotonic()
    try:
        from openpyxl import load_workbook
    except ImportError:
        _warn_once("openpyxl", "Для XLSX нужен пакет openpyxl: pip install openpyxl")
        return ""

    try:
        wb = load_workbook(str(path), read_only=True, data_only=True)
        parts: list[str] = []
        row_count = 0
        for sheet in wb.worksheets:
            parts.append(f"## Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(v).strip() for v in row if v is not None and str(v).strip()]
                if cells:
                    parts.append(" | ".join(cells))
                    row_count += 1
            parts.append("")
        wb.close()

        text = "\n".join(parts)
        log.debug(
            "xlsx extracted %s sheets=%d rows=%d chars=%d total=%.2fs",
            path,
            len(parts),
            row_count,
            len(text),
            time.monotonic() - t0,
        )
        return text
    except Exception as e:
        log.warning("xlsx error %s: %s", path, e)
        return ""


def _extract_xls(path: Path) -> str:
    t0 = time.monotonic()
    try:
        import xlrd
    except ImportError:
        _warn_once("xlrd", "Для XLS нужен пакет xlrd: pip install xlrd")
        return ""

    try:
        wb = xlrd.open_workbook(str(path))
        parts: list[str] = []
        row_count = 0
        for sheet in wb.sheets():
            parts.append(f"## Sheet: {sheet.name}")
            for row_idx in range(sheet.nrows):
                cells = []
                for cell in sheet.row(row_idx):
                    val = cell.value
                    val = str(val).strip() if val is not None else ""
                    if val:
                        cells.append(val)
                if cells:
                    parts.append(" | ".join(cells))
                    row_count += 1
            parts.append("")

        text = "\n".join(parts)
        log.debug(
            "xls extracted %s sheets=%d rows=%d chars=%d total=%.2fs",
            path,
            wb.nsheets,
            row_count,
            len(text),
            time.monotonic() - t0,
        )
        return text
    except Exception as e:
        log.warning("xls error %s: %s", path, e)
        return ""


# ── Dispatcher ────────────────────────────────────────────────


def extract_text(
    path: Path,
    ocr_client=None,
    ocr_model: str | None = None,
    *,
    on_ocr_page_start: Callable[[int, int], None] | None = None,
    on_ocr_page_done: Callable[[int, int], None] | None = None,
) -> str:
    """Extract text from any supported file type.

    For PDFs with OCR fallback, *on_ocr_page_start* and *on_ocr_page_done*
    are called with ``(page_1based, total_pages)`` around each vision-LLM call.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_text(
            path,
            ocr_client=ocr_client,
            ocr_model=ocr_model,
            on_ocr_page_start=on_ocr_page_start,
            on_ocr_page_done=on_ocr_page_done,
        )
    if ext == ".docx":
        return extract_docx_text(path)
    if ext in (".xlsx", ".xls"):
        return extract_excel_text(path)
    return read_text_file(path)
