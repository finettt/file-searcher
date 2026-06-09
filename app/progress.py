"""Progress tracking with ETA calculation for index builds."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressEvent:
    """Single progress update pushed to WebSocket clients."""

    phase: str  # "discovering" | "extracting" | "embedding" | "saving" | "done" | "error"
    current_file: str = ""  # relative path
    current: int = 0  # files done
    total: int = 0  # files total
    chunks: int = 0  # chunks created
    skipped: int = 0  # files skipped
    # Timing
    elapsed: float = 0.0  # seconds since start
    eta: float = 0.0  # estimated seconds remaining
    pct: float = 0.0  # 0-100
    # Human-readable message
    message: str = ""
    # OCR stats
    ocr_pages: int = 0          # total OCR pages processed so far
    ocr_current_page: int = 0   # page being OCR-ed right now (1-based, 0 = not in OCR)
    ocr_total_pages: int = 0    # total pages in the file currently being OCR-ed


class ProgressTracker:
    """Collects progress events during index building."""

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._total_files: int = 0
        self._processed: int = 0
        self._skipped: int = 0
        self._chunks: int = 0
        self._current_file: str = ""
        self._phase: str = "idle"
        # OCR counters
        self._ocr_pages: int = 0          # cumulative OCR pages done
        self._ocr_current_page: int = 0   # page in progress (0 when idle)
        self._ocr_total_pages: int = 0    # total pages of current file

    def start(self, total_files: int) -> None:
        self._start_time = time.time()
        self._total_files = total_files
        self._processed = 0
        self._skipped = 0
        self._chunks = 0
        self._ocr_pages = 0
        self._ocr_current_page = 0
        self._ocr_total_pages = 0
        self._phase = "extracting"

    @property
    def elapsed(self) -> float:
        if self._start_time:
            return time.time() - self._start_time
        return 0.0

    @property
    def eta(self) -> float:
        if self._processed < 2 or self._total_files == 0:
            return 0.0
        avg_per_file = self.elapsed / self._processed
        remaining = self._total_files - self._processed
        return round(avg_per_file * remaining, 1)

    @property
    def pct(self) -> float:
        if self._total_files == 0:
            return 0.0
        return round((self._processed / self._total_files) * 100, 1)

    def _make_event(self, phase: str, message: str, *, eta_override: float | None = None) -> ProgressEvent:
        """Build a ProgressEvent from current tracker state."""
        return ProgressEvent(
            phase=phase,
            current_file=self._current_file,
            current=self._processed,
            total=self._total_files,
            chunks=self._chunks,
            skipped=self._skipped,
            elapsed=round(self.elapsed, 1),
            eta=eta_override if eta_override is not None else self.eta,
            pct=self.pct,
            message=message,
            ocr_pages=self._ocr_pages,
            ocr_current_page=self._ocr_current_page,
            ocr_total_pages=self._ocr_total_pages,
        )

    def ocr_page_start(self, rel_path: str, page: int, total_pages: int) -> ProgressEvent:
        """Called just before each OCR page request is sent to the vision LLM."""
        self._current_file = rel_path
        self._ocr_current_page = page
        self._ocr_total_pages = total_pages
        msg = f"OCR {rel_path} — page {page}/{total_pages}"
        return self._make_event(self._phase, msg)

    def ocr_page_done(self, rel_path: str, page: int, total_pages: int) -> ProgressEvent:
        """Called after an OCR page is successfully processed."""
        self._ocr_pages += 1
        self._ocr_current_page = page
        self._ocr_total_pages = total_pages
        msg = f"OCR {rel_path} — page {page}/{total_pages} done"
        return self._make_event(self._phase, msg)

    def file_done(self, rel_path: str, chunk_count: int = 0, skipped: bool = False) -> ProgressEvent:
        self._current_file = rel_path
        self._processed += 1
        self._ocr_current_page = 0  # reset per-file OCR page counter
        if skipped:
            self._skipped += 1
        else:
            self._chunks += chunk_count
        msg = f"Processing {self._processed}/{self._total_files} — {rel_path}"
        return self._make_event(self._phase, msg)

    def set_phase(self, phase: str, message: str = "") -> ProgressEvent:
        self._phase = phase
        eta = self.eta if phase != "embedding" else 0.0
        return self._make_event(phase, message or f"{phase}…", eta_override=eta)

    def done(self, total_chunks: int) -> ProgressEvent:
        self._ocr_current_page = 0
        return ProgressEvent(
            phase="done",
            current=self._total_files,
            total=self._total_files,
            chunks=total_chunks,
            skipped=self._skipped,
            elapsed=round(self.elapsed, 1),
            eta=0.0,
            pct=100.0,
            message=f"Index built: {total_chunks} chunks from {self._total_files} files",
            ocr_pages=self._ocr_pages,
            ocr_current_page=0,
            ocr_total_pages=self._ocr_total_pages,
        )

    def error(self, message: str) -> ProgressEvent:
        return self._make_event("error", f"Error: {message}", eta_override=0.0)

    def cancelled(self) -> ProgressEvent:
        return self._make_event(
            "cancelled",
            f"Cancelled after {self._processed}/{self._total_files} files",
            eta_override=0.0,
        )
