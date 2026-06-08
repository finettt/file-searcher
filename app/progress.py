"""Progress tracking with ETA calculation for index builds."""

from __future__ import annotations

import time
from dataclasses import dataclass


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

    def start(self, total_files: int) -> None:
        self._start_time = time.time()
        self._total_files = total_files
        self._processed = 0
        self._skipped = 0
        self._chunks = 0
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

    def file_done(self, rel_path: str, chunk_count: int = 0, skipped: bool = False) -> ProgressEvent:
        self._current_file = rel_path
        self._processed += 1
        if skipped:
            self._skipped += 1
        else:
            self._chunks += chunk_count
        msg = f"Processing {self._processed}/{self._total_files} — {rel_path}"
        return ProgressEvent(
            phase=self._phase,
            current_file=rel_path,
            current=self._processed,
            total=self._total_files,
            chunks=self._chunks,
            skipped=self._skipped,
            elapsed=round(self.elapsed, 1),
            eta=self.eta,
            pct=self.pct,
            message=msg,
        )

    def set_phase(self, phase: str, message: str = "") -> ProgressEvent:
        self._phase = phase
        return ProgressEvent(
            phase=phase,
            current_file=self._current_file,
            current=self._processed,
            total=self._total_files,
            chunks=self._chunks,
            skipped=self._skipped,
            elapsed=round(self.elapsed, 1),
            eta=self.eta if phase != "embedding" else 0.0,
            pct=self.pct,
            message=message or f"{phase}…",
        )

    def done(self, total_chunks: int) -> ProgressEvent:
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
        )

    def error(self, message: str) -> ProgressEvent:
        return ProgressEvent(
            phase="error",
            current=self._processed,
            total=self._total_files,
            chunks=self._chunks,
            skipped=self._skipped,
            elapsed=round(self.elapsed, 1),
            eta=0.0,
            pct=self.pct,
            message=f"Error: {message}",
        )

    def cancelled(self) -> ProgressEvent:
        return ProgressEvent(
            phase="cancelled",
            current=self._processed,
            total=self._total_files,
            chunks=self._chunks,
            skipped=self._skipped,
            elapsed=round(self.elapsed, 1),
            eta=0.0,
            pct=self.pct,
            message=f"Cancelled after {self._processed}/{self._total_files} files",
        )
