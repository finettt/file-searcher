"""Tests for app/progress.py."""

from __future__ import annotations

import time

from app.progress import ProgressEvent, ProgressTracker


class TestProgressTracker:
    def test_initial_state(self):
        tracker = ProgressTracker()
        assert tracker._total_files == 0
        assert tracker._processed == 0

    def test_start_resets_state(self):
        tracker = ProgressTracker()
        tracker.start(100)
        assert tracker._total_files == 100
        assert tracker._processed == 0
        assert tracker._phase == "extracting"

    def test_elapsed_time(self):
        tracker = ProgressTracker()
        assert tracker.elapsed == 0.0
        tracker.start(10)
        time.sleep(0.1)
        assert tracker.elapsed >= 0.1

    def test_eta_calculation(self):
        tracker = ProgressTracker()
        tracker.start(10)
        tracker.file_done("file1.txt")
        tracker.file_done("file2.txt")
        # ETA requires at least 2 processed files
        assert tracker.eta >= 0

    def test_eta_zero_before_threshold(self):
        tracker = ProgressTracker()
        tracker.start(10)
        tracker.file_done("file1.txt")
        assert tracker.eta == 0.0

    def test_pct_calculation(self):
        tracker = ProgressTracker()
        tracker.start(100)
        assert tracker.pct == 0.0
        tracker.file_done("f1.txt")
        tracker.file_done("f2.txt")
        assert tracker.pct == 2.0

    def test_pct_at_100(self):
        tracker = ProgressTracker()
        tracker.start(10)
        for i in range(10):
            tracker.file_done(f"file{i}.txt")
        assert tracker.pct == 100.0

    def test_file_done_creases_processed(self):
        tracker = ProgressTracker()
        tracker.start(10)
        tracker.file_done("file.txt", chunk_count=5)
        assert tracker._processed == 1
        assert tracker._chunks == 5

    def test_file_done_counts_skipped(self):
        tracker = ProgressTracker()
        tracker.start(10)
        tracker.file_done("file.txt", skipped=True)
        assert tracker._processed == 1
        assert tracker._skipped == 1

    def test_set_phase(self):
        tracker = ProgressTracker()
        tracker.start(10)
        event = tracker.set_phase("embedding", "Working...")
        assert event.phase == "embedding"
        assert event.message == "Working..."

    def test_done(self):
        tracker = ProgressTracker()
        tracker.start(10)
        for i in range(10):
            tracker.file_done(f"file{i}.txt", chunk_count=5)
        event = tracker.done(50)
        assert event.phase == "done"
        assert event.pct == 100.0
        assert event.chunks == 50

    def test_error(self):
        tracker = ProgressTracker()
        tracker.start(10)
        tracker.file_done("file.txt")
        event = tracker.error("Something failed")
        assert event.phase == "error"
        assert "Something failed" in event.message


class TestProgressEvent:
    def test_defaults(self):
        event = ProgressEvent(phase="extracting")
        assert event.current_file == ""
        assert event.current == 0
        assert event.total == 0
        assert event.message == ""

    def test_all_fields(self):
        event = ProgressEvent(
            phase="done",
            current_file="test.txt",
            current=5,
            total=10,
            chunks=100,
            skipped=2,
            elapsed=5.5,
            eta=0.0,
            pct=50.0,
            message="Halfway there",
        )
        assert event.phase == "done"
        assert event.current_file == "test.txt"
        assert event.chunks == 100