"""File-system watcher — incremental re-index on file changes."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

from watchfiles import awatch

from . import cache, indexer
from .logging_config import get_logger

log = get_logger(__name__)

# Debounce window: collect events for this many seconds before acting
_DEBOUNCE_SECONDS = 5.0


def _relevant(path_str: str, exts: set[str]) -> bool:
    ext = os.path.splitext(path_str)[1].lower()
    return ext in exts


def _skip_dir(path_str: str) -> bool:
    skips = {"node_modules", "__pycache__", "venv", ".venv", ".git", "@eaDir", "#recycle"}
    return any(s in Path(path_str).parts for s in skips)


async def watch_directory(
    data_dir: Path,
    exts: set[str],
    qdrant: cache.QdrantIndex,
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    ocr_api_key: str | None = None,
    ocr_base_url: str | None = None,
    ocr_model: str | None = None,
    chunk_size: int = 1200,
    overlap: int = 200,
    batch_size: int = 32,
    shutdown_event: asyncio.Event | None = None,
    on_rebuild_lock: threading.Lock | None = None,
) -> None:
    """Watch *data_dir* and trigger selective rebuilds on file changes.

    Runs until *shutdown_event* is set.  Debounces rapid events into single
    batches so that editor saves (which can fire multiple inotify events) do
    not trigger N rebuilds.

    Uses *on_rebuild_lock* to serialise access to the index — pass the same
    lock used by manual rebuilds.
    """
    log.info(
        "Watcher started  root=%s  exts=%s",
        data_dir,
        ", ".join(sorted(exts)),
    )

    if not data_dir.is_dir():
        log.error("Watcher: data directory does not exist: %s", data_dir)
        return

    try:
        async for changes in awatch(
            str(data_dir),
            recursive=True,
            stop_event=shutdown_event,
        ):
            added: set[str] = set()
            modified: set[str] = set()
            deleted: set[str] = set()

            for change_type, path_str in changes:
                if not _relevant(path_str, exts) or _skip_dir(path_str):
                    continue
                rel = os.path.relpath(path_str, str(data_dir)).replace(os.sep, "/")
                if change_type == "added":
                    added.add(rel)
                elif change_type == "modified":
                    modified.add(rel)
                elif change_type == "deleted":
                    deleted.add(rel)

            if not (added or modified or deleted):
                continue

            # Debounce: wait for the filesystem to settle
            await asyncio.sleep(_DEBOUNCE_SECONDS)

            to_reindex = sorted(added | modified)
            to_delete = sorted(deleted)

            # Acquire the rebuild lock so we don't fight a manual rebuild
            if on_rebuild_lock and not on_rebuild_lock.acquire(blocking=False):
                log.info("Watcher: rebuild in progress, skipping batch")
                continue

            try:
                if to_reindex:
                    log.info("Watcher: re-indexing %d changed file(s)", len(to_reindex))
                    cancel_ev = threading.Event()
                    progress = object()  # placeholder — could wire up to WS in future

                    try:
                        gen = indexer.build_index_selective(
                            data_dir,
                            to_reindex,
                            qdrant=qdrant,
                            model=model,
                            api_key=api_key,
                            base_url=base_url,
                            ocr_api_key=ocr_api_key,
                            ocr_base_url=ocr_base_url,
                            ocr_model=ocr_model,
                            extensions=exts,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            batch_size=batch_size,
                            progress=None,
                            cancel_event=cancel_ev,
                        )

                        def _consume() -> None:
                            for _ in gen:
                                pass

                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, _consume)
                        log.info("Watcher: re-indexed %d file(s) OK", len(to_reindex))
                    except Exception as exc:
                        log.error("Watcher: re-index failed: %s", exc)

                if to_delete:
                    log.info("Watcher: removing %d deleted file(s)", len(to_delete))
                    qdrant.delete_by_paths(to_delete)
            finally:
                if on_rebuild_lock:
                    on_rebuild_lock.release()

    except asyncio.CancelledError:
        log.info("Watcher stopped")
    except Exception as exc:
        log.error("Watcher error: %s", exc)
