"""Index building: file scanning, text extraction, chunking, embedding → Qdrant."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Generator
from pathlib import Path

import numpy as np
from qdrant_client.http import models as qmodels

from . import utils
from .cache import QdrantIndex
from .chunking import chunk_text, clean_text, make_embed_input
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_OVERLAP,
    DEFAULT_SPARSE_VECTOR_NAME,
)
from .extractors import extract_text
from .logging_config import get_logger
from .progress import ProgressEvent, ProgressTracker
from .scoring import document_sparse_vector

log = get_logger(__name__)


class BuildCancelled(Exception):
    """Raised when a rebuild is cancelled by user."""


# ── OpenAI client builders ────────────────────────────────────


def build_client(api_key: str, base_url: str | None = None):
    """Build an OpenAI-compatible client for embeddings."""
    try:
        from openai import OpenAI
    except ImportError:
        log.critical("Missing package: pip install openai")
        sys.exit(1)
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    log.debug("Embedding client → %s", base_url or "(default OpenAI)")
    return OpenAI(**kwargs)


def build_ocr_client(api_key: str | None, base_url: str | None = None):
    """Build an OpenAI-compatible client for OCR (returns None if not configured)."""
    if not api_key:
        log.debug("OCR client not configured (no API key)")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("OCR requested but openai package not installed")
        return None
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    log.debug("OCR client → %s", base_url or "(default OpenAI)")
    return OpenAI(**kwargs)


# ── Embedding ─────────────────────────────────────────────────


def l2_normalize_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def l2_normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm == 0:
        return x
    return x / norm


def embed_texts(
    client,
    model: str,
    texts: list[str],
    batch_size: int = 32,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    """Send texts to embedding model in batches. Returns L2-normalized matrix."""
    embed_log = get_logger("app.embedder")
    all_embeddings: list[list[float]] = []
    total = len(texts)
    embed_start = time.monotonic()

    for start in range(0, total, batch_size):
        if cancel_event and cancel_event.is_set():
            raise BuildCancelled()
        batch = texts[start : start + batch_size]
        batch_t0 = time.monotonic()
        resp = client.embeddings.create(model=model, input=batch)
        batch_dt = time.monotonic() - batch_t0
        data = sorted(resp.data, key=lambda d: d.index)
        all_embeddings.extend(item.embedding for item in data)

        done = min(start + batch_size, total)
        rate = len(batch) / batch_dt if batch_dt > 0 else 0
        embed_log.info(
            "batch %d/%d  (%d chunks in %.1fs, %.1f chunks/s)",
            done,
            total,
            len(batch),
            batch_dt,
            rate,
        )
        embed_log.debug(
            "batch detail: chars_total=%d avg_chars=%.0f",
            sum(len(t) for t in batch),
            sum(len(t) for t in batch) / len(batch) if batch else 0,
        )

    total_dt = time.monotonic() - embed_start
    embed_log.info(
        "Embedding complete: %d chunks in %.1fs (%.1f chunks/s)",
        total,
        total_dt,
        total / total_dt if total_dt > 0 else 0,
    )
    return np.asarray(all_embeddings, dtype=np.float32)


# ── Qdrant upsert helper ─────────────────────────────────────


def _upsert_batch(
    qdrant: QdrantIndex,
    chunks: list[dict],
    embeddings: np.ndarray,
    sparse_vectors: list | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Upsert a batch of chunks + embeddings + sparse vectors into Qdrant."""
    points = []
    for i, chunk in enumerate(chunks):
        point_id = str(uuid.uuid4())
        vector_data: dict = {
            DEFAULT_DENSE_VECTOR_NAME: embeddings[i].tolist(),
        }
        if sparse_vectors is not None:
            vector_data[DEFAULT_SPARSE_VECTOR_NAME] = sparse_vectors[i]
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector_data,
                payload={
                    "path": chunk["path"],
                    "abs_path": chunk["abs_path"],
                    "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                },
            )
        )
    # Batch upsert: Qdrant handles ~100 points efficiently
    upsert_batch = 100
    upsert_t0 = time.monotonic()
    for start in range(0, len(points), upsert_batch):
        if cancel_event and cancel_event.is_set():
            raise BuildCancelled()
        qdrant.client.upsert(
            collection_name=qdrant.collection,
            points=points[start : start + upsert_batch],
        )
    upsert_dt = time.monotonic() - upsert_t0
    log.info(
        "Qdrant upsert: %d points in %.2fs (%.0f pts/s)",
        len(points),
        upsert_dt,
        len(points) / upsert_dt if upsert_dt > 0 else 0,
    )


# ── Human-readable helpers ────────────────────────────────────


def _fmt_size(nbytes: int) -> str:
    """Format byte count as human string."""
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f}KB"
    return f"{nbytes / (1024 * 1024):.1f}MB"


def _fmt_time(seconds: float) -> str:
    """Format seconds as human string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_eta(processed: int, total: int, elapsed: float) -> str:
    """Calculate and format ETA."""
    if processed < 2 or total == 0:
        return "calculating…"
    avg = elapsed / processed
    remaining = avg * (total - processed)
    return _fmt_time(remaining)


# ── Full index build ──────────────────────────────────────────


def build_index(
    root: Path,
    *,
    qdrant: QdrantIndex,
    model: str,
    api_key: str,
    base_url: str | None = None,
    ocr_api_key: str | None = None,
    ocr_base_url: str | None = None,
    ocr_model: str | None = None,
    extensions: set[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: ProgressTracker | None = None,
    cancel_event: threading.Event | None = None,
    checkpoint_path: str | None = None,
) -> Generator[ProgressEvent, None, None]:
    """Build a full semantic index from scratch → Qdrant.

    Yields ProgressEvents for real-time UI feedback.
    """
    build_t0 = time.monotonic()
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    if extensions is None:
        extensions = utils.parse_extensions(None)

    log.info("Starting full index build")
    log.info("  root       : %s", root)
    log.info("  extensions : %s", ", ".join(sorted(extensions)))
    log.info("  chunk_size : %d  overlap: %d", chunk_size, overlap)
    log.info("  model      : %s", model)
    log.info("  qdrant     : %s/%s", qdrant.url, qdrant.collection)

    client = build_client(api_key, base_url)
    ocr_client = build_ocr_client(ocr_api_key, ocr_base_url)

    discover_t0 = time.monotonic()
    files = utils.collect_files(root, extensions)
    discover_dt = time.monotonic() - discover_t0
    if not files:
        raise ValueError("No matching files found")

    # ── Checkpoint resume ─────────────────────────────────────
    checkpoint: dict = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            log.info(
                "Loaded checkpoint: %d files already processed",
                len(checkpoint.get("processed_files", [])),
            )
        except Exception as exc:
            log.warning("Failed to load checkpoint: %s — starting from scratch", exc)
            checkpoint = {}

    processed = set(checkpoint.get("processed_files", []))
    original_file_count = len(files)
    if processed:
        files = [f for f in files if str(f.relative_to(root)) not in processed]
        log.info(
            "Skipping %d already-processed files, %d remaining",
            original_file_count - len(files),
            len(files),
        )
        if not files:
            log.info("All files already processed — build complete")
            if not checkpoint.get("processed_files"):
                raise ValueError("No matching files found")
            if progress:
                yield progress.done(len(checkpoint["processed_files"]))
            return

    log.info(
        "Processing %d files in %s",
        len(files),
        _fmt_time(discover_dt),
    )

    if progress:
        progress.start(len(files))

    # ── Process files in streaming batches ────────────────────
    BATCH_FILES = batch_size * 5  # ~160 files per batch
    extract_t0 = time.monotonic()
    all_chunks_meta: list[dict] = []
    all_file_hashes: dict[str, str] = {}
    total_chunks = 0
    first_batch = True

    for batch_start in range(0, len(files), BATCH_FILES):
        batch_files = files[batch_start : batch_start + BATCH_FILES]
        batch_embed_inputs: list[str] = []
        batch_chunks_meta: list[dict] = []
        batch_file_hashes: dict[str, str] = {}

        # Extract + chunk for this batch
        for batch_rel_idx, path in enumerate(batch_files):
            file_idx = batch_start + batch_rel_idx  # 0-based global index
            if cancel_event and cancel_event.is_set():
                raise BuildCancelled()
            rel_path = str(path.relative_to(root))
            skipped = False
            chunk_count = 0
            file_t0 = time.monotonic()

            # Build OCR callbacks that emit progress events via the tracker
            ocr_events: list = []  # collect events to yield after extract_text returns

            def _on_ocr_start(page: int, total: int, _rp: str = rel_path) -> None:
                if progress:
                    evt = progress.ocr_page_start(_rp, page, total)
                    ocr_events.append(evt)

            def _on_ocr_done(page: int, total: int, _rp: str = rel_path) -> None:
                if progress:
                    evt = progress.ocr_page_done(_rp, page, total)
                    ocr_events.append(evt)

            try:
                file_size = path.stat().st_size
                text = clean_text(
                    extract_text(
                        path,
                        ocr_client=ocr_client,
                        ocr_model=ocr_model,
                        on_ocr_page_start=_on_ocr_start if progress else None,
                        on_ocr_page_done=_on_ocr_done if progress else None,
                    )
                )
                extract_dt = time.monotonic() - file_t0

                if not text:
                    skipped = True
                    log.debug(
                        "[%d/%d] SKIP (empty) %s  size=%s  extract=%.2fs",
                        file_idx + 1,
                        len(files),
                        rel_path,
                        _fmt_size(file_size),
                        extract_dt,
                    )
                else:
                    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                    if not chunks:
                        skipped = True
                        log.debug(
                            "[%d/%d] SKIP (no chunks) %s  text=%d chars",
                            file_idx + 1,
                            len(files),
                            rel_path,
                            len(text),
                        )
                    else:
                        for cid, (s, e, ct) in enumerate(chunks):
                            batch_embed_inputs.append(make_embed_input(rel_path, ct))
                            batch_chunks_meta.append(
                                {
                                    "path": rel_path,
                                    "abs_path": str(path),
                                    "chunk_id": cid,
                                    "start": s,
                                    "end": e,
                                    "text": ct,
                                }
                            )
                            chunk_count += 1
                        batch_file_hashes[rel_path] = utils.compute_file_hash(path)

                        elapsed = time.monotonic() - build_t0
                        eta = _fmt_eta(file_idx + 1, len(files), elapsed)
                        log.debug(
                            "[%d/%d] OK %s  size=%s  chunks=%d  chars=%d  extract=%.2fs  elapsed=%s  eta=%s",
                            file_idx + 1,
                            len(files),
                            rel_path,
                            _fmt_size(file_size),
                            chunk_count,
                            len(text),
                            extract_dt,
                            _fmt_time(elapsed),
                            eta,
                        )
                        if (file_idx + 1) % 10 == 0 or file_idx + 1 == len(files):
                            log.info(
                                "Extracted %d/%d files  (%d chunks so far, eta %s)",
                                file_idx + 1,
                                len(files),
                                total_chunks + len(batch_chunks_meta),
                                eta,
                            )
            except Exception as exc:
                log.warning(
                    "[%d/%d] ERROR %s: %s",
                    file_idx + 1,
                    len(files),
                    rel_path,
                    exc,
                )
                skipped = True

            # Yield any buffered OCR page events first, then the file-done event
            if progress:
                for ocr_evt in ocr_events:
                    yield ocr_evt
                ocr_events.clear()
                yield progress.file_done(rel_path, chunk_count, skipped)

        if not batch_embed_inputs:
            continue

        # Embed this batch
        if progress:
            yield progress.set_phase(
                "embedding",
                f"Embedding batch {batch_start // BATCH_FILES + 1} ({len(batch_embed_inputs)} chunks)...",
            )
        if cancel_event and cancel_event.is_set():
            raise BuildCancelled()

        batch_embeddings = embed_texts(
            client, model, batch_embed_inputs, batch_size=batch_size, cancel_event=cancel_event
        )
        batch_embeddings = l2_normalize_matrix(batch_embeddings)

        # Sparse vectors for this batch
        log.info("Computing sparse vectors for %d chunks...", len(batch_chunks_meta))
        batch_sparse = [document_sparse_vector(chunk["text"], chunk.get("path", "")) for chunk in batch_chunks_meta]

        if progress:
            yield progress.set_phase("saving", "Saving to Qdrant...")
        if cancel_event and cancel_event.is_set():
            raise BuildCancelled()

        # Recreate collection on first batch, then just upsert
        if first_batch:
            vector_size = batch_embeddings.shape[1]
            log.info("Recreating Qdrant collection (vector_size=%d)", vector_size)
            qdrant.recreate_collection(vector_size)
            first_batch = False

        _upsert_batch(
            qdrant,
            batch_chunks_meta,
            batch_embeddings,
            sparse_vectors=batch_sparse,
            cancel_event=cancel_event,
        )

        # Free batch memory
        del batch_embeddings, batch_sparse, batch_embed_inputs

        # Accumulate metadata
        total_chunks += len(batch_chunks_meta)
        all_chunks_meta.extend(batch_chunks_meta)
        all_file_hashes.update(batch_file_hashes)

        # Save checkpoint after each batch
        if checkpoint_path:
            _checkpoint = {
                "processed_files": list(all_file_hashes.keys()),
                "total_files": original_file_count,
                "total_chunks": total_chunks,
                "model": model,
                "root": str(root),
            }
            try:
                with open(checkpoint_path, "w") as f:
                    json.dump(_checkpoint, f)
            except Exception as exc:
                log.warning("Failed to save checkpoint: %s", exc)

    # ── Post-processing ───────────────────────────────────────
    if not all_file_hashes:
        raise ValueError("No text extracted from any file")

    # Set metadata
    if progress:
        yield progress.set_phase("saving", "Finalizing metadata...")
    if cancel_event and cancel_event.is_set():
        raise BuildCancelled()

    fs_hash = utils.compute_fs_hash(root, extensions)
    meta = {
        "created_at": time.time(),
        "root": str(root),
        "model": model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "fs_hash": fs_hash,
        "file_hashes": all_file_hashes,
    }
    qdrant.set_metadata(meta, vector_size)

    # Remove checkpoint on successful completion
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            log.debug("Checkpoint removed after successful build")
        except Exception as exc:
            log.warning("Failed to remove checkpoint: %s", exc)

    total_dt = time.monotonic() - build_t0
    n_indexed = len(files) - (progress._skipped if progress else 0)
    log.info("═══ Index build complete ═══")
    log.info("  root          : %s", root)
    log.info("  files indexed : %d / %d", n_indexed, original_file_count)
    log.info("  chunks        : %d", total_chunks)
    log.info("  embedding dim : %d", vector_size)
    log.info("  qdrant        : %s/%s", qdrant.url, qdrant.collection)
    log.info("  total time    : %s", _fmt_time(total_dt))

    if progress:
        yield progress.done(total_chunks)


# ── Selective re-index (specific files/dirs) ──────────────────


def build_index_selective(
    root: Path,
    paths: list[str],
    *,
    qdrant: QdrantIndex,
    model: str,
    api_key: str,
    base_url: str | None = None,
    ocr_api_key: str | None = None,
    ocr_base_url: str | None = None,
    ocr_model: str | None = None,
    extensions: set[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: ProgressTracker | None = None,
    cancel_event: threading.Event | None = None,
) -> Generator[ProgressEvent, None, None]:
    """Re-index only specific files/dirs, updating Qdrant in place.

    *paths* is a list of relative paths (files or directories) under *root*.
    Existing chunks for those paths are removed, then re-extracted.
    """
    build_t0 = time.monotonic()
    root = root.expanduser().resolve()

    if extensions is None:
        extensions = utils.parse_extensions(None)

    log.info("Starting selective re-index for %d path(s)", len(paths))
    for p in paths:
        log.debug("  target: %s", p)

    target_files: list[Path] = []
    for rel in paths:
        candidate = (root / rel).resolve()
        if candidate.is_dir():
            found = utils.collect_files(candidate, extensions)
            target_files.extend(found)
            log.debug("  dir %s → %d files", rel, len(found))
        elif candidate.is_file():
            target_files.append(candidate)
            log.debug("  file %s", rel)
        else:
            log.warning("Path not found: %s", candidate)

    if not target_files:
        raise ValueError("No files to re-index")

    log.info("Collected %d files for re-indexing", len(target_files))

    exact_paths: list[str] = []
    prefix_paths: list[str] = []
    for rel in paths:
        candidate = root / rel
        if candidate.is_dir():
            prefix_paths.append(rel.rstrip("/") + "/")
        else:
            exact_paths.append(rel)

    # Remove old chunks for these paths
    if exact_paths:
        log.info("Deleting old chunks for %d exact paths", len(exact_paths))
        qdrant.delete_by_paths(exact_paths)
    if prefix_paths:
        log.info("Deleting old chunks for %d directory prefixes", len(prefix_paths))
        qdrant.delete_by_path_prefixes(prefix_paths)

    if progress:
        progress.start(len(target_files))

    # Extract + chunk new files
    client = build_client(api_key, base_url)
    ocr_client = build_ocr_client(ocr_api_key, ocr_base_url)

    embed_inputs: list[str] = []
    new_chunks: list[dict] = []
    new_file_hashes: dict[str, str] = {}

    for file_idx, path in enumerate(target_files):
        if cancel_event and cancel_event.is_set():
            raise BuildCancelled()
        rel_path = str(path.relative_to(root))
        skipped = False
        chunk_count = 0
        file_t0 = time.monotonic()

        ocr_events: list = []

        def _on_ocr_start(page: int, total: int, _rp: str = rel_path) -> None:
            if progress:
                evt = progress.ocr_page_start(_rp, page, total)
                ocr_events.append(evt)

        def _on_ocr_done(page: int, total: int, _rp: str = rel_path) -> None:
            if progress:
                evt = progress.ocr_page_done(_rp, page, total)
                ocr_events.append(evt)

        try:
            file_size = path.stat().st_size
            text = clean_text(
                extract_text(
                    path,
                    ocr_client=ocr_client,
                    ocr_model=ocr_model,
                    on_ocr_page_start=_on_ocr_start if progress else None,
                    on_ocr_page_done=_on_ocr_done if progress else None,
                )
            )
            extract_dt = time.monotonic() - file_t0

            if not text:
                skipped = True
                log.debug(
                    "[%d/%d] SKIP (empty) %s  size=%s",
                    file_idx + 1,
                    len(target_files),
                    rel_path,
                    _fmt_size(file_size),
                )
            else:
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    skipped = True
                else:
                    for cid, (s, e, ct) in enumerate(chunks):
                        embed_inputs.append(make_embed_input(rel_path, ct))
                        new_chunks.append(
                            {
                                "path": rel_path,
                                "abs_path": str(path),
                                "chunk_id": cid,
                                "start": s,
                                "end": e,
                                "text": ct,
                            }
                        )
                        chunk_count += 1
                    new_file_hashes[rel_path] = utils.compute_file_hash(path)

                    elapsed = time.monotonic() - build_t0
                    log.debug(
                        "[%d/%d] OK %s  size=%s  chunks=%d  chars=%d  extract=%.2fs  elapsed=%s",
                        file_idx + 1,
                        len(target_files),
                        rel_path,
                        _fmt_size(file_size),
                        chunk_count,
                        len(text),
                        extract_dt,
                        _fmt_time(elapsed),
                    )
        except Exception as exc:
            log.warning("[%d/%d] ERROR %s: %s", file_idx + 1, len(target_files), rel_path, exc)
            skipped = True

        if progress:
            for ocr_evt in ocr_events:
                yield ocr_evt
            ocr_events.clear()
            yield progress.file_done(rel_path, chunk_count, skipped)

    # Embed newly extracted chunks
    if embed_inputs:
        if progress:
            yield progress.set_phase("embedding", f"Embedding {len(embed_inputs)} chunks…")
        if cancel_event and cancel_event.is_set():
            raise BuildCancelled()
        new_embeddings = embed_texts(client, model, embed_inputs, batch_size=batch_size, cancel_event=cancel_event)
        new_embeddings = l2_normalize_matrix(new_embeddings)

        # Compute sparse vectors for native hybrid search
        log.info("Computing sparse vectors for %d new chunks…", len(new_chunks))
        sparse_vectors = [document_sparse_vector(chunk["text"], chunk.get("path", "")) for chunk in new_chunks]

        # ensure_collection creates if missing, uses new embeddings' dim
        vector_size = new_embeddings.shape[1]
        qdrant.ensure_collection(vector_size)
        _upsert_batch(qdrant, new_chunks, new_embeddings, sparse_vectors=sparse_vectors, cancel_event=cancel_event)
    else:
        log.warning("No text extracted from any target file")

    # Update metadata
    if progress:
        yield progress.set_phase("saving", "Updating metadata…")
    if cancel_event and cancel_event.is_set():
        raise BuildCancelled()

    meta = qdrant.get_metadata()
    file_hashes = meta.get("file_hashes", {})
    for p in exact_paths:
        file_hashes.pop(p, None)
    for pfx in prefix_paths:
        file_hashes = {k: v for k, v in file_hashes.items() if not k.startswith(pfx)}
    file_hashes.update(new_file_hashes)

    fs_hash = utils.compute_fs_hash(root, extensions)
    meta.update(
        {
            "created_at": time.time(),
            "fs_hash": fs_hash,
            "file_hashes": file_hashes,
        }
    )

    # Get actual vector dim from Qdrant (may differ from embeddings if reused)
    # Get actual vector dim from Qdrant (may differ from embeddings if reused)
    info = qdrant.client.get_collection(qdrant.collection)
    vec_cfg = info.config.params.vectors
    if isinstance(vec_cfg, qmodels.VectorParams):
        vector_size = vec_cfg.size
    elif isinstance(vec_cfg, dict) and DEFAULT_DENSE_VECTOR_NAME in vec_cfg:
        vector_size = vec_cfg[DEFAULT_DENSE_VECTOR_NAME].size
    else:
        vector_size = 768  # fallback

    qdrant.set_metadata(meta, vector_size)

    total_chunks = qdrant.count()
    total_dt = time.monotonic() - build_t0
    log.info("═══ Selective re-index complete ═══")
    log.info("  files processed : %d", len(target_files))
    log.info("  new chunks      : %d", len(new_chunks))
    log.info("  total in index  : %d", total_chunks)
    log.info("  total time      : %s", _fmt_time(total_dt))

    if progress:
        yield progress.done(total_chunks)
