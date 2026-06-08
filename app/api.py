"""FastAPI application: routes, WebSocket progress, diff detection."""

from __future__ import annotations

import asyncio
import csv
import io
import os
from collections import deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import utils
from .cache import QdrantIndex
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HOST,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_OVERLAP,
    DEFAULT_PORT,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_URL,
    DEFAULT_SNIPPET_CHARS,
    DEFAULT_TOP_K,
    TEMPLATE_DIR,
)
from .extractors import extract_text
from .indexer import build_index, build_index_selective
from .logging_config import get_logger
from .progress import ProgressTracker
from .searcher import do_search

log = get_logger(__name__)


# ── WebSocket connection manager ──────────────────────────────


class ConnectionManager:
    """Manage WebSocket clients subscribed to progress events."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []
        self._queue: deque[dict] = deque(maxlen=200)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    def enqueue(self, event: dict) -> None:
        self._queue.append(event)

    async def broadcast(self) -> None:
        if not self._queue:
            return
        payloads = [self._queue.popleft() for _ in range(len(self._queue))]
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                for payload in payloads:
                    await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── App factory ───────────────────────────────────────────────


def create_app(
    folder: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    ocr_api_key: str | None = None,
    ocr_base_url: str | None = None,
    ocr_model: str | None = None,
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    extensions: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    by_chunk: bool = False,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    filebrowser_url: str = "",
) -> FastAPI:
    """Create and configure the FastAPI application."""

    data_dir = Path(folder).expanduser().resolve()
    exts = utils.parse_extensions(extensions)

    # Resolve env var defaults
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    ocr_api_key = ocr_api_key or os.getenv("OCR_API_KEY")
    ocr_base_url = ocr_base_url or os.getenv("OCR_LLM_BASE_URL")
    ocr_model = ocr_model or os.getenv("OCR_LLM_MODEL")
    filebrowser_url = filebrowser_url or os.getenv("FILEBROWSER_URL", "")
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    qdrant_collection = qdrant_collection or os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)

    # Qdrant client
    qdrant = QdrantIndex(url=qdrant_url, collection=qdrant_collection)

    # Rebuild state
    _rebuilding = False
    _rebuild_lock = False
    _background_tasks: set[asyncio.Task] = set()

    app = FastAPI(title="File Searcher")
    app.add_middleware(CORSMiddleware, allow_origins=["*"])

    # ── Helpers ─────────────────────────────────────────────

    def is_stale() -> bool:
        info = qdrant.get_info()
        fs_hash = info.get("fs_hash", "")
        if not fs_hash:
            return True
        try:
            return utils.compute_fs_hash(data_dir, exts) != fs_hash
        except Exception:
            return True

    def load_html_template(name: str = "index.html") -> str:
        path = TEMPLATE_DIR / name
        if not path.exists():
            return "<h1>Template not found</h1>"
        return path.read_text(encoding="utf-8")

    html_template = load_html_template("index.html")

    # ── Background rebuild runner ───────────────────────────

    async def _run_rebuild(selective_paths: list[str] | None = None):
        nonlocal _rebuilding, _rebuild_lock
        if _rebuild_lock:
            log.debug("Rebuild requested but lock held — ignoring")
            return
        _rebuild_lock = True
        _rebuilding = True
        if selective_paths:
            log.info("Selective rebuild started for %d path(s)", len(selective_paths))
        else:
            log.info("Full rebuild started")

        progress = ProgressTracker()
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def run_build() -> None:
            try:
                if selective_paths:
                    gen = build_index_selective(
                        data_dir,
                        selective_paths,
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
                        progress=progress,
                    )
                else:
                    gen = build_index(
                        data_dir,
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
                        progress=progress,
                    )

                for evt in gen:
                    loop.call_soon_threadsafe(event_queue.put_nowait, evt.__dict__)
            except Exception as e:
                log.error("Rebuild failed: %s", e, exc_info=True)
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    progress.error(str(e)).__dict__,
                )
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

        worker = asyncio.create_task(asyncio.to_thread(run_build))

        try:
            while True:
                payload = await event_queue.get()
                if payload is None:
                    break
                manager.enqueue(payload)
                await manager.broadcast()
                # Yield to event loop so WS messages flush
                await asyncio.sleep(0)
            await worker
        finally:
            _rebuilding = False
            _rebuild_lock = False

    # ── Routes ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return html_template

    @app.get("/api/health")
    async def api_health():
        """Healthcheck endpoint for Docker / load balancers."""
        if not qdrant.collection_exists():
            return JSONResponse(status_code=503, content={"status": "no_index"})
        info = qdrant.get_info()
        return JSONResponse(
            content={
                "status": "ok",
                "chunks": info.get("points_count", 0),
                "model": info.get("model", "?"),
            }
        )

    @app.get("/api/status")
    async def api_status():
        if not qdrant.collection_exists():
            return JSONResponse(
                content={
                    "stale": True,
                    "building": _rebuilding,
                    "info": "No index built yet",
                    "filebrowser_url": filebrowser_url,
                }
            )
        info = qdrant.get_info()
        stale = is_stale()
        info_str = f"{info.get('model', '?')} | {info.get('points_count', 0)} chunks | {info.get('root', '?')}"
        return JSONResponse(
            content={
                "stale": stale,
                "building": _rebuilding,
                "info": info_str,
                "filebrowser_url": filebrowser_url,
            }
        )

    @app.get("/api/diff")
    async def api_diff():
        """Compare current filesystem against stored index.

        Returns lists of added, removed, and modified files.
        """
        current_hashes = utils.compute_fs_hash_map(data_dir, exts)

        if not qdrant.collection_exists():
            return JSONResponse(
                content={
                    "has_index": False,
                    "added": sorted(current_hashes.keys()),
                    "removed": [],
                    "modified": [],
                    "total_current": len(current_hashes),
                }
            )

        stored_hashes = qdrant.get_all_file_hashes()
        if not stored_hashes:
            return JSONResponse(
                content={
                    "has_index": True,
                    "can_diff": False,
                    "stale": is_stale(),
                    "info": "Index predates per-file hashing. Full rebuild needed.",
                }
            )

        current_paths = set(current_hashes.keys())
        stored_paths = set(stored_hashes.keys())

        added = sorted(current_paths - stored_paths)
        removed = sorted(stored_paths - current_paths)
        modified = sorted(p for p in current_paths & stored_paths if current_hashes[p] != stored_hashes[p])

        return JSONResponse(
            content={
                "has_index": True,
                "can_diff": True,
                "stale": bool(added or removed or modified),
                "added": added,
                "removed": removed,
                "modified": modified,
                "total_current": len(current_hashes),
                "total_stored": len(stored_hashes),
            }
        )

    @app.post("/api/search")
    async def api_search(body: dict):
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(status_code=400, content={"detail": "Empty query"})

        try:
            top_k_val = int(body.get("top_k", top_k))
        except (TypeError, ValueError):
            top_k_val = top_k
        top_k_val = max(1, min(top_k_val, 200))

        by_chunk_val = bool(body.get("by_chunk", by_chunk))
        ext_filter = body.get("ext_filter") or None

        if not qdrant.collection_exists() or qdrant.count() == 0:
            # Auto-build on first search
            await _run_rebuild(None)

        if not qdrant.collection_exists() or qdrant.count() == 0:
            return JSONResponse(status_code=503, content={"detail": "Index build failed"})

        try:
            res = do_search(
                qdrant,
                query,
                model=model,
                api_key=api_key,
                base_url=base_url,
                top_k=top_k_val,
                by_chunk=by_chunk_val,
                snippet_chars=snippet_chars,
                lexical_weight=lexical_weight,
                ext_filter=ext_filter,
            )
        except Exception as e:
            log.error("Search error: %s", e, exc_info=True)
            return JSONResponse(status_code=500, content={"detail": str(e)})

        info = qdrant.get_info()
        return JSONResponse(
            content={
                "results": res,
                "index_info": f"{info.get('model', '?')} | {info.get('points_count', 0)} chunks",
                "index_stale": is_stale(),
                "filebrowser_url": filebrowser_url,
            }
        )

    @app.post("/api/rebuild")
    async def api_rebuild():
        """Trigger a full index rebuild."""
        if _rebuilding:
            return JSONResponse(status_code=409, content={"detail": "Rebuild already in progress"})
        task = asyncio.create_task(_run_rebuild(None))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return JSONResponse(content={"status": "started"})

    @app.post("/api/rebuild-selective")
    async def api_rebuild_selective(body: dict):
        """Re-index specific files or directories.

        Body: {"paths": ["relative/path1", "relative/dir2/"]}
        """
        if _rebuilding:
            return JSONResponse(status_code=409, content={"detail": "Rebuild already in progress"})
        paths = body.get("paths", [])
        if not paths:
            return JSONResponse(status_code=400, content={"detail": "No paths provided"})
        task = asyncio.create_task(_run_rebuild(paths))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return JSONResponse(content={"status": "started", "paths": paths})

    @app.get("/api/file")
    async def api_file(path: str):
        full_path = (data_dir / path).resolve()
        if not str(full_path).startswith(str(data_dir)):
            return JSONResponse(status_code=403, content={"detail": "Access denied"})
        if not full_path.exists():
            return JSONResponse(status_code=404, content={"detail": "File not found"})
        try:
            text = extract_text(full_path)
        except Exception:
            text = ""
        if not text:
            return JSONResponse(content={"text": "(binary or empty)", "lines": 0, "truncated": False})
        truncated = len(text) > 50_000
        lines = text.count("\n") + 1
        return JSONResponse(
            content={
                "text": text[:50_000],
                "lines": lines,
                "truncated": truncated,
            }
        )

    @app.post("/api/export")
    async def api_export(body: dict):
        items = body.get("results", [])
        fmt = body.get("format", "json")
        if fmt == "csv":
            out = io.StringIO()
            fieldnames = ["rank", "path", "score", "sem", "lex", "chunk_id", "snippet"]
            writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(items)
            return Response(
                content=out.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=results.csv"},
            )
        return JSONResponse(content=items)

    @app.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket):
        """WebSocket endpoint for real-time index build progress."""
        await manager.connect(websocket)
        try:
            while True:
                await manager.broadcast()
                await asyncio.sleep(0.15)
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    # Store config on app for uvicorn runner
    app.state.host = host
    app.state.port = port

    return app


# ── Runner ────────────────────────────────────────────────────


def run_app(app: FastAPI) -> None:
    """Start the uvicorn server."""
    host = getattr(app.state, "host", DEFAULT_HOST)
    port = getattr(app.state, "port", DEFAULT_PORT)
    log.info("Web UI → http://%s:%s", host, port)
    log.info("FileBrowser → %s", os.getenv("FILEBROWSER_URL", "(not configured)"))
    log.info("Qdrant → %s", os.getenv("QDRANT_URL", "(not configured)"))
    uvicorn.run(app, host=host, port=port, log_level="info")
