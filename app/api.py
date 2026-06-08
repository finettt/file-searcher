"""FastAPI application: routes, WebSocket progress, diff detection."""

from __future__ import annotations

import asyncio
import csv
import io
import os
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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
from .models import (
    ExportRequest,
    RebuildSelectiveRequest,
    SearchRequest,
)
from .progress import ProgressTracker
from .searcher import do_search

log = get_logger(__name__)


# ── WebSocket connection manager ──────────────────────────────


class ConnectionManager:
    """Manage WebSocket clients subscribed to progress events.

    Each app instance gets its own manager (no global singleton).
    Uses an asyncio.Event for efficient, event-driven broadcasting
    instead of busy-polling.
    """

    def __init__(self) -> None:
        self.active: list[WebSocket] = []
        self._queue: deque[dict] = deque(maxlen=200)
        self._event = asyncio.Event()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    def enqueue(self, event: dict) -> None:
        self._queue.append(event)
        self._event.set()

    async def wait_and_broadcast(self) -> None:
        """Wait for enqueued events, then broadcast to all clients."""
        await self._event.wait()
        self._event.clear()
        await self._broadcast()

    async def _broadcast(self) -> None:
        if not self._queue:
            return
        payloads = list(self._queue)
        self._queue.clear()
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                for payload in payloads:
                    await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ── Application state ─────────────────────────────────────────


@dataclass
class AppState:
    """All application configuration and runtime state in one place.

    Replaces 20+ closure-captured variables with a single structured object
    accessible via ``request.app.state.ctx``.
    """

    data_dir: Path
    exts: set[str]
    model: str
    api_key: str
    base_url: str | None
    ocr_api_key: str | None
    ocr_base_url: str | None
    ocr_model: str | None
    qdrant: QdrantIndex
    host: str
    port: int
    batch_size: int
    chunk_size: int
    overlap: int
    top_k: int
    by_chunk: bool
    snippet_chars: int
    lexical_weight: float
    filebrowser_url: str
    html_template: str

    # Runtime mutable state
    rebuilding: bool = False
    rebuild_lock: bool = False
    background_tasks: set[asyncio.Task] = field(default_factory=set)
    manager: ConnectionManager = field(default_factory=ConnectionManager)


def _get_ctx(request: Request) -> AppState:
    """Retrieve the AppState from the request."""
    return request.app.state.ctx


# ── Helpers ───────────────────────────────────────────────────


def _is_stale(ctx: AppState) -> bool:
    info = ctx.qdrant.get_info()
    fs_hash = info.get("fs_hash", "")
    if not fs_hash:
        return True
    try:
        return utils.compute_fs_hash(ctx.data_dir, ctx.exts) != fs_hash
    except Exception:
        return True


def _load_html_template(name: str = "index.html") -> str:
    path = TEMPLATE_DIR / name
    if not path.exists():
        return "<h1>Template not found</h1>"
    return path.read_text(encoding="utf-8")


# ── Background rebuild runner ─────────────────────────────────


async def _run_rebuild(ctx: AppState, selective_paths: list[str] | None = None) -> None:
    if ctx.rebuild_lock:
        log.debug("Rebuild requested but lock held — ignoring")
        return
    ctx.rebuild_lock = True
    ctx.rebuilding = True
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
                    ctx.data_dir,
                    selective_paths,
                    qdrant=ctx.qdrant,
                    model=ctx.model,
                    api_key=ctx.api_key,
                    base_url=ctx.base_url,
                    ocr_api_key=ctx.ocr_api_key,
                    ocr_base_url=ctx.ocr_base_url,
                    ocr_model=ctx.ocr_model,
                    extensions=ctx.exts,
                    chunk_size=ctx.chunk_size,
                    overlap=ctx.overlap,
                    batch_size=ctx.batch_size,
                    progress=progress,
                )
            else:
                gen = build_index(
                    ctx.data_dir,
                    qdrant=ctx.qdrant,
                    model=ctx.model,
                    api_key=ctx.api_key,
                    base_url=ctx.base_url,
                    ocr_api_key=ctx.ocr_api_key,
                    ocr_base_url=ctx.ocr_base_url,
                    ocr_model=ctx.ocr_model,
                    extensions=ctx.exts,
                    chunk_size=ctx.chunk_size,
                    overlap=ctx.overlap,
                    batch_size=ctx.batch_size,
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
            ctx.manager.enqueue(payload)
            # Yield to event loop so WS messages flush
            await asyncio.sleep(0)
        await worker
    finally:
        ctx.rebuilding = False
        ctx.rebuild_lock = False


# ── Lifespan (graceful shutdown) ──────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Manage app startup and shutdown."""
    yield
    # Graceful shutdown: cancel and await all background tasks
    ctx: AppState = app.state.ctx
    if ctx.background_tasks:
        log.info("Shutting down: cancelling %d background task(s)", len(ctx.background_tasks))
        for task in ctx.background_tasks:
            task.cancel()
        await asyncio.gather(*ctx.background_tasks, return_exceptions=True)
        ctx.background_tasks.clear()


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

    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    ocr_api_key = ocr_api_key or os.getenv("OCR_API_KEY")
    ocr_base_url = ocr_base_url or os.getenv("OCR_LLM_BASE_URL")
    ocr_model = ocr_model or os.getenv("OCR_LLM_MODEL")
    filebrowser_url = filebrowser_url or os.getenv("FILEBROWSER_URL", "")
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    qdrant_collection = qdrant_collection or os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)

    qdrant = QdrantIndex(url=qdrant_url, collection=qdrant_collection)

    ctx = AppState(
        data_dir=data_dir,
        exts=exts,
        model=model,
        api_key=api_key,
        base_url=base_url,
        ocr_api_key=ocr_api_key,
        ocr_base_url=ocr_base_url,
        ocr_model=ocr_model,
        qdrant=qdrant,
        host=host,
        port=port,
        batch_size=batch_size,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        by_chunk=by_chunk,
        snippet_chars=snippet_chars,
        lexical_weight=lexical_weight,
        filebrowser_url=filebrowser_url,
        html_template=_load_html_template("index.html"),
    )

    app = FastAPI(title="File Searcher", lifespan=_lifespan)
    app.state.ctx = ctx
    app.add_middleware(CORSMiddleware, allow_origins=["*"])

    # ── Routes ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        return _get_ctx(request).html_template

    @app.get("/api/health")
    async def api_health(request: Request):
        """Healthcheck endpoint for Docker / load balancers."""
        ctx = _get_ctx(request)
        if not ctx.qdrant.collection_exists():
            return JSONResponse(status_code=503, content={"status": "no_index"})
        info = ctx.qdrant.get_info()
        return JSONResponse(
            content={
                "status": "ok",
                "chunks": info.get("points_count", 0),
                "model": info.get("model", "?"),
            }
        )

    @app.get("/api/status")
    async def api_status(request: Request):
        ctx = _get_ctx(request)
        if not ctx.qdrant.collection_exists():
            return JSONResponse(
                content={
                    "stale": True,
                    "building": ctx.rebuilding,
                    "info": "No index built yet",
                    "filebrowser_url": ctx.filebrowser_url,
                }
            )
        info = ctx.qdrant.get_info()
        stale = _is_stale(ctx)
        info_str = f"{info.get('model', '?')} | {info.get('points_count', 0)} chunks | {info.get('root', '?')}"
        return JSONResponse(
            content={
                "stale": stale,
                "building": ctx.rebuilding,
                "info": info_str,
                "filebrowser_url": ctx.filebrowser_url,
            }
        )

    @app.get("/api/diff")
    async def api_diff(request: Request):
        """Compare current filesystem against stored index."""
        ctx = _get_ctx(request)
        current_hashes = utils.compute_fs_hash_map(ctx.data_dir, ctx.exts)

        if not ctx.qdrant.collection_exists():
            return JSONResponse(
                content={
                    "has_index": False,
                    "added": sorted(current_hashes.keys()),
                    "removed": [],
                    "modified": [],
                    "total_current": len(current_hashes),
                }
            )

        stored_hashes = ctx.qdrant.get_all_file_hashes()
        if not stored_hashes:
            return JSONResponse(
                content={
                    "has_index": True,
                    "can_diff": False,
                    "stale": _is_stale(ctx),
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
    async def api_search(body: SearchRequest, request: Request):
        ctx = _get_ctx(request)
        query = body.query.strip()
        if not query:
            return JSONResponse(status_code=400, content={"detail": "Empty query"})

        top_k_val = max(1, min(body.top_k, 200))

        if not ctx.qdrant.collection_exists() or ctx.qdrant.count() == 0:
            # Auto-build on first search
            await _run_rebuild(ctx, None)

        if not ctx.qdrant.collection_exists() or ctx.qdrant.count() == 0:
            return JSONResponse(status_code=503, content={"detail": "Index build failed"})

        try:
            res = do_search(
                ctx.qdrant,
                query,
                model=ctx.model,
                api_key=ctx.api_key,
                base_url=ctx.base_url,
                top_k=top_k_val,
                by_chunk=body.by_chunk,
                snippet_chars=ctx.snippet_chars,
                lexical_weight=ctx.lexical_weight,
                ext_filter=body.ext_filter,
            )
        except Exception as e:
            log.error("Search error: %s", e, exc_info=True)
            return JSONResponse(status_code=500, content={"detail": str(e)})

        info = ctx.qdrant.get_info()
        return JSONResponse(
            content={
                "results": res,
                "index_info": f"{info.get('model', '?')} | {info.get('points_count', 0)} chunks",
                "index_stale": _is_stale(ctx),
                "filebrowser_url": ctx.filebrowser_url,
            }
        )

    @app.post("/api/rebuild")
    async def api_rebuild(request: Request):
        """Trigger a full index rebuild."""
        ctx = _get_ctx(request)
        if ctx.rebuilding:
            return JSONResponse(status_code=409, content={"detail": "Rebuild already in progress"})
        task = asyncio.create_task(_run_rebuild(ctx, None))
        ctx.background_tasks.add(task)
        task.add_done_callback(ctx.background_tasks.discard)
        return JSONResponse(content={"status": "started"})

    @app.post("/api/rebuild-selective")
    async def api_rebuild_selective(body: RebuildSelectiveRequest, request: Request):
        """Re-index specific files or directories."""
        ctx = _get_ctx(request)
        if ctx.rebuilding:
            return JSONResponse(status_code=409, content={"detail": "Rebuild already in progress"})
        task = asyncio.create_task(_run_rebuild(ctx, body.paths))
        ctx.background_tasks.add(task)
        task.add_done_callback(ctx.background_tasks.discard)
        return JSONResponse(content={"status": "started", "paths": body.paths})

    @app.get("/api/file")
    async def api_file(path: str, request: Request):
        ctx = _get_ctx(request)
        full_path = (ctx.data_dir / path).resolve()
        if not full_path.is_relative_to(ctx.data_dir):
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
    async def api_export(body: ExportRequest):
        if body.format == "csv":
            out = io.StringIO()
            fieldnames = ["rank", "path", "score", "sem", "lex", "chunk_id", "snippet"]
            writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(body.results)
            return Response(
                content=out.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=results.csv"},
            )
        return JSONResponse(content=body.results)

    @app.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket):
        """WebSocket endpoint for real-time index build progress."""
        ctx = websocket.app.state.ctx
        await ctx.manager.connect(websocket)
        try:
            while True:
                await ctx.manager.wait_and_broadcast()
        except WebSocketDisconnect:
            ctx.manager.disconnect(websocket)
        except Exception:
            ctx.manager.disconnect(websocket)

    return app


# ── Runner ────────────────────────────────────────────────────


def run_app(app: FastAPI) -> None:
    """Start the uvicorn server."""
    ctx: AppState = app.state.ctx
    log.info("Web UI → http://%s:%s", ctx.host, ctx.port)
    log.info("FileBrowser → %s", ctx.filebrowser_url or "(not configured)")
    log.info("Qdrant → %s", ctx.qdrant.url)
    uvicorn.run(app, host=ctx.host, port=ctx.port, log_level="info")
