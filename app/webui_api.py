"""Web UI gateway service: serves HTML and proxies API calls to the indexer service."""

from __future__ import annotations

import asyncio
import csv
import io
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from websockets.client import connect as ws_connect

from .config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    TEMPLATE_DIR,
)
from .log_utils import get_log_config
from .logging_config import get_logger
from .models import ExportRequest

log = get_logger(__name__)

# ── Default URLs ───────────────────────────────────────────────
DEFAULT_INDEXER_URL = os.getenv("INDEXER_URL", "http://localhost:8002")


# ── Helpers ───────────────────────────────────────────────────


def _load_html_template(name: str = "index.html") -> str:
    path = TEMPLATE_DIR / name
    if not path.exists():
        return "<h1>Template not found</h1>"
    return path.read_text(encoding="utf-8")


# ── App factory ───────────────────────────────────────────────


def create_webui_app(
    *,
    indexer_url: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    hide_health: bool = False,
) -> FastAPI:
    """Create and configure the web UI gateway FastAPI application.

    All /api/* and /ws/* routes are transparently proxied to the indexer service.
    The /api/export route is handled locally as it only formats already-returned data.
    """
    indexer_url = (indexer_url or os.getenv("INDEXER_URL", DEFAULT_INDEXER_URL)).rstrip("/")
    html_template = _load_html_template("index.html")

    app = FastAPI(title="File Searcher — Web UI")
    app.state.indexer_url = indexer_url
    app.state.host = host
    app.state.port = port
    app.state.html_template = html_template
    app.state.hide_health = hide_health
    app.add_middleware(CORSMiddleware, allow_origins=["*"])

    # Shared async HTTP client (connection-pooled, reused across requests)
    http_client = httpx.AsyncClient(base_url=indexer_url, timeout=120.0)

    @app.on_event("shutdown")
    async def _shutdown():
        await http_client.aclose()

    # ── HTML root ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return app.state.html_template

    # ── Local-only: export (pure formatting, no indexer needed) ──

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

    # ── Generic HTTP proxy helper ─────────────────────────────

    async def _proxy(request: Request, path: str) -> Response:
        """Forward an incoming request to the indexer service and relay the response."""
        url = f"/{path}"
        # Preserve query string
        if request.url.query:
            url = f"{url}?{request.url.query}"

        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

        try:
            upstream = await http_client.request(
                method=request.method,
                url=url,
                content=body,
                headers=headers,
            )
        except httpx.ConnectError:
            return JSONResponse(
                status_code=503,
                content={"detail": "Indexer service unavailable"},
            )
        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={"detail": "Indexer service timed out"},
            )

        # Relay response headers (strip hop-by-hop headers)
        _excluded = {"transfer-encoding", "connection", "keep-alive", "te", "trailers", "upgrade"}
        relay_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _excluded}

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=relay_headers,
            media_type=upstream.headers.get("content-type"),
        )

    # ── Proxied API routes ────────────────────────────────────

    @app.get("/api/health")
    async def api_health(request: Request):
        return await _proxy(request, "api/health")

    @app.get("/api/status")
    async def api_status(request: Request):
        return await _proxy(request, "api/status")

    @app.get("/api/diff")
    async def api_diff(request: Request):
        return await _proxy(request, "api/diff")

    @app.post("/api/search")
    async def api_search(request: Request):
        return await _proxy(request, "api/search")

    @app.post("/api/rebuild")
    async def api_rebuild(request: Request):
        return await _proxy(request, "api/rebuild")

    @app.post("/api/rebuild-selective")
    async def api_rebuild_selective(request: Request):
        return await _proxy(request, "api/rebuild-selective")

    @app.post("/api/cancel-rebuild")
    async def api_cancel_rebuild(request: Request):
        return await _proxy(request, "api/cancel-rebuild")

    @app.get("/api/file")
    async def api_file(request: Request):
        return await _proxy(request, "api/file")

    # ── WebSocket relay ───────────────────────────────────────

    @app.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket):
        """Relay WebSocket progress events from the indexer to the browser."""
        await websocket.accept()
        upstream_ws_url = indexer_url.replace("http://", "ws://").replace("https://", "wss://")
        upstream_ws_url = f"{upstream_ws_url}/ws/progress"

        try:
            async with ws_connect(upstream_ws_url) as upstream_ws:

                async def browser_to_upstream():
                    """Forward messages from browser to indexer (keepalive / ping)."""
                    try:
                        async for msg in websocket.iter_text():
                            await upstream_ws.send(msg)
                    except WebSocketDisconnect:
                        pass
                    except Exception as exc:
                        log.debug("browser→indexer relay closed: %s", exc)

                async def upstream_to_browser():
                    """Forward progress messages from indexer to browser."""
                    try:
                        async for msg in upstream_ws:
                            try:
                                await websocket.send_text(msg if isinstance(msg, str) else msg.decode())
                            except Exception as exc:
                                log.debug("indexer→browser send failed: %s", exc)
                                break
                    except Exception as exc:
                        log.debug("indexer→browser relay closed: %s", exc)

                await asyncio.gather(
                    browser_to_upstream(),
                    upstream_to_browser(),
                )
        except Exception as exc:
            log.warning("WebSocket relay to indexer failed: %s", exc)
        finally:
            try:
                await websocket.close()
            except Exception as exc:
                log.debug("WebSocket close failed (client already gone): %s", exc)

    return app


# ── Runner ────────────────────────────────────────────────────


def run_webui(app: FastAPI) -> None:
    """Start the uvicorn server for the web UI service."""
    host = app.state.host
    port = app.state.port
    indexer_url = app.state.indexer_url
    log.info("Web UI → http://%s:%s", host, port)
    log.info("Indexer → %s", indexer_url)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        log_config=get_log_config(
            hide_health=app.state.hide_health,
            health_paths=("/api/health",),
        ),
    )
