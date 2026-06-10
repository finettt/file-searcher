"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request models ────────────────────────────────────────────


class SearchRequest(BaseModel):
    """POST /api/search request body."""

    query: str
    top_k: int = Field(default=5, ge=1, le=200)
    by_chunk: bool = False
    ext_filter: list[str] | None = None


class RebuildSelectiveRequest(BaseModel):
    """POST /api/rebuild-selective request body."""

    paths: list[str] = Field(..., min_length=1)


class ExportRequest(BaseModel):
    """POST /api/export request body."""

    results: list[dict] = []
    format: str = "json"


# ── Response models ───────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    chunks: int = 0
    model: str = "?"


class StatusResponse(BaseModel):
    stale: bool
    building: bool
    info: str
    filebrowser_url: str = ""


class DiffResponse(BaseModel):
    has_index: bool
    can_diff: bool = True
    stale: bool = False
    added: list[str] = []
    removed: list[str] = []
    modified: list[str] = []
    total_current: int = 0
    total_stored: int = 0
    info: str = ""


class SearchResultItem(BaseModel):
    rank: int
    path: str
    score: float         # RRF fusion score (backward compat)
    sem: float
    lex: float
    rerank: float = -1.0         # cross-encoder score; -1.0 means reranker was unavailable
    reranker_used: bool = True   # False when the cross-encoder fell back to RRF ordering
    chunk_id: int
    start: int
    end: int
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    index_info: str
    index_stale: bool
    filebrowser_url: str = ""


class FileResponse(BaseModel):
    text: str
    lines: int
    truncated: bool


class RebuildResponse(BaseModel):
    status: str
    paths: list[str] = []
