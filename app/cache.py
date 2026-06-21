"""Qdrant collection manager — create, inspect, and manage the vector collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import (
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_VECTOR_NAME,
)
from .logging_config import get_logger

log = get_logger(__name__)

# Sentinel point holds collection-level metadata
_META_POINT_ID = "00000000-0000-0000-0000-000000000000"


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for external service calls.

    Tracks consecutive failures. After ``threshold`` consecutive failures,
    the circuit trips OPEN and stays open for ``recovery_timeout`` seconds.
    A single success resets the failure count.
    """

    name: str
    threshold: int = 5
    recovery_timeout: float = 30.0
    _failures: int = 0
    _last_failure_time: float = 0.0
    _state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    @property
    def is_open(self) -> bool:
        if self._state == "CLOSED":
            return False
        if self._state == "OPEN":
            # Check if recovery timeout has passed
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                return False
            return True
        # HALF_OPEN - allow one request through
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._failures >= self.threshold:
            self._state = "OPEN"

    def __str__(self) -> str:
        return f"CircuitBreaker({self.name}: {self._state}, failures={self._failures})"


class QdrantIndex:
    """Thin wrapper around QdrantClient for index operations."""

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        collection: str = DEFAULT_QDRANT_COLLECTION,
    ) -> None:
        self.url = url
        self.collection = collection
        self._client: QdrantClient | None = None
        self._circuit_breaker = CircuitBreaker(name=f"qdrant:{self.collection}")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            log.debug("Connecting to Qdrant at %s", self.url)
            self._client = QdrantClient(url=self.url, timeout=120)
            log.info("Qdrant connected  url=%s collection=%s", self.url, self.collection)
        return self._client

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    def _call_with_retry(
        self,
        operation: str,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> Any:
        """Call *fn* with retry and circuit breaker protection.

        Args:
            operation: Human-readable name for logging (e.g. "query_points").
            fn: The Qdrant client method to call.
            *args, **kwargs: Passed to fn.

        Returns:
            The return value of fn.

        Raises:
            The last exception if all retries fail and circuit breaker trips.
        """
        if self._circuit_breaker.is_open:
            log.warning(
                "Circuit breaker OPEN for %s/%s — skipping %s",
                self.collection,
                operation,
                self._circuit_breaker,
            )
            raise ConnectionError(f"Circuit breaker OPEN for {operation} on {self.collection}")

        last_exc = None
        for attempt in range(1 + max_retries):
            try:
                result = fn(*args, **kwargs)
                self._circuit_breaker.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "%s attempt %d/%d failed: %s",
                    operation,
                    attempt + 1,
                    1 + max_retries,
                    exc,
                )
                if attempt < max_retries:
                    # Exponential backoff: 0.5s, 1s, 2s...
                    delay = 0.5 * (2**attempt)
                    time.sleep(delay)

        self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]

    # ── Collection lifecycle ──────────────────────────────────

    def ensure_collection(self, vector_size: int) -> bool:
        """Create collection if it doesn't exist. Returns True if created.

        Creates a collection with named dense vector ('text') and sparse
        vector ('text-sparse') configs for native hybrid search.
        """
        existing = [c.name for c in self._call_with_retry("get_collections", self.client.get_collections).collections]
        if self.collection in existing:
            log.debug("Collection already exists: %s", self.collection)
            return False
        t0 = time.monotonic()
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DEFAULT_DENSE_VECTOR_NAME: qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                DEFAULT_SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(),
            },
        )
        for field in ("path", "chunk_id"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD
                if field == "path"
                else qmodels.PayloadSchemaType.INTEGER,
            )
        log.info(
            "Collection created  name=%s vector_size=%d dense=%s sparse=%s time=%.2fs",
            self.collection,
            vector_size,
            DEFAULT_DENSE_VECTOR_NAME,
            DEFAULT_SPARSE_VECTOR_NAME,
            time.monotonic() - t0,
        )
        return True

    def recreate_collection(self, vector_size: int) -> None:
        """Drop and recreate collection (full rebuild)."""
        existing = [c.name for c in self._call_with_retry("get_collections", self.client.get_collections).collections]
        if self.collection in existing:
            log.info("Dropping collection: %s", self.collection)
            self._call_with_retry("delete_collection", self.client.delete_collection, self.collection)
        self.ensure_collection(vector_size)

    def collection_exists(self) -> bool:
        try:
            existing = [
                c.name for c in self._call_with_retry("get_collections", self.client.get_collections).collections
            ]
        except Exception:
            return False
        exists = self.collection in existing
        log.debug("collection_exists(%s) → %s", self.collection, exists)
        return exists

    # ── Point count ───────────────────────────────────────────

    def count(self) -> int:
        """Return number of points (chunks) in the collection."""
        if not self.collection_exists():
            return 0
        info = self._call_with_retry("get_collection", self.client.get_collection, self.collection)
        n = info.points_count or 0
        log.debug("collection count(%s) → %d", self.collection, n)
        return n

    # ── Metadata (stored as a special sentinel point) ─────────

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve collection-level metadata from the sentinel point."""
        if not self.collection_exists():
            return {}
        try:
            points = self._call_with_retry(
                "retrieve",
                self.client.retrieve,
                collection_name=self.collection,
                ids=[_META_POINT_ID],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                return points[0].payload.get("_meta", {})
        except Exception:
            log.debug("Failed to retrieve metadata sentinel", exc_info=True)
        return {}

    def set_metadata(self, meta: dict[str, Any], vector_size: int) -> None:
        """Store collection-level metadata in a sentinel point.

        Uses named vectors to be compatible with the hybrid collection schema.
        """
        t0 = time.monotonic()
        self._call_with_retry(
            "upsert",
            self.client.upsert,
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=_META_POINT_ID,
                    vector={
                        DEFAULT_DENSE_VECTOR_NAME: [0.0] * vector_size,
                    },
                    payload={"_meta": meta, "_sentinel": True},
                ),
            ],
        )
        log.debug("metadata written  collection=%s time=%.3fs", self.collection, time.monotonic() - t0)

    # ── Deletion helpers ──────────────────────────────────────

    def delete_by_paths(self, paths: list[str]) -> None:
        """Delete all points whose 'path' payload matches any of *paths*."""
        if not paths:
            return
        log.info("Deleting points for %d paths", len(paths))
        for p in paths:
            log.debug("  delete path: %s", p)
        self._call_with_retry(
            "delete",
            self.client.delete,
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(
                            key="path",
                            match=qmodels.MatchValue(value=p),
                        )
                        for p in paths
                    ]
                )
            ),
        )

    def delete_by_path_prefixes(self, prefixes: list[str]) -> None:
        """Delete all points whose 'path' starts with any prefix."""
        if not prefixes:
            return
        log.info("Scanning for points matching %d prefix(es)", len(prefixes))
        # Qdrant lacks native prefix match, so scroll and delete
        ids_to_delete: list[str] = []
        offset = None
        while True:
            result = self._call_with_retry(
                "scroll",
                self.client.scroll,
                collection_name=self.collection,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["path"],
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                path = point.payload.get("path", "")
                if any(path.startswith(pfx) for pfx in prefixes):
                    ids_to_delete.append(point.id)
            if next_offset is None:
                break
            offset = next_offset

        log.info("Deleting %d points matching prefix(es)", len(ids_to_delete))
        if ids_to_delete:
            self._call_with_retry(
                "delete",
                self.client.delete,
                collection_name=self.collection,
                points_selector=qmodels.PointsSelector(
                    points=ids_to_delete,
                ),
            )

    # ── Scroll all paths (for diff) ───────────────────────────

    def get_all_file_hashes(self) -> dict[str, str]:
        """Return {path: file_hash} from the metadata sentinel."""
        meta = self.get_metadata()
        return meta.get("file_hashes", {})

    def get_all_paths(self) -> set[str]:
        """Return set of all unique 'path' values in the collection."""
        paths: set[str] = set()
        offset = None
        while True:
            result = self._call_with_retry(
                "scroll",
                self.client.scroll,
                collection_name=self.collection,
                scroll_filter=qmodels.Filter(
                    must_not=[
                        qmodels.FieldCondition(
                            key="_sentinel",
                            match=qmodels.MatchValue(value=True),
                        )
                    ]
                ),
                limit=1000,
                offset=offset,
                with_payload=["path"],
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                paths.add(point.payload.get("path", ""))
            if next_offset is None:
                break
            offset = next_offset
        return paths

    # ── Collection info for status ────────────────────────────

    def get_info(self) -> dict[str, Any]:
        """Return collection info dict for /api/status."""
        if not self.collection_exists():
            return {}
        try:
            info = self._call_with_retry("get_collection", self.client.get_collection, self.collection)
        except Exception:
            return {}
        meta = self.get_metadata()
        return {
            "points_count": info.points_count or 0,
            "model": meta.get("model", "?"),
            "root": meta.get("root", "?"),
            "created_at": meta.get("created_at", 0),
            "fs_hash": meta.get("fs_hash", ""),
            "file_hashes": meta.get("file_hashes", {}),
        }
