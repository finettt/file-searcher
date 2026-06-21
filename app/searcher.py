"""Native hybrid search via Qdrant (dense + sparse) + cross-encoder reranking."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from qdrant_client.http import models as qmodels

from .cache import QdrantIndex
from .chunking import make_snippet
from .config import (
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_RERANKER_TOP_CANDIDATES,
    DEFAULT_SPARSE_VECTOR_NAME,
)
from .indexer import build_client, embed_texts, l2_normalize_vector
from .logging_config import get_logger
from .reranker import rerank as rerank_chunks
from .scoring import (
    bm25_scores,
    bm25_top_k,
    query_sparse_vector,
    rank_by_file,
    rrf_fusion,
)

log = get_logger(__name__)


def _normalize_ext_filter(ext_filter: list[str] | None) -> set[str] | None:
    """Normalize extensions to lowercase ``.ext`` form."""
    if not ext_filter:
        return None
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext_filter}


def _build_rerank_pool(
    chunks: list[dict],
    scores: np.ndarray,
    *,
    top_k: int,
    by_chunk: bool,
) -> list[int]:
    """Select the candidate pool to send to the reranker.

    ``by_chunk=True`` keeps chunk-level candidates and uses the standard
    top-50 BM25/RRF slice. ``by_chunk=False`` de-duplicates by file first,
    keeping up to ``top_k * 4`` unique files, capped by the reranker pool size.
    """
    if by_chunk:
        return bm25_top_k(scores, DEFAULT_RERANKER_TOP_CANDIDATES)

    unique_file_limit = min(DEFAULT_RERANKER_TOP_CANDIDATES, top_k * 4)
    return [idx for idx, _ in rank_by_file(scores, chunks, unique_file_limit)]


def _collection_has_sparse(qdrant: QdrantIndex) -> bool:
    """Check whether the collection was created with sparse vector support."""
    try:
        info = qdrant.client.get_collection(qdrant.collection)
        sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
        if sparse_cfg and isinstance(sparse_cfg, dict):
            return DEFAULT_SPARSE_VECTOR_NAME in sparse_cfg
        return False
    except Exception:
        return False


def do_search(
    qdrant: QdrantIndex,
    query: str,
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    reranker_base_url: str,
    reranker_model: str,
    top_k: int = 5,
    by_chunk: bool = False,
    snippet_chars: int = 300,
    ext_filter: list[str] | None = None,
) -> list[dict]:
    """Run hybrid search against Qdrant and rerank with a cross-encoder.

    Pipeline (native hybrid — sparse vectors present):
    1. Embed query → dense vector; tokenize query → sparse vector
    2. Qdrant ``query_points`` with two ``prefetch`` legs (dense + sparse)
       fused via ``Fusion.RRF`` — all fusion happens server-side
    3. Build reranker pool from the fused results
    4. Cross-encode with llama.cpp ``/v1/rerank``; fall back to RRF order on failure
    5. Apply optional extension filter and return top-k results

    Fallback (legacy — no sparse vectors in collection):
    Same as before: dense search → client-side BM25 → client-side RRF → rerank.

    The ``score`` field in each result contains the RRF fusion score for
    backward compatibility. ``rerank`` contains the cross-encoder score (-1.0
    when the reranker is unavailable and the RRF fallback is used).
    """
    t0 = time.monotonic()
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    allowed_exts = _normalize_ext_filter(ext_filter)

    # Detect whether the collection supports native hybrid search
    has_sparse = _collection_has_sparse(qdrant)

    log.info(
        "search start  query=%r top_k=%d by_chunk=%s hybrid=%s ext_filter=%s",
        query,
        top_k,
        by_chunk,
        has_sparse,
        sorted(allowed_exts) if allowed_exts else [],
    )

    client = build_client(api_key, base_url)

    embed_t0 = time.monotonic()
    embeddings = embed_texts(client, model, [query], batch_size=1)
    if not embeddings.size:
        log.warning("No embeddings returned for query")
        return []
    query_emb = l2_normalize_vector(embeddings[0])
    embed_dt = time.monotonic() - embed_t0
    log.debug(
        "query embedded  model=%s dim=%d time=%.3fs",
        model,
        len(query_emb),
        embed_dt,
    )

    # Build filter: exclude metadata sentinel point
    qdrant_filter = qmodels.Filter(
        must_not=[
            qmodels.FieldCondition(
                key="_sentinel",
                match=qmodels.MatchValue(value=True),
            ),
        ]
    )

    # Fetch more candidates for re-ranking
    search_limit = max(top_k * 10, 100)

    qdrant_t0 = time.monotonic()

    try:
        if has_sparse:
            # ── Native hybrid: dense + sparse prefetch → server-side RRF ──
            query_sparse = query_sparse_vector(query)
            results = qdrant.client.query_points(
                collection_name=qdrant.collection,
                prefetch=[
                    qmodels.Prefetch(
                        query=query_emb.tolist(),
                        using=DEFAULT_DENSE_VECTOR_NAME,
                        limit=search_limit,
                        filter=qdrant_filter,
                    ),
                    qmodels.Prefetch(
                        query=query_sparse,
                        using=DEFAULT_SPARSE_VECTOR_NAME,
                        limit=search_limit,
                        filter=qdrant_filter,
                    ),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=search_limit,
                with_payload=True,
            )
        else:
            # ── Legacy: dense-only search ──
            results = qdrant.client.query_points(
                collection_name=qdrant.collection,
                query=query_emb.tolist(),
                query_filter=qdrant_filter,
                limit=search_limit,
                with_payload=True,
            )
    except Exception as exc:
        log.warning(
            "Qdrant search query failed, falling back to empty results: %s",
            exc,
        )
        return []

    qdrant_dt = time.monotonic() - qdrant_t0

    point_count = len(results.points)
    log.info(
        "qdrant query complete  collection=%s candidates=%d limit=%d hybrid=%s time=%.3fs",
        qdrant.collection,
        point_count,
        search_limit,
        has_sparse,
        qdrant_dt,
    )

    if not results.points:
        log.info("search end  no results  total=%.3fs", time.monotonic() - t0)
        return []

    # Build payload from Qdrant results
    chunks: list[dict] = []
    fusion_scores: list[float] = []

    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            {
                "path": payload.get("path", ""),
                "abs_path": payload.get("abs_path", ""),
                "chunk_id": payload.get("chunk_id", 0),
                "start": payload.get("start", 0),
                "end": payload.get("end", 0),
                "text": payload.get("text", ""),
            }
        )
        fusion_scores.append(point.score or 0.0)

    scores_arr = np.array(fusion_scores, dtype=np.float32)

    if has_sparse:
        # Qdrant already fused dense + sparse via RRF — scores_arr IS the
        # fused score.  We don't need client-side BM25 or RRF.
        scores = scores_arr
        # For backward compat, sem and lex are both set to the fusion score
        # since we can't decompose server-side RRF into individual components.
        sem_arr = scores_arr
        lex = np.zeros_like(scores_arr)
        bm25_dt = 0.0
    else:
        # Legacy path: client-side BM25 + RRF
        sem_arr = scores_arr
        bm25_t0 = time.monotonic()
        lex = bm25_scores(query, chunks)
        bm25_dt = time.monotonic() - bm25_t0
        scores = rrf_fusion(sem_arr, lex)

    rerank_pool_indices = _build_rerank_pool(chunks, scores, top_k=top_k, by_chunk=by_chunk)

    log.debug(
        "candidate fusion done  candidates=%d bm25_time=%.3fs hybrid=%s scores_max=%.4f rerank_pool=%d",
        len(chunks),
        bm25_dt,
        has_sparse,
        float(scores_arr.max()) if len(scores_arr) else 0.0,
        len(rerank_pool_indices),
    )

    if not rerank_pool_indices:
        log.info("search end  empty rerank pool  total=%.3fs", time.monotonic() - t0)
        return []

    rerank_input = [chunks[idx] for idx in rerank_pool_indices]

    # Cross-encoder rerank — fall back to RRF order when the reranker is down.
    # rerank_input is already ordered by score descending (via bm25_top_k or
    # rank_by_file), so the fallback index order == descending relevance order.
    reranker_used = True
    try:
        # Request all pool scores; we apply top_k after ext_filter so we don't
        # lose results to pre-filter truncation.
        rerank_results = rerank_chunks(
            query,
            rerank_input,
            base_url=reranker_base_url,
            model=reranker_model,
        )
    except Exception as exc:
        log.warning(
            "reranker unavailable (%s), falling back to RRF order",
            exc,
        )
        reranker_used = False
        # Sentinel score -1.0 is outside the [0,1] reranker range, so consumers
        # can distinguish "reranker was down" from a genuine near-zero score.
        rerank_results = [(i, -1.0) for i in range(len(rerank_input))]

    output: list[dict] = []
    for rerank_idx, rerank_score in rerank_results:
        original_idx = rerank_pool_indices[rerank_idx]
        item = chunks[original_idx]

        if allowed_exts:
            ext = Path(item["path"]).suffix.lower()
            if ext not in allowed_exts:
                continue

        rrf_score = round(float(scores[original_idx]), 4)
        output.append(
            {
                "rank": len(output) + 1,
                "path": item["path"],
                "score": rrf_score,  # RRF fusion — backward compat
                "sem": round(float(sem_arr[original_idx]), 4),
                "lex": round(float(lex[original_idx]), 2),
                "rerank": round(float(rerank_score), 4),  # cross-encoder score (-1.0 on fallback)
                "reranker_used": reranker_used,  # consumers can distinguish fallback
                "chunk_id": item["chunk_id"],
                "start": item["start"],
                "end": item["end"],
                "snippet": make_snippet(item["text"], limit=snippet_chars),
            }
        )
        # Early exit when no ext filter is active — avoid iterating the rest of
        # the pool unnecessarily (e.g. 50-item pool with top_k=5).
        if not allowed_exts and len(output) >= top_k:
            break

    output = output[:top_k]

    total_dt = time.monotonic() - t0
    log.info(
        "search end  returned=%d rerank_pool=%d reranker_used=%s hybrid=%s total=%.3fs",
        len(output),
        len(rerank_pool_indices),
        reranker_used,
        has_sparse,
        total_dt,
    )
    for item in output[: min(5, len(output))]:
        log.debug(
            "result rank=%d path=%s chunk=%d score=%.4f sem=%.4f lex=%.2f rerank=%.4f",
            item["rank"],
            item["path"],
            item["chunk_id"],
            item["score"],
            item["sem"],
            item["lex"],
            item["rerank"],
        )

    return output
