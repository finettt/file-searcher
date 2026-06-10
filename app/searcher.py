"""Semantic + lexical hybrid search via Qdrant + cross-encoder reranking."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from qdrant_client.http import models as qmodels

from .cache import QdrantIndex
from .chunking import make_snippet
from .config import DEFAULT_RERANKER_TOP_CANDIDATES
from .indexer import build_client, embed_texts, l2_normalize_vector
from .logging_config import get_logger
from .reranker import rerank as rerank_chunks
from .scoring import bm25_scores, bm25_top_k, rank_by_file, rrf_fusion

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
        return bm25_top_k(scores, chunks, DEFAULT_RERANKER_TOP_CANDIDATES)

    unique_file_limit = min(DEFAULT_RERANKER_TOP_CANDIDATES, top_k * 4)
    return [idx for idx, _ in rank_by_file(scores, chunks, unique_file_limit)]


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

    1. Embed query → search Qdrant for top candidates (semantic)
    2. Score returned chunks with BM25 (lexical)
    3. Fuse semantic + lexical rankings with RRF
    4. Build reranker pool (top-50 chunks or top_k*4 unique files)
    5. Cross-encode with llama.cpp ``/v1/rerank``
    6. Apply optional extension filter and return final results
    """
    t0 = time.monotonic()
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    allowed_exts = _normalize_ext_filter(ext_filter)

    log.info(
        "search start  query=%r top_k=%d by_chunk=%s fusion=rrf ext_filter=%s",
        query,
        top_k,
        by_chunk,
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
    filter_conditions: list[qmodels.Condition] = [
        qmodels.FieldCondition(
            key="_sentinel",
            match=qmodels.MatchValue(value=True),
        ),
    ]
    must_not = list(filter_conditions)
    qdrant_filter = qmodels.Filter(must_not=must_not) if must_not else None

    # Fetch more candidates for BM25 re-ranking
    search_limit = max(top_k * 10, 100)

    qdrant_t0 = time.monotonic()
    results = qdrant.client.query_points(
        collection_name=qdrant.collection,
        query=query_emb.tolist(),
        query_filter=qdrant_filter,
        limit=search_limit,
        with_payload=True,
    )
    qdrant_dt = time.monotonic() - qdrant_t0

    point_count = len(results.points)
    log.info(
        "qdrant query complete  collection=%s candidates=%d limit=%d time=%.3fs",
        qdrant.collection,
        point_count,
        search_limit,
        qdrant_dt,
    )

    if not results.points:
        log.info("search end  no results  total=%.3fs", time.monotonic() - t0)
        return []

    # Build payload from Qdrant results
    chunks: list[dict] = []
    semantic_scores: list[float] = []

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
        semantic_scores.append(point.score or 0.0)

    sem_arr = np.array(semantic_scores, dtype=np.float32)

    # BM25 lexical scoring on the returned chunks
    bm25_t0 = time.monotonic()
    lex = bm25_scores(query, chunks)
    bm25_dt = time.monotonic() - bm25_t0

    # Reciprocal Rank Fusion of semantic and lexical rankings
    scores = rrf_fusion(sem_arr, lex)
    rerank_pool_indices = _build_rerank_pool(chunks, scores, top_k=top_k, by_chunk=by_chunk)

    log.debug(
        "candidate fusion done  candidates=%d bm25_time=%.3fs fusion=rrf sem_max=%.4f lex_max=%.4f rerank_pool=%d",
        len(chunks),
        bm25_dt,
        float(sem_arr.max()) if len(sem_arr) else 0.0,
        float(lex.max()) if len(lex) else 0.0,
        len(rerank_pool_indices),
    )

    if not rerank_pool_indices:
        log.info("search end  empty rerank pool  total=%.3fs", time.monotonic() - t0)
        return []

    rerank_input = [chunks[idx] for idx in rerank_pool_indices]
    rerank_results = rerank_chunks(
        query,
        rerank_input,
        base_url=reranker_base_url,
        model=reranker_model,
        top_n=min(top_k, len(rerank_input)),
    )

    output: list[dict] = []
    for rerank_idx, rerank_score in rerank_results:
        original_idx = rerank_pool_indices[rerank_idx]
        item = chunks[original_idx]

        if allowed_exts:
            ext = Path(item["path"]).suffix.lower()
            if ext not in allowed_exts:
                continue

        final_score = round(float(rerank_score), 4)
        output.append(
            {
                "rank": len(output) + 1,
                "path": item["path"],
                "score": final_score,
                "sem": round(float(sem_arr[original_idx]), 4),
                "lex": round(float(lex[original_idx]), 2),
                "rerank": final_score,
                "chunk_id": item["chunk_id"],
                "start": item["start"],
                "end": item["end"],
                "snippet": make_snippet(item["text"], limit=snippet_chars),
            }
        )
        if len(output) >= top_k:
            break

    total_dt = time.monotonic() - t0
    log.info(
        "search end  returned=%d rerank_pool=%d total=%.3fs",
        len(output),
        len(rerank_pool_indices),
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
