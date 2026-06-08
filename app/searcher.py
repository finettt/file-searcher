"""Semantic + lexical hybrid search via Qdrant."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from qdrant_client.http import models as qmodels

from .cache import QdrantIndex
from .chunking import make_snippet
from .indexer import build_client, embed_texts, l2_normalize_vector
from .logging_config import get_logger
from .scoring import bm25_scores, rank_by_chunk, rank_by_file

log = get_logger(__name__)


def do_search(
    qdrant: QdrantIndex,
    query: str,
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    top_k: int = 5,
    by_chunk: bool = False,
    snippet_chars: int = 300,
    lexical_weight: float = 0.3,
    ext_filter: list[str] | None = None,
) -> list[dict]:
    """Run hybrid search against Qdrant.

    1. Embed query → search Qdrant for top candidates (semantic)
    2. Re-rank with BM25 (lexical) on the returned chunks
    3. Return ranked list of result dicts
    """
    t0 = time.monotonic()
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    log.info(
        "search start  query=%r top_k=%d by_chunk=%s lexical_weight=%.2f ext_filter=%s",
        query,
        top_k,
        by_chunk,
        lexical_weight,
        ext_filter or [],
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

    # BM25 re-ranking on the returned chunks
    bm25_t0 = time.monotonic()
    lex = bm25_scores(query, chunks)
    lex_max = float(lex.max()) if lex.max() > 0 else 1.0
    lex_norm = lex / lex_max
    bm25_dt = time.monotonic() - bm25_t0

    # Combined semantic and lexical scores
    scores = sem_arr + lexical_weight * lex_norm

    log.debug(
        "rerank stats  candidates=%d bm25_time=%.3fs sem_max=%.4f lex_max=%.4f",
        len(chunks),
        bm25_dt,
        float(sem_arr.max()) if len(sem_arr) else 0.0,
        float(lex.max()) if len(lex) else 0.0,
    )

    # Rank
    rank_t0 = time.monotonic()
    if by_chunk:
        ranked = rank_by_chunk(scores, top_k * 4 if ext_filter else top_k)
    else:
        ranked = rank_by_file(scores, chunks, top_k * 4 if ext_filter else top_k)
    rank_dt = time.monotonic() - rank_t0

    # Final output
    output: list[dict] = []
    for rank, (idx, score) in enumerate(ranked, 1):
        item = chunks[idx]
        if ext_filter:
            ext = Path(item["path"]).suffix.lower()
            allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext_filter}
            if ext not in allowed:
                continue
        output.append(
            {
                "rank": rank,
                "path": item["path"],
                "score": round(float(score), 4),
                "sem": round(float(sem_arr[idx]), 4),
                "lex": round(float(lex[idx]), 2),
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
        "search end  returned=%d rank_time=%.3fs total=%.3fs",
        len(output),
        rank_dt,
        total_dt,
    )
    for item in output[: min(5, len(output))]:
        log.debug(
            "result rank=%d path=%s chunk=%d score=%.4f sem=%.4f lex=%.2f",
            item["rank"],
            item["path"],
            item["chunk_id"],
            item["score"],
            item["sem"],
            item["lex"],
        )

    return output
