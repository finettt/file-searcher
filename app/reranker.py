"""Cross-encoder reranking via llama.cpp /v1/rerank endpoint."""

from __future__ import annotations

import time

import httpx

from .logging_config import get_logger

log = get_logger(__name__)

# Qwen3-Reranker prepends an instruction prompt of ~100 tokens to every
# document.  At ~3 chars/token, 1800 chars ≈ 600 doc-tokens; adding the
# ~100-token instruction prefix keeps the total input well within the
# --batch-size 2048 set in the compose files.  This guard is
# defense-in-depth against edge cases where individual chunks are unusually
# large.
MAX_DOC_CHARS = 1800  # ≈ 600 doc-tokens + ~100 prompt tokens < 2048


def rerank(
    query: str,
    chunks: list[dict],
    *,
    base_url: str,
    model: str = "Qwen3-Reranker-0.6B",
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Call the llama.cpp /v1/rerank endpoint and return ranked (index, score) pairs.

    The endpoint follows the OpenAI-compatible rerank API activated by the
    ``--rerank`` flag in llama.cpp server.

    Args:
        query: The search query string.
        chunks: List of chunk dicts, each containing a ``"text"`` key.
        base_url: Base URL for the reranker service, e.g. ``"http://localhost:8004/v1"``.
        model: Model alias used by the reranker server.
        top_n: How many results to request from the reranker (defaults to len(chunks)).

    Returns:
        List of ``(original_chunk_index, rerank_score)`` tuples sorted by score
        descending (highest relevance first).

    Raises:
        httpx.HTTPError: If the reranker service is unreachable or returns an error.
    """
    if not chunks:
        return []

    documents = [c.get("text", "")[:MAX_DOC_CHARS] for c in chunks]
    payload: dict = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    url = base_url.rstrip("/") + "/rerank"

    log.info(
        "reranker start  url=%s model=%s docs=%d top_n=%s",
        url,
        model,
        len(documents),
        top_n,
    )

    t0 = time.monotonic()
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()

    dt = time.monotonic() - t0
    data = resp.json()

    # llama.cpp rerank response format:
    # {"results": [{"index": 0, "relevance_score": 0.92}, ...]}
    results_raw = data.get("results", [])

    ranked: list[tuple[int, float]] = [
        (int(item["index"]), float(item["relevance_score"]))
        for item in results_raw
    ]
    # Sort descending by score (server may already sort, but be safe)
    ranked.sort(key=lambda x: x[1], reverse=True)

    log.info(
        "reranker done  docs=%d returned=%d time=%.3fs top_score=%.4f",
        len(documents),
        len(ranked),
        dt,
        ranked[0][1] if ranked else 0.0,
    )

    return ranked