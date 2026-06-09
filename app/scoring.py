"""BM25 lexical scoring and ranking helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# ── Tokenization ──────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Simple regex tokenizer (Latin + Cyrillic + digits + path chars)."""
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_./-]+", text.lower())


# ── BM25 ──────────────────────────────────────────────────────


def bm25_scores(
    query: str,
    items: list[dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Compute BM25 scores for *query* against chunk texts + filename boost.

    Each item must have ``"text"`` and ``"path"`` keys.
    Filename tokens get 3× weight.
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return np.zeros(len(items), dtype=np.float32)

    doc_count = len(items)
    tokenized: list[list[str]] = []
    df: dict[str, int] = {}

    for item in items:
        tokens = tokenize(item.get("text", ""))
        filename_tokens = tokenize(Path(item.get("path", "")).name) * 3
        combined = tokens + filename_tokens
        tokenized.append(combined)
        for t in set(combined):
            df[t] = df.get(t, 0) + 1

    avg_dl = float(np.mean([len(t) for t in tokenized])) or 1.0
    scores = np.zeros(doc_count, dtype=np.float32)

    for i, doc_tokens in enumerate(tokenized):
        dl = len(doc_tokens)
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for qt in q_tokens:
            if qt not in tf:
                continue
            n = df.get(qt, 0)
            idf = float(np.log((doc_count - n + 0.5) / (n + 0.5) + 1))
            tf_norm = (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm
        scores[i] = score

    return scores


# ── Reciprocal Rank Fusion ────────────────────────────────────

RRF_K = 60  # Standard RRF constant; higher values dampen rank differences


def rrf_fusion(
    sem_scores: np.ndarray,
    lex_scores: np.ndarray,
    k: int = RRF_K,
) -> np.ndarray:
    """Combine semantic and lexical scores via Reciprocal Rank Fusion.

    RRF is rank-based and requires no weight tuning: each list contributes
    ``1 / (k + rank)`` to the final score (rank is 1-indexed, lower is better).

    Args:
        sem_scores: Semantic similarity scores (higher = better).
        lex_scores: BM25 lexical scores (higher = better).
        k: RRF constant (default 60, per the original RRF paper).

    Returns:
        Combined RRF scores as float32 array (higher = better).
    """
    n = len(sem_scores)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Compute ranks (1-indexed, rank 1 = highest score)
    # argsort ascending, then invert to get rank of each position
    sem_order = np.argsort(-sem_scores)
    lex_order = np.argsort(-lex_scores)

    sem_rank = np.empty(n, dtype=np.float32)
    lex_rank = np.empty(n, dtype=np.float32)
    sem_rank[sem_order] = np.arange(1, n + 1, dtype=np.float32)
    lex_rank[lex_order] = np.arange(1, n + 1, dtype=np.float32)

    return (1.0 / (k + sem_rank) + 1.0 / (k + lex_rank)).astype(np.float32)


# ── Ranking ───────────────────────────────────────────────────


def rank_by_file(
    scores: np.ndarray,
    chunks: list[dict],
    top_k: int,
) -> list[tuple[int, float]]:
    """Rank by best-per-file score. Returns [(chunk_idx, score), …]."""
    best: dict[str, tuple[int, float]] = {}
    for i, score in enumerate(scores):
        path = chunks[i]["path"]
        s = float(score)
        if path not in best or s > best[path][1]:
            best[path] = (i, s)
    ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rank_by_chunk(
    scores: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """Rank individual chunks by score. Returns [(chunk_idx, score), …]."""
    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]
