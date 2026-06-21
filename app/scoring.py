"""BM25 lexical scoring and ranking helpers."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
from qdrant_client.http import models as qmodels

from .config import DEFAULT_SPARSE_FILENAME_BOOST

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
    """Rank individual chunks by score. Returns [(chunk_idx, score), …].

    Public scoring API.  Not used by the default reranker pipeline
    (which uses :func:`bm25_top_k` + cross-encoder) but available for
    callers that want chunk-level RRF ranking without a cross-encoder step.
    """
    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def bm25_top_k(
    scores: np.ndarray,
    top_k: int,
) -> list[int]:
    """Return the indices of the top-*k* chunks ranked by RRF/BM25 score.

    Used to slice the candidate pool before feeding it to the cross-encoder
    reranker.

    Args:
        scores: Combined RRF scores (or any float array, higher = better).
        top_k: Maximum number of indices to return.

    Returns:
        List of at most *top_k* chunk indices sorted by score descending.
    """
    n = len(scores)
    if n == 0:
        return []
    actual_k = min(top_k, n)
    top_idx = np.argsort(-scores)[:actual_k]
    return [int(i) for i in top_idx]


# ── Sparse vector generation (server-side hybrid search) ──────
#
# Qdrant supports native hybrid search by storing a sparse vector alongside
# the dense embedding. We produce a hashed BM25-style sparse vector for each
# chunk at index time, and a TF-only sparse vector for each query at search
# time. Qdrant then fuses dense+sparse rankings server-side via
# ``query=models.FusionQuery(fusion=models.Fusion.RRF)``.
#
# The vocabulary is open: we hash each token to a 32-bit non-negative integer
# (the dimension index) — this matches Qdrant's ``SparseVector`` schema and
# lets us add new terms without rebuilding a vocab.

# Sparse vector index space is u32; reserve the top bit so hashed indices stay
# in a safe range across platforms.
_SPARSE_DIM = 2**31 - 1


def _hash_token(token: str) -> int:
    """Stable, language-independent token → non-negative int hash.

    Uses Python's built-in ``hash`` is **not** stable across processes, so we
    use a deterministic FNV-1a 32-bit hash instead.
    """
    h = 0x811C9DC5  # FNV offset basis
    for b in token.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime, keep 32-bit
    # Fold into the safe u31 range Qdrant accepts.
    return h % _SPARSE_DIM


def _aggregate_sparse(
    indices_values: dict[int, float],
    tokens: list[str],
    weight: float,
) -> None:
    """Accumulate weighted token counts into an index→value dict."""
    if weight == 0.0 or not tokens:
        return
    counts = Counter(tokens)
    for tok, n in counts.items():
        idx = _hash_token(tok)
        indices_values[idx] = indices_values.get(idx, 0.0) + float(n) * weight


def document_sparse_vector(
    text: str,
    path: str = "",
    *,
    filename_boost: float = DEFAULT_SPARSE_FILENAME_BOOST,
) -> qmodels.SparseVector:
    """Build a hashed term-frequency sparse vector for a document chunk.

    Mirrors the weighting used by :func:`bm25_scores`: filename tokens are
    boosted ×``filename_boost`` so matches on the file name rank higher.
    The IDF component is left to Qdrant's sparse index (which applies BM25
    scoring at query time when used with a TF query vector).
    """
    indices_values: dict[int, float] = {}
    _aggregate_sparse(indices_values, tokenize(text), weight=1.0)
    if path:
        _aggregate_sparse(
            indices_values,
            tokenize(Path(path).name),
            weight=filename_boost,
        )

    if not indices_values:
        # Qdrant rejects fully-empty sparse vectors; emit a single zero-weight
        # entry so the point can still be stored and matched.
        return qmodels.SparseVector(indices=[0], values=[0.0])

    items = sorted(indices_values.items())
    return qmodels.SparseVector(
        indices=[idx for idx, _ in items],
        values=[float(v) for _, v in items],
    )


def query_sparse_vector(query: str) -> qmodels.SparseVector | None:
    """Build a sparse query vector (TF only — Qdrant applies sparse scoring).

    Returns ``None`` for empty/all-stopword queries so callers can skip the
    sparse prefetch entirely and fall back to dense-only search.
    """
    tokens = tokenize(query)
    if not tokens:
        return None

    counts = Counter(tokens)
    pairs = sorted((_hash_token(tok), float(n)) for tok, n in counts.items())

    # ``sorted`` may surface duplicate hashed indices for distinct tokens; merge
    # them so Qdrant doesn't reject the vector with a duplicate-index error.
    merged: dict[int, float] = {}
    for idx, val in pairs:
        merged[idx] = merged.get(idx, 0.0) + val
    items = sorted(merged.items())

    return qmodels.SparseVector(
        indices=[idx for idx, _ in items],
        values=[val for _, val in items],
    )
