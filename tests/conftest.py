"""Pytest configuration and fixtures."""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Ensure the project root is on the path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mock qdrant_client and grpc before importing app modules
# These require system libraries that may not be available in test environment
mock_qdrant_client = MagicMock()
mock_qdrant_client.QdrantClient = MagicMock()
sys.modules["qdrant_client"] = mock_qdrant_client


# ── Real lightweight model stubs for types used in app logic ──
# These are needed because app code does isinstance() checks, creates
# real instances (SparseVector, Prefetch, FusionQuery), or uses enum
# values (Fusion.RRF, Distance.COSINE).


@dataclass
class SparseVector:
    """Minimal stub matching qdrant_client.http.models.SparseVector."""
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


class Distance:
    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"


class Fusion(enum.Enum):
    RRF = "rrf"


@dataclass
class FusionQuery:
    fusion: Fusion = Fusion.RRF


@dataclass
class Prefetch:
    query: Any = None
    using: str = ""
    limit: int = 10
    filter: Any = None


@dataclass
class SparseVectorParams:
    index: Any = None


# Mock qdrant_client.http.models for qmodels
mock_models = MagicMock()
mock_models.VectorParams = MagicMock()
mock_models.Distance = Distance
mock_models.FieldCondition = MagicMock()
mock_models.MatchValue = MagicMock()
mock_models.Filter = MagicMock()
mock_models.FilterSelector = MagicMock()
mock_models.PointsSelector = MagicMock()
mock_models.PointStruct = MagicMock()
mock_models.PayloadSchemaType = MagicMock()
# Real types for sparse/hybrid search
mock_models.SparseVector = SparseVector
mock_models.SparseVectorParams = SparseVectorParams
mock_models.Prefetch = Prefetch
mock_models.FusionQuery = FusionQuery
mock_models.Fusion = Fusion

# Expose the same module object through both import paths so
# `from qdrant_client.http import models as qmodels` and
# `import qdrant_client.http.models` see the same stubs.
mock_http = MagicMock()
mock_http.models = mock_models
sys.modules["qdrant_client.http"] = mock_http
sys.modules["qdrant_client.http.models"] = mock_models
