"""Pytest configuration and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the project root is on the path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mock qdrant_client and grpc before importing app modules
# These require system libraries that may not be available in test environment
mock_qdrant = MagicMock()
mock_qdrant_client = MagicMock()
mock_qdrant_client.QdrantClient = MagicMock()
sys.modules["qdrant_client"] = mock_qdrant_client
sys.modules["qdrant_client.http"] = MagicMock()
sys.modules["qdrant_client.http.models"] = MagicMock()
sys.modules["qdrant_client.grpc"] = MagicMock()
sys.modules["qdrant_client.async_qdrant_client"] = MagicMock()

# Mock qdrant_client.http.models for qmodels
mock_models = MagicMock()
mock_models.VectorParams = MagicMock()
mock_models.VectorParams.Distance = MagicMock()
mock_models.VectorParams.Distance.COSINE = "Cosine"
mock_models.FieldCondition = MagicMock()
mock_models.MatchValue = MagicMock()
mock_models.Filter = MagicMock()
mock_models.FilterSelector = MagicMock()
mock_models.PointsSelector = MagicMock()
mock_models.PointStruct = MagicMock()
mock_models.PayloadSchemaType = MagicMock()
sys.modules["qdrant_client.http.models"] = mock_models