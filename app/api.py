"""Backward-compatibility shim.

The monolithic app factory has been split into a focused module:

  - :mod:`app.indexer_api` — indexer service (search, rebuild, diff,
    file preview, progress WebSocket)

This module re-exports the old ``create_app`` / ``run_app`` names so that
existing code (``main.py``, tests) that imports from ``app.api`` continues
to work without modification.  ``create_app`` now delegates to
``create_indexer_app``, preserving the original monolithic behaviour.
"""

from __future__ import annotations

# Re-export the indexer-side state and manager for tests that inspect
# ``app.state.ctx`` directly.
from .indexer_api import (
    ConnectionManager,
)
from .indexer_api import (
    IndexerState as AppState,
)
from .indexer_api import (
    create_indexer_app as create_app,
)
from .indexer_api import (
    run_indexer as run_app,
)

__all__ = [
    "AppState",
    "ConnectionManager",
    "create_app",
    "run_app",
]
