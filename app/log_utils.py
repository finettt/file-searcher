"""Shared logging utilities."""

from __future__ import annotations

import logging


class HealthLogFilter(logging.Filter):
    """Drop uvicorn access-log lines for health-check paths.

    Attached to the ``uvicorn.access`` handler only.  Matches against the
    request path extracted from ``record.args`` for precision.
    """

    def __init__(
        self,
        hide: bool = False,
        paths: tuple[str, ...] | list[str] = ("/api/health",),
    ) -> None:
        super().__init__()
        self._hide = hide
        self._paths = tuple(paths)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._hide:
            return True
        path = self._extract_path(record)
        if path is None:
            return True
        return not any(p == path for p in self._paths)

    @staticmethod
    def _extract_path(record: logging.LogRecord) -> str | None:
        """Pull the request path from uvicorn access log record args.

        Uvicorn access records use positional args:
        ``(client_addr, method, path, http_version, status_code)``
        """
        args = record.args
        if isinstance(args, dict):
            return args.get("path")
        if isinstance(args, (list, tuple)) and len(args) >= 3:
            return args[2]
        return None


def get_log_config(
    hide_health: bool = False,
    health_paths: tuple[str, ...] = ("/api/health",),
) -> dict:
    """Return a uvicorn ``log_config`` dict that optionally suppresses health access logs."""

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "health_filter": {
                "()": "app.log_utils.HealthLogFilter",
                "hide": hide_health,
                "paths": health_paths,
            },
        },
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["health_filter"],
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }
