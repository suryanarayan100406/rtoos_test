"""Structured logging for DRISHTI. Uses rich when available; degrades to stdlib logging."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "drishti") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("DRISHTI_LOG_LEVEL", "INFO").upper()
        handler: logging.Handler
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except Exception:  # pragma: no cover - rich always present in core deps
            handler = logging.StreamHandler()
            fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        logging.basicConfig(level=level, format=fmt, handlers=[handler])
        _CONFIGURED = True
    return logging.getLogger(name)
