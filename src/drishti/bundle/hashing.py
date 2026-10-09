"""Content hashing for reproducibility and stage freshness.

A stage is re-run only when its declared inputs or params hash changes (docs/implementation/
02-SYSTEM-DESIGN.md §3). Hashes are content-addressed, never wall-clock heuristics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable hash of a JSON-serializable object (sorted keys, compact separators)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def combine(*hashes: str) -> str:
    """Order-independent combination of child hashes into one."""
    joined = "".join(sorted(h for h in hashes if h))
    return sha256_bytes(joined.encode("utf-8"))
