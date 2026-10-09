"""Guarded dependency import.

Heavy stages depend on native/GPU libraries that are installed per-environment. Rather than stub or
fake behavior when a library is absent, DRISHTI raises a clear, actionable error (AGENTS.md §5). This
is the single mechanism every heavy stage uses to access an optional dependency.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class MissingDependencyError(RuntimeError):
    """A required optional dependency is not installed in this environment."""


# maps a module name to the pip extra that provides it (for a helpful message)
_EXTRA_HINTS = {
    "av": "video",
    "cv2": "video",
    "pyproj": "geo",
    "rasterio": "geo",
    "shapely": "geo",
    "open3d": "recon",
    "trimesh": "recon",
    "pygltflib": "recon",
    "pycolmap": "poses",
    "gtsam": "spine",
    "pymavlink": "telem",
    "piexif": "telem",
    "torch": "depth",
    "transformers": "depth",
    "huggingface_hub": "depth",
    "laspy": "export",
    "xatlas": "texture",
    "PIL": "texture",
    "scipy": "texture",
    "gsplat": "gaussian",
    "openvino": "depth",
    "onnx": "depth",
    "fastapi": "server",
    "uvicorn": "server",
}


def require(module: str, *, purpose: str = "") -> ModuleType:
    """Import ``module`` or raise MissingDependencyError with an install hint.

    Example: ``cv2 = require("cv2", purpose="frame QA")``.
    """
    try:
        return importlib.import_module(module)
    except Exception as exc:  # ImportError and any import-time failure
        extra = _EXTRA_HINTS.get(module.split(".")[0])
        hint = f'  install with:  pip install -e ".[{extra}]"' if extra else ""
        ctx = f" (needed for {purpose})" if purpose else ""
        raise MissingDependencyError(
            f"optional dependency '{module}' is not available{ctx}.\n{hint}\n"
            f"underlying import error: {exc}"
        ) from exc


def available(module: str) -> bool:
    """True if ``module`` can be imported (used by `doctor` and placement, never to fake output)."""
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False
