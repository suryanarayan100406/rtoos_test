"""Optional PDAL ground classification.

This module contains the only PDAL invocation in the shipped pipeline.  Merely importing PDAL is not
evidence that SMRF ran; callers must invoke :func:`classify_ground_smrf` and record its returned point
count.  Keeping the adapter isolated also lets the normal NumPy path remain testable on lightweight
installations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..runtime.deps import require


class PdalGroundFilterError(RuntimeError):
    """Raised when PDAL is present but cannot execute the configured SMRF pipeline."""


def classify_ground_smrf(
    cloud_path: str | Path,
    *,
    window_m: float,
    slope: float,
    threshold: float,
    scalar: float,
) -> tuple[np.ndarray, int]:
    """Run real PDAL SMRF and return ``(ground_xyz, total_classified_points)``.

    PDAL's SMRF convention is ``Classification == 2`` for ground.  The PLY reader is intentionally used
    here rather than a synthetic reclassification of the raster: if a local PDAL build lacks the PLY
    reader or SMRF filter, the call fails and S9 reports the morphological fallback explicitly.
    """
    pdal = require("pdal", purpose="PDAL SMRF ground classification")
    pipeline_spec = [
        {"type": "readers.ply", "filename": str(cloud_path)},
        {
            "type": "filters.smrf",
            "window": float(window_m),
            "slope": float(slope),
            "threshold": float(threshold),
            "scalar": float(scalar),
        },
    ]
    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_spec))
        count = int(pipeline.execute())
        arrays = getattr(pipeline, "arrays", None)
    except Exception as exc:  # PDAL exposes native/plugin failures as varied exception types.
        raise PdalGroundFilterError(f"PDAL SMRF execution failed: {exc}") from exc
    if not arrays:
        raise PdalGroundFilterError("PDAL SMRF returned no point array")
    arr = arrays[0]
    names = set(arr.dtype.names or ())
    required = {"X", "Y", "Z", "Classification"}
    if not required.issubset(names):
        raise PdalGroundFilterError(
            "PDAL SMRF output lacks required fields "
            f"{sorted(required - names)}; available fields: {sorted(names)}"
        )
    ground = np.column_stack([arr["X"], arr["Y"], arr["Z"]]).astype(np.float32, copy=False)
    ground = ground[np.asarray(arr["Classification"]) == 2]
    return ground, count
