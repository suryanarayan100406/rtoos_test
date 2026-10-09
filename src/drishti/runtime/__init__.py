"""Runtime: compute detection, placement policy, guarded dependencies, and the Runtime context."""
from .compute import (
    ComputeNeed,
    DetectedCompute,
    Placement,
    chosen_backend,
    detect_compute,
    place,
)
from .context import Runtime
from .deps import MissingDependencyError, available, require
from .reliability import (
    LEVEL_CAPS,
    Decision,
    NoViableTier,
    Tier,
    choose,
    tier,
)

__all__ = [
    "LEVEL_CAPS",
    "ComputeNeed",
    "Decision",
    "DetectedCompute",
    "MissingDependencyError",
    "NoViableTier",
    "Placement",
    "Runtime",
    "Tier",
    "available",
    "choose",
    "chosen_backend",
    "detect_compute",
    "place",
    "require",
    "tier",
]
